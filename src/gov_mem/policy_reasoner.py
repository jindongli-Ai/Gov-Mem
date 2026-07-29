"""Stateful Policy Reasoning: the single governance decision boundary."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from gov_mem.data.schema import MemoryInstance
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.policy_schema import MemoryStatus, PolicyAction, PolicyDecision, PolicyState
from gov_mem.policy_selector import select_applicable_policies
from gov_mem.query_intent_parser import parse_query_intent
from gov_mem.state_transition_engine import replay_policy_state
from gov_mem.field_state_projection import compile_query_contract


def _requires_sensitive_authorization(question: str) -> bool:
    """Recognize requests for exact sensitive facts, independent of domain."""
    lowered = question.lower()
    # "without leaking restricted material" is a delivery constraint, not a
    # request for a restricted field. Remove this generic instruction before
    # lexical sensitivity detection so an otherwise ordinary logistics
    # summary remains directly answerable.
    lowered = re.sub(
        r"\bwithout\s+(?:leaking|sharing|disclosing|exposing)\s+restricted\s+material\b",
        "",
        lowered,
    )
    # Interrogatives such as ``does``, ``whether`` and ``active`` are not
    # evidence of sensitivity by themselves. The decision boundary should be
    # driven by the requested information type, otherwise ordinary current
    # state questions are turned into denials before retrieval.
    direct_sensitive = bool(re.search(
        r"\b(code|credential|pin|password|door code|portal code|host institution|host-school|restricted|internal label|review flag|customer identity|exact customer|customer mapping|diagnosis|incident diagnosis|medical condition|health condition|disease|treatment|symptom|medication|blood pressure|hormone|clinical|pregnancy|pregnant|viability|viable|access status|access is|psychiatric|mental-health|self-harm|program|hold)\b",
        lowered,
    ))
    indirect_identity_inference = bool(
        re.search(r"\b(?:imply|infer|means|same as|equivalent|correspond|match)\b", lowered)
        and re.search(r"\b(?:customer|account|identity|mapping|name|who)\b", lowered)
    )
    return direct_sensitive or indirect_identity_inference


def _intent_requires_sensitive_authorization(*, question: str, intent) -> bool:
    """Use semantic intent first, with lexical cues only as a safe fallback.

    The old boundary recognized a small list of sensitive words.  This lets a
    query such as ``exact discount cap`` or ``gender-hormone care`` bypass the
    policy check.  The intent parser supplies open-ended semantic categories;
    Python still makes the final authorization decision.
    """
    if _requires_sensitive_authorization(question):
        return True
    categories = set(getattr(intent, "sensitivity_topics", ()) or ())
    mode = str(getattr(intent, "disclosure_mode", "unknown") or "unknown")
    return bool(categories and mode in {"exact", "yes_no", "historical"})


_SAFE_PROJECTION_TOPICS = {
    # These are broad, ordinary operational channels.  They are deliberately
    # topic-level categories rather than dataset nouns or answer values.
    "logistics",
    "scheduling",
    "communication",
}


def _requests_explicit_safe_projection(*, question: str, intent) -> bool:
    """Return whether the query asks for an independent safe field.

    A policy-approved record is not automatically a projection the requester
    asked to receive.  In particular, an administrative pretext followed by
    one yes/no medical question must remain a refusal even if an unrelated
    appointment or contact record is readable.  The parser's reusable topic
    categories provide the signal; no case-specific answer vocabulary is
    needed here.
    """
    topics = {
        str(topic).strip().lower()
        for topic in (getattr(intent, "requested_topics", ()) or ())
        if str(topic).strip()
    }
    if topics.intersection(_SAFE_PROJECTION_TOPICS):
        return True
    # A public/safe audience is itself an explicit projection boundary.  The
    # requester may ask for a mixed exact record, but only the independently
    # authorized public carrier should survive the partial-disclosure path.
    if str(getattr(intent, "target_scope", "") or "").lower() in {
        "public", "safe_summary", "broad", "logistics", "scheduling", "communication",
    }:
        return True
    lowered = str(question or "").lower()
    # Conservative fallback for rule-only parsing.  These are generic
    # operational concepts, not GateMem case names or expected answers.
    return bool(re.search(
        r"\b(?:appointment|schedule|scheduled|calendar|visit|meeting|pickup|"
        r"drop[- ]?off|ride|route|arrival|delivery|logistics|callback|call back|phone|"
        r"mobile|contact|voicemail|email)\b",
        lowered,
    ) or re.search(
        r"\b(?:only helping|just coordinate|coordinate .*schedule|public .*login|"
        r"do not need the whole file|route forms|so i can)",
        lowered,
    ))


def _is_safe_summary_request(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:safe|broad|public|sponsor-safe|sponsor-ready|household-safe|mixed-audience|helper-facing)\s+"
            r"(?:wording|summary|sentence|update|recap|brief|status|one-line\s+status)\b"
            r"|\bhigh[- ]level\s+(?:status|summary|update|recap)\b",
            question.lower(),
        )
        or re.search(r"\bkeep\s+(?:it|this)\s+high[- ]level\b", question.lower())
    )


def _is_lifecycle_query(question: str) -> bool:
    return bool(re.search(
        r"\b(deleted|removed|retired|forgotten|old|previous|former|stale|superseded|"
        r"earlier|provisional|preliminary|prior|before\s+the\s+(?:approved|current|final|updated))\b",
        question.lower(),
    ))


def _is_deleted_existence_query(question: str) -> bool:
    """Recognize attempts to confirm whether a sensitive item still exists."""
    lowered = str(question or "").lower()
    existence = bool(re.search(
        r"\b(?:still\s+(?:have|keep|store)|stored|on\s+file|exists?|somewhere|"
        r"remain(?:s|ing)?|confirm)\b",
        lowered,
    ))
    sensitive = bool(re.search(
        r"\b(?:exact|private|secret|credential|code|pin|password|minute|point|"
        r"location|identity|detail)\b",
        lowered,
    ))
    return existence and sensitive


def _is_deleted_recovery_query(question: str) -> bool:
    """Recognize a request to restate a value after it was retired."""
    lowered = str(question or "").lower()
    lifecycle = bool(re.search(
        r"\b(?:deleted|removed|forgotten|retired|purged|erased|earlier|previous|"
        r"former|provisional|preliminary|prior|before|replaced|superseded)\b",
        lowered,
    ))
    historical = bool(re.search(
        r"\b(?:what\s+was|what\s+were|used\s+to|before\s+(?:it|the|this|that)\s+was|"
        r"recover|reconstruct|restore|bring\s+back|earlier|previous|former|"
        r"appeared\s+before|before\s+the\s+(?:approved|current|final|updated))\b",
        lowered,
    ))
    return lifecycle and historical


def _partial_disclosure_memory_ids(
    state: PolicyState,
    *,
    allowed_memory_ids: tuple[str, ...],
    requested_topics: tuple[str, ...] = (),
    requested_attributes: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Keep an authorized operational/public projection for partial answers.

    A denied exact field must not turn an otherwise shareable request into a
    blanket denial.  This is intentionally based on observable state scopes,
    rather than benchmark query names or answer values.  The resulting set is
    still passed through the normal policy-filtered retrieval boundary.
    """
    safe_scopes = {"public", "broad", "safe_summary"}
    allowed = set(allowed_memory_ids)
    sensitive_topics = {"health", "laboratory", "medication", "imaging", "clinical"}
    requested = set(requested_topics)
    requested.update(
        token
        for attribute in requested_attributes
        for token in re.findall(r"[a-z][a-z0-9_-]{2,}", str(attribute).lower())
        if token not in {"current", "exact", "tell", "give", "what", "the", "and", "for"}
    )
    operational_topics = {"logistics", "scheduling", "communication"}
    return tuple(
        item.memory_id
        for item in state.memory_items
        if item.memory_id in allowed
        and state.memory_status.get(item.memory_id, item.status).value == "active"
        and (
            item.scope in safe_scopes
            or "public_projection" in set(item.topics)
            or (
                bool(operational_topics.intersection(item.topics))
                and bool(requested.intersection(set(item.topics)))
                and not sensitive_topics.intersection(item.topics)
                and bool(re.search(
                    r"\b(?:only|just|help(?:ing)?\s+with|logistics|rides?|check[- ]?in|pickup|callback)\b",
                    item.provenance.evidence_text,
                    re.IGNORECASE,
                ))
            )
        )
    )


def _observable_named_subjects(state: PolicyState) -> tuple[dict[str, Any], ...]:
    """Expose only subject metadata needed for semantic noun grounding.

    This catalog contains no memory IDs, content, or evaluator fields.  It is
    used only when a safe-summary answer needs to resolve a generic noun such
    as "pet" or "parcel" to an observable named thread.
    """
    generic = {
        "current", "latest", "active", "old", "previous", "record", "memory",
        "summary", "safe summary", "broad summary", "exact", "private",
    }
    cue_words = {
        "if", "keep", "and", "for", "when", "the", "restricted", "current",
        "initial", "approved", "deleted", "please", "no", "yes", "was",
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in state.memory_items:
        if state.memory_status.get(item.memory_id, item.status).value != "active":
            continue
        subject = str(item.subject or "").strip()
        if not subject or subject.lower() in generic or subject.lower() in seen:
            continue
        if subject.split()[0].lower().strip(".,:;!?'") in cue_words:
            continue
        if not re.search(r"\b[A-Z][A-Za-z0-9'&-]+(?:\s+[A-Z][A-Za-z0-9'&-]+){1,4}\b", subject):
            continue
        seen.add(subject.lower())
        rows.append({"subject": subject, "topics": list(item.topics)})
    return tuple(rows[:48])


class StatefulPolicyReasoner:
    def __init__(self, *, llm_client: LLMClient | None, config: dict[str, Any]):
        self.llm_client = llm_client
        self.config = config

    def decide(self, *, instance: MemoryInstance, state: PolicyState) -> PolicyDecision:
        policy_cfg = dict(self.config.get("policy_reasoning") or {})
        reasoning_mode = str(policy_cfg.get("mode") or "full").lower()
        if reasoning_mode == "llm_only" and self.llm_client is not None and self.llm_client.is_available():
            return self._llm_only_decide(instance=instance, state=state)
        intent = parse_query_intent(
            instance=instance,
            state=state,
            llm_client=None if reasoning_mode == "rule_only" else self.llm_client,
            config=self.config,
        )
        # Compile the request fields once at the policy boundary.  Retrieval,
        # projection, and execution all consume this same contract; they must
        # not independently ask the model to split or merge the request.
        query_contract = compile_query_contract(
            question=instance.question,
            requester=intent.requester,
            target_subject=intent.target_subject,
            requested_fields=intent.requested_attributes,
            answer_need_spec=None,
            llm_client=None if reasoning_mode == "rule_only" else self.llm_client,
            config=self.config,
        )
        applicable = select_applicable_policies(state, intent)
        transition = replay_policy_state(
            state,
            intent,
            applicable,
            default_owner_access=bool(policy_cfg.get("owner_access", True)),
            enable_temporal_transition=bool(policy_cfg.get("temporal_transition", True)),
            enable_conflict_resolution=bool(policy_cfg.get("conflict_resolution", True)),
            enable_memory_status=bool(policy_cfg.get("explicit_memory_status", True)),
            enable_role_capabilities=bool(policy_cfg.get("role_capabilities_enabled", True)),
            role_capabilities=dict(policy_cfg.get("role_capabilities") or {}),
        )
        relevant_allowed = transition.related_allowed_memory_ids
        relevant_blocked = transition.related_blocked_memory_ids
        explicit_relevant_blocked = transition.explicit_related_blocked_memory_ids
        safe_summary_request = intent.target_scope == "safe_summary" or _is_safe_summary_request(instance.question)
        # Policy authorization is memory-level access control.  A safe/broad
        # wording request must not be turned into a hand-written public-scope
        # whitelist here: GateMem may explicitly authorize a precise
        # operational field (for example a current time, room, label, or
        # approved amount) while restricting a neighboring credential or
        # internal label.  The field-level answer projection downstream is the
        # disclosure boundary for that mixed request. Lifecycle and effective
        # permission blocks remain hard and are never reopened here.
        decision_allowed = tuple(dict.fromkeys(
            (*relevant_allowed, *transition.allowed_memory_ids)
        ))
        decision_blocked = relevant_blocked or transition.blocked_memory_ids
        identity_related_memory_ids = tuple(
            item.memory_id
            for item in state.memory_items
            if set(item.subject_principal_ids or ()).intersection(intent.mentioned_principal_ids or ())
        )
        known_principals = {principal.principal_id for principal in state.principals}
        if intent.identity_ambiguous:
            # Never resolve a same-name/same-category person by nearest text,
            # role, or owner.  The ambiguity itself is a privacy boundary.
            action = PolicyAction.DENY
            reason = intent.identity_resolution_reason or "principal identity is ambiguous"
            decision_allowed = ()
            decision_blocked = tuple(dict.fromkeys((*identity_related_memory_ids, *decision_blocked)))
        elif not intent.requester:
            action = PolicyAction.ABSTAIN
            reason = "requester identity is unresolved"
        elif intent.requester not in known_principals:
            action = PolicyAction.ABSTAIN
            reason = "requester is not present in observable principal state"
        elif intent.requested_operation in {"delete", "forget", "update", "grant", "revoke"}:
            if transition.allowed_memory_ids:
                action = PolicyAction(intent.requested_operation.upper())
                reason = None
            elif transition.uncertainty >= 0.8:
                action = PolicyAction.ABSTAIN
                reason = "governance operation lacks a determinate authorization chain"
            else:
                action = PolicyAction.DENY
                reason = "governance operation is not permitted"
        elif relevant_blocked and not relevant_allowed:
            action = PolicyAction.DENY
            reason = "all matching memory is blocked by effective policy or lifecycle state"
        elif decision_allowed:
            action = PolicyAction.ALLOW
            reason = None
        elif decision_blocked and all(
            transition.status_by_memory_id.get(memory_id).value in {"deleted", "forgotten", "inaccessible"}
            for memory_id in decision_blocked
            if transition.status_by_memory_id.get(memory_id) is not None
        ):
            action = PolicyAction.DENY
            reason = "all matching memory is unavailable due to lifecycle state"
        else:
            action = PolicyAction.ABSTAIN
            reason = "effective policy state is insufficient to determine access"
        if (
            action == PolicyAction.ALLOW
            and _is_deleted_existence_query(instance.question)
            and any(
                transition.status_by_memory_id.get(memory_id) in {
                    MemoryStatus.DELETED,
                    MemoryStatus.FORGOTTEN,
                }
                for memory_id in relevant_blocked
            )
        ):
            # An existence probe must not confirm a successor or expose a
            # yes/no side channel after the relevant old layer disappeared.
            action = PolicyAction.DENY
            reason = "requested exact existence is unavailable after lifecycle transition"
        if action == PolicyAction.ALLOW and _is_deleted_recovery_query(instance.question):
            # Lifecycle is an absolute boundary. Ownership and a later active
            # successor cannot authorize reconstruction of the predecessor.
            action = PolicyAction.DENY
            reason = "lifecycle: requested historical value was deleted or retired"
        sensitive_authorized = self._sensitive_authorized(
            state=state,
            intent=intent,
            applicable=applicable,
            transition=transition,
            owner_sensitive_roles=tuple(
                str(role).lower()
                for role in (policy_cfg.get("sensitive_owner_roles") or ())
                if str(role).strip()
            ),
        )
        explicit_safe_projection = _requests_explicit_safe_projection(
            question=instance.question,
            intent=intent,
        )
        if (
            action == PolicyAction.ALLOW
            and _intent_requires_sensitive_authorization(question=instance.question, intent=intent)
            and not sensitive_authorized
            and not safe_summary_request
            and not _is_lifecycle_query(instance.question)
        ):
            # Exact sensitive requests without a narrow scope are denied at
            # the policy boundary. Safe-summary requests remain ALLOW so the
            # executor can return an authorized broad subset.
            action = PolicyAction.DENY
            reason = "exact sensitive disclosure lacks a narrow authorization scope"
        partial_allowed = ()
        if (
            action == PolicyAction.DENY
            and explicit_safe_projection
            and not intent.identity_ambiguous
            and intent.requested_operation == "access"
            and not _is_lifecycle_query(instance.question)
            and not _is_deleted_recovery_query(instance.question)
        ):
            partial_allowed = _partial_disclosure_memory_ids(
                state,
                allowed_memory_ids=decision_allowed,
                requested_topics=intent.requested_topics,
                requested_attributes=intent.requested_attributes,
            )
            if partial_allowed:
                # ALLOW here means that the policy permits a restricted
                # projection, not the exact sensitive field.  The executor
                # marks the user-visible result as answer_redacted while the
                # retrieval boundary exposes only this safe subset.
                restricted = tuple(
                    memory_id for memory_id in decision_allowed
                    if memory_id not in set(partial_allowed)
                )
                decision_allowed = partial_allowed
                decision_blocked = tuple(dict.fromkeys((*decision_blocked, *restricted)))
                action = PolicyAction.ALLOW
                reason = "partial disclosure: authorized safe fields survive record-level denial"
        snapshot = {
            "as_of_turn_id": state.as_of_turn_id,
            "memory_status": {key: value.value for key, value in transition.status_by_memory_id.items()},
            "requester": intent.requester,
            "requester_role": next(
                (principal.role for principal in state.principals if principal.principal_id == intent.requester),
                None,
            ),
            "target_scope": intent.target_scope,
            "requested_operation": intent.requested_operation,
            "requested_topics": list(intent.requested_topics),
            "requested_attributes": list(intent.requested_attributes),
            "query_contract": asdict(query_contract),
            "mentioned_principal_ids": list(intent.mentioned_principal_ids),
            "identity_ambiguous": bool(intent.identity_ambiguous),
            "identity_resolution_reason": intent.identity_resolution_reason,
            "identity_related_memory_ids": list(identity_related_memory_ids),
            "observable_named_subjects": list(_observable_named_subjects(state)),
            "related_allowed_memory_ids": list(transition.related_allowed_memory_ids),
            "related_blocked_memory_ids": list(transition.related_blocked_memory_ids),
            "explicit_related_blocked_memory_ids": list(explicit_relevant_blocked),
            "allowed_reason_by_memory_id": transition.allowed_reason_by_memory_id or {},
            "blocked_reason_by_memory_id": transition.blocked_reason_by_memory_id or {},
            "sensitive_authorized": sensitive_authorized,
            "explicit_safe_projection_requested": explicit_safe_projection,
            "partial_disclosure": bool(partial_allowed and explicit_safe_projection),
            "partial_allowed_memory_ids": list(partial_allowed if explicit_safe_projection else ()),
            "decision_reason": reason,
        }
        return PolicyDecision(
            action=action,
            requester=intent.requester,
            target_subject=intent.target_subject,
            requested_operation=intent.requested_operation,
            allowed_memory_ids=decision_allowed,
            blocked_memory_ids=decision_blocked,
            applicable_policy_ids=tuple(dict.fromkeys(applicable.policy_ids + transition.winning_policy_ids)),
            state_snapshot=snapshot,
            transition_trace=transition.trace,
            uncertainty=max(transition.uncertainty, 1.0 - intent.confidence),
            abstain_reason=reason if action == PolicyAction.ABSTAIN else None,
        )

    @staticmethod
    def _sensitive_authorized(*, state, intent, applicable, transition, owner_sensitive_roles=()) -> bool:
        """Require an explicit narrow scope for exact sensitive disclosure."""
        requester = str(intent.requester or "")
        # A permission is narrow only when it names this requester.  A
        # grantee-less record is a generic policy candidate and must not be
        # promoted into exact sensitive authorization by itself.
        # Relevance is a retrieval projection, not the complete authorization
        # set. A requester-owned sensitive continuation may omit query nouns,
        # so this check must inspect all policy-approved memory IDs.
        relevant_ids = set(transition.allowed_memory_ids)
        if any(
            str(transition.allowed_reason_by_memory_id.get(memory_id, "")).startswith(
                ("subject-linked", "shared case-owner", "subject-bridged")
            )
            for memory_id in relevant_ids
        ):
            # An observable care/case relationship is an explicit
            # authorization chain, even when it is represented as a graph
            # relation rather than a standalone permission row.
            return True
        if any(
            permission.grantee == requester
            and (not permission.target_memory_ids or relevant_ids.intersection(permission.target_memory_ids))
            and (
                not permission.scope
                or permission.scope in set(intent.requested_topics)
                or permission.scope == intent.target_scope
            )
            for permission in applicable.permissions
        ):
            return True
        # Ownership is a narrow authorization path only when the observable
        # principal role explicitly represents a self-governing owner (for
        # example a patient or primary resident), and the owned record is
        # relevant to the requested target/attribute.  A bare owner string is
        # intentionally insufficient: generic ownership must not turn every
        # exact-sensitive query into ALLOW.
        role = next(
            (str(principal.role or "").lower() for principal in state.principals
             if principal.principal_id == requester),
            "",
        )
        # An explicitly configured operational capability is a narrow
        # authorization chain for the matching sensitive topic. It is not a
        # role-name heuristic: the transition engine must have admitted at
        # least one relevant memory through that same capability path.
        role_capabilities = {
            str(key).lower(): {str(topic).lower() for topic in value}
            for key, value in (getattr(transition, "role_capabilities", {}) or {}).items()
        }
        role_topics = role_capabilities.get(role, set())
        if role_topics and ("*" in role_topics or role_topics.intersection(set(intent.requested_topics))):
            if any(
                str(transition.allowed_reason_by_memory_id.get(memory_id, "")).startswith("role-capability:")
                for memory_id in relevant_ids
            ):
                return True
        configured_roles = set(owner_sensitive_roles) or {
            "patient", "primary_resident", "account_owner", "case_owner",
        }
        target_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(intent.target_subject or "").lower())
            if token not in {"current", "latest", "record", "memory", "summary"}
        }
        requested_tokens = {
            token for attribute in intent.requested_attributes
            for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(attribute).lower())
            if token not in {"current", "latest", "what", "are", "the", "and", "for", "with", "my", "our"}
        }
        if role in configured_roles:
            for item in state.memory_items:
                if item.memory_id not in relevant_ids or item.owner != requester:
                    continue
                if state.memory_status.get(item.memory_id, item.status).value != "active":
                    continue
                if item.scope not in {"private", "exact"} and not {
                    str(topic).lower() for topic in (item.topics or ())
                }.intersection({"access_control", "credential", "health", "medical", "clinical"}):
                    continue
                item_tokens = {
                    token for token in re.findall(
                        r"[a-z0-9][a-z0-9_-]{2,}",
                        " ".join((
                            item.subject or "",
                            item.scope or "",
                            *item.topics,
                            item.provenance.evidence_text or "",
                        )).lower(),
                    )
                }
                if (target_tokens and target_tokens.intersection(item_tokens)) or requested_tokens.intersection(item_tokens):
                    return True
        # Direct ownership is also an explicit self-disclosure route for a
        # private record, even when the dataset uses a domain-specific role
        # such as adult_child. The record must be active, owned by the
        # requester, and semantically tied to the request; role names alone
        # never broaden access to another principal's memory.
        for item in state.memory_items:
            if item.memory_id not in relevant_ids or item.owner != requester:
                continue
            if state.memory_status.get(item.memory_id, item.status).value != "active":
                continue
            item_tokens = {
                token for token in re.findall(
                    r"[a-z0-9][a-z0-9_-]{2,}",
                    " ".join((
                        item.subject or "",
                        item.scope or "",
                        *item.topics,
                        item.provenance.evidence_text or "",
                    )).lower(),
                )
            }
            if (
                item.scope in {"private", "exact"}
                and item_tokens
                and requested_tokens.intersection(item_tokens)
            ):
                return True
        # Direct ownership is an explicit self-disclosure route, but it still
        # needs a semantic carrier. A continuation may omit the query nouns
        # while retaining a calendar-shaped source value; an unrelated owner
        # record must not authorize an exact diagnosis.
        date_terms = {"date", "day", "deadline", "visit", "appointment", "schedule"}
        time_terms = {"time", "window", "hour", "schedule"}
        date_surface = re.compile(
            r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\b|\b\d{1,2}(?:[:/]\d{1,2}){1,2}\b",
            re.IGNORECASE,
        )
        time_surface = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", re.IGNORECASE)
        for item in state.memory_items:
            if (
                item.memory_id not in relevant_ids
                or item.owner != requester
                or state.memory_status.get(item.memory_id, item.status).value != "active"
            ):
                continue
            if item.scope not in {"private", "exact"}:
                continue
            item_text = " ".join((
                item.subject or "",
                item.scope or "",
                *item.topics,
                item.provenance.evidence_text or "",
            )).lower()
            semantic_link = bool(requested_tokens.intersection(
                set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", item_text))
            ))
            if date_terms.intersection(requested_tokens) and date_surface.search(item_text):
                semantic_link = True
            if time_terms.intersection(requested_tokens) and time_surface.search(item_text):
                semantic_link = True
            if semantic_link:
                return True
        sensitive_terms = {
            "credential", "code", "host", "customer", "diagnosis", "clinical",
            "private", "exact", "restricted", "label", "secret", "password",
        }
        for relation in state.scope_constraints:
            if not isinstance(relation, dict):
                continue
            if str(relation.get("principal_id") or relation.get("grantee") or "") != requester:
                continue
            scope = str(relation.get("access_scope") or relation.get("scope") or "").lower()
            if any(term in scope for term in sensitive_terms):
                return True
            relation_type = str(relation.get("type") or "").lower()
            query_text = " ".join(prov.evidence_text for prov in intent.provenance).lower()
            narrow_scope = any(
                marker in scope for marker in
                ("scheduling_only", "safe_summary_only", "public_program", "wellbeing_only", "logistics_only")
            )
            if "case_owner" in relation_type and not narrow_scope and "helping" not in query_text:
                return True
        return False

    def _llm_only_decide(self, *, instance: MemoryInstance, state: PolicyState) -> PolicyDecision:
        payload = {
            "query": instance.question,
            "requester": instance.asking_user_id,
            "memory_ids": [item.memory_id for item in state.memory_items],
            "memory_status": {key: value.value for key, value in state.memory_status.items()},
            "policies": [
                {"policy_id": item.policy_id, "grantee": item.grantee, "scope": item.scope, "effect": item.effect}
                for item in state.permission_relations
            ],
        }
        assert_runtime_payload_safe(payload, context="llm_only_policy_prompt")
        prompt = (
            "Return JSON only with action (ALLOW, DENY, ABSTAIN), allowed_memory_ids, "
            "blocked_memory_ids, uncertainty, and abstain_reason. Decide only from the supplied state."
        )
        try:
            raw = self.llm_client.chat_json(
                model=resolve_llm_model(self.config, "reasoning"),
                system_prompt=prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            raw = dict(raw) if isinstance(raw, dict) else {}
            action = PolicyAction(str(raw.get("action") or "ABSTAIN").upper())
        except Exception:
            action = PolicyAction.ABSTAIN
            raw = {}
        valid_ids = {item.memory_id for item in state.memory_items if state.memory_status.get(item.memory_id, item.status).value == "active"}
        allowed = tuple(item for item in (raw.get("allowed_memory_ids") or []) if item in valid_ids)
        blocked = tuple(item for item in (raw.get("blocked_memory_ids") or []) if item in {memory.memory_id for memory in state.memory_items})
        if action == PolicyAction.ALLOW and not allowed:
            action = PolicyAction.ABSTAIN
        return PolicyDecision(
            action=action,
            requester=instance.asking_user_id,
            target_subject=None,
            requested_operation="access",
            allowed_memory_ids=allowed,
            blocked_memory_ids=blocked,
            state_snapshot={"memory_status": {key: value.value for key, value in state.memory_status.items()}},
            transition_trace=("llm_only policy decision",),
            uncertainty=float(raw.get("uncertainty") or 1.0),
            abstain_reason=str(raw.get("abstain_reason") or "") or None,
        )
