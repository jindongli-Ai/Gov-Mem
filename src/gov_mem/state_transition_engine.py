"""Pure temporal replay of PolicyState for one query."""

from __future__ import annotations

from dataclasses import dataclass

from gov_mem.policy_conflict_resolver import resolve_permission
from gov_mem.general_lexicon import GENERAL_OBJECT_PREFIXES
from dataclasses import replace
import re

from gov_mem.policy_schema import MemoryStatus, OperationKind, OperationState, PolicyAction, PolicyState, QueryIntent
from gov_mem.policy_selector import ApplicablePolicies


@dataclass(frozen=True)
class TransitionResult:
    allowed_memory_ids: tuple[str, ...]
    blocked_memory_ids: tuple[str, ...]
    status_by_memory_id: dict[str, MemoryStatus]
    winning_policy_ids: tuple[str, ...]
    trace: tuple[str, ...]
    uncertainty: float
    related_allowed_memory_ids: tuple[str, ...] = ()
    related_blocked_memory_ids: tuple[str, ...] = ()
    explicit_related_blocked_memory_ids: tuple[str, ...] = ()
    allowed_reason_by_memory_id: dict[str, str] | None = None
    blocked_reason_by_memory_id: dict[str, str] | None = None
    role_capabilities: dict[str, list[str]] | None = None


def apply_execution_state_update(state: PolicyState, *, action: PolicyAction, memory_ids: tuple[str, ...], evidence_text: str = "") -> PolicyState:
    """Advance state only from a structured execution action."""
    statuses = dict(state.memory_status)
    if action == PolicyAction.DELETE:
        for memory_id in memory_ids:
            statuses[memory_id] = MemoryStatus.DELETED
        kind = OperationKind.DELETE
    elif action == PolicyAction.FORGET:
        for memory_id in memory_ids:
            statuses[memory_id] = MemoryStatus.FORGOTTEN
        kind = OperationKind.FORGET
    elif action == PolicyAction.UPDATE:
        for memory_id in memory_ids:
            statuses[memory_id] = MemoryStatus.ACTIVE
        kind = OperationKind.UPDATE
    else:
        kind = OperationKind.ACCESS
    operation = OperationState(
        operation_id=f"execution_{len(state.operation_history) + 1}",
        kind=kind,
        actor=None,
        target_memory_ids=memory_ids,
        effective_at=None,
        provenance=(),
    )
    memories = tuple(replace(item, status=statuses.get(item.memory_id, item.status)) for item in state.memory_items)
    return replace(state, memory_items=memories, memory_status=statuses, operation_history=state.operation_history + (operation,))


def replay_policy_state(
    state: PolicyState,
    intent: QueryIntent,
    applicable: ApplicablePolicies,
    *,
    default_owner_access: bool = True,
    enable_temporal_transition: bool = True,
    enable_conflict_resolution: bool = True,
    enable_memory_status: bool = True,
    enable_role_capabilities: bool = True,
    role_capabilities: dict[str, list[str]] | None = None,
) -> TransitionResult:
    """Compute effective access without mutating the source state."""
    allowed: list[str] = []
    blocked: list[str] = []
    allowed_reasons: dict[str, str] = {}
    blocked_reasons: dict[str, str] = {}
    winning: list[str] = []
    trace: list[str] = list(applicable.trace)
    uncertain = 0.0
    permissions_by_memory: dict[str, tuple] = {}
    for permission in applicable.permissions:
        for memory_id in permission.target_memory_ids:
            permissions_by_memory.setdefault(memory_id, tuple())
            permissions_by_memory[memory_id] = permissions_by_memory[memory_id] + (permission,)

    memory_by_id = {item.memory_id: item for item in state.memory_items}

    def inherited_subject_permissions(item):
        """Carry a policy only across an explicit, ordered state transition.

        Subject wording alone is not an object lineage. A single subject can
        have separate private, public, credential, and historical records.
        Permission inheritance therefore requires a later memory plus an
        observable update/share operation that targets the previously
        authorized record.
        """
        if not intent.target_subject:
            return ()

        def tokens(value: object) -> set[str]:
            return {
                token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(value or "").lower())
                if token not in {"current", "record", "memory", "state", "update", "after", "before"}
            }

        subject_tokens = tokens(intent.target_subject)
        item_tokens = tokens(" ".join((item.subject or "", item.provenance.evidence_text or "")))
        if subject_tokens and not subject_tokens.intersection(item_tokens):
            return ()
        item_turn = item.provenance.turn_index
        inherited = []
        for permission in applicable.permissions:
            for target_id in permission.target_memory_ids:
                target = memory_by_id.get(target_id)
                if target is None:
                    continue
                target_tokens = tokens(" ".join((target.subject or "", target.provenance.evidence_text or "")))
                if subject_tokens and not subject_tokens.intersection(target_tokens):
                    continue
                target_turn = target.provenance.turn_index
                if item_turn is None or target_turn is None or item_turn <= target_turn:
                    continue
                linked = any(
                    operation.kind == OperationKind.UPDATE
                    and target_id in operation.target_memory_ids
                    and operation.provenance.turn_index is not None
                    and operation.provenance.turn_index < item_turn
                    and (
                        not ((item_tokens & target_tokens) - subject_tokens)
                        or bool(
                            ((item_tokens & target_tokens) - subject_tokens)
                            & tokens(operation.provenance.evidence_text)
                        )
                    )
                    for operation in state.operation_history
                )
                if linked:
                    inherited.append(permission)
                    break
        return tuple(dict.fromkeys(inherited))

    requester_role = next(
        (principal.role for principal in state.principals if principal.principal_id == intent.requester),
        None,
    )
    # Role capabilities are a retrieval/candidate hint only.  A job title is
    # not an authorization chain for a concrete memory, so it must never turn
    # an unknown permission state into ALLOW.
    capabilities = role_capabilities or {}
    allowed_topics = set(capabilities.get(str(requester_role or "").lower(), []))
    requester_text = str(intent.requester or "").lower()
    requester_revoked = any(
        permission.effect == "deny"
        and (
            permission.grantee == intent.requester
            or (permission.target_subject and requester_text in permission.target_subject.lower())
        )
        for permission in applicable.permissions
    )
    requester_revoked = requester_revoked or (
        str(requester_role or "").lower() == "family_member"
        and any(
            operation.kind.value == "revoke"
            and (requester_text in operation.provenance.evidence_text.lower() or "family access revoked" in operation.provenance.evidence_text.lower())
            for operation in state.operation_history
        )
    )

    target_principal_ids = set(intent.mentioned_principal_ids or ())

    def item_matches_target_subject(item) -> bool:
        """Require explicit person binding when the query names a person."""
        if target_principal_ids:
            bound = set(item.subject_principal_ids or ())
            if bound:
                return bool(bound.intersection(target_principal_ids))
            # Do not infer a person from a bare role or owner.  A lexical
            # fallback is permitted only for records that visibly name the
            # uniquely resolved target principal.
            target_names = [
                principal.display_name
                for principal in state.principals
                if principal.principal_id in target_principal_ids and principal.display_name
            ]
            text = " ".join((item.subject or "", item.provenance.evidence_text or "")).lower()
            return any(str(name).lower() in text for name in target_names)
        if not intent.target_subject:
            return True
        lowered_target = str(intent.target_subject).lower()
        text = " ".join((item.subject or "", item.provenance.evidence_text or "")).lower()
        if lowered_target in text:
            return True
        # Shared-memory shorthand often drops a generic object prefix in
        # later messages (``Project Maple`` -> ``Maple``).  Recover only this
        # observable alias form; never match a substring such as Maplemark.
        target_tokens = [
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", lowered_target)
        ]
        generic_prefixes = GENERAL_OBJECT_PREFIXES
        if len(target_tokens) >= 2 and target_tokens[0] in generic_prefixes:
            source_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text))
            return all(token in source_tokens for token in target_tokens[1:])
        return False

    def relevance(item) -> float:
        """Weak semantic candidate signal, deliberately separate from auth."""
        query_text = " ".join(prov.evidence_text for prov in intent.provenance if prov.evidence_text)
        query = " ".join(
            [query_text, intent.target_subject or "", intent.target_scope or "", *intent.requested_topics,
             *intent.requested_attributes, *(intent.mentioned_entities or [])]
        )
        text = " ".join([item.subject or "", item.scope or "", item.provenance.evidence_text or "", *item.topics])
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "where", "when",
            "how", "can", "could", "would", "please", "tell", "me", "give", "for", "and", "or", "to",
            "of", "on", "in", "with", "this", "that", "it", "they", "does", "do", "about", "current",
            "information", "memory", "record", "request", "access", "shared", "private", "confidential",
            "answer", "yes", "only", "need", "still", "tell", "send", "finish",
        }
        token = lambda value: {
            word for word in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(value or "").lower())
            if word not in stop
        }
        query_tokens = token(query)
        text_tokens = token(text)
        subject_hit = bool(intent.target_subject and str(intent.target_subject).lower() in text.lower())
        entity_hit = any(
            len(entity.strip()) > 2 and entity.lower() in text.lower()
            for entity in intent.mentioned_entities
        )
        if subject_filter_active and not item_matches_target_subject(item):
            return 0.0
        overlap = len(query_tokens & text_tokens) / max(1, len(query_tokens))
        topic_hit = bool(set(intent.requested_topics) & set(item.topics))
        return min(1.0, overlap + (0.65 if subject_hit else 0.0) + (0.45 if entity_hit else 0.0) + (0.15 if topic_hit else 0.0))

    def relation_scope_tokens(scope_id: str) -> set[str]:
        return {
            token for token in re.split(r"[_\W]+", scope_id.lower())
            if len(token) > 2 and token not in {"project", "program", "case", "course", "class", "team", "group", "user", "principal"}
        }

    def relation_scopes(principal_id: str | None) -> set[tuple[str, str, str]]:
        if not principal_id:
            return set()
        result: set[tuple[str, str, str]] = set()
        endpoint_keys = {
            "principal_id", "grantee", "guest_id", "member_id", "resident_id",
            "owner_id", "user_id", "agent_id", "delegate_id", "contractor_id",
        }
        for relation in state.scope_constraints:
            if not isinstance(relation, dict):
                continue
            endpoints = {
                str(relation.get(key) or "")
                for key in endpoint_keys
                if str(relation.get(key) or "")
            }
            if principal_id not in endpoints:
                continue
            relation_type = str(relation.get("type") or "").lower()
            for key, value in relation.items():
                if key.endswith("_id") and key not in {"principal_id", "for_principal_id", "owner_id", "grantee_id", "grantor_id"}:
                    result.add((key, str(value), relation_type))
            access_scope = str(relation.get("access_scope") or "").strip()
            if access_scope:
                result.add(("access_scope", access_scope, relation_type))
        return result

    requester_scopes = relation_scopes(intent.requester)

    def subject_linked_owner_allow(item) -> tuple[bool, str | None]:
        """Use an observable case/care relationship as an auth path."""
        requester_id = str(intent.requester or "")
        if not requester_id or not item.owner:
            return False, None
        target_keys = {"patient_id", "subject_id", "case_subject_id", "for_principal_id"}
        owner_keys = {
            "principal_id", "clinician_id", "advisor_id", "registrar_id",
            "case_owner_id", "owner_id",
        }
        for relation in state.scope_constraints:
            if not isinstance(relation, dict):
                continue
            if requester_id not in {str(relation.get(key) or "") for key in target_keys}:
                continue
            owners = {str(relation.get(key) or "") for key in owner_keys}
            if item.owner not in owners:
                continue
            relation_type = str(relation.get("type") or "observable_case_relation")
            return True, f"subject-linked relation:{relation_type}"
        has_subject_case = any(
            isinstance(relation, dict)
            and requester_id in {str(relation.get(key) or "") for key in target_keys}
            for relation in state.scope_constraints
        )
        owner_role = next(
            (str(principal.role or "").lower() for principal in state.principals if principal.principal_id == item.owner),
            "",
        )
        if has_subject_case and owner_role in {
            "clinician", "nurse", "pharmacist", "social_worker", "lab_tech",
            "scheduler", "reception", "care_coordinator",
        }:
            return True, "subject-linked care context"
        return False, None

    def shared_case_owner_allow(item) -> tuple[bool, str | None]:
        """Authorize through two observable principals on one concrete case.

        A case-owner relationship and a support relationship are state
        evidence, not blanket role grants. The requester and the record owner
        must share the same case/project/program relation.
        """
        requester_id = str(intent.requester or "")
        if not requester_id or not item.owner:
            return False, None
        target_subject_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(intent.target_subject or "").lower())
            if token not in {"current", "record", "summary", "memory"}
        }
        item_subject_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(item.subject or "").lower())
            if token not in {"current", "record", "summary", "memory"}
        }
        if (
            len(item_subject_tokens) >= 2
            and target_subject_tokens.intersection(item_subject_tokens)
            and not target_subject_tokens.issubset(item_subject_tokens)
        ):
            return False, None

        def relation_pairs(principal_id: str) -> tuple[set[tuple[str, str]], set[str]]:
            pairs: set[tuple[str, str]] = set()
            types: set[str] = set()
            for relation in state.scope_constraints:
                if not isinstance(relation, dict):
                    continue
                if str(relation.get("principal_id") or relation.get("grantee") or "") != principal_id:
                    continue
                types.add(str(relation.get("type") or "").lower())
                for key, value in relation.items():
                    if key.endswith("_id") and key not in {"principal_id", "grantee_id", "grantor_id", "owner_id", "for_principal_id"}:
                        pairs.add((key, str(value)))
            return pairs, types

        requester_pairs, requester_types = relation_pairs(requester_id)
        owner_pairs, _ = relation_pairs(str(item.owner))
        shared = requester_pairs & owner_pairs
        owner_like = any("owner" in relation_type for relation_type in requester_types)
        object_keys = {"case_id", "project_id", "program_id", "thread_id", "account_id"}
        matching = sorted((key, value) for key, value in shared if key in object_keys)
        if owner_like and matching:
            key, value = matching[0]
            return True, f"shared case-owner relation:{key}={value}"

        # A case owner may be related to the case through its subject
        # principal (for example ``for_principal_id``), while a support owner
        # carries the concrete ``case_id``. Resolve that bridge only through
        # the observable relationship graph and the requested subject.
        subject_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(intent.target_subject or "").lower())
            if token not in {"current", "record", "summary", "memory"}
        }
        target_objects: set[str] = set()
        subject_principals: set[str] = set()
        for relation in state.scope_constraints:
            if not isinstance(relation, dict):
                continue
            relation_values = " ".join(str(value) for key, value in relation.items() if key.endswith("_id")).lower()
            relation_tokens = {
                token for token in re.split(r"[_\W]+", relation_values) if len(token) > 2
            }
            if subject_tokens and not subject_tokens.issubset(relation_tokens):
                continue
            subject_principal = str(relation.get("principal_id") or "")
            if subject_principal:
                subject_principals.add(subject_principal)
            for key, value in relation.items():
                if key in object_keys:
                    target_objects.add(str(value))
        requester_for_subject = any(
            str(relation.get("principal_id") or "") == requester_id
            and str(relation.get("type") or "").lower().find("owner") >= 0
            and str(relation.get("for_principal_id") or "") in subject_principals
            for relation in state.scope_constraints
            if isinstance(relation, dict)
        )
        if requester_for_subject and subject_tokens:
            item_text_tokens = {
                token for token in re.findall(
                    r"[a-z0-9][a-z0-9_-]{2,}",
                    " ".join((item.subject or "", item.provenance.evidence_text or "")).lower(),
                )
            }
            if subject_tokens.issubset(item_text_tokens):
                # A registrar/case owner may read records explicitly tied to
                # the observable case even when those records are authored by
                # different operational owners.  This is a concrete
                # case-owner chain, not a role-wide grant.
                return True, "subject-bridged case-owner relation:target_subject"
        owner_object_values = {
            value for key, value in owner_pairs if key in object_keys
        }
        if requester_for_subject and target_objects.intersection(owner_object_values):
            value = sorted(target_objects.intersection(owner_object_values))[0]
            return True, f"subject-bridged case-owner relation:case={value}"
        return False, None

    def shared_scope_allow(item) -> tuple[bool, str | None]:
        """Observable relationship authorization for shared work objects."""
        # An explicitly public projection is independently shareable.  It is
        # not a blanket grant for nearby private records: only the concrete
        # memory whose observable evidence marks it public is admitted.
        item_text = str(item.provenance.evidence_text or "")
        query_text = " ".join(prov.evidence_text for prov in intent.provenance if prov.evidence_text)
        if (
            status == MemoryStatus.ACTIVE
            and re.search(r"\bpublic\b", item_text, re.IGNORECASE)
            and re.search(r"\bpublic\b", query_text, re.IGNORECASE)
        ):
            return True, "explicit public disclosure"
        if (
            intent.target_scope in {"public", "safe_summary", "broad"}
            and status == MemoryStatus.ACTIVE
            and (item.scope in {"public", "broad", "safe_summary"} or "public_projection" in item.topics)
        ):
            return True, "explicit public disclosure projection"
        if not requester_scopes or not item.owner:
            return False, None
        owner_scopes = relation_scopes(item.owner)
        requester_scope_pairs = {(key, value) for key, value, _ in requester_scopes}
        owner_scope_pairs = {(key, value) for key, value, _ in owner_scopes}
        shared_pairs = requester_scope_pairs & owner_scope_pairs
        # Project membership and equivalent team/care relationships can expose
        # ordinary operational facts. Contractor, visitor, assistant, and
        # delegate relationships remain narrow unless an explicit permission
        # record authorizes the concrete memory/scope.
        broad_relation_types = {
            "project_member", "team_member", "case_team", "care_team",
            "household_member", "resident", "course_staff", "instructor",
            "advisor", "budget_owner", "legal_reviewer", "executive_sponsor",
        }
        requester_types = {
            relation_type for key, value, relation_type in requester_scopes
            if (key, value) in shared_pairs
        }
        if not (requester_types & broad_relation_types):
            # Narrow operational delegations can still authorize the concrete
            # operational capability named by their observable access_scope.
            # This does not grant arbitrary records: the requester and record
            # owner must share an object token and the item must mention the
            # owner's capability token.
            requester_access = [value for key, value, _ in requester_scopes if key == "access_scope"]
            owner_access = [value for key, value, _ in owner_scopes if key == "access_scope"]
            generic = {"only", "and", "the", "for", "access", "scope", "current"}
            # A narrow delegate's capability is matched against the semantic
            # category of the concrete record, not only against a literal
            # token copied from the owner's scope.  The latter was too strict
            # for ordinary operational records: a logistics delegate should
            # be able to read a delivery time/location owned by a guest whose
            # scope says ``produce_drop``.  This remains object-bounded and
            # never grants records merely because they share a household.
            capability_topic_aliases = {
                "logistics": {"logistics", "scheduling", "location", "transport", "household"},
                "scheduling": {"logistics", "scheduling", "location"},
                "communication": {"communication"},
                "watering": {"household", "environment", "location"},
                "produce": {"food", "household", "logistics"},
                "drop": {"logistics", "location", "household"},
            }
            sensitive_topics = {
                "access_control", "privacy", "identity", "medical", "health",
                "laboratory", "imaging", "finance", "economics", "legal",
            }
            item_topics = set(item.topics or ())
            item_text = " ".join((item.subject or "", item.provenance.evidence_text or "")).lower()
            for requester_scope in requester_access:
                requester_tokens = relation_scope_tokens(requester_scope)
                for owner_scope in owner_access:
                    owner_tokens = relation_scope_tokens(owner_scope)
                    shared_object = (requester_tokens & owner_tokens) - {"logistics", "scheduling", "communication"}
                    requester_capability = (requester_tokens - shared_object) - generic
                    owner_capability = (owner_tokens - shared_object) - generic
                    if not shared_object or not requester_capability:
                        continue
                    if item_topics & sensitive_topics:
                        continue
                    literal_capability = (
                        requester_capability | owner_capability
                    ) & set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", item_text))
                    semantic_capability = set().union(*(
                        capability_topic_aliases.get(token, {token})
                        for token in requester_capability
                    ))
                    if literal_capability or item_topics.intersection(semantic_capability):
                        return True, "shared observable access_scope capability"
            return False, None
        if not shared_pairs:
            return False, None
        text = " ".join([intent.target_subject or "", item.subject or "", item.provenance.evidence_text or ""]).lower()
        for key, scope_id in sorted(shared_pairs):
            tokens = relation_scope_tokens(scope_id)
            if tokens and any(token in text for token in tokens):
                return True, f"shared observable {key}={scope_id}"
        return False, None

    def role_capability_allow(item) -> tuple[bool, str | None]:
        """Apply explicit role capability only to matching active topics.

        This is a bounded RBAC candidate path: the role, requested topic, and
        concrete memory topic must all agree. It does not authorize a role to
        read every record, and lifecycle/deny resolution still runs first.
        """
        if not enable_role_capabilities or status != MemoryStatus.ACTIVE:
            return False, None
        role_topics = {str(topic).lower() for topic in allowed_topics}
        if "*" not in role_topics and not role_topics.intersection(set(intent.requested_topics)):
            return False, None
        item_topics = {str(topic).lower() for topic in (item.topics or ())}
        if "*" not in role_topics and not role_topics.intersection(item_topics):
            return False, None
        if intent.target_scope and item.scope and intent.target_scope not in {item.scope, "private"}:
            return False, None
        if relevance(item) <= 0.0:
            return False, None
        return True, f"role-capability:{str(requester_role or '').lower()}"

    # A semantic parser may name the principal (for example
    # ``patient_nora_diaz``) even when the request itself contains no concrete
    # entity mention. Only activate exact subject filtering if the parsed
    # subject is actually grounded in the observable memory catalog.
    # A compound request can intentionally span several observable subjects
    # (for example a private case value plus a public logistics room).  Keep
    # authorization unchanged, but do not let one singular parsed subject
    # discard other authorized fields before content retrieval.
    compound_request = len(tuple(intent.requested_attributes or ())) >= 3
    subject_filter_active = bool(
        (target_principal_ids or intent.target_subject)
        and (target_principal_ids or not compound_request)
        and any(item_matches_target_subject(item) for item in state.memory_items)
    )

    def owner_related_to_target(item) -> bool:
        """Ground records whose owner is linked to the requested object."""
        if not intent.target_subject or not item.owner:
            return False
        target_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(intent.target_subject).lower())
            if token not in {"current", "record", "summary", "memory"}
        }
        item_subject_tokens = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(item.subject or "").lower())
            if token not in {"current", "record", "summary", "case", "memory"}
        }
        if (
            len(item_subject_tokens) >= 2
            and target_tokens.intersection(item_subject_tokens)
            and not target_tokens.issubset(item_subject_tokens)
        ):
            return False
        for relation in state.scope_constraints:
            if not isinstance(relation, dict) or str(relation.get("principal_id") or "") != item.owner:
                continue
            for key, value in relation.items():
                if not key.endswith("_id") or key in {"principal_id", "for_principal_id", "owner_id"}:
                    continue
                relation_tokens = {
                    token for token in re.split(r"[_\W]+", str(value).lower())
                    if len(token) > 2
                }
                if target_tokens and target_tokens.issubset(relation_tokens):
                    return True
        return False

    for item in state.memory_items:
        if intent.target_memory_ids and item.memory_id not in intent.target_memory_ids:
            continue
        status = state.memory_status.get(item.memory_id, item.status) if enable_memory_status else MemoryStatus.ACTIVE
        permissions = permissions_by_memory.get(item.memory_id, ())
        # A permission with an explicit target id is local to that target. It
        # must never become a global deny/allow for unrelated memories.
        if not permissions:
            permissions = tuple(
                permission for permission in applicable.permissions
                if not permission.target_memory_ids
                and (
                    not permission.scope
                    or permission.scope in set(intent.requested_topics)
                    or permission.scope == intent.target_scope
                )
            )
        if not permissions:
            permissions = inherited_subject_permissions(item)
            if permissions:
                trace.append(
                    f"subject-policy lineage inherited for {item.memory_id}: "
                    f"{[permission.policy_id for permission in permissions]}"
                )
        if not enable_temporal_transition and permissions:
            permissions = tuple(sorted(permissions, key=lambda value: value.valid_from or "")[:1])
        if not enable_conflict_resolution and permissions:
            permissions = permissions[-1:]
        resolution = resolve_permission(
            permissions,
            memory_status=status,
            requester=intent.requester,
        )
        effect = resolution.effect
        reason = resolution.winning_policy_id or ""
        if effect == "unknown" and default_owner_access and item.owner == intent.requester and status == MemoryStatus.ACTIVE:
            effect = "allow"
            reason = "owner"
            trace.append(f"owner access allowed for {item.memory_id}")
        if effect == "unknown" and status == MemoryStatus.ACTIVE:
            shared_allowed, shared_reason = shared_scope_allow(item)
            if shared_allowed:
                effect = "allow"
                reason = shared_reason or "shared_scope"
                trace.append(f"shared-scope access allowed for {item.memory_id}: {reason}")
        if effect == "unknown" and status == MemoryStatus.ACTIVE:
            case_allowed, case_reason = shared_case_owner_allow(item)
            if case_allowed:
                effect = "allow"
                reason = case_reason or "shared_case_owner"
                trace.append(f"case-owner access allowed for {item.memory_id}: {reason}")
        if effect == "unknown" and status == MemoryStatus.ACTIVE:
            linked_allowed, linked_reason = subject_linked_owner_allow(item)
            if linked_allowed:
                effect = "allow"
                reason = linked_reason or "subject_linked_relation"
                trace.append(f"subject-linked access allowed for {item.memory_id}: {reason}")
        if effect == "unknown" and status == MemoryStatus.ACTIVE:
            role_allowed, role_reason = role_capability_allow(item)
            if role_allowed:
                effect = "allow"
                reason = role_reason or "role_capability"
                trace.append(f"role-capability access allowed for {item.memory_id}: {reason}")
        is_relevant = relevance(item) > 0.0
        explicitly_targeted = bool(permissions_by_memory.get(item.memory_id))
        # Authorization is resolved independently of content relevance.  The
        # latter only limits the decision's candidate set after permission has
        # been computed; it never creates a grant.
        # Owner authorization is independent of lexical relevance. A
        # shorthand update such as "Saturday tag color is brick" can be a
        # valid owner record even when it does not repeat the target object;
        # content retrieval remains responsible for selecting answer
        # evidence after this authorization boundary.
        owner_authorized = bool(
            default_owner_access
            and item.owner == intent.requester
            and status == MemoryStatus.ACTIVE
        )
        candidate = owner_authorized or explicitly_targeted or is_relevant or (
            intent.target_scope == "safe_summary"
            and status == MemoryStatus.ACTIVE
            and (item.scope in {"public", "broad", "safe_summary"} or "public_projection" in item.topics)
        ) or (
            item.owner == intent.requester and not intent.requested_topics and not intent.target_subject
        ) or reason in {"shared observable access_scope capability", "role_capability"} or str(reason).startswith((
            "role-capability:", "subject-linked", "shared case-owner", "subject-bridged",
        ))
        # A direct owner's later continuation may omit the object phrase that
        # appeared in the original record (for example, "the exact note
        # remains ..." after an earlier "Thursday note").  Subject filtering
        # must not turn that valid owner record into an inaccessible record;
        # content retrieval still performs field-local binding inside the
        # already authorized owner set.  Keep strict filtering for records
        # owned by other principals so this exception cannot broaden sharing.
        explicit_shared_or_linked_authorization = bool(
            str(reason).startswith((
                "shared observable",
                "shared case-owner",
                "subject-linked",
                "subject-bridged",
                "role-capability:",
            ))
        )
        if (
            subject_filter_active
            and not item_matches_target_subject(item)
            and item.owner != intent.requester
            and not explicit_shared_or_linked_authorization
        ):
            candidate = False
        if effect == "allow" and status == MemoryStatus.ACTIVE and candidate:
            allowed.append(item.memory_id)
            allowed_reasons[item.memory_id] = reason or "permission"
        elif (effect in {"deny", "unknown"} or status != MemoryStatus.ACTIVE) and candidate:
            blocked.append(item.memory_id)
            if status != MemoryStatus.ACTIVE:
                blocked_reasons[item.memory_id] = f"lifecycle:{status.value}"
            elif effect == "deny":
                blocked_reasons[item.memory_id] = reason or "explicit_deny"
            else:
                blocked_reasons[item.memory_id] = "unknown_authorization"
            if effect == "unknown" and status == MemoryStatus.ACTIVE:
                # Absence of an owner/grant/delegation chain is uncertainty,
                # not a silent denial. The reasoner converts this to ABSTAIN.
                uncertain += 1.0
        if resolution.winning_policy_id:
            winning.append(resolution.winning_policy_id)
        trace.extend(f"{item.memory_id}: {line}" for line in resolution.trace)

    if not allowed and not blocked:
        uncertain = 1.0
        trace.append("no target memory matched the intent")
    related_allowed = tuple(item_id for item_id in allowed if relevance(next(item for item in state.memory_items if item.memory_id == item_id)) > 0.0)
    related_blocked = tuple(item_id for item_id in blocked if relevance(next(item for item in state.memory_items if item.memory_id == item_id)) > 0.0)
    redaction_relevant_blocked = tuple(
        item_id for item_id in related_blocked
        if str(blocked_reasons.get(item_id, "")) == "explicit_deny"
    )
    return TransitionResult(
        allowed_memory_ids=tuple(dict.fromkeys(allowed)),
        blocked_memory_ids=tuple(dict.fromkeys(blocked)),
        status_by_memory_id={item.memory_id: state.memory_status.get(item.memory_id, item.status) for item in state.memory_items},
        winning_policy_ids=tuple(dict.fromkeys(winning)),
        trace=tuple(trace),
        uncertainty=min(1.0, uncertain),
        related_allowed_memory_ids=related_allowed,
        related_blocked_memory_ids=related_blocked,
        explicit_related_blocked_memory_ids=redaction_relevant_blocked,
        allowed_reason_by_memory_id=allowed_reasons,
        blocked_reason_by_memory_id=blocked_reasons,
        role_capabilities=role_capabilities or {},
    )
