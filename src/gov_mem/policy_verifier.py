"""Final permission/privacy verification before a response is delivered."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.policy_schema import PolicyAction, PolicyDecision


_DELIVERY_ACTIONS = {"answer", "answer_redacted", "refuse", "no_memory"}
_SENSITIVE_FIELD_TERMS = {
    "diagnosis", "psychiatric", "psychiatry", "mental-health", "mental",
    "self-harm", "incident", "review flag", "customer identity",
    "customer mapping", "account number", "credential", "password",
    "passcode", "access code", "pin", "token", "private address",
}
_SENSITIVE_VALUE_TERMS = {
    "diagnos", "psychiatr", "mental-health", "self-harm", "pregnan",
    "miscarriage", "losing the pregnancy", "no heartbeat", "heartbeat",
    "medical condition", "clinical", "medication", "symptom", "disease",
    "blood pressure", "hormone", "credential", "password", "passcode",
    "access code", "private address", "account number",
}
@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    delivery_action: str
    reasons: tuple[str, ...] = ()
    symbolic_checks: tuple[str, ...] = ()
    llm_checked: bool = False
    llm_passed: bool | None = None
    claim_provenance: dict[str, Any] | None = None


_NON_DETERMINISTIC_FIELD_MARKERS = {
    "unknown", "unclear", "uncertain", "unavailable", "not available",
    "not found", "not provided", "not specified", "restricted", "redacted",
    "conflict", "conflicting", "cannot determine", "unable to determine",
}
_CLAIM_CONNECTORS = re.compile(
    r"\b(?:is|are|was|were|will be|has|have|had|at|on|from|until|located|"
    r"scheduled|available|assigned|uses|needs|requires|starts|ends)\b",
    re.IGNORECASE,
)
_SAFE_NON_CLAIM_PREFIXES = re.compile(
    r"^(?:i\s+(?:cannot|can't|can only|can provide)|that information|"
    r"under the current|under current|the requested information)",
    re.IGNORECASE,
)


def _normalized_tokens(value: object) -> list[str]:
    """Normalize a value for token-contiguous, punctuation-tolerant matching."""
    return re.findall(r"[\w]+", str(value or "").casefold(), flags=re.UNICODE)


def _phrase_occurs(text: object, phrase: object) -> bool:
    phrase_tokens = _normalized_tokens(phrase)
    text_tokens = _normalized_tokens(text)
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index:index + width] == phrase_tokens
        for index in range(max(0, len(text_tokens) - width + 1))
    )


def _field_label_occurs(answer_text: str, label: str) -> bool:
    label_tokens = _normalized_tokens(label)
    if label.strip() and re.search(
        rf"(?<!\w){re.escape(label.strip())}\s*:",
        str(answer_text or ""),
        re.IGNORECASE,
    ):
        return True
    # Very short labels such as "date" are too common in prose to establish
    # that a deterministic claim was made. A structured value claim still
    # passes through the selected-value checks below.
    return len(label_tokens) >= 2 and _phrase_occurs(answer_text, label)


def _field_has_non_deterministic_marker(answer_text: str, label: str) -> bool:
    if not _field_label_occurs(answer_text, label):
        return False
    answer_tokens = _normalized_tokens(answer_text)
    label_tokens = _normalized_tokens(label)
    for index in range(len(answer_tokens) - len(label_tokens) + 1):
        if answer_tokens[index:index + len(label_tokens)] != label_tokens:
            continue
        tail = " ".join(answer_tokens[index + len(label_tokens):index + len(label_tokens) + 12])
        if any(marker in tail for marker in _NON_DETERMINISTIC_FIELD_MARKERS):
            return True
    return False


def _claim_units(answer_text: str) -> list[str]:
    return [
        unit.strip()
        for unit in re.split(r"[.!?;]+", str(answer_text or ""))
        if unit.strip()
    ]


def _claim_provenance_check(
    *,
    allowed_memory_ids: set[str],
    evidence_payload: list[dict[str, Any]],
    answer_contract: dict[str, Any],
    answer_text: str,
) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    """Audit the final answer against the authorized field/value provenance closure.

    This is intentionally a structural verifier. It does not try to discover
    arbitrary facts in prose; field selection and temporal resolution have
    already happened upstream. Its hard gate covers every claim that the
    structured contract declares deliverable and every deterministic claim
    attached to a non-supported field.
    """
    projection = answer_contract.get("field_state_projection") or {}
    fields = list(projection.get("fields") or [])
    if not fields:
        fields = list(answer_contract.get("requested_fields") or [])
    rows_by_id = {
        str(row.get("memory_id")): row
        for row in evidence_payload
        if str(row.get("memory_id") or "")
    }
    audit: dict[str, Any] = {
        "enabled": bool(fields),
        "passed": True,
        "checked_fields": 0,
        "checked_claims": 0,
        "supported_claims": [],
        "unsupported_claims": [],
        "missing_provenance": [],
        "out_of_scope_sources": [],
        "ungrounded_values": [],
    }
    if not fields:
        return True, audit, [], ["claim_provenance_not_applicable"]

    reasons: list[str] = []
    checks = ["claim_provenance_enabled"]
    supported_values_all: list[str] = []
    supported_labels: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            continue
        audit["checked_fields"] += 1
        field_id = str(field.get("field_id") or field.get("label") or "unknown_field")
        label = str(field.get("label") or field_id)
        status = str(field.get("status") or "").casefold()
        supported = status in {"supported", "covered"}
        selected_values = [
            str(value).strip()
            for value in field.get("selected_values") or []
            if str(value).strip()
        ]
        if supported and not selected_values:
            fallback_value = str(field.get("answer_text") or "").strip()
            if fallback_value:
                selected_values = [fallback_value]
        source_ids = {
            str(value) for value in field.get("source_memory_ids") or []
            if str(value)
        }
        provenance = [item for item in field.get("provenance") or [] if isinstance(item, dict)]
        provenance_ids = {
            str(item.get("memory_id")) for item in provenance
            if str(item.get("memory_id") or "")
        }

        if not supported:
            # An explicit "unknown/restricted" answer is a status report, not
            # a recovered fact. A concrete label claim on such a field is not.
            if _field_label_occurs(answer_text, label) and not _field_has_non_deterministic_marker(answer_text, label):
                audit["unsupported_claims"].append({"field_id": field_id, "label": label})
                reasons.append(f"unsupported_field_claim:{label}")
            continue

        audit["checked_claims"] += len(selected_values)
        claim_record = {
            "field_id": field_id,
            "label": label,
            "values": list(selected_values),
            "source_memory_ids": sorted(source_ids),
        }
        audit["supported_claims"].append(claim_record)
        supported_values_all.extend(selected_values)
        supported_labels.append(label)
        if not source_ids:
            audit["missing_provenance"].append(field_id)
            reasons.append(f"claim_missing_source:{label}")
        if provenance and not source_ids.issuperset(provenance_ids):
            audit["missing_provenance"].append(field_id)
            reasons.append(f"claim_provenance_mismatch:{label}")
        out_of_scope = sorted(source_ids - allowed_memory_ids)
        if out_of_scope:
            audit["out_of_scope_sources"].append({"field_id": field_id, "memory_ids": out_of_scope})
            reasons.append(f"claim_source_outside_allowed_set:{label}")

        for value in selected_values:
            if not _phrase_occurs(answer_text, value):
                reasons.append(f"selected_value_missing_from_answer:{label}")
                continue
            source_rows = [rows_by_id[memory_id] for memory_id in source_ids if memory_id in rows_by_id]
            source_supported = any(_phrase_occurs(row.get("text"), value) for row in source_rows)
            if not source_supported:
                audit["ungrounded_values"].append({"field_id": field_id, "value": value})
                reasons.append(f"claim_value_not_supported_by_source:{label}")
            if provenance:
                span_supported = any(
                    _phrase_occurs(item.get("source_span"), value)
                    for item in provenance
                    if str(item.get("source_span") or "").strip()
                )
                # A missing span is allowed for legacy records, but an
                # explicitly supplied span must remain consistent with value.
                if any(str(item.get("source_span") or "").strip() for item in provenance) and not span_supported:
                    reasons.append(f"claim_value_outside_provenance_span:{label}")
        if selected_values:
            checks.append(f"claim_values_grounded:{field_id}")

    # The production projection renderer emits one labeled value fragment per
    # field. For legacy/free-form callers, catch an additional factual sentence
    # that is neither a selected value nor a non-deterministic status. This is
    # deliberately a conservative syntax check, not a domain vocabulary.
    if supported_values_all:
        for unit in _claim_units(answer_text):
            if _SAFE_NON_CLAIM_PREFIXES.search(unit):
                continue
            if any(_phrase_occurs(unit, value) for value in supported_values_all):
                continue
            if any(_field_label_occurs(unit, label) for label in supported_labels):
                continue
            if _CLAIM_CONNECTORS.search(unit):
                audit["unsupported_claims"].append({"text": unit})
                reasons.append("unsupported_answer_claim")

    if reasons:
        audit["passed"] = False
    else:
        checks.append("claim_provenance_verified")
    return not reasons, audit, reasons, checks


def _symbolic_checks(
    *,
    decision: PolicyDecision,
    evidence_payload: list[dict[str, Any]],
    answer_contract: dict[str, Any],
    delivery_action: str,
    answer_text: str,
    config: dict[str, Any],
) -> tuple[bool, list[str], list[str], dict[str, Any]]:
    allowed = set(decision.allowed_memory_ids)
    blocked = set(decision.blocked_memory_ids)
    accessed = {str(row.get("memory_id") or "") for row in evidence_payload}
    checks: list[str] = []
    reasons: list[str] = []
    passed = True

    if not accessed.issubset(allowed):
        passed = False
        reasons.append("evidence_outside_allowed_memory_set")
    else:
        checks.append("evidence_subset_of_allowed_set")
    if accessed.intersection(blocked):
        passed = False
        reasons.append("blocked_memory_reached_answer_stage")
    else:
        checks.append("blocked_memory_absent_from_answer_stage")
    if decision.state_snapshot.get("identity_ambiguous"):
        if accessed or delivery_action != "refuse":
            passed = False
            reasons.append("ambiguous_principal_was_not_refused")
        else:
            checks.append("ambiguous_principal_refused")

    # A partial disclosure is not allowed to reveal even the existence or
    # category of a restricted field when that field has no policy-approved
    # source.  This catches answers such as "the diagnosis is unchanged"
    # when only an access-status projection was allowed.  The check is
    # deliberately source-relative: a sensitive term is acceptable when it is
    # explicitly present in the evidence that policy allowed to the answer.
    if not bool(decision.state_snapshot.get("sensitive_authorized")):
        allowed_text = " ".join(str(row.get("text") or "").casefold() for row in evidence_payload)
        restricted_attributes = " ".join(
            str(value).casefold()
            for value in decision.state_snapshot.get("requested_attributes") or ()
        )
        leaked_field_terms = [
            term for term in sorted(_SENSITIVE_FIELD_TERMS)
            if term in restricted_attributes
            and re.search(rf"\b{re.escape(term)}\b", str(answer_text or "").casefold())
            and not re.search(rf"\b{re.escape(term)}\b", allowed_text)
        ]
        if leaked_field_terms:
            passed = False
            reasons.append("unauthorized_sensitive_field_mentioned:" + ",".join(leaked_field_terms))
        else:
            checks.append("unauthorized_sensitive_fields_not_mentioned")

        # Source provenance alone is not enough: a request/clarification can
        # be quoted by the answer projection as if it were a medical fact.
        # Fail closed when a supported projected value is itself sensitive.
        # This is field-level and value-driven, so ordinary authorized
        # logistics fields remain answerable for delegated requesters.
        projected_sensitive_fields = []
        for field in list(answer_contract.get("requested_fields") or []):
            status = str(field.get("status") or "").casefold()
            value = " ".join(str(field.get(key) or "") for key in ("answer_text", "selected_values"))
            value_lower = value.casefold()
            if status in {"covered", "supported"} and any(term in value_lower for term in _SENSITIVE_VALUE_TERMS):
                projected_sensitive_fields.append(str(field.get("label") or "unknown field"))
        projection = answer_contract.get("field_state_projection") or {}
        for field in list(projection.get("fields") or []):
            status = str(field.get("status") or "").casefold()
            values = " ".join(str(field.get(key) or "") for key in ("selected_values", "candidate_values"))
            if status == "supported" and any(term in values.casefold() for term in _SENSITIVE_VALUE_TERMS):
                projected_sensitive_fields.append(str(field.get("label") or "unknown field"))
        if projected_sensitive_fields:
            passed = False
            labels = ",".join(dict.fromkeys(projected_sensitive_fields))
            reasons.append("unauthorized_sensitive_projection:" + labels)

    fields = list(answer_contract.get("requested_fields") or [])
    for field in fields:
        source_ids = {str(value) for value in field.get("source_memory_ids") or []}
        if not source_ids.issubset(allowed):
            passed = False
            reasons.append(f"field_source_outside_allowed_set:{field.get('label')}")
    if fields:
        checks.append("field_provenance_is_policy_bounded")

    claim_enabled = bool((config.get("policy_verifier") or {}).get("claim_provenance_enabled", True))
    if claim_enabled:
        claim_passed, claim_audit, claim_reasons, claim_checks = _claim_provenance_check(
            allowed_memory_ids=allowed,
            evidence_payload=evidence_payload,
            answer_contract=answer_contract,
            answer_text=answer_text,
        )
    else:
        claim_passed = True
        claim_reasons = []
        claim_checks = ["claim_provenance_disabled_for_control"]
        claim_audit = {
            "enabled": False,
            "passed": True,
            "checked_fields": 0,
            "checked_claims": 0,
            "supported_claims": [],
            "unsupported_claims": [],
            "missing_provenance": [],
            "out_of_scope_sources": [],
            "ungrounded_values": [],
            "reason": "explicit_control_configuration",
        }
    checks.extend(claim_checks)
    reasons.extend(claim_reasons)
    if not claim_passed:
        passed = False

    if delivery_action in {"answer", "answer_redacted"} and not str(answer_text or "").strip():
        passed = False
        reasons.append("empty_delivered_answer")
    if decision.action in {PolicyAction.DENY, PolicyAction.ABSTAIN} and delivery_action == "answer":
        passed = False
        reasons.append("delivery_action_promoted_denied_decision")
    omitted_fields = tuple(
        str(value).strip().casefold()
        for value in answer_contract.get("restricted_fields_omitted") or ()
        if str(value).strip()
    )
    if decision.action == PolicyAction.ALLOW and delivery_action == "answer" and omitted_fields:
        # The answering model may conservatively call an ordinary uncovered
        # field "restricted" even when policy granted the complete request.
        # Treat that as an answer-completeness signal, not a privacy decision.
        # Only a partial policy projection or an actually sensitive omitted
        # field should force answer_redacted.
        sensitive_omitted = any(
            term in field
            for field in omitted_fields
            for term in _SENSITIVE_FIELD_TERMS
        )
        if bool(decision.state_snapshot.get("partial_disclosure")) or (
            sensitive_omitted and not bool(decision.state_snapshot.get("sensitive_authorized"))
        ):
            passed = False
            reasons.append("restricted_fields_omitted_but_action_not_redacted")
    return passed, checks, reasons, claim_audit


def _is_lifecycle_content_query(question: str) -> bool:
    return bool(re.search(
        r"\b(?:deleted|removed|retired|forgotten|old|previous|former|earlier|"
        r"provisional|preliminary|prior|before the current|before the approved|"
        r"used to|in pieces|fragment|reconstruct|recover|after .* removed|"
        r"before .* removed|before .* converted|before .* replaced)\b",
        str(question or "").lower(),
    ))


def _is_deleted_recovery_query(question: str) -> bool:
    """Identify safety-style requests to recover explicitly deleted content."""
    lowered = str(question or "").lower()
    return bool(
        re.search(r"\b(?:deleted|removed|retired|forgotten|purged|erased)\b", lowered)
        and _is_lifecycle_content_query(lowered)
    )


def _fallback_action(decision: PolicyDecision, *, question: str, current_action: str) -> str:
    if decision.state_snapshot.get("identity_ambiguous"):
        return "refuse"
    if decision.action == PolicyAction.DENY:
        statuses = {
            str(value).lower()
            for memory_id, value in (decision.state_snapshot.get("memory_status") or {}).items()
            if memory_id in set(decision.blocked_memory_ids)
        }
        return (
            "no_memory"
            if (
                _is_deleted_recovery_query(question)
                or (
                    statuses.intersection({"deleted", "forgotten", "inaccessible", "superseded"})
                    and _is_lifecycle_content_query(question)
                )
            )
            else "refuse"
        )
    if decision.action == PolicyAction.ABSTAIN:
        return "no_memory"
    return current_action if current_action in _DELIVERY_ACTIONS else "no_memory"


def verify_policy_delivery(
    *,
    question: str,
    decision: PolicyDecision,
    evidence_payload: list[dict[str, Any]],
    answer_contract: dict[str, Any],
    answer_text: str,
    delivery_action: str,
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> VerificationResult:
    """Run a symbolic hard gate and an optional base-LLM second opinion."""
    symbolic_passed, checks, reasons, claim_audit = _symbolic_checks(
        decision=decision,
        evidence_payload=evidence_payload,
        answer_contract=answer_contract,
        delivery_action=delivery_action,
        answer_text=answer_text,
        config=config,
    )
    verifier_cfg = dict(config.get("policy_verifier") or {})
    # Keep the verifier's second LLM opinion opt-in for callers that do not
    # provide the production config.  The production Stateful Policy config
    # explicitly enables it; unit tests and lightweight callers should not
    # accidentally route their answering mock through a verifier schema.
    llm_enabled = bool(verifier_cfg.get("llm_enabled", False))
    # The symbolic verifier is the hard safety gate.  The LLM verifier sees a
    # policy-approved projection and is useful for audit/review, but its
    # conservative free-form recommendation must not turn an authorized,
    # source-grounded utility answer into a refusal.  Callers can opt into
    # enforcement for controlled experiments with ``llm_advisory_only: false``.
    llm_advisory_only = bool(verifier_cfg.get("llm_advisory_only", True))
    llm_checked = False
    llm_passed: bool | None = None
    llm_action: str | None = None
    if (
        llm_enabled
        and llm_client is not None
        and llm_client.is_available()
        and symbolic_passed
        and decision.action == PolicyAction.ALLOW
    ):
        payload = {
            "question": question,
            "policy_decision": {
                "action": decision.action.value,
                "requester": decision.requester,
                "target_subject": decision.target_subject,
                "allowed_memory_ids": list(decision.allowed_memory_ids),
                "blocked_memory_count": len(decision.blocked_memory_ids),
                "sensitive_authorized": bool(decision.state_snapshot.get("sensitive_authorized")),
                "identity_ambiguous": bool(decision.state_snapshot.get("identity_ambiguous")),
            },
            "delivery_action": delivery_action,
            "answer_text": answer_text,
            "answer_contract": answer_contract,
            # Only policy-approved evidence is visible to this verifier.
            "allowed_evidence": evidence_payload,
        }
        assert_runtime_payload_safe(payload, context="policy_privacy_verifier_prompt")
        system_prompt = (
            "You are the final permission and privacy verifier. Return JSON only with "
            "passed, recommended_action, reasons. Do not answer the user. Verify that "
            "the delivery action follows the policy decision, each covered field has "
            "an allowed provenance source, no exact sensitive detail is delivered "
            "without sensitive_authorized, and an ambiguous identity is refused. "
            "You only see policy-approved evidence; do not request or infer blocked evidence. "
            "recommended_action must be answer, answer_redacted, refuse, or no_memory."
        )
        try:
            raw = llm_client.chat_json(
                model=resolve_llm_model(config, "answering"),
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            llm_checked = True
            llm_passed = bool(raw.get("passed")) if isinstance(raw, dict) else False
            candidate = str(raw.get("recommended_action") or "").strip().lower() if isinstance(raw, dict) else ""
            if candidate in _DELIVERY_ACTIONS:
                llm_action = candidate
            if isinstance(raw, dict):
                llm_reasons = raw.get("reasons") or []
                if isinstance(llm_reasons, str):
                    llm_reasons = [llm_reasons]
                reasons.extend(str(item) for item in llm_reasons if str(item).strip())
        except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
            llm_checked = False
            llm_passed = None

    passed = symbolic_passed
    final_action = _fallback_action(decision, question=question, current_action=delivery_action)
    if not symbolic_passed:
        final_action = "no_memory" if decision.action == PolicyAction.ALLOW else _fallback_action(decision, question=question, current_action=final_action)
    elif llm_advisory_only or llm_passed is not False:
        # In advisory mode the LLM result is retained in the audit trace, but
        # the delivery action remains the action produced by the structured
        # policy/execution path.  Symbolic checks above still fail closed.
        if llm_action and decision.action == PolicyAction.ALLOW:
            if not llm_advisory_only:
                final_action = llm_action
        passed = True
    else:
        passed = False
        if decision.action == PolicyAction.ALLOW:
            final_action = "no_memory"
    # A verifier may downgrade an allowed response, never promote a denied or
    # abstained decision into content delivery.
    if decision.action != PolicyAction.ALLOW:
        final_action = _fallback_action(decision, question=question, current_action=final_action)
    return VerificationResult(
        passed=passed,
        delivery_action=final_action,
        reasons=tuple(dict.fromkeys(reasons)),
        symbolic_checks=tuple(checks),
        llm_checked=llm_checked,
        llm_passed=llm_passed,
        claim_provenance=claim_audit,
    )
