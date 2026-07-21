from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import md5
from typing import Any

from gov_mem.experience.failure_case import FailureCase


@dataclass
class FailurePattern:
    pattern_id: str
    failure_type: str
    description: str
    trigger_signature: dict[str, Any]
    affected_domains: list[str]
    affected_roles: list[str]
    affected_slots: list[str]
    recommended_rule: str
    recommended_skill: str
    support_cases: list[str]
    confidence: float
    created_from_dev_only: bool = True
    provenance: dict[str, Any] = field(default_factory=dict)


class PatternInducer:
    def induce(
        self,
        *,
        failure_cases: list[FailureCase],
        min_support: int = 2,
    ) -> list[FailurePattern]:
        grouped: dict[tuple[str, str, str, str], list[FailureCase]] = {}
        for case in failure_cases:
            signature = _structural_signature(case)
            key = (
                str(case.failure_type),
                signature["lifecycle_state"],
                signature["authorization_state"],
                signature["evidence_state"],
            )
            grouped.setdefault(key, []).append(case)

        patterns: list[FailurePattern] = []
        for (failure_type, lifecycle_state, authorization_state, evidence_state), cases in sorted(grouped.items()):
            if len(cases) < min_support:
                continue
            affected_roles = _collect_roles(cases)
            affected_slots = _collect_slots(cases)
            trigger_signature = {
                "lifecycle_state": lifecycle_state,
                "authorization_state": authorization_state,
                "evidence_state": evidence_state,
                "slot_policy_state": "sensitive" if _has_denied_slots(cases) else "ordinary",
                "predicted_actions": sorted({case.predicted_action for case in cases if case.predicted_action}),
                "symbolic_actions": sorted(
                    {
                        str((case.symbolic_decision or {}).get("action_constraint") or "")
                        for case in cases
                        if str((case.symbolic_decision or {}).get("action_constraint") or "").strip()
                    }
                ),
            }
            recommended_rule, recommended_skill = _recommend_pattern_fix(
                failure_type=failure_type,
                lifecycle_state=lifecycle_state,
                authorization_state=authorization_state,
                evidence_state=evidence_state,
            )
            pattern_id = md5(
                f"{failure_type}:{lifecycle_state}:{authorization_state}:{evidence_state}:{'|'.join(affected_slots[:6])}".encode("utf-8")
            ).hexdigest()[:12]
            support_cases = [case.case_id for case in cases]
            patterns.append(
                FailurePattern(
                    pattern_id=pattern_id,
                    failure_type=failure_type,
                    description=_describe_pattern(
                        failure_type=failure_type,
                        lifecycle_state=lifecycle_state,
                        authorization_state=authorization_state,
                        evidence_state=evidence_state,
                        support_count=len(cases),
                        affected_slots=affected_slots,
                    ),
                    trigger_signature=trigger_signature,
                    affected_domains=[],
                    affected_roles=affected_roles,
                    affected_slots=affected_slots,
                    recommended_rule=recommended_rule,
                    recommended_skill=recommended_skill,
                    support_cases=support_cases,
                    confidence=min(0.6 + 0.05 * len(cases), 0.95),
                    provenance={
                        "created_from_dev_only": True,
                        "support_count": len(cases),
                        "support_case_ids": support_cases,
                    },
                )
            )
        return patterns


def failure_pattern_to_dict(pattern: FailurePattern) -> dict[str, Any]:
    return asdict(pattern)


def _collect_roles(cases: list[FailureCase]) -> list[str]:
    roles: list[str] = []
    for case in cases:
        for group in case.retrieved_governance_evidence or []:
            if not isinstance(group, dict):
                continue
            for key in ("roles", "relations", "policies"):
                for row in group.get(key) or []:
                    if not isinstance(row, dict):
                        continue
                    for scope in row.get("access_scope") or []:
                        scope_text = str(scope).strip()
                        if scope_text:
                            roles.append(scope_text)
    return list(dict.fromkeys(roles))[:8]


def _collect_slots(cases: list[FailureCase]) -> list[str]:
    slots: list[str] = []
    for case in cases:
        for slot_name in list((case.symbolic_decision or {}).get("allowed_slots") or []):
            slots.append(str(slot_name))
        for slot_name in list((case.symbolic_decision or {}).get("denied_slots") or []):
            slots.append(str(slot_name))
        for row in case.retrieved_utility_evidence or []:
            if isinstance(row, dict):
                for slot_name in dict(row.get("slots") or {}).keys():
                    slots.append(str(slot_name))
    return list(dict.fromkeys(slot for slot in slots if slot))[:12]


def _recommend_pattern_fix(
    *,
    failure_type: str,
    lifecycle_state: str,
    authorization_state: str,
    evidence_state: str,
) -> tuple[str, str]:
    if failure_type in {"deleted_reconstruction", "stale_state"} or lifecycle_state != "active_only":
        return "enforce_versioned_lifecycle_precedence", "lifecycle_integrity_skill"
    if failure_type in {"leakage", "under_redaction"}:
        return "enforce_slot_level_authorization_projection", "authorization_projection_skill"
    if failure_type in {"missing_utility", "no_memory_collapse"} and authorization_state == "authorized":
        return "realize_all_authorized_typed_slots", "typed_utility_realization_skill"
    if failure_type in {"missing_utility", "no_memory_collapse"} or evidence_state != "complete":
        return "require_complete_provenance_before_realization", "provenance_completion_skill"
    return "calibrate_restrictive_action_from_memory_state", "restrictive_action_calibration_skill"


def _describe_pattern(
    *,
    failure_type: str,
    lifecycle_state: str,
    authorization_state: str,
    evidence_state: str,
    support_count: int,
    affected_slots: list[str],
) -> str:
    slot_text = ", ".join(affected_slots[:4]) if affected_slots else "unspecified slots"
    return (
        f"{failure_type} under lifecycle={lifecycle_state}, authorization={authorization_state}, "
        f"evidence={evidence_state}; observed in {support_count} dev cases around {slot_text}."
    )


def _structural_signature(case: FailureCase) -> dict[str, str]:
    symbolic = dict(case.symbolic_decision or {})
    allowed = {str(item) for item in symbolic.get("allowed_slots") or [] if str(item)}
    denied = {str(item) for item in symbolic.get("denied_slots") or [] if str(item)}
    rows = [row for row in case.retrieved_utility_evidence or [] if isinstance(row, dict)]
    lifecycle_tokens = " ".join(
        str(value).lower()
        for row in rows
        for value in (row.get("lifecycle_status"), dict(row.get("metadata") or {}).get("memory_status"))
        if value
    )
    if "deleted" in lifecycle_tokens:
        lifecycle_state = "contains_deleted"
    elif "superseded" in lifecycle_tokens or "canceled" in lifecycle_tokens:
        lifecycle_state = "contains_superseded"
    else:
        lifecycle_state = "active_only"
    action = str(symbolic.get("action_constraint") or "").lower()
    authorization_state = "authorized" if allowed and action in {"answer", "answer_redacted"} else "restricted" if denied else "undetermined"
    available = {
        str(slot)
        for row in rows
        for slot in dict(row.get("slots") or {}).keys()
        if str(slot)
    }
    requested = allowed | denied
    evidence_state = "complete" if requested and requested.issubset(available) else "partial"
    return {"lifecycle_state": lifecycle_state, "authorization_state": authorization_state, "evidence_state": evidence_state}


def _has_denied_slots(cases: list[FailureCase]) -> bool:
    return any((case.symbolic_decision or {}).get("denied_slots") for case in cases)
