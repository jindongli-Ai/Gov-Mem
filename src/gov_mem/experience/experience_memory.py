from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any

from gov_mem.experience.failure_case import DevFailureCaseBuilder, FailureCase
from gov_mem.experience.experience_store import ExperienceStore
from gov_mem.evolution.dev_guard import load_dev_attestation


@dataclass
class Experience:
    exp_id: str
    failure_type: str
    pattern: str
    pattern_id: str
    abstraction_level: str
    pattern_signature: dict[str, Any]
    trigger_conditions: list[str]
    correction_strategy: str
    applicable_domains: list[str]
    applicable_actions: list[str]
    applicable_roles: list[str]
    applicable_slots: list[str]
    reusable_skill_hypothesis: str
    evidence: list[str]
    confidence: float
    created_from_dev_only: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperienceMemoryBuilder:
    def build(self, *, failure_cases: list[FailureCase]) -> list[Experience]:
        grouped: dict[tuple[str, str, str], list[FailureCase]] = {}
        for case in failure_cases:
            signature = _governance_signature(case)
            key = (
                str(case.failure_type),
                signature["lifecycle_state"],
                signature["authorization_state"],
            )
            grouped.setdefault(key, []).append(case)

        experiences: list[Experience] = []
        for (failure_type, lifecycle_state, authorization_state), cases in sorted(grouped.items()):
            support_ids = [case.case_id for case in cases]
            trigger_conditions = _collect_trigger_conditions(cases)
            correction_strategy = _correction_strategy(failure_type)
            pattern = _pattern_description(failure_type=failure_type, lifecycle_state=lifecycle_state, authorization_state=authorization_state, trigger_conditions=trigger_conditions)
            applicable_roles = _collect_roles(cases)
            applicable_slots = _collect_slots(cases)
            pattern_signature = _pattern_signature(
                failure_type=failure_type,
                lifecycle_state=lifecycle_state,
                authorization_state=authorization_state,
                trigger_conditions=trigger_conditions,
                applicable_roles=applicable_roles,
                applicable_slots=applicable_slots,
            )
            pattern_id = md5(
                f"{failure_type}:{lifecycle_state}:{authorization_state}:{'|'.join(trigger_conditions)}:{'|'.join(applicable_slots[:6])}".encode("utf-8")
            ).hexdigest()[:12]
            experiences.append(
                Experience(
                    exp_id=md5(f"exp:{pattern_id}:{len(cases)}".encode("utf-8")).hexdigest()[:12],
                    failure_type=failure_type,
                    pattern=pattern,
                    pattern_id=pattern_id,
                    abstraction_level="generalized_failure_pattern",
                    pattern_signature=pattern_signature,
                    trigger_conditions=trigger_conditions,
                    correction_strategy=correction_strategy,
                    applicable_domains=[],
                    applicable_actions=sorted({case.predicted_action for case in cases if case.predicted_action}),
                    applicable_roles=applicable_roles,
                    applicable_slots=applicable_slots,
                    reusable_skill_hypothesis=_reusable_skill_hypothesis(
                        failure_type=failure_type,
                        applicable_slots=applicable_slots,
                    ),
                    evidence=support_ids,
                    confidence=min(0.55 + 0.1 * len(cases), 0.9),
                    metadata={
                        "support_count": len(cases),
                        "created_from_dev_only": True,
                        "source_case_ids": support_ids,
                    },
                )
            )
        return experiences


def experience_to_dict(item: Experience) -> dict[str, Any]:
    return asdict(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dev-only experience memory from failed Gov-Mem runs.")
    parser.add_argument("--run_dir", action="append", required=True, help="Development run directory. Can be passed multiple times.")
    parser.add_argument("--dev_attestation", required=True, help="Development-only provenance attestation JSON.")
    parser.add_argument("--output_dir", default="outputs/experience", help="Directory to write failure_cases.jsonl and experience_memory.jsonl")
    parser.add_argument("--backbone", default="rag_policy_amem", help="Backbone name to attach to failure cases")
    args = parser.parse_args()

    attestation = load_dev_attestation(args.dev_attestation)
    if {str(Path(run).resolve()) for run in args.run_dir} - {str(Path(run).resolve()) for run in attestation["source_runs"]}:
        raise ValueError("Each run_dir must be declared in the development attestation")

    builder = DevFailureCaseBuilder()
    failure_cases: list[FailureCase] = []
    for run_dir in args.run_dir:
        failure_cases.extend(builder.build_from_run(run_dir=run_dir, backbone=args.backbone))

    deduped_cases: list[FailureCase] = []
    seen_ids: set[str] = set()
    for case in failure_cases:
        if case.case_id in seen_ids:
            continue
        seen_ids.add(case.case_id)
        deduped_cases.append(case)

    experiences = ExperienceMemoryBuilder().build(failure_cases=deduped_cases)
    for experience in experiences:
        experience.metadata["dev_attestation"] = attestation
    store = ExperienceStore(output_dir=Path(args.output_dir))
    store.save_failure_cases(deduped_cases)
    store.save_experiences(experiences)
    print(f"failure_cases={len(deduped_cases)}")
    print(f"experiences={len(experiences)}")
    print(f"output_dir={Path(args.output_dir).resolve()}")


def _collect_trigger_conditions(cases: list[FailureCase]) -> list[str]:
    conditions: list[str] = []
    for case in cases:
        if case.predicted_action:
            conditions.append(f"predicted_action={case.predicted_action}")
        if case.symbolic_decision.get("action_constraint"):
            conditions.append(f"symbolic_action={case.symbolic_decision.get('action_constraint')}")
        if case.symbolic_decision.get("rules_fired"):
            for rule_name in list(case.symbolic_decision.get("rules_fired") or [])[:3]:
                conditions.append(f"rule={rule_name}")
    return list(dict.fromkeys(conditions))


def _correction_strategy(failure_type: str) -> str:
    mapping = {
        "leakage": "Tighten governance projection and ensure denied-slot surfaces are removed before realization.",
        "deleted_reconstruction": "Strengthen deletion blocking so blocked facts cannot enter either context or final realization.",
        "under_redaction": "Force redaction-aware realization when symbolic governance allows only partial disclosure.",
        "over_refusal": "Relax overly conservative action arbitration when the symbolic decision and official expectation both allow a constrained answer.",
        "stale_state": "Increase preference for active state and suppress superseded or deleted state during state resolution and realization.",
        "no_memory_collapse": "Preserve required-slot utility evidence so the system does not collapse into no_memory when answerable evidence exists.",
        "missing_utility": "Improve evidence packing and answer surface coverage so benchmark-required utility fields are fully rendered.",
    }
    return mapping.get(failure_type, "Review retrieval, governance reasoning, and realization jointly before updating the framework.")


def _pattern_description(*, failure_type: str, lifecycle_state: str, authorization_state: str, trigger_conditions: list[str]) -> str:
    joined = ", ".join(trigger_conditions[:4]) or "generic trigger pattern"
    return f"{failure_type} pattern with lifecycle={lifecycle_state}, authorization={authorization_state}, under {joined}."


def _collect_roles(cases: list[FailureCase]) -> list[str]:
    roles: list[str] = []
    for case in cases:
        governance_rows = list(case.retrieved_governance_evidence or [])
        for group in governance_rows:
            if not isinstance(group, dict):
                continue
            for key in ("roles", "relations", "policies"):
                for row in group.get(key) or []:
                    if not isinstance(row, dict):
                        continue
                    for scope in row.get("access_scope") or []:
                        text = str(scope).strip()
                        if text and text not in roles:
                            roles.append(text)
    return roles[:12]


def _collect_slots(cases: list[FailureCase]) -> list[str]:
    slots: list[str] = []
    for case in cases:
        symbolic = dict(case.symbolic_decision or {})
        for key in ("allowed_slots", "denied_slots"):
            for slot in symbolic.get(key) or []:
                text = str(slot).strip()
                if text and text not in slots:
                    slots.append(text)
        for row in case.retrieved_utility_evidence or []:
            if not isinstance(row, dict):
                continue
            for slot in dict(row.get("slots") or {}).keys():
                text = str(slot).strip()
                if text and text not in slots:
                    slots.append(text)
    return slots[:16]


def _pattern_signature(
    *,
    failure_type: str,
    lifecycle_state: str,
    authorization_state: str,
    trigger_conditions: list[str],
    applicable_roles: list[str],
    applicable_slots: list[str],
) -> dict[str, Any]:
    return {
        "failure_type": failure_type,
        "lifecycle_state": lifecycle_state,
        "authorization_state": authorization_state,
        "trigger_conditions": list(trigger_conditions),
        "applicable_roles": list(applicable_roles),
        "applicable_slots": list(applicable_slots),
    }


def _reusable_skill_hypothesis(
    *,
    failure_type: str,
    applicable_slots: list[str],
) -> str:
    _ = applicable_slots
    if failure_type in {"deleted_reconstruction", "stale_state"}:
        return "lifecycle_integrity_skill"
    if failure_type in {"leakage", "under_redaction"}:
        return "authorization_projection_skill"
    if failure_type in {"missing_utility", "no_memory_collapse"}:
        return "typed_utility_realization_skill"
    return "restrictive_action_calibration_skill"


def _governance_signature(case: FailureCase) -> dict[str, str]:
    symbolic = dict(case.symbolic_decision or {})
    denied = list(symbolic.get("denied_slots") or [])
    allowed = list(symbolic.get("allowed_slots") or [])
    evidence_text = str(case.retrieved_utility_evidence).lower()
    lifecycle = "deleted" if "deleted" in evidence_text else "superseded" if "superseded" in evidence_text else "active"
    authorization = "restricted" if denied else "authorized" if allowed else "undetermined"
    return {"lifecycle_state": lifecycle, "authorization_state": authorization}


if __name__ == "__main__":
    main()
