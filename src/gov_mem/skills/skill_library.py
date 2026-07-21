from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from gov_mem.experience.pattern_inducer import FailurePattern
from gov_mem.skills.governance_skill import GovernanceSkill
from gov_mem.utils.io import read_jsonl, write_jsonl


class GovernanceSkillLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.skills: list[GovernanceSkill] = []
        if self.path.exists():
            self.skills = [GovernanceSkill(**_normalize_skill_row(dict(row))) for row in read_jsonl(self.path)]

    def all(self) -> list[GovernanceSkill]:
        return list(self.skills)


class GovernanceSkillLibraryBuilder:
    def build(self, *, patterns: list[FailurePattern]) -> list[GovernanceSkill]:
        grouped: dict[str, list[FailurePattern]] = {}
        for pattern in patterns:
            grouped.setdefault(str(pattern.recommended_skill), []).append(pattern)

        self._backfill_required_skill_groups(grouped=grouped, patterns=patterns)

        skills: list[GovernanceSkill] = []
        for skill_name, source_patterns in sorted(grouped.items()):
            if not source_patterns:
                continue
            domains = _merge_unique(value for pattern in source_patterns for value in pattern.affected_domains)
            applicable_domains: list[str] = []
            roles = _merge_unique(value for pattern in source_patterns for value in pattern.affected_roles)
            slots = _merge_unique(value for pattern in source_patterns for value in pattern.affected_slots)
            trigger_conditions = _merge_unique(
                condition
                for pattern in source_patterns
                for condition in _flatten_trigger_signature(pattern.trigger_signature)
            )
            symbolic_rule_patch, prompt_patch, verifier_patch, priority = _skill_template(
                skill_name=skill_name,
                domains=domains,
                slots=slots,
            )
            attestations = [
                dict((pattern.provenance or {}).get("dev_attestation") or {})
                for pattern in source_patterns
            ]
            dev_attestation = attestations[0] if attestations and all(item == attestations[0] for item in attestations) else {}
            skills.append(
                GovernanceSkill(
                    skill_id=skill_name,
                    name=skill_name,
                    description=_skill_description(skill_name, domains),
                    abstraction_source="dev_failure_patterns",
                    abstraction_signature={
                        "recommended_skill": skill_name,
                        "domains": [],
                        "roles": roles,
                        "slots": slots,
                        "source_pattern_ids": [pattern.pattern_id for pattern in source_patterns],
                    },
                    trigger_conditions=trigger_conditions,
                    applicable_domains=applicable_domains,
                    applicable_roles=roles,
                    applicable_slots=slots,
                    symbolic_rule_patch=symbolic_rule_patch,
                    prompt_patch=prompt_patch,
                    verifier_patch=verifier_patch,
                    priority=priority,
                    source_patterns=[pattern.pattern_id for pattern in source_patterns],
                    success_count=0,
                    failure_count=sum(len(pattern.support_cases) for pattern in source_patterns),
                    confidence=min(max(sum(pattern.confidence for pattern in source_patterns) / len(source_patterns), 0.5), 0.95),
                    metadata={
                        "created_from_dev_only": True,
                        "pattern_count": len(source_patterns),
                        "dev_attestation": dev_attestation,
                    },
                )
            )
        return skills

    @staticmethod
    def _backfill_required_skill_groups(
        *,
        grouped: dict[str, list[FailurePattern]],
        patterns: list[FailurePattern],
    ) -> None:
        required = {
            "authorization_projection_skill",
            "lifecycle_integrity_skill",
            "provenance_completion_skill",
            "restrictive_action_calibration_skill",
            "typed_utility_realization_skill",
        }
        if "restrictive_action_calibration_skill" not in grouped:
            grouped["restrictive_action_calibration_skill"] = [
                pattern
                for pattern in patterns
                if pattern.failure_type in {"over_refusal", "deleted_reconstruction"}
            ][:4]
        if "lifecycle_integrity_skill" not in grouped:
            grouped["lifecycle_integrity_skill"] = [
                pattern
                for pattern in patterns
                if pattern.failure_type in {"stale_state", "deleted_reconstruction"}
            ][:4]
        for skill_name in sorted(required):
            grouped.setdefault(skill_name, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build governance skill library from generalized failure patterns.")
    parser.add_argument("--pattern_path", required=True, help="Path to failure_patterns.jsonl")
    parser.add_argument("--output_path", default="outputs/skills/governance_skill_library.jsonl", help="Output library path")
    args = parser.parse_args()

    patterns = [FailurePattern(**row) for row in read_jsonl(args.pattern_path)]
    skills = GovernanceSkillLibraryBuilder().build(patterns=patterns)
    write_jsonl(args.output_path, [asdict(skill) for skill in skills])
    print(f"patterns={len(patterns)}")
    print(f"skills={len(skills)}")
    print(f"output_path={Path(args.output_path).resolve()}")


def _merge_unique(values) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _normalize_skill_row(row: dict) -> dict:
    row.setdefault("abstraction_source", "legacy_skill_library")
    row.setdefault("abstraction_signature", {})
    row.setdefault("metadata", {})
    return row


def _flatten_trigger_signature(signature: dict) -> list[str]:
    conditions: list[str] = []
    blocked_keys = {"query_type", "expected_action", "attack_type", "attack_types"}
    for key, value in dict(signature or {}).items():
        if str(key).strip() in blocked_keys:
            continue
        if isinstance(value, list):
            for item in value:
                text = str(item).strip()
                if text:
                    conditions.append(f"{key}={text}")
        else:
            text = str(value).strip()
            if text:
                conditions.append(f"{key}={text}")
    return conditions


def _skill_template(*, skill_name: str, domains: list[str], slots: list[str]) -> tuple[list[str], str, list[str], int]:
    if skill_name == "authorization_projection_skill":
        return (
            ["skill_enforce_slot_level_authorization", "skill_remove_denied_slot_surfaces"],
            "Render only slots supported by an explicit authorization path; remove denied values without suppressing independently authorized utility.",
            ["verify_denied_slots_absent", "verify_rendered_slots_authorized"],
            95,
        )
    if skill_name == "lifecycle_integrity_skill":
        return (
            ["skill_block_deleted_reconstruction", "skill_prefer_latest_active_version"],
            "Resolve memory versions before answering: deleted values are not reconstructable and superseded values cannot replace the latest active state.",
            ["verify_deleted_surface_absent", "verify_superseded_surface_absent"],
            100,
        )
    if skill_name == "typed_utility_realization_skill":
        return (
            ["skill_realize_certificate_typed_slots"],
            "When the authorization certificate is complete, realize every requested certified slot using its source-validated value and a minimal canonical surface.",
            ["verify_certified_slots_present", "verify_no_uncertified_value"],
            90,
        )
    if skill_name == "provenance_completion_skill":
        return (
            ["skill_require_complete_provenance_path"],
            "Do not realize a typed value unless its active version, source span, and principal-compatible authorization path are complete.",
            ["verify_complete_provenance_path"],
            94,
        )
    if skill_name == "restrictive_action_calibration_skill":
        return (
            ["skill_calibrate_restrictive_action_from_memory_state"],
            "Use no_memory for absent or lifecycle-blocked content, refuse for present but wholly unauthorized content, and answer_redacted when authorized utility remains.",
            ["verify_restrictive_action_matches_memory_state"],
            92,
        )
    return (
        ["skill_generic_governance_patch"],
        f"Apply a governance-state patch for typed slots {slots or ['generic']}.",
        ["verify_generic_governance_constraint"],
        70,
    )


def _skill_description(skill_name: str, domains: list[str]) -> str:
    _ = domains
    return f"{skill_name} induced from development-only structural governance failures."


def _skill_applicable_domains(*, skill_name: str, domains: list[str]) -> list[str]:
    _ = skill_name, domains
    return []


if __name__ == "__main__":
    main()
