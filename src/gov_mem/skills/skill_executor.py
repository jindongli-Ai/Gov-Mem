from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gov_mem.skills.governance_skill import GovernanceSkill, governance_skill_to_dict
from gov_mem.skills.skill_context import SkillQueryContext


@dataclass
class RetrievedSkillBundle:
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    activated_rules: list[str] = field(default_factory=list)
    prompt_patches: list[str] = field(default_factory=list)
    verifier_patches: list[str] = field(default_factory=list)
    action_patches: list[str] = field(default_factory=list)
    loaded_rule_updates: list[str] = field(default_factory=list)
    loaded_prompt_updates: list[str] = field(default_factory=list)
    loaded_policy_updates: list[str] = field(default_factory=list)
    skill_trace: list[str] = field(default_factory=list)
    affected_decision_fields: list[str] = field(default_factory=list)
    explicit_deleted_request: bool = False
    explicit_historical_request: bool = False
    explicit_current_request: bool = False


class GovernanceSkillExecutor:
    def execute(
        self,
        *,
        context: SkillQueryContext,
        selected_skills: list[tuple[float, GovernanceSkill, list[str]]],
    ) -> RetrievedSkillBundle:
        bundle = RetrievedSkillBundle(
            explicit_deleted_request=bool(context.explicit_deleted_request),
            explicit_historical_request=bool(context.explicit_historical_request),
            explicit_current_request=bool(context.explicit_current_request),
        )
        for score, skill, reasons in selected_skills:
            bundle.selected_skills.append(
                {
                    **governance_skill_to_dict(skill),
                    "selection_score": round(float(score), 4),
                    "selection_reasons": reasons,
                }
            )
            bundle.activated_rules.extend(skill.symbolic_rule_patch)
            if skill.prompt_patch.strip():
                bundle.prompt_patches.append(skill.prompt_patch.strip())
            bundle.verifier_patches.extend(skill.verifier_patch)
            bundle.action_patches.extend(_action_patches_for_skill(skill=skill, context=context))
            bundle.skill_trace.append(f"selected {skill.name} score={score:.2f} reasons={reasons}")

        bundle.activated_rules = list(dict.fromkeys(bundle.activated_rules))
        bundle.prompt_patches = list(dict.fromkeys(bundle.prompt_patches))
        bundle.verifier_patches = list(dict.fromkeys(bundle.verifier_patches))
        bundle.action_patches = list(dict.fromkeys(bundle.action_patches))
        bundle.affected_decision_fields = _affected_fields(bundle)
        return bundle


def retrieved_skill_bundle_to_dict(bundle: RetrievedSkillBundle) -> dict[str, Any]:
    return asdict(bundle)


def _action_patches_for_skill(*, skill: GovernanceSkill, context: SkillQueryContext) -> list[str]:
    if skill.name in {"lifecycle_integrity_skill", "deletion_blocking_skill"} and (context.explicit_deleted_request or context.explicit_historical_request):
        return ["force_no_memory_on_deleted_query"]
    if skill.name == "authorization_projection_skill" and context.detected_sensitive_slots:
        return ["enforce_certificate_bounded_slot_projection"]
    if skill.name == "typed_utility_realization_skill" and context.certificate_authorized:
        return ["prefer_certificate_grounded_typed_answer"]
    if skill.name == "provenance_completion_skill" and not context.certificate_authorized:
        return ["fail_closed_on_incomplete_provenance_path"]
    if skill.name in {"restrictive_action_calibration_skill", "no_memory_vs_refusal_skill"}:
        return ["disambiguate_no_memory_vs_refusal"]
    return []


def _affected_fields(bundle: RetrievedSkillBundle) -> list[str]:
    fields: list[str] = []
    if bundle.activated_rules:
        fields.append("symbolic_reasoner")
    if bundle.action_patches:
        fields.append("action_arbitrator")
    if bundle.prompt_patches:
        fields.append("renderer")
    if bundle.verifier_patches:
        fields.append("verifier")
    return fields
