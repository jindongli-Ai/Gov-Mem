from __future__ import annotations

from gov_mem.skills.governance_skill import GovernanceSkill
from gov_mem.skills.skill_context import SkillQueryContext


class GovernanceSkillRetriever:
    def retrieve(
        self,
        *,
        context: SkillQueryContext,
        skills: list[GovernanceSkill],
        top_k: int = 4,
    ) -> list[tuple[float, GovernanceSkill, list[str]]]:
        scored: list[tuple[float, GovernanceSkill, list[str]]] = []
        required_slots = {slot for slot in context.required_slots if slot}
        sensitive_slots = {slot for slot in context.detected_sensitive_slots if slot}
        lifecycle_flags = {flag for flag in context.lifecycle_flags if flag}
        for skill in skills:
            score = 0.0
            reasons: list[str] = []
            if skill.applicable_domains:
                # New libraries are domain-invariant. Keep legacy artifacts
                # loadable, but never reward benchmark-domain identity.
                reasons.append("legacy_domain_scope_ignored")
            else:
                score += 0.5
                reasons.append("domain_invariant_skill")
            if context.requester_role and context.requester_role in skill.applicable_roles:
                score += 1.5
                reasons.append(f"requester_role={context.requester_role}")
            if context.owner_relation and context.owner_relation in skill.applicable_roles:
                score += 1.5
                reasons.append(f"owner_relation={context.owner_relation}")
            slot_overlap = required_slots & set(skill.applicable_slots)
            if slot_overlap:
                score += 1.2 + 0.2 * len(slot_overlap)
                reasons.append(f"slot_overlap={sorted(slot_overlap)}")
            sensitive_overlap = sensitive_slots & set(skill.applicable_slots)
            if sensitive_overlap:
                score += 1.0
                reasons.append(f"sensitive_overlap={sorted(sensitive_overlap)}")
            trigger_hits = [
                trigger
                for trigger in skill.trigger_conditions
                if _trigger_matches(trigger=trigger, context=context, lifecycle_flags=lifecycle_flags)
            ]
            if trigger_hits:
                score += min(2.0, 0.5 * len(trigger_hits))
                reasons.append(f"trigger_hits={trigger_hits[:4]}")
            heuristic_score, heuristic_reasons = _heuristic_skill_match(
                skill=skill,
                context=context,
                lifecycle_flags=lifecycle_flags,
                required_slots=required_slots,
            )
            if heuristic_score > 0:
                score += heuristic_score
                reasons.extend(heuristic_reasons)
            if not trigger_hits and not heuristic_reasons and not slot_overlap and not sensitive_overlap:
                continue
            if score > 0:
                scored.append((score + 0.01 * skill.priority, skill, reasons))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


def _trigger_matches(*, trigger: str, context: SkillQueryContext, lifecycle_flags: set[str]) -> bool:
    text = str(trigger or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith(("query_type=", "expected_action=", "attack_types=", "attack_type=")):
        return False
    if lowered.startswith("symbolic_actions=") or lowered.startswith("symbolic_action="):
        target = lowered.split("=", 1)[1]
        return any(target == str(item).lower() for item in context.symbolic_predicates)
    if lowered.startswith("predicted_actions=") or lowered.startswith("predicted_action="):
        return lowered.split("=", 1)[1] in context.query_intent.lower()
    if lowered.startswith("rule="):
        return lowered.split("=", 1)[1] in lifecycle_flags
    if lowered.startswith("lifecycle_state="):
        return lowered.split("=", 1)[1] in lifecycle_flags
    if lowered.startswith("certificate_state="):
        target = lowered.split("=", 1)[1]
        actual = "authorized" if context.certificate_authorized else "incomplete"
        return target == actual
    if lowered.startswith("evidence_state="):
        target = lowered.split("=", 1)[1]
        actual = "complete" if context.evidence_coverage >= 1.0 else "partial"
        return target == actual
    if lowered.startswith("slot_policy_state="):
        target = lowered.split("=", 1)[1]
        actual = "sensitive" if context.detected_sensitive_slots else "ordinary"
        return target == actual
    return lowered in context.query_intent.lower()


def _heuristic_skill_match(
    *,
    skill: GovernanceSkill,
    context: SkillQueryContext,
    lifecycle_flags: set[str],
    required_slots: set[str],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    if skill.name in {"lifecycle_integrity_skill", "deletion_blocking_skill"}:
        if context.explicit_deleted_request or context.explicit_historical_request:
            score += 2.0
            reasons.append("lifecycle_integrity_required")
        if context.explicit_current_request and ({"supersedes", "superseded"} & set(context.graph_lifecycle_flags)):
            score += 1.5
            reasons.append("version_resolution_required")
    elif skill.name in {"restrictive_action_calibration_skill", "no_memory_vs_refusal_skill"}:
        if context.certificate_reason or context.explicit_deleted_request:
            score += 1.5
            reasons.append("restrictive_action_disambiguation_required")
    elif skill.name == "authorization_projection_skill":
        if context.detected_sensitive_slots or context.certificate_reason.startswith("no_explicit_graph_allow"):
            score += 1.5
            reasons.append("slot_authorization_projection_required")
    elif skill.name == "typed_utility_realization_skill":
        if required_slots and (context.evidence_coverage < 1.0 or context.certificate_authorized):
            score += 1.5
            reasons.append("typed_slot_realization_required")
    elif skill.name == "provenance_completion_skill":
        if required_slots and not context.certificate_authorized:
            score += 1.5
            reasons.append("provenance_path_incomplete")
    return score, reasons
