from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.experience.failure_case import FailureCase
from gov_mem.experience.pattern_inducer import PatternInducer
from gov_mem.skills.skill_context import SkillQueryContext
from gov_mem.skills.skill_library import GovernanceSkillLibraryBuilder
from gov_mem.skills.skill_retriever import GovernanceSkillRetriever


def _failure(case_id: str, domain: str, slot: str) -> FailureCase:
    return FailureCase(
        case_id=case_id,
        domain=domain,
        backbone="smoke",
        query=f"What is the current {slot}?",
        predicted_action="answer",
        official_result={"query_type": "must_not_survive", "expected_action": "must_not_survive"},
        failure_type="missing_utility",
        retrieved_utility_evidence=[{"slots": {slot: f"value-{case_id}"}, "lifecycle_status": "active"}],
        retrieved_governance_evidence=[],
        symbolic_decision={"action_constraint": "answer", "allowed_slots": [slot], "denied_slots": []},
        final_answer="",
        suspected_causes=["utility_surface_missing_or_incomplete"],
    )


def main() -> None:
    domains = ["medical", "education", "household", "office"]
    cases = [
        _failure(f"{domain}-{index}", domain, f"renamed_slot_{index}")
        for domain in domains
        for index in range(2)
    ]
    patterns = PatternInducer().induce(failure_cases=cases, min_support=2)
    assert patterns, "no structural patterns induced"
    serialized = str([
        {
            "description": pattern.description,
            "trigger_signature": pattern.trigger_signature,
            "affected_domains": pattern.affected_domains,
            "recommended_rule": pattern.recommended_rule,
            "recommended_skill": pattern.recommended_skill,
        }
        for pattern in patterns
    ])
    for forbidden in ("must_not_survive", "medical", "education", "household", "office"):
        assert forbidden not in serialized, f"forbidden taxonomy leaked into pattern: {forbidden}"

    skills = GovernanceSkillLibraryBuilder().build(patterns=patterns)
    assert skills, "no governance skills built"
    assert all(not skill.applicable_domains for skill in skills)
    assert all(
        not any(token in skill.name for token in domains)
        for skill in skills
    )

    retriever = GovernanceSkillRetriever()
    selections: list[list[str]] = []
    for domain in domains:
        context = SkillQueryContext(
            domain=domain,
            requester_role="delegate",
            owner_relation="collaborator",
            query_intent="current typed utility",
            required_slots=["unseen_slot"],
            detected_sensitive_slots=[],
            lifecycle_flags=["current"],
            evidence_coverage=1.0,
            certificate_authorized=True,
            certificate_reason="all_requested_slots_have_latest_active_explicit_allow_path",
            certified_slots=["unseen_slot"],
        )
        selections.append([skill.name for _, skill, _ in retriever.retrieve(context=context, skills=skills)])
    assert selections and all(selected == selections[0] for selected in selections[1:]), selections
    assert "typed_utility_realization_skill" in selections[0], selections[0]
    print(f"patterns={len(patterns)} skills={len(skills)} selected={selections[0]}")
    print("structural_evolution_smoke=PASS")


if __name__ == "__main__":
    main()
