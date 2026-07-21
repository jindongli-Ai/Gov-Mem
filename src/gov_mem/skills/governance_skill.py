from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GovernanceSkill:
    skill_id: str
    name: str
    description: str
    trigger_conditions: list[str]
    applicable_domains: list[str]
    applicable_roles: list[str]
    applicable_slots: list[str]
    symbolic_rule_patch: list[str]
    prompt_patch: str
    verifier_patch: list[str]
    priority: int
    source_patterns: list[str]
    success_count: int
    failure_count: int
    confidence: float
    abstraction_source: str = "legacy_skill_library"
    abstraction_signature: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def governance_skill_to_dict(skill: GovernanceSkill) -> dict[str, Any]:
    return asdict(skill)
