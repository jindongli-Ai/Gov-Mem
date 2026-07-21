from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RuntimeAdaptationAudit:
    instance_id: str
    domain: str
    query: str
    adaptation_enabled: bool
    adaptation_triggered: bool
    skill_library_enabled: bool
    experience_memory_enabled: bool
    self_evolving_enabled: bool
    selected_experience_ids: list[str] = field(default_factory=list)
    selected_experience_pattern_ids: list[str] = field(default_factory=list)
    selected_skill_ids: list[str] = field(default_factory=list)
    loaded_rule_updates: list[str] = field(default_factory=list)
    loaded_prompt_updates: list[str] = field(default_factory=list)
    loaded_policy_updates: list[str] = field(default_factory=list)
    action_patches: list[str] = field(default_factory=list)
    affected_decision_fields: list[str] = field(default_factory=list)
    trigger_reasons: list[str] = field(default_factory=list)
    runtime_context_summary: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def runtime_adaptation_audit_to_dict(audit: RuntimeAdaptationAudit) -> dict[str, Any]:
    return asdict(audit)
