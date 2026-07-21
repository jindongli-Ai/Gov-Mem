from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SkillQueryContext:
    domain: str
    requester_role: str
    owner_relation: str
    query_intent: str
    required_slots: list[str]
    detected_sensitive_slots: list[str]
    lifecycle_flags: list[str]
    graph_lifecycle_flags: list[str] = field(default_factory=list)
    explicit_deleted_request: bool = False
    explicit_historical_request: bool = False
    explicit_current_request: bool = False
    symbolic_predicates: list[Any] = field(default_factory=list)
    evidence_coverage: float = 0.0
    certificate_authorized: bool = False
    certificate_reason: str = ""
    certified_slots: list[str] = field(default_factory=list)


def skill_query_context_to_dict(context: SkillQueryContext) -> dict[str, Any]:
    return asdict(context)
