from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GovernanceDecision:
    action_constraint: str
    allowed_slots: list[str] = field(default_factory=list)
    denied_slots: list[str] = field(default_factory=list)
    redacted_slots: list[str] = field(default_factory=list)
    explicit_deleted_request: bool = False
    explicit_historical_request: bool = False
    explicit_current_request: bool = False
    blocked_facts: list[str] = field(default_factory=list)
    active_facts: list[str] = field(default_factory=list)
    superseded_facts: list[str] = field(default_factory=list)
    deleted_facts: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    proof_trace: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0


def governance_decision_to_dict(decision: GovernanceDecision) -> dict[str, Any]:
    return asdict(decision)
