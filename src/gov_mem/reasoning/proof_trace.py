from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ProofStep:
    rule_name: str
    conclusion: str
    supporting_predicates: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0


def proof_step_to_dict(step: ProofStep) -> dict[str, Any]:
    return asdict(step)
