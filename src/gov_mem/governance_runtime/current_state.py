from __future__ import annotations

from dataclasses import asdict, dataclass, field

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frames
from gov_mem.governance_runtime.state_ledger import build_current_state_ledger


@dataclass
class CurrentState:
    active_items: list[dict] = field(default_factory=list)
    canceled_items: list[dict] = field(default_factory=list)
    deleted_items: list[dict] = field(default_factory=list)
    superseded_items: list[dict] = field(default_factory=list)
    uncertain_items: list[dict] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


def resolve_current_state(evidence: list[RetrievedEvidence]) -> tuple[CurrentState, list]:
    frames = compile_evidence_frames(evidence)
    ledger = build_current_state_ledger(frames)
    state = CurrentState(
        active_items=[asdict(item) for item in ledger.active_events.values()],
        canceled_items=[asdict(item) for item in ledger.canceled_events.values()],
        deleted_items=[asdict(item) for item in ledger.deleted_events.values()],
        superseded_items=[asdict(item) for item in ledger.superseded_events.values()],
        uncertain_items=[],
        trace=list(ledger.trace),
    )
    return state, frames
