"""Small support surface used by the paper-facing Gov-Mem v4 backbone.

The active path intentionally keeps its chunk/result types separate from the
historical backbone helpers.  No semantic vocabulary or raw-text classifier
belongs in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gov_mem.data.schema import (
    AnswerResult,
    GovernedActionDecision,
    MemoryInstance,
    MemoryItem,
    QueryPlan,
    ReasoningState,
    RetrievedEvidence,
)
from gov_mem.data.timestamps import normalize_timestamp
from gov_mem.utils.io import ensure_dir, write_jsonl


@dataclass
class RAGChunk:
    chunk_id: str
    instance_id: str
    text: str
    source_message_ids: list[str]
    speaker_ids: list[str]
    timestamp_range: tuple[str | None, str | None]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneRunResult:
    query_plan: QueryPlan
    retrieval_result: dict[str, Any]
    reasoning_state: ReasoningState
    action_decision: GovernedActionDecision | None
    answer_result: AnswerResult
    debug_payload: dict[str, Any]


def save_rag_chunks(output_dir, dataset_name: str, instance_id: str, chunks: list[RAGChunk]) -> None:
    path = ensure_dir(output_dir / "rag_chunks" / dataset_name / instance_id) / "chunks.jsonl"
    write_jsonl(path, [asdict(chunk) for chunk in chunks])


def chunk_to_memory_item(chunk: RAGChunk) -> MemoryItem:
    return MemoryItem(
        memory_id=chunk.chunk_id,
        instance_id=chunk.instance_id,
        user_id=(chunk.speaker_ids[0] if chunk.speaker_ids else None),
        scope="chunk",
        content=chunk.text,
        memory_type="chunk",
        entities=[],
        time=chunk.timestamp_range[0],
        source_message_ids=chunk.source_message_ids,
        confidence=1.0,
        privacy_level=None,
        tags=[str(chunk.metadata.get("chunk_type") or "chunk")],
        metadata=dict(chunk.metadata),
    )


def build_reasoning_state(
    evidence: list[RetrievedEvidence],
    *,
    trace: list[str] | None = None,
    slot_coverage: dict | None = None,
    selected_frames: list | None = None,
    current_state_ledger: dict | None = None,
    required_slot_plan: dict | None = None,
) -> ReasoningState:
    return ReasoningState(
        selected_evidence=evidence,
        reasoning_trace=trace or [],
        conflicts=[],
        conclusion_hint="Backbone evidence aggregation.",
        selected_frames=selected_frames or [],
        current_state_ledger=current_state_ledger or {},
        required_slot_plan=required_slot_plan or {},
        slot_coverage=slot_coverage or {},
    )


__all__ = [
    "BackboneRunResult",
    "RAGChunk",
    "build_reasoning_state",
    "chunk_to_memory_item",
    "save_rag_chunks",
]
