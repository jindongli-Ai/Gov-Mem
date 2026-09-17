"""Lexicon-free structural operators for Gov-Mem.

The paper-facing v4 backbone calls :func:`build_required_slot_plan` with a
query-conditioned LLM contract. These helpers perform only structural work:
stable ordering, dynamic-slot coverage, and provenance selection. They never
infer semantic fields, policy scope, lifecycle cues, or domains from raw text.
"""

from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import QueryPlan, RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame


def _slot_key(value: object) -> str:
    """Normalize a runtime-proposed slot label without a vocabulary lookup."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _source_order(row: RetrievedEvidence) -> tuple[int, str, float]:
    metadata = dict(getattr(row, "metadata", {}) or {})
    record = metadata.get("structured_record")
    turn_index = record.get("turn_index") if isinstance(record, dict) else None
    if not isinstance(turn_index, int):
        turn_index = metadata.get("source_turn_index")
    if not isinstance(turn_index, int):
        turn_index = -1
    return turn_index, str(getattr(row, "time", "") or ""), float(getattr(row, "score", 0.0) or 0.0)


def apply_not_filter(plan: QueryPlan, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    """Apply an explicit plan-level entity filter, with no semantic parsing."""

    if "NOT" not in list(plan.reasoning_ops or []) or not list(plan.target_users or []):
        return list(evidence)
    targets = {_slot_key(value) for value in plan.target_users if _slot_key(value)}
    if not targets:
        return list(evidence)
    return [
        row for row in evidence
        if not getattr(row, "user_id", None) or _slot_key(row.user_id) in targets
    ]


def apply_temporal_order(plan: QueryPlan, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    """Use visible provenance order only; never inspect raw text for status."""

    if "temporal_order" not in list(plan.reasoning_ops or []):
        return list(evidence)
    return sorted(list(evidence), key=_source_order, reverse=True)


def detect_conflicts(evidence: list[RetrievedEvidence]) -> list[dict[str, str]]:
    """Report provenance-level duplicate-source disagreement without adjudication."""

    first_by_identity: dict[tuple[str, str], RetrievedEvidence] = {}
    conflicts: list[dict[str, str]] = []
    for row in evidence:
        identity = (str(row.user_id or ""), str(row.memory_type or ""))
        previous = first_by_identity.get(identity)
        if previous is not None and str(previous.content or "") != str(row.content or ""):
            conflicts.append({"memory_id_a": previous.memory_id, "memory_id_b": row.memory_id})
        else:
            first_by_identity[identity] = row
    return conflicts


def apply_compare(evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    return list(evidence)


def build_required_slot_plan(
    question: str,
    query_plan: QueryPlan,
    action_decision: Any = None,
    query_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an open-vocabulary slot plan from runtime semantic structure.

    ``question`` and ``action_decision`` are compatibility arguments but are
    intentionally never parsed. In v4, fields come from the semantic compiler;
    without it, only an already-structured plan attribute is allowed.
    """

    del question, action_decision
    fields = list((query_analysis or {}).get("fields") or []) if query_analysis is not None else []
    required_slots = [
        str(item.get("name") or "").strip()
        for item in fields
        if isinstance(item, dict) and str(item.get("name") or "").strip()
        and bool(item.get("required", True))
    ]
    if not required_slots:
        semantic_spec = dict(getattr(query_plan, "semantic_spec", {}) or {})
        source_slots = semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or []
        if isinstance(source_slots, list):
            required_slots = [str(value).strip() for value in source_slots if str(value or "").strip()]
    required_slots = list(dict.fromkeys(required_slots))
    temporal = dict((query_analysis or {}).get("temporal") or {})
    request_shape = str((query_analysis or {}).get("request_shape") or "unknown").casefold()
    return {
        "target_frame_types": ["general_fact"],
        "required_slots": required_slots,
        "optional_slots": [],
        "target_entities": [],
        "temporal_policy": "historical" if str(temporal.get("orientation") or "").casefold() == "historical" else "current_state",
        "mixed": request_shape == "multi_field" or len(required_slots) > 1,
        "domains": ["query_conditioned_contract"] if required_slots else [],
    }


def _frame_slot_keys(frame: Any) -> set[str]:
    slots = dict(getattr(frame, "slots", {}) or {})
    attributes = dict(getattr(frame, "semantic_attributes", {}) or {})
    return {_slot_key(value) for value in [*slots.keys(), *attributes.keys()] if _slot_key(value)}


def _covered_required_slot_keys(frame: Any, required_slots: list[str]) -> set[str]:
    available = _frame_slot_keys(frame)
    return {slot for slot in required_slots if _slot_key(slot) and _slot_key(slot) in available}


def select_evidence_by_slot_coverage(
    *,
    frames: list[Any],
    evidence: list[RetrievedEvidence],
    required_slot_plan: dict[str, Any],
    requester: Any,
    config: dict[str, Any],
) -> tuple[list[RetrievedEvidence], list[Any], dict[str, Any]]:
    """Select source rows that cover dynamic semantic slots.

    Coverage comes solely from verified atom-derived frame keys. It is not an
    authorization decision and no raw-text keyword/regex fallback is present.
    """

    del requester
    required_slots = [str(value) for value in list(required_slot_plan.get("required_slots") or []) if str(value).strip()]
    max_selected = max(1, int((config.get("reasoning") or {}).get("max_selected_frames", 12)))
    remaining = list(zip(frames, evidence))
    selected_frames: list[Any] = []
    selected_evidence: list[RetrievedEvidence] = []
    covered: set[str] = set()
    while remaining and len(selected_frames) < max_selected:
        best_index, best_gain = 0, -1
        for index, (frame, _row) in enumerate(remaining):
            gain = len(_covered_required_slot_keys(frame, required_slots) - covered)
            if gain > best_gain:
                best_index, best_gain = index, gain
        frame, row = remaining.pop(best_index)
        if selected_frames and best_gain <= 0:
            break
        selected_frames.append(frame)
        selected_evidence.append(row)
        covered.update(_covered_required_slot_keys(frame, required_slots))
        if required_slots and len(covered) == len(required_slots):
            break
    if not selected_frames and evidence:
        selected_evidence = [evidence[0]]
        selected_frames = [frames[0]] if frames else [compile_evidence_frame(evidence[0])]
        covered.update(_covered_required_slot_keys(selected_frames[0], required_slots))
    missing = [slot for slot in required_slots if slot not in covered]
    return selected_evidence, selected_frames, {
        "required_slots": required_slots,
        "covered_slots": sorted(covered),
        "missing_slots": missing,
        "coverage_ratio": (len(covered) / len(required_slots)) if required_slots else 1.0,
    }
