from __future__ import annotations

from gov_mem.data.schema import AnswerResult, MemoryInstance, ReasoningState
from gov_mem.governance_runtime.access import normalize_role


class FailureAnalyzer:
    def analyze(
        self,
        *,
        instance: MemoryInstance,
        answer_result: AnswerResult,
        reasoning_state: ReasoningState,
        correct: bool,
    ) -> str | None:
        if correct:
            return None
        if answer_result.action not in {"answer", "answer_redacted", "refuse", "no_memory"}:
            return "answer_format_error"
        if not reasoning_state.selected_evidence:
            return "retrieval_miss"
        slot_coverage = reasoning_state.slot_coverage or {}
        required_slots = set(slot_coverage.get("required_slots", []))
        covered_slots = set(slot_coverage.get("covered_slots", []))
        missing_slots = set(slot_coverage.get("missing_slots", []))
        if answer_result.action == "refuse" and reasoning_state.selected_evidence:
            return "access_filter_error"
        if answer_result.action == "no_memory" and reasoning_state.selected_evidence:
            if any(
                getattr(frame, "lifecycle_status", None) in {"deleted", "superseded"}
                for frame in (reasoning_state.selected_frames or [])
            ):
                return "access_filter_error"
        if required_slots and not covered_slots:
            return "slot_selection_error"
        if any(
            frame.lifecycle_status in {"canceled", "superseded"} for frame in reasoning_state.selected_frames
        ) and any(
            frame.lifecycle_status == "active" for frame in reasoning_state.selected_frames
        ) and "status" in missing_slots:
            return "state_tracking_error"
        if answer_result.answer_structured and missing_slots:
            return "renderer_omission"
        requester_role = normalize_role(((instance.metadata.get("requester") or {}).get("role")))
        if requester_role == "owner" and instance.asking_user_id and not any(
            row.user_id == instance.asking_user_id for row in reasoning_state.selected_evidence if row.user_id
        ):
            return "wrong_user_grounding"
        if reasoning_state.conflicts:
            return "conflict_unresolved"
        if any("temporal" in step.lower() for step in reasoning_state.reasoning_trace):
            return "temporal_error"
        return "official_match_miss"
