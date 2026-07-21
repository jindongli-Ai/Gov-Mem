from __future__ import annotations

from pathlib import Path
from typing import Any

from gov_mem.backbones.common import BackboneRunResult
from gov_mem.backbones.rag_policy import RAGPolicyBackbone
from gov_mem.data.schema import AnswerResult, GovernedActionDecision, MemoryInstance, ReasoningState, RetrievedEvidence
from gov_mem.governance_runtime.access import build_principal, infer_owner_user_id
from gov_mem.governance_runtime.current_state import resolve_current_state
from gov_mem.governance_runtime.state_ledger import build_current_state_ledger, ledger_to_dict
from gov_mem.governance_runtime.utility_answering import (
    answer_structured_to_dict,
    audit_answer_structured_slots,
    build_answer_structured,
    correct_action_with_runtime_evidence,
    render_with_surface_replay,
    verify_rendered_answer_contains_slots,
)
from gov_mem.llm.client import LLMClient
from gov_mem.reasoning.operators import build_required_slot_plan, select_evidence_by_slot_coverage


def _explode_state_evidence_rows(evidence: list[RetrievedEvidence], required_slot_plan: dict[str, Any]) -> list[RetrievedEvidence]:
    target_frame_types = set(required_slot_plan.get("target_frame_types") or [])
    current_state_like = bool(target_frame_types & {"household_plan", "project_state", "research_state"})
    if not current_state_like:
        return evidence
    expanded: list[RetrievedEvidence] = []
    for row in evidence:
        raw_lines = [line.strip() for line in str(row.content or "").splitlines() if line.strip()]
        if len(raw_lines) <= 1:
            expanded.append(row)
            continue
        for idx, raw_line in enumerate(raw_lines):
            line_text = raw_line
            if raw_line.startswith("[") and "] " in raw_line:
                line_text = raw_line.split("] ", 1)[1].strip()
            expanded.append(
                RetrievedEvidence(
                    memory_id=f"{row.memory_id}::line{idx}",
                    content=line_text,
                    score=row.score,
                    retrieval_source=row.retrieval_source,
                    reason=row.reason,
                    user_id=row.user_id,
                    memory_type=row.memory_type,
                    scope=row.scope,
                    entities=list(row.entities),
                    time=row.time,
                    source_message_ids=list(row.source_message_ids),
                    metadata={
                        **dict(row.metadata or {}),
                        "parent_memory_id": row.memory_id,
                        "source_line_index": idx,
                        "linewise_expanded": True,
                    },
                )
            )
    return expanded


class GovMemIncrementalBackbone:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_client: LLMClient,
        config: dict[str, Any],
        output_dir: Path,
        dataset_name: str,
    ):
        self.config = config
        self.policy_backbone = RAGPolicyBackbone(
            llm_client=llm_client,
            embedding_client=embedding_client,
            config=config,
            output_dir=output_dir,
            dataset_name=dataset_name,
        )

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        base = self.policy_backbone.run_instance(instance)
        allowed = list(base.reasoning_state.selected_evidence)
        required_slot_plan = build_required_slot_plan(instance.question, base.query_plan)
        expanded_allowed = _explode_state_evidence_rows(allowed, required_slot_plan)
        current_state, frames = resolve_current_state(expanded_allowed)
        selected_evidence, selected_frames, slot_coverage = select_evidence_by_slot_coverage(
            frames=frames,
            evidence=expanded_allowed,
            required_slot_plan=required_slot_plan,
            requester=instance.asking_user_id,
            config=self.config,
        )
        ledger = build_current_state_ledger(selected_frames)
        reasoning_state = ReasoningState(
            selected_evidence=selected_evidence,
            reasoning_trace=[
                f"govmem_incremental started from {len(allowed)} policy-filtered evidence rows.",
                f"expanded to {len(expanded_allowed)} line-level rows for state resolution.",
                f"selected {len(selected_frames)} frames after typed slot coverage selection.",
            ],
            conflicts=[],
            conclusion_hint="Incremental Gov-Mem over RAG-Policy+A-Mem.",
            selected_frames=selected_frames,
            current_state_ledger=ledger_to_dict(ledger),
            required_slot_plan=required_slot_plan,
            slot_coverage=slot_coverage,
        )
        principal = build_principal(
            requester_id=instance.asking_user_id,
            requester_role=((instance.metadata.get("requester") or {}).get("role")),
            owner_user_id=infer_owner_user_id(
                messages=list(instance.messages),
                evidence_rows=selected_evidence,
                requester_id=instance.asking_user_id,
            ),
        )
        raw_action = base.action_decision.action if base.action_decision is not None else ("answer" if selected_evidence else "no_memory")
        action_decision = GovernedActionDecision(
            action=raw_action,
            answer_mode="redacted" if raw_action == "answer_redacted" else ("direct" if raw_action == "answer" else "abstain"),
            privacy_decision="partial" if raw_action == "answer_redacted" else ("allowed" if raw_action == "answer" else "unknown"),
            forgetting_decision=None,
            evidence_memory_ids=[row.memory_id for row in selected_evidence],
            rationale_summary="Incremental Gov-Mem action from policy-filtered evidence.",
        )
        answer_structured = build_answer_structured(
            question=instance.question,
            action=action_decision.action,
            reasoning_state=reasoning_state,
        )
        slot_audit = audit_answer_structured_slots(
            answer_structured=answer_structured,
            question=instance.question,
            query_plan=required_slot_plan,
            current_state_ledger=reasoning_state.current_state_ledger,
            selected_frames=selected_frames,
            config=self.config,
        )
        action_decision, correction_trace = correct_action_with_runtime_evidence(
            action_decision,
            answer_structured,
            principal,
            slot_audit,
            self.config,
            query_type=str(getattr(base.query_plan, "query_type", "") or ""),
            question=instance.question,
        )
        answer_structured.action = action_decision.action
        rendered = render_with_surface_replay(
            answer_structured,
            slot_audit,
            action_decision.__dict__,
            principal,
            self.config,
        )
        verifier = verify_rendered_answer_contains_slots(rendered, slot_audit, self.config)
        answer_result = AnswerResult(
            prediction=rendered,
            answer_text=rendered,
            used_memory_ids=[row.memory_id for row in selected_evidence],
            reasoning_summary=action_decision.rationale_summary,
            action=action_decision.action,
            answer_structured=answer_structured_to_dict(answer_structured),
            raw_response={
                "slot_audit": answer_structured_to_dict(slot_audit) if hasattr(slot_audit, "__dataclass_fields__") else {},
                "rendered_answer_verifier": verifier,
                "action_correction_trace": correction_trace,
                "selected_frame_typed_slots": [sorted(f"{frame.frame_type}.{slot}" for slot in frame.slots.keys()) for frame in selected_frames],
                "event_ledger_summary": {
                    "active_events": list((reasoning_state.current_state_ledger or {}).get("active_events", {}).keys()),
                    "canceled_events": list((reasoning_state.current_state_ledger or {}).get("canceled_events", {}).keys()),
                    "superseded_events": list((reasoning_state.current_state_ledger or {}).get("superseded_events", {}).keys()),
                    "deleted_events": list((reasoning_state.current_state_ledger or {}).get("deleted_events", {}).keys()),
                },
            },
        )
        base.debug_payload.update(
            {
                "experiment_mode": "govmem_rag_policy_incremental",
                "selected_evidence": [
                    {
                        "evidence_id": row.memory_id,
                        "source_type": row.metadata.get("source_type", "chunk"),
                        "text": row.content,
                        "slots": row.metadata.get("slots", {}),
                        "score": row.score,
                    }
                    for row in selected_evidence
                ],
                "current_state": {
                    "active_items": current_state.active_items,
                    "canceled_items": current_state.canceled_items,
                    "deleted_items": current_state.deleted_items,
                    "superseded_items": current_state.superseded_items,
                    "uncertain_items": current_state.uncertain_items,
                    "trace": current_state.trace,
                },
                "slot_coverage": slot_coverage,
                "action_correction_trace": correction_trace,
                "slot_audit": answer_result.raw_response["slot_audit"],
                "rendered_answer_verifier": verifier,
                "selected_frame_typed_slots": answer_result.raw_response["selected_frame_typed_slots"],
                "event_ledger_summary": answer_result.raw_response["event_ledger_summary"],
            }
        )
        return BackboneRunResult(
            query_plan=base.query_plan,
            retrieval_result=base.retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=base.debug_payload,
        )
