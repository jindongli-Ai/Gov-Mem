from __future__ import annotations

from gov_mem.data.schema import EvidenceFrame, QueryPlan, ReasoningState, RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frames
from gov_mem.governance_runtime.state_ledger import build_current_state_ledger, ledger_to_dict
from gov_mem.reasoning.operators import (
    apply_compare,
    apply_not_filter,
    apply_temporal_order,
    build_required_slot_plan,
    detect_conflicts,
    select_evidence_by_slot_coverage,
)


class SymbolicReasoner:
    def reason(self, *, plan: QueryPlan, evidence: list[RetrievedEvidence], config: dict | None = None) -> ReasoningState:
        cfg = config or {}
        trace: list[str] = [f"Initial evidence count: {len(evidence)}"]
        current = apply_not_filter(plan, evidence)
        if len(current) != len(evidence):
            trace.append(f"NOT filter removed {len(evidence) - len(current)} evidence items.")
        if "temporal_order" in plan.reasoning_ops:
            current = apply_temporal_order(plan, current)
            trace.append("Applied temporal ordering.")
        if "compare" in plan.reasoning_ops:
            current = apply_compare(current)
            trace.append("Applied compare operator.")

        conflicts = detect_conflicts(current) if "conflict_check" in plan.reasoning_ops else []
        if conflicts:
            trace.append(f"Detected {len(conflicts)} potential conflicts.")

        runtime_cfg = dict((cfg.get("governance_runtime") or {}))
        frames = compile_evidence_frames(current) if bool(runtime_cfg.get("enable_evidence_frame_compiler", True)) else []
        trace.append(f"Compiled {len(frames)} evidence frames.")
        question = ""
        runtime_meta = cfg.get("_runtime_instance_metadata") or {}
        if isinstance(runtime_meta, dict):
            question = str(runtime_meta.get("question") or "")
        required_slot_plan = build_required_slot_plan(question, plan)
        selected_evidence, selected_frames, slot_coverage = select_evidence_by_slot_coverage(
            frames=frames,
            evidence=current,
            required_slot_plan=required_slot_plan,
            requester=((cfg.get("_runtime_instance_metadata") or {}).get("requester") or {}).get("principal_id"),
            config=cfg,
        )
        trace.append(
            f"Selected {len(selected_frames)} frames covering {len(slot_coverage.get('covered_slots', []))} required slots."
        )
        ledger = build_current_state_ledger(selected_frames) if bool(runtime_cfg.get("enable_current_state_ledger", True)) else None
        if ledger is not None:
            trace.append(
                f"Built current-state ledger with {len(ledger.active_slots)} active, {len(ledger.canceled_slots)} canceled slots."
            )

        conclusion_hint = self._build_conclusion_hint(plan, selected_evidence, conflicts, slot_coverage)
        return ReasoningState(
            selected_evidence=selected_evidence,
            selected_frames=selected_frames,
            current_state_ledger=ledger_to_dict(ledger) if ledger is not None else {},
            required_slot_plan=required_slot_plan,
            slot_coverage=slot_coverage,
            reasoning_trace=trace,
            conflicts=conflicts,
            conclusion_hint=conclusion_hint,
        )

    @staticmethod
    def _build_conclusion_hint(
        plan: QueryPlan,
        evidence: list[RetrievedEvidence],
        conflicts: list[dict],
        slot_coverage: dict,
    ) -> str | None:
        if not evidence:
            return "Evidence is limited; answer conservatively."
        parts = [f"Relevant evidence count: {len(evidence)}"]
        covered = slot_coverage.get("covered_slots", [])
        missing = slot_coverage.get("missing_slots", [])
        if covered:
            parts.append(f"Covered slots: {', '.join(covered)}")
        if missing:
            parts.append(f"Missing slots: {', '.join(missing[:6])}")
        if plan.target_users:
            parts.append(f"Target users: {', '.join(plan.target_users)}")
        if plan.target_entities:
            parts.append(f"Target entities: {', '.join(plan.target_entities)}")
        if conflicts:
            parts.append("Conflicts detected; prefer current active state.")
        return ". ".join(parts)
