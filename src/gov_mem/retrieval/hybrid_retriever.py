from __future__ import annotations

from gov_mem.data.schema import QueryPlan, RetrievedEvidence
from gov_mem.governance_runtime.access import (
    build_principal,
    is_requester_authorized,
    normalize_role,
    requester_has_logistics_access,
    resolve_slot_access,
    requires_redaction_for_requester,
)
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frames, frame_to_dict
from gov_mem.retrieval.dense_retriever import DenseRetriever
from gov_mem.retrieval.symbolic_retriever import SymbolicRetriever


class HybridRetriever:
    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever,
        symbolic_retriever: SymbolicRetriever,
        final_top_k: int,
    ):
        self.dense_retriever = dense_retriever
        self.symbolic_retriever = symbolic_retriever
        self.final_top_k = final_top_k

    def retrieve(self, *, plan: QueryPlan, dense_index, symbolic_store, memory_by_id, requester=None, config=None) -> dict:
        dense = self.dense_retriever.retrieve(plan=plan, index=dense_index, memory_by_id=memory_by_id)
        symbolic = self.symbolic_retriever.retrieve(plan=plan, store=symbolic_store)

        merged: dict[str, RetrievedEvidence] = {}
        for row in dense + symbolic:
            existing = merged.get(row.memory_id)
            if existing is None:
                merged[row.memory_id] = row
                continue
            existing.score = max(existing.score, row.score)
            existing.retrieval_source = "hybrid"
            existing.reason = f"{existing.reason}; {row.reason}"

        reranked = sorted(
            merged.values(),
            key=lambda item: _hybrid_rank_score(item, plan),
            reverse=True,
        )
        before_filter = reranked[: self.final_top_k]
        after_filter, filtered = apply_privacy_filter(
            evidence=before_filter,
            requester=requester,
            query_plan=plan,
            config=config or {},
        )
        frames = compile_evidence_frames(after_filter)
        requester_role = None
        if isinstance(config, dict):
            requester_role = (
                ((config.get("_runtime_instance_metadata") or {}).get("requester") or {}).get("role")
                or ((config.get("_runtime_instance_metadata") or {}).get("observable") or {}).get("asker_role")
            )
        principal = build_principal(
            requester_id=requester,
            requester_role=requester_role,
            owner_user_id=frames[0].owner_user if frames else None,
        )
        slot_access_results = [
            resolve_slot_access(frame=frame, principal=principal, meta=(row.metadata or {}))
            for frame, row in zip(frames, after_filter)
        ]
        slot_coverage_before = _slot_coverage(compile_evidence_frames(before_filter))
        slot_coverage_after = _slot_coverage(frames)
        return {
            "retrieved_before_privacy_filter": before_filter,
            "retrieved_after_privacy_filter": after_filter,
            "filtered_evidence": filtered,
            "frame_candidates": [frame_to_dict(frame) for frame in frames],
            "slot_access_results": slot_access_results,
            "slot_coverage_before_selection": slot_coverage_before,
            "slot_coverage_after_selection": slot_coverage_after,
        }


def _hybrid_rank_score(item: RetrievedEvidence, plan: QueryPlan) -> float:
    score = item.score
    if item.user_id and item.user_id in plan.target_users:
        score += 0.4
    if item.memory_type and item.memory_type in plan.required_memory_types:
        score += 0.25
    if set(item.entities) & set(plan.target_entities):
        score += 0.2
    if item.retrieval_source == "hybrid":
        score += 0.15
    return score


def apply_privacy_filter(evidence, requester, query_plan, config):
    ablation = config.get("ablation", {}) if isinstance(config, dict) else {}
    memory_cfg = config.get("memory", {}) if isinstance(config, dict) else {}
    use_privacy_filter = bool(ablation.get("use_privacy_filter", True)) and bool(memory_cfg.get("enable_privacy_filter", True))
    use_forgetting = bool(ablation.get("use_forgetting", True)) and bool(memory_cfg.get("enable_forgetting", True))
    if not use_privacy_filter and not use_forgetting:
        return evidence, []

    kept = []
    filtered = []
    requester_role = None
    if isinstance(config, dict):
        requester_role = (
            ((config.get("_runtime_instance_metadata") or {}).get("requester") or {}).get("role")
            or ((config.get("_runtime_instance_metadata") or {}).get("observable") or {}).get("asker_role")
        )
    requester_role = normalize_role(requester_role) if requester_role else None
    logistics_delegation = requester_has_logistics_access(
        requester_id=requester,
        requester_role=requester_role,
        evidence_rows=evidence,
    )
    for row in evidence:
        meta = row.metadata or {}
        memory_status = meta.get("memory_status") or (row.metadata.get("memory_status") if row.metadata else None)
        if use_forgetting and memory_status in {"deleted", "superseded"}:
            filtered.append({"memory_id": row.memory_id, "reason": "deleted_or_superseded", "memory_status": memory_status})
            continue
        if use_privacy_filter and not is_requester_authorized(
            meta=meta,
            requester_id=requester,
            requester_role=requester_role,
            owner_user_id=row.user_id,
        ):
            if logistics_delegation and row.metadata and row.metadata.get("redaction_required") and any(
                token in row.content.lower()
                for token in ["appointment", "arrival", "location", "suite", "parking", "schedule", "scheduled", "time"]
            ):
                row.metadata["requires_redaction"] = False
                kept.append(row)
                continue
            filtered.append({"memory_id": row.memory_id, "reason": "unauthorized_user"})
            continue
        if requires_redaction_for_requester(
            meta=meta,
            requester_id=requester,
            requester_role=requester_role,
            owner_user_id=row.user_id,
        ):
            row.metadata["requires_redaction"] = True
        kept.append(row)
    return kept, filtered


def _slot_coverage(frames) -> dict:
    coverage: dict[str, int] = {}
    for frame in frames:
        for slot_name in frame.slots.keys():
            coverage[slot_name] = coverage.get(slot_name, 0) + 1
    return coverage
