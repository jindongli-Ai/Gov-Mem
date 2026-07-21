from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from gov_mem.backbones.common import (
    BackboneRunResult,
    answer_with_retrieved_evidence,
    build_rag_chunks,
    build_reasoning_state,
    retrieve_rag_chunks,
    save_rag_chunks,
)
from gov_mem.data.schema import GovernedActionDecision, MemoryInstance
from gov_mem.llm.client import LLMClient
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.reasoning.operators import build_required_slot_plan


class RAGNaiveBackbone:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_client: LLMClient,
        config: dict[str, Any],
        output_dir: Path,
        dataset_name: str,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = dataset_name

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        chunks = build_rag_chunks(instance, self.config)
        save_rag_chunks(self.output_dir, self.dataset_name, instance.instance_id, chunks)
        plan, retrieval_result, evidence = retrieve_rag_chunks(
            instance=instance,
            chunks=chunks,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            config=self.config,
            planning_client=self.llm_client,
            planning_model=resolve_llm_model(self.config, "query_planning"),
        )
        answer_result = answer_with_retrieved_evidence(
            instance=instance,
            evidence=evidence,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "answering"),
            action="answer" if evidence else "no_memory",
            used_chunk_ids=[row.memory_id for row in evidence],
            semantic_spec=plan.semantic_spec,
        )
        compiled_frames = []
        for frame in (answer_result.raw_response or {}).get("compiled_frames", []):
            if isinstance(frame, dict):
                compiled_frames.append(frame)
        reasoning_state = build_reasoning_state(
            evidence,
            trace=[
                f"rag_naive selected {len(evidence)} retrieved chunks after diversified retrieval.",
                f"surface replay selected {len((answer_result.raw_response or {}).get('surface_lines', []))} evidence lines.",
            ],
            slot_coverage=(answer_result.raw_response or {}).get("slot_verifier", {}),
            selected_frames=[],
            required_slot_plan=build_required_slot_plan(instance.question, plan),
        )
        action_decision = GovernedActionDecision(
            action=answer_result.action,
            answer_mode="direct" if answer_result.action == "answer" else "abstain",
            privacy_decision="unknown",
            forgetting_decision=None,
            evidence_memory_ids=answer_result.used_memory_ids,
            rationale_summary=answer_result.reasoning_summary,
        )
        debug_payload = {
            "experiment_mode": "rag_naive",
            "rag_chunks": [asdict(chunk) for chunk in chunks],
            "retrieved_chunks": [
                {
                    "chunk_id": row.memory_id,
                    "text": row.content,
                    "score": row.score,
                    "source_message_ids": row.source_message_ids,
                }
                for row in evidence
            ],
            "atomic_memories": [],
            "retrieved_atomic_memories": [],
            "policy_decisions": [],
            "selected_evidence": [
                {
                    "evidence_id": row.memory_id,
                    "source_type": "chunk",
                    "text": row.content,
                    "score": row.score,
                }
                for row in evidence
            ],
            "current_state": {},
            "slot_coverage": (answer_result.raw_response or {}).get("slot_verifier", {}),
            "action_correction_trace": [],
            "surface_lines": (answer_result.raw_response or {}).get("surface_lines", []),
            "compiled_frames": compiled_frames,
            "retrieval_queries": retrieval_result.get("query_variants", []),
            "retrieval_candidates": retrieval_result.get("retrieval_candidates", []),
        }
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=debug_payload,
        )
