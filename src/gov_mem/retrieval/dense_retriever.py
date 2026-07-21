from __future__ import annotations

from gov_mem.data.schema import MemoryItem, QueryPlan, RetrievedEvidence
from gov_mem.llm.client import LLMClient
from gov_mem.memory.dense_index import DenseMemoryIndex


class DenseRetriever:
    def __init__(self, *, llm_client: LLMClient, embedding_model: str, top_k: int):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(
        self,
        *,
        plan: QueryPlan,
        index: DenseMemoryIndex,
        memory_by_id: dict[str, MemoryItem],
    ) -> list[RetrievedEvidence]:
        rows = index.query(
            query_texts=plan.dense_queries,
            top_k=self.top_k,
            llm_client=self.llm_client,
            embedding_model=self.embedding_model,
        )
        evidence: list[RetrievedEvidence] = []
        for memory_id, score in rows:
            item = memory_by_id[memory_id]
            evidence.append(
                RetrievedEvidence(
                    memory_id=memory_id,
                    content=item.content,
                    score=float(score),
                    retrieval_source="dense",
                    reason="dense semantic match",
                    user_id=item.user_id,
                    memory_type=item.memory_type,
                    scope=item.scope,
                    entities=item.entities,
                    time=item.time,
                    source_message_ids=item.source_message_ids,
                    metadata=dict(item.metadata or {}),
                )
            )
        return evidence
