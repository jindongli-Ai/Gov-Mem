from __future__ import annotations

from gov_mem.data.schema import QueryPlan, RetrievedEvidence
from gov_mem.memory.symbolic_store import SymbolicMemoryStore


class SymbolicRetriever:
    def __init__(self, *, top_k: int):
        self.top_k = top_k

    def retrieve(self, *, plan: QueryPlan, store: SymbolicMemoryStore) -> list[RetrievedEvidence]:
        filters = plan.symbolic_filters or {}
        items = store.filter(
            user_ids=list(filters.get("user_ids", []) or []),
            entities=list(filters.get("entities", []) or []),
            memory_types=list(filters.get("memory_types", []) or []),
            scopes=list(filters.get("scopes", []) or []),
            source_message_ids=list(filters.get("source_message_ids", []) or []),
            time_values=list(filters.get("time_values", []) or []),
            top_k=self.top_k,
        )
        evidence: list[RetrievedEvidence] = []
        for rank, item in enumerate(items):
            score = 1.0 / float(rank + 1)
            evidence.append(
                RetrievedEvidence(
                    memory_id=item.memory_id,
                    content=item.content,
                    score=score,
                    retrieval_source="symbolic",
                    reason="symbolic filter match",
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
