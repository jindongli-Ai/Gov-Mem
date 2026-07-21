from __future__ import annotations

import math
from dataclasses import dataclass

from gov_mem.data.schema import MemoryItem
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().replace("\n", " ").split() if token]


def _embed_text_heuristic(text: str) -> dict[str, float]:
    vector: dict[str, float] = {}
    for token in _tokenize(text):
        vector[token] = vector.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
    return {key: value / norm for key, value in vector.items()}


def _cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


@dataclass
class DenseIndexRow:
    memory_id: str
    text: str
    embedding: list[float] | None = None
    sparse_embedding: dict[str, float] | None = None


class DenseMemoryIndex:
    def __init__(self, rows: list[DenseIndexRow], backend: str):
        self.rows = rows
        self.backend = backend

    @classmethod
    def build(
        cls,
        *,
        items: list[MemoryItem],
        llm_client: LLMClient,
        embedding_model: str,
    ) -> "DenseMemoryIndex":
        texts = [_memory_item_to_embedding_text(item) for item in items]
        try:
            embeddings = llm_client.embed_texts(model=embedding_model, texts=texts)
            rows = [
                DenseIndexRow(memory_id=item.memory_id, text=text, embedding=embedding)
                for item, text, embedding in zip(items, texts, embeddings)
            ]
            return cls(rows=rows, backend="openai_embedding")
        except LLMClientUnavailableError:
            rows = [
                DenseIndexRow(
                    memory_id=item.memory_id,
                    text=text,
                    sparse_embedding=_embed_text_heuristic(text),
                )
                for item, text in zip(items, texts)
            ]
            return cls(rows=rows, backend="sparse_heuristic")

    def query(
        self,
        *,
        query_texts: list[str],
        top_k: int,
        llm_client: LLMClient,
        embedding_model: str,
    ) -> list[tuple[str, float]]:
        if not query_texts:
            return []

        if self.backend == "openai_embedding":
            try:
                query_embeddings = llm_client.embed_texts(model=embedding_model, texts=query_texts)
            except LLMClientUnavailableError:
                query_embeddings = []
            if query_embeddings:
                scores: dict[str, float] = {}
                for query_embedding in query_embeddings:
                    for row in self.rows:
                        if not row.embedding:
                            continue
                        score = _cosine_dense(query_embedding, row.embedding)
                        scores[row.memory_id] = max(score, scores.get(row.memory_id, float("-inf")))
                return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        scores: dict[str, float] = {}
        query_vectors = [_embed_text_heuristic(text) for text in query_texts]
        for query_vector in query_vectors:
            for row in self.rows:
                score = _cosine_sparse(query_vector, row.sparse_embedding or {})
                scores[row.memory_id] = max(score, scores.get(row.memory_id, float("-inf")))
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def _memory_item_to_embedding_text(item: MemoryItem) -> str:
    entities = " ".join(item.entities)
    return f"[{item.scope}] [{item.memory_type}] [{item.user_id or 'none'}] [{entities}] {item.content}"


def _cosine_dense(left: list[float], right: list[float]) -> float:
    numer = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left)) or 1.0
    right_norm = math.sqrt(sum(r * r for r in right)) or 1.0
    return numer / (left_norm * right_norm)

