from __future__ import annotations

import math
from dataclasses import dataclass

import requests

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
    def __init__(self, rows: list[DenseIndexRow], backend: str, fallback_reason: str | None = None):
        self.rows = rows
        self.backend = backend
        self.last_query_backend = backend
        self.fallback_reason = fallback_reason

    @classmethod
    def build(
        cls,
        *,
        items: list[MemoryItem],
        llm_client: LLMClient,
        embedding_model: str,
        embedding_texts: list[str] | None = None,
        allow_fallback: bool = True,
    ) -> "DenseMemoryIndex":
        texts = list(embedding_texts) if embedding_texts is not None else [
            _memory_item_to_embedding_text(item) for item in items
        ]
        if len(texts) != len(items):
            raise ValueError(
                f"Embedding text count mismatch: items={len(items)} texts={len(texts)}"
            )
        try:
            embeddings = llm_client.embed_texts(model=embedding_model, texts=texts)
            # A provider can return a successful HTTP response with a partial
            # or empty data array.  ``zip`` would silently truncate the index,
            # turning an embedding transport defect into a false retrieval
            # miss.  Fall back to the deterministic in-process index unless
            # every authorized item has a non-empty vector.
            if len(embeddings) != len(items) or any(not vector for vector in embeddings):
                raise ValueError(
                    f"Embedding response count/vector mismatch: requested={len(items)} received={len(embeddings)}"
                )
            rows = [
                DenseIndexRow(memory_id=item.memory_id, text=text, embedding=embedding)
                for item, text, embedding in zip(items, texts, embeddings)
            ]
            return cls(rows=rows, backend="openai_embedding")
        except (LLMClientUnavailableError, requests.RequestException, ValueError) as exc:
            if not allow_fallback:
                raise
            return cls(
                rows=_sparse_rows(items=items, texts=texts),
                backend="sparse_heuristic",
                fallback_reason=type(exc).__name__,
            )

    def query(
        self,
        *,
        query_texts: list[str],
        top_k: int,
        llm_client: LLMClient,
        embedding_model: str,
        allow_fallback: bool = True,
    ) -> list[tuple[str, float]]:
        if not query_texts:
            return []

        if self.backend == "openai_embedding":
            try:
                query_embeddings = llm_client.embed_texts(model=embedding_model, texts=query_texts)
            except (LLMClientUnavailableError, requests.RequestException, ValueError) as exc:
                if not allow_fallback:
                    raise
                query_embeddings = []
                self.fallback_reason = type(exc).__name__
            if len(query_embeddings) == len(query_texts) and all(query_embeddings):
                self.last_query_backend = "openai_embedding"
                scores: dict[str, float] = {}
                for query_embedding in query_embeddings:
                    for row in self.rows:
                        if not row.embedding:
                            continue
                        score = _cosine_dense(query_embedding, row.embedding)
                        scores[row.memory_id] = max(score, scores.get(row.memory_id, float("-inf")))
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                out: list[tuple[str, float]] = []
                for memory_id, score in ranked[: min(top_k * 5, len(ranked))]:
                    if score <= 0:
                        continue
                    out.append((memory_id, score))
                    if len(out) >= top_k:
                        break
                return out

        # Keep retrieval useful when the provider becomes unavailable after
        # index construction; dense rows still retain their source text.
        self.last_query_backend = "sparse_heuristic_query_fallback"
        sparse_rows = self.rows if self.backend == "sparse_heuristic" else [
            DenseIndexRow(
                memory_id=row.memory_id,
                text=row.text,
                sparse_embedding=_embed_text_heuristic(row.text),
            )
            for row in self.rows
        ]
        scores: dict[str, float] = {}
        query_vectors = [_embed_text_heuristic(text) for text in query_texts]
        for query_vector in query_vectors:
            for row in sparse_rows:
                score = _cosine_sparse(query_vector, row.sparse_embedding or {})
                scores[row.memory_id] = max(score, scores.get(row.memory_id, float("-inf")))
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def _memory_item_to_embedding_text(item: MemoryItem) -> str:
    entities = " ".join(item.entities)
    return f"[{item.scope}] [{item.memory_type}] [{item.user_id or 'none'}] [{entities}] {item.content}"


def _sparse_rows(*, items: list[MemoryItem], texts: list[str]) -> list[DenseIndexRow]:
    return [
        DenseIndexRow(
            memory_id=item.memory_id,
            text=text,
            sparse_embedding=_embed_text_heuristic(text),
        )
        for item, text in zip(items, texts)
    ]


def _cosine_dense(left: list[float], right: list[float]) -> float:
    numer = sum(l * r for l, r in zip(left, right))
    left_norm = math.sqrt(sum(l * l for l in left)) or 1.0
    right_norm = math.sqrt(sum(r * r for r in right)) or 1.0
    return numer / (left_norm * right_norm)
