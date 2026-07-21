from __future__ import annotations

import re
from typing import Any

from gov_mem.graph.governed_graph import GovernedMemoryGraph


class GovernedGraphRetriever:
    def retrieve_paths(
        self,
        *,
        graph: GovernedMemoryGraph,
        query: str,
        requested_attributes: list[str] | None = None,
        max_paths: int = 8,
    ) -> list[dict[str, Any]]:
        query_tokens = {token for token in re.findall(r"[a-z0-9_]+", query.lower()) if len(token) >= 3}
        requested = {str(slot).strip() for slot in (requested_attributes or []) if str(slot).strip()}
        if not query_tokens and not requested:
            return []
        node_by_id = {node.node_id: node for node in graph.nodes}
        paths: list[dict[str, Any]] = []
        for edge in graph.edges:
            source = node_by_id.get(edge.source_id)
            target = node_by_id.get(edge.target_id)
            source_text = f"{(source.label if source else '')} {source.attributes if source else ''}".lower()
            target_text = f"{(target.label if target else '')} {target.attributes if target else ''}".lower()
            edge_slots = {str((edge.attributes or {}).get("slot_name") or "")}
            edge_slots.discard("")
            overlap = sum(1 for token in query_tokens if token in source_text or token in target_text)
            schema_overlap = len(edge_slots & requested)
            if overlap <= 0 and schema_overlap <= 0:
                continue
            paths.append(
                {
                    "edge_type": edge.edge_type,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "source_type": source.node_type if source else None,
                    "target_type": target.node_type if target else None,
                    "source_label": source.label if source else None,
                    "target_label": target.label if target else None,
                    "score": float(overlap + 2 * schema_overlap),
                    "provenance": dict(edge.provenance or {}),
                }
            )
        paths.sort(key=lambda row: (float(row["score"]), str(row["edge_type"])), reverse=True)
        return paths[:max_paths]
