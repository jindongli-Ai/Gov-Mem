from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gov_mem.graph.governed_graph import GovernedMemoryGraph
from gov_mem.utils.io import append_jsonl, ensure_dir, read_json, write_json


class GovernedGraphStore:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def save(self, *, instance_id: str, graph: GovernedMemoryGraph) -> dict[str, str]:
        per_instance_path = ensure_dir(self.output_dir / "graphs" / instance_id) / "governed_graph.json"
        write_json(per_instance_path, graph.to_dict())

        aggregate_graph_path = self.output_dir / "debug" / "governed_graph.json"
        aggregate_payload = read_json(aggregate_graph_path) if aggregate_graph_path.exists() else {}
        if not isinstance(aggregate_payload, dict):
            aggregate_payload = {}
        aggregate_payload[instance_id] = {
            "graph_id": graph.graph_id,
            "instance_id": graph.instance_id,
            "metadata": dict(graph.metadata or {}),
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "graph_path": str(per_instance_path),
        }
        write_json(aggregate_graph_path, aggregate_payload)

        edge_path = self.output_dir / "debug" / "governed_graph_edges.jsonl"
        for edge in graph.edges:
            append_jsonl(
                edge_path,
                {
                    "instance_id": instance_id,
                    "graph_id": graph.graph_id,
                    "edge_id": edge.edge_id,
                    "edge_type": edge.edge_type,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "attributes": dict(edge.attributes or {}),
                    "provenance": dict(edge.provenance or {}),
                },
            )

        return {
            "per_instance_graph": str(per_instance_path),
            "aggregate_graph": str(aggregate_graph_path),
            "aggregate_edges": str(edge_path),
        }

    def load_aggregate(self) -> dict[str, Any]:
        aggregate_graph_path = self.output_dir / "debug" / "governed_graph.json"
        if not aggregate_graph_path.exists():
            return {}
        return read_json(aggregate_graph_path)
