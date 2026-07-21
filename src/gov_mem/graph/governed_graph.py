from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    edge_id: str
    edge_type: str
    source_id: str
    target_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernedMemoryGraph:
    graph_id: str
    instance_id: str | None
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        if any(existing.node_id == node.node_id for existing in self.nodes):
            return
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        if any(existing.edge_id == edge.edge_id for existing in self.edges):
            return
        self.edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "instance_id": self.instance_id,
            "metadata": dict(self.metadata or {}),
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
        }
