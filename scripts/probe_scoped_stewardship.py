#!/usr/bin/env python3
"""Run only the source-grounded scoped-stewardship adjudicator on a saved case graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gov_mem.governance_runtime.query_policy_authorization import attach_query_scoped_policy_authorizations
from gov_mem.graph.governed_graph import GovernedMemoryGraph, GraphEdge, GraphNode
from gov_mem.llm.client import LLMClient, LLMConfig


def _load_graph(path: Path) -> GovernedMemoryGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GovernedMemoryGraph(
        graph_id=str(payload["graph_id"]),
        instance_id=payload.get("instance_id"),
        metadata=dict(payload.get("metadata") or {}),
        nodes=[GraphNode(**row) for row in list(payload.get("nodes") or [])],
        edges=[GraphEdge(**row) for row in list(payload.get("edges") or [])],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--debug", required=True)
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    graph = _load_graph(Path(args.graph))
    debug = json.loads(Path(args.debug).read_text(encoding="utf-8"))
    requester_id = str((debug.get("principal_relation_ledger") or {}).get("requester_id") or "")
    owner_by_message = dict((debug.get("information_owner_ledger") or {}).get("owner_by_message_id") or {})
    utility = dict(debug.get("utility_source_locator") or {})
    utility_ids = {str(value) for value in list(utility.get("source_message_ids") or []) if str(value)}
    utility_ids.update(
        str(value)
        for value in list((debug.get("authorization_context_locator") or {}).get("source_message_ids") or [])
        if str(value)
    )
    selected_fact_ids = [str(value) for value in list(utility.get("selected_fact_message_ids") or []) if str(value)]
    owner_id = next((str(owner_by_message.get(message_id) or "") for message_id in selected_fact_ids if owner_by_message.get(message_id)), "")
    if not requester_id or not owner_id:
        raise SystemExit("Saved case has no closed requester/owner pair for stewardship probing.")
    nodes = {node.node_id: node for node in graph.nodes}
    aligned_ids = {
        edge.target_id
        for edge in graph.edges
        if edge.edge_type == "has_slot"
        and edge.source_id in nodes
        and str((nodes[edge.source_id].attributes or {}).get("owner_id") or "") == owner_id
        and set(str(value) for value in list((nodes[edge.source_id].provenance or {}).get("source_message_ids") or [])) & set(selected_fact_ids)
    }
    alignment = {"bindings": {"probe_record_collection": {"anchor_slot_node_ids": sorted(aligned_ids)}}}
    client = LLMClient(LLMConfig(
        provider="yunwu", api_base="https://yunwu.ai/v1", api_key_env="YUNWU_API_KEY",
        temperature=0.0, max_retries=1, request_timeout=60,
    ))
    result = attach_query_scoped_policy_authorizations(
        graph=graph,
        semantic_alignment=alignment,
        principal_relation_ledger={
            "requester_id": requester_id, "owner_id": None,
            "effective_relation": "unknown", "effective_status": "unknown",
        },
        owner_id=owner_id,
        governance_policy_atom_ids=None,
        utility_source_message_ids=utility_ids,
        llm_client=client,
        model_name=args.model,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
