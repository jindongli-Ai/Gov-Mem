#!/usr/bin/env python3
"""Checks closed-set query-scoped policy authorization."""

from __future__ import annotations

import re

from gov_mem.governance_runtime.query_policy_authorization import (
    attach_query_scoped_policy_authorizations,
)
from gov_mem.governance_runtime.provenance_authorization import _scope_matches_principal
from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
from gov_mem.graph.governed_graph import GovernedMemoryGraph, GraphEdge, GraphNode


class _GrantLLM:
    def __init__(self, owner_id="owner"):
        self.owner_id = owner_id

    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"grants": [{
            "policy_atom_id": "policy-1",
            "slot_node_ids": ["slot-1"],
            "effect": "allow",
            "relation": "delegate",
            "governed_owner_id": self.owner_id,
            "policy_support_span": "The delegate may receive the operational field.",
        }]}


class _AliasedGrantLLM:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"policy_authorization": {"authorizations": [{
            "atom_id": "policy-1",
            "slots": [{"slot_node_id": "slot-1"}],
            "authorization_effect": "allow",
            "requester_owner_relation": "delegated",
            "owner_id": "owner",
            "support_span": "The delegate may receive the operational field.",
        }]}}


class _RelationAssignmentLLM:
    def __init__(self):
        self.calls = 0

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {"grants": []}
        assignment_id = re.search(r"relation_assignment::[a-f0-9]+", kwargs["user_prompt"]).group(0)
        return {"grants": [{
            "assignment_id": assignment_id,
            "slot_node_ids": ["slot-1"],
            "effect": "allow",
            "relation": "authorized_staff",
            "governed_owner_id": "owner",
            "support_span": "Assigned to the active record.",
        }]}


class _StewardshipLLM:
    def __init__(self, *, grant=True, slot_id=None, owner_id="owner"):
        self.grant = grant
        self.slot_id = slot_id
        self.owner_id = owner_id

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        if not self.grant:
            return {"grants": []}
        candidate_id = re.search(r"stewardship_candidate::[a-f0-9]+", kwargs["user_prompt"]).group(0)
        return {"grants": [{
            "candidate_id": candidate_id,
            "slot_node_ids": [self.slot_id or "slot-record-a"],
            "effect": "allow",
            "governed_owner_id": self.owner_id,
            "support_span": "I placed the alpha record and will maintain it.",
        }]}


class _DirectRequesterPolicyLLM:
    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        if "direct, requester-specific policy capability" not in kwargs["system_prompt"]:
            return {"grants": []}
        return {"grants": [{
            "policy_atom_id": "policy-1",
            "slot_node_ids": ["slot-1"],
            "effect": "allow",
            "requester_id": "family-user",
            "governed_owner_id": "owner",
            "requester_reference_span": "The delegate",
            "permission_support_span": "may receive the operational field.",
            "permission_semantics": "explicit_allow",
        }]}


class _ForbiddenDirectPolicyLLM:
    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        if "direct, requester-specific policy capability" not in kwargs["system_prompt"]:
            return {"grants": []}
        # A warning contains neither a requester reference nor a permission.
        return {"grants": [{
            "policy_atom_id": "policy-1", "slot_node_ids": ["slot-1"], "effect": "allow",
            "requester_id": "family-user", "governed_owner_id": "owner",
            "requester_reference_span": "The delegate",
            "permission_support_span": "must not share the operational field.",
            "permission_semantics": "not_allow",
        }]}


def _graph():
    graph = GovernedMemoryGraph(graph_id="query-policy", instance_id="query-policy")
    graph.add_node(GraphNode(
        "semantic::fact", "FactNode", "The operational field is ready.", attributes={"owner_id": "owner"},
    ))
    graph.add_node(GraphNode("slot-1", "SlotNode", "field=ready", {"slot_name": "field", "slot_value": "ready"}))
    graph.add_edge(GraphEdge("has-slot", "has_slot", "semantic::fact", "slot-1"))
    graph.add_node(GraphNode(
        "semantic::policy-1", "PolicyNode", "The delegate may receive the operational field.",
        provenance={"source_atom_id": "policy-1"},
    ))
    return graph


def _ledger():
    return {
        "requester_id": "delegate-user",
        "owner_id": "owner",
        "effective_relation": "delegate",
        "effective_status": "proven",
    }


def _stewardship_graph(*, speaker="operator", owner="owner"):
    graph = GovernedMemoryGraph(graph_id="stewardship", instance_id="stewardship")
    for record_id, slot_id, text, value in (
        ("semantic::record-a", "slot-record-a", "I placed the alpha record and will maintain it.", "alpha"),
        ("semantic::record-b", "slot-record-b", "The separate bravo record is ready.", "bravo"),
    ):
        graph.add_node(GraphNode(
            record_id, "FactNode", text, attributes={"owner_id": owner},
            provenance={"speaker": speaker, "source_atom_id": record_id, "source_message_ids": [record_id]},
        ))
        graph.add_node(GraphNode(slot_id, "SlotNode", f"field={value}", {"slot_name": "field", "slot_value": value}))
        graph.add_edge(GraphEdge(f"has-{slot_id}", "has_slot", record_id, slot_id))
    return graph


def _unknown_ledger():
    return {"requester_id": "operator", "owner_id": None, "effective_relation": "unknown", "effective_status": "unknown"}


def main() -> None:
    assert _scope_matches_principal("role::authorized_staff", "authorized_staff")
    alignment = {"bindings": {"record": {"anchor_slot_node_ids": ["slot-1"]}}}
    graph = _graph()
    result = attach_query_scoped_policy_authorizations(
        graph=graph,
        semantic_alignment=alignment,
        principal_relation_ledger=_ledger(),
        owner_id="owner",
        governance_policy_atom_ids={"policy-1"},
        llm_client=_GrantLLM(),
        model_name="fake",
    )
    assert result["available"], result
    assert any(
        edge.edge_type == "allows" and edge.source_id == "role::delegate" and edge.target_id == "slot-1"
        for edge in graph.edges
    )
    aliased_graph = _graph()
    aliased = attach_query_scoped_policy_authorizations(
        graph=aliased_graph,
        semantic_alignment=alignment,
        principal_relation_ledger=_ledger(),
        owner_id="owner",
        governance_policy_atom_ids={"policy-1"},
        llm_client=_AliasedGrantLLM(),
        model_name="fake",
    )
    assert aliased["available"], aliased
    rejected = attach_query_scoped_policy_authorizations(
        graph=_graph(),
        semantic_alignment=alignment,
        principal_relation_ledger=_ledger(),
        owner_id="owner",
        governance_policy_atom_ids={"policy-1"},
        llm_client=_GrantLLM(owner_id="different-owner"),
        model_name="fake",
    )
    assert not rejected["available"], rejected
    direct_policy_graph = _graph()
    direct_policy = attach_query_scoped_policy_authorizations(
        question="What is the operational field and the restricted interpretation?",
        graph=direct_policy_graph,
        semantic_alignment=alignment,
        principal_relation_ledger={
            "requester_id": "family-user", "owner_id": "owner",
            "effective_relation": "unknown", "effective_status": "unknown",
        },
        owner_id="owner", governance_policy_atom_ids={"policy-1"},
        llm_client=_DirectRequesterPolicyLLM(), model_name="fake",
    )
    assert direct_policy["available"], direct_policy
    assert any(
        edge.edge_type == "allows" and edge.provenance.get("direct_requester_policy_authorization")
        and edge.provenance.get("requester_id") == "family-user"
        for edge in direct_policy_graph.edges
    )
    forbidden_graph = _graph()
    forbidden_graph.nodes[-1].label = "The delegate must not share the operational field."
    forbidden = attach_query_scoped_policy_authorizations(
        question="What is the operational field?", graph=forbidden_graph,
        semantic_alignment=alignment,
        principal_relation_ledger={
            "requester_id": "family-user", "owner_id": "owner",
            "effective_relation": "unknown", "effective_status": "unknown",
        },
        owner_id="owner", governance_policy_atom_ids={"policy-1"},
        llm_client=_ForbiddenDirectPolicyLLM(), model_name="fake",
    )
    assert not forbidden["available"], forbidden
    direct_policy_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_attributes": ["operational_field", "restricted_interpretation"]},
        graph=direct_policy_graph, principal_relation="unknown", owner_id="owner",
        semantic_alignment={"bindings": {
            "operational_field": {"slot_name": "field", "anchor_slot_node_ids": ["slot-1"]},
        }},
        principal_relation_ledger={
            "requester_id": "family-user", "owner_id": "owner",
            "effective_relation": "unknown", "effective_status": "unknown",
        },
    )
    assert direct_policy_certificate["authorized"], direct_policy_certificate
    assert direct_policy_certificate["requires_redaction"], direct_policy_certificate
    assert direct_policy_certificate["unresolved_requested_attributes"] == ["restricted_interpretation"]
    assert [item["slot_node_id"] for item in direct_policy_certificate["realizations"]] == ["slot-1"]
    relation_graph = _graph()
    relation_result = attach_query_scoped_policy_authorizations(
        graph=relation_graph,
        semantic_alignment=alignment,
        principal_relation_ledger={
            "requester_id": "staff-user", "owner_id": "owner", "effective_relation": "authorized_staff",
            "effective_status": "proven", "records": [{
                "requester_id": "staff-user", "owner_id": "owner", "relation": "authorized_staff", "status": "proven",
                "supports": [{
                    "message_id": "m-assignment", "source_span": "Assigned to the active record.",
                    "evidence_kind": "explicit_assignment",
                }],
            }],
        },
        owner_id="owner", governance_policy_atom_ids={"policy-1"}, llm_client=_RelationAssignmentLLM(), model_name="fake",
    )
    assert relation_result["available"], relation_result
    assert any(
        edge.edge_type == "allows" and edge.provenance.get("relation_assignment_authorization")
        for edge in relation_graph.edges
    )

    # A source-grounded operational capability is valid only for the exact
    # record spoken and maintained by the requester, without creating a broad
    # requester-owner relation.
    stewardship_graph = _stewardship_graph()
    stewardship = attach_query_scoped_policy_authorizations(
        graph=stewardship_graph,
        semantic_alignment={"bindings": {"record": {"anchor_slot_node_ids": ["slot-record-a"]}}},
        principal_relation_ledger=_unknown_ledger(), owner_id="owner", governance_policy_atom_ids=None,
        llm_client=_StewardshipLLM(), model_name="fake",
    )
    assert stewardship["available"] and stewardship["grants"][0]["authority_kind"] == "scoped_stewardship", stewardship
    certificate = certify_graph_slot_paths(
        semantic_spec={"requested_attributes": ["record"]}, graph=stewardship_graph,
        principal_relation="unknown", owner_id="owner",
        semantic_alignment={"bindings": {"record": {"slot_name": "field", "anchor_slot_node_ids": ["slot-record-a"]}}},
        principal_relation_ledger=_unknown_ledger(),
    )
    assert certificate["authorized"] and certificate["scoped_capability_authorized"], certificate

    # Same actor/title but no operational statement is not permission.
    no_stewardship = attach_query_scoped_policy_authorizations(
        graph=_stewardship_graph(), semantic_alignment={"bindings": {"record": {"anchor_slot_node_ids": ["slot-record-a"]}}},
        principal_relation_ledger=_unknown_ledger(), owner_id="owner", governance_policy_atom_ids=None,
        llm_client=_StewardshipLLM(grant=False), model_name="fake",
    )
    assert not no_stewardship["available"], no_stewardship

    # Capability for record A cannot be used for the same owner's record B.
    cross_record_graph = _stewardship_graph()
    cross_record = attach_query_scoped_policy_authorizations(
        graph=cross_record_graph, semantic_alignment={"bindings": {"record": {"anchor_slot_node_ids": ["slot-record-a", "slot-record-b"]}}},
        principal_relation_ledger=_unknown_ledger(), owner_id="owner", governance_policy_atom_ids=None,
        llm_client=_StewardshipLLM(), model_name="fake",
    )
    assert cross_record["available"], cross_record
    cross_record_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_attributes": ["record"]}, graph=cross_record_graph, principal_relation="unknown", owner_id="owner",
        semantic_alignment={"bindings": {"record": {"slot_name": "field", "anchor_slot_node_ids": ["slot-record-b"]}}},
        principal_relation_ledger=_unknown_ledger(),
    )
    assert not cross_record_certificate["authorized"], cross_record_certificate

    # A provenance edge tied to a different owner is rejected deterministically.
    cross_owner_graph = _stewardship_graph(owner="other-owner")
    cross_owner = attach_query_scoped_policy_authorizations(
        graph=cross_owner_graph, semantic_alignment={"bindings": {"record": {"anchor_slot_node_ids": ["slot-record-a"]}}},
        principal_relation_ledger=_unknown_ledger(), owner_id="owner", governance_policy_atom_ids=None,
        llm_client=_StewardshipLLM(), model_name="fake",
    )
    assert not cross_owner["available"], cross_owner
    print("query_policy_authorization_smoke=PASS")


if __name__ == "__main__":
    main()
