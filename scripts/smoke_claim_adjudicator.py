#!/usr/bin/env python3
"""Current-state fallback must ignore meta mentions but keep real updates."""

from gov_mem.data.schema import RetrievedEvidence
import requests

from gov_mem.legacy.claim_adjudicator import (
    _fallback_slot_is_structurally_compatible,
    _paired_operational_role_conflict,
    _resolve_current_collection_components,
    _slot_matches_attribute,
    _stage2_attribute_fallback,
    adjudicate_claims,
)
from gov_mem.governance_runtime.provenance_authorization import _is_summary_attribute
from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
from gov_mem.graph.governed_graph import GraphEdge, GraphNode, GovernedMemoryGraph
from gov_mem.memory.dense_index import DenseIndexRow, DenseMemoryIndex


class FakeClient:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"decisions": [{
            "attribute": "approved_budget",
            "memory_id": "base",
            "slot_name": "approved_budget",
            "value": "221,000 USD",
            "evidence_span": "Maple budget is 221,000 USD.",
            "decision": "answer",
            "confidence": 1.0,
        }]}


class FailingEmbeddingClient:
    def embed_texts(self, **_kwargs):
        raise requests.HTTPError("provider unavailable")


def row(memory_id, text, message_id, slots, *, changed_fields=None, served=True):
    return RetrievedEvidence(
        memory_id=memory_id,
        content=text,
        score=1.0,
        retrieval_source="test",
        reason="test",
        source_message_ids=[message_id],
        time="2026-01-01T00:00:00",
        metadata={
            "slots": slots,
            "stage2_semantic_rerank": {
                "served_attributes": ["approved_budget"] if served else [],
                "typed_fields": [{
                    "attribute": "approved_budget",
                    "slot_name": "approved_budget",
                    "value": next(iter(slots.values())),
                }] if served else [],
            },
            "semantic_tags": {
                "claims": [{
                    "property_label": "approved_budget",
                    "value_span": next(iter(slots.values())),
                    "claim_span": text,
                }],
                "state_delta": {
                    "operation": "set" if changed_fields else "none",
                    "changed_fields": changed_fields or {},
                },
            },
        },
    )


def main():
    # Retrieval must remain usable when an optional embedding provider fails.
    # The dense index falls back to source-text overlap without changing the
    # governance or answer-selection contracts.
    dense_index = DenseMemoryIndex(
        rows=[DenseIndexRow(memory_id="budget", text="approved budget")],
        backend="openai_embedding",
    )
    assert dense_index.query(
        query_texts=["approved budget"],
        top_k=1,
        llm_client=FailingEmbeddingClient(),
        embedding_model="unused",
    )[0][0] == "budget"

    role_spec = {
        "requested_attributes": ["backup_helper_phrase", "helper_window"],
        "attribute_bindings": [
            {"attribute": "backup_helper_phrase", "support_span": "backup helper phrase"},
            {"attribute": "helper_window", "support_span": "helper window"},
        ],
    }
    mixed_record = {"slots": {"helper_time_range": "8:40 AM to 10:20 AM"}}
    assert not _slot_matches_attribute(
        "backup_helper_phrase", "helper_time_range", role_spec, mixed_record
    )
    assert _slot_matches_attribute("helper_window", "helper_time_range", role_spec, mixed_record)
    mixed_record["slots"]["visit_window"] = "8:25 AM to 10:05 AM"
    assert not _slot_matches_attribute("helper_window", "visit_window", role_spec, mixed_record)
    assert _paired_operational_role_conflict("dropoff_window", "pickup_window")
    assert not _slot_matches_attribute(
        "dropoff_window", "pickup_window", {
            "requested_attributes": ["dropoff_window", "pickup_window"],
            "attribute_bindings": [
                {"attribute": "dropoff_window", "support_span": "Friday dropoff window"},
                {"attribute": "pickup_window", "support_span": "Saturday pickup window"},
            ],
        },
    )

    current_collection = _resolve_current_collection_components(
        [
            {"attribute": "label_color", "memory_id": "old", "slot_name": "initial_color", "value": "silver"},
            {"attribute": "label_color", "memory_id": "new", "slot_name": "label_color", "value": "moss"},
            {"attribute": "overflow_point", "memory_id": "old-route", "slot_name": "first_overflow_point", "value": "hall bench lower basket"},
            {"attribute": "overflow_point", "memory_id": "new-route", "slot_name": "supply_overflow_point", "value": "from hall bench lower basket to laundry shelf upper tub"},
            {"attribute": "current_supply_state", "memory_id": "state", "slot_name": "resident_held_digits", "value": "2816"},
            {"attribute": "current_supply_state", "memory_id": "state-text", "slot_name": "supply_logistics", "value": "Sunday restocking"},
        ],
        candidate_by_id={
            "old": {"source_message_ids": ["message-old-color"], "timestamp": "2026-08-28T10:00:00"},
            "new": {"source_message_ids": ["message-new-color"], "timestamp": "2026-08-29T10:00:00"},
            "old-route": {"source_message_ids": ["message-old-route"], "timestamp": "2026-08-28T11:00:00"},
            "new-route": {"source_message_ids": ["message-new-route"], "timestamp": "2026-08-29T11:00:00"},
            "state": {"source_message_ids": ["message-state"], "timestamp": "2026-08-30T10:00:00"},
            "state-text": {"source_message_ids": ["message-state-text"], "timestamp": "2026-08-30T09:00:00"},
        },
    )
    resolved_values = {(item["attribute"], item["value"]) for item in current_collection}
    assert ("label_color", "moss") in resolved_values and ("label_color", "silver") not in resolved_values
    assert ("overflow_point", "from hall bench lower basket to laundry shelf upper tub") in resolved_values
    assert not any(item["value"] == "2816" for item in current_collection)
    assert _is_summary_attribute("current_plan_items")

    retired_subject = GovernedMemoryGraph(graph_id="historical-subject", instance_id="smoke")
    retired_subject.add_node(GraphNode(
        node_id="fact::retired", node_type="SemanticNode",
        label="Former object was not disclosed.",
        attributes={"atom_type": "fact_atom", "owner_id": "owner", "lifecycle": "active"},
    ))
    retired_subject.add_node(GraphNode(
        node_id="slot::subject", node_type="SlotNode", label="former object",
        attributes={"slot_name": "claim_subject_value", "slot_value": "former object", "slot_role": "claim_subject_value"},
        provenance={"source_atom_id": "fact::retired", "source_memory_id": "memory::retired", "source_message_ids": ["message::retired"]},
    ))
    retired_subject.add_node(GraphNode(
        node_id="role::self", node_type="RoleNode", label="self",
    ))
    retired_subject.add_edge(GraphEdge("has-slot", "has_slot", "fact::retired", "slot::subject"))
    retired_subject.add_edge(GraphEdge("allows-slot", "allows", "role::self", "slot::subject"))
    historical_certificate = certify_graph_slot_paths(
        semantic_spec={
            "requested_attributes": ["deleted_object"],
            "temporal_scope": "historical",
            "attribute_bindings": [{
                "attribute": "deleted_object",
                "slot_name": "claim_subject_value",
                "anchor_slot_node_id": "slot::subject",
            }],
        },
        graph=retired_subject,
        principal_relation="owner",
        owner_id="owner",
        utility_atom_ids={"fact::retired"},
        semantic_alignment={"bindings": {
            "deleted_object": {
                "attribute": "deleted_object",
                "slot_name": "claim_subject_value",
                "anchor_slot_node_id": "slot::subject",
            },
        }},
        stage2_authorized_atom_ids={"fact::retired"},
    )
    assert not historical_certificate["authorized"]
    assert "missing_active_graph_slot" in historical_certificate["reason"]

    # A noisy Stage-2 batch must not reuse one scalar slot for several
    # unrelated properties, while a claim-local status span may bridge an
    # open slot label such as pin_status.
    slot_reuse_spec = {
        "requested_attributes": ["current_path", "current_sort", "whether_any_helper_digits_remain"],
        "attribute_bindings": [
            {"attribute": "current_path", "evidence_slot_hint": "current_state"},
            {"attribute": "current_sort", "evidence_slot_hint": "current_state"},
            {"attribute": "whether_any_helper_digits_remain", "evidence_slot_hint": "current_state"},
        ],
    }
    reused_path_record = {
        "stage2_typed_fields": [
            {"attribute": attribute, "slot_name": "bins_path", "value": "front desk cart"}
            for attribute in slot_reuse_spec["requested_attributes"]
        ],
        "claim_slots": [{"slot_name": "bins_path", "claim_span": "Current Bins path is front desk cart."}],
    }
    assert _fallback_slot_is_structurally_compatible(
        attribute="current_path",
        slot_name="bins_path",
        semantic_spec=slot_reuse_spec,
        record=reused_path_record,
    )
    assert not _fallback_slot_is_structurally_compatible(
        attribute="current_sort",
        slot_name="bins_path",
        semantic_spec=slot_reuse_spec,
        record=reused_path_record,
    )
    assert not _fallback_slot_is_structurally_compatible(
        attribute="whether_any_helper_digits_remain",
        slot_name="bins_path",
        semantic_spec=slot_reuse_spec,
        record=reused_path_record,
    )
    assert _fallback_slot_is_structurally_compatible(
        attribute="whether_any_helper_digits_remain",
        slot_name="pin_status",
        semantic_spec=slot_reuse_spec,
        record={
            "stage2_typed_fields": [{
                "attribute": "whether_any_helper_digits_remain",
                "slot_name": "pin_status",
                "value": "no helper digits remain",
            }],
            "claim_slots": [{"slot_name": "pin_status", "claim_span": "no helper digits remain."}],
        },
    )

    spec = {
        "requested_attributes": ["approved_budget"],
        "temporal_scope": "current",
        "attribute_bindings": [{
            "attribute": "approved_budget",
            "support_span": "approved budget",
            "evidence_slot_hint": "approved_budget",
        }],
    }
    # Locator-selected utility closures may carry a valid typed slot that
    # Stage 2 did not repeat explicitly. Keep it available for chronology;
    # an ordinary Stage-2 record must still use an explicit binding.
    closure_record = {
        "memory_id": "closure",
        "source_text": "The current approved budget is 225,000 USD.",
        "timestamp": "2026-01-02T00:00:00",
        "source_message_ids": ["t130"],
        "stage2_served_attributes": ["approved_budget"],
        "utility_source_closure": True,
        "slots": {"approved_budget": "225,000 USD"},
        "state_delta": {"changed_fields": {}},
    }
    assert _stage2_attribute_fallback(
        "approved_budget", [closure_record], spec, question="What is the current approved budget?"
    )["value"] == "225,000 USD"
    ordinary_record = dict(closure_record, utility_source_closure=False)
    assert _stage2_attribute_fallback(
        "approved_budget", [ordinary_record], spec, question="What is the current approved budget?"
    ) is None

    meta = row(
        "meta",
        "Nothing in this message changes the approved budget.",
        "t129",
        {"approved_budget": "approved budget"},
    )
    base = row(
        "base",
        "Maple budget is 221,000 USD.",
        "t058",
        {"approved_budget": "221,000 USD"},
    )
    selected, _ = adjudicate_claims(
        question="What is the current approved budget?",
        semantic_spec=spec,
        evidence=[base, meta],
        llm_client=FakeClient(),
        model_name="test",
    )
    assert [(item["memory_id"], item["value"]) for item in selected] == [("base", "221,000 USD")], selected

    update = row(
        "update",
        "The approved budget is now 225,000 USD.",
        "t130",
        {"approved_budget": "225,000 USD"},
        changed_fields={"approved_budget": "225,000 USD"},
    )
    selected, _ = adjudicate_claims(
        question="What is the current approved budget?",
        semantic_spec=spec,
        evidence=[base, update],
        llm_client=FakeClient(),
        model_name="test",
    )
    assert [(item["memory_id"], item["value"]) for item in selected] == [("update", "225,000 USD")], selected
    print("claim_adjudicator_smoke=PASS")


if __name__ == "__main__":
    main()
