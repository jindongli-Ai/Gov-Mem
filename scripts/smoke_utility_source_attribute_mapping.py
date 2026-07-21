#!/usr/bin/env python3
"""Source-local utility bridge must preserve exact attribute/value bindings."""

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.semantic_reranker import map_utility_source_attributes


class FakeClient:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"mappings": [
            {
                "memory_id": "points",
                "support_span": "Opening points are shelf A and drawer B.",
                "served_attributes": ["current_stocking_points"],
                "typed_fields": [{
                    "attribute": "current_stocking_points",
                    "slot_name": "stocking_points",
                    "value": "shelf A and drawer B",
                }],
                "authorized_for_requester": True,
                "authorization_support_span": "Opening points are shelf A and drawer B.",
            },
            {
                "memory_id": "overflow",
                "support_span": "Overflow moves to bin C.",
                "served_attributes": ["current_overflow_note"],
                "typed_fields": [{
                    "attribute": "current_overflow_note",
                    "slot_name": "overflow_location_destination",
                    "value": "bin C",
                }],
                "authorized_for_requester": True,
                "authorization_support_span": "Overflow moves to bin C.",
            },
        ]}


class MisbindingClient:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"mappings": [{
            "memory_id": "summary",
            "support_span": "Current state: shelf A, IP-7, and bin C.",
            "served_attributes": ["current_label", "current_stocking_points"],
            "typed_fields": [
                {"attribute": "current_label", "slot_name": "location_access_details", "value": "shelf A, IP-7, and bin C"},
                {"attribute": "current_stocking_points", "slot_name": "location_access_details", "value": "shelf A, IP-7, and bin C"},
            ],
            "authorized_for_requester": True,
            "authorization_support_span": "Current state: shelf A, IP-7, and bin C.",
        }]}


class OmittedTerseClaimClient:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        # Both mapping passes omit the terse current label; the closed-set
        # typed recovery must preserve it for later chronology adjudication.
        return {"mappings": []}


def row(memory_id, text, slots):
    return RetrievedEvidence(
        memory_id=memory_id,
        content=text,
        score=1.0,
        retrieval_source="test",
        reason="test",
        metadata={"slots": slots},
    )


def main():
    selected, decisions, filtered, debug = map_utility_source_attributes(
        question="Give the current stocking points and overflow note.",
        semantic_spec={"requested_attributes": ["current_stocking_points", "current_overflow_note"]},
        evidence=[
            row("points", "Opening points are shelf A and drawer B.", {"stocking_points": "shelf A and drawer B"}),
            row("overflow", "Overflow moves to bin C.", {"overflow_location_destination": "bin C"}),
        ],
        llm_client=FakeClient(),
        model_name="test",
    )
    assert {item.memory_id for item in selected} == {"points", "overflow"}, selected
    by_id = {item["chunk_id"]: item for item in decisions}
    assert by_id["points"]["served_attributes"] == ["current_stocking_points"], decisions
    assert by_id["overflow"]["served_attributes"] == ["current_overflow_note"], decisions
    assert not filtered, filtered
    assert debug["accepted_count"] == 2, debug

    selected, decisions, filtered, debug = map_utility_source_attributes(
        question="Give the current label and stocking points.",
        semantic_spec={
            "requested_attributes": ["current_label", "current_stocking_points"],
            "attribute_bindings": [
                {"attribute": "current_label", "evidence_slot_hint": "label_text_identifier"},
                {"attribute": "current_stocking_points", "evidence_slot_hint": "locations_where_items_are_stocked"},
            ],
        },
        evidence=[row(
            "summary",
            "Current state: shelf A, IP-7, and bin C.",
            {"location_access_details": "shelf A, IP-7, and bin C"},
        )],
        llm_client=MisbindingClient(),
        model_name="test",
    )
    assert not selected, selected
    assert debug["accepted_count"] == 0, debug

    selected, decisions, filtered, debug = map_utility_source_attributes(
        question="What is the current label?",
        semantic_spec={
            "requested_attributes": ["current_label"],
            "attribute_bindings": [
                {"attribute": "current_label", "support_span": "current label"},
            ],
        },
        evidence=[row("latest", "BB-6.", {"driver_label": "BB-6"})],
        llm_client=OmittedTerseClaimClient(),
        model_name="test",
    )
    assert {item.memory_id for item in selected} == {"latest"}, selected
    assert decisions[0]["typed_fields"] == [{
        "attribute": "current_label",
        "slot_name": "driver_label",
        "value": "BB-6",
    }], decisions
    assert debug["direct_typed_recovery_count"] == 1, debug
    print("utility_source_attribute_mapping_smoke=PASS")


if __name__ == "__main__":
    main()
