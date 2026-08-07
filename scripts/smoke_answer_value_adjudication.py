#!/usr/bin/env python3
"""Final answer-value adjudication must replace contextual phrases with values."""

from gov_mem.legacy.semantic_alignment import (
    _should_escalate_answer_value,
    align_requested_attributes,
)
from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.memory.governed_atom import GovernedMemoryAtom


def _atom(atom_id: str, text: str, slots: dict[str, str]) -> GovernedMemoryAtom:
    return GovernedMemoryAtom(
        atom_id=atom_id,
        atom_type="fact_atom",
        text=text,
        slots=slots,
        owner_id="owner",
        subject_id=atom_id,
        speaker_id="owner",
        source_turn=1,
        timestamp="2026-01-01T00:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["self"],
        related_entities=[],
        confidence=1.0,
        provenance={"source_message_ids": [atom_id], "evidence_span": text},
    )


def _slot_ids_by_value(graph):
    return {
        str((node.attributes or {}).get("slot_value") or ""): node.node_id
        for node in graph.nodes
        if node.node_type == "SlotNode"
    }


class FinalGroundingLLM:
    def __init__(self, slot_ids: dict[str, str]):
        self.slot_ids = slot_ids
        self.final_calls = 0

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        prompt = str(kwargs.get("user_prompt") or "")
        if "This is final answer-value grounding" in prompt:
            self.final_calls += 1
            rows = []
            if "'current_room'" in prompt:
                rows.append({
                    "attribute": "current_room",
                    "slot_node_ids": [self.slot_ids["LH-318"]],
                    "binding_kind": "scalar",
                    "query_support_span": "current room",
                    "fact_support_spans": [{
                        "slot_node_id": self.slot_ids["LH-318"],
                        "source_atom_id": "room_update",
                        "fact_support_span": "Current room LH-318",
                    }],
                    "reason": "concrete current value, not update instruction",
                    "rejected_neighbor_roles": ["meta_update_instruction"],
                })
            if "'current_overflow_note'" in prompt:
                rows.append({
                    "attribute": "current_overflow_note",
                    "slot_node_ids": [self.slot_ids["hall pantry lower wicker bin"]],
                    "binding_kind": "scalar",
                    "query_support_span": "current overflow note",
                    "fact_support_spans": [{
                        "slot_node_id": self.slot_ids["hall pantry lower wicker bin"],
                        "source_atom_id": "overflow_update",
                        "fact_support_span": "Current overflow remains hall pantry lower wicker bin only",
                    }],
                    "reason": "current remaining value, not deletion boundary",
                    "rejected_neighbor_roles": ["deletion_boundary"],
                })
            return {"bindings": rows}
        return {"bindings": [
            {
                "attribute": "current_room",
                "slot_node_ids": [self.slot_ids["should override older ones"]],
                "binding_kind": "scalar",
                "query_support_span": "current room",
                "fact_support_spans": [{
                    "slot_node_id": self.slot_ids["should override older ones"],
                    "source_atom_id": "room_update",
                    "fact_support_span": "should override older ones",
                }],
            },
            {
                "attribute": "current_overflow_note",
                "slot_node_ids": [self.slot_ids["Deleting the old Pantry overflow note"]],
                "binding_kind": "scalar",
                "query_support_span": "current overflow note",
                "fact_support_spans": [{
                    "slot_node_id": self.slot_ids["Deleting the old Pantry overflow note"],
                    "source_atom_id": "overflow_update",
                    "fact_support_span": "Deleting the old Pantry overflow note",
                }],
            },
        ]}


def main() -> None:
    assert _should_escalate_answer_value(
        attribute="broad_wording",
        semantic_spec={"request_shape": "fact"},
        binding={"anchor_slot_node_ids": ["slot::wording"]},
        candidates=[{
            "slot_node_id": "slot::wording",
            "slot_name": "safe_wording",
            "slot_value": "for broad summaries remains",
            "source_text": "the safe wording for broad summaries remains 'temporary residence reassignment'",
        }],
    )
    graph = GovernedGraphBuilder().build(
        graph_id="answer_value_adjudication",
        instance_id="smoke",
        atoms=[
            _atom(
                "room_update",
                "Current room LH-318; the source note says this should override older ones.",
                {"current_room": "should override older ones", "room_code": "LH-318"},
            ),
            _atom(
                "overflow_update",
                "Deleting the old Pantry overflow note. Current overflow remains hall pantry lower wicker bin only.",
                {
                    "current_overflow_note": "Deleting the old Pantry overflow note",
                    "overflow_location": "hall pantry lower wicker bin",
                },
            ),
        ],
    )
    llm = FinalGroundingLLM(_slot_ids_by_value(graph))
    result = align_requested_attributes(
        question="What are the current room and current overflow note?",
        semantic_spec={
            "requested_attributes": ["current_room", "current_overflow_note"],
            "temporal_scope": "current",
            "attribute_bindings": [
                {"attribute": "current_room", "support_span": "current room"},
                {"attribute": "current_overflow_note", "support_span": "current overflow note"},
            ],
        },
        graph=graph,
        owner_id="owner",
        utility_atom_ids=None,
        llm_client=llm,
        model_name="stub",
        semantic_contract_certifiable=True,
        allow_record_local_completion=True,
    )
    assert result["available"], result
    bindings = result["bindings"]
    assert bindings["current_room"]["slot_name"] == "room_code", result
    assert bindings["current_overflow_note"]["slot_name"] == "overflow_location", result
    trace = result["diagnostics"]["answer_value_adjudication"]
    assert trace["attempted"], result
    assert set(trace["updated_attributes"]) == {"current_room", "current_overflow_note"}, result
    assert llm.final_calls == 2, result
    print("answer_value_adjudication_smoke=PASS")


if __name__ == "__main__":
    main()
