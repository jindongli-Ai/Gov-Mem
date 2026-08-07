"""Regression checks for source-grounded dynamic record decomposition."""

import ast

from gov_mem.data.schema import MemoryInstance
from gov_mem.memory.amem_memory import (
    AtomicMemory,
    AtomicMemoryExtractor,
    _ground_semantic_tags,
    _ground_record_annotations,
    _materialize_record_atom,
)
from gov_mem.legacy.graph_slot_renderer import render_graph_authorized_slots


def main() -> None:
    source = "Active records: alpha window at 09:00; beta window at 14:00."
    records = _ground_record_annotations([
        {
            "discourse_act": "update",
            "assertion_confidence": 0.9,
            "event_identity": {"entity_key": "alpha_window", "entity_surface_span": "alpha window"},
            "attributes": {"time": "09:00"},
            "surface_values": {"time": "09:00"},
            "claims": [{"property_label": "time", "value_span": "09:00", "claim_span": "alpha window at 09:00"}],
            "state_delta": {"operation": "set", "changed_fields": {"time": "09:00"}},
            "evidence_span": "alpha window at 09:00",
        },
        {
            "discourse_act": "update",
            "assertion_confidence": 0.9,
            "event_identity": {"entity_key": "beta_window", "entity_surface_span": "beta window"},
            "attributes": {"time": "14:00"},
            "surface_values": {"time": "14:00"},
            "claims": [{"property_label": "time", "value_span": "14:00", "claim_span": "beta window at 14:00"}],
            "state_delta": {"operation": "set", "changed_fields": {"time": "14:00"}},
            "evidence_span": "beta window at 14:00",
        },
    ], source_text=source)
    assert len(records) == 2, records
    candidate = AtomicMemory(
        memory_id="source", instance_id="smoke", owner_user="owner", memory_type="event",
        content=source, entities=[], slots={"time": "09:00", "secondary_time": "14:00"},
        source_message_ids=["m1"], timestamp="2026-01-01T00:00:00", lifecycle_status="active",
        access_tags={}, confidence=1.0,
    )
    atoms = [_materialize_record_atom(item=candidate, semantic_tags=record) for record in records]
    assert atoms[0].content == "alpha window at 09:00", atoms[0]
    assert atoms[0].slots == {"time": "09:00"}, atoms[0]
    assert atoms[1].content == "beta window at 14:00", atoms[1]
    assert atoms[1].slots == {"time": "14:00"}, atoms[1]

    class RecordAnnotator:
        def chat_json(self, **kwargs):
            payload = ast.literal_eval(kwargs["user_prompt"].split("Candidates: ", 1)[1])
            return {"annotations": [{
                "candidate_id": payload[0]["candidate_id"],
                "discourse_act": "update", "assertion_confidence": 0.9,
                "event_identity": {"entity_key": "record_bundle", "entity_surface_span": "appointment records"},
                "attributes": {"time": "09:00"}, "surface_values": {"time": "09:00"},
                "claims": [{"property_label": "time", "value_span": "09:00", "claim_span": "alpha appointment at 09:00"}],
                "state_delta": {"operation": "set", "changed_fields": {"time": "09:00"}},
                "evidence_span": "alpha appointment at 09:00",
                "record_annotations": [
                    {
                        "discourse_act": "update", "assertion_confidence": 0.9,
                        "event_identity": {"entity_key": "alpha_appointment", "entity_surface_span": "alpha appointment"},
                        "attributes": {"time": "09:00"}, "surface_values": {"time": "09:00"},
                        "claims": [{"property_label": "time", "value_span": "09:00", "claim_span": "alpha appointment at 09:00"}],
                        "state_delta": {"operation": "set", "changed_fields": {"time": "09:00"}},
                        "evidence_span": "alpha appointment at 09:00",
                    },
                    {
                        "discourse_act": "update", "assertion_confidence": 0.9,
                        "event_identity": {"entity_key": "beta_appointment", "entity_surface_span": "beta appointment"},
                        "attributes": {"time": "14:00"}, "surface_values": {"time": "14:00"},
                        "claims": [{"property_label": "time", "value_span": "14:00", "claim_span": "beta appointment at 14:00"}],
                        "state_delta": {"operation": "set", "changed_fields": {"time": "14:00"}},
                        "evidence_span": "beta appointment at 14:00",
                    },
                ],
            }]}

    instance = MemoryInstance(
        instance_id="record-decomposition", domain="generic", conversation_id=None,
        messages=[{"message_id": "m1", "speaker_id": "owner", "timestamp": "2026-01-01T00:00:00",
                   "text": "Upcoming appointment records: alpha appointment at 09:00; beta appointment at 14:00."}],
        question="List my appointment records.", asking_user_id="owner", choices=None, answer=None,
    )
    extracted = AtomicMemoryExtractor(llm_client=RecordAnnotator(), model_name="stub").extract(instance)
    decomposed = [item for item in extracted if "record::" in item.memory_id]
    assert len(decomposed) == 2, extracted
    assert {item.content for item in decomposed} == {"alpha appointment at 09:00", "beta appointment at 14:00"}, decomposed

    class OmissionRepairAnnotator:
        def chat_json(self, **kwargs):
            payload = ast.literal_eval(kwargs["user_prompt"].split("Candidates: ", 1)[1])
            if "Repair missing factual claim structures" not in kwargs["user_prompt"]:
                return {"annotations": []}
            return {"annotations": [{
                "candidate_id": payload[0]["candidate_id"],
                "discourse_act": "update", "assertion_confidence": 0.9,
                "event_identity": {"entity_key": "single_appointment", "entity_surface_span": "single appointment"},
                "attributes": {"time": "10:00"}, "surface_values": {"time": "10:00"},
                "claims": [{"property_label": "time", "value_span": "10:00", "claim_span": "single appointment at 10:00"}],
                "state_delta": {"operation": "set", "changed_fields": {"time": "10:00"}},
                "evidence_span": "single appointment at 10:00",
            }]}

    omitted_instance = MemoryInstance(
        instance_id="required-source-repair", domain="generic", conversation_id=None,
        messages=[{"message_id": "m-required", "speaker_id": "owner", "timestamp": "2026-01-01T00:00:00",
                   "text": "The single appointment at 10:00 remains active."}],
        question="What remains active?", asking_user_id="owner", choices=None, answer=None,
    )
    repaired = AtomicMemoryExtractor(llm_client=OmissionRepairAnnotator(), model_name="stub").extract(
        omitted_instance,
        annotation_source_message_ids={"m-required"},
        required_annotation_source_message_ids={"m-required"},
    )
    assert any(
        (item.access_tags.get("semantic_tags") or {}).get("evidence_span") == "single appointment at 10:00"
        for item in repaired
    ), repaired

    class RequiredRecordCompiler:
        def __init__(self):
            self.record_calls = 0

        def chat_json(self, **kwargs):
            if "Return {\"records\":" not in kwargs["user_prompt"]:
                return {"annotations": []}
            self.record_calls += 1
            payload = ast.literal_eval(kwargs["user_prompt"].split("Candidates: ", 1)[1])
            candidate_id = payload[0]["candidate_id"]
            records = [
                {
                    "candidate_id": candidate_id, "discourse_act": "update", "assertion_confidence": 0.9,
                    "event_identity": {"entity_key": "alpha_record", "entity_surface_span": "alpha record"},
                    "attributes": {"time": "09:00"}, "surface_values": {"time": "09:00"},
                    "claims": [{"property_label": "time", "value_span": "09:00", "claim_span": "alpha record at 09:00"}],
                    "state_delta": {"operation": "set", "changed_fields": {"time": "09:00"}},
                    "evidence_span": "alpha record at 09:00",
                },
            ]
            if self.record_calls > 1:
                records.append(
                {
                    "candidate_id": candidate_id, "discourse_act": "update", "assertion_confidence": 0.9,
                    "event_identity": {"entity_key": "beta_record", "entity_surface_span": "beta record"},
                    "attributes": {"time": "14:00"}, "surface_values": {"time": "14:00"},
                    "claims": [{"property_label": "time", "value_span": "14:00", "claim_span": "beta record at 14:00"}],
                    "state_delta": {"operation": "set", "changed_fields": {"time": "14:00"}},
                    "evidence_span": "beta record at 14:00",
                })
            return {"records": records}

    required_records_instance = MemoryInstance(
        instance_id="required-record-compiler", domain="generic", conversation_id=None,
        messages=[{"message_id": "m-records", "speaker_id": "owner", "timestamp": "2026-01-01T00:00:00",
                   "text": "alpha record at 09:00; beta record at 14:00."}],
        question="List the records.", asking_user_id="owner", choices=None, answer=None,
    )
    direct_compiled_tag = _ground_semantic_tags({
        "discourse_act": "update", "assertion_confidence": 0.9,
        "event_identity": {"entity_key": "alpha_record", "entity_surface_span": "alpha record"},
        "attributes": {"time": "09:00"}, "surface_values": {"time": "09:00"},
        "claims": [{"property_label": "time", "value_span": "09:00", "claim_span": "alpha record at 09:00"}],
        "state_delta": {"operation": "set", "changed_fields": {"time": "09:00"}},
        "evidence_span": "alpha record at 09:00",
    }, source_text="alpha record at 09:00; beta record at 14:00.")
    assert direct_compiled_tag["attributes"] == {"time": "09:00"}, direct_compiled_tag
    required_compiler = RequiredRecordCompiler()
    compiled_records = AtomicMemoryExtractor(llm_client=required_compiler, model_name="stub").extract(
        required_records_instance,
        annotation_source_message_ids={"m-records"},
        required_annotation_source_message_ids={"m-records"},
    )
    assert required_compiler.record_calls == 2, required_compiler.record_calls
    assert {item.content for item in compiled_records if "record::" in item.memory_id} == {
        "alpha record at 09:00", "beta record at 14:00",
    }, compiled_records
    rendered = render_graph_authorized_slots(
        certificate={"realizations": [
            {"attribute": "record_collection", "slot_name": "first_field", "value": "value-A", "source_atom_id": "record-a"},
            {"attribute": "record_collection", "slot_name": "second_field", "value": "value-B", "source_atom_id": "record-a"},
        ]},
        action="answer",
    )
    assert rendered.answer_text == "record collection: first field: value-A; second field: value-B.", rendered
    record_local_rendered = render_graph_authorized_slots(
        certificate={"realizations": [
            {"attribute": "future_schedule", "slot_name": "appointment_datetime", "value": "Tue Mar 17 1:00 PM", "source_atom_id": "march-old", "source_memory_id": "march-old", "timestamp": "2026-03-06T10:39", "source_text": "Tue Mar 17 1:00 PM infectious disease follow-up"},
            {"attribute": "future_schedule", "slot_name": "appointment_type", "value": "infectious disease follow-up", "source_atom_id": "march-old", "source_memory_id": "march-old", "timestamp": "2026-03-06T10:39", "source_text": "Tue Mar 17 1:00 PM infectious disease follow-up"},
            {"attribute": "future_schedule", "slot_name": "start_datetime", "value": "Tue Mar 17 1:00 PM", "source_atom_id": "march-new", "source_memory_id": "march-new", "timestamp": "2026-03-09T11:33", "source_text": "Tue Mar 17 1:00 PM infectious disease follow-up"},
            {"attribute": "future_schedule", "slot_name": "event_type", "value": "infectious disease follow-up", "source_atom_id": "march-new", "source_memory_id": "march-new", "timestamp": "2026-03-09T11:33", "source_text": "Tue Mar 17 1:00 PM infectious disease follow-up"},
            {"attribute": "future_schedule", "slot_name": "start_datetime", "value": "Mon Jun 8 8:30 AM", "source_atom_id": "june", "source_memory_id": "june", "timestamp": "2026-03-09T11:33", "source_text": "Mon Jun 8 8:30 AM repeat RPR"},
            {"attribute": "future_schedule", "slot_name": "event_type", "value": "repeat RPR", "source_atom_id": "june", "source_memory_id": "june", "timestamp": "2026-03-09T11:33", "source_text": "Mon Jun 8 8:30 AM repeat RPR"},
        ]}, action="answer",
    )
    assert record_local_rendered.answer_text == (
        "Tue Mar 17 1:00 PM infectious disease follow-up; "
        "Tue Mar 17 1:00 PM infectious disease follow-up; "
        "Mon Jun 8 8:30 AM repeat RPR."
    ), record_local_rendered
    print("atomic_record_decomposition_smoke=PASS")


if __name__ == "__main__":
    main()
