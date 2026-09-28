from __future__ import annotations

from gov_mem.data.schema import MemoryInstance
from gov_mem.extraction.v8_event_extractor import extract_v8_events
from gov_mem.extraction.v8_principal_registry import build_v8_principal_registry


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _instance() -> MemoryInstance:
    messages = [
        {
            "turn_id": "t001", "message_id": "t001",
            "speaker_id": "patient_elena", "speaker_role": "patient",
            "text": "Linda is my mother.", "timestamp": "2026-01-01T09:00",
        },
        {
            "turn_id": "t002", "message_id": "t002",
            "speaker_id": "patient_elena", "speaker_role": "patient",
            "text": "She can receive appointment times and parking details only.",
            "timestamp": "2026-01-01T09:01",
        },
    ]
    return MemoryInstance(
        instance_id="medical_ckpt_1", conversation_id="medical_episode_1",
        domain="medical", messages=messages, question="When is the appointment?",
        asking_user_id="family_linda", choices=None, answer=None,
        metadata={
            "requester": {"principal_id": "family_linda", "role": "family_member"},
            "observable": {"as_of_turn_id": "t002"},
            "raw_sample": {"episode": {"entities": {
                "principals": [
                    {"principal_id": "patient_elena", "role": "patient", "display_name": "Elena"},
                    {"principal_id": "family_linda", "role": "family_member", "display_name": "Linda"},
                ],
                "relationships": [{
                    "type": "family_member", "family_id": "family_linda",
                    "patient_id": "patient_elena",
                    "access_level": "logistics_only_then_revoked",
                }],
            }}},
        },
    )


def test_registry_keeps_identity_relationship_but_drops_future_scope_summary():
    registry = build_v8_principal_registry(_instance())
    assert registry.contains("family_linda")
    assert registry.static_relationships == [{
        "type": "family_member",
        "family_id": "family_linda",
        "patient_id": "patient_elena",
    }]


def test_permission_can_use_cross_turn_sources_and_closed_principal_id():
    instance = _instance()
    registry = build_v8_principal_registry(instance)
    llm = FakeLLM({
        "entities": [], "facts": [], "relations": [],
        "governance_events": [{
            "event_type": "permission", "effect": "allow",
            "issuer_principal_id": "patient_elena",
            "grantee_principal_id": "family_linda", "action": "receive",
            "resource_id": "visit_logistics",
            "resource_surface": "appointment times and parking details",
            "included_scopes": ["appointment times", "parking details"],
            "excluded_scopes": [], "valid_from_turn_id": "t002",
            "source_spans": [
                {"turn_id": "t001", "span": "Linda is my mother"},
                {"turn_id": "t002", "span": "appointment times and parking details"},
            ],
            "confidence": 0.95,
        }],
    })
    batch = extract_v8_events(
        instance=instance, new_messages=[instance.messages[1]],
        context_messages=[instance.messages[0]], registry=registry,
        existing_scenes=[], existing_entities=[], llm_client=llm,
        model_name="gemini-2.5-flash-lite",
    )
    assert len(batch.governance_events) == 1
    event = batch.governance_events[0]
    assert event.grantee_principal_id == "family_linda"
    assert [span.turn_id for span in event.source_spans] == ["t001", "t002"]


def test_lifecycle_effect_without_grantee_is_not_rejected_as_missing_grantee():
    from gov_mem.extraction.v8_event_validator import validate_v8_extraction
    registry = build_v8_principal_registry(_instance())
    batch = validate_v8_extraction({"governance_events": [{
        "effect": "delete", "action": "delete", "resource_id": "old_note",
        "resource_surface": "parking details", "source_spans": [{"turn_id": "t002", "span": "parking details"}],
    }]}, turn_text={"t002": "She can receive appointment times and parking details only."}, registry=registry)
    assert len(batch.governance_events) == 1
    assert batch.governance_events[0].event_type == "lifecycle"


def test_scene_is_explicitly_extracted_and_grounded():
    instance = _instance()
    llm = FakeLLM({
        "scenes": [{
            "scene_id": "scene_visit", "scene_type": "appointment",
            "canonical_name": "appointment times", "participant_principal_ids": ["patient_elena"],
            "status": "active",
            "source_spans": [{"turn_id": "t002", "span": "appointment times"}],
            "confidence": 0.9,
        }],
        "entities": [], "facts": [], "relations": [], "governance_events": [],
    })
    batch = extract_v8_events(
        instance=instance, new_messages=[instance.messages[1]],
        context_messages=[instance.messages[0]],
        registry=build_v8_principal_registry(instance), existing_scenes=[],
        existing_entities=[], llm_client=llm, model_name="gemini-2.5-flash-lite",
    )
    assert len(batch.scenes) == 1
    assert batch.scenes[0].scene_id == "scene_visit"
    assert batch.audit["accepted_counts"]["scenes"] == 1


def test_completed_deletion_backstop_adds_tombstone_when_llm_omits_it():
    instance = _instance()
    deletion = {
        "turn_id": "t003", "message_id": "t003",
        "speaker_id": "patient_elena", "speaker_role": "patient",
        "text": "Temporary callback 415-555-0100 cleared from shared assistant memory.",
        "timestamp": "2026-01-01T09:02",
    }
    llm = FakeLLM({"entities": [], "facts": [], "relations": [], "governance_events": []})
    batch = extract_v8_events(
        instance=instance, new_messages=[deletion], context_messages=instance.messages,
        registry=build_v8_principal_registry(instance), existing_scenes=[],
        existing_entities=[], llm_client=llm, model_name="gemini-2.5-flash-lite",
    )
    assert len(batch.facts) == 1
    assert batch.facts[0].lifecycle == "delete"
    assert batch.facts[0].value == "Temporary callback 415-555-0100"
    assert "415-555-0100" in batch.facts[0].source_spans[0].span


def test_deletion_request_does_not_create_completed_tombstone():
    instance = _instance()
    request = {
        "turn_id": "t003", "message_id": "t003",
        "speaker_id": "patient_elena", "speaker_role": "patient",
        "text": "After Tuesday, please delete the temporary callback note from memory.",
        "timestamp": "2026-01-01T09:02",
    }
    llm = FakeLLM({"entities": [], "facts": [], "relations": [], "governance_events": []})
    batch = extract_v8_events(
        instance=instance, new_messages=[request], context_messages=instance.messages,
        registry=build_v8_principal_registry(instance), existing_scenes=[],
        existing_entities=[], llm_client=llm, model_name="gemini-2.5-flash-lite",
    )
    assert batch.facts == []


def test_direct_deletion_request_creates_immediate_tombstone():
    instance = _instance()
    request = {
        "turn_id": "t003", "message_id": "t003",
        "speaker_id": "patient_elena", "speaker_role": "patient",
        "text": "Deletion request: remove callback 415-555-0100 from shared memory.",
        "timestamp": "2026-01-01T09:02",
    }
    llm = FakeLLM({"scenes": [], "entities": [], "facts": [], "relations": [], "governance_events": []})
    batch = extract_v8_events(
        instance=instance, new_messages=[request], context_messages=instance.messages,
        registry=build_v8_principal_registry(instance), existing_scenes=[],
        existing_entities=[], llm_client=llm, model_name="gemini-2.5-flash-lite",
    )
    assert len(batch.facts) == 1
    assert batch.facts[0].lifecycle == "delete"
    assert batch.facts[0].value == "callback 415-555-0100"


def test_ontology_is_small_and_domain_independent():
    from gov_mem.extraction.v8_scene_schema import v8_scene_schema
    assert v8_scene_schema("unseen_domain") == v8_scene_schema("medical")
    schema = v8_scene_schema()
    assert len(schema["entity_types"]) <= 12
    assert "operational_duty" in schema["relation_types"]
    assert "domain" not in schema


def test_exact_source_grounding_rejects_changed_punctuation():
    from gov_mem.extraction.v8_event_validator import validate_v8_extraction
    from gov_mem.extraction.v8_principal_registry import V8PrincipalRegistry
    batch = validate_v8_extraction({"facts": [{"field_name": "balance", "value": "7",
          "source_spans": [{"turn_id": "t1", "span": "balance is 7"}]}]},
          turn_text={"t1": "balance is -7"}, registry=V8PrincipalRegistry({}, []))
    assert not batch.facts


def test_compressed_fact_restores_exact_source_including_qualifiers():
    from gov_mem.extraction.v8_event_validator import validate_v8_extraction
    instance = _instance()
    source = 'Use the backup channel only with explicit approval.'
    def check(value):
        return validate_v8_extraction(
            {'facts': [{'field_name': 'channel', 'value': value, 'lifecycle': 'update',
                        'source_spans': [{'turn_id': 't003', 'span': source}]}]},
            turn_text={'t003': source}, registry=build_v8_principal_registry(instance))
    batch = check('backup channel, explicit approval')
    assert batch.facts[0].value == source
    assert len(batch.audit['source_repairs']) == 1
    assert not check('primary channel').facts
    assert not check('approval backup channel').facts


def test_text_prefill_extracts_source_grounded_graph_without_json():
    class TextLLM:
        def chat_text(self, **kwargs):
            assert 'Return JSON only' not in kwargs['system_prompt']
            return ('EVENTS\nEVENT\tp1\tevent_type=permission\teffect=allow\taction=receive'
                    '\tgrantee_principal_id=family_linda\tresource_id=times\tresource_surface=appointment times\n'
                    'SOURCE\tp1\tturn_id=t002\tspan=appointment times and parking details\nACTION\tno_memory\nEND')
    instance = _instance()
    batch = extract_v8_events(instance=instance, new_messages=[instance.messages[1]],
        context_messages=[instance.messages[0]], registry=build_v8_principal_registry(instance),
        existing_scenes=[], existing_entities=[], llm_client=TextLLM(), model_name='test', response_protocol='lines')
    assert len(batch.governance_events) == 1
    assert batch.governance_events[0].resource_surface == 'appointment times'
    assert not batch.rejected
