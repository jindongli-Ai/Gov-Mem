from gov_mem.governance_runtime.v8_prompt_sources import collect_prompt_sources
from gov_mem.extraction.v8_event_validator import validate_v8_extraction
from gov_mem.extraction.v8_principal_registry import V8PrincipalRegistry


def test_sources_are_exactly_exposed_and_mixed_chunks_have_no_false_attribution():
    candidates = [
        {"source_message_ids": ["old"], "text": "Visible older excerpt."},
        {"source_message_ids": ["left", "right"], "text": "Mixed chunk without offsets."}]
    graph = {"events": [{"source_spans": [{"turn_id": "g", "span": "Excerpt one."},
                                         {"turn_id": "g", "span": "Excerpt two."}]}]}
    ingestion = {"new_turns": [{"turn_id": "new", "text": "New information."}]}
    sources = collect_prompt_sources(candidates, graph, ingestion)
    assert sources == {"old": ["Visible older excerpt."], "g": ["Excerpt one.", "Excerpt two."],
                       "new": ["New information."]}
    # Never join separated quotations into a fake contiguous source.
    batch = validate_v8_extraction({"facts": [{"field_name": "status", "value": "one. Excerpt",
        "lifecycle": "update", "source_spans": [{"turn_id": "g", "span": "one. Excerpt"}]}]},
        turn_text=sources, registry=V8PrincipalRegistry({}, []))
    assert not batch.facts


def test_old_retrieved_lifecycle_is_context_not_a_new_graph_event():
    sources = collect_prompt_sources(
        [{"source_message_ids": ["old"], "text": "The record was canceled."}], {},
        {"new_turns": [{"turn_id": "new", "text": "We discussed another matter."}]})
    fact = {"field_name": "record", "value": "canceled", "lifecycle": "cancel",
            "source_spans": [{"turn_id": "old", "span": "The record was canceled."}]}
    batch = validate_v8_extraction({"facts": [fact]}, turn_text=sources,
        registry=V8PrincipalRegistry({}, []), allowed_output_turn_ids={"new"})
    assert not batch.facts
    assert batch.rejected[0]["reason"] == "context_only_record"
    fact["source_spans"].append({"turn_id": "new", "span": "Invented new cancellation."})
    batch = validate_v8_extraction({"facts": [fact]}, turn_text=sources,
        registry=V8PrincipalRegistry({}, []), allowed_output_turn_ids={"new"})
    assert not batch.facts
    assert batch.rejected[0]["reason"] == "new_source_not_grounded"


def test_new_lifecycle_can_cite_old_retrieved_source_and_new_change():
    sources = collect_prompt_sources(
        [{"source_message_ids": ["old"], "text": "The callback is 123."}], {},
        {"new_turns": [{"turn_id": "new", "text": "Delete that callback from memory."}]})
    batch = validate_v8_extraction({"facts": [{
        "field_name": "callback", "value": "123", "lifecycle": "delete",
        "source_spans": [{"turn_id": "old", "span": "The callback is 123."},
                         {"turn_id": "new", "span": "Delete that callback from memory."}]}]},
        turn_text=sources, registry=V8PrincipalRegistry({}, []), allowed_output_turn_ids={"new"})
    assert batch.facts[0].value == "123"
    assert {s.turn_id for s in batch.facts[0].source_spans} == {"old", "new"}


def test_lifecycle_uses_cited_words_instead_of_generated_boolean_or_deleted_value():
    registry = V8PrincipalRegistry({}, [])
    cases = [
        ("false", "update", "The device is not locked."),
        ("retired confidential wording", "delete", "Delete the retired note from memory."),
    ]
    for value, lifecycle, source in cases:
        batch = validate_v8_extraction({"facts": [{"field_name": "state", "value": value,
            "lifecycle": lifecycle, "source_spans": [{"turn_id": "t1", "span": source}]}]},
            turn_text={"t1": source}, registry=registry, allowed_output_turn_ids={"t1"})
        assert batch.facts[0].value == source
        assert batch.audit["source_repairs"][0]["reason"] == "lifecycle_stored_as_exact_source"
    # An unseen quote is never repaired by guessing.
    batch = validate_v8_extraction({"facts": [{"field_name": "state", "value": "false",
        "lifecycle": "update", "source_spans": [{"turn_id": "future", "span": "not locked"}]}]},
        turn_text={"t1": "locked"}, registry=registry)
    assert not batch.facts
