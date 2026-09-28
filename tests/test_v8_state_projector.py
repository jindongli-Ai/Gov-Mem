from __future__ import annotations

from gov_mem.governance_runtime.v8_event_store import V8EventStore
from gov_mem.governance_runtime.v8_state_projector import project_v8_state


def _event(event_id, effect, turn_id):
    return {
        "event_id": event_id, "event_type": "permission", "effect": effect,
        "issuer_principal_id": "owner", "grantee_principal_id": "guest",
        "action": "receive", "resource_id": "visit_logistics",
        "resource_surface": "visit logistics", "included_scopes": ["visit time"],
        "excluded_scopes": [], "source_spans": [{"turn_id": turn_id, "span": effect}],
    }


def test_projection_excludes_cached_future_revocation():
    store = V8EventStore(conversation_id="episode")
    store.governance_events = [_event("e1", "allow", "t001"), _event("e2", "revoke", "t003")]
    early = project_v8_state(
        store, visible_turn_ids=["t001", "t002"], requester_principal_id="guest",
        question="What is the visit time?",
    )
    late = project_v8_state(
        store, visible_turn_ids=["t001", "t002", "t003"], requester_principal_id="guest",
        question="What is the visit time?",
    )
    assert early["current_permissions"][0]["decision"] == "allow"
    assert early["future_event_count_excluded"] == 1
    assert late["current_permissions"][0]["decision"] == "deny"


def test_projection_exposes_source_grounded_delete_tombstone():
    store = V8EventStore(conversation_id="episode")
    store.facts = [{
        "fact_id": "deleted_phone", "scene_id": None, "subject_id": "callback",
        "field_name": "record_status", "value": "cleared", "lifecycle": "delete",
        "source_spans": [{"turn_id": "t002", "span": "Number 415-555-0100 cleared from memory."}],
    }]
    state = project_v8_state(
        store, visible_turn_ids=["t001", "t002"], requester_principal_id="owner",
        question="What was the number?",
    )
    assert state["lifecycle_tombstones"][0]["lifecycle"] == "delete"


def test_same_scene_observations_merge_only_after_their_turn():
    from gov_mem.extraction.v8_event_schema import V8ExtractionBatch, V8Scene, V8SourceSpan, V8Relation
    store = V8EventStore(conversation_id="synthetic")
    for turn, participants in [("t1", ["owner"]), ("t2", ["assistant"])]:
        store.append(V8ExtractionBatch(scenes=[V8Scene(
            scene_id="scene", scene_type="activity", canonical_name="review",
            participant_principal_ids=participants, source_spans=[V8SourceSpan(turn, "review")])]), processed_turn_ids=[turn])
    store.append(V8ExtractionBatch(relations=[V8Relation(
        relation_id="r", relation_type="participates_in", source_id="operator", target_id="scene",
        source_spans=[V8SourceSpan("t3", "operator joins the review")])]), processed_turn_ids=["t3"])
    early = project_v8_state(store, visible_turn_ids=["t1"], requester_principal_id="owner", question="")
    late = project_v8_state(store, visible_turn_ids=["t1", "t2", "t3"], requester_principal_id="owner", question="")
    assert early["current_scenes"][0]["participant_principal_ids"] == ["owner"]
    assert late["current_scenes"][0]["participant_principal_ids"] == ["owner", "assistant", "operator"]
    assert len(late["relations"]) == 1
    assert len(store.scenes) == 2  # Projection must not mutate append-only storage.


def test_same_scene_same_field_distinct_subjects_do_not_overwrite():
    store = V8EventStore(conversation_id="synthetic", facts=[
        {"scene_id": "s", "subject_id": p, "field_name": "status", "value": v,
         "source_spans": [{"turn_id": "t1", "span": v}]}
        for p, v in [("p1", "ready"), ("p2", "waiting")]])
    state = project_v8_state(store, visible_turn_ids=["t1"], requester_principal_id="p1", question="")
    assert len(state["current_facts"]) == 2


def test_scopes_conditions_and_issuers_are_preserved():
    store = V8EventStore(conversation_id="synthetic")
    allow = _event("a", "allow", "t1")
    deny = {**_event("b", "deny", "t2"), "included_scopes": ["private note"], "condition": "after review"}
    other_issuer = {**_event("c", "allow", "t2"), "issuer_principal_id": "another_owner"}
    store.governance_events = [allow, deny, other_issuer]
    state = project_v8_state(store, visible_turn_ids=["t1", "t2"], requester_principal_id="guest", question="")
    assert len(state["requester_permissions"]) == 3
    assert state["requester_permissions"][1]["activation"] == "conditional"


def test_expired_permission_and_pending_revocation():
    store = V8EventStore(conversation_id="synthetic")
    store.governance_events = [
        {**_event("a", "allow", "t1"), "valid_until": "2026-01-01T10:00:00"},
        {**_event("b", "revoke", "t2"), "valid_from_turn_id": "t9"}]
    state = project_v8_state(store, visible_turn_ids=["t1", "t2"], requester_principal_id="guest", question="",
                             checkpoint_time="2026-01-01T11:00:00")
    assert len(state["requester_permissions"]) == 1
    assert state["requester_permissions"][0]["activation"] == "expired"
    assert len(state["requester_permission_events"]) == 2


def test_whole_resource_revocation_supersedes_scoped_allow():
    store = V8EventStore(conversation_id="synthetic", governance_events=[
        _event("a", "allow", "t1"), {**_event("b", "revoke", "t2"), "included_scopes": []}])
    state = project_v8_state(store, visible_turn_ids=["t1", "t2"], requester_principal_id="guest", question="")
    assert len(state["requester_permissions"]) == 1
    assert state["requester_permissions"][0]["decision"] == "deny"


def test_lifecycle_events_without_grantee_reach_query_context_without_future_leakage():
    from gov_mem.backbones.govmem_v8_late_governance import _graph_side_context
    from gov_mem.extraction.v8_line_protocol import bind_claim_events
    from gov_mem.data.schema import MemoryInstance
    store = V8EventStore(conversation_id="synthetic")
    store.governance_events = [
        {"event_id": "d1", "event_type": "lifecycle", "effect": "delete",
         "resource_id": "retired_note", "action": "delete",
         "source_spans": [{"turn_id": "t2", "span": "Delete the retired note."}]},
        {"event_id": "u1", "event_type": "lifecycle", "effect": "update",
         "resource_id": "current_note", "action": "use",
         "source_spans": [{"turn_id": "t3", "span": "A new note is available."}]},
    ]
    instance = MemoryInstance(instance_id="q", conversation_id="synthetic", domain="synthetic",
        messages=[], question="Old note?", asking_user_id="owner", choices=None, answer=None)
    for turns, expected in [(["t1"], []), (["t1", "t2"], ["d1"]), (["t1", "t2", "t3"], ["d1", "u1"])]:
        state = project_v8_state(store, visible_turn_ids=turns, requester_principal_id="owner", question="")
        graph = _graph_side_context(instance=instance, store=store, state_projection=state,
                                    static_relationships=[], visible_turn_ids=turns)
        assert [e["event_id"] for e in graph["lifecycle_events"]] == expected
        assert graph["lifecycle_tombstones"] == []
        if "d1" in expected:
            raw = {"claims": [{"permission_event_ids": ["d1"]}]}
            bind_claim_events(raw, graph)
            assert raw["claims"][0]["resource_id"] == "retired_note"


def test_future_start_remains_pending_even_if_expiry_is_present():
    store = V8EventStore(conversation_id="synthetic", governance_events=[
        {**_event("p", "revoke", "t1"), "valid_from_turn_id": "t9", "valid_until": "after review"},
        {**_event("d", "delete", "t1"), "event_type": "lifecycle", "valid_from_turn_id": "t9",
         "valid_until": "2026-01-01T09:00:00"}])
    state = project_v8_state(store, visible_turn_ids=["t1"], requester_principal_id="guest", question="",
                             checkpoint_time="2026-01-01T11:00:00")
    assert state["current_permissions"] == []
    assert state["lifecycle_state"][0]["activation"] == "pending_or_unresolved"
