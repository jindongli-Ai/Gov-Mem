"""Synthetic contracts for language scope decisions and narrow symbolic checks.

The earlier V8 lexical/scene/sibling-veto expectations are intentionally
replaced: those policies were removed by the user's in-place V8 redesign.
"""
import pytest
from gov_mem.governance_runtime.v8_claim_reasoner import _validate_claims
from gov_mem.governance_runtime.v8_safe_evidence import build_v8_safe_evidence
from gov_mem.governance_runtime.v8_symbolic_critic import criticize_v8_claims
from gov_mem.data.schema import RetrievedEvidence


def claim(**kwargs):
    return {"claim_id": "c", "slot": "meeting room", "value": "R-7", "quote": "meeting room R-7",
            "delivery": "exact", "source_message_ids": ["t1"], "source_memory_id": "m1",
            "requested": True, "temporal_state": "current", **kwargs}


def review(claims, permissions=None, **state):
    return criticize_v8_claims({"claims": claims},
        state_projection={"current_permissions": permissions or [], **state},
        requester_principal_id="visitor", graph_context={"scene_hints": [
            {"canonical_name": "meeting", "participant_principal_ids": ["operator"]}]})


def deny(**kwargs):
    return {"event_id": "p1", "grantee_principal_id": "visitor", "decision": "deny",
            "activation": "active", "resource_id": "resource_a", "action": "receive", "scene_id": "scene_a",
            "source_spans": [{"turn_id": "t2", "span": "Do not disclose this record to the visitor."}],
            "included_scopes": [], "excluded_scopes": [], **kwargs}


def test_missing_permission_or_scene_member_does_not_veto():
    for basis in ("none", "explicit_permission", "operational_duty"):
        item = claim(authorization_basis=basis, sensitivity="location")
        assert review([item])["claims"][0]["delivery"] == "exact"


def test_lexical_overlap_is_not_resource_scope_matching():
    item = claim(slot="room callback policy", value="enabled")
    assert review([item], [deny(resource_surface="room callback number")])["claims"][0]["delivery"] == "exact"


def test_bound_unconditional_deny_is_enforced_after_language_matching():
    item = claim(resource_id="resource_a", action="receive", scene_id="scene_a", permission_event_ids=["p1"])
    reviewed = review([item], [deny()])
    assert reviewed["claims"][0]["delivery"] == "block"
    assert reviewed["symbolic_critic"]["veto_count"] == 1


@pytest.mark.parametrize("change", [
    {"activation": "expired"}, {"condition": "only after the review"},
    {"included_scopes": ["private notes"]}, {"excluded_scopes": ["room"]},
    {"grantee_principal_id": "someone_else"}, {"action": "share"}, {"scene_id": "other"},
])
def test_ambiguous_scope_and_different_action_are_not_hard_vetoes(change):
    item = claim(resource_id="resource_a", action="receive", scene_id="scene_a", permission_event_ids=["p1"])
    assert review([item], [deny(**change)])["claims"][0]["delivery"] == "exact"


def test_subject_and_record_neighbors_do_not_propagate_veto():
    first = claim(resource_id="resource_a", action="receive", scene_id="scene_a", permission_event_ids=["p1"], subject_principal_id="owner")
    second = claim(claim_id="other", slot="public address", value="8 East Street", subject_principal_id="owner")
    assert [c["delivery"] for c in review([first, second], [deny()])["claims"]] == ["block", "exact"]


def test_shared_value_does_not_mean_same_deleted_resource():
    item = claim(value="12", slot="room", resource_id="room")
    assert review([item], lifecycle_tombstones=[{"value": "12", "subject_id": "retired_code"}])["claims"][0]["delivery"] == "exact"


def test_scoped_allow_does_not_relabel_complete_release_as_summary():
    reviewed = review([claim()], [deny(decision="allow", excluded_scopes=["access code"])])
    assert reviewed["claims"][0]["delivery"] == "exact"


def evidence():
    return [RetrievedEvidence(memory_id="m", content="Room R-7; private token X-908.", score=1,
                              retrieval_source="dense", reason="test", source_message_ids=["t1"])]


def test_requester_authored_sensitive_value_is_not_automatically_authorized():
    raw = {"query_slots": [{"slot": "token"}], "claims": [claim(slot="token", value="X-908",
        quote="private token X-908", candidate_id="candidate_0", delivery="block",
        authorization_basis="none", restriction_kind="outside_scope",
        restriction_evidence=[{"turn_id": "t2", "span": "Only rooms may be disclosed."}])]}
    accepted, rejected = _validate_claims(raw, evidence(), requester_principal_id="visitor",
         source_texts={"t2": ["Only rooms may be disclosed."]})
    assert not rejected
    assert accepted[0]["delivery"] == "block"


def test_block_requires_actual_visible_restriction():
    raw = {"claims": [claim(candidate_id="candidate_0", quote="Room R-7", delivery="block")]}
    with pytest.raises(ValueError, match="grounded"):
        _validate_claims(raw, evidence(), requester_principal_id="visitor")


def test_claim_cannot_normalize_away_negative_sign_or_invent_quote():
    raw = {"claims": [claim(candidate_id="candidate_0", quote="Room R 7", value="R 7")]}
    accepted, rejected = _validate_claims(raw, evidence(), requester_principal_id="visitor")
    assert not accepted and rejected[0]["reason"] == "claim_not_exactly_grounded"


def test_safe_view_omits_denied_value_even_in_free_text_reason_and_slot():
    ledger = {"query_slots": [{"slot": "room"}, {"slot": "token X-908"}], "claims": [
        claim(slot="room"), claim(claim_id="blocked", slot="token X-908", value="X-908", delivery="block",
        reason="The token is X-908; do not show it", quote="private token X-908")]}
    safe, audit = build_v8_safe_evidence(ledger)
    assert "X-908" not in str(audit)
    assert "X-908" not in str([row.__dict__ for row in safe])
    assert audit["blocked_requested_count"] == 1


def test_background_does_not_count_as_partial_answer():
    safe, audit = build_v8_safe_evidence({"claims": [
        claim(delivery="block"), claim(claim_id="context", requested=False, slot="background")]})
    assert safe == [] and audit["released_requested_count"] == 0


def test_overlapping_protected_source_span_is_masked_but_other_occurrence_survives():
    text = "Room 42, token 42"
    denied = claim(claim_id="denied", slot="token", value="42", delivery="block", value_start=15, value_end=17)
    wide = claim(claim_id="excerpt", slot="overview", value=text, value_start=0, value_end=len(text))
    room = claim(claim_id="room", value="42", value_start=5, value_end=7)
    safe, _ = build_v8_safe_evidence({"claims": [wide, room, denied]})
    assert safe[0].content == "overview: Room 42, token [withheld]"
    assert safe[1].content == "meeting room: 42"


def test_duplicate_values_require_an_unambiguous_source_binding():
    source = RetrievedEvidence(memory_id="m", content="Room 42, token 42", score=1,
                               retrieval_source="dense", reason="test", source_message_ids=["t1"])
    raw = {"claims": [claim(candidate_id="candidate_0", slot="token", value="42", quote=source.content)]}
    accepted, rejected = _validate_claims(raw, [source], requester_principal_id="visitor")
    assert not accepted and rejected[0]["reason"] == "ambiguous_source_span"
    raw["claims"][0]["value_start"] = 15
    accepted, rejected = _validate_claims(raw, [source], requester_principal_id="visitor")
    assert not rejected and accepted[0]["value_start"] == 15


def test_wrong_candidate_index_is_rebound_only_to_unique_exact_visible_quote():
    raw = {'query_slots':[{'slot':'room'}], 'claims':[claim(candidate_id='candidate_91',slot='room',quote='Room R-7')]}
    claims, rejected = _validate_claims(raw, evidence(), requester_principal_id='visitor')
    assert not rejected
    assert claims[0]['source_binding_repaired'] is True
    assert claims[0]['candidate_id'] == 'candidate_0'
    raw['claims'][0]['quote'] = 'An unseen sentence'
    claims, rejected = _validate_claims(raw, evidence(), requester_principal_id='visitor')
    assert not claims and rejected


def test_blocked_paraphrase_uses_exact_protected_quote_without_releasing_it():
    raw={'query_slots':[{'slot':'token'}], 'claims':[claim(slot='token',candidate_id='candidate_0',
         value='The secret token is X-908',quote='private token X-908',delivery='block',
         restriction_kind='explicit_restriction',restriction_evidence=[{'turn_id':'t2','span':'Do not disclose tokens.'}])]}
    accepted,rejected=_validate_claims(raw,evidence(),requester_principal_id='visitor',
                                     source_texts={'t2':['Do not disclose tokens.']})
    assert not rejected and accepted[0]['value']=='private token X-908'
    safe,audit=build_v8_safe_evidence({'claims':accepted})
    assert safe==[] and 'X-908' not in str(audit)


def test_compact_claim_value_is_its_exact_quote():
    raw = {'query_slots': [{'slot': 'room'}], 'claims': [
        {'slot': 'room', 'candidate_id': 'candidate_0', 'value': 'R-7', 'delivery': 'exact'}]}
    source = RetrievedEvidence(memory_id='m', content='Use room R-7.', source_message_ids=['t1'], score=1, retrieval_source='dense', reason='test')
    accepted, rejected = _validate_claims(raw, [source], requester_principal_id='visitor')
    assert not rejected
    assert accepted[0]['quote'] == accepted[0]['value'] == 'R-7'


def test_ungrounded_release_is_omitted_but_malformed_denial_is_fatal():
    from types import SimpleNamespace
    from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
    instance = SimpleNamespace(question='Which room?', metadata={}, asking_user_id='visitor')
    source = RetrievedEvidence(memory_id='m', content='Use room R-7.', source_message_ids=['t1'], score=1, retrieval_source='dense', reason='test')
    raw = {'query_slots': [{'slot': 'room'}], 'answer_action': 'answer', 'claims': [
        {'slot': 'room', 'candidate_id': 'candidate_0', 'value': 'R-7', 'delivery': 'exact'},
        {'slot': 'room', 'candidate_id': 'candidate_0', 'value': 'invented', 'delivery': 'exact'}]}
    llm = SimpleNamespace(chat_json=lambda **kwargs: raw)
    def run():
        return reason_v8_claims(instance=instance, evidence=[source], state_projection={}, graph_context={},
                               llm_client=llm, model_name='gemini-2.5-flash-lite')
    result = run()
    assert len(result['claims']) == 1
    assert result['audit']['omitted_ungrounded_release_count'] == 1
    raw['claims'][1]['delivery'] = 'block'
    with pytest.raises(ValueError, match='grounding failed'):
        run()


@pytest.mark.parametrize("allow_scene, expected", [("scene_b", "block"), ("scene_a", "exact")])
def test_allow_in_other_scene_does_not_disable_explicit_bound_deny(allow_scene, expected):
    item = claim(resource_id="resource_a", action="receive", scene_id="scene_a",
                 permission_event_ids=["p1"])
    permissions = [deny(), deny(event_id="p2", decision="allow", scene_id=allow_scene)]
    assert review([item], permissions)["claims"][0]["delivery"] == expected


def test_identical_denied_values_keep_both_source_spans_for_late_redaction():
    text = "Room R-7; private token X-908."
    evidence_rows = [
        RetrievedEvidence(memory_id=f"m{i}", content=text, score=1, retrieval_source="dense",
                          reason="test", source_message_ids=[f"t{i}"])
        for i in range(2)]
    blocked = [{
        "slot": "token", "candidate_id": f"candidate_{i}", "value": "X-908",
        "delivery": "block", "restriction_kind": "explicit_restriction",
        "restriction_evidence": [{"turn_id": "policy", "span": "Do not share the token."}]
    } for i in range(2)]
    raw = {"query_slots": [{"slot": "token"}, {"slot": "room"}],
           "claims": blocked + [{"slot": "room", "candidate_id": "candidate_1",
                                "value": text, "delivery": "exact"}]}
    claims, rejected = _validate_claims(raw, evidence_rows, requester_principal_id="visitor",
        source_texts={"policy": ["Do not share the token."]})
    assert not rejected
    assert len([c for c in claims if c["delivery"] == "block"]) == 2
    safe, audit = build_v8_safe_evidence({"query_slots": raw["query_slots"], "claims": claims})
    assert safe and "R-7" in safe[0].content
    assert all("X-908" not in row.content for row in safe)


def test_invalid_ambiguous_duplicate_cannot_suppress_later_grounded_claim():
    rows = [RetrievedEvidence(memory_id="m", content="7 then 7", score=1,
        retrieval_source="dense", reason="test", source_message_ids=["t"])]
    base = {"slot": "number", "candidate_id": "candidate_0", "value": "7", "delivery": "exact"}
    raw = {"query_slots": [{"slot": "number"}], "claims": [base, {**base, "value_start": 7}]}
    claims, rejected = _validate_claims(raw, rows, requester_principal_id="owner")
    assert len(claims) == 1 and claims[0]["value_start"] == 7
    assert rejected[0]["reason"] == "ambiguous_source_span"


def test_grounded_deleted_classification_is_symbolically_blocked_without_repair():
    raw = {"query_slots": [{"slot": "token"}], "claims": [claim(
        slot="token", candidate_id="candidate_0", value="X-908", quote="private token X-908",
        temporal_state="deleted", restriction_kind="deleted",
        restriction_evidence=[{"turn_id": "t2", "span": "Delete the private token."}])]}
    claims, rejected = _validate_claims(raw, evidence(), requester_principal_id="visitor",
        source_texts={"t2": ["Delete the private token."]})
    assert not rejected and claims[0]["delivery"] == "exact"
    reviewed = review(claims)
    assert reviewed["claims"][0]["delivery"] == "block"
    safe, _ = build_v8_safe_evidence(reviewed)
    assert not safe
    missing, rejected = _validate_claims(raw, evidence(), requester_principal_id="visitor", source_texts={})
    assert missing == [] and rejected[0]["reason"] == "deleted_release_not_grounded"


def test_malformed_deleted_release_with_grounded_span_is_dropped():
    raw = {"query_slots": [{"slot": "token"}], "claims": [claim(
        slot="token", candidate_id="candidate_0", value="X-908", quote="private token X-908",
        temporal_state="deleted", restriction_evidence=[{"turn_id": "t2", "span": "Delete the private token."}]) ]}
    claims, rejected = _validate_claims(raw, evidence(), requester_principal_id="visitor",
        source_texts={"t2": ["Delete the private token."]})
    assert claims == [] and rejected[0]["reason"] == "deleted_release_not_grounded"


@pytest.mark.parametrize("change", [
    {"effect": "update"}, {"effect": "cancel"}, {"effect": "supersede"},
    {"activation": "pending_or_unresolved"}, {"activation": "expired"},
    {"condition": "if the owner confirms"}, {"included_scopes": ["only the old code"]},
    {"excluded_scopes": ["the public code"]}, {"grantee_principal_id": "other"},
    {"resource_id": "another_record"}, {"scene_id": "another_context"},
])
def test_nonapplicable_lifecycle_event_does_not_create_a_deletion(change):
    event = {"event_id": "d1", "event_type": "lifecycle", "effect": "delete", "activation": "active",
        "resource_id": "record", "action": "delete", "source_spans": [{"turn_id": "t2", "span": "Delete the note."}], **change}
    item = claim(resource_id="record", action="delete", permission_event_ids=["d1"])
    assert review([item], lifecycle_state=[event])["claims"][0]["delivery"] == "exact"
    event.update(effect="delete", activation="active")
    item["permission_event_ids"] = []
    assert review([item], lifecycle_state=[event])["claims"][0]["delivery"] == "exact"
