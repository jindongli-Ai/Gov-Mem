"""Offline protocol recovery must preserve source grounding and denial semantics."""
import pytest

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.extraction.v8_line_protocol import bind_claim_events
from gov_mem.governance_runtime.v8_claim_reasoner import _validate_claims
from gov_mem.governance_runtime.v8_shallow_memory import normalize_shallow_json, expand_kept_candidates
from gov_mem.llm.json_parser import parse_json_response


def test_literal_controls_are_preserved_without_structural_repair():
    assert parse_json_response('{"value":"a\tb\nc"}') == {"value": "a\tb\nc"}
    for bad in ['{"value":"a\tb"broken"}', '{"value":"a\tb"', '{"value":"a\\q"}']:
        with pytest.raises(ValueError):
            parse_json_response(bad)


def normalize(claim):
    return normalize_shallow_json({"query_slots": [{"slot": "note"}], "claims": [
        {"slot": "note", "candidate_id": "candidate_0", "bind": [], **claim}]}, ingestion=False)


def validate(raw, sources=None):
    evidence = [RetrievedEvidence(memory_id="m", content="Private note: amber.", score=1,
        retrieval_source="dense", reason="test", source_message_ids=["t1"])]
    return _validate_claims(raw, evidence, requester_principal_id="guest", source_texts=sources)


def test_quote_only_release_still_requires_exact_source():
    raw = normalize({"quote": "amber", "delivery": "exact"})
    accepted, rejected = validate(raw)
    assert accepted[0]["value"] == "amber" and not rejected
    raw = normalize({"quote": "invented", "delivery": "exact"})
    accepted, rejected = validate(raw)
    assert not accepted and rejected[0]["reason"] == "claim_not_exactly_grounded"


def test_explicit_keep_false_omission_is_ignored_without_contract_failure():
    normalized = normalize({"keep": False})
    assert normalized["claims"] == []


def test_incomplete_claim_is_ignored_and_audited_without_release():
    normalized = normalize_shallow_json({
        "query_slots": [{"slot": "note"}],
        "claims": [
            {"candidate_id": "candidate_0", "value": "amber", "delivery": "exact"},
            {"slot": "note", "candidate_id": "candidate_0", "keep": True},
        ],
    }, ingestion=False)
    assert len(normalized["claims"]) == 1
    assert normalized["claims"][0]["slot"] == "note"
    assert normalized["contract_rejections"] == [
        {"claim_index": 0, "reason": "missing_slot_or_candidate_id"}
    ]


def test_candidate_only_denial_stays_blocked_and_requires_grounded_restriction():
    raw = normalize({"delivery": "block", "restriction_kind": "explicit_restriction",
        "restriction_evidence": [{"turn_id": "policy", "span": "Do not share the note."}]})
    expand_kept_candidates(raw, [{"candidate_id": "candidate_0", "text": "Private note: amber."}])
    accepted, rejected = validate(raw, {"policy": ["Do not share the note."]})
    assert not rejected and accepted[0]["delivery"] == "block"
    assert accepted[0]["value"] == "Private note: amber."
    with pytest.raises(ValueError, match="grounded"):
        validate(raw, {"policy": ["An unrelated policy."]})
    with pytest.raises(ValueError, match="value and delivery"):
        normalize({"delivery": "exact"})


def test_visible_source_binding_is_provenance_only_and_future_ids_fail():
    policy = {"event_id": "p1", "resource_id": "note", "action": "receive", "effect": "deny"}
    claim = {"permission_event_ids": ["t1"], "resource_id": "fabricated", "action": "receive"}
    bind_claim_events({"claims": [claim]}, {"policies": [policy]}, source_texts={"t1": ["Note."]})
    assert claim["permission_event_ids"] == []
    assert claim["governance_source_turn_ids"] == ["t1"]
    assert "resource_id" not in claim and "action" not in claim
    claim["permission_event_ids"] = ["future_turn"]
    with pytest.raises(ValueError, match="unknown event"):
        bind_claim_events({"claims": [claim]}, {}, source_texts={"t1": ["Note."]})


def test_multiple_restrictions_cannot_manufacture_canonical_binding():
    graph = {"policies": [{"event_id": "p1", "resource_id": "note", "effect": "deny"},
                          {"event_id": "p2", "resource_id": "address", "effect": "deny"}]}
    raw = normalize({"value": "amber", "delivery": "block", "bind": ["p1", "p2"],
        "restriction_kind": "explicit_restriction",
        "restriction_evidence": [{"turn_id": "policy", "span": "Do not share the note."}]})
    bind_claim_events(raw, graph)
    accepted, _ = validate(raw, {"policy": ["Do not share the note."]})
    assert accepted[0]["delivery"] == "block" and "resource_id" not in accepted[0]
    raw["claims"][0]["delivery"] = "exact"
    with pytest.raises(ValueError, match="incompatible"):
        bind_claim_events(raw, graph)


def test_grounded_denial_does_not_require_redundant_empty_binding():
    raw = {"query_slots": [{"slot": "note"}], "claims": [{"slot": "note", "candidate_id": "candidate_0",
        "delivery": "block", "restriction_kind": "explicit_restriction",
        "restriction_evidence": [{"turn_id": "policy", "span": "Do not share the note."}]}]}
    normalized = normalize_shallow_json(raw, ingestion=False)
    expand_kept_candidates(normalized, [{"candidate_id": "candidate_0", "text": "Private note: amber."}])
    accepted, rejected = validate(normalized, {"policy": ["Do not share the note."]})
    assert not rejected and accepted[0]["delivery"] == "block"
    assert accepted[0]["permission_event_ids"] == []
    with pytest.raises(ValueError, match="grounded"):
        validate(normalized, {})
    normalized_keep = normalize_shallow_json({"query_slots": [], "claims": [{"slot": "note", "candidate_id": "candidate_0",
        "keep": True}]}, ingestion=False)
    assert normalized_keep["claims"][0]["permission_event_ids"] == []


def test_source_candidate_ids_are_exact_and_do_not_map_arbitrary_numbers():
    from gov_mem.governance_runtime.v8_claim_reasoner import _candidate_payload
    ev = [RetrievedEvidence(memory_id="chunk", content="Private note: amber.", score=1,
        retrieval_source="dense", reason="test", source_message_ids=["t158"])]
    payload = _candidate_payload(ev, 0, "source")
    assert payload[0]["candidate_id"] == "t158"
    raw = {"query_slots": [], "claims": [{"slot": "note", "candidate_id": "t158", "value": "amber", "delivery": "exact"}]}
    claims, rejected = _validate_claims(raw, ev, requester_principal_id="owner", candidate_ids=["t158"])
    assert not rejected and claims[0]["candidate_id"] == "t158"
    for invalid in ["candidate_158", "t999", "158"]:
        with pytest.raises(ValueError, match="unknown candidate"):
            expand_kept_candidates({"claims": [{"candidate_id": invalid, "_keep_candidate": True}]}, payload)
    duplicate = _candidate_payload(ev * 2, 0, "source")
    assert [c["candidate_id"] for c in duplicate] == ["candidate_0", "candidate_1"]


def test_only_exact_already_supplied_source_can_become_candidate():
    from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
    from types import SimpleNamespace
    from copy import deepcopy
    evidence = [RetrievedEvidence(memory_id="m", content="The public visit is on Friday.",
        source_message_ids=["t1"], score=1, retrieval_source="dense", reason="test")]
    row = SimpleNamespace(question="Visit details?", metadata={}, asking_user_id="owner")
    graph = {"lifecycle_events": [{"event_id": "t2_update", "resource_id": "visit", "effect": "update",
        "source_spans": [{"turn_id": "t2", "span": "Bring both signed forms; no deposit is due."}]}]}
    raw = {"query_slots": [{"slot": "forms"}], "answer_action": "answer", "claims": [
        {"slot": "forms", "candidate_id": "t2", "keep": True, "bind": []}]}
    def run():
        return reason_v8_claims(instance=row, evidence=evidence, state_projection={}, graph_context=graph,
            llm_client=SimpleNamespace(chat_json=lambda **kw: deepcopy(raw)), model_name="offline",
            memory_mode="shallow", candidate_id_style="source")
    ledger = run()
    assert ledger["claims"][0]["value"] == "Bring both signed forms; no deposit is due."
    assert ledger["claims"][0]["source_message_ids"] == ["t2"]
    assert len(evidence) == 1  # No mutation/extra retrieval of bank records.
    raw["claims"][0]["candidate_id"] = "future_t3"
    with pytest.raises(ValueError, match="unknown candidate"):
        run()
    raw["claims"][0] = {"slot": "forms", "candidate_id": "t2", "value": "An unseen deposit number",
                        "delivery": "exact", "bind": []}
    with pytest.raises(ValueError, match="grounding"):
        run()


def test_disjoint_prompt_fragments_cannot_be_joined_into_a_fake_candidate():
    from gov_mem.governance_runtime.v8_claim_reasoner import _include_visible_source_candidates
    candidates, evidence = _include_visible_source_candidates([], [], {
        "t1": ["The old code", "was deleted."],
        "t2": ["Bring both forms.", "both forms"],
    })
    assert [c["candidate_id"] for c in candidates] == ["t2"]
    assert evidence[0].content == "Bring both forms."
