import json
from pathlib import Path

import pytest

from gov_mem.backbones.govmem_v8_late_governance import GovMemV8LateGovernanceBackbone
from gov_mem.data.schema import MemoryInstance
from gov_mem.extraction.v8_line_protocol import parse_lines
from gov_mem.extraction.v8_event_extractor import extract_v8_events
from gov_mem.extraction.v8_principal_registry import build_v8_principal_registry
from gov_mem.governance_runtime.v8_event_store import V8EventStore
from gov_mem.governance_runtime.v8_shallow_memory import shallow_graph_context, expand_kept_candidates
from gov_mem.utils.config import load_yaml_config


def instance():
    return MemoryInstance(instance_id="q1", conversation_id="episode", domain="synthetic",
        messages=[
            {"turn_id": "t1", "speaker_id": "owner", "text": "Do not disclose the private note to guest.", "timestamp": "2026-01-01T09:00"},
            {"turn_id": "t2", "speaker_id": "owner", "text": "The visit is July 9. Bring both signed forms; no deposit is due.", "timestamp": "2026-01-01T09:01"},
            {"turn_id": "t3", "speaker_id": "owner", "text": "The private note says amber.", "timestamp": "2026-01-01T09:02"}],
        question="Visit details?", asking_user_id="guest", choices=None, answer=None,
        metadata={"requester": {"principal_id": "guest"},
                  "raw_sample": {"episode": {"entities": {"principals": [
                      {"principal_id": "owner", "role": "owner"},
                      {"principal_id": "guest", "role": "guest"}], "relationships": []}}}})


class Embeddings:
    def embed_texts(self, *, model, texts):
        return [[1.0, 0.0] for _ in texts]


def config():
    return {"llm": {"base_model": "gemini-2.5-flash-lite"}, "embedding": {"model": "fake"},
            "v8_memory": {"mode": "shallow"}, "v8_query": {"response_protocol": "lines"}}


class LedgerLLM:
    def __init__(self):
        self.requests = []
        self.answer_calls = 0
    def chat_text(self, **kwargs):
        assert kwargs["model"] == "gemini-2.5-flash-lite"
        payload = json.loads(kwargs["user_prompt"])
        self.requests.append(payload)
        graph = payload["graph_context"]
        assert graph["mode"] == "shallow"
        assert not {"scene_hints", "entities", "relations", "current_fact_hints", "ontology"} & graph.keys()
        deny = payload["question"] == "Private note?"
        record = next(c for c in payload["rag_candidates"] if ("amber" if deny else "July 9") in c["text"])
        output = "SLOT\ts1\tslot=requested information\n"
        # Intentionally try to release a bound prohibited value. Symbolic must
        # veto without relying on the model choosing delivery=block.
        if deny:
            assert graph["permission_events"][0]["event_id"] == "p1"
            output += f"CLAIM\tc1\tslot=requested information\tcandidate_id={record['candidate_id']}\tvalue=amber\tdelivery=exact\tbind=p1\n"
        else:
            output += f"KEEP\tk1\tslot=requested information\tcandidate_id={record['candidate_id']}\tbind=NONE\n"
        if "ingestion" in payload:
            assert "domain_schema" not in payload["ingestion"]
            output += ("EVENTS\nEVENT\tp1\tevent_type=permission\teffect=deny\tissuer_principal_id=owner"
                       "\tgrantee_principal_id=guest\taction=receive\tresource_id=private_note\tresource_surface=private note\n"
                       "SOURCE\tp1\tturn_id=t1\tspan=Do not disclose the private note to guest.\n")
        return output + "ACTION\tanswer\nEND"
    def chat_json(self, **kwargs):
        if "Stage-2 governance reasoner" in kwargs["system_prompt"]:
            payload = json.loads(kwargs["user_prompt"])
            raw = parse_lines(self.chat_text(**kwargs), ingestion="ingestion" in payload, shallow=True)
            if "events" in raw:
                raw["events"] = raw["events"]["governance_events"]
            for claim in raw["claims"]:
                claim["bind"] = claim.pop("permission_event_ids")
                if claim.pop("_keep_candidate", False):
                    claim["keep"] = True
            return raw
        self.answer_calls += 1
        payload = json.loads(kwargs["user_prompt"])
        assert "amber" not in kwargs["user_prompt"]
        text = payload["allowed_claims"][0]["value"]
        assert "both signed forms; no deposit is due." in text
        return {"answer": text, "used_claim_ids": []}


@pytest.mark.parametrize("protocol", ["lines", "json"])
@pytest.mark.parametrize("candidate_style", ["indexed", "source"])
def test_shallow_default_config_and_two_checkpoint_neurosymbolic_flow(tmp_path, protocol, candidate_style):
    root = Path(__file__).resolve().parents[1]
    assert load_yaml_config(root / "configs/govmem_v8_late_governance_gemini25flashlite.yaml")["v8_memory"]["mode"] == "shallow"
    llm = LedgerLLM()
    cfg = config()
    cfg["v8_query"]["response_protocol"] = protocol
    cfg["v8_query"]["candidate_id_style"] = candidate_style
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=Embeddings(),
        config=cfg, output_dir=tmp_path, dataset_name="synthetic")
    row = instance()
    result = backbone.run_instance(row)
    assert result.answer_result.action == "answer"
    assert "no deposit is due" in result.answer_result.answer_text
    assert result.debug_payload["v8_cost"]["logical_chat_calls"] == 2
    store = backbone._stores["episode"]
    assert not store.scenes and not store.entities and not store.facts and not store.relations
    assert len(store.governance_events) == 1
    # Same prefix, different question: cached one-hop policy is enforced.
    row.instance_id = "q2"
    row.question = "Private note?"
    result = backbone.run_instance(row)
    assert "ingestion" not in llm.requests[-1]
    assert result.answer_result.action == "refuse"
    assert result.answer_result.raw_response["v8_claim_ledger"]["symbolic_critic"]["veto_count"] == 1
    assert llm.answer_calls == 1
    assert result.debug_payload["v8_cost"]["logical_chat_calls"] == 1


def test_shallow_context_is_one_hop_and_never_authorizes_by_relationship():
    state = {"requester_permission_events": [], "current_permissions": [],
             "current_entities": [{"entity_id": "unneeded"}], "lifecycle_events": []}
    relations = [{"type": "related_to", "from_id": "guest", "to_id": "a"},
                 {"type": "related_to", "from_id": "a", "to_id": "b"}]
    graph = shallow_graph_context(state, requester="guest", identity_relations=relations)
    assert graph["requester_identity_evidence"] == relations[:1]
    assert not graph["permission_events"] and "entities" not in graph


@pytest.mark.parametrize("kind", ["SCENE", "ENTITY", "FACT", "RELATION"])
def test_shallow_protocol_rejects_world_graph_records(kind):
    with pytest.raises(ValueError, match="only EVENT"):
        parse_lines(f"EVENTS\n{kind}\tx\tvalue=foo\nACTION\tno_memory\nEND", ingestion=True, shallow=True)


def test_keep_does_not_need_to_regenerate_quotes_newlines_or_numbers():
    text = 'Status: no fee.\nBring "A" and "B" at 10:15; path C:\\new.'
    parsed = parse_lines("SLOT\ts\tslot=status\nKEEP\tk\tslot=status\tcandidate_id=c0\tbind=NONE\nACTION\tanswer\nEND",
                         ingestion=False, shallow=True)
    expand_kept_candidates(parsed, [{"candidate_id": "c0", "text": text}])
    assert parsed["claims"][0]["value"] == text
    with pytest.raises(ValueError, match="unknown candidate"):
        expand_kept_candidates({"claims": [{"_keep_candidate": True, "candidate_id": "missing"}]}, [])


def test_shallow_prefill_only_extracts_events_and_no_lexical_tombstones():
    row = instance()
    class Prefill:
        def chat_text(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            assert set(payload) == {"principal_registry", "context_turns", "new_turns", "resource_registry"}
            return ("EVENTS\nEVENT\te1\tevent_type=lifecycle\teffect=delete\taction=delete\tresource_id=private_note"
                    "\tresource_surface=private note\nSOURCE\te1\tturn_id=t4\tspan=The private note was deleted from memory.\nACTION\tno_memory\nEND")
    batch = extract_v8_events(instance=row,
        new_messages=[{"turn_id": "t4", "text": "The private note was deleted from memory."}],
        context_messages=[], registry=build_v8_principal_registry(row), existing_scenes=[],
        existing_entities=[], llm_client=Prefill(), model_name="fake", response_protocol="lines", memory_mode="shallow")
    assert len(batch.governance_events) == 1
    assert not batch.facts and not batch.entities and not batch.scenes and not batch.relations
    assert batch.audit["deterministic_completed_deletion_tombstones_added"] == 0


def test_incompatible_full_graph_cache_cannot_be_silently_reused(tmp_path):
    path = tmp_path / "store.json"
    V8EventStore(conversation_id="ep").save(path)
    with pytest.raises(ValueError, match="mode changed"):
        V8EventStore.load(path, conversation_id="ep", memory_mode="shallow")


def test_shallow_projection_exposes_deterministic_expiry_without_graph_hops():
    from gov_mem.governance_runtime.v8_state_projector import project_v8_state
    store = V8EventStore(conversation_id="ep", memory_mode="shallow")
    store.governance_events = [{
        "event_id": "t1_allow", "event_type": "permission", "effect": "allow",
        "resource_id": "visit", "resource_surface": "visit", "action": "receive",
        "issuer_principal_id": "owner", "grantee_principal_id": "guest",
        "valid_until": "2026-01-01T10:00:00",
        "source_spans": [{"turn_id": "t1", "span": "Visit access ends at ten."}]}]
    for time, expected in [("2026-01-01T09:00:00", "active"), ("2026-01-01T11:00:00", "expired")]:
        state = project_v8_state(store, visible_turn_ids=["t1"], requester_principal_id="guest",
                                 question="", checkpoint_time=time)
        graph = shallow_graph_context(state, requester="guest", identity_relations=[])
        assert graph["permission_state"] == [{"event_id": "t1_allow", "decision": "allow", "activation": expected}]
        assert graph["checkpoint_time"] == time


def test_shallow_prefill_can_ground_new_revocation_in_prior_resource_quote():
    row = instance()
    resources = [{"resource_id": "private_note", "resource_surface": "private note",
                  "source_spans": [{"turn_id": "old", "span": "The private note is shared with guest."}]}]
    class Prefill:
        def chat_text(self, **kwargs):
            assert json.loads(kwargs["user_prompt"])["resource_registry"] == resources
            return ("EVENTS\nEVENT\tnew_revoke\tevent_type=permission\teffect=revoke\taction=receive"
                    "\tresource_id=private_note\tresource_surface=private note\tgrantee_principal_id=guest\n"
                    "SOURCE\tnew_revoke\tturn_id=old\tspan=The private note is shared with guest.\n"
                    "SOURCE\tnew_revoke\tturn_id=new\tspan=Revoke that access now.\nACTION\tno_memory\nEND")
    batch = extract_v8_events(instance=row, new_messages=[{"turn_id": "new", "text": "Revoke that access now."}],
        context_messages=[], registry=build_v8_principal_registry(row), existing_scenes=[], existing_entities=[],
        llm_client=Prefill(), model_name="fake", response_protocol="lines", memory_mode="shallow", resource_registry=resources)
    assert not batch.rejected
    assert batch.governance_events[0].effect == "revoke"
    assert {s.turn_id for s in batch.governance_events[0].source_spans} == {"old", "new"}


def test_keep_expansion_still_redacts_an_explicit_block_in_same_source():
    from gov_mem.data.schema import RetrievedEvidence
    from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
    from gov_mem.governance_runtime.v8_safe_evidence import build_v8_safe_evidence
    class Mixed:
        def chat_text(self, **kwargs):
            return ("SLOT\ts1\tslot=visit\nSLOT\ts2\tslot=token\n"
                    "KEEP\tk1\tslot=visit\tcandidate_id=candidate_0\tbind=NONE\n"
                    "CLAIM\tc1\tslot=token\tcandidate_id=candidate_0\tvalue=Q-88\tdelivery=block\tbind=NONE"
                    "\trestriction_kind=explicit_restriction\n"
                    "RESTRICTION\tc1\tturn_id=t1\tspan=Do not share the token.\n"
                    "ACTION\tpartial_answer\nEND")
    evidence = [RetrievedEvidence(memory_id="m", content="Visit July 9; token Q-88. Do not share the token.",
        score=1, retrieval_source="dense", reason="test", source_message_ids=["t1"])]
    ledger = reason_v8_claims(instance=instance(), evidence=evidence, state_projection={}, graph_context={},
        llm_client=Mixed(), model_name="fake", response_protocol="lines", memory_mode="shallow")
    safe, audit = build_v8_safe_evidence(ledger)
    assert safe and "July 9" in safe[0].content
    assert "Q-88" not in safe[0].content
    assert audit["blocked_requested_count"] > 0


@pytest.mark.parametrize("bad_claim", ["unknown_candidate", "missing_value", "missing_bind"])
def test_shallow_valid_delta_is_durable_before_claim_contract_repair(tmp_path, bad_claim):
    class BadThenValid(LedgerLLM):
        def __init__(self):
            super().__init__()
            self.stage2 = []
        def chat_json(self, **kwargs):
            raw = super().chat_json(**kwargs)
            if "Stage-2 governance reasoner" in kwargs["system_prompt"]:
                self.stage2.append(json.loads(kwargs["user_prompt"]))
                if len(self.stage2) == 1:
                    if bad_claim == "unknown_candidate":
                        raw["claims"][0]["candidate_id"] = "invented"
                    elif bad_claim == "missing_bind":
                        raw["claims"][0].pop("bind")
                    else:
                        raw["claims"][0].pop("keep")
                        raw["claims"][0]["delivery"] = "exact"
            return raw
    llm = BadThenValid()
    cfg = config()
    cfg["v8_query"] = {"response_protocol": "json", "max_contract_repairs": 1}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=Embeddings(),
        config=cfg, output_dir=tmp_path, dataset_name="synthetic")
    result = backbone.run_instance(instance())
    assert result.answer_result.action == "answer"
    assert "ingestion" in llm.stage2[0] and "ingestion" not in llm.stage2[1]
    stored = json.loads((tmp_path / "v8_event_cache/synthetic/episode.json").read_text())
    assert stored["processed_turn_ids"] == ["t1", "t2", "t3"]
    assert len(stored["extraction_audits"]) == 1
    assert result.debug_payload["v8_cost"]["logical_chat_calls"] == 3


def test_shallow_grounded_delete_binding_vetoes_release_before_answering(tmp_path):
    class DeletedLLM:
        def chat_json(self, **kwargs):
            assert "Stage-2 governance reasoner" in kwargs["system_prompt"], "Deleted value reached Stage 3"
            request = json.loads(kwargs["user_prompt"])
            candidate = next(c for c in request["rag_candidates"] if "amber" in c["text"])
            raw = {"query_slots": [{"slot": "note"}], "answer_action": "answer", "claims": [
                {"slot": "note", "candidate_id": candidate["candidate_id"], "value": "amber",
                 "delivery": "exact", "bind": ["t4_delete"]}]}
            if "ingestion" in request:
                raw["events"] = [{"event_id": "t4_delete", "event_type": "lifecycle", "effect": "delete",
                    "action": "delete", "resource_id": "note", "resource_surface": "private note",
                    "source_spans": [{"turn_id": "t4", "span": "Delete the private note from memory."}]}]
            return raw
    cfg = config()
    cfg["v8_query"]["response_protocol"] = "json"
    backbone = GovMemV8LateGovernanceBackbone(llm_client=DeletedLLM(), embedding_client=Embeddings(),
        config=cfg, output_dir=tmp_path, dataset_name="synthetic")
    row = instance()
    row.messages.append({"turn_id": "t4", "speaker_id": "owner", "text": "Delete the private note from memory."})
    row.question = "What was the private note?"
    for query_id in ["q1", "q2"]:
        row.instance_id = query_id
        result = backbone.run_instance(row)
        assert result.answer_result.action == "no_memory"
        assert "amber" not in result.answer_result.answer_text
        assert result.debug_payload["v8_claim_ledger"]["claims"][0]["symbolic_veto_reasons"] == ["bound_lifecycle_deletion_veto"]
        assert result.debug_payload["v8_cost"]["logical_chat_calls"] == 1


def test_context_compaction_preserves_every_source_and_nonduplicate_resource():
    from copy import deepcopy
    from gov_mem.governance_runtime.v8_shallow_memory import compact_shallow_context
    from gov_mem.governance_runtime.v8_prompt_sources import collect_prompt_sources
    event = {"event_id": "e1", "resource_id": "note", "resource_surface": "private note",
        "event_type": "lifecycle", "effect": "delete", "action": "delete", "condition": None,
        "included_scopes": [], "excluded_scopes": [], "confidence": 0.9,
        "source_spans": [{"turn_id": "t2", "span": "Delete the private note."}]}
    duplicate = {k: event[k] for k in ("resource_id", "resource_surface", "source_spans")}
    earlier = {**duplicate, "source_spans": [{"turn_id": "t1", "span": "The private note exists."}]}
    other = {**duplicate, "scene_id": "other_context"}
    graph = {"lifecycle_events": [event], "resource_registry": [duplicate, earlier, other]}
    saved = deepcopy(graph)
    compact = compact_shallow_context(graph)
    assert graph == saved
    assert compact["resource_registry"] == [earlier, other]
    assert collect_prompt_sources([], compact, None) == collect_prompt_sources([], graph, None)
    assert compact["lifecycle_events"][0]["effect"] == "delete"
    assert "condition" not in compact["lifecycle_events"][0]


def test_advisory_projection_folds_only_same_binding_and_never_deletions():
    from copy import deepcopy
    from gov_mem.governance_runtime.v8_shallow_memory import shallow_lifecycle_view
    def event(identity, effect="update", **kw):
        return {"event_id": identity, "event_type": "lifecycle", "effect": effect,
            "resource_id": "r", "action": "use", "source_spans": [{"turn_id": identity, "span": identity}], **kw}
    events = [event("old"), event("delete", "delete"), event("new"),
        event("other", resource_id="other"), event("scoped", included_scopes=["only old values"]),
        event("pending"), event("conditional", condition="after confirmation"),
        event("issuer", issuer_principal_id="someone_else"), event("context", scene_id="other"),
        event("grantee", grantee_principal_id="someone_else"), event("timed", valid_until="later")]
    saved = deepcopy(events)
    state = {"lifecycle_events": events, "lifecycle_state": [
        {"event_id": e["event_id"], "activation": "pending_or_unresolved" if e["event_id"] == "pending" else "active"}
        for e in events]}
    projected, folded = shallow_lifecycle_view(state)
    assert folded == 1
    assert [e["event_id"] for e in projected] == [e["event_id"] for e in events[1:]]
    assert events == saved
    # No activation proof, no folding.
    assert shallow_lifecycle_view({"lifecycle_events": events}) == (events, 0)
    graph = shallow_graph_context(state, requester="owner", identity_relations=[])
    assert "old" not in [e["event_id"] for e in graph["lifecycle_state"]]
    assert "delete" in [e["event_id"] for e in graph["lifecycle_events"]]
