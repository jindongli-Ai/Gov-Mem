from __future__ import annotations

import json
import pytest

from gov_mem.backbones.govmem_v8_late_governance import (
    GovMemV8LateGovernanceBackbone,
    _answer_from_safe_evidence,
)
from gov_mem.data.schema import MemoryInstance


class FakeEmbeddingClient:
    def embed_texts(self, *, model, texts):
        return [[1.0, 0.0] if "appointment" in text.casefold() else [0.0, 1.0] for text in texts]


class SequencedLLM:
    def __init__(self):
        self.calls = []

    def chat_json(self, *, model, system_prompt, user_prompt):
        self.calls.append(system_prompt)
        if "Stage-2 governance reasoner" in system_prompt:
            return {
                "query_slots": [{"slot": "appointment date", "required": True}],
                "claims": [{
                    "claim_id": "claim_date", "slot": "appointment date",
                    "value": "June 20", "candidate_id": "candidate_0",
                    "quote": "appointment is June 20", "entity_id": None,
                    "temporal_state": "current", "sensitivity": "ordinary",
                    "authorization": "allow", "delivery": "exact", "reason": "current",
                }],
                "answer_action": "answer",
                "events": {"scenes": [], "entities": [], "relations": [], "governance_events": [],
                           "facts": [{"field_name": "appointment date", "value": "June 20",
                                      "source_spans": [{"turn_id": "t002", "span": "appointment is June 20"}]}]},
            }
        return {"action": "answer", "answer": "The appointment is June 20.", "used_claim_ids": ["claim_date"]}


class FailingExtractionLLM(SequencedLLM):
    def chat_json(self, *, model, system_prompt, user_prompt):
        if "append-only memory events" in system_prompt:
            raise RuntimeError("temporary")
        return super().chat_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt)


def _instance():
    return MemoryInstance(
        instance_id="medical_ep_ckpt_1", conversation_id="medical_ep", domain="medical",
        messages=[
            {
                "turn_id": "t001", "message_id": "t001", "speaker_id": "patient",
                "speaker_role": "patient", "text": "We discussed parking.",
                "timestamp": "2026-01-01T09:00", "turn_kind": "dialogue",
            },
            {
                "turn_id": "t002", "message_id": "t002", "speaker_id": "scheduler",
                "speaker_role": "scheduler", "text": "The appointment is June 20.",
                "timestamp": "2026-01-01T09:01", "turn_kind": "dialogue",
            },
        ],
        question="When is the appointment?", asking_user_id="patient",
        choices=None, answer=None,
        metadata={
            "requester": {"principal_id": "patient", "role": "patient"},
            "observable": {"as_of_turn_id": "t002"},
            "raw_sample": {"episode": {"entities": {
                "principals": [
                    {"principal_id": "patient", "role": "patient", "display_name": "Patient"},
                    {"principal_id": "scheduler", "role": "scheduler", "display_name": "Scheduler"},
                ],
                "relationships": [],
            }}},
        },
    )


def test_v8_backbone_keeps_rag_first_and_answers_from_safe_claims(tmp_path):
    llm = SequencedLLM()
    config = {
        "llm": {"provider": "test", "base_model": "gemini-2.5-flash-lite"},
        "embedding": {"model": "fake", "allow_fallback": False},
        "rag": {"naive_top_k": 20},
        "v8_extraction": {"max_new_turns_per_call": 32, "context_turns": 8},
        "v8_query": {"max_candidate_chars": 2400},
    }
    backbone = GovMemV8LateGovernanceBackbone(
        llm_client=llm, embedding_client=FakeEmbeddingClient(), config=config,
        output_dir=tmp_path, dataset_name="checkpoint_benchmark",
    )
    result = backbone.run_instance(_instance())
    assert result.answer_result.answer_text == "The appointment is June 20."
    assert result.answer_result.used_memory_ids == ["chunk_0002_t002_t002"]
    assert [row.memory_id for row in result.retrieval_result["retrieved_before_privacy_filter"]] == [
        "chunk_0002_t002_t002"
    ]
    assert result.retrieval_result["v8_safe_evidence"]["released_claim_count"] == 1
    assert result.retrieval_result["v8_dense_top_k_count"] == 1
    assert result.retrieval_result["v8_adjacent_context_count"] == 1
    assert len(llm.calls) == 2
    assert result.debug_payload["v8_cost"]["logical_chat_calls"] == 2


def test_failed_extraction_does_not_advance_cache_watermark(tmp_path):
    config = {
        "llm": {"provider": "test", "base_model": "gemini-2.5-flash-lite"},
        "embedding": {"model": "fake", "allow_fallback": False},
        "rag": {"naive_top_k": 20},
        "v8_extraction": {"max_new_turns_per_call": 32, "context_turns": 8},
        "v8_query": {"max_candidate_chars": 2400},
    }
    backbone = GovMemV8LateGovernanceBackbone(
        llm_client=FailingExtractionLLM(), embedding_client=FakeEmbeddingClient(),
        config=config, output_dir=tmp_path, dataset_name="checkpoint_benchmark",
    )
    with pytest.raises(RuntimeError, match="temporary"):
        backbone.run_instance(_instance())
    assert backbone._stores["medical_ep"].processed_turn_ids == []


def test_same_visible_prefix_reuses_event_cache_without_extraction_call(tmp_path):
    config = {
        "llm": {"provider": "test", "base_model": "gemini-2.5-flash-lite"},
        "embedding": {"model": "fake", "allow_fallback": False},
        "rag": {"naive_top_k": 20},
        "v8_extraction": {"max_new_turns_per_call": 32, "context_turns": 8},
        "v8_query": {"max_candidate_chars": 2400},
    }
    llm = SequencedLLM()
    backbone = GovMemV8LateGovernanceBackbone(
        llm_client=llm, embedding_client=FakeEmbeddingClient(), config=config,
        output_dir=tmp_path, dataset_name="checkpoint_benchmark",
    )
    backbone.run_instance(_instance())
    extraction_calls_before = sum("append-only memory events" in prompt for prompt in llm.calls)
    second = backbone.run_instance(_instance())
    extraction_calls_after = sum("append-only memory events" in prompt for prompt in llm.calls)
    assert extraction_calls_before == 1
    assert extraction_calls_after == 1
    assert second.debug_payload["v8_extraction_audit"]["new_turn_count"] == 0


def test_deleted_only_safe_view_returns_no_memory_action():
    result = _answer_from_safe_evidence(
        instance=_instance(), safe_evidence=[],
        safe_audit={"blocked_claims": [{
            "claim_id": "phone", "slot": "callback number",
            "symbolic_veto_reasons": ["source_grounded_deletion_tombstone"],
        }]},
        llm_client=SequencedLLM(), model_name="gemini-2.5-flash-lite",
        stage2_prompt_audit={},
    )
    assert result.action == "no_memory"
    assert "deleted" in result.answer_text


def test_refusal_takes_precedence_over_confirming_deleted_state():
    result = _answer_from_safe_evidence(
        instance=_instance(), safe_evidence=[],
        safe_audit={
            "claim_ledger_action": "refuse",
            "blocked_claims": [{
                "claim_id": "token", "slot": "credential",
                "symbolic_veto_reasons": ["source_grounded_deletion_tombstone"],
            }],
        },
        llm_client=SequencedLLM(), model_name="gemini-2.5-flash-lite",
        stage2_prompt_audit={},
    )
    assert result.action == "refuse"
    assert "access policy" in result.answer_text


def test_explicit_deleted_history_request_returns_no_memory_even_if_unauthorized():
    instance = _instance()
    instance.question = "Recover the exact old credential."
    result = _answer_from_safe_evidence(
        instance=instance, safe_evidence=[],
        safe_audit={
            "claim_ledger_action": "refuse",
            "blocked_claims": [{
                "claim_id": "token", "slot": "old credential",
                "symbolic_veto_reasons": ["source_grounded_deletion_tombstone"],
            }],
        },
        llm_client=SequencedLLM(), model_name="gemini-2.5-flash-lite",
        stage2_prompt_audit={},
    )
    assert result.action == "no_memory"


def test_joint_delta_governs_current_query_and_never_reaches_answer_prompt(tmp_path):
    """All four checkpoints of a synthetic episode, with one grant/revocation.

    This tests orchestration/cost/isolation, not model semantic accuracy.
    """
    from copy import deepcopy
    instance = _instance()
    instance.conversation_id = "synthetic_policy_episode"
    instance.messages = [
        {"turn_id": "t1", "message_id": "t1", "speaker_id": "scheduler", "speaker_role": "operator",
         "text": "The appointment room is R-7; the appointment token is X-908.", "timestamp": "2026-01-01T09:00"}]
    instance.question = "What are the appointment room and token?"
    observed = []

    class EpisodeLLM:
        def __init__(self): self.calls = []
        def chat_json(self, *, model, system_prompt, user_prompt):
            payload = json.loads(user_prompt)
            self.calls.append(payload)
            if "rag_candidates" not in payload:
                observed.append(payload)
                return {"answer": "; ".join(c["value"] for c in payload["allowed_claims"])}
            candidate = next(c for c in payload["rag_candidates"] if "X-908" in c["text"])
            n = len(instance.messages)
            claims = []
            for field, value in [("room", "R-7"), ("token", "X-908")]:
                blocked = (n >= 2 and field == "token") or n >= 3
                turn = "t3" if n >= 3 else "t2"
                restriction = instance.messages[2 if n >= 3 else 1]["text"] if blocked else ""
                claims.append({"claim_id": field, "slot": field, "requested": True, "value": value,
                               "candidate_id": candidate["candidate_id"], "quote": candidate["text"],
                               "delivery": "block" if blocked else "exact",
                               "restriction_kind": "explicit_restriction" if blocked else None,
                               "restriction_evidence": [{"turn_id": turn, "span": restriction}] if blocked else []})
            events = {k: [] for k in ("scenes", "entities", "facts", "relations", "governance_events")}
            if payload.get("ingestion"):
                for message in payload["ingestion"]["new_turns"]:
                    if message["turn_id"] in {"t2", "t3"}:
                        events["governance_events"].append({
                            "event_id": "p_" + message["turn_id"], "event_type": "permission",
                            "effect": "deny", "issuer_principal_id": "scheduler", "grantee_principal_id": "patient",
                            "resource_id": "appointment_token" if message["turn_id"] == "t2" else "appointment_details",
                            "resource_surface": "appointment token" if message["turn_id"] == "t2" else "appointment room",
                            "action": "receive", "source_spans": [{"turn_id": message["turn_id"], "span": message["text"]}]})
            return {"claims": claims, "query_slots": [{"slot": "room"}, {"slot": "token"}],
                    "answer_action": "refuse" if n >= 3 else "partial_answer" if n == 2 else "answer", "events": events}

    llm = EpisodeLLM()
    config = {"llm": {"base_model": "gemini-2.5-flash-lite"},
              "embedding": {"model": "fake", "allow_fallback": False}, "rag": {"naive_top_k": 20}}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
               config=config, output_dir=tmp_path, dataset_name="synthetic")
    results = []
    for i, text in enumerate([None, "Do not disclose the appointment token to the patient.",
                              "Access revoked: do not disclose the appointment room or token to the patient.", None]):
        if text:
            instance.messages.append({"turn_id": f"t{i+1}", "message_id": f"t{i+1}",
                                      "speaker_id": "scheduler", "text": text, "timestamp": f"2026-01-01T09:0{i}"})
        instance.instance_id = f"synthetic_checkpoint_{i}"
        results.append(backbone.run_instance(deepcopy(instance)))
    assert [r.answer_result.action for r in results] == ["answer", "answer_redacted", "refuse", "refuse"]
    assert [r.debug_payload["v8_cost"]["logical_chat_calls"] for r in results] == [2, 2, 1, 1]
    assert len(llm.calls) == 6
    assert "X-908" not in json.dumps(observed[1])
    assert "R-7" in json.dumps(observed[1])
    assert results[-1].debug_payload["v8_extraction_audit"]["new_turn_count"] == 0
    assert len(results[1].debug_payload["v8_state_projection"]["requester_permissions"]) == 1
    assert results[0].debug_payload["v8_state_projection"]["requester_permissions"] == []
    # Resume from disk: no ingestion is repeated, and the same historical
    # checkpoint cannot see the cached later denial.
    earlier = deepcopy(instance)
    earlier.messages = earlier.messages[:1]
    instance.messages = earlier.messages  # deterministic fake follows requested prefix
    resumed = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
                config=config, output_dir=tmp_path, dataset_name="synthetic")
    result = resumed.run_instance(earlier)
    assert result.answer_result.action == "answer"
    prompt = llm.calls[-2]
    assert prompt["graph_context"]["requester_permissions"] == []
    assert "ingestion" not in prompt
    assert "Access revoked" not in json.dumps(prompt)


def test_full_scoped_answer_is_not_redacted(tmp_path):
    from gov_mem.governance_runtime.v8_safe_evidence import build_v8_safe_evidence
    safe, audit = build_v8_safe_evidence({"claims": [{
        "claim_id": "c", "slot": "appointment date", "value": "June 20", "delivery": "summary",
        "requested": True, "symbolic_advisories": ["scoped_summary_release"]}]})
    answer = _answer_from_safe_evidence(instance=_instance(), safe_evidence=safe, safe_audit=audit,
              llm_client=SequencedLLM(), model_name="test", stage2_prompt_audit={})
    assert answer.action == "answer"


def test_missing_events_does_not_mark_new_turns_processed(tmp_path):
    class MissingEvents(SequencedLLM):
        def chat_json(self, **kwargs):
            raw = super().chat_json(**kwargs)
            raw.pop("events", None)
            return raw
    llm = MissingEvents()
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
        config={"llm": {"base_model": "test"}, "embedding": {"model": "fake"}},
        output_dir=tmp_path, dataset_name="synthetic")
    with pytest.raises(ValueError, match="five event arrays"):
        backbone.run_instance(_instance())
    assert backbone._stores["medical_ep"].processed_turn_ids == []
    assert len(llm.calls) == 1


def test_cold_prefix_extra_call_is_bounded_counted_and_cached(tmp_path):
    class WithPrefill(SequencedLLM):
        def chat_json(self, *, model, system_prompt, user_prompt):
            if "Stage-2 governance reasoner" not in system_prompt and "append-only memory events" in system_prompt:
                self.calls.append(system_prompt)
                return {k: [] for k in ("scenes", "entities", "facts", "relations", "governance_events")}
            return super().chat_json(model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    config = {"llm": {"base_model": "test"}, "embedding": {"model": "fake"},
              "v8_extraction": {"inline_max_new_turns": 1, "max_new_turns_per_call": 1, "max_prefill_calls": 1}}
    llm = WithPrefill()
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
               config=config, output_dir=tmp_path, dataset_name="synthetic")
    result = backbone.run_instance(_instance())
    assert result.debug_payload["v8_cost"]["logical_chat_calls"] == 3
    assert result.debug_payload["v8_cost"]["prefill_calls"] == 1
    assert len(llm.calls) == 3
    assert backbone.run_instance(_instance()).debug_payload["v8_cost"]["logical_chat_calls"] == 2


def test_cold_prefix_budget_failure_makes_no_paid_request(tmp_path):
    llm = SequencedLLM()
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
        config={"llm": {"base_model": "test"}, "embedding": {"model": "fake"},
                "v8_extraction": {"inline_max_new_turns": 1, "max_prefill_calls": 0}},
        output_dir=tmp_path, dataset_name="synthetic")
    with pytest.raises(ValueError, match="budget exceeded"):
        backbone.run_instance(_instance())
    assert llm.calls == []


def test_runner_stops_paid_work_after_first_failure_and_saves_telemetry(tmp_path):
    import logging
    from copy import deepcopy
    from types import SimpleNamespace
    from gov_mem.pipeline import GovMemRunner
    runner = GovMemRunner.__new__(GovMemRunner)
    first, second = _instance(), deepcopy(_instance())
    second.instance_id = "medical_ep_ckpt_2"
    runner.load_dataset = lambda: SimpleNamespace(instances=[first, second], dataset_name="synthetic", metadata={})
    runner.experiment_mode = "govmem_v8_late_governance"
    runner.output_dir, runner.resume, runner.experience_bank = tmp_path, False, None
    runner.config = {"pipeline": {"fail_fast": True}}
    runner.logger = logging.getLogger("test_v8_failure")
    processed, saved = [], []
    def fail(instance, *args):
        processed.append(instance.instance_id)
        raise ValueError("invalid model contract")
    runner._process_instance_backbone = fail
    runner._save_run_metadata = lambda: saved.append(True)
    with pytest.raises(RuntimeError, match="paid telemetry saved"):
        runner.run()
    assert processed == [first.instance_id]
    assert saved == [True]


def test_invalid_advisory_fact_does_not_discard_valid_answer_or_relax_grounding(tmp_path):
    class CombinedFact(SequencedLLM):
        def chat_json(self, **kwargs):
            raw = super().chat_json(**kwargs)
            if 'events' in raw:
                raw['events']['facts'][0]['value'] = 'June 20, arrive early'  # Not one exact source span.
            return raw
    backbone = GovMemV8LateGovernanceBackbone(llm_client=CombinedFact(), embedding_client=FakeEmbeddingClient(),
        config={'llm': {'base_model': 'test'}, 'embedding': {'model': 'fake'}},
        output_dir=tmp_path, dataset_name='synthetic')
    result = backbone.run_instance(_instance())
    assert result.answer_result.action == 'answer'
    assert result.debug_payload['v8_event_store']['facts'] == []
    assert result.debug_payload['v8_extraction_audit']['batches'][0]['rejected_count'] == 1


def test_invalid_lifecycle_fact_remains_a_critical_extraction_failure():
    from gov_mem.backbones.govmem_v8_late_governance import _critical_extraction_rejections
    from gov_mem.extraction.v8_event_schema import V8ExtractionBatch
    batch = V8ExtractionBatch(rejected=[{'kind':'fact','reason':'value_not_source_grounded','lifecycle':'delete'}])
    assert _critical_extraction_rejections(batch)


@pytest.mark.parametrize('always_fail', [False, True])
def test_contract_repair_is_bounded_and_metered(tmp_path, always_fail):
    class MissingEvents(SequencedLLM):
        def chat_json(self, **kwargs):
            raw = super().chat_json(**kwargs)
            if always_fail or len(self.calls) == 1:
                raw.pop('events', None)
            return raw
    llm = MissingEvents()
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
        config={'llm': {'base_model': 'test'}, 'embedding': {'model': 'fake'},
                'v8_query': {'max_contract_repairs': 1}}, output_dir=tmp_path, dataset_name='synthetic')
    if always_fail:
        with pytest.raises(ValueError, match='five event arrays'):
            backbone.run_instance(_instance())
        assert len(llm.calls) == 2
        assert backbone._stores['medical_ep'].processed_turn_ids == []
    else:
        result = backbone.run_instance(_instance())
        assert len(llm.calls) == result.debug_payload['v8_cost']['logical_chat_calls'] == 3
        assert 'previous generation failed' in llm.calls[1]
        assert backbone._stores['medical_ep'].processed_turn_ids == ['t001', 't002']


def test_evaluation_retains_errors_and_attempts_entire_episode(tmp_path):
    import logging
    from copy import deepcopy
    from types import SimpleNamespace
    from gov_mem.pipeline import GovMemRunner
    runner = GovMemRunner.__new__(GovMemRunner)
    first, second = _instance(), deepcopy(_instance())
    second.instance_id = 'medical_ep_ckpt_2'
    runner.load_dataset = lambda: SimpleNamespace(instances=[first, second], dataset_name='synthetic', metadata={})
    runner.experiment_mode = 'govmem_v8_late_governance'
    runner.output_dir, runner.resume, runner.experience_bank = tmp_path, False, None
    runner.config = {'pipeline': {'fail_fast': True, 'record_execution_errors': True}}
    runner.logger = logging.getLogger('test_v8_evaluation_errors')
    runner.stage = 'ingest'
    runner.run_official_benchmark_eval = False
    processed = []
    def fail(instance, *args):
        processed.append(instance.instance_id)
        raise ValueError('invalid model contract')
    runner._process_instance_backbone = fail
    runner._save_run_metadata = lambda: None
    runner.run()
    rows = [json.loads(line) for line in (tmp_path/'predictions/synthetic/predictions.jsonl').read_text().splitlines()]
    assert processed == [first.instance_id, second.instance_id]
    assert len(rows) == 2
    assert all(r['action'] == 'error' and r['answer'] == '' and r['execution_status'] == 'error' for r in rows)


@pytest.mark.parametrize('deny', [False, True])
def test_line_protocol_end_to_end_activates_bound_symbolic_veto(tmp_path, deny):
    instance = _instance()
    instance.messages[0]['text'] = 'Do not disclose the appointment date.'
    class LineLLM:
        def __init__(self): self.calls = []
        def chat_text(self, *, model, system_prompt, user_prompt):
            self.calls.append(system_prompt)
            assert 'Return JSON only' not in system_prompt
            assert 'Stage-2 governance reasoner' in system_prompt
            binding = 'p1' if deny else 'NONE'
            text = ('SLOT\ts1\tslot=appointment date\trequired=true\n'
                    'CLAIM\tc1\tslot=appointment date\tcandidate_id=candidate_0'
                    f'\tvalue=June 20\tdelivery=exact\tbind={binding}\nEVENTS\n')
            if deny:
                text += ('EVENT\tp1\tevent_type=permission\teffect=deny\taction=receive'
                         '\tresource_id=date\tresource_surface=appointment date\tgrantee_principal_id=patient\n'
                         'SOURCE\tp1\tturn_id=t001\tspan=Do not disclose the appointment date.\n')
            return text + 'ACTION\tanswer\nEND'
        def chat_json(self, *, model, system_prompt, user_prompt):
            self.calls.append(system_prompt)
            assert 'Return JSON only' in system_prompt
            return {'action': 'answer', 'answer': 'The appointment is June 20.',
                    'used_claim_ids': ['c1']}
    llm = LineLLM()
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
        config={'llm': {'base_model': 'gemini-2.5-flash-lite'}, 'embedding': {'model': 'fake'},
                'v8_query': {'response_protocol': 'lines'}}, output_dir=tmp_path, dataset_name='synthetic')
    result = backbone.run_instance(instance)
    ledger = result.answer_result.raw_response['v8_claim_ledger']
    assert ledger['response_protocol'] == 'lines'
    assert ledger['raw_text'].endswith('END')
    if deny:
        assert len(llm.calls) == 1
        assert result.answer_result.action == 'refuse'
        assert 'June 20' not in result.answer_result.answer_text
        assert ledger['claims'][0]['symbolic_veto_reasons'] == ['bound_explicit_permission_veto']
    else:
        assert len(llm.calls) == 2
        assert result.answer_result.answer_text == 'The appointment is June 20.'


@pytest.mark.parametrize("repair_limit", [0, 1])
def test_valid_graph_delta_survives_bad_claim_and_is_not_reextracted(tmp_path, repair_limit):
    class OnceBadClaim(SequencedLLM):
        def __init__(self):
            super().__init__()
            self.stage2_payloads = []
        def chat_json(self, **kwargs):
            raw = super().chat_json(**kwargs)
            if "Stage-2 governance reasoner" in kwargs["system_prompt"]:
                payload = json.loads(kwargs["user_prompt"])
                self.stage2_payloads.append(payload)
                if len(self.stage2_payloads) == 1:
                    raw["claims"][0]["value"] = "invented date"
                    raw["claims"][0]["quote"] = "invented date"
                else:
                    assert "ingestion" not in payload
                    raw.pop("events")
            return raw
    llm = OnceBadClaim()
    config = {"llm": {"base_model": "gemini-2.5-flash-lite"}, "embedding": {"model": "fake"},
              "v8_query": {"max_contract_repairs": repair_limit}}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm,
        embedding_client=FakeEmbeddingClient(), config=config, output_dir=tmp_path, dataset_name="synthetic")
    if repair_limit == 0:
        first = backbone.run_instance(_instance())
        assert first.answer_result.raw_response["governance_audit_status"]["status"] == "insufficient_evidence"
        # Verify durable cache reuse, not just in-memory reuse.
        store_file = tmp_path / "v8_event_cache/synthetic/medical_ep.json"
        assert json.loads(store_file.read_text())["processed_turn_ids"] == ["t001", "t002"]
        backbone = GovMemV8LateGovernanceBackbone(llm_client=llm,
            embedding_client=FakeEmbeddingClient(), config=config, output_dir=tmp_path, dataset_name="synthetic")
    result = backbone.run_instance(_instance())
    assert result.answer_result.answer_text == "The appointment is June 20."
    assert "ingestion" in llm.stage2_payloads[0]
    assert all("ingestion" not in p for p in llm.stage2_payloads[1:])
    assert len(backbone._stores["medical_ep"].extraction_audits) == 1


def test_public_policy_mixed_request_keeps_utility_and_hides_only_blocked_value(tmp_path):
    instance = _instance()
    instance.messages[1]["text"] = "The appointment is June 20. Private credential Z-19."
    instance.question = "When is the appointment and what is the credential?"
    class MixedLLM:
        def chat_text(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            assert payload["access_policy"]["text"] == "Private credentials require explicit authorization."
            return ("SLOT\ts1\tslot=appointment\nSLOT\ts2\tslot=credential\n"
                    "CLAIM\tc1\tslot=appointment\tcandidate_id=candidate_0\tvalue=June 20\tdelivery=exact\tbind=NONE\n"
                    "CLAIM\tc2\tslot=credential\tcandidate_id=candidate_0\tvalue=Z-19\tdelivery=block\tbind=NONE\trestriction_kind=outside_scope\n"
                    "RESTRICTION\tc2\tturn_id=application_access_policy\tspan=Private credentials require explicit authorization.\n"
                    "EVENTS\nACTION\tpartial_answer\nEND")
        def chat_json(self, **kwargs):
            assert "Z-19" not in kwargs["user_prompt"]
            payload = json.loads(kwargs["user_prompt"])
            assert [c["value"] for c in payload["allowed_claims"]] == ["June 20"]
            assert payload["blocked_slots"][0]["slot"] == "credential"
            return {"answer": "The appointment is June 20. I cannot provide the credential.", "used_claim_ids": []}
    config = {"llm": {"base_model": "gemini-2.5-flash-lite"}, "embedding": {"model": "fake"},
              "v8_query": {"response_protocol": "lines"},
              "access_policy": {"enabled": True, "by_domain": {
                  "medical": "Private credentials require explicit authorization."}}}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=MixedLLM(),
        embedding_client=FakeEmbeddingClient(), config=config, output_dir=tmp_path, dataset_name="synthetic")
    result = backbone.run_instance(instance)
    assert result.answer_result.action == "answer_redacted"
    assert "June 20" in result.answer_result.answer_text
    assert "Z-19" not in result.answer_result.answer_text


def test_joint_extraction_validates_old_retrieved_source_outside_ingestion_context(tmp_path):
    class LifecycleLLM(SequencedLLM):
        def chat_json(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            if "Stage-2 governance reasoner" not in kwargs["system_prompt"] or payload["question"] != "What was the previous appointment?":
                return super().chat_json(**kwargs)
            assert [t["turn_id"] for t in payload["ingestion"]["new_turns"]] == ["t003"]
            assert not payload["ingestion"]["context_turns"]
            candidate = next(c for c in payload["rag_candidates"] if "June 20" in c["text"])
            return {
                "query_slots": [{"slot": "appointment", "required": True}],
                "claims": [{"slot": "appointment", "candidate_id": candidate["candidate_id"],
                    "value": "June 20", "delivery": "block", "restriction_kind": "deleted",
                    "restriction_evidence": [{"turn_id": "t003", "span": "Delete the previous appointment from memory."}]}],
                "answer_action": "no_memory",
                "events": {"scenes": [], "entities": [], "relations": [], "governance_events": [],
                    "facts": [{"field_name": "appointment", "value": "June 20", "lifecycle": "delete",
                        "source_spans": [{"turn_id": "t002", "span": "The appointment is June 20."},
                                         {"turn_id": "t003", "span": "Delete the previous appointment from memory."}]}]},
            }
    config = {"llm": {"base_model": "gemini-2.5-flash-lite"}, "embedding": {"model": "fake"},
              "v8_extraction": {"context_turns": 0}}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=LifecycleLLM(),
        embedding_client=FakeEmbeddingClient(), config=config, output_dir=tmp_path, dataset_name="synthetic")
    instance = _instance()
    backbone.run_instance(instance)
    instance.instance_id = "medical_ep_ckpt_2"
    instance.question = "What was the previous appointment?"
    instance.messages.append({"turn_id": "t003", "message_id": "t003", "speaker_id": "patient",
        "text": "Delete the previous appointment from memory.", "timestamp": "2026-01-01T09:02"})
    result = backbone.run_instance(instance)
    assert "June 20" not in result.answer_result.answer_text
    store = backbone._stores["medical_ep"]
    deletion = next(f for f in store.facts if f["lifecycle"] == "delete")
    assert {s["turn_id"] for s in deletion["source_spans"]} == {"t002", "t003"}
    assert store.processed_turn_ids == ["t001", "t002", "t003"]


def test_enabled_slot_graph_is_observer_and_does_not_supply_stage2_evidence(tmp_path, monkeypatch):
    from gov_mem.data.schema import RetrievedEvidence
    llm = SequencedLLM()
    config = {'llm': {'provider': 'test', 'base_model': 'fake'},
              'embedding': {'model': 'fake', 'allow_fallback': False},
              'rag': {'naive_top_k': 1},
              'memory_governed_slot_graph': {'enabled': True}}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm,
        embedding_client=FakeEmbeddingClient(), config=config,
        output_dir=tmp_path, dataset_name='checkpoint_benchmark')
    called = []
    def graph_retrieve(**kwargs):
        called.append(kwargs)
        return [RetrievedEvidence(memory_id='graph-source', content='The appointment is June 20.',
                 source_message_ids=['t002'], score=1.0, reason='graph match', retrieval_source='governed_slot_graph')], {
                     'enabled': True, 'llm_calls': 1, 'graph': {}, 'retrieved_source_ids': ['graph-source']}
    monkeypatch.setattr(backbone._slot_graph, 'retrieve', graph_retrieve)
    # A graph proposal is deliberately ignored by the answer pipeline.
    from gov_mem.memory.dense_index import DenseMemoryIndex
    monkeypatch.setattr(DenseMemoryIndex, 'query', lambda *a, **kw: [('chunk_0002_t002_t002', 1.0)])
    instance = _instance()
    instance.metadata['observable']['as_of_turn_id'] = 't003'
    instance.messages.append({'turn_id': 't003', 'message_id': 't003', 'speaker_id': 'patient', 'text': instance.question})
    result = backbone.run_instance(instance)
    assert called
    assert all('t003' not in c.source_message_ids for c in called[0]['chunks'])
    assert result.answer_result.answer_text == 'The appointment is June 20.'
    assert result.answer_result.raw_response['governed_slot_graph']['enabled']
    assert result.answer_result.raw_response['governed_slot_graph']['selection']['graph_selected'] == 0
    assert result.answer_result.raw_response['governed_slot_graph']['selection']['dense_memory_ids'] == [
        'chunk_0002_t002_t002']
    assert [row.memory_id for row in result.retrieval_result['retrieved_before_privacy_filter']] == [
        'chunk_0002_t002_t002']
    assert result.answer_result.raw_response['v8_cost']['logical_chat_calls'] == 3


def test_slot_graph_observer_reports_explicit_findings_without_evidence_effect():
    from gov_mem.backbones.govmem_v8_late_governance import _slot_graph_observer_audit
    audit = _slot_graph_observer_audit({"graph": {
        "policies": [{"effect": "deny", "source_chunk_id": "c1", "source_message_id": "t1", "span": "do not share"}],
        "lifecycle": [{"lifecycle_type": "delete", "source_chunk_id": "c2", "source_message_id": "t2", "span": "delete it"}],
    }})
    assert audit["mode"] == "observer_only"
    assert audit["finding_count"] == 2
    assert audit["effects_answer_evidence"] is False
    assert audit["effects_authorization"] is False


def test_graph_audit_is_passed_to_stage2_as_advisory(tmp_path, monkeypatch):
    from gov_mem.data.schema import RetrievedEvidence
    llm = SequencedLLM()
    config = {'llm': {'provider': 'test', 'base_model': 'fake'},
              'embedding': {'model': 'fake', 'allow_fallback': False},
              'rag': {'naive_top_k': 1}, 'memory_governed_slot_graph': {'enabled': True}}
    backbone = GovMemV8LateGovernanceBackbone(llm_client=llm, embedding_client=FakeEmbeddingClient(),
        config=config, output_dir=tmp_path, dataset_name='checkpoint_benchmark')
    def graph_retrieve(**kwargs):
        return [], {'enabled': True, 'llm_calls': 1, 'graph': {
            'policies': [{'effect': 'deny', 'source_message_id': 't001', 'source_chunk_id': 'c1', 'span': 'do not share'}],
            'lifecycle': []}}
    monkeypatch.setattr(backbone._slot_graph, 'retrieve', graph_retrieve)
    from gov_mem.memory.dense_index import DenseMemoryIndex
    monkeypatch.setattr(DenseMemoryIndex, 'query', lambda *a, **kw: [('chunk_0002_t002_t002', 1.0)])
    instance = _instance(); instance.metadata['observable']['as_of_turn_id'] = 't003'
    instance.messages.append({'turn_id': 't003', 'message_id': 't003', 'speaker_id': 'patient', 'text': instance.question})
    backbone.run_instance(instance)
    assert any('GOVERNED SLOT GRAPH AUDIT' in prompt for prompt in llm.calls)
