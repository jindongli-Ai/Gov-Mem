from __future__ import annotations

from types import SimpleNamespace

from gov_mem.backbones.rag_naive import (
    _append_missing_verified_safe_wording,
    _build_turn_chunks,
    _format_retrieved_memory,
    _direct_answer,
    _source_grounded_access_context,
    _answer_redacted_confirmation_boundary_reason,
    _explicit_private_negative_delivery_boundary_reason,
    _normalize_claim_contract,
    _build_provenance_explanation,
    _run_claim_provenance_verifier,
    _arbitrate_in_scope_operational_delivery,
    _sanitize_answer_redacted_sensitive_values,
    _realize_certified_operational_answer,
    _preserve_symbolic_state_carriers,
    _merge_question_safety_into_contract,
    _is_current_query_item,
)
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.llm.client import LLMClientUnavailableError
from gov_mem.backbones.stage2_typed_rerank import Stage2Decision
from gov_mem.backbones.stage2_typed_rerank import _mixed_reasoning_prompt
from gov_mem.data.schema import AnswerResult, MemoryInstance, RetrievedEvidence


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_json(self, *, model, system_prompt, user_prompt):
        self.calls.append((model, system_prompt, user_prompt))
        return self.response


def test_semantic_contract_sensitive_confirmation_is_not_released():
    reason = _answer_redacted_confirmation_boundary_reason(
        question="Is the current case blocker still the clearance form?",
        answer="No, the current blocker is the reentry form.",
        action="answer_redacted",
        query_analysis={
            "semantic_contract_sensitive": True,
            "safety": {
                "sensitive": False,
                "exact": False,
                "confirmation": True,
                "existence": False,
                "privacy_scope": "ordinary",
                "available": True,
            },
        },
    )
    assert reason == "answer_redacted contained a sensitive confirmation"


def test_sensitive_confirmation_guard_handles_prefixed_yes_no_template():
    reason = _answer_redacted_confirmation_boundary_reason(
        question="For a quick yes or no: is the current case blocker still the clearance form?",
        answer="No, the blocker is a different form.",
        action="answer_redacted",
        query_analysis={
            "semantic_contract_sensitive": True,
            "safety": {
                "sensitive": True, "exact": False, "confirmation": True,
                "existence": False, "privacy_scope": "private", "available": True,
            },
        },
    )
    assert reason == "answer_redacted contained a sensitive confirmation"


def test_explicit_private_negative_query_cannot_be_softened_to_redacted_answer():
    reason = _explicit_private_negative_delivery_boundary_reason(
        query_analysis={
            "safety": {"sensitive": True, "privacy_scope": "private"},
            "authorization": {"negative": True},
        },
        authorization_decision="deny",
        action="answer_redacted",
    )
    assert reason == "explicit private negative authorization cannot be delivered as a redacted answer"


def test_grounded_compiler_private_negative_is_not_overridden_by_stage2_analysis():
    reason = _explicit_private_negative_delivery_boundary_reason(
        query_analysis={
            "safety": {"sensitive": False, "privacy_scope": "ordinary"},
            "authorization": {"negative": False},
        },
        authorization_decision="deny",
        action="answer_redacted",
        semantic_delivery_signal={
            "grounded_restricted_atom": True,
            "grounded_negative_policy_atom": True,
        },
    )
    assert reason == "grounded private negative policy cannot be delivered as a redacted answer"


def test_restricted_atom_without_negative_policy_does_not_force_refusal():
    assert _explicit_private_negative_delivery_boundary_reason(
        query_analysis={}, authorization_decision="unknown", action="answer_redacted",
        semantic_delivery_signal={
            "grounded_restricted_atom": True,
            "grounded_negative_policy_atom": False,
        },
    ) is None


def _instance() -> MemoryInstance:
    return MemoryInstance(
        instance_id="education_ckpt_01",
        domain="education",
        conversation_id="episode_01",
        messages=[
            {
                "turn_id": "t001",
                "message_id": "t001",
                "speaker_id": "student_lina",
                "speaker_role": "student",
                "text": "The current date is May 12, 2026.",
                "timestamp": "2026-05-01T09:00",
                "turn_kind": "dialogue",
                "source_turn": {
                    "turn_id": "t001",
                    "timestamp": "2026-05-01T09:00",
                    "speaker": {"principal_id": "student_lina", "role": "student"},
                    "turn_kind": "dialogue",
                    "text": "The current date is May 12, 2026.",
                    "record_refs": ["date_record"],
                },
            },
            {
                "message_id": "t002",
                "speaker_id": "advisor_nikhil",
                "speaker_role": "advisor",
                "text": "The blocker is room booking.",
                "timestamp": "2026-05-01T09:01",
            },
        ],
        question="What is current?",
        asking_user_id="student_lina",
        choices=None,
        answer=None,
        metadata={
            "requester": {"principal_id": "student_lina", "role": "student"},
            "observable": {"as_of_turn_id": "t002"},
        },
    )


def test_rag_naive_uses_one_turn_chunk_per_message():
    chunks = _build_turn_chunks(_instance())

    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk_0001_t001_t001",
        "chunk_0002_t002_t002",
    ]
    assert [chunk.metadata["chunk_type"] for chunk in chunks] == ["turn", "turn"]
    assert chunks[0].text == "[student:student_lina] The current date is May 12, 2026."
    assert chunks[0].source_message_ids == ["t001"]


def test_query_turn_is_not_used_as_memory_evidence():
    instance = _instance()
    item = SimpleNamespace(
        metadata={"structured_record": {"text": instance.question}},
    )
    assert _is_current_query_item(item, instance)


def test_query_turn_identity_uses_checkpoint_boundary_when_text_is_normalized():
    """A checkpoint query may differ textually from its visible turn."""
    instance = _instance()
    item = SimpleNamespace(
        metadata={
            "structured_record": {
                "turn_id": "t002",
                "text": "As of now, what is the current plan for Priya Shah?",
            },
        },
    )
    instance.question = "As of now, what is my current plan for Priya?"
    assert _is_current_query_item(item, instance)


def test_non_query_turn_with_similar_text_is_retained_by_boundary_identity():
    instance = _instance()
    item = SimpleNamespace(
        metadata={
            "structured_record": {
                "turn_id": "t001",
                "text": instance.question,
            },
        },
    )
    assert not _is_current_query_item(item, instance)


def test_source_grounded_access_context_projects_assignment_path_only():
    instance = _instance()
    instance = MemoryInstance(
        **{
            **instance.__dict__,
            "asking_user_id": "advisor_ava",
        }
    )
    contract = SimpleNamespace(target_entities=[{
        "canonical_reference": "Cinder Clearance",
        "surface_form": "Cinder Clearance",
    }])
    trace = {
        "graph_nodes": [
            {"node_id": "principal::advisor_ava", "node_type": "principal"},
            {"node_id": "principal::student_mateo", "node_type": "principal"},
            {"node_id": "entity::cinder_clearance", "node_type": "entity", "value": "cinder_clearance"},
        ],
        "graph_edges": [
            {"edge_type": "advisor_for", "source": "principal::advisor_ava", "target": "principal::student_mateo", "attributes": {}},
            {"edge_type": "case_subject", "source": "principal::student_mateo", "target": "entity::cinder_clearance", "attributes": {}},
            {"edge_type": "about", "source": "evidence::m1", "target": "entity::cinder_clearance"},
        ],
        "temporal_authorization": {"decision": "unknown"},
    }

    result = _source_grounded_access_context(
        instance=instance, symbolic_trace=trace, semantic_contract=contract,
    )

    assert result["source_grounded"] is True
    assert result["explicit_policy_decision"] == "unknown"
    assert [[edge["relation"] for edge in path] for path in result["assignment_paths"]] == [
        ["advisor_for", "case_subject"],
    ]


def test_source_grounded_access_context_does_not_project_evidence_about_edges():
    contract = SimpleNamespace(target_entities=[{"canonical_reference": "Cinder Clearance"}])
    result = _source_grounded_access_context(
        instance=_instance(),
        semantic_contract=contract,
        symbolic_trace={
            "graph_nodes": [{"node_id": "entity::cinder_clearance", "value": "cinder_clearance"}],
            "graph_edges": [{
                "edge_type": "about", "source": "evidence::m1", "target": "entity::cinder_clearance",
            }],
        },
    )

    assert result["source_grounded"] is False
    assert result["assignment_paths"] == []


def test_source_grounded_slots_allow_ordinary_delivery_without_graph_path():
    contract = SimpleNamespace(requested_slots=[SimpleNamespace(
        slot_id="qslot_0", slot_name="current value", required_for_answer=True,
        sensitivity_possible=False, authorization_relevant=False,
    )])
    result = _source_grounded_access_context(
        instance=_instance(),
        semantic_contract=contract,
        semantic_atoms=[{
            "slot_id": "qslot_0",
            "source": {"chunk_id": "chunk_0001_t001_t001"},
        }],
        symbolic_trace={
            "graph_nodes": [],
            "graph_edges": [],
            "temporal_authorization": {"decision": "unknown"},
        },
        query_analysis={"safety": {
            "delivery_mode": "ordinary_operational",
            "confirmation": False,
            "existence": False,
        }},
    )

    assert result["assignment_paths"] == []
    assert result["semantic_delivery_support"] is True
    assert result["source_grounded"] is True
    assert result["operational_assignment_in_scope"] is True


def test_source_grounded_access_context_matches_each_target_phrase_independently():
    instance = MemoryInstance(
        **{**_instance().__dict__, "asking_user_id": "advisor_ava"}
    )
    contract = SimpleNamespace(target_entities=[
        {"canonical_reference": "Cinder Clearance", "surface_form": "Cinder Clearance"},
        {"canonical_reference": "Cinder Connectors", "surface_form": "Cinder Connectors"},
    ])
    trace = {
        "graph_nodes": [
            {"node_id": "principal::advisor_ava", "value": "advisor_ava"},
            {"node_id": "principal::student_mateo", "value": "student_mateo"},
            {"node_id": "entity::cinder_clearance", "value": "cinder_clearance"},
            {"node_id": "entity::cinder_connectors", "value": "cinder_connectors"},
        ],
        "graph_edges": [
            {"edge_type": "advisor_for", "source": "principal::advisor_ava", "target": "principal::student_mateo", "attributes": {}},
            {"edge_type": "case_subject", "source": "principal::student_mateo", "target": "entity::cinder_clearance", "attributes": {}},
        ],
        "temporal_authorization": {"decision": "unknown"},
    }

    result = _source_grounded_access_context(
        instance=instance, symbolic_trace=trace, semantic_contract=contract,
    )

    assert result["target_nodes"] == ["entity::cinder_clearance", "entity::cinder_connectors"]
    assert result["source_grounded"] is True


def test_in_scope_operational_delivery_recovers_only_certified_unscoped_assignment():
    base = AnswerResult(
        prediction="Room 301",
        answer_text="Room 301",
        used_memory_ids=["m1"],
        reasoning_summary="model was conservative",
        action="answer_redacted",
    )
    recovered = _arbitrate_in_scope_operational_delivery(
        base,
        {"operational_assignment_in_scope": True},
    )
    assert recovered.action == "answer"
    assert recovered.answer_text == base.answer_text


def test_in_scope_operational_delivery_does_not_override_scope_or_policy_boundaries():
    base = AnswerResult(
        prediction="Room 301",
        answer_text="Room 301",
        used_memory_ids=["m1"],
        reasoning_summary="model was conservative",
        action="answer_redacted",
    )
    for context in (
        {"operational_assignment_in_scope": False},
        {"operational_assignment_in_scope": False, "assignment_paths": [[{"attributes": {"access_scope": "limited"}}]]},
    ):
        assert _arbitrate_in_scope_operational_delivery(base, context).action == "answer_redacted"


def test_redacted_answer_does_not_echo_grounded_sensitive_value():
    base = AnswerResult(
        prediction="The private pod is Pod 1405.",
        answer_text="The private pod is Pod 1405.",
        used_memory_ids=["m1"],
        reasoning_summary="model returned a summary",
        action="answer_redacted",
    )
    contract = SimpleNamespace(requested_slots=[SimpleNamespace(
        slot_id="qslot_0", sensitivity_possible=True, authorization_relevant=True,
    )])
    sanitized = _sanitize_answer_redacted_sensitive_values(
        base,
        semantic_atoms=[{"slot_id": "qslot_0", "value": "Pod 1405"}],
        semantic_contract=contract,
        authorization_decision="deny",
    )
    assert sanitized.action == "answer_redacted"
    assert "Pod 1405" not in sanitized.answer_text


def test_redacted_answer_does_not_echo_sensitive_value_when_authorization_unknown():
    base = AnswerResult(
        prediction="The current PIN is 8125.", answer_text="The current PIN is 8125.",
        used_memory_ids=["m1"], reasoning_summary="model returned a redacted answer",
        action="answer_redacted",
    )
    contract = SimpleNamespace(requested_slots=[SimpleNamespace(
        slot_id="qslot_0", sensitivity_possible=True, authorization_relevant=True,
    )])
    sanitized = _sanitize_answer_redacted_sensitive_values(
        base,
        semantic_atoms=[{"slot_id": "qslot_0", "value": "8125"}],
        semantic_contract=contract,
        authorization_decision="unknown",
    )
    assert sanitized.action == "answer_redacted"


def test_refusal_explanation_does_not_echo_protected_value():
    result = AnswerResult(
        prediction="I cannot provide 8125.",
        answer_text="I cannot provide 8125.",
        used_memory_ids=["m1"],
        reasoning_summary="refused",
        action="refuse",
    )
    contract = SimpleNamespace(requested_slots=[SimpleNamespace(
        slot_id="qslot_0", sensitivity_possible=True, authorization_relevant=True,
    )])
    sanitized = _sanitize_answer_redacted_sensitive_values(
        result,
        semantic_atoms=[{"slot_id": "qslot_0", "value": "8125"}],
        semantic_contract=contract,
        query_analysis={"safety": {
            "sensitive": True, "exact": True, "confirmation": False,
            "existence": False, "privacy_scope": "private",
            "delivery_mode": "protected_fact",
        }},
        authorization_decision="unknown",
    )
    assert "8125" not in sanitized.answer_text
    assert sanitized.action == "refuse"


def test_question_safety_lifts_slot_boundary_without_authorizing_access():
    contract = SimpleNamespace(requested_slots=[SimpleNamespace(
        slot_id="qslot_0", sensitivity_possible=False, authorization_relevant=False,
    )])
    changed = _merge_question_safety_into_contract(
        semantic_contract=contract,
        query_analysis={"safety": {
            "sensitive": True, "exact": True, "confirmation": False,
            "existence": False, "privacy_scope": "private",
            "category": "credential", "delivery_mode": "protected_fact",
        }},
        question="Give me the current access code.",
    )
    assert changed
    assert contract.requested_slots[0].sensitivity_possible
    assert contract.requested_slots[0].authorization_relevant


def test_redacted_answer_keeps_non_sensitive_summary_without_grounded_value_echo():
    base = AnswerResult(
        prediction="A high-level summary is available.",
        answer_text="A high-level summary is available.",
        used_memory_ids=["m1"],
        reasoning_summary="model returned a summary",
        action="answer_redacted",
    )
    contract = SimpleNamespace(requested_slots=[SimpleNamespace(
        slot_id="qslot_0", sensitivity_possible=True, authorization_relevant=True,
    )])
    sanitized = _sanitize_answer_redacted_sensitive_values(
        base,
        semantic_atoms=[{"slot_id": "qslot_0", "value": "Pod 1405"}],
        semantic_contract=contract,
    )
    assert sanitized.answer_text == base.answer_text


def test_certified_operational_realization_recovers_all_grounded_slots():
    base = AnswerResult(
        prediction="Requested details are restricted.",
        answer_text="Requested details are restricted.",
        used_memory_ids=[],
        reasoning_summary="model over-refused",
        action="refuse",
    )
    contract = SimpleNamespace(requested_slots=[
        SimpleNamespace(slot_id="qslot_0", slot_name="current date", required_for_answer=True),
        SimpleNamespace(slot_id="qslot_1", slot_name="current amount", required_for_answer=True),
    ])
    realized = _realize_certified_operational_answer(
        base,
        access_context={"operational_assignment_in_scope": True},
        semantic_atoms=[
            {"slot_id": "qslot_0", "value": "December 1, 2027", "source": {"chunk_id": "m1"}},
            {"slot_id": "qslot_1", "value": "3,855 USD", "source": {"chunk_id": "m2"}},
        ],
        semantic_contract=contract,
    )
    assert realized.action == "answer"
    assert "December 1, 2027" in realized.answer_text
    assert "3,855 USD" in realized.answer_text


def test_certified_operational_realization_repairs_incomplete_answer_action():
    base = AnswerResult(
        prediction="The current date is December 1, 2027.",
        answer_text="The current date is December 1, 2027.",
        used_memory_ids=["m1"],
        reasoning_summary="model omitted a field",
        action="answer",
    )
    contract = SimpleNamespace(requested_slots=[
        SimpleNamespace(slot_id="qslot_0", slot_name="current date", required_for_answer=True),
        SimpleNamespace(slot_id="qslot_1", slot_name="current amount", required_for_answer=True),
    ])
    realized = _realize_certified_operational_answer(
        base,
        access_context={"operational_assignment_in_scope": True},
        semantic_atoms=[
            {"slot_id": "qslot_0", "value": "December 1, 2027", "source": {"chunk_id": "m1"}},
            {"slot_id": "qslot_1", "value": "3,855 USD", "source": {"chunk_id": "m2"}},
        ],
        semantic_contract=contract,
    )
    assert realized.action == "answer"
    assert "3,855 USD" in realized.answer_text


def test_certified_operational_realization_does_not_cross_scoped_boundary():
    base = AnswerResult(
        prediction="Requested details are restricted.", answer_text="Requested details are restricted.",
        used_memory_ids=[], reasoning_summary="model over-refused", action="refuse",
    )
    contract = SimpleNamespace(requested_slots=[
        SimpleNamespace(slot_id="qslot_0", slot_name="private value", required_for_answer=True),
    ])
    result = _realize_certified_operational_answer(
        base,
        access_context={"operational_assignment_in_scope": False},
        semantic_atoms=[{"slot_id": "qslot_0", "value": "secret", "source": {"chunk_id": "m1"}}],
        semantic_contract=contract,
    )
    assert result.action == "refuse"


def test_symbolic_carrier_preservation_projects_unique_verified_slot_span():
    selected = [RetrievedEvidence(
        memory_id="m1", content="Current window is 10:00.", score=1.0,
        retrieval_source="dense", reason="test",
        metadata={"structured_record": {"text": "Current window is 10:00.", "turn_id": "t1"}},
    )]
    available = selected + [RetrievedEvidence(
        memory_id="m2",
        content="The device pair is the den tablet and projector; unrelated detail.",
        score=0.5, retrieval_source="dense", reason="test",
        metadata={"structured_record": {
            "text": "The device pair is the den tablet and projector; unrelated detail.",
            "turn_id": "t2",
        }},
    )]
    rows = _preserve_symbolic_state_carriers(
        selected=selected,
        available=available,
        symbolic_trace={
            "semantic_compiler": {
                "query_contract": {"requested_slots": [{
                    "slot_id": "qslot_1", "required_for_answer": True,
                }]},
                "final_grounded_atoms": [{
                    "atom_id": "a1", "slot_id": "qslot_1",
                    "source": {"chunk_id": "m2", "span": "the den tablet and projector"},
                }],
            },
            "temporal_authorization": {"decision": "unknown"},
            "state_ledger": {"fields": {}},
        },
    )
    restored = next(row for row in rows if row.memory_id == "m2")
    assert restored.content == "the den tablet and projector"
    assert "unrelated detail" not in restored.content


def test_symbolic_carrier_preservation_does_not_bypass_explicit_deny():
    selected = [RetrievedEvidence(
        memory_id="m1", content="Current window is 10:00.", score=1.0,
        retrieval_source="dense", reason="test",
    )]
    available = selected + [RetrievedEvidence(
        memory_id="m2", content="The device pair is the den tablet and projector.",
        score=0.5, retrieval_source="dense", reason="test",
    )]
    rows = _preserve_symbolic_state_carriers(
        selected=selected, available=available,
        symbolic_trace={
            "semantic_compiler": {
                "query_contract": {"requested_slots": [{"slot_id": "qslot_1"}]},
                "final_grounded_atoms": [{
                    "slot_id": "qslot_1",
                    "source": {"chunk_id": "m2", "span": "the den tablet and projector"},
                }],
            },
            "temporal_authorization": {"decision": "deny"},
            "state_ledger": {"fields": {}},
        },
    )
    assert [row.memory_id for row in rows] == ["m1"]


def test_rag_naive_retrieval_record_restores_gate_mem_typed_fields():
    record = _build_turn_chunks(_instance())[0].metadata["structured_record"]

    assert record["message_id"] == "t001"
    assert record["turn_id"] == "t001"
    assert record["turn_index"] == 0
    assert record["timestamp"] == "2026-05-01T09:00:00"
    assert record["speaker"] == {
        "principal_id": "student_lina",
        "role": "student",
    }
    assert record["turn_kind"] == "dialogue"
    assert record["text"] == "The current date is May 12, 2026."
    assert record["checkpoint"] == {"as_of_turn_id": "t002"}
    assert record["source_turn"]["record_refs"] == ["date_record"]


def test_rag_naive_stage2_context_uses_valid_json_structured_record():
    chunk = _build_turn_chunks(_instance())[0]
    evidence = [
        RetrievedEvidence(
            memory_id=chunk.chunk_id,
            content=chunk.text,
            score=1.0,
            retrieval_source="dense",
            reason="test",
            user_id="student_lina",
            source_message_ids=["t001"],
            time="2026-05-01T09:00",
            metadata=chunk.metadata,
        )
    ]

    context = _format_retrieved_memory(evidence)

    structured_json = context.split("[STRUCTURED_RECORD] ", 1)[1]
    import json
    parsed = json.loads(structured_json)
    assert parsed["speaker"]["role"] == "student"
    assert parsed["speaker"]["principal_id"] == "student_lina"
    assert parsed["timestamp"] == "2026-05-01T09:00:00"
    assert parsed["source_turn"]["record_refs"] == ["date_record"]


def test_stage2_reasoning_prompt_receives_typed_provenance_without_reextraction():
    chunk = _build_turn_chunks(_instance())[0]
    evidence = [
        RetrievedEvidence(
            memory_id=chunk.chunk_id,
            content=chunk.text,
            score=1.0,
            retrieval_source="dense",
            reason="test",
            user_id="student_lina",
            source_message_ids=["t001"],
            time="2026-05-01T09:00",
            metadata=chunk.metadata,
        )
    ]

    _, prompt = _mixed_reasoning_prompt(
        question="What is the current date?",
        requested_slots=["date"],
        evidence=evidence,
        max_candidate_chars=2400,
    )

    assert '"structured_record"' in prompt
    assert '"role": "student"' in prompt
    assert '"principal_id": "student_lina"' in prompt
    assert '"timestamp": "2026-05-01T09:00:00"' in prompt
    assert '"turn_kind": "dialogue"' in prompt


def test_stage2_reasoning_prompt_receives_v4_symbolic_annotations():
    chunk = _build_turn_chunks(_instance())[0]
    metadata = dict(chunk.metadata)
    metadata["symbolic_provenance"] = {
        "record_complete": True,
        "role_consistent_with_roster": True,
        "checkpoint_consistent": True,
    }
    metadata["symbolic_consistency"] = {
        "passed": False,
        "violation_count": 1,
        "violation_kinds": ["principal_role_conflict"],
    }
    evidence = [
        RetrievedEvidence(
            memory_id=chunk.chunk_id,
            content=chunk.text,
            score=1.0,
            retrieval_source="dense",
            reason="test",
            metadata=metadata,
        )
    ]

    _, prompt = _mixed_reasoning_prompt(
        question="What is the current date?",
        requested_slots=["date"],
        evidence=evidence,
        max_candidate_chars=2400,
    )

    assert '"symbolic_annotations"' in prompt
    assert '"principal_role_conflict"' in prompt


def test_stage2_reasoning_prompt_keeps_validity_certificate_internal():
    chunk = _build_turn_chunks(_instance())[0]
    metadata = dict(chunk.metadata)
    metadata["symbolic_validity_certificate"] = {
        "mode": "shadow",
        "state": "explicit_inactive",
        "current_answer_eligibility": "blocked_in_enforced_mode",
    }
    evidence = [
        RetrievedEvidence(
            memory_id=chunk.chunk_id,
            content=chunk.text,
            score=1.0,
            retrieval_source="dense",
            reason="test",
            metadata=metadata,
        )
    ]

    _, prompt = _mixed_reasoning_prompt(
        question="What is the current date?",
        requested_slots=["date"],
        evidence=evidence,
        max_candidate_chars=2400,
    )

    assert "symbolic_validity_certificate" not in prompt
    assert "blocked_in_enforced_mode" not in prompt


def test_stage1_embedding_input_can_match_raw_gate_mem_turn_text():
    class FakeEmbeddingClient:
        def embed_texts(self, *, model, texts):
            return [[1.0] for _ in texts]

    items = [
        type("Item", (), {
            "memory_id": "m1",
            "scope": "private",
            "memory_type": "chunk",
            "user_id": "student_lina",
            "entities": [],
            "content": "[student:student_lina] The current date is May 12, 2026.",
        })()
    ]
    index = DenseMemoryIndex.build(
        items=items,
        llm_client=FakeEmbeddingClient(),
        embedding_model="text-embedding-3-small",
        embedding_texts=[items[0].content],
        allow_fallback=False,
    )

    assert index.backend == "openai_embedding"
    assert index.rows[0].text == items[0].content


def test_stage1_formal_retrieval_does_not_silently_replace_embedding_backend():
    class UnavailableEmbeddingClient:
        def embed_texts(self, *, model, texts):
            raise LLMClientUnavailableError("embedding unavailable")

    items = [
        type("Item", (), {
            "memory_id": "m1",
            "scope": "private",
            "memory_type": "chunk",
            "user_id": "student_lina",
            "entities": [],
            "content": "turn text",
        })()
    ]
    try:
        DenseMemoryIndex.build(
            items=items,
            llm_client=UnavailableEmbeddingClient(),
            embedding_model="text-embedding-3-small",
            embedding_texts=["turn text"],
            allow_fallback=False,
        )
    except LLMClientUnavailableError:
        pass
    else:
        raise AssertionError("formal Stage 1 must not silently fall back to sparse retrieval")


def test_rag_naive_direct_answer_preserves_official_fields_without_projection():
    llm = FakeLLM(
        {
            "action": "answer",
            "answer": "May 12, 2026; blocker: room booking.",
            "answer_structured": {"date": "May 12, 2026"},
            "used_record_ids": ["chunk_0001_t001_t001"],
        }
    )
    evidence = [
        RetrievedEvidence(
            memory_id="chunk_0001_t001_t001",
            content="[student:student_lina] The current date is May 12, 2026.",
            score=0.9,
            retrieval_source="dense",
            reason="test",
            user_id="student_lina",
            metadata={"speaker_id": "student_lina", "chunk_type": "turn"},
        )
    ]

    result = _direct_answer(
        instance=_instance(),
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
    )

    assert result.action == "answer"
    assert result.answer_text == "May 12, 2026; blocker: room booking."
    assert result.used_memory_ids == ["chunk_0001_t001_t001"]
    assert result.answer_structured == {}
    assert len(llm.calls) == 1
    assert "{global_access_policy_block}" not in llm.calls[0][1]
    assert "Do not infer authorization from a role alone." in llm.calls[0][1]
    assert "[MEMORY PROVIDED]" in llm.calls[0][2]
    assert "The current date is May 12, 2026." in llm.calls[0][2]


def test_rag_naive_claim_verifier_maps_source_message_ids_and_records_audit():
    llm = FakeLLM({
        "action": "answer",
        "answer": "appointment date: May 12, 2026",
        "used_record_ids": ["t001"],
        "claim_contract": {
            "fields": [{
                "field_id": "appointment_date",
                "label": "appointment date",
                "status": "supported",
                "selected_values": ["May 12, 2026"],
                "source_memory_ids": ["t001"],
                "provenance": [{
                    "memory_id": "t001",
                    "source_span": "current date is May 12, 2026",
                }],
            }],
        },
    })
    chunk = _build_turn_chunks(_instance())[0]
    evidence = [RetrievedEvidence(
        memory_id=chunk.chunk_id,
        content=chunk.text,
        score=1.0,
        retrieval_source="dense",
        reason="test",
        user_id="student_lina",
        source_message_ids=["t001"],
        metadata=chunk.metadata,
    )]
    answer = _direct_answer(
        instance=_instance(),
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
    )

    answer, audit = _run_claim_provenance_verifier(
        instance=_instance(),
        evidence=evidence,
        answer_result=answer,
        raw_answer=answer.raw_response["rag_naive_raw"],
        config={"policy_verifier": {"enabled": True, "llm_enabled": False}},
    )

    assert audit["claim_provenance"]["passed"] is True
    assert "claim_provenance_verified" in audit["symbolic_checks"]
    contract = answer.raw_response["claim_contract"]
    assert contract["field_state_projection"]["fields"][0]["source_memory_ids"] == [chunk.chunk_id]
    assert answer.raw_response["answer_grounding"]["policy_privacy_verifier"] == audit
    assert len(llm.calls) == 1


def test_rag_naive_claim_verifier_shadow_mode_preserves_answer_on_contract_failure():
    answer = _direct_answer(
        instance=_instance(),
        evidence=[RetrievedEvidence(
            memory_id="room",
            content="The appointment is in Room 9.",
            score=1.0,
            retrieval_source="dense",
            reason="test",
            source_message_ids=["t002"],
        )],
        llm_client=FakeLLM({
            "action": "answer",
            "answer": "appointment date: August 4",
            "used_record_ids": ["t002"],
            "claim_contract": {"fields": [{
                "label": "appointment date",
                "status": "supported",
                "selected_values": ["August 4"],
                "source_memory_ids": ["t002"],
            }]},
        }),
        model_name="gpt-4o-mini-2024-07-18",
    )

    answer, audit = _run_claim_provenance_verifier(
        instance=_instance(),
        evidence=[RetrievedEvidence(
            memory_id="room",
            content="The appointment is in Room 9.",
            score=1.0,
            retrieval_source="dense",
            reason="test",
            source_message_ids=["t002"],
        )],
        answer_result=answer,
        raw_answer=answer.raw_response["rag_naive_raw"],
        config={"policy_verifier": {
            "enabled": True,
            "llm_enabled": False,
            "claim_provenance_enforcement": False,
        }},
    )

    assert audit["passed"] is False
    assert audit["enforced"] is False
    assert answer.action == "answer"
    assert answer.answer_text == "appointment date: August 4"


def test_provenance_explanation_is_non_intervention_and_source_closed():
    evidence = [RetrievedEvidence(
        memory_id="chunk_date",
        content="[student:student_lina] The current date is May 12, 2026.",
        score=1.0,
        retrieval_source="dense",
        reason="test",
        user_id="student_lina",
        time="2026-05-01T09:00:00",
        source_message_ids=["t001"],
        metadata={
            "structured_record": {
                "turn_id": "t001",
                "timestamp": "2026-05-01T09:00:00",
                "speaker": {"principal_id": "student_lina", "role": "student"},
            }
        },
    )]
    answer = SimpleNamespace(
        action="answer",
        answer_text="The date is May 12, 2026.",
        used_memory_ids=["chunk_date"],
        raw_response={
            "claim_contract": {
                "requested_fields": [{
                    "field_id": "date",
                    "label": "date",
                    "status": "supported",
                    "selected_values": ["May 12, 2026"],
                    "source_memory_ids": ["chunk_date"],
                    "provenance": [{
                        "memory_id": "chunk_date",
                        "source_span": "current date is May 12, 2026",
                    }],
                }]
            }
        },
    )
    decision = Stage2Decision(
        route="typed_scalar",
        applied=True,
        original_memory_ids=["chunk_date"],
        selected_memory_ids=["chunk_date"],
    )
    explanation = _build_provenance_explanation(
        instance=_instance(),
        evidence=evidence,
        answer_result=answer,
        stage2_decision=decision,
        symbolic_trace={
            "consistency": {"violations": []},
            "validity_projection": {"state_counts": {"active": 1}},
            "temporal_authorization": {
                "enabled": True,
                "decision": "allow",
                "enforcement_applied": True,
            },
            "authorization_evidence_boundary": {"filtered_memory_ids": []},
            "state_ledger": {"fields": {"date": {"status": "resolved"}}, "conflicts": []},
        },
        claim_audit={
            "passed": True,
            "enforced": False,
            "reasons": [],
            "claim_provenance": {
                "passed": True,
                "checked_fields": 1,
                "checked_claims": 1,
                "supported_claims": [{"field_id": "date", "values": ["May 12, 2026"]}],
            },
        },
    )

    assert explanation["intervention"] is False
    assert explanation["answer_unchanged"] is True
    assert explanation["scored_by_gatemem"] is False
    assert explanation["final_action"] == "answer"
    assert explanation["selected_evidence"][0]["memory_id"] == "chunk_date"
    assert explanation["symbolic"]["temporal_authorization"]["decision"] == "allow"
    assert explanation["claim_level"]["status"] == "verified"


def test_claim_contract_sources_are_filled_from_typed_state_ledger():
    row = RetrievedEvidence(
        memory_id="chunk_date",
        content="The appointment date is August 4.",
        score=1.0,
        retrieval_source="dense",
        reason="test",
        source_message_ids=["t001"],
        metadata={"symbolic_state_ledger": {
            "fields": {
                "date": {
                    "status": "resolved",
                    "value": "August 4",
                    "source_memory_id": "chunk_date",
                    "quote": "The appointment date is August 4.",
                }
            }
        }},
    )

    contract = _normalize_claim_contract(
        raw={"claim_contract": {"fields": [{
            "field_id": "appointment_date",
            "label": "Appointment Date",
            "status": "supported",
            "selected_values": ["August 4"],
        }]}},
        answer_text="appointment date: August 4",
        evidence=[row],
    )

    field = contract["requested_fields"][0]
    assert field["source_memory_ids"] == ["chunk_date"]
    assert field["provenance"] == [{
        "memory_id": "chunk_date",
        "source_span": "The appointment date is August 4.",
    }]


def test_claim_contract_keeps_a_span_for_each_retrieved_source_chunk():
    evidence = [
        RetrievedEvidence(
            memory_id="chunk_one",
            content="Continue apixaban 5 milligrams twice daily.",
            score=1.0,
            retrieval_source="dense",
            reason="test",
            source_message_ids=["t001"],
        ),
        RetrievedEvidence(
            memory_id="chunk_two",
            content="Increase metoprolol to 37.5 milligrams twice daily.",
            score=0.9,
            retrieval_source="dense",
            reason="test",
            source_message_ids=["t002"],
        ),
    ]

    contract = _normalize_claim_contract(
        raw={"claim_contract": {"fields": [{
            "field_id": "medication_plan",
            "label": "Medication Plan",
            "status": "supported",
            "selected_values": [
                "Continue apixaban 5 milligrams twice daily.",
                "Increase metoprolol to 37.5 milligrams twice daily.",
            ],
            "source_memory_ids": ["t001", "t002"],
        }]}},
        answer_text=(
            "Continue apixaban 5 milligrams twice daily. "
            "Increase metoprolol to 37.5 milligrams twice daily."
        ),
        evidence=evidence,
    )

    assert contract["requested_fields"][0]["provenance"] == [
        {
            "memory_id": "chunk_one",
            "source_span": "Continue apixaban 5 milligrams twice daily.",
        },
        {
            "memory_id": "chunk_two",
            "source_span": "Increase metoprolol to 37.5 milligrams twice daily.",
        },
    ]


def test_nested_field_keyed_claim_contract_is_normalized():
    row = RetrievedEvidence(
        memory_id="chunk_date",
        content="The current review date is August 4.",
        score=1.0,
        retrieval_source="dense",
        reason="test",
        source_message_ids=["t001"],
    )

    contract = _normalize_claim_contract(
        raw={"answer_structured": {"claim_contract": {
            "current_review_date": {
                "label": "current review date",
                "status": "supported",
                "selected_values": ["August 4"],
                "source_memory_ids": ["t001"],
                "provenance": {
                    "memory_id": "t001",
                    "source_span": "current review date is August 4",
                },
            },
        }}},
        answer_text="The current review date is August 4.",
        evidence=[row],
    )

    assert contract["requested_fields"] == [{
        "field_id": "current_review_date",
        "label": "current review date",
        "status": "supported",
        "selected_values": ["August 4"],
        "source_memory_ids": ["chunk_date"],
        "provenance": [{
            "memory_id": "chunk_date",
            "source_span": "current review date is August 4",
        }],
    }]


def test_mixed_answer_action_is_not_changed_without_rerealization():
    llm = FakeLLM({
        "action": "answer_redacted",
        "answer": "A high-level status is available.",
        "used_record_ids": ["status"],
    })
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        original_memory_ids=["status"],
        selected_memory_ids=["status"],
        query_analysis={"fields": [{"name": "current status"}]},
    )
    row = RetrievedEvidence(
        memory_id="status",
        content="The current status is active.",
        score=1.0,
        retrieval_source="dense",
        reason="test",
    )

    result = _direct_answer(
        instance=_instance(),
        evidence=[row],
        stage2_decision=decision,
        llm_client=llm,
        model_name="gpt-5.4-mini",
    )

    assert result.action == "answer_redacted"
    assert result.answer_text == "A high-level status is available."


def test_stage2_answer_instruction_preserves_complete_mixed_field_contract():
    llm = FakeLLM({"action": "answer", "answer": "The side door keypad is required after 4:20 PM."})
    instance = _instance()
    instance.question = "What is the current arrival window and approved entrance?"
    evidence = [
        RetrievedEvidence(
            memory_id="schedule",
            content="Current arrival window is 4:20 PM to 6:10 PM.",
            score=0.9,
            retrieval_source="dense",
            reason="test",
            metadata={"projection_requested_slots": ["visit_window", "entry_method"]},
        ),
        RetrievedEvidence(
            memory_id="entry",
            content="Use the side door keypad only after 4:20 PM.",
            score=0.8,
            retrieval_source="dense",
            reason="test",
            metadata={"projection_requested_slots": ["visit_window", "entry_method"]},
        ),
    ]
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        projection_applied=True,
        original_memory_ids=["schedule", "entry"],
        selected_memory_ids=["schedule", "entry"],
    )

    _direct_answer(
        instance=instance,
        evidence=evidence,
        stage2_decision=decision,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
    )

    assert "answer every explicitly requested field" in llm.calls[0][2]
    assert "only, after, before" in llm.calls[0][2]


def test_verified_safe_wording_restores_omitted_concrete_clause():
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        long_context_applied=True,
        long_context_fields=["safe_wording", "date", "household_plan.visit_window"],
    )
    evidence = [
        RetrievedEvidence(
            memory_id="safe-wording",
            content="verified safe wording",
            score=1.0,
            retrieval_source="stage2_long_context",
            reason="test",
            metadata={
                "stage2_long_context_slot": "safe_wording",
                "stage2_long_context_quote": (
                    "Current local-summary branch is: Saturday buzz after 9:35 AM; "
                    "Sunday desk release after 1:10 PM with Omar 2:25 PM to 2:40 PM."
                ),
            },
        )
    ]

    repaired = _append_missing_verified_safe_wording(
        answer="Saturday buzz after 9:35 AM; Sunday desk release after 1:10 PM.",
        evidence=evidence,
        decision=decision,
    )

    assert repaired == "Saturday buzz after 9:35 AM; Sunday desk release after 1:10 PM."


def test_verified_safe_wording_does_not_restore_sensitive_carrier():
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        long_context_applied=True,
        long_context_fields=["safe_wording"],
    )
    evidence = [
        RetrievedEvidence(
            memory_id="sensitive-wording",
            content="verified safe wording",
            score=1.0,
            retrieval_source="stage2_long_context",
            reason="test",
            metadata={
                "stage2_long_context_slot": "safe_wording",
                "stage2_long_context_quote": "Public wording only; PIN 9136 after 1:10 PM.",
            },
        )
    ]

    repaired = _append_missing_verified_safe_wording(
        answer="Public wording only.",
        evidence=evidence,
        decision=decision,
    )

    assert "9136" not in repaired


def test_stage2_answer_instruction_stays_off_sensitive_mixed_requests():
    llm = FakeLLM({"action": "answer", "answer": "The current token is available."})
    instance = _instance()
    instance.question = "What is the current arrival window and exact active token?"
    evidence = [
        RetrievedEvidence(
            memory_id="schedule",
            content="Current arrival window is 4:20 PM to 6:10 PM.",
            score=0.9,
            retrieval_source="dense",
            reason="test",
            metadata={"projection_requested_slots": ["visit_window", "access_token"]},
        )
    ]
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        projection_applied=True,
        original_memory_ids=["schedule"],
        selected_memory_ids=["schedule"],
    )

    _direct_answer(
        instance=instance,
        evidence=evidence,
        stage2_decision=decision,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
    )

    assert "Stage 2 field-completeness check" in llm.calls[0][2]


def test_redacted_sensitive_confirmation_is_replaced_with_refusal():
    llm = FakeLLM({
        "action": "answer_redacted",
        "answer": "No, there is no hidden spare-key location on file for this unit.",
        "used_record_ids": ["private-row"],
    })
    instance = _instance()
    instance.question = "Is there still a hidden spare-key location on file for this unit?"

    result = _direct_answer(
        instance=instance,
        evidence=[],
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
    )

    assert result.action == "answer_redacted"
    assert result.answer_text == "No, there is no hidden spare-key location on file for this unit."
    assert result.used_memory_ids == ["private-row"]


def test_redacted_ordinary_confirmation_is_left_unchanged():
    llm = FakeLLM({
        "action": "answer_redacted",
        "answer": "Yes, the review is scheduled for Monday.",
        "used_record_ids": ["schedule-row"],
    })
    instance = _instance()
    instance.question = "Is the current review date Monday?"

    result = _direct_answer(
        instance=instance,
        evidence=[],
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
    )

    assert result.action == "answer_redacted"
    assert result.answer_text == "Yes, the review is scheduled for Monday."
