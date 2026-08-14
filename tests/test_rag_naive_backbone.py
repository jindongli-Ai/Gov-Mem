from __future__ import annotations

from gov_mem.backbones.rag_naive import (
    _append_missing_verified_safe_wording,
    _build_turn_chunks,
    _format_retrieved_memory,
    _direct_answer,
)
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.llm.client import LLMClientUnavailableError
from gov_mem.backbones.stage2_typed_rerank import Stage2Decision
from gov_mem.backbones.stage2_typed_rerank import _mixed_reasoning_prompt
from gov_mem.data.schema import MemoryInstance, RetrievedEvidence


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_json(self, *, model, system_prompt, user_prompt):
        self.calls.append((model, system_prompt, user_prompt))
        return self.response


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


def test_rag_naive_retrieval_record_restores_gate_mem_typed_fields():
    record = _build_turn_chunks(_instance())[0].metadata["structured_record"]

    assert record["message_id"] == "t001"
    assert record["turn_id"] == "t001"
    assert record["turn_index"] == 0
    assert record["timestamp"] == "2026-05-01T09:00"
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
    assert parsed["timestamp"] == "2026-05-01T09:00"
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
    assert '"timestamp": "2026-05-01T09:00"' in prompt
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
    assert "[MEMORY PROVIDED]" in llm.calls[0][2]
    assert "The current date is May 12, 2026." in llm.calls[0][2]


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

    assert "Omar 2:25 PM to 2:40 PM" in repaired


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

    assert "Stage 2 field-completeness check" not in llm.calls[0][2]


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

    assert result.action == "refuse"
    assert result.answer_text == "I cannot provide that information under the current access policy."
    assert result.used_memory_ids == []


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
