from __future__ import annotations

from gov_mem.backbones.rag_naive import _build_turn_chunks, _direct_answer
from gov_mem.backbones.stage2_typed_rerank import Stage2Decision
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
                "message_id": "t001",
                "speaker_id": "student_lina",
                "speaker_role": "student",
                "text": "The current date is May 12, 2026.",
                "timestamp": "2026-05-01T09:00",
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
        metadata={"requester": {"principal_id": "student_lina", "role": "student"}},
    )


def test_rag_naive_uses_one_turn_chunk_per_message():
    chunks = _build_turn_chunks(_instance())

    assert [chunk.chunk_id for chunk in chunks] == [
        "education_ckpt_01_msg_0000",
        "education_ckpt_01_msg_0001",
    ]
    assert [chunk.metadata["chunk_type"] for chunk in chunks] == ["turn", "turn"]
    assert chunks[0].text == "[student:student_lina] The current date is May 12, 2026."
    assert chunks[0].source_message_ids == ["t001"]


def test_rag_naive_direct_answer_preserves_official_fields_without_projection():
    llm = FakeLLM(
        {
            "action": "answer",
            "answer": "May 12, 2026; blocker: room booking.",
            "answer_structured": {"date": "May 12, 2026"},
            "used_record_ids": ["education_ckpt_01_msg_0000"],
        }
    )
    evidence = [
        RetrievedEvidence(
            memory_id="education_ckpt_01_msg_0000",
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
    assert result.used_memory_ids == ["education_ckpt_01_msg_0000"]
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
