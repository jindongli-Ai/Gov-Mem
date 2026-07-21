"""Closed-set Stage 2 reranker must retain complementary active records only."""

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.semantic_reranker import semantic_rerank_evidence


class FakeClient:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"decisions": [
            {"memory_id": "a", "classification": "answer_member", "support_span": "active token A", "served_attributes": ["token"], "authorized_for_requester": False, "authorization_support_span": "active token A"},
            {"memory_id": "b", "classification": "redactable_member", "support_span": "active label B", "served_attributes": ["label"], "authorized_for_requester": False, "authorization_support_span": "active label B"},
            {"memory_id": "c", "classification": "unrelated", "support_span": "unrelated note"},
        ]}


def row(memory_id, text, lifecycle="active"):
    return RetrievedEvidence(
        memory_id=memory_id, content=text, score=1.0, retrieval_source="test", reason="test",
        metadata={"memory_status": lifecycle, "slots": {"value": text}},
    )


def main():
    selected, decisions, filtered, debug = semantic_rerank_evidence(
        question="Give the active token and label.", semantic_spec={"requested_attributes": ["token", "label"]},
        evidence=[
            row("a", "active token A"), row("b", "active label B"),
            row("c", "unrelated note"), row("old", "deleted token", "deleted"),
        ],
        requester_id="owner", owner_id="owner", relation_to_owner="owner",
        llm_client=FakeClient(), model_name="test",
    )
    assert [item.memory_id for item in selected] == ["a", "b"], selected
    assert {item["chunk_id"] for item in decisions} == {"a", "b", "c"}, decisions
    assert {item["reason"] for item in filtered} == {"stage2_unrelated", "nonactive_lifecycle"}, filtered
    assert debug["accepted_count"] == 2, debug
    assert all("served_attributes" in item for item in decisions), decisions
    print("semantic_reranker_smoke=PASS")


if __name__ == "__main__":
    main()
