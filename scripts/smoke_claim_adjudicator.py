#!/usr/bin/env python3
"""Current-state fallback must ignore meta mentions but keep real updates."""

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.claim_adjudicator import adjudicate_claims, _slot_matches_attribute


class FakeClient:
    def is_available(self):
        return True

    def chat_json(self, **_kwargs):
        return {"decisions": [{
            "attribute": "approved_budget",
            "memory_id": "base",
            "slot_name": "approved_budget",
            "value": "221,000 USD",
            "evidence_span": "Maple budget is 221,000 USD.",
            "decision": "answer",
            "confidence": 1.0,
        }]}


def row(memory_id, text, message_id, slots, *, changed_fields=None, served=True):
    return RetrievedEvidence(
        memory_id=memory_id,
        content=text,
        score=1.0,
        retrieval_source="test",
        reason="test",
        source_message_ids=[message_id],
        time="2026-01-01T00:00:00",
        metadata={
            "slots": slots,
            "stage2_semantic_rerank": {
                "served_attributes": ["approved_budget"] if served else [],
                "typed_fields": [{
                    "attribute": "approved_budget",
                    "slot_name": "approved_budget",
                    "value": next(iter(slots.values())),
                }] if served else [],
            },
            "semantic_tags": {
                "claims": [{
                    "property_label": "approved_budget",
                    "value_span": next(iter(slots.values())),
                    "claim_span": text,
                }],
                "state_delta": {
                    "operation": "set" if changed_fields else "none",
                    "changed_fields": changed_fields or {},
                },
            },
        },
    )


def main():
    role_spec = {
        "requested_attributes": ["backup_helper_phrase", "helper_window"],
        "attribute_bindings": [
            {"attribute": "backup_helper_phrase", "support_span": "backup helper phrase"},
            {"attribute": "helper_window", "support_span": "helper window"},
        ],
    }
    mixed_record = {"slots": {"helper_time_range": "8:40 AM to 10:20 AM"}}
    assert not _slot_matches_attribute(
        "backup_helper_phrase", "helper_time_range", role_spec, mixed_record
    )
    assert _slot_matches_attribute("helper_window", "helper_time_range", role_spec, mixed_record)
    mixed_record["slots"]["visit_window"] = "8:25 AM to 10:05 AM"
    assert not _slot_matches_attribute("helper_window", "visit_window", role_spec, mixed_record)

    spec = {
        "requested_attributes": ["approved_budget"],
        "temporal_scope": "current",
        "attribute_bindings": [{
            "attribute": "approved_budget",
            "support_span": "approved budget",
            "evidence_slot_hint": "approved_budget",
        }],
    }
    meta = row(
        "meta",
        "Nothing in this message changes the approved budget.",
        "t129",
        {"approved_budget": "approved budget"},
    )
    base = row(
        "base",
        "Maple budget is 221,000 USD.",
        "t058",
        {"approved_budget": "221,000 USD"},
    )
    selected, _ = adjudicate_claims(
        question="What is the current approved budget?",
        semantic_spec=spec,
        evidence=[base, meta],
        llm_client=FakeClient(),
        model_name="test",
    )
    assert [(item["memory_id"], item["value"]) for item in selected] == [("base", "221,000 USD")], selected

    update = row(
        "update",
        "The approved budget is now 225,000 USD.",
        "t130",
        {"approved_budget": "225,000 USD"},
        changed_fields={"approved_budget": "225,000 USD"},
    )
    selected, _ = adjudicate_claims(
        question="What is the current approved budget?",
        semantic_spec=spec,
        evidence=[base, update],
        llm_client=FakeClient(),
        model_name="test",
    )
    assert [(item["memory_id"], item["value"]) for item in selected] == [("update", "225,000 USD")], selected
    print("claim_adjudicator_smoke=PASS")


if __name__ == "__main__":
    main()
