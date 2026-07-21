from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.provenance_authorization import build_slot_governance_certificate


def main() -> None:
    row = RetrievedEvidence(
        memory_id="m1",
        content="The event is Monday, October 3.",
        score=1.0,
        retrieval_source="test",
        reason="test",
        source_message_ids=["t1"],
    )
    owner = build_slot_governance_certificate(
        semantic_spec={"requested_slots": ["date"]},
        evidence=[row],
        symbolic_decision={"allowed_slots": [], "denied_slots": []},
        query_echo_evidence=[],
        principal_relation="owner",
    )
    non_owner = build_slot_governance_certificate(
        semantic_spec={"requested_slots": ["date"]},
        evidence=[row],
        symbolic_decision={"allowed_slots": [], "denied_slots": []},
        query_echo_evidence=[],
        principal_relation="external",
    )
    assert owner["slots"]["date"]["authorization"] == "allow"
    assert non_owner["slots"]["date"]["authorization"] == "undetermined"
    print("owner_authorization_smoke=PASS")


if __name__ == "__main__":
    main()
