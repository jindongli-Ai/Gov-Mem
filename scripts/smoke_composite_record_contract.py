"""Composite named records must retain record-collection semantics."""

from gov_mem.planning.query_planner import _normalize_certifiable_contract


def main() -> None:
    question = "What is the current reminder recap?"
    contract = _normalize_certifiable_contract({
        "requested_attributes": ["reminder_recap"],
        "attribute_bindings": [{
            "attribute": "reminder_recap", "support_span": "reminder recap",
            "semantic_role": "requested_property", "evidence_slot_hint": "",
            "need_kind": "record_collection",
        }],
    }, question)
    assert contract["request_shape"] == "list", contract
    assert contract["certifiable_needs"][0]["need_kind"] == "record_collection", contract
    print("composite_record_contract_smoke=PASS")


if __name__ == "__main__":
    main()
