#!/usr/bin/env python3
"""Regression check for ordered reference events in semantic contracts."""

from gov_mem.planning.query_planner import _normalize_certifiable_contract


def main() -> None:
    question = "What is my current future schedule after today's injection visit?"
    contract = _normalize_certifiable_contract({
        "requested_attributes": ["future_schedule", "injection_visit"],
        "attribute_bindings": [
            {"attribute": "future_schedule", "support_span": "future schedule", "semantic_role": "requested_property", "need_kind": "record_collection"},
            {"attribute": "injection_visit", "support_span": "injection visit", "semantic_role": "requested_property", "need_kind": "record_collection"},
        ],
    }, question)
    assert contract["requested_attributes"] == ["future_schedule"], contract
    assert contract["attribute_bindings"][0]["support_span"] == "future schedule", contract
    assert contract["disclosure_constraints"] == [{
        "constraint_kind": "temporal_access_boundary", "support_span": "after today's injection visit",
    }], contract
    print("ordered_query_contract_smoke=PASS")


if __name__ == "__main__":
    main()
