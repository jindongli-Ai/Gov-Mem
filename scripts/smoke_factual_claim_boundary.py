#!/usr/bin/env python3
"""The typed boundary must separate factual values from policy prose."""

from gov_mem.governance_runtime.factual_claim_quality import factual_value_is_eligible
from gov_mem.legacy.graph_slot_renderer import graph_certificate_typed_compatibility


def main() -> None:
    fact_spec = {
        "request_shape": "fact",
        "requested_attributes": ["approved_budget"],
        "attribute_bindings": [{"attribute": "approved_budget", "slot_name": "approved_budget"}],
    }
    assert not factual_value_is_eligible(
        attribute="approved_budget",
        slot_name="approved_budget",
        value="no single calendar edit should be interpreted as a change in launch date",
        semantic_spec=fact_spec,
    )
    assert factual_value_is_eligible(
        attribute="current_status",
        slot_name="status",
        value="no remaining blocker",
        semantic_spec=fact_spec,
    )
    assert factual_value_is_eligible(
        attribute="renewal_state",
        slot_name="renewal_state",
        value="no longer auto-renews",
        semantic_spec=fact_spec,
    )
    policy_spec = {"request_shape": "policy", "requested_attributes": ["sharing_rule"]}
    assert factual_value_is_eligible(
        attribute="sharing_rule",
        slot_name="sharing_rule",
        value="do not share this record outside the treatment team",
        semantic_spec=policy_spec,
    )

    certificate = {
        "authorized": True,
        "slots": {"approved_budget": {"value": "do not return the budget"}},
        "realizations": [{
            "attribute": "approved_budget",
            "slot_name": "approved_budget",
            "value": "do not return the budget",
            "typed_slot_value": "do not return the budget",
            "source_text": "The policy says do not return the budget.",
            "source_memory_id": "m1",
        }],
    }
    assert not graph_certificate_typed_compatibility(
        certificate=certificate,
        semantic_spec=fact_spec,
    )
    print("factual_claim_boundary_smoke=PASS")


if __name__ == "__main__":
    main()
