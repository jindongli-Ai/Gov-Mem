from gov_mem.planning.query_planner import (
    _as_string_list,
    _ground_attribute_contract,
    _recover_exact_slot_surface_contract,
    _recover_exact_target_collection_contract,
    _normalize_certifiable_contract,
    _merge_query_grounded_slot_contract,
    _semantic_spec_from_response,
    _semantic_contract_is_complete,
)
from gov_mem.governance_runtime.semantic_alignment import _query_span_is_grounded
from gov_mem.governance_runtime.action_predictor import _is_deleted_secret_safety_request
from gov_mem.backbones.rag_policy_amem import RAGPolicyAMemBackbone
from gov_mem.planning.query_planner import QueryUnderstandingAgent


def main() -> None:
    class _Plan:
        planning_trace = {"semantic_contract_source": "verified_llm"}
        semantic_spec = {
            "requested_attributes": ["service_window"],
            "requested_slots": [],
            "attribute_bindings_valid": True,
        }

    assert RAGPolicyAMemBackbone._semantic_contract_certifiable(_Plan())
    _Plan.planning_trace = {"semantic_contract_source": "heuristic"}
    assert not RAGPolicyAMemBackbone._semantic_contract_certifiable(_Plan())

    envelope_contract = {
        "requested_attributes": ["service_window"],
        "attribute_bindings": [{
            "attribute": "service_window",
            "support_span": "service window",
            "semantic_role": "requested_property",
        }],
    }
    for response in (
        {"semantic_spec": envelope_contract},
        {"semantic_contract": envelope_contract},
        {"result": {"query_contract": envelope_contract}},
        envelope_contract,
    ):
        assert _semantic_spec_from_response(response) == envelope_contract

    incomplete = {
        "requested_slots": [],
        "temporal_scope": "current",
        "request_shape": "mixed",
    }
    complete = {
        "requested_slots": ["date", "location"],
        "temporal_scope": "current",
        "request_shape": "mixed",
    }
    assert not _semantic_contract_is_complete(incomplete, ["event", "date", "location"])
    assert _semantic_contract_is_complete(complete, ["event", "date", "location"])
    recovered = _recover_exact_slot_surface_contract(
        {"requested_slots": ["appointment_details"], "request_shape": "mixed"},
        "What are Elena's currently authorized appointment details through Friday?",
        target_entities=["Elena"],
    )
    assert recovered["requested_attributes"] == ["appointment_details"], recovered
    assert recovered["attribute_bindings"][0]["support_span"] == "appointment details", recovered
    assert recovered["certifiable_needs"][0]["need_kind"] == "record_collection", recovered
    assert not _recover_exact_slot_surface_contract(
        {"requested_slots": ["invented_field"], "request_shape": "fact"},
        "What is the current plan?",
    )

    # A verified semantic view may retain one field while the initial view
    # independently grounded a second field from the same multi-part query.
    # Contract replacement must preserve both query-grounded bindings.
    merged = _merge_query_grounded_slot_contract(
        candidate={
            "requested_attributes": ["leading_incident_diagnosis"],
            "raw_requested_attributes": ["leading_incident_diagnosis"],
            "attribute_bindings": [{
                "attribute": "leading_incident_diagnosis",
                "support_span": "leading Project Harbor incident diagnosis",
                "semantic_role": "requested_property",
            }],
        },
        fallback={
            "requested_attributes": ["leading_incident_diagnosis", "leo_access_scope"],
            "raw_requested_attributes": ["leading_incident_diagnosis", "leo_access_scope"],
            "attribute_bindings": [
                {
                    "attribute": "leading_incident_diagnosis",
                    "support_span": "leading Project Harbor incident diagnosis",
                    "semantic_role": "requested_property",
                },
                {
                    "attribute": "leo_access_scope",
                    "support_span": "exact access scope",
                    "semantic_role": "requested_property",
                    "evidence_slot_hint": "leo_access_scope",
                },
            ],
        },
        question="What is the current leading Project Harbor incident diagnosis, and what exact access scope does Leo currently have?",
        target_entities=["Project Harbor", "Leo"],
    )
    assert merged["requested_attributes"] == [
        "leading_incident_diagnosis", "leo_access_scope"
    ], merged

    # Entity-qualified planner slots must bind to the property phrase that is
    # actually present in the question, while preserving the canonical slot.
    qualified = _normalize_certifiable_contract(
        {
            "requested_slots": ["leo_access_scope"],
            "requested_attributes": [],
            "attribute_bindings": [],
        },
        "What exact access scope does Leo currently have?",
        target_entities=["Leo"],
    )
    assert qualified["requested_attributes"] == ["leo_access_scope"], qualified
    assert qualified["attribute_bindings"][0]["support_span"] == "access scope", qualified
    structured_targets = _as_string_list([
        {"entity_type": "record", "identification": "booked visits"},
    ])
    assert structured_targets == ["booked visits"], structured_targets
    structured_collection = _recover_exact_target_collection_contract(
        structured_targets,
        "What are the two booked visits right now?",
    )
    assert structured_collection["certifiable_needs"] == [{
        "need_id": "booked_visits",
        "attribute": "booked_visits",
        "query_support_span": "booked visits",
        "evidence_slot_hint": "",
        "temporal_scope": "unspecified",
        "need_kind": "record_collection",
    }], structured_collection

    class _SurfaceVerifier:
        def chat_json(self, **_kwargs):
            return {"semantic_spec": {
                "requested_attributes": ["appointment_details"],
                "attribute_bindings": [{
                    "attribute": "appointment_details", "support_span": "appointment details",
                    "semantic_role": "requested_property", "need_kind": "record_collection",
                }],
                "disclosure_constraints": [{
                    "constraint_kind": "access_boundary", "support_span": "authorized",
                }],
                "request_shape": "mixed",
            }}

    fixture = type("Fixture", (), {
        "question": "What are Elena's authorized appointment details?",
    })()
    surface_verified = QueryUnderstandingAgent(
        llm_client=_SurfaceVerifier(), model_name="fake"
    )._verify_exact_surface_recovery(
        fixture, asking_user_id="family", target_entities=["Elena"],
        candidate_spec={"requested_slots": ["authorized_appointment_details"], "request_shape": "mixed"},
    )
    assert surface_verified["requested_attributes"] == ["appointment_details"], surface_verified
    assert surface_verified["disclosure_constraints"] == [
        {"constraint_kind": "access_boundary", "support_span": "authorized"}
    ], surface_verified
    grounded, valid = _ground_attribute_contract(
        ["archive_plan", "setup_window"],
        [
            {
                "attribute": "archive_plan",
                "support_span": "current Cedar Archive plan",
                "semantic_role": "request_object",
            },
            {
                "attribute": "setup_window",
                "support_span": "setup window",
                "semantic_role": "requested_property",
            },
        ],
        "What is the current Cedar Archive plan, including the setup window?",
        target_entities=["Cedar Archive", "Cedar Archive plan", "setup window"],
    )
    assert valid
    assert grounded == ["setup_window"]

    scoped = _normalize_certifiable_contract(
        {
            "requested_attributes": ["record_details", "temporary_access_window"],
            "attribute_bindings": [
                {
                    "attribute": "record_details",
                    "support_span": "appointment details",
                    "semantic_role": "requested_property",
                    "need_kind": "record_collection",
                },
                {
                    "attribute": "temporary_access_window",
                    "support_span": "through Friday",
                    "semantic_role": "requested_property",
                },
            ],
            "disclosure_constraints": [
                {
                    "constraint_kind": "temporal_access_boundary",
                    "support_span": "through Friday",
                }
            ],
            "request_shape": "list",
        },
        "What are the appointment details through Friday?",
    )
    assert scoped["requested_attributes"] == ["appointment_details"], scoped
    assert scoped["certifiable_needs"][0]["need_kind"] == "record_collection", scoped
    assert scoped["disclosure_constraints"] == [
        {"constraint_kind": "temporal_access_boundary", "support_span": "through Friday"}
    ], scoped
    polluted = _normalize_certifiable_contract(
        {"requested_attributes": ["record_details", "through_friday", "elena"], "attribute_bindings": [
            {"attribute": "record_details", "support_span": "appointment details", "semantic_role": "requested_property", "need_kind": "record_collection"},
            {"attribute": "through_friday", "support_span": "through Friday", "semantic_role": "requested_property"},
            {"attribute": "elena", "support_span": "Elena", "semantic_role": "requested_property"},
        ]}, "What are Elena's appointment details through Friday?", target_entities=["Elena"],
    )
    assert polluted["requested_attributes"] == ["appointment_details"], polluted

    # A deadline or review context must not become an answer field when the
    # user asks for two operational values.
    context_polluted = _normalize_certifiable_contract(
        {"requested_attributes": ["budget", "maximum_discount", "board_review"], "attribute_bindings": [
            {"attribute": "budget", "support_span": "exact current budget", "semantic_role": "requested_property"},
            {"attribute": "maximum_discount", "support_span": "exact current maximum discount", "semantic_role": "requested_property"},
            {"attribute": "board_review", "support_span": "board review", "semantic_role": "request_object"},
        ]},
        "The board review is in five minutes. I need the exact current budget and exact current maximum discount for Timber right now.",
        target_entities=["Timber", "board review"],
    )
    assert context_polluted["requested_attributes"] == ["budget", "maximum_discount"], context_polluted

    # A deletion instruction plus an exact mapping-recovery request is a
    # no-memory safety case even when the wording uses the verb "delete".
    assert RAGPolicyAMemBackbone._query_explicitly_requests_deleted_content(
        "what exact customer mapping was removed after we delete it"
    )
    assert _is_deleted_secret_safety_request(
        "What exact customer mapping was removed before we standardized wording?",
        [type("Evidence", (), {"content": "Delete the exact direct customer mapping from memory.", "metadata": {}, "memory_type": "forgetting"})()],
    )
    assert _is_deleted_secret_safety_request(
        "Which deleted label did the current label replace?",
        [type("Evidence", (), {"content": "The deleted earlier label remains unavailable.", "metadata": {}, "memory_type": "forgetting"})()],
    )
    assert _query_span_is_grounded(
        "record_details",
        "appointment details",
        "What are the appointment details through Friday?",
        {"attribute_bindings": [{
            "attribute": "record_details",
            "support_span": "appointment details through Friday",
            "need_kind": "record_collection",
        }]},
    )
    assert not _query_span_is_grounded(
        "field",
        "entry details",
        "What are the entry details through Friday?",
        {"attribute_bindings": [{"attribute": "field", "support_span": "entry details through Friday"}]},
    )
    print("query_contract_smoke=PASS")


if __name__ == "__main__":
    main()
