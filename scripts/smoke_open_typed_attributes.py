from gov_mem.data.schema import MemoryInstance, QueryPlan, RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.memory.amem_memory import (
    _lifecycle_from_semantic_tags,
    _normalize_semantic_tags,
    _semantic_annotation_is_informative,
)
from gov_mem.planning.query_planner import (
    QueryUnderstandingAgent,
    _ground_attribute_contract,
    _ground_requested_attributes,
    _consensus_attribute_bindings,
    _normalize_semantic_spec,
)
from gov_mem.reasoning.operators import build_required_slot_plan
from gov_mem.backbones.rag_policy_amem import (
    _absolute_temporal_anchor_value,
    _is_temporal_interval_value,
    _match_typed_attribute,
    _structured_authority_claim_is_eligible,
    _temporal_anchor_value,
)


tags = _normalize_semantic_tags(
    {
        "discourse_act": "update",
        "assertion_confidence": 0.98,
        "event_identity": {"entity_key": "service_visit"},
        "attributes": {
            "Label Strip Color": "amber",
            "Approved Areas": ["scan table", "album shelf"],
        },
        "surface_values": {"Label Strip Color": "amber"},
        "state_delta": {"operation": "set", "changed_fields": ["label_strip_color"]},
    }
)
assert tags["attributes"] == {
    "label_strip_color": "amber",
    "approved_areas": ["scan table", "album shelf"],
}
assert tags["surface_values"] == {"label_strip_color": "amber"}
assert _match_typed_attribute(
    {"service_window_start": "09:20", "service_window_end": "10:25"},
    "service_window",
) == ("service_window", "09:20 to 10:25", 1.0)
assert _match_typed_attribute(
    {"apricot_archive_keypad_code": "2745"},
    "current_numeric_code",
) == ("apricot_archive_keypad_code", "2745", 0.5)
assert _match_typed_attribute(
    {"approved_areas_for_helper_handling": ["scan_table", "album_shelf"]},
    "approved_areas",
) == (
    "approved_areas_for_helper_handling",
    ["scan_table", "album_shelf"],
    1.0,
)
assert _match_typed_attribute(
    {"helper_handling_areas": ["scan_table", "album_shelf"]},
    "approved_areas",
) == (
    "helper_handling_areas",
    ["scan_table", "album_shelf"],
    0.5,
)
assert _match_typed_attribute(
    {"helper_window_start_time": "09:20", "helper_window_end_time": "10:25"},
    "setup_window",
) is None
assert _match_typed_attribute(
    {"max_discount_percent": "8%"},
    "current_approved_maximum_discount",
) == ("max_discount_percent", "8%", 0.5)
assert _temporal_anchor_value({"date": "October 3"}) == "October 3"
assert _absolute_temporal_anchor_value({"date": "Saturday"}) is None
assert _absolute_temporal_anchor_value({"date": "October 3"}) == "October 3"
assert _is_temporal_interval_value("9:10 AM to 10:35 AM")
assert not _is_temporal_interval_value("2745")
malformed_tags = _normalize_semantic_tags({"event_identity": "event", "state_delta": "set"})
assert malformed_tags["event_identity"] == {}
assert malformed_tags["state_delta"] == {"operation": "none"}
assert not _semantic_annotation_is_informative(malformed_tags)
assert _semantic_annotation_is_informative(
    {"discourse_act": "assertion", "attributes": {"status": "active"}}
)
assert _lifecycle_from_semantic_tags(
    "superseded",
    {"discourse_act": "update", "state_delta": {"operation": "set"}},
) == "active"
assert _match_typed_attribute({"keypad_code_current": "2745"}, "current_numeric_code")[1] == "2745"
assert _match_typed_attribute({"grip_label_strip_color": "apricot"}, "label_strip_color")[1] == "apricot"
assert _match_typed_attribute({"unrelated_field": "x"}, "label_strip_color") is None

authoritative_frame = type("Frame", (), {
    "discourse_act": "update",
    "state_delta": {"authority_effect": "authoritative", "changed_fields": {"date": "v2"}},
})()
non_authoritative_frame = type("Frame", (), {
    "discourse_act": "update",
    "state_delta": {"authority_effect": "non_authoritative", "changed_fields": {"date": "v3"}},
})()
unknown_frame = type("Frame", (), {"discourse_act": "unknown", "state_delta": {}})()
assert _structured_authority_claim_is_eligible(authoritative_frame)
assert not _structured_authority_claim_is_eligible(non_authoritative_frame)
assert not _structured_authority_claim_is_eligible(unknown_frame)

row = RetrievedEvidence(
    memory_id="m1",
    content="The current visit uses an amber label strip.",
    score=0.9,
    retrieval_source="test",
    reason="test",
    user_id="owner",
    time="2026-01-02T10:00:00",
    source_message_ids=["s1"],
    metadata={"semantic_tags": tags, "memory_status": "active"},
)
frame = compile_evidence_frame(row)
assert frame.subject_entity == "service_visit"
assert frame.semantic_attributes["label_strip_color"] == "amber"
assert frame.slots["label_strip_color"] == "amber"

delta_only = RetrievedEvidence(
    memory_id="m2",
    content="A structured update.",
    score=0.8,
    retrieval_source="test",
    reason="test",
    metadata={
        "semantic_tags": {
            "discourse_act": "update",
            "state_delta": {"operation": "set", "changed_fields": {"service_window": "09:00-10:00"}},
        }
    },
)
delta_frame = compile_evidence_frame(delta_only)
assert delta_frame.semantic_attributes["service_window"] == "09:00-10:00"
assert delta_frame.slots["service_window"] == "09:00-10:00"

replacement = RetrievedEvidence(
    memory_id="m3",
    content="The code changed from 1111 to 2222.",
    score=0.9,
    retrieval_source="test",
    reason="test",
    metadata={
        "semantic_tags": {
            "discourse_act": "update",
            "attributes": {"keypad_code": "2222", "previous_keypad_code": "1111"},
            "state_delta": {
                "operation": "supersede",
                "changed_fields": {"keypad_code": "2222"},
            },
        }
    },
)
replacement_frame = compile_evidence_frame(replacement)
assert replacement_frame.semantic_attributes == {"keypad_code": "2222"}
assert "previous_keypad_code" not in replacement_frame.slots

semantic_spec = _normalize_semantic_spec(
    {
        "requested_slots": [],
        "requested_attributes": ["Label Strip Color", "approved areas"],
        "temporal_scope": "current",
        "request_shape": "list",
    }
)
assert semantic_spec["requested_attributes"] == ["label_strip_color", "approved_areas"]
assert semantic_spec["request_shape"] == "list"
assert _ground_requested_attributes(
    ["service_window", "unrequested_field"],
    "List the current service window.",
) == ["service_window"]
grounded_bindings, bindings_valid = _ground_attribute_contract(
    ["approved_budget", "service_window"],
    [
        {"attribute": "service_window", "support_span": "service window", "semantic_role": "requested_property"},
        {"attribute": "approved_budget", "support_span": "missing span", "semantic_role": "requested_property"},
    ],
    "List the current service window.",
)
assert bindings_valid and grounded_bindings == ["service_window"]

role_grounded, role_bindings_valid = _ground_attribute_contract(
    ["project_plan", "service_window", "safe_wording", "backup_helper_phrase"],
    [
        {"attribute": "project_plan", "support_span": "current Project Plan", "semantic_role": "target_entity"},
        {"attribute": "service_window", "support_span": "service window", "semantic_role": "requested_property"},
        {"attribute": "safe_wording", "support_span": "backup helper phrase", "semantic_role": "requested_property"},
        {"attribute": "backup_helper_phrase", "support_span": "backup helper phrase", "semantic_role": "requested_property"},
    ],
    "List the current Project Plan, including its service window and backup helper phrase.",
    target_entities=["Project Plan"],
)
assert role_bindings_valid
assert role_grounded == ["service_window", "backup_helper_phrase"]

explicit_role_grounded, explicit_role_valid = _ground_attribute_contract(
    ["current_plan", "service_window"],
    [
        {
            "attribute": "current_plan",
            "support_span": "current Project Plan",
            "semantic_role": "request_object",
        },
        {
            "attribute": "service_window",
            "support_span": "service window",
            "semantic_role": "requested_property",
        },
    ],
    "What is the current Project Plan, including its service window?",
    target_entities=["Project Plan"],
)
assert explicit_role_valid
assert explicit_role_grounded == ["service_window"]

surface_delta = RetrievedEvidence(
    memory_id="m4",
    content="The support amount is 4,190 USD and the discount cap is 8%.",
    score=0.9,
    retrieval_source="test",
    reason="test",
    metadata={
        "semantic_tags": {
            "discourse_act": "assertion",
            "surface_values": {
                "support_amount": "4,190 USD",
                "discount_cap": "8%",
            },
            "state_delta": {
                "operation": "set",
                "changed_fields": {"support_amount": 4190, "discount_cap": 8},
            },
        }
    },
)
surface_frame = compile_evidence_frame(surface_delta)
assert surface_frame.slots["support_amount"] == "4,190 USD"
assert surface_frame.slots["discount_cap"] == "8%"

consensus_grounded, consensus_valid = _ground_attribute_contract(
    ["service_window", "entry_method", "numeric_code", "approved_areas"],
    [
        {"attribute": "service_window", "support_span": "service window", "semantic_role": "requested_property"},
        {"attribute": "entry_method", "support_span": "entry method", "semantic_role": "requested_property"},
        {"attribute": "numeric_code", "support_span": "numeric code", "semantic_role": "requested_property"},
        {"attribute": "approved_areas", "support_span": "approved areas", "semantic_role": "requested_property"},
    ],
    "List the service window, entry method, numeric code, and approved areas.",
    target_entities=["service plan"],
)
assert consensus_valid
assert consensus_grounded == [
    "service_window", "entry_method", "numeric_code", "approved_areas"
]

span_family_grounded, span_family_valid = _ground_attribute_contract(
    ["visit_window", "setup_window", "access_badge", "numeric_code"],
    [
        {"attribute": "visit_window", "support_span": "setup window", "semantic_role": "requested_property"},
        {"attribute": "setup_window", "support_span": "including the setup window", "semantic_role": "requested_property"},
        {"attribute": "access_badge", "support_span": "numeric code", "semantic_role": "requested_property"},
        {"attribute": "numeric_code", "support_span": "current numeric code", "semantic_role": "requested_property"},
    ],
    "Give the plan, including the setup window and current numeric code.",
    target_entities=["plan"],
)
assert span_family_valid
assert span_family_grounded == ["setup_window", "numeric_code"]

consensus_bindings = _consensus_attribute_bindings(
    [
        {"attribute": "date", "support_span": "As of now", "semantic_role": "temporal_modifier"},
        {"attribute": "access_badge", "support_span": "current numeric code", "semantic_role": "requested_property"},
        {"attribute": "setup_window", "support_span": "setup window", "semantic_role": "requested_property"},
    ],
    [
        {"attribute": "current_numeric_code", "support_span": "current numeric code", "semantic_role": "requested_property"},
        {"attribute": "setup_window", "support_span": "the setup window", "semantic_role": "requested_property"},
    ],
)
assert {row["attribute"] for row in consensus_bindings} == {
    "access_badge", "current_numeric_code", "setup_window"
}

instance = MemoryInstance(
    instance_id="contract",
    domain=None,
    conversation_id="conversation",
    messages=[],
    question="List the current plan items, including entry method, approved areas, and label strip color.",
    asking_user_id="owner",
    choices=None,
    answer=None,
)
normalized_plan = QueryUnderstandingAgent.__new__(QueryUnderstandingAgent)._normalize_plan(
    instance,
    {
        "query_type": "factual",
        "target_entities": ["configuration"],
        "semantic_spec": {
            "requested_slots": ["entry_method", "approved_areas"],
            "requested_attributes": ["label_strip_color"],
            "temporal_scope": "current",
            "request_shape": "list",
        },
    },
    asking_user_id="owner",
)
assert normalized_plan.semantic_spec["requested_slots"] == ["entry_method", "approved_areas"]
assert normalized_plan.semantic_spec["raw_requested_attributes"] == ["label_strip_color"]
assert normalized_plan.semantic_spec["requested_attributes"] == []

malformed_plan = QueryUnderstandingAgent.__new__(QueryUnderstandingAgent)._normalize_plan(
    instance,
    {
        "query_type": "factual",
        "target_users": "owner",
        "target_entities": "configuration",
        "required_memory_types": "factual",
        "dense_queries": "query",
        "reasoning_ops": "AND",
        "symbolic_filters": "invalid",
        "semantic_spec": {
            "requested_attributes": ["label_strip_color"],
            "temporal_scope": "current",
            "request_shape": "list",
        },
    },
    asking_user_id="owner",
)
assert isinstance(malformed_plan.symbolic_filters["user_ids"], list)
assert isinstance(malformed_plan.symbolic_filters["entities"], list)
assert isinstance(malformed_plan.symbolic_filters["memory_types"], list)
assert "invalid" not in malformed_plan.symbolic_filters
assert malformed_plan.semantic_spec["raw_requested_attributes"] == ["label_strip_color"]
assert malformed_plan.semantic_spec["requested_attributes"] == []

plan = QueryPlan(
    query_type="factual",
    target_users=[],
    target_entities=["service_visit"],
    required_memory_types=[],
    symbolic_filters={},
    dense_queries=[],
    reasoning_ops=[],
    semantic_spec=semantic_spec,
)
required = build_required_slot_plan("Give the current visit properties.", plan)
assert "label_strip_color" in required["required_slots"]
assert "approved_areas" in required["required_slots"]

print("open_typed_attributes_smoke=PASS")
