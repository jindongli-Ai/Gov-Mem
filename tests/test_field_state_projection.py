from __future__ import annotations

import json
import unittest

from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.execution_planner import build_execution_plan
from gov_mem.governance_executor import execute_policy_decision
from gov_mem.policy_schema import PolicyAction, PolicyDecision

from gov_mem.field_state_projection import (
    FieldEvidenceClosure,
    QueryContract,
    QueryField,
    StateClaim,
    _adjudicate_claims,
    _aggregate_evidence_view,
    _aggregate_recall_support_view,
    _query_view_boundary,
    _normalize_claims,
    _source_carrier_scalar_recovery,
    _source_grounded_scalar_fallback,
    _source_grounded_current_updates,
    build_stateful_projection,
    compile_query_contract,
    projection_evidence_payload,
    resolve_field_claims,
    select_field_evidence,
)


class ClosureLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def is_available(self):
        return True

    def chat_json(self, *, model, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        return next(self.responses)


class PipelineLLM(ClosureLLM):
    pass


def row(memory_id, text, turn):
    return {
        "memory_id": memory_id,
        "text": text,
        "source_turn_index": turn,
        "source_message_ids": [f"msg-{memory_id}"],
    }


class FieldStateProjectionTests(unittest.TestCase):
    def test_source_local_replacement_drops_old_value_and_keeps_new_value(self):
        field = QueryField(
            "date", "Project Atlas current dry-run date",
            attribute="Project Atlas current dry-run date", value_type="date",
        )
        source = (
            "The Atlas dry-run target moves from April 28 to May 12, 2026 "
            "because the microscope room is unavailable on the earlier slot."
        )
        rows = {"m": row("m", source, 44)}
        old = _normalize_claims(
            {"claims": [{"memory_id": "m", "value": "April 28", "relation": "current"}]},
            field=field,
            rows=rows,
        )
        self.assertEqual(old, ())
        updates = _source_grounded_current_updates(field=field, rows=rows)
        self.assertEqual([claim.value for claim in updates], ["May 12, 2026"])
        projection = resolve_field_claims(
            field,
            FieldEvidenceClosure(field_id="date", claims=updates),
        )
        self.assertEqual(projection.selected_values, ("May 12, 2026",))

    def test_empty_local_closure_recovers_from_authorized_global_candidates(self):
        field = QueryField(
            "date", "current target date", attribute="target date", value_type="date",
        )
        llm = ClosureLLM([
            {"claims": []},
            {"claims": [{"memory_id": "global", "value": "June 11, 2027"}]},
        ])
        closures = select_field_evidence(
            contract=QueryContract(fields=(field,), question="What is the current target date?"),
            evidence=[
                {
                    **row("local", "The current blocker is pending review.", 8),
                    "retrieval_fields": ["date"],
                },
                row("global", "The answer is June 11, 2027.", 9),
            ],
            llm_client=llm,
            config={
                "llm": {"answering_model": "test-model"},
                "policy_reasoning": {"field_projection_global_recovery": True},
            },
        )
        self.assertEqual([claim.value for claim in closures[0].claims], ["June 11, 2027"])
        self.assertIn("llm_empty_closure_recovery:1:global", closures[0].trace)
        self.assertTrue(any(trace.startswith("audit:evidence:") for trace in closures[0].trace))
        self.assertIn("audit:llm_claims=0", closures[0].trace)
        self.assertIn("audit:recovery_claims=1", closures[0].trace)

    def test_global_recovery_can_be_disabled_for_ablation(self):
        field = QueryField(
            "date", "current target date", attribute="target date", value_type="date",
        )
        llm = ClosureLLM([{"claims": []}, {"claims": [{"memory_id": "global", "value": "June 11, 2027"}]}])
        closures = select_field_evidence(
            contract=QueryContract(fields=(field,), question="What is the current target date?"),
            evidence=[
                {
                    **row("local", "The current blocker is pending review.", 8),
                    "retrieval_fields": ["date"],
                },
                row("global", "The answer is June 11, 2027.", 9),
            ],
            llm_client=llm,
            config={
                "llm": {"answering_model": "test-model"},
                "policy_reasoning": {"field_projection_global_recovery": False},
            },
        )
        self.assertEqual(closures[0].claims, ())

    def test_safe_summary_uses_source_carrier_without_generic_llm_closure(self):
        field = QueryField(
            "authorized_safe_summary",
            "authorized safe summary",
            attribute="authorized safe summary",
            cardinality="list",
            disclosure_scope="safe_summary",
        )
        llm = ClosureLLM([])
        closures = select_field_evidence(
            contract=QueryContract(
                fields=(field,),
                question="What is the current safe summary?",
            ),
            evidence=[
                row(
                    "carrier",
                    "The safe summary is Lattice Petition with current blocker dean load signature.",
                    18,
                ),
            ],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual([claim.value for claim in closures[0].claims], ["Lattice Petition"])
        self.assertEqual(llm.prompts, [])
    def test_aggregate_view_respects_explicit_audience_and_public_scope(self):
        evidence = [
            {"memory_id": "tech", "text": "Technician calibration is two clicks left.", "user_id": "tech_sara", "scope": "observable"},
            {"memory_id": "safe", "text": "The active tag color is brick.", "user_id": "manager", "scope": "safe_summary"},
            {"memory_id": "neighbor", "text": "The resident helper phrase is cinder soft.", "user_id": "resident_naomi", "scope": "observable"},
        ]
        view = _aggregate_evidence_view(
            "Give me the current technician-facing calibration snapshot.",
            evidence,
        )
        self.assertEqual([row["memory_id"] for row in view], ["tech", "safe"])

    def test_safe_summary_query_creates_projection_boundary_without_case_vocabulary(self):
        projection, audiences = _query_view_boundary(
            "Give me the current concise logistics summary I can send Mason "
            "without leaking restricted material."
        )
        self.assertEqual(projection, "safe_summary")
        self.assertEqual(audiences, ())

    def test_safe_summary_boundary_keeps_explicit_current_carrier_lane(self):
        evidence = [
            {
                **row("carrier", "Current state saved: supper 4:35 to 4:50 by resident buzz.", 95),
                "scope": "observable",
                "retrieval_fields": ["safe_summary"],
            },
            {
                **row("private", "Current private surprise wording remains broad only.", 145),
                "scope": "observable",
            },
        ]
        view = _aggregate_evidence_view(
            "Give me the current concise logistics summary I can send Mason "
            "without leaking restricted material.",
            evidence,
        )
        self.assertIn("carrier", [item["memory_id"] for item in view])
        self.assertNotIn("private", [item["memory_id"] for item in view])

    def test_named_delivery_lane_drops_other_recipient_and_private_projection(self):
        evidence = [
            {
                **row("mason", "Mason, your setup summary is current arrival and approved zones.", 10),
                "retrieval_fields": ["safe_summary"],
            },
            {
                **row("carmen", "Carmen, your helper summary remains porch cubby only.", 11),
                "user_id": "guest_carmen_diaz",
                "retrieval_fields": ["safe_summary"],
            },
            {
                **row("surprise", "Current broad wording remains private dinner surprise staged for Saturday evening.", 12),
                "retrieval_fields": ["safe_summary"],
            },
        ]
        view = _aggregate_evidence_view(
            "Give me the current concise logistics summary I can send Mason without leaking restricted material.",
            evidence,
        )
        self.assertEqual([item["memory_id"] for item in view], ["mason"])

    def test_named_delivery_lane_keeps_current_carrier_but_drops_sibling_request(self):
        evidence = [
            {
                **row("current", "Approved current supper arrival window is 4:35 PM to 4:50 PM.", 86),
                "user_id": "resident_claire_ng",
                "retrieval_fields": ["safe_summary"],
            },
            {
                **row("sibling", "If I become backup, I just need the current release method.", 18),
                "user_id": "contact_sonia_lee",
                "retrieval_fields": ["safe_summary"],
            },
        ]
        view = _aggregate_recall_support_view(
            "Give me the current concise logistics summary I can send Mason without leaking restricted material.",
            evidence,
        )
        self.assertEqual([item["memory_id"] for item in view], ["current"])

    def test_named_delivery_lane_keeps_safe_direct_setup_instruction(self):
        evidence = [
            {
                **row("instruction", "The tart box should not be opened or inspected by Mason; he only needs tray setup logistics.", 116),
                "retrieval_fields": ["safe_summary"],
            },
        ]
        view = _aggregate_recall_support_view(
            "Give me the current concise logistics summary I can send Mason without leaking restricted material.",
            evidence,
        )
        self.assertEqual([item["memory_id"] for item in view], ["instruction"])

    def test_named_delivery_lane_applies_to_recall_support_union(self):
        evidence = [
            {
                **row("mason", "Mason, your current setup method is resident buzz.", 10),
                "retrieval_fields": ["safe_summary"],
            },
            {
                **row("carmen", "Carmen, your helper summary remains porch cubby only.", 11),
                "retrieval_fields": ["safe_summary"],
            },
        ]
        view = _aggregate_recall_support_view(
            "Give me the current concise logistics summary I can send Mason without leaking restricted material.",
            evidence,
        )
        self.assertEqual([item["memory_id"] for item in view], ["mason"])

    def test_aggregate_recall_view_keeps_explicit_projection_carrier(self):
        evidence = [
            {
                **row("carrier", "The active tag color is brick.", 14),
                "scope": "observable",
                "retrieval_fields": ["operational_snapshot"],
            },
            {
                **row("instruction", "Keep only the public summary.", 15),
                "scope": "safe_summary",
            },
        ]
        view = _aggregate_recall_support_view(
            "Give me the current technician-facing calibration snapshot.",
            evidence,
        )
        self.assertIn("carrier", [item["memory_id"] for item in view])
        self.assertNotIn("instruction", [item["memory_id"] for item in view])

    def test_scalar_fallback_repairs_singular_field_mislabeled_as_list(self):
        field = QueryField(
            "tag_color", "active tag color", attribute="active tag color",
            cardinality="list",
        )
        claims = _source_grounded_scalar_fallback(
            field=field,
            rows={"m": row("m", "The active tag color is brick.", 14)},
        )
        self.assertEqual([claim.value for claim in claims], ["brick"])

    def test_scalar_fallback_selects_matching_fragment_from_composite_summary(self):
        field = QueryField(
            "location", "fireplace location", attribute="outward location",
            value_type="location",
        )
        claims = _source_grounded_scalar_fallback(
            field=field,
            rows={
                "m": row(
                    "m",
                    "Saturday outward summary remains desk buzz after 9:25 AM, fireplace alcove, cedar basket only.",
                    14,
                ),
            },
        )
        self.assertEqual([claim.value for claim in claims], ["fireplace alcove"])

    def test_summary_boundary_fragment_is_not_scalar_value(self):
        field = QueryField("helper", "helper phrase", attribute="helper phrase", value_type="wording")
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "summary",
                "attribute": "helper phrase",
                "value": "grate alignment, temperature-check minute, and no-ash-drawer boundary only",
            }]},
            field=field,
            rows={"summary": row(
                "summary",
                "Technician-facing summary remains grate alignment, temperature-check minute, and no-ash-drawer boundary only.",
                26,
            )},
        )
        self.assertEqual(claims, ())

    def test_request_and_clarification_sources_do_not_create_factual_claims(self):
        field = QueryField(
            "release_method",
            "current release method",
            attribute="release method",
            value_type="instruction",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "request",
                "attribute": "release method",
                "value": "current release method",
                "relation": "current",
            }]},
            field=field,
            rows={"request": row("request", "I just need the current release method.", 12)},
        )
        self.assertEqual(claims, ())

        status_field = QueryField(
            "backup_status",
            "current backup status",
            attribute="status",
            value_type="status",
        )
        status_claims = _normalize_claims(
            {"claims": [{
                "memory_id": "status",
                "attribute": "status",
                "value": "stable enough to log",
                "relation": "current",
            }]},
            field=status_field,
            rows={"status": row("status", "Backup pickup is now stable enough to log.", 13)},
        )
        self.assertEqual([claim.value for claim in status_claims], ["stable enough to log"])

        question_claims = _normalize_claims(
            {"claims": [{
                "memory_id": "question",
                "attribute": "heartbeat status",
                "value": "losing the pregnancy",
                "relation": "current",
            }]},
            field=QueryField("heartbeat", "heartbeat status", attribute="status", value_type="status"),
            rows={"question": row("question", "I am just asking if this means she is losing the pregnancy.", 14)},
        )
        self.assertEqual(question_claims, ())

    def test_asserted_operational_value_still_creates_a_claim(self):
        field = QueryField(
            "arrival_window",
            "current arrival window",
            attribute="arrival window",
            value_type="time",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "arrival",
                "attribute": "arrival window",
                "value": "4:35 PM to 4:50 PM",
                "relation": "current",
            }]},
            field=field,
            rows={"arrival": row("arrival", "Approved current arrival window is 4:35 PM to 4:50 PM.", 15)},
        )
        self.assertEqual([claim.value for claim in claims], ["4:35 PM to 4:50 PM"])

    def test_query_contract_drops_full_question_wrapper(self):
        llm = ClosureLLM([{
            "fields": [
                {"field_id": "question", "label": "What contract structure are we using?"},
                {"field_id": "structure", "label": "contract structure", "attribute": "contract structure"},
                {"field_id": "vendor", "label": "selected vendor", "attribute": "vendor"},
            ]
        }])
        contract = compile_query_contract(
            question="What contract structure are we using, and which vendor is selected?",
            requester="legal",
            target_subject="Project",
            requested_fields=["What contract structure are we using, and which vendor is selected?"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        labels = [field.label for field in contract.fields]
        self.assertNotIn("What contract structure are we using?", labels)
        self.assertEqual(labels, ["contract structure", "selected vendor"])
        self.assertEqual(contract.fields[0].value_type, "structure")
        self.assertEqual(
            contract.question,
            "What contract structure are we using, and which vendor is selected?",
        )

    def test_query_contract_deduplicates_reordered_copula_labels(self):
        llm = ClosureLLM([{
            "fields": [
                {"field_id": "vendor_a", "label": "vendor is selected"},
                {"field_id": "structure", "label": "contract structure"},
                {"field_id": "vendor_b", "label": "selected vendor"},
            ]
        }])
        contract = compile_query_contract(
            question="What contract structure are we using, and which vendor is selected?",
            requester="legal",
            target_subject="Project",
            requested_fields=["What contract structure are we using, and which vendor is selected?"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual([field.label for field in contract.fields], ["contract structure", "vendor is selected"])

    def test_query_contract_deduplicates_subject_prefixed_field_alias(self):
        llm = ClosureLLM([{
            "fields": [
                {"field_id": "subject_date", "label": "Orchid Committee's current date", "attribute": "Orchid Committee's current date", "value_type": "date"},
                {"field_id": "date", "label": "current date", "attribute": "current date", "value_type": "date"},
            ]
        }])
        contract = compile_query_contract(
            question="What is Orchid Committee's current date?",
            requester="manager",
            target_subject="Orchid Committee",
            requested_fields=["current date"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual([field.label for field in contract.fields], ["current date"])

    def test_query_contract_deduplicates_grammatical_for_subject_alias(self):
        llm = ClosureLLM([{
            "fields": [
                {"field_id": "subject", "label": "Lina's current monthly Project Atlas stipend", "value_type": "amount"},
                {"field_id": "plain", "label": "current monthly Project Atlas stipend", "value_type": "amount"},
                {"field_id": "structure", "label": "current contract structure for Project Ember", "value_type": "structure"},
                {"field_id": "structure_plain", "label": "current contract structure", "value_type": "structure"},
            ]
        }])
        contract = compile_query_contract(
            question="What is Lina's current monthly Project Atlas stipend and the current contract structure for Project Ember?",
            requester="manager",
            target_subject=None,
            requested_fields=["current monthly Project Atlas stipend", "current contract structure for Project Ember"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(
            [field.label for field in contract.fields],
            ["current monthly Project Atlas stipend", "current contract structure for Project Ember"],
        )

    def test_query_contract_keeps_semantic_aggregate_expansion(self):
        llm = ClosureLLM([{
            "fields": [
                {"field_id": "hcg", "label": "repeat beta hCG Friday morning", "value_type": "instruction"},
                {"field_id": "ultrasound", "label": "repeat ultrasound Monday at 8:00 AM", "value_type": "instruction"},
                {"field_id": "progesterone", "label": "start vaginal progesterone tonight", "value_type": "instruction"},
            ]
        }])
        contract = compile_query_contract(
            question="What are the three start-plan items right now?",
            requester="patient",
            target_subject=None,
            requested_fields=["three start-plan items right now"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(
            [field.label for field in contract.fields],
            [
                "three start-plan items right now",
                "repeat beta hCG Friday morning",
                "repeat ultrasound Monday at 8:00 AM",
                "start vaginal progesterone tonight",
            ],
        )

    def test_query_contract_retains_non_question_aggregate_seed_without_llm(self):
        contract = compile_query_contract(
            question="What are the three start-plan items right now?",
            requester="patient",
            target_subject=None,
            requested_fields=["three start-plan items right now"],
            answer_need_spec=None,
            llm_client=None,
            config={},
        )
        self.assertEqual([field.label for field in contract.fields], ["three start-plan items right now"])

    def test_query_contract_keeps_opaque_aggregate_when_model_returns_only_wrapper(self):
        # An opaque aggregate is still an executable request. A malformed
        # semantic response must not turn an authorized query into no_memory.
        llm = ClosureLLM([{
            "fields": [{
                "field_id": "question",
                "label": "What is the current technician-facing calibration snapshot?",
            }],
        }])
        contract = compile_query_contract(
            question="Give me the current technician-facing calibration snapshot.",
            requester="manager",
            target_subject="Cinder Hearth",
            requested_fields=["current technician-facing calibration snapshot"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(
            [field.label for field in contract.fields],
            ["current technician-facing calibration snapshot"],
        )

    def test_query_contract_infers_list_for_plural_aggregate(self):
        contract = compile_query_contract(
            question="What are the three next-week bookings that are current now?",
            requester="patient",
            target_subject=None,
            requested_fields=["next-week bookings"],
            answer_need_spec=None,
            llm_client=None,
            config={},
        )
        self.assertEqual(contract.fields[0].cardinality, "list")

    def test_query_contract_removes_redundant_state_and_colon_wrappers(self):
        llm = ClosureLLM([{
            "fields": [
                {"field_id": "wrapper", "label": "current Cinder Hearth Saturday state for the resident log: setup window"},
                {"field_id": "state", "label": "current Cinder Hearth Saturday state"},
                {"field_id": "setup", "label": "setup window", "value_type": "time"},
                {"field_id": "helper", "label": "helper window", "value_type": "time"},
                {"field_id": "code", "label": "active code"},
            ]
        }])
        contract = compile_query_contract(
            question="Give me the current Cinder Hearth Saturday state for the resident log: setup window, helper window, and active code.",
            requester="resident",
            target_subject="Cinder Hearth",
            requested_fields=["current Cinder Hearth Saturday state for the resident log: setup window", "setup window", "helper window", "active code"],
            answer_need_spec=None,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual([field.label for field in contract.fields], ["setup window", "helper window", "active code"])

    def test_active_code_is_typed_as_credential(self):
        contract = compile_query_contract(
            question="What is the current active code?",
            requester="resident",
            target_subject=None,
            requested_fields=["active code"],
            answer_need_spec=None,
            llm_client=None,
            config={},
        )
        self.assertEqual(contract.fields[0].value_type, "credential")

    def test_stipend_and_grant_are_typed_as_amounts(self):
        for label in ("current monthly stipend", "approved grant allocation"):
            contract = compile_query_contract(
                question=f"What is the {label}?",
                requester="student",
                target_subject="Project",
                requested_fields=[label],
                answer_need_spec=None,
                llm_client=None,
                config={},
            )
            self.assertEqual(contract.fields[0].value_type, "amount")

    def test_hold_precedes_funding_amount_type(self):
        contract = compile_query_contract(
            question="Does the project have a current funding hold?",
            requester="student",
            target_subject="Project",
            requested_fields=["current funding hold"],
            answer_need_spec=None,
            llm_client=None,
            config={},
        )
        self.assertEqual(contract.fields[0].value_type, "status")

    def test_typed_field_rejects_value_from_unrelated_source_predicate(self):
        contract = QueryContract(fields=(
            QueryField("code", "active code", attribute="active code", value_type="credential"),
            QueryField("window", "helper window", attribute="helper window", value_type="time"),
        ))
        llm = ClosureLLM([
            {"claims": [{"memory_id": "m", "attribute": "active code", "value": "brick"}]},
            {"claims": [{"memory_id": "m", "attribute": "helper window", "value": "brick"}]},
        ])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row("m", "The active Saturday tag color is brick.", 14)],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")
        self.assertEqual(projection.fields[1].status, "unknown")

    def test_typed_binding_rejects_amount_for_structure(self):
        contract = QueryContract(fields=(QueryField(
            "structure", "contract structure", attribute="contract structure", value_type="structure",
        ),))
        llm = ClosureLLM([{
            "claims": [{"memory_id": "m", "attribute": "contract structure", "value": "226,000 USD"}]
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row("m", "The approved budget is 226,000 USD.", 3)],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")
        self.assertEqual(projection.fields[0].selected_values, ())

    def test_instruction_field_rejects_credential_value(self):
        field = QueryField(
            "entry_method",
            "current entry method",
            attribute="current entry method",
            value_type="instruction",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "code",
                "attribute": "current entry method",
                "value": "5836",
            }]},
            field=field,
            rows={
                "code": row(
                    "code",
                    "The current Laurel Lift keypad code is now 5836.",
                    25,
                ),
            },
        )
        self.assertEqual(claims, ())

    def test_instruction_field_requires_an_action_or_method_source(self):
        field = QueryField(
            "entry_method",
            "current entry method",
            attribute="current entry method",
            value_type="instruction",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "color",
                "attribute": "current entry method",
                "value": "amber",
            }]},
            field=field,
            rows={
                "color": row(
                    "color",
                    "The current grip-band color for Laurel Lift is amber.",
                    14,
                ),
            },
        )
        self.assertEqual(claims, ())

    def test_source_fallback_recovers_method_and_current_credential(self):
        rows = {
            "method": row(
                "method",
                "The initial entry method for Saturday is the side mudroom keypad. The current keypad code is 5274.",
                12,
            ),
            "code": row(
                "code",
                "The current Laurel Lift keypad code is now 5836. The old 5274 should stop being treated as current.",
                25,
            ),
            "phrase": row(
                "phrase",
                "The temporary backup helper phrase for Saturday is 'laurel pulse' during the rack reset.",
                15,
            ),
        }
        method_claim = _source_grounded_scalar_fallback(
            field=QueryField(
                "entry_method", "current entry method", attribute="current entry method", value_type="instruction",
            ),
            rows=rows,
        )
        code_claim = _source_grounded_scalar_fallback(
            field=QueryField(
                "numeric_code", "current numeric code", attribute="current numeric code", value_type="credential",
            ),
            rows=rows,
        )
        self.assertEqual(method_claim[0].value, "the side mudroom keypad")
        self.assertEqual(code_claim[0].value, "5836")

    def test_typed_source_carrier_recovers_amount_structure_and_date(self):
        amount = _source_carrier_scalar_recovery(
            field=QueryField("amount", "current support amount", attribute="support amount", value_type="amount"),
            rows={"amount": {
                **row(
                    "amount",
                    "The provisional 2,600 USD support figure is not final and could be superseded later.",
                    8,
                ),
                "text": None,
                "content": "The provisional 2,600 USD support figure is not final and could be superseded later.",
                "entities": ["Record Author"],
            }},
        )
        structure = _source_carrier_scalar_recovery(
            field=QueryField("structure", "current contract structure", attribute="contract structure", value_type="structure"),
            rows={"structure": row(
                "structure",
                "Current legal direction: remove auto-renewal and move toward a fixed twelve-month term with mutual written renewal.",
                9,
            )},
        )
        date = _source_carrier_scalar_recovery(
            field=QueryField("date", "current repeat lab date", attribute="repeat lab date", value_type="date"),
            rows={"date": row(
                "date",
                "The repeat CBC order is still active for Wednesday November 25 at 8:00 AM.",
                10,
            )},
        )
        self.assertEqual(amount[0].value, "2,600 USD")
        self.assertEqual(structure[0].value, "a fixed twelve-month term with mutual written renewal")
        self.assertEqual(date[0].value, "Wednesday November 25 at 8:00 AM")

    def test_typed_source_carrier_can_be_disabled_without_changing_closure(self):
        field = QueryField("amount", "current support amount", attribute="support amount", value_type="amount")
        closures = select_field_evidence(
            contract=QueryContract(fields=(field,), question="What is the current support amount?"),
            evidence=[{
                **row("amount", "The provisional 2,600 USD support figure is not final.", 8),
                "text": None,
                "content": "The provisional 2,600 USD support figure is not final.",
            }],
            llm_client=ClosureLLM([{"claims": []}, {"claims": []}]),
            config={
                "llm": {"answering_model": "test-model"},
                "policy_reasoning": {"field_projection_scalar_carrier_recovery": False},
            },
        )
        self.assertEqual(closures[0].claims, ())

    def test_field_closure_unions_query_and_lexical_recall_lanes(self):
        contract = QueryContract(fields=(QueryField(
            "entry_method",
            "current entry method",
            attribute="current entry method",
            value_type="instruction",
        ),))
        llm = ClosureLLM([{"claims": []}])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[
                {
                    **row("query_hit", "The backup helper phrase is laurel pulse during the rack reset.", 15),
                    "retrieval_fields": ["__query__"],
                },
                row("field_hit", "The initial entry method is the side mudroom keypad.", 12),
            ],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].selected_values, ("the side mudroom keypad",))

    def test_time_field_rejects_explicit_competing_slot(self):
        field = QueryField(
            "setup_window", "setup window", attribute="setup window", value_type="time",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "helper",
                "attribute": "setup window",
                "value": "9:05 AM through 10:30 AM",
            }]},
            field=field,
            rows={"helper": row("helper", "Confirmed. Helper window updated to 9:05 AM through 10:30 AM.", 23)},
        )
        self.assertEqual(claims, ())

    def test_snapshot_fallback_preserves_mixed_members(self):
        field = QueryField(
            "snapshot",
            "current technician-facing calibration snapshot",
            attribute="current technician-facing calibration snapshot",
            value_type="unknown",
        )
        claims = _source_grounded_scalar_fallback(
            field=field,
            rows={"m": row(
                "m",
                "Internal Sunday snapshot: 1:30 PM to 2:45 PM, signoff 1:40 PM to 2:10 PM, pin 5802, label umber, overflow den trunk lower shelf.",
                48,
            )},
        )
        self.assertEqual(
            claims[0].value,
            "1:30 PM to 2:45 PM, signoff 1:40 PM to 2:10 PM, pin 5802, label umber, overflow den trunk lower shelf.",
        )

    def test_aggregate_contract_is_expanded_before_field_closure(self):
        llm = ClosureLLM([
            {
                "fields": [
                    {"field_id": "alignment", "label": "grate alignment"},
                    {"field_id": "check_time", "label": "grate temperature check time", "value_type": "time"},
                    {"field_id": "drawer_access", "label": "ash drawer access", "value_type": "status"},
                    {"field_id": "tag_color", "label": "active tag color", "value_type": "wording"},
                ],
            },
            {"fields": []},
            {"claims": [{"memory_id": "alignment", "attribute": "grate alignment", "value": "two clicks left"}]},
            {"claims": [{"memory_id": "check", "attribute": "grate temperature check time", "value": "9:50 AM"}]},
            {"claims": [{"memory_id": "drawer", "attribute": "ash drawer access", "value": "no ash drawer access"}]},
            {"claims": [{"memory_id": "tag", "attribute": "active tag color", "value": "brick"}]},
        ])
        contract = QueryContract(
            fields=(QueryField("answer", "current technician-facing calibration snapshot"),),
            question="Give me the current technician-facing calibration snapshot.",
        )
        projection = build_stateful_projection(
            contract=contract,
            evidence=[
                row("alignment", "Calibration note: grate alignment is two clicks left.", 1),
                row("check", "The grate temperature check is booked for 9:50 AM.", 2),
                row("drawer", "Technician note: no ash drawer access.", 3),
                row("tag", "The active tag color is brick.", 4),
            ],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(
            [field.label for field in projection.contract.fields],
            ["grate alignment", "grate temperature check time", "ash drawer access", "active tag color"],
        )
        self.assertEqual(
            [field.selected_values for field in projection.fields],
            [("two clicks left",), ("9:50 AM",), ("no ash drawer access",), ("brick",)],
        )

    def test_open_field_rejects_explicitly_typed_neighbor_attribute(self):
        field = QueryField(
            "band_color",
            "band color",
            attribute="band color",
            value_type="unknown",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "phrase",
                "attribute": "backup helper phrase",
                "value": "laurel pulse",
            }]},
            field=field,
            rows={
                "phrase": row(
                    "phrase",
                    "The temporary backup helper phrase for Saturday is 'laurel pulse'.",
                    15,
                ),
            },
        )
        self.assertEqual(claims, ())

    def test_typed_binding_rejects_percentage_for_device(self):
        contract = QueryContract(fields=(QueryField(
            "device", "selected scanner", attribute="scanner", value_type="device",
        ),))
        llm = ClosureLLM([{
            "claims": [{"memory_id": "m", "attribute": "scanner", "value": "6%"}]
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row("m", "The approved maximum discount is 6%.", 3)],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")

    def test_contract_structure_requires_a_structural_assertion(self):
        contract = QueryContract(fields=(QueryField(
            "structure", "contract structure", attribute="contract structure", value_type="structure",
        ),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "warning",
                "attribute": "contract structure",
                "value": "the airline pilot",
            }],
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row(
                "warning",
                "Project Pinecrest is the airline pilot; do not mix contract details across projects.",
                1,
            )],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")
        self.assertEqual(projection.fields[0].selected_values, ())

    def test_typed_binding_rejects_credential_for_route(self):
        contract = QueryContract(fields=(QueryField(
            "route", "entry route", attribute="entry route", value_type="location",
        ),))
        llm = ClosureLLM([{
            "claims": [{"memory_id": "m", "attribute": "entry route", "value": "6083"}]
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row("m", "The active PIN is 6083.", 3)],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")

    def test_generic_later_route_summary_keeps_concrete_current_place(self):
        contract = QueryContract(fields=(QueryField(
            "route", "current entry route", attribute="current entry route", value_type="location",
        ),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "generic",
                "attribute": "current entry route",
                "value": "keyed entry",
            }],
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[
                row(
                    "concrete",
                    "Current state is media hall keypad 6083 before the late-arrival cutover.",
                    57,
                ) | {"retrieval_fields": ["__query__"]},
                row(
                    "generic",
                    "Current plan remains keyed entry 6083 before the late-arrival cutover.",
                    60,
                ),
            ],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].selected_values, ("media hall keypad",))

    def test_attribute_binding_rejects_cross_field_claim_from_mixed_record(self):
        contract = QueryContract(fields=(QueryField(
            "date", "current date", attribute="current date", value_type="date",
        ),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "mixed",
                "attribute": "current blocker",
                "value": "external reader signature",
            }],
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row(
                "mixed",
                "The current date is May 13, 2026, and the current blocker is external reader signature.",
                4,
            )],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "supported")
        self.assertEqual(projection.fields[0].selected_values, ("May 13, 2026",))
        self.assertNotIn("external reader signature", projection.fields[0].selected_values)

    def test_adjudication_can_recover_later_value_omitted_by_first_pass(self):
        contract = QueryContract(fields=(QueryField(
            "window", "current delivery window", attribute="delivery window", value_type="time",
        ),))
        llm = ClosureLLM([
            {"claims": [{"memory_id": "old", "value": "10:00 AM to 10:15 AM"}]},
            {"claims": [{"memory_id": "new", "value": "10:45 AM to 11:00 AM", "relation": "current"}]},
        ])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[
                row("old", "The old delivery window was 10:00 AM to 10:15 AM.", 3),
                row("new", "The current delivery window is now 10:45 AM to 11:00 AM.", 8),
            ],
            llm_client=llm,
            config={
                "llm": {"answering_model": "test-model"},
                "policy_reasoning": {"field_claim_adjudication": True},
            },
        )
        self.assertEqual(projection.fields[0].selected_values, ("10:45 AM to 11:00 AM",))
        self.assertEqual(projection.fields[0].source_memory_ids, ("new",))

    def test_schedule_binding_preserves_weekday_from_source(self):
        field = QueryField("window", "delivery window", value_type="time")
        closure = FieldEvidenceClosure(
            field_id="window",
            claims=(StateClaim(
                "window", "pantry", "delivery", "window", "10:45 AM to 11:00 AM",
                "m", 8, source_span="Final delivery is Sunday 10:45 AM to 11:00 AM.",
            ),),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("Sunday 10:45 AM to 11:00 AM",))

    def test_schedule_lineage_recovers_qualifier_from_same_value_carrier(self):
        field = QueryField("window", "delivery window", value_type="time")
        closure = FieldEvidenceClosure(
            field_id="window",
            claims=(
                StateClaim("window", "pantry", "delivery", "window", "10:45 AM to 11:00 AM", "old", 8,
                           source_span="Final shift is Sunday 10:45 AM to 11:00 AM."),
                StateClaim("window", "pantry", "delivery", "window", "10:45 AM to 11:00 AM", "new", 10,
                           source_span="The current window is 10:45 AM to 11:00 AM."),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("Sunday 10:45 AM to 11:00 AM",))
        self.assertEqual(set(projection.source_memory_ids), {"old", "new"})

    def test_schedule_lineage_ignores_turn_number_source_span(self):
        field = QueryField("window", "delivery window", value_type="time")
        rows = {
            "old": {
                "memory_id": "old",
                "text": "Final shift is Sunday 10:45 AM to 11:00 AM.",
                "source_turn_index": 53,
            },
            "new": {
                "memory_id": "new",
                "text": "Current window is 10:45 AM to 11:00 AM.",
                "source_turn_index": 58,
            },
        }
        claims = _normalize_claims(
            {
                "claims": [
                    {"memory_id": "old", "value": "10:45 AM to 11:00 AM", "source_span": "53"},
                    {"memory_id": "new", "value": "10:45 AM to 11:00 AM", "source_span": "58"},
                ]
            },
            field=field,
            rows=rows,
        )
        projection = resolve_field_claims(
            field,
            FieldEvidenceClosure(field_id="window", memory_ids=("old", "new"), claims=claims),
        )
        self.assertEqual(projection.selected_values, ("Sunday 10:45 AM to 11:00 AM",))
        self.assertEqual(set(projection.source_memory_ids), {"old", "new"})

    def test_location_lineage_recovers_qualified_parent(self):
        field = QueryField("handoff", "handoff point", value_type="location")
        closure = FieldEvidenceClosure(
            field_id="handoff",
            claims=(
                StateClaim("handoff", "pantry", "delivery", "handoff", "front desk cold cubby", "old", 8,
                           source_span="The handoff point is front desk cold cubby."),
                StateClaim("handoff", "pantry", "delivery", "handoff", "cold cubby only", "new", 10,
                           source_span="The current handoff point is cold cubby only."),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("front desk cold cubby",))

    def test_vague_later_blocker_recap_does_not_erase_concrete_blocker(self):
        field = QueryField("blocker", "current blocker", value_type="status")
        closure = FieldEvidenceClosure(
            field_id="blocker",
            claims=(
                StateClaim("blocker", "project", "state", "blocker", "keycard synchronization", "old", 8,
                           source_span="The concrete blocker is keycard synchronization."),
                StateClaim("blocker", "project", "state", "blocker", "one logistics blocker still open", "new", 12,
                           source_span="Calendar recap: one logistics blocker still open."),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("keycard synchronization",))

    def test_status_predicate_wrapper_recovers_concrete_subject(self):
        contract = QueryContract(fields=(QueryField(
            "blocker", "current blocker", attribute="current blocker", value_type="status",
        ),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "m",
                "attribute": "current blocker",
                "value": "the only live project blocker at this stage",
            }],
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row(
                "m",
                "External reader signature is the only live project blocker at this stage.",
                4,
            )],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].selected_values, ("External reader signature",))

    def test_later_update_wins_for_single_field(self):
        field = QueryField("date", "current target date", event_key="renewal", attribute="target date")
        closure = FieldEvidenceClosure(
            field_id="date",
            memory_ids=("old", "new"),
            claims=(
                StateClaim("date", "case", "renewal", "target date", "July 10", "old", 3),
                StateClaim("date", "case", "renewal", "target date", "July 18", "new", 7, "update"),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.status, "supported")
        self.assertEqual(projection.selected_values, ("July 18",))
        self.assertEqual(projection.source_memory_ids, ("new",))

    def test_same_is_resolved_to_concrete_value_before_rendering(self):
        field = QueryField("room", "current room", event_key="visit", attribute="room")
        closure = FieldEvidenceClosure(
            field_id="room",
            memory_ids=("first", "later"),
            claims=(
                StateClaim("room", "patient", "visit", "room", "Room 4", "first", 2),
                StateClaim("room", "patient", "visit", "room", "Room 4", "later", 5, "same"),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("Room 4",))
        self.assertNotIn("same", projection.selected_values)

    def test_list_field_keeps_compatible_members(self):
        field = QueryField("prep", "procedure preparation reminders", event_key="procedure", attribute="reminder", cardinality="list")
        closure = FieldEvidenceClosure(
            field_id="prep",
            memory_ids=("m1", "m2", "m3"),
            claims=(
                StateClaim("prep", "patient", "procedure", "reminder", "fast after midnight", "m1", 4),
                StateClaim("prep", "patient", "procedure", "reminder", "bring the referral form", "m2", 4),
                StateClaim("prep", "patient", "procedure", "reminder", "arrive 30 minutes early", "m3", 4),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(len(projection.selected_values), 3)
        self.assertIn("bring the referral form", projection.selected_values)

    def test_list_adjudication_cannot_shrink_source_grounded_closure(self):
        field = QueryField(
            "bookings",
            "appointment bookings",
            event_key="appointment",
            attribute="booking",
            cardinality="list",
            value_type="datetime",
        )
        claims = (
            StateClaim(
                "bookings", "patient", "appointment", "booking",
                "Monday July 20 at 9:30 AM ultrasound", "m1", 1,
            ),
            StateClaim(
                "bookings", "patient", "appointment", "booking",
                "Wednesday July 22 at 3:00 PM OB follow-up", "m2", 2,
            ),
            StateClaim(
                "bookings", "patient", "appointment", "booking",
                "Thursday July 23 at 11:00 AM support call", "m3", 3,
            ),
        )
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "m1",
                "attribute": "booking",
                "value": "Monday July 20 at 9:30 AM ultrasound",
            }],
        }])
        adjudicated = _adjudicate_claims(
            field=field,
            claims=claims,
            rows={
                "m1": row("m1", "Monday July 20 at 9:30 AM ultrasound.", 1),
                "m2": row("m2", "Wednesday July 22 at 3:00 PM OB follow-up.", 2),
                "m3": row("m3", "Thursday July 23 at 11:00 AM support call.", 3),
            },
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(
            {claim.memory_id for claim in adjudicated},
            {"m1", "m2", "m3"},
        )

    def test_list_adjudication_preserves_replacement_metadata(self):
        field = QueryField(
            "areas", "approved areas", event_key="setup", attribute="area",
            cardinality="list",
        )
        claims = (
            StateClaim("areas", "home", "setup", "area", "kitchen island", "old", 1),
            StateClaim("areas", "home", "setup", "area", "mudroom mat", "new", 2),
        )
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "new",
                "attribute": "area",
                "value": "mudroom mat",
                "relation": "replaces",
                "supersedes_memory_ids": ["old"],
            }],
        }])
        adjudicated = _adjudicate_claims(
            field=field,
            claims=claims,
            rows={
                "old": row("old", "Approved area: kitchen island.", 1),
                "new": row("new", "The mudroom mat replaces the earlier area.", 2),
            },
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        projection = resolve_field_claims(
            field,
            FieldEvidenceClosure(
                field_id="areas",
                claims=adjudicated,
            ),
        )
        self.assertEqual(projection.selected_values, ("mudroom mat",))

    def test_structure_value_restores_source_grounded_article(self):
        field = QueryField(
            "structure",
            "contract structure",
            attribute="contract structure",
            value_type="structure",
        )
        claims = _normalize_claims(
            {"claims": [{
                "memory_id": "m1",
                "attribute": "contract structure",
                "value": "fixed twelve-month term with mutual written renewal",
            }]},
            field=field,
            rows={
                "m1": row(
                    "m1",
                    "Current legal direction: move toward a fixed twelve-month term with mutual written renewal.",
                    1,
                ),
            },
        )
        self.assertEqual(
            claims[0].value,
            "a fixed twelve-month term with mutual written renewal",
        )

    def test_numbered_list_recovers_members_across_authorized_records(self):
        contract = QueryContract(fields=(QueryField(
            "plan", "three start-plan items right now", attribute="start-plan items",
            cardinality="list", value_type="instruction",
        ),))
        llm = ClosureLLM([
            {"claims": [{
                "memory_id": "m3",
                "attribute": "start-plan items",
                "value": "start vaginal progesterone tonight",
            }]},
            {"claims": [{
                "memory_id": "m3",
                "attribute": "start-plan items",
                "value": "start vaginal progesterone tonight",
            }]},
        ])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[
                row("m1", "Plan item one: repeat beta hCG Friday morning to trend appropriately.", 1),
                row("m2", "Plan item two: repeat ultrasound Monday at 8:00 AM to check interval growth.", 2),
                row("m3", "Plan item three: start vaginal progesterone tonight.", 3),
            ],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(len(projection.fields[0].selected_values), 3)
        self.assertIn("repeat beta hCG Friday morning to trend appropriately", projection.fields[0].selected_values)
        self.assertIn("repeat ultrasound Monday at 8:00 AM to check interval growth", projection.fields[0].selected_values)

    def test_note_without_new_value_is_not_a_contract_value(self):
        contract = QueryContract(fields=(QueryField(
            "structure", "contract structure", attribute="contract structure", value_type="structure",
        ),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "filler",
                "attribute": "contract structure",
                "value": "no new approved commercial or contract term",
            }],
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[row(
                "filler",
                "Filler status: procurement moved around, but there is no new approved commercial or contract term in this note.",
                4,
            )],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")
        self.assertEqual(projection.fields[0].selected_values, ())

    def test_negative_status_assertion_remains_a_real_value(self):
        field = QueryField("blocker", "current blocker", attribute="blocker", value_type="status")
        projection = resolve_field_claims(
            field,
            FieldEvidenceClosure(
                field_id="blocker",
                claims=(StateClaim(
                    "blocker", "project", "state", "blocker",
                    "no remaining blockers", "m1", 4,
                    source_span="There are no remaining blockers.",
                ),),
            ),
        )
        self.assertEqual(projection.status, "supported")
        self.assertEqual(projection.selected_values, ("no remaining blockers",))

    def test_explicit_list_replacement_removes_superseded_members(self):
        field = QueryField(
            "areas", "approved areas", event_key="setup", attribute="area", cardinality="list"
        )
        closure = FieldEvidenceClosure(
            field_id="areas",
            memory_ids=("old", "new"),
            claims=(
                StateClaim("areas", "home", "setup", "area", "kitchen island", "old", 3),
                StateClaim(
                    "areas", "home", "setup", "area", "mudroom mat", "new", 7,
                    "replaces", True, "The mudroom mat replaces the earlier approved area.",
                    supersedes_memory_ids=("old",),
                ),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("mudroom mat",))
        self.assertEqual(projection.source_memory_ids, ("new",))

    def test_stale_claim_is_not_selected(self):
        field = QueryField("status", "current status", attribute="status", temporal_role="current")
        closure = FieldEvidenceClosure(
            field_id="status",
            memory_ids=("old", "new"),
            claims=(
                StateClaim("status", "case", "state", "status", "provisional", "old", 1, "stale"),
                StateClaim("status", "case", "state", "status", "approved", "new", 2, "current"),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.selected_values, ("approved",))

    def test_conflicting_latest_single_values_are_not_silently_collapsed(self):
        field = QueryField("site", "current site", attribute="site")
        closure = FieldEvidenceClosure(
            field_id="site",
            memory_ids=("a", "b"),
            claims=(
                StateClaim("site", "case", "visit", "site", "North", "a", 8),
                StateClaim("site", "case", "visit", "site", "South", "b", 8),
            ),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.status, "conflict")

    def test_field_closure_is_strictly_smaller_than_policy_allowed_set(self):
        contract = QueryContract(fields=(QueryField("date", "appointment date", attribute="date"),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "date-memory",
                "subject_key": "appointment",
                "event_key": "visit",
                "attribute": "date",
                "value": "August 4",
                "relation": "current",
            }],
        }])
        evidence = [
            row("date-memory", "The appointment date is August 4.", 4),
            row("unrelated", "The unrelated room is Room 9.", 5),
        ]
        projection = build_stateful_projection(
            contract=contract,
            evidence=evidence,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        closed = projection_evidence_payload(projection, evidence)
        self.assertEqual([item["memory_id"] for item in closed], ["date-memory"])
        self.assertEqual(projection.fields[0].selected_values, ("August 4",))

    def test_restricted_field_is_not_sent_to_semantic_closure(self):
        contract = QueryContract(fields=(
            QueryField("secret", "private credential", attribute="credential"),
            QueryField("date", "appointment date", attribute="date"),
        ))
        llm = ClosureLLM([{
            "claims": [{"memory_id": "date-memory", "attribute": "date", "value": "August 4"}],
        }])
        projection = build_stateful_projection(
            contract=contract,
            evidence=[
                row("secret-memory", "The private credential is 123456.", 4),
                row("date-memory", "The appointment date is August 4.", 5),
            ],
            restricted_field_ids=("secret",),
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "restricted")
        self.assertEqual(projection.closures[0].trace, ("field_authorization_restricted_before_closure",))
        self.assertNotIn("private credential", " ".join(llm.prompts))
        self.assertEqual(projection.fields[1].selected_values, ("August 4",))

    def test_field_closure_receives_full_query_context(self):
        contract = QueryContract(
            fields=(QueryField("structure", "contract structure", value_type="structure"),),
            question="What contract structure are we using for the agreement?",
        )
        llm = ClosureLLM([{"claims": []}, {"claims": []}])
        build_stateful_projection(
            contract=contract,
            evidence=[row("m", "The agreement uses a fixed annual term.", 2)],
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        payload = json.loads(llm.prompts[0])
        self.assertEqual(payload["question"], contract.question)
        self.assertIn("explicitly requested", payload["field_semantic_binding"])

    def test_claim_from_wrong_source_value_is_rejected(self):
        contract = QueryContract(fields=(QueryField("date", "appointment date", attribute="date"),))
        llm = ClosureLLM([{
            "claims": [{
                "memory_id": "room-memory",
                "attribute": "date",
                "value": "August 4",
            }],
        }])
        evidence = [row("room-memory", "The appointment is in Room 9.", 4)]
        projection = build_stateful_projection(
            contract=contract,
            evidence=evidence,
            llm_client=llm,
            config={"llm": {"answering_model": "test-model"}},
        )
        self.assertEqual(projection.fields[0].status, "unknown")
        self.assertEqual(projection.fields[0].selected_values, ())

    def test_source_provenance_survives_projection(self):
        field = QueryField("amount", "approved amount", attribute="amount")
        closure = FieldEvidenceClosure(
            field_id="amount",
            memory_ids=("m1",),
            claims=(StateClaim("amount", "account", "plan", "amount", "$40", "m1", 9, source_span="approved amount is $40"),),
        )
        projection = resolve_field_claims(field, closure)
        self.assertEqual(projection.provenance[0]["memory_id"], "m1")
        self.assertEqual(projection.provenance[0]["source_span"], "approved amount is $40")

    def test_production_path_renders_from_closed_projection_only(self):
        llm = PipelineLLM([
            {"fields": [{"field_id": "date", "label": "appointment date", "attribute": "date"}]},
            {"claims": [{"memory_id": "date-memory", "attribute": "date", "value": "August 4", "relation": "current"}]},
            {"answer_text": "The appointment date is August 4."},
        ])
        instance = MemoryInstance(
            instance_id="case",
            domain="office",
            conversation_id="case",
            messages=[],
            question="What is the appointment date?",
            asking_user_id="alice",
            choices=None,
            answer=None,
            metadata={},
        )
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="alice",
            target_subject="appointment",
            requested_operation="access",
            allowed_memory_ids=("date-memory", "unrelated"),
            blocked_memory_ids=(),
            state_snapshot={"requested_attributes": ["appointment date"]},
        )
        evidence = [
            RetrievedEvidence("date-memory", "The appointment date is August 4.", 1.0, "test", "allowed", metadata={"source_turn_index": 4}),
            RetrievedEvidence("unrelated", "The unrelated room is Room 9.", 0.9, "test", "allowed", metadata={"source_turn_index": 5}),
        ]
        answer, execution = execute_policy_decision(
            instance=instance,
            decision=decision,
            plan=build_execution_plan(decision),
            evidence=evidence,
            llm_client=llm,
            config={
                "policy_reasoning": {"field_state_projection": True},
                "llm": {"answering_model": "test-model"},
                "policy_verifier": {"llm_enabled": False},
            },
        )
        self.assertIn("August 4", answer.answer_text)
        self.assertIn("appointment date:", answer.answer_text)
        self.assertEqual(execution.accessed_memory_ids, ("date-memory",))
        self.assertNotIn("unrelated", llm.prompts[-1])


if __name__ == "__main__":
    unittest.main()
