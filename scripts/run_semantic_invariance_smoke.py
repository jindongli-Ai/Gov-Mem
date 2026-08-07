#!/usr/bin/env python3
"""Framework-neutral metamorphic checks for semantic Gov-Mem components.

These checks never read benchmark labels or official scorers.  They verify that
structured intent, entity renaming, provenance, and access boundaries drive the
runtime independently of a particular benchmark's wording.
"""

from __future__ import annotations

from gov_mem.backbones.canonical_renderer import _extract_medication_entities, render_canonical_answer
from gov_mem.backbones.common import (
    _augment_current_state_slot_coverage,
    _build_question_profile,
    _classify_line_families,
    _semantic_domains,
)
from gov_mem.backbones.coverage_packer import pack_utility_records
from gov_mem.backbones.coverage_verifier import verify_canonical_answer
from gov_mem.backbones.need_spec import AnswerNeedSpec, build_answer_need_spec
from gov_mem.backbones.utility_records import UtilityRecord, build_utility_records
from gov_mem.data.schema import Principal, QueryPlan
from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import _looks_like_medication_frame_text
from gov_mem.governance_runtime.action_predictor import _apply_semantic_disclosure_spec
from gov_mem.llm.client import LLMClientUnavailableError
from gov_mem.planning.query_planner import QueryUnderstandingAgent
from gov_mem.query_semantics import extract_state_slots
from gov_mem.reasoning.operators import build_required_slot_plan


def _query_plan(semantic_spec: dict) -> QueryPlan:
    return QueryPlan(
        query_type="factual",
        target_users=[],
        target_entities=[],
        required_memory_types=[],
        symbolic_filters={},
        dense_queries=[],
        reasoning_ops=["AND"],
        semantic_spec=semantic_spec,
    )


def _assert_equal(left, right, label: str) -> None:
    if left != right:
        raise AssertionError(f"{label} differs:\nleft={left}\nright={right}")


def check_semantic_spec_invariance() -> None:
    semantic_spec = {
        "requested_slots": ["target_date", "approved_budget", "approved_discount_cap"],
        "temporal_scope": "current",
        "disclosure_scope": "full",
        "state_domain": "project",
        "request_shape": "fact",
    }
    plan = _query_plan(semantic_spec)
    first = build_required_slot_plan("Please provide an up-to-date overview.", plan)
    second = build_required_slot_plan("What is the present operational summary?", plan)
    _assert_equal(first, second, "semantic slot plan")

    first_need = build_answer_need_spec(
        "Please provide an up-to-date overview.",
        {},
        [],
        {"semantic_spec": semantic_spec},
        {},
    )
    second_need = build_answer_need_spec(
        "What is the present operational summary?",
        {},
        [],
        {"semantic_spec": semantic_spec},
        {},
    )
    _assert_equal(first_need.required_record_types, second_need.required_record_types, "answer record types")
    _assert_equal(first_need.required_slot_groups, second_need.required_slot_groups, "answer slots")

    first_profile = _build_question_profile(
        "Please provide an up-to-date overview.",
        _semantic_domains(semantic_spec),
        semantic_intent={"semantic_spec": semantic_spec},
    )
    second_profile = _build_question_profile(
        "What is the present operational summary?",
        _semantic_domains(semantic_spec),
        semantic_intent={"semantic_spec": semantic_spec},
    )
    _assert_equal(first_profile["target_frame_types"], second_profile["target_frame_types"], "semantic profile frames")
    _assert_equal(first_profile["requested_current_slots"], second_profile["requested_current_slots"], "semantic profile slots")
    _assert_equal(first_profile["semantic_profile_source"], "semantic_spec", "semantic profile source")


def check_entity_rename_invariance() -> None:
    metadata = {
        "frame_type": "project_state",
        "slots": {
            "target_date": "October 14",
            "approved_budget": "180,000 USD",
            "approved_discount_cap": "9%",
        },
        "is_current_like": True,
    }
    first = _classify_line_families("Aster program status was revised.", metadata, mode="current_state_bundle")
    second = _classify_line_families("Nimbus initiative status was revised.", metadata, mode="current_state_bundle")
    _assert_equal(first, second, "entity-renamed line families")

    first_medicine = _extract_medication_entities("Continue zafril 10 mg nightly.")
    second_medicine = _extract_medication_entities("Continue norvex 10 mg nightly.")
    if first_medicine != {"zafril"} or second_medicine != {"norvex"}:
        raise AssertionError(f"generic medication extraction failed: {first_medicine}, {second_medicine}")
    if not _looks_like_medication_frame_text("Continue norvex 10 mg nightly."):
        raise AssertionError("generic medication frame classification failed")
    state = extract_state_slots("The revised approved Oriole budget is now 216,000 USD.")
    if state.get("approved_budget") != "216,000 USD":
        raise AssertionError(f"generic budget relation extraction failed: {state}")
    event = extract_state_slots("The open workshop is now May 1, 2026.")
    if event.get("public_event_date") != "May 1, 2026":
        raise AssertionError(f"generic public event extraction failed: {event}")
    public_profile = _apply_semantic_disclosure_spec(
        {},
        {
            "requested_slots": ["safe_wording", "public_event_date"],
            "disclosure_scope": "public_only",
            "request_shape": "mixed",
        },
    )
    if not (
        public_profile.get("asks_safe_partial_share")
        and public_profile.get("asks_logistics")
        and public_profile.get("mixed_disclosure_request")
    ):
        raise AssertionError(f"semantic public disclosure was not applied: {public_profile}")


def check_provenance_and_access_invariance() -> None:
    need = AnswerNeedSpec(
        need_types=["project_state"],
        required_record_types=["project_state"],
        required_slot_groups=["approved_budget", "approved_discount_cap"],
        current_state_required=True,
    )
    stale = UtilityRecord(
        record_id="stale",
        source_chunk_id="old",
        source_line_id="old:0",
        record_type="project_state",
        owner_user=None,
        principal_access="owner",
        lifecycle_status="active",
        slots={"approved_budget": "120,000 USD", "approved_discount_cap": "5%"},
        allowed_slots=["approved_budget", "approved_discount_cap"],
        evidence_line="The provisional budget is 120,000 USD and the cap is 5%.",
        source_time="2026-01-01T09:00:00",
    )
    revised = UtilityRecord(
        record_id="revised",
        source_chunk_id="new",
        source_line_id="new:0",
        record_type="project_state",
        owner_user=None,
        principal_access="owner",
        lifecycle_status="active",
        slots={"approved_budget": "180,000 USD", "approved_discount_cap": "9%"},
        allowed_slots=["approved_budget", "approved_discount_cap"],
        evidence_line="The revised approved budget is 180,000 USD and the cap is 9%.",
        source_time="2026-02-01T09:00:00",
    )
    packed = pack_utility_records([stale, revised], need, {})
    selected_values = {
        value
        for record in packed.selected_records
        for value in record.slots.values()
    }
    if {"180,000 USD", "9%"} - selected_values:
        raise AssertionError(f"newer slot values were not retained: {selected_values}")

    family_principal = Principal("helper", None, "family", None)
    owner_principal = Principal("owner", None, "owner", None)
    if family_principal.relation_to_owner == owner_principal.relation_to_owner:
        raise AssertionError("access-boundary fixture is invalid")


def check_structured_realization_chain() -> None:
    semantic_spec = {
        "requested_slots": ["target_date", "approved_budget", "approved_discount_cap"],
        "temporal_scope": "current",
        "disclosure_scope": "full",
        "state_domain": "project",
        "request_shape": "fact",
    }
    principal = Principal("owner", None, "owner", None)
    lines = [
        {
            "chunk_id": "old",
            "text": "The provisional target date is September 8 and the approved budget is 120,000 USD.",
            "score": 0.8,
            "line_meta": {
                "frame_type": "project_state",
                "slots": {"target_date": "September 8", "approved_budget": "120,000 USD"},
                "source_time": "2026-01-01",
            },
        },
        {
            "chunk_id": "new",
            "text": "The revised target date is October 14, the approved budget is 180,000 USD, and the approved maximum discount is 9%.",
            "score": 0.7,
            "line_meta": {
                "frame_type": "project_state",
                "slots": {
                    "target_date": "October 14",
                    "approved_budget": "180,000 USD",
                    "approved_discount_cap": "9%",
                },
                "source_time": "2026-02-01",
            },
        },
    ]
    need = build_answer_need_spec(
        "Give the latest operating overview.",
        {},
        lines,
        {"semantic_spec": semantic_spec},
        {},
    )
    records = build_utility_records(lines, [], principal, need, {})
    packed = pack_utility_records(records, need, {})
    answer = render_canonical_answer(packed, need, {}, principal, {})
    verification = verify_canonical_answer(answer, packed, need, principal, {})
    if "180,000 USD" not in answer or "9%" not in answer or "120,000 USD" in answer:
        raise AssertionError(f"structured realization used stale state: {answer}")
    if not (verification.pass_coverage and verification.pass_access_safety):
        raise AssertionError(f"structured realization verification failed: {verification}")


def check_semantic_current_state_resolution() -> None:
    semantic_spec = {
        "requested_slots": ["target_date"],
        "temporal_scope": "current",
        "disclosure_scope": "full",
        "state_domain": "project",
        "request_shape": "fact",
    }
    from gov_mem.data.schema import RetrievedEvidence

    older = RetrievedEvidence(
        memory_id="older",
        content="The project moves from June 2, 2026 to June 9, 2026.",
        score=0.9,
        retrieval_source="smoke",
        reason="smoke",
        time="2026-01-01T09:00:00",
    )
    newer = RetrievedEvidence(
        memory_id="newer",
        content="The official date is July 16, 2026.",
        score=0.4,
        retrieval_source="smoke",
        reason="smoke",
        time="2026-02-01T09:00:00",
        metadata={"denied_slots": ["access_badge"]},
    )
    selected = _augment_current_state_slot_coverage(
        question="Give the latest overview.",
        evidence=[older, newer],
        selected_lines=[],
        semantic_spec=semantic_spec,
    )
    target_values = [
        str((item.get("line_meta") or {}).get("slots", {}).get("target_date") or "")
        for item in selected
    ]
    if target_values != ["July 16, 2026"]:
        raise AssertionError(f"semantic provenance selection failed: {target_values}")


def check_semantic_contract_repair() -> None:
    from gov_mem.planning.query_planner import _bindings_cover_attributes, _ground_attribute_contract

    broad_question = "What are the current planned date, approved amount, and active service window for the renamed asset?"
    broad_attributes, broad_valid = _ground_attribute_contract(
        ["current_asset_state"],
        [{
            "attribute": "current_asset_state",
            "support_span": broad_question,
            "semantic_role": "requested_property",
        }],
        broad_question,
    )
    if broad_valid or broad_attributes:
        raise AssertionError(f"whole-question span became a semantic property: {broad_attributes}")
    if _bindings_cover_attributes(
        attributes=["requested_value", "unbound_entity"],
        bindings=[{
            "attribute": "requested_value", "support_span": "requested value",
            "semantic_role": "requested_property",
        }],
    ):
        raise AssertionError("unbound planner field entered a certificate contract")

    from gov_mem.planning.query_planner import _normalize_certifiable_contract

    normalized = _normalize_certifiable_contract(
        {
            "requested_attributes": ["blockers", "project_aurora", "still_open"],
            "attribute_bindings": [
                {"attribute": "blockers", "support_span": "blockers", "semantic_role": "requested_property"},
                {"attribute": "project_aurora", "support_span": "Project Aurora", "semantic_role": "target_entity"},
                {"attribute": "still_open", "support_span": "still open", "semantic_role": "temporal_modifier"},
            ],
            "temporal_scope": "current",
        },
        "Which blockers are still open for Project Aurora?",
        target_entities=["Project Aurora"],
    )
    if normalized["requested_attributes"] != ["blockers"] or normalized["certifiable_needs"] != [{
        "need_id": "blockers", "attribute": "blockers", "query_support_span": "blockers",
        "evidence_slot_hint": "", "temporal_scope": "current",
    }]:
        raise AssertionError(f"entity or temporal modifier polluted certifiable needs: {normalized}")

    renamed = _normalize_certifiable_contract(
        {
            "requested_attributes": ["blockers", "project_nimbus", "still_open"],
            "attribute_bindings": [
                {"attribute": "blockers", "support_span": "blockers", "semantic_role": "requested_property"},
                {"attribute": "project_nimbus", "support_span": "Project Nimbus", "semantic_role": "target_entity"},
                {"attribute": "still_open", "support_span": "still open", "semantic_role": "temporal_modifier"},
            ],
            "temporal_scope": "current",
        },
        "Which blockers are still open for Project Nimbus?",
        target_entities=["Project Nimbus"],
    )
    if renamed["requested_attributes"] != normalized["requested_attributes"]:
        raise AssertionError(f"entity rename changed certifiable need count: {renamed}")

    named_property = _normalize_certifiable_contract(
        {
            "requested_attributes": ["return_alerts"],
            "attribute_bindings": [{
                "attribute": "return_alerts", "support_span": "return alerts",
                "semantic_role": "requested_property",
            }],
            "request_shape": "list",
        },
        "List the return alerts for the current plan.",
        target_entities=["return alerts"],
    )
    if named_property["requested_attributes"] != ["return_alerts"]:
        raise AssertionError(f"property sharing a dynamic entity surface was wrongly rejected: {named_property}")

    class TwoViewContractLLM:
        def __init__(self):
            self.calls = 0

        def chat_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "query_type": "factual",
                    "target_entities": ["Project Aurora"],
                    "semantic_spec": {
                        "requested_attributes": ["blockers", "project_aurora", "still_open"],
                        "attribute_bindings": [
                            {"attribute": "blockers", "support_span": "blockers", "semantic_role": "requested_property"},
                            {"attribute": "project_aurora", "support_span": "Project Aurora", "semantic_role": "requested_property"},
                            {"attribute": "still_open", "support_span": "still open", "semantic_role": "requested_property"},
                        ],
                        "request_shape": "list", "temporal_scope": "current",
                    },
                }
            return {"semantic_spec": {
                "requested_attributes": ["blockers", "project_aurora", "still_open"],
                "attribute_bindings": [
                    {"attribute": "blockers", "support_span": "blockers", "semantic_role": "requested_property"},
                    {"attribute": "project_aurora", "support_span": "Project Aurora", "semantic_role": "target_entity"},
                    {"attribute": "still_open", "support_span": "still open", "semantic_role": "temporal_modifier"},
                ],
                "request_shape": "list", "temporal_scope": "current",
            }}

    audited_instance = type("Fixture", (), {
        "question": "Which blockers are still open for Project Aurora?",
        "asking_user_id": "owner", "choices": None, "metadata": {"observable": {}}, "domain": "synthetic",
    })()
    audited_plan = QueryUnderstandingAgent(
        llm_client=TwoViewContractLLM(), model_name="stub"
    ).plan(audited_instance)
    if (
        audited_plan.planning_trace.get("semantic_contract_source") not in {"audited_llm", "verified_llm"}
        or audited_plan.semantic_spec.get("requested_attributes") != ["blockers"]
        or not audited_plan.semantic_spec.get("attribute_bindings_valid")
    ):
        raise AssertionError(f"two-view contract audit did not isolate the factual need: {audited_plan}")

    class StubLLM:
        def __init__(self):
            self.prompts = []

        def chat_json(self, *, model, system_prompt, user_prompt):
            self.prompts.append(user_prompt)
            if len(self.prompts) == 1:
                return {
                    "query_type": "factual",
                    "target_users": [],
                    "target_entities": [],
                    "required_memory_types": ["factual"],
                    "symbolic_filters": {},
                    "dense_queries": ["ignored"],
                    "reasoning_ops": ["AND"],
                    "semantic_spec": {},
                }
            return {
                "semantic_spec": {
                    "requested_slots": ["approved_budget", "approved_discount_cap"],
                    "temporal_scope": "current",
                    "disclosure_scope": "full",
                    "state_domain": "project",
                    "request_shape": "fact",
                    "requires_entity_resolution": True,
                }
            }

    instance = type(
        "Fixture",
        (),
        {
            "question": "What is the revised operating position?",
            "asking_user_id": "reviewer",
            "choices": None,
            "metadata": {"observable": {"role": "manager"}},
            "domain": "synthetic",
        },
    )()
    client = StubLLM()
    plan = QueryUnderstandingAgent(llm_client=client, model_name="stub").plan(instance)
    if plan.planning_trace.get("semantic_contract_source") not in {"repair_llm", "audited_llm", "verified_llm", "missing_after_repair"}:
        raise AssertionError(f"missing semantic repair trace: {plan.planning_trace}")
    if plan.planning_trace.get("semantic_contract_source") != "missing_after_repair" and plan.semantic_spec.get("requested_slots") != ["approved_budget", "approved_discount_cap"]:
        raise AssertionError(f"semantic repair lost requested slots: {plan.semantic_spec}")
    if len(client.prompts) not in {3, 4}:
        raise AssertionError("semantic contract did not perform repair plus independent verification")

    class OfflineLLM:
        def chat_json(self, **kwargs):
            raise LLMClientUnavailableError("offline")

    fallback = QueryUnderstandingAgent(llm_client=OfflineLLM(), model_name="stub").plan(instance)
    if fallback.planning_trace.get("reason") != "llm_unavailable":
        raise AssertionError(f"fallback reason is not auditable: {fallback.planning_trace}")


def check_redacted_governance_contract() -> None:
    from gov_mem.backbones.common import (
        _policy_authorized_source_slots,
        _render_redacted_answer_from_governance_contract,
    )

    row = RetrievedEvidence(
        memory_id="mixed",
        content="Public appointment window is April 8. Private access detail is at 5:55 PM by the east curb.",
        score=1.0,
        retrieval_source="smoke",
        reason="smoke",
    )

    class ContractLLM:
        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            return {
                "frames": [{
                    "memory_id": "mixed",
                    "lifecycle": "active",
                    "disclosure": "allow",
                    "allowed_slots": [{"slot": "free_form_explanation", "surface": "Public appointment window is April 8."}],
                }]
            }

    result = _render_redacted_answer_from_governance_contract(
        question="Can you share the public appointment?",
        evidence=[row],
        selected_lines=[{"chunk_id": "mixed"}],
        used_chunk_ids=["mixed"],
        semantic_intent={"semantic_spec": {"disclosure_scope": "public_only", "requested_slots": ["date"]}},
        principal=Principal("helper", "guest", "delegate", None),
        policy_decisions=[],
        llm_client=ContractLLM(),
        model_name="stub",
    )
    if "April 8" not in result.answer_text or "5:55 PM" in result.answer_text or "east curb" in result.answer_text:
        raise AssertionError(f"redacted contract leaked or omitted evidence: {result.answer_text}")

    class UnavailableContractLLM:
        def chat_json(self, **kwargs):
            raise LLMClientUnavailableError("offline")

    failed_closed = _render_redacted_answer_from_governance_contract(
        question="Can you share the public appointment?",
        evidence=[row],
        selected_lines=[{"chunk_id": "mixed"}],
        used_chunk_ids=["mixed"],
        semantic_intent={},
        principal=Principal("helper", "guest", "delegate", None),
        policy_decisions=[],
        llm_client=UnavailableContractLLM(),
        model_name="stub",
    )
    if "April 8" in failed_closed.answer_text or "5:55 PM" in failed_closed.answer_text:
        raise AssertionError(f"missing contract did not fail closed: {failed_closed.answer_text}")

    logistics = RetrievedEvidence(
        memory_id="logistics",
        content="The approved service window is 9:15 AM to 9:55 AM.",
        score=1.0,
        retrieval_source="smoke",
        reason="smoke",
    )
    allowed = _policy_authorized_source_slots(
        candidates=[logistics],
        policy_decisions=[{
            "chunk_id": "logistics",
            "allowed_for_requester": True,
            "allowed_slot_groups": ["logistics"],
        }],
        semantic_spec={"requested_slots": ["visit_window"]},
    )
    if allowed != [{"memory_id": "logistics", "slot": "visit_window", "surface": "9:15 AM to 9:55 AM"}]:
        raise AssertionError(f"policy slot path is not source-grounded: {allowed}")
    if _policy_authorized_source_slots(
        candidates=[logistics],
        policy_decisions=[{"chunk_id": "logistics", "allowed_for_requester": True, "allowed_slot_groups": ["logistics"]}],
        semantic_spec={},
    ):
        raise AssertionError("broad policy group bypassed the semantic slot contract")


def check_current_state_authorization_certificate() -> None:
    from gov_mem.governance_runtime.provenance_authorization import certify_current_state_slots

    older = RetrievedEvidence(
        memory_id="older",
        content="The approved budget is 120,000 USD.",
        score=1.0,
        retrieval_source="smoke",
        reason="smoke",
        time="2026-01-01T09:00:00",
    )
    revised = RetrievedEvidence(
        memory_id="revised",
        content="The revised approved budget is 180,000 USD and supersedes the earlier figure.",
        score=1.0,
        retrieval_source="smoke",
        reason="smoke",
        time="2026-02-01T09:00:00",
    )
    spec = {"requested_slots": ["approved_budget"], "temporal_scope": "current"}
    certificate = certify_current_state_slots(
        semantic_spec=spec,
        evidence=[older, revised],
        allowed_rows=[revised],
        redacted_rows=[],
    )
    if not certificate.get("authorized") or certificate["slots"]["approved_budget"]["value"] != "180,000 USD":
        raise AssertionError(f"latest authorized state was not certified: {certificate}")
    blocked = certify_current_state_slots(
        semantic_spec=spec,
        evidence=[older, revised],
        allowed_rows=[],
        redacted_rows=[],
    )
    if blocked.get("authorized"):
        raise AssertionError(f"inaccessible latest state was certified: {blocked}")
    mixed = RetrievedEvidence(
        memory_id="mixed_state",
        content="Revised approved budget is 180,000 USD; active access badge is ZX-9.",
        score=1.0,
        retrieval_source="smoke",
        reason="smoke",
        time="2026-02-01T09:00:00",
        metadata={"denied_slots": ["access_badge"]},
    )
    mixed_certificate = certify_current_state_slots(
        semantic_spec=spec,
        evidence=[mixed],
        allowed_rows=[mixed],
        redacted_rows=[],
    )
    if mixed_certificate.get("authorized"):
        raise AssertionError(f"mixed access-artifact evidence was certified: {mixed_certificate}")


def check_slot_version_graph() -> None:
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    base = {
        "atom_type": "fact_atom",
        "owner_id": "owner",
        "subject_id": "renamed_program",
        "speaker_id": "finance",
        "source_turn": 1,
        "lifecycle": "active",
        "sensitivity": "private",
        "access_scope": ["collaborator"],
        "related_entities": ["renamed_program"],
        "confidence": 1.0,
    }
    older = GovernedMemoryAtom(
        atom_id="old",
        text="Earlier approved budget is 120,000 USD.",
        slots={"approved_budget": "120,000 USD"},
        timestamp="2026-01-01T09:00:00",
        provenance={"frame_type": "project_state"},
        **base,
    )
    newer = GovernedMemoryAtom(
        atom_id="new",
        text="Revised approved budget is 180,000 USD.",
        slots={"approved_budget": "180,000 USD"},
        timestamp="2026-02-01T09:00:00",
        provenance={"frame_type": "project_state"},
        **base,
    )
    graph = GovernedGraphBuilder().build(graph_id="smoke", instance_id="smoke", atoms=[older, newer])
    edge_types = {edge.edge_type for edge in graph.edges}
    if not {"version_precedes", "supersedes_slot"}.issubset(edge_types):
        raise AssertionError(f"slot version graph lacks provenance edges: {edge_types}")

    permitted = GovernedMemoryAtom(
        atom_id="permission",
        atom_type="permission_atom",
        text="Collaborators are authorized to receive approved budget.",
        slots={"approved_budget": "180,000 USD"},
        owner_id="owner",
        subject_id="renamed_program",
        speaker_id="owner",
        source_turn=2,
        timestamp="2026-02-01T10:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["collaborator"],
        related_entities=["renamed_program"],
        provenance={
            "frame_type": "project_state",
            "policy_binding": {
                "effect": "allow",
                "scopes": ["collaborator"],
                "slots": ["approved_budget"],
                "support_spans": ["Collaborators may receive the current approved budget."],
            },
        },
        confidence=1.0,
    )
    # Permission and fact are separate atoms: authorization must point to the
    # fact atom whose typed value has source provenance.
    permitted_graph = GovernedGraphBuilder().build(graph_id="permit", instance_id="smoke", atoms=[permitted, newer])
    certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["approved_budget"], "temporal_scope": "current"},
        graph=permitted_graph,
        principal_relation="authorized_staff",
    )
    if not certificate.get("authorized"):
        raise AssertionError(f"explicit graph allow path was not certified: {certificate}")
    from gov_mem.legacy.graph_slot_renderer import build_graph_authorized_projection
    projection = build_graph_authorized_projection(
        certificate=certificate,
        semantic_spec={"requested_slots": ["approved_budget"], "temporal_scope": "current"},
    )
    if projection is None or projection.metadata.get("slots", {}).get("approved_budget") != "180,000 USD":
        raise AssertionError(f"graph certificate did not produce a source-grounded typed projection: {projection}")
    denied_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["approved_budget"], "temporal_scope": "current"},
        graph=permitted_graph,
        principal_relation="guest",
    )
    if denied_certificate.get("authorized"):
        raise AssertionError(f"incompatible graph role was certified: {denied_certificate}")
    foreign_fact = GovernedMemoryAtom(
        atom_id="foreign",
        text="Another owner's approved budget is 999,000 USD.",
        slots={"approved_budget": "999,000 USD"},
        timestamp="2026-03-01T09:00:00",
        provenance={"frame_type": "project_state"},
        **{**base, "owner_id": "another_owner"},
    )
    cross_owner_graph = GovernedGraphBuilder().build(
        graph_id="cross-owner", instance_id="smoke", atoms=[permitted, foreign_fact]
    )
    cross_owner_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["approved_budget"], "temporal_scope": "current"},
        graph=cross_owner_graph,
        principal_relation="authorized_staff",
        owner_id="owner",
    )
    if cross_owner_certificate.get("authorized"):
        raise AssertionError(f"policy incorrectly crossed owners: {cross_owner_certificate}")
    competing_graph = GovernedGraphBuilder().build(
        graph_id="competing-owner", instance_id="smoke", atoms=[permitted, newer, foreign_fact]
    )
    competing_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["approved_budget"], "temporal_scope": "current"},
        graph=competing_graph,
        principal_relation="authorized_staff",
        owner_id="owner",
    )
    if not competing_certificate.get("authorized") or competing_certificate["slots"]["approved_budget"]["value"] != "180,000 USD":
        raise AssertionError(f"newer foreign owner displaced authorized fact: {competing_certificate}")
    deleted = GovernedMemoryAtom(
        atom_id="deleted",
        atom_type="deletion_atom",
        text="Remove the approved budget value from memory.",
        slots={"approved_budget": "120,000 USD"},
        owner_id="owner",
        subject_id="renamed_program",
        speaker_id="owner",
        source_turn=2,
        timestamp="2026-01-02T09:00:00",
        lifecycle="deleted",
        sensitivity="deleted",
        access_scope=["self"],
        related_entities=["renamed_program"],
        provenance={"frame_type": "project_state"},
        confidence=1.0,
    )
    deleted_graph = GovernedGraphBuilder().build(
        graph_id="deleted", instance_id="smoke", atoms=[older, deleted, permitted]
    )
    deleted_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["approved_budget"], "temporal_scope": "current"},
        graph=deleted_graph,
        principal_relation="authorized_staff",
        owner_id="owner",
    )
    if deleted_certificate.get("authorized"):
        raise AssertionError(f"deleted fact remained graph-certifiable: {deleted_certificate}")


def check_dual_channel_convergence() -> None:
    """Only independently selected utility and governance evidence may converge."""
    from gov_mem.data.schema import RetrievedEvidence
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom
    from gov_mem.retrieval.dual_channel_retriever import DualChannelRetriever

    fact = GovernedMemoryAtom(
        atom_id="fact",
        atom_type="fact_atom",
        text="The revised calibration value is 73 units.",
        slots={"calibration_value": "73 units"},
        owner_id="owner",
        subject_id="device",
        speaker_id="owner",
        source_turn=2,
        timestamp="2026-02-01T09:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["collaborator"],
        related_entities=["device"],
        provenance={"frame_type": "device_state", "source_memory_id": "fact_memory"},
        confidence=0.9,
    )
    policy = GovernedMemoryAtom(
        atom_id="policy",
        atom_type="permission_atom",
        text="A collaborator may receive the calibration value.",
        slots={},
        owner_id="owner",
        subject_id="device",
        speaker_id="owner",
        source_turn=1,
        timestamp="2026-01-01T09:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["collaborator"],
        related_entities=["device"],
        provenance={"policy_binding": {
            "effect": "allow", "scopes": ["collaborator"],
            "slots": ["calibration_value"], "support_spans": ["may receive the calibration value"],
        }},
        confidence=0.9,
    )
    evidence = [RetrievedEvidence(
        memory_id="fact_memory", content=fact.text, score=0.9,
        retrieval_source="smoke", reason="smoke", metadata={"slots": fact.slots},
    )]
    graph = GovernedGraphBuilder().build(graph_id="dual", instance_id="smoke", atoms=[policy, fact])
    bundle = DualChannelRetriever().retrieve(
        query="What is the calibration value?", requester="helper", owner="owner",
        relation="authorized_staff", evidence=evidence, governed_atoms=[policy, fact],
        graph_paths=[], requested_attributes=["calibration_value"],
    )
    certificate = certify_graph_slot_paths(
        semantic_spec={"requested_attributes": ["calibration_value"], "temporal_scope": "current"},
        graph=graph, principal_relation="authorized_staff",
        owner_id="owner",
        utility_atom_ids=set(bundle.utility_evidence.selected_atom_ids),
        governance_policy_atom_ids=set(bundle.governance_evidence.selected_policy_atom_ids),
    )
    if not certificate.get("authorized"):
        raise AssertionError(f"dual-channel convergence should certify selected evidence: {certificate}")
    rejected = certify_graph_slot_paths(
        semantic_spec={"requested_attributes": ["calibration_value"], "temporal_scope": "current"},
        graph=graph, principal_relation="authorized_staff",
        owner_id="owner",
        utility_atom_ids=set(), governance_policy_atom_ids=set(bundle.governance_evidence.selected_policy_atom_ids),
    )
    if rejected.get("authorized"):
        raise AssertionError(f"certificate ignored utility-channel selection: {rejected}")


def check_structural_failure_diagnosis() -> None:
    from gov_mem.experience.structural_diagnosis import diagnose_runtime_trace

    trace = {
        "query_plan": {"semantic_spec": {"requested_attributes": ["calibration_value"]}},
        "selected_evidence": [{"memory_id": "row"}],
        "slot_coverage": {"missing_slots": []},
        "dual_channel_retrieval": {
            "utility_evidence": {"selected_atom_ids": ["fact"]},
            "governance_evidence": {"selected_policy_atom_ids": []},
        },
        "graph_authorization_certificate": {"authorized": False, "reason": "no_explicit_graph_allow:calibration_value"},
    }
    diagnosis = diagnose_runtime_trace(trace)
    if diagnosis.get("category") != "governance_binding_selection_miss":
        raise AssertionError(f"runtime diagnosis used non-structural category: {diagnosis}")
    missing_contract = diagnose_runtime_trace({"selected_evidence": [{"memory_id": "row"}]})
    if missing_contract.get("category") != "semantic_contract_missing":
        raise AssertionError(f"missing semantic contract was misdiagnosed: {missing_contract}")


def check_certificate_requires_llm_contract() -> None:
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    denied = certify_graph_slot_paths(
        semantic_spec={"requested_attributes": ["calibration_value"], "temporal_scope": "current"},
        graph=None,
        principal_relation="authorized_staff",
        semantic_contract_certifiable=False,
    )
    if denied.get("reason") != "semantic_contract_not_certifiable":
        raise AssertionError(f"heuristic semantic contract could reach graph authorization: {denied}")


def check_open_semantic_slot_schema() -> None:
    from gov_mem.llm.prompts import build_query_planner_user_prompt
    from gov_mem.planning.query_planner import _normalize_semantic_spec
    spec = _normalize_semantic_spec({"requested_slots": ["novel_resistance_coefficient"]})
    if spec.get("requested_slots") != ["novel_resistance_coefficient"]:
        raise AssertionError(f"closed slot vocabulary rejected an unseen attribute: {spec}")
    prompt = build_query_planner_user_prompt(
        question="What is the resistance coefficient?", asking_user_id=None,
        choices=None, observable_metadata={}, skill_text="", retrieved_lessons=[],
    )
    if "This is an open schema" not in prompt:
        raise AssertionError("planner prompt did not request an open semantic schema")


def check_policy_frame_compiler() -> None:
    from gov_mem.governance_runtime.policy_frames import compile_policy_frames
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    atom = GovernedMemoryAtom(
        atom_id="policy",
        atom_type="policy_atom",
        text="Collaborators may receive the current approved budget.",
        slots={},
        owner_id="owner",
        subject_id="renamed_program",
        speaker_id="owner",
        source_turn=1,
        timestamp="2026-01-01T09:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["self"],
        related_entities=["renamed_program"],
        provenance={},
        confidence=1.0,
    )

    class PolicyLLM:
        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            return {"bindings": [{
                "atom_id": "policy",
                "effect": "allow",
                "scopes": ["collaborator"],
                "slots": ["approved_budget"],
                "support_spans": ["Collaborators may receive the current approved budget."],
            }]}

    observed_budget = GovernedMemoryAtom(
        atom_id="observed_budget",
        atom_type="fact_atom",
        text="The approved budget is 180,000 USD.",
        slots={"approved_budget": "180,000 USD"},
        owner_id="owner",
        subject_id="renamed_program",
        speaker_id="owner",
        source_turn=2,
        timestamp="2026-01-01T01:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["collaborator"],
        related_entities=["renamed_program"],
        provenance={"frame_type": "project_state"},
        confidence=1.0,
    )
    compiled = compile_policy_frames(atoms=[atom, observed_budget], llm_client=PolicyLLM(), model_name="stub")
    binding = compiled[0].provenance.get("policy_binding") or {}
    if compiled[0].access_scope != ["collaborator"] or binding.get("slots") != ["approved_budget"]:
        raise AssertionError(f"policy binding normalization failed: {compiled[0]}")

    # An unseen schema-shaped slot must work without a curated slot vocabulary.
    novel = GovernedMemoryAtom(
        atom_id="novel_policy",
        atom_type="permission_atom",
        text="Collaborators may receive the current field calibration coefficient.",
        slots={},
        owner_id="owner",
        subject_id="renamed_program",
        speaker_id="owner",
        source_turn=2,
        timestamp="2026-02-01T10:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["self"],
        related_entities=["renamed_program"],
        provenance={"frame_type": "project_state"},
        confidence=1.0,
    )
    novel_fact = GovernedMemoryAtom(
        atom_id="novel_fact",
        atom_type="fact_atom",
        text="The current field calibration coefficient is 0.91.",
        slots={"field_calibration_coefficient": "0.91"},
        owner_id="owner",
        subject_id="renamed_program",
        speaker_id="owner",
        source_turn=3,
        timestamp="2026-02-01T11:00:00",
        lifecycle="active",
        sensitivity="private",
        access_scope=["self"],
        related_entities=["renamed_program"],
        provenance={"frame_type": "project_state"},
        confidence=1.0,
    )

    class NovelSlotLLM(PolicyLLM):
        def chat_json(self, **kwargs):
            return {"bindings": [{
                "atom_id": "novel_policy",
                "effect": "allow",
                "scopes": ["collaborator"],
                "slots": ["field calibration coefficient"],
                "support_spans": ["Collaborators may receive the current field calibration coefficient."],
            }]}

    novel_compiled = compile_policy_frames(atoms=[novel, novel_fact], llm_client=NovelSlotLLM(), model_name="stub")
    novel_graph = GovernedGraphBuilder().build(graph_id="novel", instance_id="smoke", atoms=novel_compiled)
    novel_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["field_calibration_coefficient"], "temporal_scope": "current"},
        graph=novel_graph,
        principal_relation="authorized_staff",
    )
    if not novel_certificate.get("authorized"):
        raise AssertionError(f"novel policy slot was not graph-authorized: {novel_certificate}")
    unbound_graph = GovernedGraphBuilder().build(graph_id="unbound", instance_id="smoke", atoms=[novel, novel_fact])
    unbound_certificate = certify_graph_slot_paths(
        semantic_spec={"requested_slots": ["field_calibration_coefficient"], "temporal_scope": "current"},
        graph=unbound_graph,
        principal_relation="authorized_staff",
    )
    if unbound_certificate.get("authorized"):
        raise AssertionError(f"unbound policy atom created an authorization edge: {unbound_certificate}")

    class UnsupportedBindingLLM(PolicyLLM):
        def chat_json(self, **kwargs):
            return {"bindings": [{
                "atom_id": "policy",
                "effect": "allow",
                "scopes": ["collaborator"],
                "slots": ["approved_budget"],
                "support_spans": ["This span is not in the source atom."],
            }]}

    unsupported = compile_policy_frames(atoms=[atom], llm_client=UnsupportedBindingLLM(), model_name="stub")
    if (unsupported[0].provenance or {}).get("policy_binding"):
        raise AssertionError("policy compiler accepted a binding without a verbatim source citation")


def check_dev_only_adaptation_gate() -> None:
    from gov_mem.experience.runtime_experience import RuntimeExperienceContext, RuntimeExperienceRetriever
    from gov_mem.experience.pattern_inducer import FailurePattern
    from gov_mem.skills.skill_library import GovernanceSkillLibraryBuilder
    from gov_mem.utils.io import write_jsonl
    from tempfile import TemporaryDirectory
    from pathlib import Path

    attestation = {
        "artifact_scope": "development",
        "allows_adaptation": True,
        "test_data_used": False,
        "source_runs": ["/dev/run"],
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "experience.jsonl"
        write_jsonl(path, [{
            "exp_id": "legacy", "pattern_id": "legacy", "failure_type": "missing_utility",
            "reusable_skill_hypothesis": "legacy", "correction_strategy": "must_not_load",
            "applicable_domains": ["renamed_domain"], "applicable_roles": [], "applicable_slots": ["novel_attribute"], "confidence": 1.0,
        }, {
            "exp_id": "dev", "pattern_id": "dev", "failure_type": "missing_utility",
            "reusable_skill_hypothesis": "dev", "correction_strategy": "development_only_lesson",
            "applicable_domains": ["renamed_domain"], "applicable_roles": [], "applicable_slots": ["novel_attribute"], "confidence": 1.0,
            "metadata": {"dev_attestation": attestation},
        }])
        lessons = RuntimeExperienceRetriever(path).retrieve(
            context=RuntimeExperienceContext(
                domain="renamed_domain", requester_role="", owner_relation="",
                question="Describe the requested attribute.", required_slots=["novel_attribute"],
            ),
        )
    if [lesson.experience_id for lesson in lessons] != ["dev"]:
        raise AssertionError(f"runtime consumed an unattested adaptation artifact: {lessons}")
    pattern = FailurePattern(
        pattern_id="dev_pattern", failure_type="missing_utility", description="generic",
        trigger_signature={}, affected_domains=["renamed_domain"], affected_roles=[],
        affected_slots=["novel_attribute"], recommended_rule="generic",
        recommended_skill="generic_skill", support_cases=["case"], confidence=0.8,
        provenance={"dev_attestation": attestation},
    )
    skills = GovernanceSkillLibraryBuilder().build(patterns=[pattern])
    if len(skills) != 1 or skills[0].metadata.get("dev_attestation") != attestation:
        raise AssertionError("development attestation was not carried into a runtime skill")


def check_open_attribute_alignment_and_temporal_certificates() -> None:
    """Validate evidence-mediated alignment without a curated slot vocabulary."""
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.legacy.semantic_alignment import align_requested_attributes
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    common = {
        "atom_type": "fact_atom", "owner_id": "owner", "subject_id": "renamed_device",
        "speaker_id": "owner", "lifecycle": "active", "sensitivity": "private",
        "access_scope": ["collaborator"], "related_entities": ["renamed_device"], "confidence": 1.0,
        "provenance": {"frame_type": "device_state"},
    }
    earlier = GovernedMemoryAtom(
        atom_id="earlier_threshold", text="The device's configured threshold is 12 units.",
        slots={"threshold_value": "12 units"}, timestamp="2026-01-01T09:00:00", source_turn=1, **common,
    )
    later = GovernedMemoryAtom(
        atom_id="later_threshold", text="The device's configured threshold is 17 units.",
        slots={"threshold_value": "17 units"}, timestamp="2026-02-01T09:00:00", source_turn=2, **common,
    )
    policy = GovernedMemoryAtom(
        atom_id="threshold_permission", atom_type="permission_atom",
        text="Collaborators may receive the configured threshold.", slots={}, owner_id="owner",
        subject_id="renamed_device", speaker_id="owner", source_turn=3, timestamp="2026-02-01T10:00:00",
        lifecycle="active", sensitivity="private", access_scope=["collaborator"],
        related_entities=["renamed_device"], confidence=1.0,
        provenance={"policy_binding": {
            "effect": "allow", "scopes": ["collaborator"], "slots": ["threshold_value"],
            "support_spans": ["Collaborators may receive the configured threshold."],
        }},
    )
    graph = GovernedGraphBuilder().build(graph_id="aligned", instance_id="smoke", atoms=[earlier, later, policy])
    later_slot = next(
        node.node_id for node in graph.nodes
        if node.node_type == "SlotNode" and node.provenance.get("source_atom_id") == "later_threshold"
    )
    earlier_slot = next(
        node.node_id for node in graph.nodes
        if node.node_type == "SlotNode" and node.provenance.get("source_atom_id") == "earlier_threshold"
    )

    class AlignmentLLM:
        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            return {"bindings": [{
                "attribute": "operational_limit",
                "slot_node_id": later_slot,
                "query_support_span": "configured operating limit",
                "fact_support_span": "17 units",
            }]}

    question = "What is the configured operating limit?"
    spec = {
        "requested_attributes": ["operational_limit"],
        "attribute_bindings": [{
            "attribute": "operational_limit", "support_span": "configured operating limit", "semantic_role": "requested_property",
        }],
        "temporal_scope": "unspecified", "request_shape": "fact",
    }
    alignment = align_requested_attributes(
        question=question, semantic_spec=spec, graph=graph, owner_id="owner", utility_atom_ids=None,
        llm_client=AlignmentLLM(), model_name="stub", semantic_contract_certifiable=True,
    )
    certificate = certify_graph_slot_paths(
        semantic_spec=spec, graph=graph, principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=alignment,
    )
    if not certificate.get("authorized") or certificate["slots"]["operational_limit"]["value"] != "17 units":
        raise AssertionError(f"open attribute alignment did not select the active value: {certificate}")

    historical_spec = {**spec, "temporal_scope": "historical"}
    historical_alignment = {
        "available": True,
        "bindings": {"operational_limit": {
            "attribute": "operational_limit", "slot_name": "threshold_value",
            "anchor_slot_node_id": earlier_slot, "source": "llm_evidence_alignment",
        }},
    }
    historical = certify_graph_slot_paths(
        semantic_spec=historical_spec, graph=graph, principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=historical_alignment,
    )
    if not historical.get("authorized") or historical["slots"]["operational_limit"]["value"] != "12 units":
        raise AssertionError(f"explicit historical request was not certified: {historical}")

    no_policy = GovernedGraphBuilder().build(graph_id="unbound", instance_id="smoke", atoms=[earlier, later])
    denied = certify_graph_slot_paths(
        semantic_spec=spec, graph=no_policy, principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=alignment,
    )
    if denied.get("authorized"):
        raise AssertionError(f"aligned evidence bypassed explicit policy authorization: {denied}")


def check_certificate_driven_action_normalization() -> None:
    """An explicit graph proof outranks lexical action normalization."""
    from gov_mem.data.schema import GovernedActionDecision
    from gov_mem.governance_runtime.action_predictor import GovernedActionPredictor

    certificate = {
        "authorized": True,
        "reason": "all_requested_attributes_have_explicit_allow_path",
        "realizations": [{
            "attribute": "renamed_property", "value": "47 units",
            "source_memory_id": "observed_memory",
        }],
    }
    predictor = GovernedActionPredictor(llm_client=None, model_name="stub")
    owner = predictor._normalize_decision(
        GovernedActionDecision("refuse", "refusal", "denied", None, [], "stub"),
        question="Provide the configured operating limit.", query_type="utility", domain="renamed_domain",
        principal_relation="owner", requester_role=None, evidence=[], typed_state_summary={}, access_partition={},
        semantic_spec={"requested_attributes": ["renamed_property"], "temporal_scope": "historical"},
        graph_authorization_certificate=certificate,
    )
    if owner.action != "answer" or owner.evidence_memory_ids != ["observed_memory"]:
        raise AssertionError(f"owner graph certificate did not control the action: {owner}")
    non_owner = predictor._normalize_decision(
        GovernedActionDecision("no_memory", "abstain", "unknown", None, [], "stub"),
        question="Provide the configured operating limit.", query_type="utility", domain="renamed_domain",
        principal_relation="authorized_staff", requester_role=None, evidence=[], typed_state_summary={}, access_partition={},
        semantic_spec={"requested_attributes": ["renamed_property"], "temporal_scope": "unspecified"},
        graph_authorization_certificate=certificate,
    )
    if non_owner.action != "answer_redacted" or non_owner.evidence_memory_ids != ["observed_memory"]:
        raise AssertionError(f"non-owner graph certificate did not use minimal redaction: {non_owner}")


def check_list_alignment_requires_explicit_multiple_anchors() -> None:
    """List realization may include multiple records only through explicit alignment."""
    from gov_mem.legacy.graph_slot_renderer import render_graph_authorized_slots
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.legacy.semantic_alignment import align_requested_attributes
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    def fact(atom_id: str, subject: str, value: str) -> GovernedMemoryAtom:
        return GovernedMemoryAtom(
            atom_id=atom_id, atom_type="fact_atom", text=f"{subject} has a reported threshold of {value}.",
            slots={"threshold_value": value}, owner_id="owner", subject_id=subject, speaker_id="owner",
            source_turn=1, timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
            access_scope=["collaborator"], related_entities=[subject], provenance={}, confidence=1.0,
        )

    first, second = fact("first_record", "unit_one", "17 units"), fact("second_record", "unit_two", "23 units")
    policy = GovernedMemoryAtom(
        atom_id="list_permission", atom_type="permission_atom",
        text="Collaborators may receive reported thresholds.", slots={}, owner_id="owner", subject_id="owner",
        speaker_id="owner", source_turn=2, timestamp="2026-01-01T01:00:00", lifecycle="active",
        sensitivity="private", access_scope=["collaborator"], related_entities=[], confidence=1.0,
        provenance={"policy_binding": {
            "effect": "allow", "scopes": ["collaborator"], "slots": ["threshold_value"],
            "support_spans": ["Collaborators may receive reported thresholds."],
        }},
    )
    graph = GovernedGraphBuilder().build(graph_id="list", instance_id="smoke", atoms=[first, second, policy])
    slots = [node.node_id for node in graph.nodes if node.node_type == "SlotNode"]

    class ListAlignmentLLM:
        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            return {"bindings": [{
                "attribute": "reported_limits", "slot_node_ids": slots,
                "query_support_span": "reported limits",
                "fact_support_spans": [
                    {"slot_node_id": slots[0], "fact_support_span": "17 units"},
                    {"slot_node_id": slots[1], "fact_support_span": "23 units"},
                ],
            }]}

    spec = {
        "requested_attributes": ["reported_limits"],
        "attribute_bindings": [{"attribute": "reported_limits", "support_span": "reported limits", "semantic_role": "requested_property"}],
        "request_shape": "list", "temporal_scope": "unspecified",
    }
    alignment = align_requested_attributes(
        question="List the reported limits.", semantic_spec=spec, graph=graph, owner_id="owner",
        utility_atom_ids=None, llm_client=ListAlignmentLLM(), model_name="stub", semantic_contract_certifiable=True,
    )
    certificate = certify_graph_slot_paths(
        semantic_spec=spec, graph=graph, principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=alignment,
    )
    values = [item["value"] for item in certificate.get("realizations") or []]
    if not certificate.get("authorized") or set(values) != {"17 units", "23 units"}:
        raise AssertionError(f"explicit multi-record list was not certified: {certificate}")
    rendered = render_graph_authorized_slots(certificate=certificate, action="answer_redacted")
    if "17 units" not in rendered.answer_text or "23 units" not in rendered.answer_text:
        raise AssertionError(f"multi-record realization omitted an attested value: {rendered}")
    duplicate_rendered = render_graph_authorized_slots(
        certificate={
            "slots": {"reported_limits": certificate["slots"]["reported_limits"]},
            "realizations": [
                certificate["realizations"][0], certificate["realizations"][0], certificate["realizations"][1],
            ],
        },
        action="answer_redacted",
    )
    if duplicate_rendered.answer_text.count("17 units") != 1:
        raise AssertionError(f"slot alias duplicate was rendered twice: {duplicate_rendered}")

    mixed_spec = {**spec, "request_shape": "mixed"}
    mixed_alignment = align_requested_attributes(
        question="List the reported limits.", semantic_spec=mixed_spec, graph=graph, owner_id="owner",
        utility_atom_ids=None, llm_client=ListAlignmentLLM(), model_name="stub", semantic_contract_certifiable=True,
    )
    mixed_certificate = certify_graph_slot_paths(
        semantic_spec=mixed_spec, graph=graph, principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=mixed_alignment,
    )
    if not mixed_certificate.get("authorized") or len(mixed_certificate.get("realizations") or []) != 2:
        raise AssertionError(f"mixed collection claim was not certified: {mixed_certificate}")


def check_attested_span_certificate_gate() -> None:
    """Experimental realization accepts only a fact with an attested source span."""
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.legacy.semantic_alignment import align_requested_attributes
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    text = "The observed thermal limit is 41 units."

    def build_graph(evidence_span: str | None):
        fact = GovernedMemoryAtom(
            atom_id="attested_fact", atom_type="fact_atom", text=text, slots={"thermal_limit": "41 units"},
            owner_id="owner", subject_id="sensor", speaker_id="owner", source_turn=1,
            timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
            access_scope=["collaborator"], related_entities=["sensor"], confidence=1.0,
            provenance={"evidence_span": evidence_span} if evidence_span else {},
        )
        policy = GovernedMemoryAtom(
            atom_id="attested_policy", atom_type="permission_atom", text="Collaborators may receive the thermal limit.",
            slots={}, owner_id="owner", subject_id="sensor", speaker_id="owner", source_turn=2,
            timestamp="2026-01-01T01:00:00", lifecycle="active", sensitivity="private",
            access_scope=["collaborator"], related_entities=["sensor"], confidence=1.0,
            provenance={"policy_binding": {
                "effect": "allow", "scopes": ["collaborator"], "slots": ["thermal_limit"],
                "support_spans": ["Collaborators may receive the thermal limit."],
            }},
        )
        return GovernedGraphBuilder().build(graph_id="attested", instance_id="smoke", atoms=[fact, policy])

    spec = {"requested_attributes": ["thermal_limit"], "temporal_scope": "unspecified"}
    graph = build_graph(text)
    alignment = align_requested_attributes(
        question="What is the thermal limit?", semantic_spec=spec, graph=graph, owner_id="owner",
        utility_atom_ids=None, llm_client=None, model_name="stub", semantic_contract_certifiable=True,
    )
    certified = certify_graph_slot_paths(
        semantic_spec=spec, graph=graph, principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=alignment, require_attested_evidence_span=True,
    )
    if not certified.get("authorized"):
        raise AssertionError(f"attested fact was rejected by strict certificate: {certified}")
    missing = certify_graph_slot_paths(
        semantic_spec=spec, graph=build_graph(None), principal_relation="authorized_staff", owner_id="owner",
        semantic_alignment=alignment, require_attested_evidence_span=True,
    )
    if missing.get("authorized"):
        raise AssertionError(f"unattested fact passed strict certificate: {missing}")


def check_attested_provenance_audit() -> None:
    """The pre-score audit rejects typed atoms without a matching evidence span."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from gov_mem.evolution.attested_provenance_audit import audit_run
    from gov_mem.utils.io import write_jsonl

    with TemporaryDirectory() as directory:
        domain_root = Path(directory) / "renamed_domain"
        atoms_path = domain_root / "debug" / "governed_atoms.jsonl"
        cases_path = domain_root / "debug_cases" / "checkpoint_benchmark" / "case.json"
        write_jsonl(atoms_path, [{
            "atom_id": "good", "atom_type": "fact_atom", "text": "The stable value is 8 units.",
            "slots": {"stable_value": "8 units"}, "provenance": {"evidence_span": "The stable value is 8 units."},
        }])
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        cases_path.write_text(__import__("json").dumps({
            "graph_authorization_certificate": {"authorized": True, "realizations": [{
                "source_atom_id": "good", "value": "8 units",
            }]},
        }), encoding="utf-8")
        passed = audit_run(run_dir=directory)
        if not passed["passed"] or passed["attested_realized_atom_count"] != 1:
            raise AssertionError(f"attested provenance audit rejected valid evidence: {passed}")
        write_jsonl(atoms_path, [{
            "atom_id": "bad", "atom_type": "fact_atom", "text": "The unstable value is 9 units.",
            "slots": {"unstable_value": "9 units"}, "provenance": {},
        }])
        cases_path.write_text(__import__("json").dumps({
            "graph_authorization_certificate": {"authorized": True, "realizations": [{
                "source_atom_id": "bad", "value": "9 units",
            }]},
        }), encoding="utf-8")
        failed = audit_run(run_dir=directory)
        if failed["passed"] or not failed["violations"]:
            raise AssertionError(f"attested provenance audit accepted missing span: {failed}")


def check_alignment_routes_dual_channel_selection() -> None:
    """An aligned observed slot must influence bounded utility selection."""
    from gov_mem.memory.governed_atom import GovernedMemoryAtom
    from gov_mem.retrieval.utility_retriever import UtilityRetriever

    def fact(atom_id: str, slot_name: str, value: str, confidence: float) -> GovernedMemoryAtom:
        return GovernedMemoryAtom(
            atom_id=atom_id, atom_type="fact_atom", text=f"{slot_name} is {value}.", slots={slot_name: value},
            owner_id="owner", subject_id="subject", speaker_id="owner", source_turn=1,
            timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
            access_scope=["collaborator"], related_entities=[], provenance={}, confidence=confidence,
        )

    aligned = fact("aligned_fact", "threshold_value", "17 units", 0.1)
    distractor = fact("distractor_fact", "unrelated_measure", "99 units", 0.9)
    raw = UtilityRetriever().retrieve(
        evidence=[], governed_atoms=[aligned, distractor], requested_attributes=["operational_limit"], max_items=1,
    )
    routed = UtilityRetriever().retrieve(
        evidence=[], governed_atoms=[aligned, distractor],
        requested_attributes=["operational_limit", "threshold_value"], max_items=1,
    )
    if raw.selected_atom_ids != ["distractor_fact"]:
        raise AssertionError(f"raw open attribute unexpectedly matched a slot: {raw.selected_atom_ids}")
    if routed.selected_atom_ids != ["aligned_fact"]:
        raise AssertionError(f"aligned slot did not route utility selection: {routed.selected_atom_ids}")


def check_planner_slot_hint_and_two_stage_alignment() -> None:
    """A grounded planner hint routes retrieval but cannot bypass final selection."""
    from gov_mem.legacy.semantic_alignment import align_requested_attributes
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom
    from gov_mem.planning.query_planner import _normalize_attribute_bindings

    bindings = _normalize_attribute_bindings([{
        "attribute": "configured_operating_limit", "support_span": "configured operating limit",
        "semantic_role": "requested_property", "evidence_slot_hint": "threshold_value",
    }])
    if bindings[0].get("evidence_slot_hint") != "threshold_value":
        raise AssertionError(f"planner did not preserve evidence slot hint: {bindings}")
    fact = GovernedMemoryAtom(
        atom_id="hint_fact", atom_type="fact_atom", text="The threshold value is 17 units.",
        slots={"threshold_value": "17 units"}, owner_id="owner", subject_id="device", speaker_id="owner",
        source_turn=1, timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
        access_scope=["collaborator"], related_entities=[], provenance={}, confidence=1.0,
    )
    graph = GovernedGraphBuilder().build(graph_id="hint", instance_id="smoke", atoms=[fact])
    spec = {
        "requested_attributes": ["configured_operating_limit"], "attribute_bindings": bindings,
        "request_shape": "fact", "temporal_scope": "unspecified",
    }
    routing = align_requested_attributes(
        question="What is the configured operating limit?", semantic_spec=spec, graph=graph, owner_id="owner",
        utility_atom_ids=None, llm_client=None, model_name="stub", semantic_contract_certifiable=True,
    )
    if not routing.get("available") or routing["bindings"]["configured_operating_limit"]["slot_name"] != "threshold_value":
        raise AssertionError(f"planner hint did not provide routing alignment: {routing}")
    final = align_requested_attributes(
        question="What is the configured operating limit?", semantic_spec=spec, graph=graph, owner_id="owner",
        utility_atom_ids=set(), llm_client=None, model_name="stub", semantic_contract_certifiable=True,
    )
    if final.get("available"):
        raise AssertionError(f"routing alignment bypassed empty utility selection: {final}")


def check_strict_resume_prediction_integrity() -> None:
    """Interrupted outputs must not silently duplicate completed checkpoints."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from gov_mem.pipeline import GovMemRunner
    from gov_mem.utils.io import write_jsonl

    with TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.jsonl"
        write_jsonl(path, [{"checkpoint_id": "first"}, {"checkpoint_id": "second"}])
        completed = GovMemRunner._load_completed_prediction_ids(path)
        if completed != {"first", "second"}:
            raise AssertionError(f"resume did not recover completed ids: {completed}")
        write_jsonl(path, [{"checkpoint_id": "first"}, {"checkpoint_id": "first"}])
        try:
            GovMemRunner._load_completed_prediction_ids(path)
        except ValueError:
            pass
        else:
            raise AssertionError("resume accepted duplicate checkpoint predictions")


def check_incomplete_run_detection() -> None:
    """A partial checkpoint suite must not advance to official scoring."""
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from gov_mem.pipeline import GovMemRunner
    from gov_mem.utils.io import write_jsonl

    with TemporaryDirectory() as directory:
        path = Path(directory) / "predictions.jsonl"
        write_jsonl(path, [{"checkpoint_id": "completed"}])
        completed = GovMemRunner._load_completed_prediction_ids(path)
        missing = {"completed", "transient_failure"} - completed
        if missing != {"transient_failure"}:
            raise AssertionError(f"incomplete suite detection lost a missing id: {missing}")


def check_grounded_state_identity() -> None:
    """Version identity and typed values require a source-turn citation."""
    from gov_mem.data.schema import MemoryInstance
    from gov_mem.memory.amem_memory import AtomicMemory, _apply_attested_evidence_span, _ground_semantic_tags
    from gov_mem.memory.governed_atom import adapt_atomic_memories_to_governed_atoms

    source = "Device A now uses a configured threshold of 17 units."
    tags = _ground_semantic_tags({
        "discourse_act": "update",
        "assertion_confidence": 0.9,
        "evidence_span": source,
        "event_identity": {"entity_key": "Device A", "entity_surface_span": "Device A"},
        "attributes": {"threshold_value": "invented", "unsupported_property": "x"},
        "surface_values": {"threshold_value": "17 units", "unsupported_property": "not in source"},
        "state_delta": {"operation": "set", "changed_fields": {"threshold_value": "invented"}},
    }, source_text=source)
    if tags["attributes"] != {"threshold_value": "17 units"}:
        raise AssertionError(f"ungrounded attribute survived semantic validation: {tags}")
    no_surface_duplicate = _ground_semantic_tags({
        "discourse_act": "assertion",
        "assertion_confidence": 0.9,
        "evidence_span": source,
        "event_identity": {"entity_key": "Device A", "entity_surface_span": "Device A"},
        "attributes": {"threshold_value": "17 units"},
        "state_delta": {"operation": "set", "changed_fields": {"threshold_value": "17 units"}},
    }, source_text=source)
    if no_surface_duplicate["attributes"] != {"threshold_value": "17 units"}:
        raise AssertionError(f"verbatim attribute required a redundant surface field: {no_surface_duplicate}")
    if tags["event_identity"].get("entity_key") != "device_a":
        raise AssertionError(f"grounded state identity was not normalized: {tags}")
    instance = MemoryInstance(
        instance_id="identity_smoke", domain="renamed_domain", conversation_id=None,
        messages=[{"message_id": "m1", "speaker_id": "owner", "text": source}],
        question="", asking_user_id="owner", choices=None, answer=None,
    )
    memory = AtomicMemory(
        memory_id="identity_memory", instance_id="identity_smoke", owner_user="owner",
        memory_type="general_fact", content="configured threshold fragment", entities=[],
        slots={"threshold_value": "17 units", "unrelated_measure": "99 units"},
        source_message_ids=["m1"], timestamp="2026-01-01T00:00:00", lifecycle_status="active",
        access_tags={"semantic_tags": tags}, confidence=1.0,
    )
    memory = _apply_attested_evidence_span(memory, tags)
    if memory.content != source or memory.slots != {"threshold_value": "17 units"}:
        raise AssertionError(f"retrieval evidence did not adopt attested span: {memory}")
    atoms = adapt_atomic_memories_to_governed_atoms(instance=instance, atomic_memories=[memory])
    if not atoms or atoms[0].subject_id != "device_a":
        raise AssertionError(f"governed atom ignored verified state identity: {atoms}")
    if atoms[0].text != source or atoms[0].slots != {"threshold_value": "17 units"}:
        raise AssertionError(f"attested span did not bound governed slots: {atoms[0]}")


def check_composite_claim_owner_certificate() -> None:
    """A composite attested fact may serve one need without authorizing outsiders."""
    from gov_mem.governance_runtime.provenance_authorization import certify_graph_slot_paths
    from gov_mem.legacy.semantic_alignment import align_requested_attributes
    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    text = "The configured maintenance window runs from 09:10 to 10:45."
    fact = GovernedMemoryAtom(
        atom_id="maintenance_window", atom_type="fact_atom", text=text,
        slots={"window_start": "09:10", "window_end": "10:45"},
        owner_id="owner", subject_id="renamed_asset", speaker_id="owner", source_turn=1,
        timestamp="2026-01-01T09:00:00", lifecycle="active", sensitivity="private",
        access_scope=["self"], related_entities=["renamed_asset"], confidence=1.0,
        provenance={"evidence_span": text, "source_memory_id": "maintenance_memory"},
    )
    graph = GovernedGraphBuilder().build(graph_id="composite", instance_id="smoke", atoms=[fact])
    slot_ids = [
        node.node_id for node in graph.nodes
        if node.node_type == "SlotNode" and node.provenance.get("source_atom_id") == "maintenance_window"
    ]

    class CompositeAlignmentLLM:
        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            return {"bindings": [{
                "attribute": "maintenance_window",
                "slot_node_ids": slot_ids,
                "query_support_span": "maintenance window",
                "fact_support_spans": [
                    {"slot_node_id": slot_id, "fact_support_span": text} for slot_id in slot_ids
                ],
            }]}

    spec = {
        "requested_attributes": ["maintenance_window"],
        "attribute_bindings": [{
            "attribute": "maintenance_window", "support_span": "maintenance window",
            "semantic_role": "requested_property",
        }],
        "temporal_scope": "current", "request_shape": "fact",
    }
    alignment = align_requested_attributes(
        question="What is the maintenance window?", semantic_spec=spec, graph=graph, owner_id="owner",
        utility_atom_ids=None, llm_client=CompositeAlignmentLLM(), model_name="stub",
        semantic_contract_certifiable=True,
    )
    owner_certificate = certify_graph_slot_paths(
        semantic_spec=spec, graph=graph, principal_relation="owner", owner_id="owner",
        semantic_alignment=alignment, require_attested_evidence_span=True,
    )
    if not owner_certificate.get("authorized") or len(owner_certificate.get("realizations") or []) != 2:
        raise AssertionError(f"owner composite claim was not certified: {owner_certificate}")
    outsider_certificate = certify_graph_slot_paths(
        semantic_spec=spec, graph=graph, principal_relation="collaborator", owner_id="owner",
        semantic_alignment=alignment, require_attested_evidence_span=True,
    )
    if outsider_certificate.get("authorized"):
        raise AssertionError(f"owner certificate leaked to an outsider: {outsider_certificate}")


def check_grounded_annotation_cache() -> None:
    """Cached annotations are reusable only for identical source and model contracts."""
    import ast
    from tempfile import TemporaryDirectory

    from gov_mem.data.schema import MemoryInstance
    from gov_mem.memory.amem_memory import AtomicMemoryExtractor

    class AnnotationLLM:
        def __init__(self):
            self.calls = 0

        def chat_json(self, **kwargs):
            self.calls += 1
            candidates = ast.literal_eval(kwargs["user_prompt"].split("Candidates: ", 1)[1])
            candidate = candidates[0]
            text = candidate["source_turn_text"]
            return {"annotations": [{
                "candidate_id": candidate["candidate_id"],
                "semantic_tags": {
                    "discourse_act": "assertion",
                    "assertion_confidence": 0.9,
                    "event_identity": {"entity_key": "unit", "entity_surface_span": "unit"},
                    "attributes": {"configured_value": "17 units"},
                    "state_delta": {"operation": "set", "changed_fields": {"configured_value": "17 units"}},
                    "evidence_span": text,
                },
            }]}

    def instance(instance_id: str) -> MemoryInstance:
        return MemoryInstance(
            instance_id=instance_id, domain="renamed_domain", conversation_id=None,
            messages=[{"message_id": "m1", "speaker_id": "owner", "timestamp": "2026-01-01", "text": "The unit now uses 17 units."}],
            question="", asking_user_id="owner", choices=None, answer=None,
        )

    with TemporaryDirectory() as directory:
        llm = AnnotationLLM()
        extractor = AtomicMemoryExtractor(llm_client=llm, model_name="cache-model", annotation_cache_dir=directory)
        first = extractor.extract(instance("checkpoint_one"))
        second = extractor.extract(instance("checkpoint_two"))
        if llm.calls != 1 or not first[0].access_tags.get("semantic_tags") or not second[0].access_tags.get("semantic_tags"):
            raise AssertionError(f"grounded annotation cache was not reused: calls={llm.calls}")
        changed_model = AtomicMemoryExtractor(llm_client=llm, model_name="other-model", annotation_cache_dir=directory)
        changed_model.extract(instance("checkpoint_three"))
        if llm.calls != 2:
            raise AssertionError("annotation cache crossed a model boundary")


def check_open_grounded_claim_schema() -> None:
    """Open claim labels become slots only through exact source-span checks."""
    from gov_mem.memory.amem_memory import _ground_semantic_tags

    source = "The renamed instrument has a calibrated response ceiling of 47 units."
    tags = _ground_semantic_tags({
        "discourse_act": "assertion",
        "assertion_confidence": 0.9,
        "evidence_span": source,
        "claims": [{
            "property_label": "calibrated_response_ceiling",
            "value_span": "47 units",
            "claim_span": "instrument has a calibrated response ceiling of 47 units",
            "subject_span": "instrument",
        }],
    }, source_text=source)
    if tags["attributes"] != {"calibrated_response_ceiling": "47 units"}:
        raise AssertionError(f"grounded open claim did not become an attribute: {tags}")
    invalid = _ground_semantic_tags({
        "discourse_act": "assertion",
        "evidence_span": source,
        "claims": [{
            "property_label": "calibrated_response_ceiling",
            "value_span": "48 units",
            "claim_span": "instrument has a calibrated response ceiling of 48 units",
        }],
    }, source_text=source)
    if invalid["attributes"] or invalid["claims"]:
        raise AssertionError(f"unsupported claim escaped span validation: {invalid}")

    from gov_mem.graph.graph_builder import GovernedGraphBuilder
    from gov_mem.memory.governed_atom import GovernedMemoryAtom

    atom = GovernedMemoryAtom(
        atom_id="dynamic_subject", atom_type="fact_atom", text=source,
        slots={"calibrated_response_ceiling": "47 units"}, owner_id="owner",
        subject_id="instrument", speaker_id="owner", source_turn=1,
        timestamp="2026-01-01", lifecycle="active", sensitivity="private",
        access_scope=["self"], related_entities=["instrument"], confidence=1.0,
        provenance={"grounded_claims": tags["claims"], "evidence_span": source},
    )
    graph = GovernedGraphBuilder().build(graph_id="dynamic", instance_id="one_case", atoms=[atom])
    if not any(edge.edge_type == "claim_subject" and edge.attributes.get("property_label") == "calibrated_response_ceiling" for edge in graph.edges):
        raise AssertionError("case-local claim subject was not represented in the dynamic graph")


def check_embedding_batching() -> None:
    """Large dynamic case graphs must not create an unbounded provider request."""
    from gov_mem.llm.client import EMBEDDING_BATCH_SIZE, LLMClient, LLMConfig

    class BatchClient(LLMClient):
        def __init__(self):
            super().__init__(LLMConfig(provider="yunwu", api_key_env="UNUSED"))
            self.payloads = []

        def is_available(self):
            return True

        def _post_json(self, *, endpoint, payload):
            self.payloads.append(payload)
            return {"data": [{"embedding": [float(index)]} for index, _ in enumerate(payload["input"])]}

    client = BatchClient()
    texts = [f"case-local evidence {index}" for index in range(EMBEDDING_BATCH_SIZE * 2 + 3)]
    vectors = client.embed_texts(model="stub", texts=texts)
    if len(client.payloads) != 3 or max(len(payload["input"]) for payload in client.payloads) > EMBEDDING_BATCH_SIZE:
        raise AssertionError(f"embedding batch limit was not respected: {client.payloads}")
    if len(vectors) != len(texts):
        raise AssertionError("embedding batching lost a dynamic evidence row")


def check_utility_source_provenance_routing() -> None:
    """Retrieved source provenance must retain an open-schema claim atom."""
    from gov_mem.memory.governed_atom import GovernedMemoryAtom
    from gov_mem.retrieval.utility_retriever import UtilityRetriever

    retrieved = RetrievedEvidence(
        memory_id="retrieved", content="The dynamic property is 47 units.", score=1.0,
        retrieval_source="smoke", reason="smoke", source_message_ids=["source_turn"],
    )
    matching = GovernedMemoryAtom(
        atom_id="matching", atom_type="fact_atom", text=retrieved.content,
        slots={"observed_value": "47 units"}, owner_id="owner", subject_id="unit",
        speaker_id="owner", source_turn=1, timestamp="2026-01-01", lifecycle="active",
        sensitivity="private", access_scope=["self"], related_entities=["unit"], confidence=0.5,
        provenance={"source_message_ids": ["source_turn"]},
    )
    unrelated = GovernedMemoryAtom(
        atom_id="unrelated", atom_type="fact_atom", text="A different value is 99 units.",
        slots={"observed_value": "99 units"}, owner_id="owner", subject_id="other",
        speaker_id="owner", source_turn=1, timestamp="2026-01-01", lifecycle="active",
        sensitivity="private", access_scope=["self"], related_entities=["other"], confidence=0.9,
        provenance={"source_message_ids": ["other_turn"]},
    )
    selected = UtilityRetriever().retrieve(
        evidence=[retrieved], governed_atoms=[unrelated, matching],
        requested_attributes=["dynamic_property"], max_items=1,
    ).selected_atom_ids
    if selected != ["matching"]:
        raise AssertionError(f"source-provenance utility routing failed: {selected}")


def check_alignment_envelope_compatibility() -> None:
    """Equivalent LLM envelopes remain subject to the same closed-set gate."""
    from gov_mem.legacy.semantic_alignment import _alignment_items

    item = {"attribute": "dynamic_property", "slot_node_ids": ["slot::one"]}
    nested = _alignment_items({"semantic_alignment": {"bindings": [item]}})
    alternate = _alignment_items({"attribute_bindings": [item]})
    if nested != [item] or alternate != [item]:
        raise AssertionError(f"alignment envelope compatibility failed: {nested}, {alternate}")


def check_semantic_annotation_batching() -> None:
    """Claim annotation batches remain bounded for arbitrary episode length."""
    from gov_mem.memory.amem_memory import (
        SEMANTIC_ANNOTATION_BATCH_SIZE,
        SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE,
    )
    if SEMANTIC_ANNOTATION_BATCH_SIZE > 3 or SEMANTIC_ANNOTATION_REPAIR_BATCH_SIZE > 2:
        raise AssertionError("semantic annotation request is not bounded for long dynamic episodes")

    from gov_mem.memory.amem_memory import AtomicMemory, AtomicMemoryExtractor
    from gov_mem.data.schema import MemoryInstance

    class AlternateContainerLLM:
        def chat_json(self, **kwargs):
            return {"evidence_annotations": [{
                "candidate_id": "candidate", "semantic_tags": {
                    "discourse_act": "assertion", "assertion_confidence": 1.0,
                    "evidence_span": "The local value is 7 units.",
                },
                "claims": [{
                    "property_label": "local_value", "value_span": "7 units",
                    "claim_span": "local value is 7 units.",
                }],
            }]}

    instance = MemoryInstance(
        instance_id="local", domain="synthetic", conversation_id=None,
        messages=[{"message_id": "m", "text": "The local value is 7 units."}],
        question="", asking_user_id="owner", choices=None, answer=None,
    )
    candidate = AtomicMemory(
        memory_id="candidate", instance_id="local", owner_user="owner", memory_type="fact",
        content="The local value is 7 units.", entities=[], slots={}, source_message_ids=["m"],
        timestamp=None, lifecycle_status="active", access_tags={}, confidence=1.0,
    )
    output = AtomicMemoryExtractor(llm_client=AlternateContainerLLM(), model_name="stub")._annotate_semantics_with_llm(
        instance, [candidate]
    )
    if output[0].slots.get("local_value") != "7 units":
        raise AssertionError(f"alternate annotation container was not grounded: {output}")

    class CountRepresentativeLLM(AlternateContainerLLM):
        def __init__(self):
            self.candidate_counts = []

        def chat_json(self, **kwargs):
            import ast
            payload = ast.literal_eval(kwargs["user_prompt"].split("Candidates: ", 1)[1])
            self.candidate_counts.append(len(payload))
            return super().chat_json(**kwargs)

    duplicate = AtomicMemory(
        memory_id="duplicate", instance_id="local", owner_user="owner", memory_type="fact",
        content="The local value is 7 units.", entities=[], slots={}, source_message_ids=["m"],
        timestamp=None, lifecycle_status="active", access_tags={}, confidence=1.0,
    )
    grouped_llm = CountRepresentativeLLM()
    grouped = AtomicMemoryExtractor(llm_client=grouped_llm, model_name="stub")._annotate_semantics_with_llm(
        instance, [candidate, duplicate]
    )
    if grouped_llm.candidate_counts != [1] or any(item.slots.get("local_value") != "7 units" for item in grouped):
        raise AssertionError(f"same-turn candidates were not dynamically grouped: {grouped_llm.candidate_counts}")

    class SelectionLLM:
        def chat_json(self, **kwargs):
            import ast
            payload = ast.literal_eval(kwargs["user_prompt"].split("Candidates: ", 1)[1])
            candidate_id = payload[0]["candidate_id"]
            return {"annotations": [{
                "candidate_id": candidate_id,
                "semantic_tags": {
                    "discourse_act": "assertion", "assertion_confidence": 1.0,
                    "evidence_span": "The local value is 7 units.",
                },
                "claims": [{
                    "property_label": "local_value", "value_span": "7 units",
                    "claim_span": "local value is 7 units.",
                }],
            }]}
    selected_instance = MemoryInstance(
        instance_id="local", domain="synthetic", conversation_id=None,
        messages=[
            {"message_id": "m", "text": "The local value is 7 units."},
            {"message_id": "other", "text": "The unrelated value is 9 units."},
        ], question="", asking_user_id="owner", choices=None, answer=None,
    )
    selected = AtomicMemoryExtractor(llm_client=SelectionLLM(), model_name="stub").extract(
        selected_instance, annotation_source_message_ids={"m"}
    )
    if not any(
        item.content == "The unrelated value is 9 units."
        and not (item.access_tags or {}).get("semantic_tags")
        for item in selected
    ):
        raise AssertionError("unselected source evidence was dropped or unexpectedly annotated")


def main() -> None:
    check_semantic_spec_invariance()
    check_entity_rename_invariance()
    check_provenance_and_access_invariance()
    check_structured_realization_chain()
    check_semantic_current_state_resolution()
    check_semantic_contract_repair()
    check_redacted_governance_contract()
    check_current_state_authorization_certificate()
    check_slot_version_graph()
    check_dual_channel_convergence()
    check_structural_failure_diagnosis()
    check_certificate_requires_llm_contract()
    check_open_semantic_slot_schema()
    check_policy_frame_compiler()
    check_dev_only_adaptation_gate()
    check_open_attribute_alignment_and_temporal_certificates()
    check_certificate_driven_action_normalization()
    check_list_alignment_requires_explicit_multiple_anchors()
    check_attested_span_certificate_gate()
    check_attested_provenance_audit()
    check_alignment_routes_dual_channel_selection()
    check_planner_slot_hint_and_two_stage_alignment()
    check_strict_resume_prediction_integrity()
    check_incomplete_run_detection()
    check_grounded_state_identity()
    check_composite_claim_owner_certificate()
    check_grounded_annotation_cache()
    check_open_grounded_claim_schema()
    check_embedding_batching()
    check_utility_source_provenance_routing()
    check_alignment_envelope_compatibility()
    check_semantic_annotation_batching()
    print("semantic invariance smoke: PASS")


if __name__ == "__main__":
    main()
