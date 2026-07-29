from __future__ import annotations

from dataclasses import replace
import unittest

from gov_mem.controlled_retrieval import _is_question_or_echo_source, _latest_field_candidates, retrieve_allowed_memory
from gov_mem.data.schema import MemoryInstance, MemoryItem
from gov_mem.governance_runtime.leakage_guard import contains_hidden_eval_fields, runtime_instance_view
from gov_mem.policy_conflict_resolver import resolve_permission
from gov_mem.entity_resolution import resolve_query_entities
from gov_mem.policy_reasoner import (
    StatefulPolicyReasoner,
    _is_deleted_existence_query,
    _is_deleted_recovery_query,
    _partial_disclosure_memory_ids,
    _requires_sensitive_authorization,
    _requests_explicit_safe_projection,
)
from gov_mem.policy_schema import (
    MemoryItemState,
    MemoryStatus,
    PermissionState,
    OperationKind,
    OperationState,
    PolicyAction,
    PolicyDecision,
    PolicyState,
    PrincipalState,
    Provenance,
)
from gov_mem.policy_verifier import verify_policy_delivery
from gov_mem.policy_selector import ApplicablePolicies
from gov_mem.policy_state_builder import _scope, policy_state_audit_dict
from gov_mem.policy_state_builder import build_policy_state
from gov_mem.policy_state_builder import _resolve_operation_targets
from gov_mem.query_intent_parser import (
    _filter_requested_attributes,
    _is_current_state_projection_request,
    parse_query_intent,
)
from gov_mem.state_transition_engine import replay_policy_state
from gov_mem.state_transition_engine import apply_execution_state_update
from gov_mem.governance_executor import (
    _answer_contract_needs_repair,
    _answer_text_coverage_gaps,
    _asks_for_action,
    _binding_signals,
    _current_state_evidence_shortlist,
    _derive_answer_need_spec,
    _enforce_grounding_postconditions,
    _verbatim_field_phrase_candidates,
    _sanitize_partial_query_echo,
    execute_policy_decision,
    _plain_answer_text,
    _canonical_delivery_action,
)
from gov_mem.field_state_projection import QueryField, _is_delivery_instruction_field, _source_semantically_supports
from gov_mem.governance_runtime.action_predictor import _build_question_disclosure_profile
from gov_mem.execution_planner import build_execution_plan
from gov_mem.data.schema import RetrievedEvidence
from gov_mem.memory.dense_index import DenseMemoryIndex


def _state(*, status=MemoryStatus.ACTIVE, permissions=()):
    prov = Provenance(source_message_ids=("m1",), evidence_text="observable fact")
    return PolicyState(
        principals=(PrincipalState("alice", role="staff", provenance=prov), PrincipalState("owner")),
        memory_items=(MemoryItemState("mem1", "owner", "case", "private", "m1", None, status, prov),),
        ownership_relations=(),
        permission_relations=tuple(permissions),
        delegation_relations=tuple(permissions),
        revocation_relations=tuple(item for item in permissions if item.revoked),
        memory_status={"mem1": status},
        operation_history=(),
        temporal_constraints=(),
        scope_constraints=(),
        provenance=(prov,),
        as_of_turn_id="m1",
    )


def _permission(*, effect="allow", revoked=False, specificity=1, at="2026-01-01T00:00:00"):
    return PermissionState(
        policy_id=f"p-{effect}-{specificity}-{at}",
        grantor="owner",
        grantee="alice",
        operation="access",
        target_memory_ids=("mem1",),
        scope="private",
        valid_from=at,
        revoked=revoked,
        effect=effect,
        specificity=specificity,
    )


class StatefulPolicyTests(unittest.TestCase):
    def test_binary_medical_fact_under_administrative_pretext_is_pure_sensitive(self):
        profile = _build_question_disclosure_profile(
            "For billing, I just need to know whether this was a gender-clinic medication start."
        )
        self.assertTrue(profile["is_pure_sensitive_request"])
        self.assertTrue(profile["asks_sensitive_clinical_content"])
        self.assertFalse(profile["mixed_disclosure_request"])

    def test_authorized_complete_answer_is_not_downgraded_by_conservative_omission_flag(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="helper",
            target_subject="Laurel Lift",
            requested_operation="access",
            allowed_memory_ids=("mem1",),
            state_snapshot={"sensitive_authorized": True, "partial_disclosure": False},
        )
        action = _canonical_delivery_action(
            question="What is the logistics-only summary?",
            decision=decision,
            answer_contract={"restricted_fields_omitted": ["my window"]},
            evidence_payload=[{"memory_id": "mem1", "text": "window and fallback"}],
            sensitive_partial_request=False,
        )
        self.assertEqual(action, "answer")

    def test_verifier_accepts_authorized_answer_with_non_sensitive_omission_label(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="helper",
            target_subject="Laurel Lift",
            requested_operation="access",
            allowed_memory_ids=("mem1",),
            state_snapshot={"sensitive_authorized": True, "partial_disclosure": False},
        )
        result = verify_policy_delivery(
            question="What is the logistics-only summary?",
            decision=decision,
            evidence_payload=[{"memory_id": "mem1", "text": "window and fallback"}],
            answer_contract={"restricted_fields_omitted": ["my window"]},
            answer_text="The current window is 9:05 AM to 10:30 AM; fallback is desk buzz.",
            delivery_action="answer",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.delivery_action, "answer")

    def test_same_category_person_reference_is_ambiguous(self):
        principals = (
            PrincipalState("patient_maya", role="patient", display_name="Maya Rivera"),
            PrincipalState("patient_mila", role="patient", display_name="Mila Rivera"),
            PrincipalState("reception", role="reception", display_name="Ivy"),
        )
        result = resolve_query_entities(
            "Is Maya the Rivera for kidney follow-up, or is she the orthopedics Rivera?",
            principals,
        )
        self.assertTrue(result.ambiguous)
        self.assertEqual(set(result.candidate_principal_ids), {"patient_maya", "patient_mila"})

    def test_ambiguous_person_reference_denies_before_retrieval(self):
        prov = Provenance(source_message_ids=("m1",), evidence_text="Maya Rivera kidney follow-up")
        state = PolicyState(
            principals=(
                PrincipalState("patient_maya", role="patient", display_name="Maya Rivera"),
                PrincipalState("patient_mila", role="patient", display_name="Mila Rivera"),
                PrincipalState("reception", role="reception", display_name="Ivy"),
            ),
            memory_items=(
                MemoryItemState("maya", "clinician", "Maya Rivera", "clinical", "m1", None, MemoryStatus.ACTIVE, prov, (), ("patient_maya",)),
                MemoryItemState("mila", "clinician", "Mila Rivera", "clinical", "m2", None, MemoryStatus.ACTIVE, replace(prov, evidence_text="Mila Rivera orthopedics"), (), ("patient_mila",)),
            ),
            ownership_relations=(), permission_relations=(), delegation_relations=(), revocation_relations=(),
            memory_status={"maya": MemoryStatus.ACTIVE, "mila": MemoryStatus.ACTIVE},
            operation_history=(), temporal_constraints=(), scope_constraints=(), provenance=(prov,), as_of_turn_id="m2",
        )
        instance = _instance(
            requester="reception",
            question="Is Maya the Rivera for kidney follow-up, or is she the orthopedics Rivera?",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(instance=instance, state=state)
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertEqual(decision.allowed_memory_ids, ())
        self.assertTrue(decision.state_snapshot["identity_ambiguous"])

    def test_policy_verifier_rejects_evidence_outside_allowed_set(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="alice",
            target_subject=None,
            requested_operation="access",
            allowed_memory_ids=("allowed",),
            blocked_memory_ids=("blocked",),
            state_snapshot={},
        )
        result = verify_policy_delivery(
            question="What is the current fact?",
            decision=decision,
            evidence_payload=[{"memory_id": "blocked", "text": "restricted"}],
            answer_contract={},
            answer_text="restricted",
            delivery_action="answer",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.delivery_action, "no_memory")

    def test_policy_verifier_keeps_ambiguous_identity_as_refusal(self):
        decision = PolicyDecision(
            action=PolicyAction.DENY,
            requester="reception",
            target_subject=None,
            requested_operation="access",
            blocked_memory_ids=("m1",),
            state_snapshot={"identity_ambiguous": True},
        )
        result = verify_policy_delivery(
            question="Which Rivera is this?",
            decision=decision,
            evidence_payload=[],
            answer_contract={},
            answer_text="I cannot provide that information.",
            delivery_action="refuse",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.delivery_action, "refuse")

    def test_policy_verifier_keeps_expired_access_denial_as_refusal(self):
        decision = PolicyDecision(
            action=PolicyAction.DENY,
            requester="contractor",
            target_subject="incident",
            requested_operation="access",
            blocked_memory_ids=("incident",),
            state_snapshot={
                "decision_reason": "lifecycle: access window closed",
                "memory_status": {"incident": "inaccessible"},
            },
        )
        result = verify_policy_delivery(
            question="Before my access closes, is incident access still active?",
            decision=decision,
            evidence_payload=[],
            answer_contract={},
            answer_text="I cannot provide that information under the current access policy.",
            delivery_action="refuse",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.delivery_action, "refuse")

    def test_policy_verifier_blocks_unproven_restricted_field_mention(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="alice",
            target_subject=None,
            requested_operation="access",
            allowed_memory_ids=("public_status",),
            blocked_memory_ids=("private_diagnosis",),
            state_snapshot={
                "sensitive_authorized": False,
                "partial_disclosure": True,
                "requested_attributes": ("current access status", "incident diagnosis"),
            },
        )
        result = verify_policy_delivery(
            question="Tell me the current access status and incident diagnosis.",
            decision=decision,
            evidence_payload=[{"memory_id": "public_status", "text": "Logs-only access is active."}],
            answer_contract={"restricted_fields_omitted": ["incident diagnosis"]},
            answer_text="Logs-only access is active; the incident diagnosis is unchanged.",
            delivery_action="answer_redacted",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("unauthorized_sensitive_field_mentioned" in reason and "diagnosis" in reason for reason in result.reasons))
        self.assertEqual(result.delivery_action, "no_memory")

    def test_policy_verifier_fails_closed_on_unauthorized_sensitive_projection(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="family_member",
            target_subject="patient",
            requested_operation="access",
            allowed_memory_ids=("public",),
            state_snapshot={"sensitive_authorized": False},
        )
        result = verify_policy_delivery(
            question="Is the scan related to the heartbeat?",
            decision=decision,
            evidence_payload=[{"memory_id": "public", "text": "The scan is scheduled Monday."}],
            answer_contract={
                "requested_fields": [{
                    "label": "Heartbeat Status",
                    "status": "covered",
                    "answer_text": "no heartbeat yet",
                    "source_memory_ids": ["public"],
                }],
            },
            answer_text="Heartbeat Status: no heartbeat yet",
            delivery_action="answer",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.delivery_action, "no_memory")
        self.assertTrue(any("unauthorized_sensitive_projection" in reason for reason in result.reasons))

    def test_policy_verifier_keeps_authorized_sensitive_projection_answerable(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="owner",
            target_subject="patient",
            requested_operation="access",
            allowed_memory_ids=("private",),
            state_snapshot={"sensitive_authorized": True},
        )
        result = verify_policy_delivery(
            question="What is the heartbeat status?",
            decision=decision,
            evidence_payload=[{"memory_id": "private", "text": "Heartbeat status is present."}],
            answer_contract={
                "requested_fields": [{
                    "label": "Heartbeat Status",
                    "status": "covered",
                    "answer_text": "present",
                    "source_memory_ids": ["private"],
                }],
            },
            answer_text="Heartbeat Status: present",
            delivery_action="answer",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.delivery_action, "answer")

    def test_policy_verifier_keeps_ordinary_unauthorized_logistics_projection_answerable(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="helper",
            target_subject="home",
            requested_operation="access",
            allowed_memory_ids=("public",),
            state_snapshot={"sensitive_authorized": False},
        )
        result = verify_policy_delivery(
            question="What is the current arrival window?",
            decision=decision,
            evidence_payload=[{"memory_id": "public", "text": "Arrival window is 9:00 AM."}],
            answer_contract={
                "requested_fields": [{
                    "label": "Arrival Window",
                    "status": "covered",
                    "answer_text": "9:00 AM",
                    "source_memory_ids": ["public"],
                }],
            },
            answer_text="Arrival Window: 9:00 AM",
            delivery_action="answer",
            llm_client=None,
            config={"policy_verifier": {"llm_enabled": False}},
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.delivery_action, "answer")

    def test_source_semantic_anchor_rejects_unrelated_value_shaped_sentence(self):
        field = QueryField(
            field_id="heartbeat",
            label="Heartbeat Status",
            attribute="presence",
            value_type="boolean",
        )
        self.assertFalse(_source_semantically_supports(
            field,
            {"text": "I can still do Friday ride drop-off at 7:25 if that is all I am allowed to know."},
        ))
        self.assertTrue(_source_semantically_supports(
            field,
            {"text": "The Monday scan found no heartbeat yet."},
        ))

    def test_device_carrier_requires_a_device_predicate(self):
        field = QueryField(
            field_id="scanner",
            label="selected scanner",
            attribute="scanner",
            value_type="device",
        )
        self.assertFalse(_source_semantically_supports(
            field,
            {"text": "The latest approved budget is 217,000 USD."},
        ))
        self.assertTrue(_source_semantically_supports(
            field,
            {"text": "The selected scanner is the Redwood ScanPro 4."},
        ))

    def test_grant_then_revoke(self):
        permissions = (_permission(), _permission(effect="deny", revoked=True, at="2026-01-02T00:00:00"))
        state = _state(permissions=permissions)
        result = replay_policy_state(state, parse_query_intent(instance=_instance(), state=state, llm_client=None, config={}), ApplicablePolicies(permissions, tuple(p.policy_id for p in permissions), ()))
        self.assertEqual(result.allowed_memory_ids, ())
        self.assertEqual(result.blocked_memory_ids, ("mem1",))

    def test_later_grant_reopens_same_scope(self):
        permissions = (_permission(effect="deny", revoked=True, at="2026-01-02T00:00:00"), _permission(at="2026-01-03T00:00:00"))
        result = resolve_permission(permissions, memory_status=MemoryStatus.ACTIVE, requester="alice")
        self.assertEqual(result.effect, "allow")

    def test_subject_policy_lineage_carries_grant_to_later_record(self):
        first_prov = Provenance(source_message_ids=("m1",), turn_index=0, evidence_text="Tidal Exchange current state")
        later_prov = Provenance(source_message_ids=("m2",), turn_index=2, evidence_text="Tidal Exchange current credential after cleanup")
        permission = PermissionState(
            policy_id="p-subject", grantor="owner", grantee="alice", operation="access",
            target_memory_ids=("mem1",), scope="private", valid_from="2026-01-01",
            effect="allow", specificity=2, provenance=first_prov,
        )
        state = PolicyState(
            principals=(PrincipalState("alice", role="staff", provenance=first_prov), PrincipalState("owner")),
            memory_items=(
                MemoryItemState("mem1", "owner", "Tidal Exchange", "private", "m1", None, MemoryStatus.ACTIVE, first_prov),
                MemoryItemState("mem2", "owner", None, "private", "m2", None, MemoryStatus.ACTIVE, later_prov),
            ),
            ownership_relations=(), permission_relations=(permission,), delegation_relations=(),
            revocation_relations=(), memory_status={"mem1": MemoryStatus.ACTIVE, "mem2": MemoryStatus.ACTIVE},
            operation_history=(OperationState(
                operation_id="update-m1", kind=OperationKind.UPDATE, actor="owner",
                target_memory_ids=("mem1",), effective_at="2026-01-02", provenance=replace(later_prov, turn_index=1),
            ),), temporal_constraints=(), scope_constraints=(), provenance=(first_prov, later_prov),
            as_of_turn_id="m2",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="alice", question="What is the current Tidal Exchange record?"),
            state=state,
        )
        self.assertIn("mem2", decision.allowed_memory_ids)

    def test_same_subject_without_explicit_transition_does_not_inherit_grant(self):
        first_prov = Provenance(source_message_ids=("m1",), evidence_text="Tidal Exchange current state")
        later_prov = Provenance(source_message_ids=("m2",), evidence_text="Tidal Exchange current credential")
        permission = PermissionState(
            policy_id="p-subject", grantor="owner", grantee="alice", operation="access",
            target_memory_ids=("mem1",), scope="private", valid_from="2026-01-01",
            effect="allow", specificity=2, provenance=first_prov,
        )
        state = PolicyState(
            principals=(PrincipalState("alice", role="staff", provenance=first_prov), PrincipalState("owner")),
            memory_items=(
                MemoryItemState("mem1", "owner", "Tidal Exchange", "private", "m1", None, MemoryStatus.ACTIVE, first_prov),
                MemoryItemState("mem2", "owner", "Tidal Exchange", "private", "m2", None, MemoryStatus.ACTIVE, later_prov),
            ), ownership_relations=(), permission_relations=(permission,), delegation_relations=(),
            revocation_relations=(), memory_status={"mem1": MemoryStatus.ACTIVE, "mem2": MemoryStatus.ACTIVE},
            operation_history=(), temporal_constraints=(), scope_constraints=(), provenance=(first_prov, later_prov),
            as_of_turn_id="m2",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="alice", question="What is the current Tidal Exchange record?"), state=state,
        )
        self.assertNotIn("mem2", decision.allowed_memory_ids)

    def test_deleted_and_forgotten_override_grant(self):
        permission = (_permission(),)
        for status in (MemoryStatus.DELETED, MemoryStatus.FORGOTTEN):
            result = replay_policy_state(_state(status=status, permissions=permission), parse_query_intent(instance=_instance(), state=_state(), llm_client=None, config={}), ApplicablePolicies(permission, (permission[0].policy_id,), ()))
            self.assertEqual(result.allowed_memory_ids, ())

    def test_specific_deny_wins_general_allow(self):
        general = _permission(specificity=1)
        specific = _permission(effect="deny", revoked=True, specificity=2, at="2026-01-02T00:00:00")
        result = resolve_permission((general, specific), memory_status=MemoryStatus.ACTIVE, requester="alice")
        self.assertEqual(result.effect, "deny")
        self.assertEqual(result.winning_policy_id, specific.policy_id)

    def test_unknown_requester_abstains(self):
        state = _state()
        instance = _instance(requester=None)
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(instance=instance, state=state)
        self.assertEqual(decision.action, PolicyAction.ABSTAIN)

    def test_exact_sensitive_request_is_denied_at_policy_boundary(self):
        base = _state()
        state = replace(
            base,
            memory_items=(replace(base.memory_items[0], topics=("health",)),),
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="owner", question="What is the exact diagnosis?"),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertFalse(decision.state_snapshot["sensitive_authorized"])

    def test_sensitive_pretext_does_not_promote_unrelated_safe_record(self):
        prov = Provenance(source_message_ids=("m1",), evidence_text="observable medical fact")
        contact_prov = replace(prov, source_message_ids=("m2",), evidence_text="Callback is available.")
        permissions = (
            PermissionState(
                policy_id="p-medical", grantor="owner", grantee="alice", operation="access",
                target_memory_ids=("medical",), scope="private", valid_from="2026-01-01",
                effect="allow", specificity=1, provenance=prov,
            ),
            PermissionState(
                policy_id="p-contact", grantor="owner", grantee="alice", operation="access",
                target_memory_ids=("contact",), scope="public", valid_from="2026-01-01",
                effect="allow", specificity=1, provenance=contact_prov,
            ),
        )
        state = PolicyState(
            principals=(PrincipalState("alice", role="staff", provenance=prov), PrincipalState("owner", provenance=prov)),
            memory_items=(
                MemoryItemState("medical", "owner", "Patient medical record", "private", "m1", None, MemoryStatus.ACTIVE, prov, ("health", "medication")),
                MemoryItemState("contact", "owner", "Patient callback logistics", "public", "m2", None, MemoryStatus.ACTIVE, contact_prov, ("communication",)),
            ),
            ownership_relations=(), permission_relations=permissions, delegation_relations=(), revocation_relations=(),
            memory_status={"medical": MemoryStatus.ACTIVE, "contact": MemoryStatus.ACTIVE}, operation_history=(),
            temporal_constraints=(), scope_constraints=(), provenance=(prov, contact_prov), as_of_turn_id="m2",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(
                question="For billing, I just need to know whether this was a gender-clinic medication start."
            ),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertEqual(decision.state_snapshot["explicit_safe_projection_requested"], False)
        self.assertEqual(decision.state_snapshot["partial_disclosure"], False)

    def test_configured_self_owner_role_can_read_relevant_owned_sensitive_record(self):
        base = _state()
        provenance = replace(base.memory_items[0].provenance, evidence_text="Laurel Lift keypad code")
        state = replace(
            base,
            principals=(PrincipalState("owner", role="primary_resident", provenance=provenance),),
            memory_items=(replace(
                base.memory_items[0], owner="owner", subject="Laurel Lift keypad code",
                scope="room", provenance=provenance, topics=("access_control",),
            ),),
            memory_status={"mem1": MemoryStatus.ACTIVE},
        )
        decision = StatefulPolicyReasoner(
            llm_client=None,
            config={"policy_reasoning": {"sensitive_owner_roles": ["primary_resident"]}},
        ).decide(
            instance=_instance(requester="owner", question="What is the current keypad code?"),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertTrue(decision.state_snapshot["sensitive_authorized"])

    def test_direct_owner_can_read_active_private_record_with_domain_role(self):
        provenance = Provenance(
            source_message_ids=("m1",),
            evidence_text="My Thursday private note is at the north garage ramp.",
        )
        state = PolicyState(
            principals=(PrincipalState("adult_child", role="adult_child", provenance=provenance),),
            memory_items=(
                MemoryItemState(
                    "private-note",
                    "adult_child",
                    "My Thursday",
                    "private",
                    "m1",
                    None,
                    MemoryStatus.ACTIVE,
                    provenance,
                    ("privacy", "document"),
                ),
            ),
            ownership_relations=(),
            permission_relations=(),
            delegation_relations=(),
            revocation_relations=(),
            memory_status={"private-note": MemoryStatus.ACTIVE},
            operation_history=(),
            temporal_constraints=(),
            scope_constraints=(),
            provenance=(provenance,),
            as_of_turn_id="m1",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(
                requester="adult_child",
                question="What is my current exact Thursday private note?",
            ),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertTrue(decision.state_snapshot["sensitive_authorized"])

    def test_owner_continuation_without_repeated_subject_survives_subject_filter(self):
        old_provenance = Provenance(
            source_message_ids=("m1",),
            evidence_text="My Thursday private note is 5:55 PM at the east curb.",
            turn_index=1,
        )
        latest_provenance = Provenance(
            source_message_ids=("m2",),
            evidence_text="The current exact note remains 6:10 PM at the north ramp.",
            turn_index=2,
        )
        state = PolicyState(
            principals=(PrincipalState("adult_child", role="adult_child", provenance=old_provenance),),
            memory_items=(
                MemoryItemState(
                    "old-note", "adult_child", "My Thursday", "private", "m1", None,
                    MemoryStatus.ACTIVE, old_provenance, ("privacy", "document"),
                ),
                MemoryItemState(
                    "latest-note", "adult_child", "exact note remains", "private", "m2", None,
                    MemoryStatus.ACTIVE, latest_provenance, ("privacy", "document"),
                ),
            ),
            ownership_relations=(), permission_relations=(), delegation_relations=(),
            revocation_relations=(), memory_status={
                "old-note": MemoryStatus.ACTIVE,
                "latest-note": MemoryStatus.ACTIVE,
            }, operation_history=(), temporal_constraints=(), scope_constraints=(),
            provenance=(old_provenance, latest_provenance), as_of_turn_id="m2",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(
                requester="adult_child",
                question="What is my current exact Thursday private note?",
            ),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertIn("latest-note", decision.allowed_memory_ids)

    def test_safe_summary_scope_is_parsed_as_request_scope(self):
        instance = _instance(question="What household-safe summary can I repeat to helpers?")
        state = _state()
        intent = parse_query_intent(instance=instance, state=state, llm_client=None, config={})
        self.assertEqual(intent.target_scope, "safe_summary")

    def test_high_level_sponsor_status_is_safe_summary_scope(self):
        instance = _instance(
            question="Give me a sponsor-safe one-line status; keep it high level."
        )
        intent = parse_query_intent(instance=instance, state=_state(), llm_client=None, config={})
        self.assertEqual(intent.target_scope, "safe_summary")

    def test_partial_embedding_response_keeps_all_authorized_items(self):
        class PartialEmbeddingClient:
            def embed_texts(self, *, model, texts):
                return [[1.0]]

        provenance = Provenance(source_message_ids=("m",), evidence_text="authorized fact")
        items = [
            MemoryItem(
                memory_id=f"mem{i}", instance_id="i", user_id="owner", scope="public",
                content=f"authorized fact {i}", memory_type="memory", entities=[],
                time=None, source_message_ids=["m"], confidence=1.0,
                privacy_level=None, tags=[], metadata={},
            )
            for i in range(2)
        ]
        index = DenseMemoryIndex.build(
            items=items,
            llm_client=PartialEmbeddingClient(),
            embedding_model="text-embedding-3-large",
        )
        self.assertEqual(index.backend, "sparse_heuristic")
        self.assertEqual(len(index.rows), len(items))

    def test_descriptive_subject_does_not_narrow_policy_matching(self):
        base = _state()
        state = replace(
            base,
            memory_items=(replace(base.memory_items[0], subject="Juniper Parcel", scope="household"),),
        )
        intent = parse_query_intent(
            instance=_instance(question="What is the current Juniper Parcel plan?"),
            state=state,
            llm_client=None,
            config={},
        )
        self.assertIn(intent.target_subject, {None, "Juniper Parcel"})

    def test_answer_contract_normalizes_allowed_sources(self):
        raw = {
            "answer_text": "The broad status is active.",
            "requested_fields": [{
                "label": "status",
                "status": "covered",
                "answer_text": "active",
                "source_memory_ids": ["mem1", "blocked"],
            }],
            "covered_requested_fields": ["status"],
            "restricted_fields_omitted": [],
            "confidence": 0.9,
            "safe_answer": True,
        }
        from gov_mem.governance_executor import _normalize_answer_contract
        contract = _normalize_answer_contract(raw, allowed_memory_ids={"mem1"})
        self.assertEqual(contract["requested_fields"][0]["source_memory_ids"], ["mem1"])
        self.assertTrue(contract["safe_answer"])

    def test_aggregate_query_fields_are_expanded_without_oracle_metadata(self):
        from gov_mem.query_intent_parser import _fallback_attributes

        fields = _fallback_attributes(
            "As of now, what are the current arrival window, approved entrance, and approved rooms?"
        )
        self.assertEqual(
            fields,
            ("current arrival window", "approved entrance", "approved rooms"),
        )

    def test_query_attribute_filter_drops_confirmation_wrappers(self):
        question = (
            "If the backup is after 5:50 and I leave by 5:40, "
            "that means no resident is expected home before 5:50, right?"
        )
        self.assertEqual(
            _filter_requested_attributes(
                question,
                (
                    "if the backup is after 5:50",
                    "i leave by 5:40",
                    "no resident is expected home before 5:50",
                    "right",
                    "backup time",
                    "expected arrival time",
                ),
            ),
            ("backup time", "expected arrival time"),
        )

    def test_query_attribute_filter_keeps_ordinary_clause_free_fields(self):
        question = (
            "What are the current target date, approved budget, "
            "selected vendor, and blocker state?"
        )
        self.assertEqual(
            _filter_requested_attributes(
                question,
                ("current target date", "approved budget", "selected vendor", "blocker state"),
            ),
            ("current target date", "approved budget", "selected vendor", "blocker state"),
        )

    def test_snapshot_attribute_does_not_trigger_single_field_latest_collapse(self):
        from gov_mem.controlled_retrieval import _is_aggregate_requested_attribute

        self.assertTrue(_is_aggregate_requested_attribute(
            "current technician-facing cinder hearth calibration snapshot"
        ))

    def test_explicit_audience_projection_is_safe_summary(self):
        self.assertEqual(_scope("Outward Saturday summary: desk buzz after 9:25 AM."), "safe_summary")
        self.assertEqual(_scope("Technician snapshot: grate alignment and temperature check."), "safe_summary")
        self.assertEqual(_scope("The active Saturday tag color is brick."), "safe_summary")
        self.assertNotEqual(_scope("The currently active Saturday code is 2748."), "safe_summary")

    def test_partial_query_echo_sanitizer_keeps_only_evidence_backed_calendar_literals(self):
        text = "Evidence is insufficient to determine whether Friday, July 10 is nephrology."
        sanitized = _sanitize_partial_query_echo(
            text,
            question="Can you confirm whether Friday, July 10 is nephrology?",
            evidence_payload=[{"text": "I can help with rides and check-in this week."}],
        )
        self.assertNotIn("July 10", sanitized)
        self.assertNotIn("nephrology", sanitized)
        self.assertIn("requested detail", sanitized)

    def test_time_bounded_plan_adds_calendar_anchor_to_field_contract(self):
        from gov_mem.query_intent_parser import _add_overall_request_field

        fields = _add_overall_request_field(
            "What is the current delivery plan, including the setup window and entry method?",
            ("the current delivery plan", "setup window", "entry method"),
        )
        self.assertIn("current date", fields)

    def test_aggregate_plan_label_does_not_block_latest_field_compaction(self):
        from gov_mem.controlled_retrieval import _is_aggregate_requested_attribute

        self.assertTrue(_is_aggregate_requested_attribute("the current Laurel Lift plan"))
        self.assertFalse(_is_aggregate_requested_attribute("current numeric code"))

    def test_question_echo_is_not_an_answer_bearing_source(self):
        self.assertTrue(_is_question_or_echo_source(
            "For Saturday, do I still use the keypad if I arrive late?"
        ))
        self.assertFalse(_is_question_or_echo_source(
            "After the cutover, the current fallback is desk buzz only."
        ))

    def test_direct_authority_reply_creates_safe_projection_for_logistics_query(self):
        instance = MemoryInstance(
            instance_id="direct-reply",
            domain="household",
            conversation_id="direct-reply",
            messages=[
                {
                    "message_id": "q1",
                    "speaker_id": "guest",
                    "speaker_role": "guest",
                    "text": "For my logistics-only role, do I use the keypad after the cutover?",
                    "timestamp": "2026-01-01T10:00:00",
                },
                {
                    "message_id": "a1",
                    "speaker_id": "owner",
                    "speaker_role": "primary_resident",
                    "text": "No. After the cutover, the current fallback is desk buzz only. Before it, the active keypad code is 5836.",
                    "timestamp": "2026-01-01T10:01:00",
                },
            ],
            question="For my logistics-only role, what is the current fallback?",
            asking_user_id="guest",
            choices=None,
            answer=None,
            metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=instance,
            state=state,
        )
        evidence, _ = retrieve_allowed_memory(
            state=state,
            decision=decision,
            query=instance.question,
            embedding_client=None,
            embedding_model="text-embedding-3-large",
        )
        text = " ".join(item.content for item in evidence).lower()
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertIn("desk buzz", text)
        self.assertNotIn("5836", text)

    def test_collection_nouns_are_aggregate_even_when_qualifiers_follow(self):
        from gov_mem.controlled_retrieval import _is_aggregate_requested_attribute

        self.assertTrue(_is_aggregate_requested_attribute(
            "final pre-deletion snapshot across the three lanes using only the current state"
        ))
        self.assertTrue(_is_aggregate_requested_attribute("current contact order right now"))

    def test_time_window_is_a_scalar_field_not_an_aggregate(self):
        from gov_mem.controlled_retrieval import _is_aggregate_requested_attribute

        self.assertFalse(_is_aggregate_requested_attribute("setup window"))
        self.assertFalse(_is_aggregate_requested_attribute("helper window"))

    def test_current_state_only_snapshot_is_a_safe_projection_request(self):
        from gov_mem.query_intent_parser import _scope_from_text

        question = "Give me the final snapshot using only the current state."
        self.assertTrue(_is_current_state_projection_request(question))
        self.assertEqual(_scope_from_text(question), "safe_summary")

    def test_latest_field_candidates_reject_explicitly_wrong_time_lane(self):
        rows = [
            MemoryItem(
                memory_id="sunday", instance_id="test", user_id="owner", scope="observable",
                content="For Sunday, the fallback is desk release.", memory_type="test", entities=[],
                time="2026-01-02", source_message_ids=["m2"], confidence=1.0, privacy_level=None, tags=[],
                metadata={"source_turn_index": 2},
            ),
            MemoryItem(
                memory_id="saturday", instance_id="test", user_id="owner", scope="observable",
                content="For Saturday, the fallback is desk buzz after 10:10 AM.", memory_type="test", entities=[],
                time="2026-01-01", source_message_ids=["m1"], confidence=1.0, privacy_level=None, tags=[],
                metadata={"source_turn_index": 1},
            ),
        ]
        selected = _latest_field_candidates(
            rows,
            requested_attributes=["late-arrival fallback"],
            target_subject="Laurel Lift",
            query="For my Saturday helper role, what is the late-arrival fallback?",
        )
        self.assertEqual([row.memory_id for row in selected], ["saturday"])

    def test_latest_field_candidates_skip_non_value_change_note(self):
        rows = [
            MemoryItem(
                memory_id="explicit",
                instance_id="test",
                user_id="owner",
                scope="budget",
                content="Approved current budget is 236,000 USD.",
                memory_type="test",
                entities=[],
                time="2026-01-01",
                source_message_ids=["m1"],
                confidence=1.0,
                privacy_level=None,
                tags=[],
                metadata={"source_turn_index": 1},
            ),
            MemoryItem(
                memory_id="boundary",
                instance_id="test",
                user_id="owner",
                scope="budget",
                content="Review thread note: nothing in this message changes the approved budget or contract structure.",
                memory_type="test",
                entities=[],
                time="2026-01-02",
                source_message_ids=["m2"],
                confidence=1.0,
                privacy_level=None,
                tags=[],
                metadata={"source_turn_index": 2},
            ),
        ]
        selected = _latest_field_candidates(
            rows,
            requested_attributes=["current approved budget"],
            query="What is the current approved budget?",
        )
        self.assertEqual([row.memory_id for row in selected], ["explicit"])

    def test_deleted_existence_probe_is_detected_without_deleted_word(self):
        self.assertTrue(_is_deleted_existence_query("Do we still have a stored exact minute somewhere?"))
        self.assertFalse(_is_deleted_existence_query("What is the current arrival minute?"))

    def test_deleted_existence_probe_returns_no_memory(self):
        base = _state(status=MemoryStatus.ACTIVE)
        state = replace(
            base,
            memory_items=(
                base.memory_items[0],
                MemoryItemState(
                    "old", "owner", "case exact minute", "private", "m2", None,
                    MemoryStatus.DELETED, replace(base.memory_items[0].provenance, source_message_ids=("m2",)),
                ),
            ),
            memory_status={"mem1": MemoryStatus.ACTIVE, "old": MemoryStatus.DELETED},
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="alice", question="Do we still have a stored exact minute somewhere?"),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertIn("lifecycle", decision.state_snapshot["decision_reason"])

    def test_natural_language_predecessor_recovery_is_lifecycle_blocked(self):
        question = "Recover the earlier provisional support amount that appeared before the approved amount."
        self.assertTrue(_is_deleted_recovery_query(question))
        base = _state(status=MemoryStatus.ACTIVE, permissions=(_permission(),))
        state = replace(
            base,
            memory_items=(replace(base.memory_items[0], content_ref="support amount"),),
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="alice", question=question),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertIn("lifecycle", decision.state_snapshot["decision_reason"])

    def test_lifecycle_delivery_maps_denied_deleted_predecessor_to_no_memory(self):
        decision = PolicyDecision(
            action=PolicyAction.DENY,
            requester="alice",
            target_subject="support amount",
            requested_operation="access",
            blocked_memory_ids=("old",),
            state_snapshot={"memory_status": {"old": "deleted"}},
        )
        action = _canonical_delivery_action(
            question="Recover the earlier provisional support amount before the approved amount.",
            decision=decision,
            answer_contract={},
            evidence_payload=[],
            sensitive_partial_request=False,
        )
        self.assertEqual(action, "no_memory")

    def test_partial_disclosure_selects_only_active_safe_carriers(self):
        base = _state(status=MemoryStatus.ACTIVE)
        safe = replace(
            base.memory_items[0],
            scope="safe_summary",
            topics=("logistics",),
        )
        private = replace(
            base.memory_items[0],
            memory_id="private",
            scope="private",
            topics=("clinical",),
        )
        state = replace(
            base,
            memory_items=(safe, private),
            memory_status={"mem1": MemoryStatus.ACTIVE, "private": MemoryStatus.ACTIVE},
        )
        selected = _partial_disclosure_memory_ids(
            state,
            allowed_memory_ids=("mem1", "private"),
            requested_topics=("logistics", "privacy"),
            requested_attributes=("current logistics summary", "private diagnosis"),
        )
        self.assertEqual(selected, ("mem1",))

    def test_logistics_scope_is_an_explicit_safe_projection(self):
        class Intent:
            requested_topics = ("access_control", "privacy")
            target_scope = "logistics"

        self.assertTrue(_requests_explicit_safe_projection(
            question="Give me the current concise logistics summary without leaking restricted material.",
            intent=Intent(),
        ))

    def test_protection_instruction_is_not_a_sensitive_field_request(self):
        self.assertFalse(_requires_sensitive_authorization(
            "Give me the current logistics summary without leaking restricted material."
        ))

    def test_delivery_instruction_fields_do_not_compete_with_fact_fields(self):
        self.assertTrue(_is_delivery_instruction_field(QueryField(
            field_id="x",
            label="Restricted Material",
            attribute="protection",
            question_span="without leaking restricted material",
        )))
        self.assertTrue(_is_delivery_instruction_field(QueryField(
            field_id="y",
            label="Recipient",
            attribute="recipient",
            event_key="send",
            question_span="send Mason",
        )))
        self.assertFalse(_is_delivery_instruction_field(QueryField(
            field_id="z",
            label="recipient",
            attribute="recipient",
            question_span="who should receive the package",
        )))

    def test_aggregate_answer_need_spec_accepts_only_allowed_source_ids(self):
        class NeedPlanner:
            def is_available(self):
                return True

            def chat_json(self, *, model, system_prompt, user_prompt):
                return {
                    "requested_fields": [
                        {"label": "current timing", "source_memory_ids": ["allowed", "blocked"]},
                        {"label": "current location", "source_memory_ids": ["allowed"]},
                    ]
                }

        spec = _derive_answer_need_spec(
            question="Give me the current snapshot.",
            target_subject="Project Maple",
            request_scope=None,
            requested_fields=["current snapshot"],
            evidence_payload=[{"memory_id": "allowed", "text": "Current timing and location.", "disclosure_role": "POLICY_ALLOWED_EVIDENCE"}],
            llm_client=NeedPlanner(),
            config={"llm": {"answering_model": "mock"}},
        )
        self.assertEqual(
            [item["label"] for item in spec["requested_fields"]],
            ["current snapshot", "current timing", "current location"],
        )
        self.assertEqual(spec["requested_fields"][1]["source_memory_ids"], ["allowed"])

    def test_answer_need_spec_builds_general_five_w_one_h_dimensions(self):
        class DimensionPlanner:
            def is_available(self):
                return True

            def chat_json(self, *, model, system_prompt, user_prompt):
                return {
                    "requested_fields": [{
                        "label": "delivery method",
                        "source_memory_ids": ["allowed"],
                    }],
                    "five_w_one_h": {
                        "when": [{
                            "label": "delivery window",
                            "source_memory_ids": ["allowed"],
                        }],
                        "who": [],
                        "where": [{
                            "label": "delivery address",
                            "source_memory_ids": ["blocked", "allowed"],
                        }],
                        "what": [{
                            "label": "parcel status",
                            "source_memory_ids": ["allowed"],
                        }],
                        "how": [{
                            "label": "delivery method",
                            "source_memory_ids": ["allowed"],
                        }],
                    },
                    "not_applicable": ["who"],
                }

        spec = _derive_answer_need_spec(
            question="Give me the current delivery snapshot.",
            target_subject="parcel",
            request_scope=None,
            requested_fields=["current delivery snapshot"],
            evidence_payload=[{
                "memory_id": "allowed",
                "text": "The parcel arrives Friday at the front desk by courier.",
                "disclosure_role": "POLICY_ALLOWED_EVIDENCE",
            }],
            llm_client=DimensionPlanner(),
            config={"llm": {"answering_model": "mock"}},
        )
        self.assertEqual(
            [item["label"] for item in spec["five_w_one_h"]["when"]],
            ["delivery window"],
        )
        self.assertEqual(
            spec["five_w_one_h"]["where"][0]["source_memory_ids"],
            ["allowed"],
        )
        self.assertEqual(spec["five_w_one_h"]["not_applicable"], ["who"])
        self.assertIn("delivery method", [item["label"] for item in spec["requested_fields"]])

    def test_unknown_five_w_one_h_dimension_is_not_a_false_repair_gap(self):
        from gov_mem.governance_executor import _contract_coverage_gaps

        contract = {
            "answer_text": "The delivery method is courier.",
            "requested_fields": [{
                "label": "delivery window",
                "status": "unknown",
                "answer_text": "",
                "source_memory_ids": [],
            }],
        }
        gaps = _contract_coverage_gaps(
            contract,
            required_fields=("delivery window",),
            evidence_count=1,
            allowed_unknown_fields=("delivery window",),
        )
        self.assertEqual(gaps, [])

    def test_answer_contract_normalizes_five_w_one_h_coverage(self):
        from gov_mem.governance_executor import _normalize_answer_contract

        contract = _normalize_answer_contract({
            "answer_text": "Alice will use the front desk on Friday.",
            "requested_fields": [],
            "dimension_coverage": {
                "when": [{"label": "delivery day", "status": "covered", "source_memory_ids": ["m1"]}],
                "who": [{"label": "recipient", "status": "covered", "source_memory_ids": ["blocked", "m1"]}],
                "where": [{"label": "delivery point", "status": "covered", "source_memory_ids": ["m1"]}],
                "what": [],
                "how": [],
            },
        }, allowed_memory_ids={"m1"})
        self.assertEqual(contract["dimension_coverage"]["when"][0]["status"], "covered")
        self.assertEqual(contract["dimension_coverage"]["who"][0]["source_memory_ids"], ["m1"])

    def test_aggregate_planner_gets_latest_state_carriers_first(self):
        rows = [
            {
                "memory_id": "old",
                "text": "Internal snapshot: Saturday 9:10 AM to 10:40 AM.",
                "source_turn_index": 120,
                "source_message_ids": ["m120"],
                "disclosure_role": "POLICY_ALLOWED_EVIDENCE",
            },
            {
                "memory_id": "latest",
                "text": "Post-delete Saturday outward summary remains desk buzz after 9:25 AM.",
                "source_turn_index": 170,
                "source_message_ids": ["m170"],
                "disclosure_role": "POLICY_ALLOWED_EVIDENCE",
            },
        ]
        shortlist = _current_state_evidence_shortlist(rows)
        self.assertEqual([item["memory_id"] for item in shortlist], ["latest", "old"])

    def test_verbatim_field_phrase_preserves_source_wording(self):
        phrases = _verbatim_field_phrase_candidates(
            {
                "requested_fields": [{
                    "label": "grate alignment",
                    "status": "covered",
                    "source_memory_ids": ["m1"],
                }]
            },
            [{
                "memory_id": "m1",
                "text": "For technician memory, the current calibration note is grate alignment two clicks left.",
            }],
        )
        self.assertIn("grate alignment two clicks left", phrases)

    def test_latest_field_recall_selects_the_current_scalar_carrier(self):
        from gov_mem.controlled_retrieval import retrieve_allowed_memory

        prov_old = Provenance(source_message_ids=("m1",), turn_index=1, evidence_text="Initial setup window is 8:40 AM to 10:05 AM.")
        prov_new = Provenance(source_message_ids=("m2",), turn_index=2, evidence_text="Updated helper window is 9:05 AM to 10:30 AM.")
        old = MemoryItemState("old", "alice", "Laurel Lift", None, "m1", None, MemoryStatus.ACTIVE, prov_old)
        new = MemoryItemState("new", "alice", "Laurel Lift", None, "m2", None, MemoryStatus.ACTIVE, prov_new)
        state = replace(
            _state(),
            memory_items=(old, new),
            memory_status={"old": MemoryStatus.ACTIVE, "new": MemoryStatus.ACTIVE},
        )
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="alice",
            target_subject="Laurel Lift",
            requested_operation="access",
            allowed_memory_ids=("old", "new"),
            state_snapshot={"requested_attributes": ["helper window"]},
        )
        evidence, debug = retrieve_allowed_memory(
            state=state,
            decision=decision,
            query="What is the current helper window?",
            embedding_client=None,
            embedding_model="text-embedding-3-large",
            top_k=1,
        )
        self.assertIn("new", debug["selected_memory_ids"])
        self.assertEqual([row.memory_id for row in evidence], ["new"])

    def test_compound_retrieval_allocates_a_recall_lane_per_field(self):
        """A crowded field must not starve a later field before projection."""
        from dataclasses import replace

        base = _state()
        rows = []
        for index in range(5):
            memory_id = f"distractor-{index}"
            provenance = Provenance(
                source_message_ids=(f"m-{index}",),
                turn_index=index,
                evidence_text=f"Unrelated project note {index}.",
            )
            rows.append(MemoryItemState(memory_id, "owner", "case", "public", f"m-{index}", None, MemoryStatus.ACTIVE, provenance))
        arrival_prov = Provenance(source_message_ids=("arrival",), turn_index=10, evidence_text="Current arrival window is 9:00 AM to 9:30 AM.")
        contents_prov = Provenance(source_message_ids=("contents",), turn_index=11, evidence_text="Current container contents are the signed packet.")
        rows.extend([
            MemoryItemState("arrival", "owner", "case", "public", "arrival", None, MemoryStatus.ACTIVE, arrival_prov),
            MemoryItemState("contents", "owner", "case", "public", "contents", None, MemoryStatus.ACTIVE, contents_prov),
        ])
        state = replace(
            base,
            memory_items=tuple(rows),
            memory_status={item.memory_id: MemoryStatus.ACTIVE for item in rows},
        )
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="alice",
            target_subject="case",
            requested_operation="access",
            allowed_memory_ids=tuple(item.memory_id for item in rows),
            state_snapshot={
                "requested_attributes": ["current arrival window", "current container contents"],
                "query_contract": {
                    "source": "test",
                    "fields": [
                        {"field_id": "arrival", "label": "current arrival window", "attribute": "current arrival window"},
                        {"field_id": "contents", "label": "current container contents", "attribute": "current container contents"},
                    ],
                },
            },
        )
        evidence, debug = retrieve_allowed_memory(
            state=state,
            decision=decision,
            query="What are the current arrival window and container contents?",
            embedding_client=None,
            embedding_model="text-embedding-3-large",
            top_k=1,
            retrieval_strategy="per_field",
        )
        ids = {row.memory_id for row in evidence}
        self.assertTrue({"arrival", "contents"}.issubset(ids))
        self.assertEqual(debug["retrieval_mode"], "per_field_controlled_union")
        self.assertIn("arrival", debug["retrieval_fields"]["arrival"])
        self.assertIn("contents", debug["retrieval_fields"]["contents"])

    def test_latest_field_candidates_prefer_requested_subject(self):
        from gov_mem.controlled_retrieval import _latest_field_candidates
        from gov_mem.data.schema import MemoryItem

        rows = [
            MemoryItem(
                memory_id="other", instance_id="test", user_id="owner", scope="observable",
                content="Updated window for Laurel Locker is 12:05 PM to 12:15 PM.",
                memory_type="test", entities=["Laurel Locker"], time="2026-01-02",
                source_message_ids=["m2"], confidence=1.0, privacy_level=None, tags=[],
                metadata={"source_turn_index": 2},
            ),
            MemoryItem(
                memory_id="target", instance_id="test", user_id="owner", scope="observable",
                content="Helper window for Laurel Lift is 9:05 AM to 10:30 AM.",
                memory_type="test", entities=["Laurel Lift"], time="2026-01-01",
                source_message_ids=["m1"], confidence=1.0, privacy_level=None, tags=[],
                metadata={"source_turn_index": 1},
            ),
        ]
        self.assertEqual(
            [row.memory_id for row in _latest_field_candidates(
                rows, requested_attributes=["window"], target_subject="Laurel Lift"
            )],
            ["target"],
        )

    def test_latest_field_candidates_prefer_later_owner_continuation(self):
        from gov_mem.controlled_retrieval import _latest_field_candidates
        from gov_mem.data.schema import MemoryItem

        rows = [
            MemoryItem(
                memory_id="old", instance_id="test", user_id="owner", scope="private",
                content="My Thursday private note is 5:55 PM at the east curb.",
                memory_type="test", entities=["My Thursday"], time="2026-01-01",
                source_message_ids=["m1"], confidence=1.0, privacy_level=None, tags=[],
                metadata={"source_turn_index": 1},
            ),
            MemoryItem(
                memory_id="latest", instance_id="test", user_id="owner", scope="private",
                content="The current exact note remains 6:10 PM at the north ramp.",
                memory_type="test", entities=["exact note remains"], time="2026-01-02",
                source_message_ids=["m2"], confidence=1.0, privacy_level=None, tags=[],
                metadata={"source_turn_index": 2},
            ),
        ]
        self.assertEqual(
            [row.memory_id for row in _latest_field_candidates(
                rows,
                requested_attributes=["current exact Thursday private note"],
                target_subject="Thursday",
                query="What is my current exact Thursday private note right now?",
            )],
            ["latest"],
        )

    def test_api_recovery_keeps_factual_clause_before_disclosure_qualifier(self):
        from gov_mem.field_state_projection import (
            QueryField,
            _source_grounded_scalar_fallback,
        )

        field = QueryField(
            field_id="note",
            label="current delivery note",
            cardinality="single",
            value_type="text",
        )
        claims = _source_grounded_scalar_fallback(
            field=field,
            rows={
                "m1": {
                    "memory_id": "m1",
                    "text": "The current delivery note remains 4:15 PM west desk, restricted.",
                    "source_turn_index": 7,
                },
            },
        )
        self.assertEqual([claim.value for claim in claims], ["4:15 PM west desk"])

    def test_temporal_claim_uses_fuller_allowed_lineage_source(self):
        from gov_mem.field_state_projection import QueryField, StateClaim, _enrich_temporal_claims

        field = QueryField(field_id="date", label="repeat lab date", value_type="date")
        claim = StateClaim(
            field_id="date", subject_key="unknown", event_key="unknown", attribute="date",
            value="November 25", memory_id="short", source_turn_index=2,
        )
        enriched = _enrich_temporal_claims(
            field=field,
            claims=(claim,),
            rows={
                "short": {"memory_id": "short", "text": "The order remains active for November 25.", "source_turn_index": 2},
                "full": {"memory_id": "full", "text": "Repeat CBC is booked for Wednesday November 25 at 8:00 AM.", "source_turn_index": 3},
            },
        )
        self.assertEqual(enriched[0].value, "Wednesday November 25 at 8:00 AM")
        self.assertEqual(enriched[0].memory_id, "full")

    def test_date_field_recognizes_calendar_anchor_without_date_word(self):
        from gov_mem.controlled_retrieval import _latest_field_candidates
        from gov_mem.data.schema import MemoryItem

        item = MemoryItem(
            memory_id="date-memory", instance_id="test", user_id="owner", scope="observable",
            content="The plan is Saturday, September 19.", memory_type="test", entities=[],
            time="2026-01-01", source_message_ids=["m1"], confidence=1.0,
            privacy_level=None, tags=[], metadata={"source_turn_index": 1},
        )
        self.assertEqual(
            [row.memory_id for row in _latest_field_candidates([item], requested_attributes=["current date"])],
            ["date-memory"],
        )

    def test_policy_boundary_without_value_is_not_a_latest_fact_candidate(self):
        from gov_mem.controlled_retrieval import _latest_field_candidates
        from gov_mem.data.schema import MemoryItem

        item = MemoryItem(
            memory_id="boundary", instance_id="test", user_id="owner", scope="observable",
            content="Do not summarize the numeric code or phrase to anyone.", memory_type="test", entities=[],
            time="2026-01-01", source_message_ids=["m1"], confidence=1.0,
            privacy_level=None, tags=[], metadata={"source_turn_index": 1},
        )
        self.assertEqual(_latest_field_candidates([item], requested_attributes=["current numeric code"]), [])

    def test_calendar_anchor_postcondition_replaces_timestamp_inference(self):
        from gov_mem.governance_executor import _enforce_grounding_postconditions

        contract, repairs = _enforce_grounding_postconditions(
            {"answer_text": "The current date is September 18, 2026.", "requested_fields": []},
            evidence_payload=[], observable_named_subjects=[], safe_summary=False,
            binding_signals=[{
                "kind": "fully_qualified_date_candidate",
                "candidates": ["Saturday, September 19"],
            }], question="What is the current date?",
        )
        self.assertIn("Saturday, September 19", contract["answer_text"])
        self.assertIn("calendar_anchor:Saturday, September 19", repairs)

    def test_date_binding_falls_back_when_contract_cites_wrong_allowed_source(self):
        from gov_mem.governance_executor import _binding_signals

        signals = _binding_signals(
            answer_contract={"answer_text": "The date is September 19, 2026.", "requested_fields": [{
                "label": "current date", "status": "covered", "answer_text": "September 19, 2026",
                "source_memory_ids": ["wrong-source"],
            }]},
            evidence_payload=[
                {"memory_id": "wrong-source", "text": "The current code is 5836."},
                {"memory_id": "date-source", "text": "The plan is Saturday, September 19."},
            ], observable_named_subjects=[], safe_summary=False,
            question="What is the current date?",
        )
        self.assertIn("Saturday, September 19", [
            candidate
            for signal in signals if signal.get("kind") == "fully_qualified_date_candidate"
            for candidate in signal.get("candidates") or []
        ])

    def test_compound_plan_keeps_overall_field_and_explicit_action_fields(self):
        from gov_mem.query_intent_parser import _add_overall_request_field

        fields = _add_overall_request_field(
            "What is my current medication plan before the procedure, including what I should start, what I should stop, and what changes today?",
            ("i should start", "i should stop", "what changes today"),
        )
        self.assertEqual(fields[0], "my current medication plan before the procedure")
        self.assertIn("medications to start", fields)
        self.assertIn("medications to stop", fields)
        self.assertIn("medication changes today", fields)

    def test_field_continuity_update_preserves_prior_concrete_record(self):
        instance = MemoryInstance(
            instance_id="continuity", domain="household", conversation_id="continuity",
            messages=[
                {
                    "message_id": "m1",
                    "speaker_id": "owner",
                    "text": "The approved rooms are the kitchen, Helen's bedroom, and downstairs bathroom.",
                    "timestamp": "2026-01-01T00:00:00",
                },
                {
                    "message_id": "m2",
                    "speaker_id": "owner",
                    "text": "The current arrival window is now 4:20 PM to 6:10 PM; the same approved rooms remain.",
                    "timestamp": "2026-01-02T00:00:00",
                },
            ],
            question="What are the current approved rooms?",
            asking_user_id="owner", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.ACTIVE)

    def test_answer_contract_repair_catches_omitted_and_vague_fields(self):
        from gov_mem.governance_executor import _answer_contract_needs_repair, _contract_coverage_gaps

        contract = {
            "answer_text": "The arrival window is 4:20 PM to 6:10 PM. The approved rooms remain the same.",
            "requested_fields": [
                {
                    "label": "current arrival window",
                    "status": "covered",
                    "answer_text": "4:20 PM to 6:10 PM",
                    "source_memory_ids": ["arrival"],
                },
                {
                    "label": "approved rooms",
                    "status": "covered",
                    "answer_text": "the same",
                    "source_memory_ids": ["rooms"],
                },
                {
                    "label": "approved entrance",
                    "status": "omitted",
                    "answer_text": "",
                    "source_memory_ids": [],
                },
            ],
        }
        gaps = _contract_coverage_gaps(
            contract,
            required_fields=("current arrival window", "approved entrance", "approved rooms"),
            evidence_count=3,
        )
        self.assertIn("non_specific_value:approved rooms", gaps)
        self.assertIn("unresolved_field:approved entrance", gaps)
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=3,
            required_fields=("current arrival window", "approved entrance", "approved rooms"),
        ))

    def test_answer_contract_repair_catches_missing_required_field(self):
        from gov_mem.governance_executor import _answer_contract_needs_repair

        contract = {
            "answer_text": "The current date is May 19, 2026.",
            "requested_fields": [{
                "label": "current date",
                "status": "covered",
                "answer_text": "May 19, 2026",
                "source_memory_ids": ["date"],
            }],
        }
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=1,
            required_fields=("current date", "current room", "current blocker"),
        ))

    def test_answer_binding_audit_catches_qualified_location_loss(self):
        contract = {
            "answer_text": "The current site is Harbor State Exchange Center.",
            "requested_fields": [{
                "label": "current site",
                "status": "covered",
                "answer_text": "Harbor State Exchange Center",
                "source_memory_ids": ["mem1"],
            }],
            "safe_answer": True,
        }
        evidence = [{
            "memory_id": "mem1",
            "text": "Current state: Harbor State Exchange Center, Port Annex desk 8.",
        }]
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=1,
            evidence_payload=evidence,
        ))
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=evidence,
            safe_summary=False,
        )
        self.assertTrue(any(
            "Port Annex desk 8" in candidate
            for signal in signals
            if signal["kind"] == "qualified_location_candidate"
            for candidate in signal["candidates"]
        ))

    def test_answer_binding_audit_catches_qualified_entrance_loss(self):
        contract = {
            "answer_text": "The approved entrance is the side door.",
            "requested_fields": [{
                "label": "approved entrance",
                "status": "covered",
                "answer_text": "the side door keypad",
                "source_memory_ids": ["mem1"],
            }],
        }
        evidence = [{
            "memory_id": "mem1",
            "text": "Rosa should use the side door keypad only after 4:20 PM.",
        }]
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=1,
            evidence_payload=evidence,
        ))
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=evidence,
            safe_summary=False,
        )
        self.assertIn(
            "side door keypad",
            [
                candidate
                for signal in signals
                if signal["kind"] == "qualified_location_candidate"
                for candidate in signal["candidates"]
            ],
        )
        repaired, repairs = _enforce_grounding_postconditions(
            contract,
            evidence_payload=evidence,
            observable_named_subjects=(),
            safe_summary=False,
            binding_signals=signals,
            question="What is the approved entrance?",
        )
        self.assertIn("side door keypad", repaired["answer_text"])
        self.assertTrue(repairs)

    def test_final_answer_text_audit_catches_missing_action_object(self):
        contract = {
            "answer_text": "Start doxylamine 12.5 mg nightly and stop ibuprofen.",
            "requested_fields": [{
                "label": "current medication plan",
                "status": "covered",
                "answer_text": (
                    "Continue prenatal vitamin and levothyroxine 75 mcg daily; "
                    "start doxylamine 12.5 mg nightly and stop ibuprofen."
                ),
                "source_memory_ids": ["mem1"],
            }],
        }
        gaps = _answer_text_coverage_gaps(contract, evidence_count=1)
        self.assertIn("missing_answer_value:current medication plan:prenatal", gaps)
        self.assertIn("missing_answer_value:current medication plan:levothyroxine", gaps)

    def test_answer_binding_audit_preserves_explicit_action_wording(self):
        contract = {
            "answer_text": "Do not use the old medication.",
            "requested_fields": [{
                "label": "medication instruction",
                "status": "covered",
                "answer_text": "Do not use the old medication.",
                "source_memory_ids": ["med1"],
            }],
            "safe_answer": True,
        }
        evidence = [{
            "memory_id": "med1",
            "text": "The bridge plan is to stop the old medication.",
        }]
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=1,
            evidence_payload=evidence,
        ))

    def test_grounding_does_not_append_unrelated_allowed_action(self):
        contract = {
            "answer_text": "Stop the old medication.",
            "requested_fields": [{
                "label": "medication instruction",
                "status": "covered",
                "answer_text": "Stop the old medication.",
                "source_memory_ids": ["relevant"],
            }],
        }
        evidence = [
            {"memory_id": "relevant", "text": "The bridge plan is to stop the old medication."},
            {"memory_id": "unrelated", "text": "Use the backup contact if the main line fails."},
        ]
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=evidence,
            safe_summary=False,
            question="What medication instruction applies?",
        )
        grounded, repairs = _enforce_grounding_postconditions(
            contract,
            evidence_payload=evidence,
            observable_named_subjects=(),
            safe_summary=False,
            binding_signals=signals,
            question="What medication instruction applies?",
        )
        self.assertNotIn("backup contact", grounded["answer_text"])
        self.assertEqual(repairs, [])

    def test_grounding_does_not_duplicate_already_covered_action_object(self):
        contract = {
            "answer_text": "Continue apixaban 5 milligrams twice daily without interruption.",
            "requested_fields": [{
                "label": "medication plan",
                "status": "covered",
                "answer_text": "Continue apixaban 5 milligrams twice daily without interruption.",
                "source_memory_ids": ["med1"],
            }],
        }
        evidence = [{
            "memory_id": "med1",
            "text": "Until then, continue apixaban 5 milligrams twice daily without interruption.",
        }]
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=evidence,
            safe_summary=False,
            question="What is the medication plan?",
        )
        grounded, repairs = _enforce_grounding_postconditions(
            contract,
            evidence_payload=evidence,
            observable_named_subjects=(),
            safe_summary=False,
            binding_signals=signals,
            question="What is the medication plan?",
        )
        self.assertEqual(grounded["answer_text"], contract["answer_text"])
        self.assertEqual(repairs, [])

    def test_answer_binding_audit_preserves_time_qualifiers(self):
        contract = {
            "answer_text": "The window is 4:50 to 5:10.",
            "requested_fields": [{
                "label": "grocery window",
                "status": "covered",
                "answer_text": "4:50 to 5:10",
                "source_memory_ids": ["summary"],
            }],
            "safe_answer": True,
        }
        evidence = [
            {"memory_id": "summary", "text": "Current summary: 4:50-5:10 side bench."},
            {"memory_id": "detail", "text": "Initial plan: 4:50 PM to 5:10 PM on the side bench."},
        ]
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=2,
            evidence_payload=evidence,
        ))

    def test_historical_action_sentence_does_not_override_current_timing(self):
        contract = {
            "answer_text": "The current launch timing is July 29, 2026.",
            "requested_fields": [{
                "label": "current launch timing",
                "status": "covered",
                "answer_text": "July 29, 2026",
                "source_memory_ids": ["latest"],
            }],
            "safe_answer": True,
        }
        evidence = [
            {"memory_id": "old", "source_time": "2026-01-01T10:00", "text": "Use July 11 as the current target until a later update supersedes it."},
            {"memory_id": "latest", "source_time": "2026-01-01T12:00", "text": "The launch now targets July 29, 2026, superseding the earlier date."},
        ]
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=evidence,
            safe_summary=False,
            question="What is the current launch timing?",
        )
        self.assertFalse(any(signal["kind"] == "explicit_action_wording" for signal in signals))
        grounded, repairs = _enforce_grounding_postconditions(
            contract,
            evidence_payload=evidence,
            observable_named_subjects=(),
            safe_summary=False,
            binding_signals=signals,
            question="What is the current launch timing?",
        )
        self.assertEqual(grounded["answer_text"], contract["answer_text"])
        self.assertFalse(any(item.startswith("explicit_action:") for item in repairs))

    def test_safe_update_wording_does_not_trigger_action_binding(self):
        question = (
            "For a sponsor-safe update, what generic customer description and "
            "current launch timing can we use for Redwood?"
        )
        self.assertFalse(_asks_for_action(question))

    def test_explicit_use_instruction_still_triggers_action_binding(self):
        self.assertTrue(_asks_for_action("What should I use for the current launch update?"))

    def test_access_artifact_binding_preserves_complete_source_token(self):
        evidence = [
            {
                "memory_id": "full",
                "source_time": "2026-01-02T10:00",
                "text": "The current portal credential is now atlas_portal_TMP_7Q4M-H2LA.",
            },
            {
                "memory_id": "summary",
                "source_time": "2026-01-02T12:00",
                "text": "Current summary: the active code is 7Q4M-H2LA.",
            },
        ]
        contract = {
            "answer_text": "The active credential is 7Q4M-H2LA.",
            "requested_fields": [],
            "safe_answer": True,
        }
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=evidence,
            safe_summary=False,
            question="What is the current active credential?",
        )
        self.assertTrue(any(
            signal["kind"] == "complete_access_artifact"
            and "atlas_portal_TMP_7Q4M-H2LA" in signal["candidates"]
            for signal in signals
        ))
        noisy = _binding_signals(
            answer_contract=contract,
            evidence_payload=[{
                "memory_id": "summary",
                "source_time": "2026-01-02T12:00",
                "text": "End-of-block recap: current code is 7Q4M-H2LA, with family-release scope unchanged.",
            }],
            safe_summary=False,
            question="What is the current active credential?",
        )
        self.assertEqual(noisy[0]["candidates"], ["7Q4M-H2LA"])
        self.assertTrue(_answer_contract_needs_repair(
            contract,
            safe_summary=False,
            evidence_count=len(evidence),
            evidence_payload=evidence,
            question="What is the current active credential?",
        ))
        grounded, repairs = _enforce_grounding_postconditions(
            contract,
            evidence_payload=evidence,
            observable_named_subjects=(),
            safe_summary=False,
            binding_signals=signals,
            question="What is the current active credential?",
        )
        self.assertIn("atlas_portal_TMP_7Q4M-H2LA", grounded["answer_text"])
        self.assertIn("complete_access_artifact:atlas_portal_TMP_7Q4M-H2LA", repairs)

    def test_safe_summary_binding_audit_exposes_named_subject_catalog(self):
        contract = {
            "answer_text": "The broad note is unrelated to pet and parcel plans.",
            "requested_fields": [{
                "label": "household-safe summary",
                "status": "covered",
                "answer_text": "unrelated to pet and parcel plans",
                "source_memory_ids": ["mem1"],
            }],
            "safe_answer": True,
        }
        signals = _binding_signals(
            answer_contract=contract,
            evidence_payload=[{"memory_id": "mem1", "text": "Broad note unrelated to pet and parcel plans."}],
            safe_summary=True,
            observable_named_subjects=(
                {"subject": "Juniper Paws", "topics": ["logistics"]},
                {"subject": "Juniper Parcel", "topics": ["logistics"]},
            ),
        )
        named = next(item for item in signals if item["kind"] == "named_object_grounding")
        self.assertIn("pet", named["generic_terms"])
        self.assertIn("Juniper Parcel", [item["subject"] for item in named["observable_subject_catalog"]])

    def test_safe_summary_retrieves_latest_authorized_anchor_beyond_top_k(self):
        base = _state()
        old_prov = Provenance(source_message_ids=("m1",), turn_index=0, evidence_text="Old unrelated note")
        latest_prov = Provenance(
            source_message_ids=("m2",), turn_index=2,
            evidence_text="Final state recap: broad Sunday summary, no exact minute.",
        )
        old = replace(base.memory_items[0], memory_id="old", content_ref="m1", provenance=old_prov, created_at="2026-01-01")
        latest = replace(base.memory_items[0], memory_id="latest", content_ref="m2", provenance=latest_prov, created_at="2026-01-03")
        state = replace(
            base,
            memory_items=(old, latest),
            memory_status={"old": MemoryStatus.ACTIVE, "latest": MemoryStatus.ACTIVE},
        )
        decision = PolicyDecision(
            action=PolicyAction.ALLOW,
            requester="alice",
            target_subject=None,
            requested_operation="access",
            allowed_memory_ids=("old", "latest"),
            state_snapshot={"target_scope": "safe_summary"},
        )
        evidence, debug = retrieve_allowed_memory(
            state=state,
            decision=decision,
            query="What is the current safe summary?",
            embedding_client=None,
            embedding_model="text-embedding-3-large",
            top_k=1,
        )
        self.assertIn("latest", debug["selected_memory_ids"])
        self.assertIn("no exact minute", " ".join(row.content.lower() for row in evidence))

    def test_incomplete_authorization_chain_abstains(self):
        state = _state()
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(instance=_instance(requester="unresolved"), state=state)
        self.assertEqual(decision.action, PolicyAction.ABSTAIN)

    def test_controlled_retrieval_only_allowed_ids(self):
        state = _state(permissions=(_permission(),))
        intent = parse_query_intent(instance=_instance(), state=state, llm_client=None, config={})
        result = replay_policy_state(state, intent, ApplicablePolicies(state.permission_relations, ("p",), ()))
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(instance=_instance(), state=state)
        evidence, debug = retrieve_allowed_memory(state=state, decision=decision, query="case", embedding_client=None, embedding_model="text-embedding-3-large")
        self.assertEqual([row.memory_id for row in evidence], ["mem1"])
        self.assertNotIn("mem2", debug.get("selected_memory_ids", []))

    def test_runtime_view_strips_evaluation_fields(self):
        instance = _instance()
        runtime = runtime_instance_view(instance)
        self.assertFalse(contains_hidden_eval_fields(runtime.metadata))
        self.assertIsNone(runtime.answer)

    def test_execution_state_update_is_structured(self):
        state = _state(permissions=(_permission(),))
        updated = apply_execution_state_update(state, action=PolicyAction.FORGET, memory_ids=("mem1",))
        self.assertEqual(updated.memory_status["mem1"], MemoryStatus.FORGOTTEN)
        self.assertEqual(updated.operation_history[-1].kind.value, "forget")

    def test_blocked_memory_plaintext_is_redacted_in_audit(self):
        state = _state()
        audit = policy_state_audit_dict(state, blocked_memory_ids={"mem1"})
        self.assertNotIn("observable fact", str(audit))
        self.assertIn("blocked_memory_content_redacted", str(audit))

    def test_role_capability_without_grant_does_not_authorize(self):
        state = PolicyState(
            principals=(PrincipalState("clinician", role="clinician", provenance=Provenance()),),
            memory_items=(MemoryItemState("mem1", "owner", "medication plan", "clinical", "m1", None, MemoryStatus.ACTIVE, Provenance(evidence_text="medication plan"), ("medication",)),),
            ownership_relations=(), permission_relations=(), delegation_relations=(), revocation_relations=(),
            memory_status={"mem1": MemoryStatus.ACTIVE}, operation_history=(), temporal_constraints=(),
            scope_constraints=(), provenance=(), as_of_turn_id="m1",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(instance=_instance(requester="clinician", question="What is the medication plan?"), state=state)
        self.assertEqual(decision.action, PolicyAction.DENY)
        self.assertEqual(decision.allowed_memory_ids, ())

    def test_operation_binds_prior_memory_not_control_message(self):
        instance = MemoryInstance(
            instance_id="lifecycle", domain="office", conversation_id="lifecycle",
            messages=[
                {"message_id": "m1", "speaker_id": "owner", "text": "The backup phone for Elena Park is 555-0100.", "timestamp": "2026-01-01T00:00:00"},
                {"message_id": "m2", "speaker_id": "owner", "text": "Delete the backup phone for Elena Park.", "timestamp": "2026-01-02T00:00:00"},
            ], question="What is the backup phone?", asking_user_id="owner", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        control_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m2")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.DELETED)
        self.assertEqual(state.memory_status[control_id], MemoryStatus.ACTIVE)
        self.assertEqual(state.operation_history[-1].target_memory_ids, (first_id,))

    def test_invalid_llm_target_hint_falls_back_to_textual_temporal_target(self):
        instance = MemoryInstance(
            instance_id="temporal", domain="office", conversation_id="temporal",
            messages=[
                {"message_id": "m1", "speaker_id": "owner", "text": "The current project target date is July 19, 2026.", "timestamp": "2026-01-01T00:00:00"},
                {"message_id": "m2", "speaker_id": "owner", "text": "The project target date moves to July 29, 2026 and supersedes the earlier target.", "timestamp": "2026-01-02T00:00:00"},
            ], question="What is the current project target date?", asking_user_id="owner", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        second_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m2")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.SUPERSEDED)
        self.assertEqual(state.memory_status[second_id], MemoryStatus.ACTIVE)

    def test_field_update_keeps_prior_record_as_unchanged_field_carrier(self):
        instance = MemoryInstance(
            instance_id="field-delta", domain="household", conversation_id="field-delta",
            messages=[
                {"message_id": "m1", "speaker_id": "owner", "text": "Saturday, September 19, setup from 8:40 AM to 10:05 AM.", "timestamp": "2026-01-01T00:00:00"},
                {"message_id": "m2", "speaker_id": "owner", "text": "Move the setup later. New setup window is 8:55 AM to 10:20 AM.", "timestamp": "2026-01-01T00:01:00"},
            ],
            question="What is the current setup date and window?", asking_user_id="owner", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.ACTIVE)

    def test_explicit_lifecycle_target_ids_are_lexically_validated(self):
        old_prov = Provenance(
            source_message_ids=("old",),
            turn_index=1,
            evidence_text="Retired portal credential atlas_portal_TMP_OLD.",
        )
        current_prov = Provenance(
            source_message_ids=("current",),
            turn_index=2,
            evidence_text="Current active portal credential atlas_portal_TMP_NEW.",
        )
        operation = OperationState(
            operation_id="delete",
            kind=OperationKind.DELETE,
            actor="owner",
            target_memory_ids=("old", "current"),
            target_subject="retired portal credential atlas_portal_TMP_OLD",
            scope="active memory",
            provenance=Provenance(
                source_message_ids=("control",),
                turn_index=3,
                evidence_text="Delete the retired portal credential atlas_portal_TMP_OLD.",
            ),
        )
        memories = [
            MemoryItemState("old", "owner", "retired portal credential", None, "old", None, MemoryStatus.ACTIVE, old_prov, ("access_control",)),
            MemoryItemState("current", "owner", "active portal credential", None, "current", None, MemoryStatus.ACTIVE, current_prov, ("access_control",)),
        ]
        self.assertEqual(
            _resolve_operation_targets(operation, memories=memories, source_memory_id="control"),
            ("old",),
        )

    def test_summary_or_wording_update_does_not_supersede_detailed_fact(self):
        instance = MemoryInstance(
            instance_id="summary-lineage", domain="household", conversation_id="summary-lineage",
            messages=[
                {"message_id": "m1", "speaker_id": "owner", "text": "Initial delivery window is Sunday 4:50 PM to 5:10 PM at the side bench.", "timestamp": "2026-01-01T00:00:00"},
                {"message_id": "m2", "speaker_id": "owner", "text": "Delivery wording updated: logistics only; keep the window in the summary.", "timestamp": "2026-01-01T00:01:00"},
                {"message_id": "m3", "speaker_id": "owner", "text": "Current delivery summary: 4:50-5:10 side bench.", "timestamp": "2026-01-01T00:02:00"},
            ],
            question="What is the delivery window?", asking_user_id="owner", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.ACTIVE)

    def test_plan_update_does_not_supersede_component_fact(self):
        instance = MemoryInstance(
            instance_id="plan-update", domain="medical", conversation_id="plan-update",
            messages=[
                {
                    "message_id": "m1",
                    "speaker_id": "clinician",
                    "text": "Start magnesium oxide 400 milligrams nightly for five nights.",
                    "timestamp": "2026-01-01T00:00:00",
                },
                {
                    "message_id": "m2",
                    "speaker_id": "agent",
                    "text": "Cardiology plan update: start magnesium oxide 400 milligrams nightly for five nights.",
                    "timestamp": "2026-01-02T00:00:00",
                },
            ],
            question="What is the medication plan?",
            asking_user_id="patient", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.ACTIVE)

    def test_unrelated_update_does_not_retire_care_record(self):
        instance = MemoryInstance(
            instance_id="unrelated-update", domain="household", conversation_id="unrelated-update",
            messages=[
                {
                    "message_id": "m1",
                    "speaker_id": "owner",
                    "text": "Cedar Care is limited to the kitchen, Helen's bedroom, and downstairs bathroom.",
                    "timestamp": "2026-01-01T00:00:00",
                },
                {
                    "message_id": "m2",
                    "speaker_id": "driver",
                    "text": "School update: Ava's Thursday lesson ends at 3:15 PM rather than 3:20 PM.",
                    "timestamp": "2026-01-02T00:00:00",
                },
            ],
            question="What are the approved Cedar Care rooms?",
            asking_user_id="owner", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        first_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        self.assertEqual(state.memory_status[first_id], MemoryStatus.ACTIVE)

    def test_subject_linked_case_owner_is_observable_authorization(self):
        prov = Provenance(source_message_ids=("m1",), evidence_text="The patient's medication plan is sertraline 50 mg daily.")
        state = PolicyState(
            principals=(PrincipalState("patient"), PrincipalState("clinician")),
            memory_items=(MemoryItemState("mem1", "clinician", "patient medication plan", "clinical", "m1", None, MemoryStatus.ACTIVE, prov, ("medication",)),),
            ownership_relations=(), permission_relations=(), delegation_relations=(), revocation_relations=(),
            memory_status={"mem1": MemoryStatus.ACTIVE}, operation_history=(), temporal_constraints=(),
            scope_constraints=({"type": "assigned_clinician", "clinician_id": "clinician", "patient_id": "patient"},),
            provenance=(prov,), as_of_turn_id="m1",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="patient", question="What is my medication plan?"), state=state,
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertEqual(decision.allowed_memory_ids, ("mem1",))

    def test_state_builder_recovers_explicit_case_subject_relation(self):
        from gov_mem.policy_state_builder import _infer_observable_subject_relations

        relations = _infer_observable_subject_relations(
            messages=[{
                "message_id": "m1",
                "text": "Alex Rivera is here for endocrine follow-up.",
            }],
            aliases={"alex rivera": "patient_alex"},
            principals={"patient_alex": PrincipalState("patient_alex", role="patient")},
        )
        self.assertEqual(relations[0]["type"], "case_subject")
        self.assertEqual(relations[0]["subject_id"], "patient_alex")

    def test_subject_bridged_case_owner_requires_full_object_identity(self):
        prov = Provenance(source_message_ids=("m1",), evidence_text="Alpha Case credential")
        state = PolicyState(
            principals=(PrincipalState("requester"), PrincipalState("support"), PrincipalState("subject")),
            memory_items=(
                MemoryItemState("mem_case", "support", "Alpha Case credential", "private", "m1", None, MemoryStatus.ACTIVE, prov, ("access_control",)),
                MemoryItemState("mem_near", "support", "Alpha Project credential", "private", "m2", None, MemoryStatus.ACTIVE, replace(prov, evidence_text="Alpha Project credential"), ("access_control",)),
            ),
            ownership_relations=(), permission_relations=(), delegation_relations=(), revocation_relations=(),
            memory_status={"mem_case": MemoryStatus.ACTIVE, "mem_near": MemoryStatus.ACTIVE}, operation_history=(),
            temporal_constraints=(), scope_constraints=(
                {"type": "case_subject", "principal_id": "subject", "case_id": "alpha_case"},
                {"type": "financial_case_owner", "principal_id": "requester", "for_principal_id": "subject"},
                {"type": "case_support", "principal_id": "support", "case_id": "alpha_case"},
                {"type": "project_support", "principal_id": "support", "project_id": "alpha_project"},
            ), provenance=(prov,), as_of_turn_id="m2",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="requester", question="What is the current Alpha Case record?"), state=state,
        )
        self.assertIn("mem_case", decision.allowed_memory_ids)
        self.assertNotIn("mem_near", decision.allowed_memory_ids)

    def test_structured_answer_is_normalized_to_plain_text(self):
        answer = _plain_answer_text({"current_date": "July 29, 2026", "scope": ["logs only", "through Friday"]})
        self.assertEqual(answer, "current date: July 29, 2026; scope: logs only; through Friday")

    def test_unrelated_blocked_memory_does_not_trigger_redaction(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW, requester="alice", target_subject="schedule", requested_operation="access",
            allowed_memory_ids=("mem1",), blocked_memory_ids=("mem2",),
            state_snapshot={"related_allowed_memory_ids": ["mem1"], "related_blocked_memory_ids": []},
        )
        evidence = [RetrievedEvidence("mem1", "The meeting is at 10:00.", 1.0, "test", "allowed", "alice", "public", "test", ("m1",), None, {})]
        answer, _ = execute_policy_decision(instance=_instance(question="When is the meeting?"), decision=decision, plan=build_execution_plan(decision), evidence=evidence, llm_client=None, config={})
        self.assertEqual(answer.action, "answer")

    def test_mixed_related_evidence_uses_partial_disclosure(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW, requester="alice", target_subject="case", requested_operation="access",
            allowed_memory_ids=("mem1",), blocked_memory_ids=("mem2",),
            state_snapshot={"related_allowed_memory_ids": ["mem1"], "explicit_related_blocked_memory_ids": ["mem2"]},
        )
        evidence = [RetrievedEvidence("mem1", "The appointment is Tuesday.", 1.0, "test", "allowed", "alice", "public", "test", ("m1",), None, {})]
        answer, _ = execute_policy_decision(instance=_instance(question="What is the appointment and diagnosis?"), decision=decision, plan=build_execution_plan(decision), evidence=evidence, llm_client=None, config={})
        self.assertEqual(answer.action, "answer_redacted")

    def test_superseded_related_memory_does_not_force_redaction(self):
        decision = PolicyDecision(
            action=PolicyAction.ALLOW, requester="alice", target_subject="case", requested_operation="access",
            allowed_memory_ids=("mem1",), blocked_memory_ids=("mem_old",),
            state_snapshot={
                "related_allowed_memory_ids": ["mem1"],
                "related_blocked_memory_ids": ["mem_old"],
                "explicit_related_blocked_memory_ids": [],
                "blocked_reason_by_memory_id": {"mem_old": "lifecycle:superseded"},
            },
        )
        evidence = [RetrievedEvidence("mem1", "The current value is 9%.", 1.0, "test", "allowed", "alice", "public", "test", ("m1",), None, {})]
        answer, _ = execute_policy_decision(instance=_instance(question="What is the current value?"), decision=decision, plan=build_execution_plan(decision), evidence=evidence, llm_client=None, config={})
        self.assertEqual(answer.action, "answer")

    def test_shared_observable_scope_authorizes_operational_fact(self):
        prov = Provenance(source_message_ids=("m1",), evidence_text="Project Maple budget is 221,000 USD.")
        state = PolicyState(
            principals=(PrincipalState("pm"), PrincipalState("finance")),
            memory_items=(MemoryItemState("mem1", "finance", "Project Maple", "budget", "m1", None, MemoryStatus.ACTIVE, prov, ("finance",)),),
            ownership_relations=(), permission_relations=(), delegation_relations=(), revocation_relations=(),
            memory_status={"mem1": MemoryStatus.ACTIVE}, operation_history=(), temporal_constraints=(),
            scope_constraints=(
                {"type": "project_member", "project_id": "project_maple", "principal_id": "pm"},
                {"type": "budget_owner", "project_id": "project_maple", "principal_id": "finance"},
            ),
            provenance=(prov,), as_of_turn_id="m1",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(instance=_instance(requester="pm", question="What is Project Maple's current budget?"), state=state)
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertEqual(decision.allowed_memory_ids, ("mem1",))
        self.assertEqual(decision.state_snapshot["allowed_reason_by_memory_id"]["mem1"], "shared observable project_id=project_maple")

    def test_contractor_relationship_does_not_grant_broad_project_access(self):
        prov = Provenance(source_message_ids=("m1",), evidence_text="Project Maple incident diagnosis is pending.")
        state = PolicyState(
            principals=(PrincipalState("contractor"), PrincipalState("security")),
            memory_items=(MemoryItemState("mem1", "security", "Project Maple", "incident", "m1", None, MemoryStatus.ACTIVE, prov, ("safety",)),),
            ownership_relations=(), permission_relations=(), delegation_relations=(), revocation_relations=(),
            memory_status={"mem1": MemoryStatus.ACTIVE}, operation_history=(), temporal_constraints=(),
            scope_constraints=(
                {"type": "contractor_for_project", "project_id": "project_maple", "principal_id": "contractor"},
                {"type": "project_member", "project_id": "project_maple", "principal_id": "security"},
            ),
            provenance=(prov,), as_of_turn_id="m1",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(requester="contractor", question="What is the Project Maple incident diagnosis?"),
            state=state,
        )
        self.assertNotEqual(decision.action, PolicyAction.ALLOW)
        self.assertEqual(decision.allowed_memory_ids, ())

    def test_requester_owned_sensitive_continuation_keeps_exact_authorization(self):
        provenance = Provenance(
            source_message_ids=("m187",),
            turn_index=187,
            evidence_text="Wednesday November 25 at 8:00 AM.",
        )
        state = PolicyState(
            principals=(PrincipalState("lab_tech_sonia", role="lab_tech", provenance=provenance),),
            memory_items=(MemoryItemState(
                "lab-date", "lab_tech_sonia", "Wednesday November", "private",
                "m187", None, MemoryStatus.ACTIVE, provenance,
            ),),
            ownership_relations=(), permission_relations=(), delegation_relations=(),
            revocation_relations=(), memory_status={"lab-date": MemoryStatus.ACTIVE},
            operation_history=(), temporal_constraints=(), scope_constraints=(),
            provenance=(provenance,), as_of_turn_id="m187",
        )
        decision = StatefulPolicyReasoner(llm_client=None, config={}).decide(
            instance=_instance(
                requester="lab_tech_sonia",
                question="What is the repeat lab date before the December visit?",
            ),
            state=state,
        )
        self.assertEqual(decision.action, PolicyAction.ALLOW)
        self.assertTrue(decision.state_snapshot["sensitive_authorized"])
        self.assertIn("lab-date", decision.allowed_memory_ids)

    def test_deleted_value_probe_does_not_delete_safe_summary_carrier(self):
        instance = MemoryInstance(
            instance_id="safe-summary-probe", domain="household", conversation_id="safe-summary-probe",
            messages=[
                {"message_id": "m1", "speaker_id": "manager", "text": "Current helper-safe summary of Mina's thread: separate Sunday evening off-site handoff, unrelated to pet and parcel plans, no exact minute, no curb.", "timestamp": "2026-01-01T00:00:00"},
                {"message_id": "m2", "speaker_id": "partner", "text": "Was Mina's deleted exact minute 7:15 PM or not?", "timestamp": "2026-01-01T00:01:00"},
                {"message_id": "m3", "speaker_id": "owner", "text": "No one should confirm deleted exact timing.", "timestamp": "2026-01-01T00:02:00"},
            ],
            question="What is the helper-safe summary?", asking_user_id="manager", choices=None, answer=None, metadata={},
        )
        state = build_policy_state(instance, llm_client=None, config={})
        summary_id = next(item.memory_id for item in state.memory_items if item.content_ref == "m1")
        self.assertEqual(state.memory_status[summary_id], MemoryStatus.ACTIVE)
        self.assertFalse(any(summary_id in op.target_memory_ids for op in state.operation_history))

    def test_grounding_pass_can_replace_failed_repair_draft(self):
        class GroundingLLM:
            def __init__(self):
                self.calls = []
            def is_available(self):
                return True
            def chat_json(self, *, model, system_prompt, user_prompt):
                self.calls.append(system_prompt)
                if "This is final answer-value grounding" in system_prompt:
                    return {
                        "answer_text": "The current site is Port Annex desk 8.",
                        "requested_fields": [{"label": "current site", "status": "covered", "answer_text": "Port Annex desk 8", "source_memory_ids": ["mem1"]}],
                        "covered_requested_fields": ["current site"],
                        "restricted_fields_omitted": [],
                        "confidence": 1.0,
                        "safe_answer": True,
                    }
                return {
                    "answer_text": "The current site is Harbor State Exchange Center.",
                    "requested_fields": [{"label": "current site", "status": "covered", "answer_text": "Harbor State Exchange Center", "source_memory_ids": ["mem1"]}],
                    "covered_requested_fields": ["current site"],
                    "restricted_fields_omitted": [],
                    "confidence": 0.7,
                    "safe_answer": True,
                }
        decision = PolicyDecision(
            action=PolicyAction.ALLOW, requester="alice", target_subject="case", requested_operation="access",
            allowed_memory_ids=("mem1",), blocked_memory_ids=(), state_snapshot={},
        )
        evidence = [RetrievedEvidence("mem1", "Current state: Harbor State Exchange Center, Port Annex desk 8.", 1.0, "test", "allowed", "alice", "public", "test", ("m1",), None, {})]
        llm = GroundingLLM()
        answer, _ = execute_policy_decision(
            instance=_instance(question="What is the current site?"),
            decision=decision,
            plan=build_execution_plan(decision),
            evidence=evidence,
            llm_client=llm,
            config={"llm": {"answering_model": "mock"}},
        )
        self.assertIn("Port Annex desk 8", answer.answer_text)
        self.assertTrue(answer.answer_structured["answer_grounding"].get("ran"))


def _instance(requester="alice", question="What is the current case?"):
    return MemoryInstance(
        instance_id="case",
        domain="medical",
        conversation_id="case",
        messages=[{"message_id": "m1", "speaker_id": "owner", "speaker_role": "staff", "text": "observable fact", "timestamp": "2026-01-01T00:00:00"}],
        question=question,
        asking_user_id=requester,
        choices=None,
        answer=None,
        metadata={"requester": {"role": "staff"}, "evaluation": {"expected_action": "answer", "judge_spec": {}, "leak_targets": [], "query_type": "utility"}},
    )


if __name__ == "__main__":
    unittest.main()
