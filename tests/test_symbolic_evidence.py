from dataclasses import replace

from gov_mem.backbones.symbolic_evidence import (
    apply_authorization_evidence_boundary,
    build_symbolic_evidence,
    extract_authorization_assertions,
)
from gov_mem.data.schema import MemoryInstance, RetrievedEvidence


def _instance() -> MemoryInstance:
    return MemoryInstance(
        instance_id="medical_ckpt_01",
        domain="medical",
        conversation_id="episode_01",
        messages=[
            {
                "message_id": "t001",
                "speaker_id": "patient_elena",
                "speaker_role": "patient",
                "text": "Only logistics may be shared with Linda.",
                "timestamp": "2026-03-03T08:00",
                "turn_kind": "dialogue",
            },
            {
                "message_id": "t002",
                "speaker_id": "nurse_alvarez",
                "speaker_role": "nurse",
                "text": "The current callback number is 415-555-0186.",
                "timestamp": "2026-03-03T08:10",
                "turn_kind": "dialogue",
            },
        ],
        question="What is current?",
        asking_user_id="patient_elena",
        choices=None,
        answer=None,
        metadata={
            "observable": {"as_of_turn_id": "t002"},
            "raw_sample": {
                "episode": {
                    "entities": {
                        "principals": [
                            {"principal_id": "patient_elena", "role": "patient"},
                            {"principal_id": "nurse_alvarez", "role": "nurse"},
                        ],
                        "relationships": [
                            {
                                "type": "care_contact",
                                "principal_id": "nurse_alvarez",
                                "patient_id": "patient_elena",
                                "access_scope": "clinical",
                            }
                        ],
                    }
                }
            },
        },
    )


def _evidence(
    *, memory_id: str, turn_id: str, principal_id: str, role: str, text: str,
    score: float, extra_metadata: dict | None = None,
):
    metadata = {
        "structured_record": {
            "record_type": "message",
            "message_id": turn_id,
            "turn_id": turn_id,
            "turn_index": int(turn_id.removeprefix("t")) - 1,
            "timestamp": f"2026-03-03T08:{int(turn_id.removeprefix('t')) - 1:02d}",
            "speaker": {"principal_id": principal_id, "role": role},
            "turn_kind": "dialogue",
            "text": text,
            "checkpoint": {"as_of_turn_id": "t002"},
        }
    }
    metadata.update(extra_metadata or {})
    return RetrievedEvidence(
        memory_id=memory_id,
        content=f"[{role}:{principal_id}] {text}",
        score=score,
        retrieval_source="dense",
        reason="test",
        user_id=principal_id,
        source_message_ids=[turn_id],
        time="2026-03-03T08:00",
        metadata=metadata,
    )


def test_v4_symbolic_layer_preserves_all_candidates_and_adds_typed_trace():
    evidence = [
        _evidence(
            memory_id="m2",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="The current callback number is 415-555-0186.",
            score=0.8,
        ),
        _evidence(
            memory_id="m1",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="Only logistics may be shared with Linda.",
            score=0.8,
        ),
    ]

    ranked, trace = build_symbolic_evidence(instance=_instance(), evidence=evidence)

    assert {row.memory_id for row in ranked} == {"m1", "m2"}
    assert trace["symbolic_step"] == "typed_principal_entity_relation_graph_v1"
    assert trace["version"] == "Gov-Mem-v4-Symbolic-dev2"
    assert trace["structured_record_count"] == 2
    assert trace["consistency"]["passed"] is True
    assert ranked[0].metadata["symbolic_provenance"]["record_complete"] is True
    assert ranked[0].metadata["symbolic_consistency"]["passed"] is True
    assert [row.memory_id for row in ranked] == ["m2", "m1"]
    assert trace["ordering_changed"] is False
    assert trace["new_llm_calls"] == 0
    assert trace["graph_type"] == "evidence_principal_typed_relation_lifecycle"
    assert trace["graph_node_count"] == 4
    assert trace["graph_edge_count"] == 3
    assert ranked[0].metadata["graph_context"]["source_relation"] == "spoken_by"
    assert ranked[0].metadata["graph_context"]["same_speaker_evidence_count"] == 1
    assert ranked[0].metadata["graph_context"]["relation_count"] == 1
    assert ranked[0].metadata["graph_context"]["relations"][0]["edge_type"] == "care_contact"
    assert ranked[0].metadata["symbolic_validity_certificate"]["state"] == "unknown"
    assert ranked[0].metadata["symbolic_validity_certificate"]["mode"] == "shadow"
    assert trace["validity_projection"]["enforcement_applied"] is False


def test_authorization_boundary_does_not_treat_missing_policy_as_denial():
    evidence = [
        _evidence(
            memory_id="fact",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="The current callback number is 415-555-0186.",
            score=0.8,
        )
    ]

    released, audit = apply_authorization_evidence_boundary(
        evidence=evidence,
        certificate={
            "decision": "unknown",
            "current_authorization": [],
        },
    )

    assert [row.memory_id for row in released] == ["fact"]
    assert audit["enforcement_applied"] is False


def test_authorization_boundary_filters_only_grounded_denied_resource():
    evidence = [
        _evidence(
            memory_id="secret",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="The confidential callback number is 415-555-0186.",
            score=0.8,
        ),
        _evidence(
            memory_id="ordinary",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="The clinic opens at 08:00.",
            score=0.7,
        ),
    ]

    released, audit = apply_authorization_evidence_boundary(
        evidence=evidence,
        certificate={
            "decision": "deny",
            "current_authorization": [{
                "principal": "patient_elena",
                "resource": "confidential callback number",
                "decision": "deny",
                "supporting_evidence_ids": ["policy"],
            }],
        },
    )

    assert [row.memory_id for row in released] == ["ordinary"]
    assert audit["filtered_memory_ids"] == ["secret"]


def test_authorization_boundary_does_not_match_single_shared_token():
    evidence = [
        _evidence(
            memory_id="ordinary",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="The callback desk is open today.",
            score=0.7,
        )
    ]

    released, audit = apply_authorization_evidence_boundary(
        evidence=evidence,
        certificate={
            "decision": "deny",
            "current_authorization": [{
                "principal": "patient_elena",
                "resource": "confidential callback number",
                "decision": "deny",
                "supporting_evidence_ids": [],
            }],
        },
    )

    assert [row.memory_id for row in released] == ["ordinary"]
    assert audit["filtered_memory_ids"] == []


def test_authorization_boundary_projects_only_grounded_denied_claim_span():
    mixed = _evidence(
        memory_id="mixed", turn_id="t002", principal_id="coach_ivy", role="coach",
        text=("The private review date is September 9. "
              "The public room is Hall A."), score=0.8,
    )
    released, audit = apply_authorization_evidence_boundary(
        evidence=[mixed],
        certificate={
            "decision": "deny",
            "current_authorization": [{
                "principal": "student_aria",
                "resource": "private review date",
                "decision": "deny",
                "supporting_evidence_ids": ["policy"],
            }],
        },
        semantic_atoms=[{
            "atom_id": "private-date",
            "slot_name": "private review date",
            "source": {
                "chunk_id": "mixed",
                "span": "The private review date is September 9.",
            },
        }],
    )

    assert [row.memory_id for row in released] == ["mixed"]
    assert "September 9" not in released[0].content
    assert "The public room is Hall A." in released[0].content
    assert audit["filtered_memory_ids"] == []
    assert audit["projected_memory_ids"] == ["mixed"]


def test_authorization_boundary_fails_closed_without_grounded_claim_projection():
    mixed = _evidence(
        memory_id="mixed", turn_id="t002", principal_id="coach_ivy", role="coach",
        text="The private review date is September 9. The public room is Hall A.", score=0.8,
    )
    released, audit = apply_authorization_evidence_boundary(
        evidence=[mixed],
        certificate={
            "decision": "deny",
            "current_authorization": [{
                "principal": "student_aria",
                "resource": "private review date",
                "decision": "deny",
                "supporting_evidence_ids": ["policy"],
            }],
        },
        semantic_atoms=[],
    )

    assert released == []
    assert audit["filtered_memory_ids"] == ["mixed"]


def test_source_grounded_restricted_atom_does_not_become_authorization_decision():
    row = _evidence(
        memory_id="private", turn_id="t002", principal_id="coach_ivy", role="coach",
        text="The case blocker remains private: clearance form.", score=0.8,
    )
    released, audit = apply_authorization_evidence_boundary(
        evidence=[row],
        certificate={"decision": "unknown", "current_authorization": []},
        semantic_atoms=[{
            "atom_id": "restricted-slot", "slot_id": "qslot_0",
            "slot_name": "current case blocker",
            "sensitivity_semantics": {"type": "restricted"},
            "source": {
                "chunk_id": "private",
                "span": "The case blocker remains private: clearance form.",
            },
        }],
    )

    assert [item.memory_id for item in released] == ["private"]
    assert audit["enforcement_applied"] is False


def test_legacy_raw_policy_scope_text_is_not_a_v4_authorization_source():
    instance = replace(_instance(), question="What clinical result is current?")
    evidence = [
        _evidence(
            memory_id="policy",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="Only logistics may be shared with Linda; clinical results and medications remain restricted.",
            score=0.9,
        )
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance,
        evidence=evidence,
        policy_consistency_enabled=True,
    )

    certificate = trace["policy_consistency"]
    assert certificate["decision"] == "unknown"
    assert certificate["scope_decisions"] == {}
    assert certificate["enforcement_applied"] is False
    assert ranked[0].metadata["symbolic_policy_certificate"] == certificate
    assert "symbolic_policy_facts" not in ranked[0].metadata


def test_legacy_raw_policy_scope_text_cannot_create_a_v4_allow():
    instance = replace(_instance(), question="What clinical result is current?")
    evidence = [
        _evidence(
            memory_id="old_policy",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="Clinical results remain restricted to Elena.",
            score=0.9,
        ),
        _evidence(
            memory_id="new_policy",
            turn_id="t002",
            principal_id="patient_elena",
            role="patient",
            text="Linda may receive the current clinical results.",
            score=0.8,
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance,
        evidence=evidence,
        policy_consistency_enabled=True,
    )

    assert trace["policy_consistency"]["decision"] == "unknown"
    assert trace["policy_consistency"]["supporting_evidence_ids"] == []


def test_dev4_policy_consistency_is_disabled_by_default():
    _, trace = build_symbolic_evidence(instance=_instance(), evidence=[])

    assert trace["policy_consistency"]["enabled"] is False


def test_v4_symbolic_layer_detects_role_and_checkpoint_conflicts():
    bad = _evidence(
        memory_id="bad",
        turn_id="t002",
        principal_id="nurse_alvarez",
        role="patient",
        text="The current callback number is 415-555-0186.",
        score=0.9,
    )

    _, trace = build_symbolic_evidence(instance=_instance(), evidence=[bad])

    kinds = {item["kind"] for item in trace["consistency"]["violations"]}
    assert "principal_role_conflict" in kinds
    assert trace["consistency"]["passed"] is False
    assert trace["consistency"]["violation_count"] == 1


def test_v4_symbolic_relation_is_directional_not_duplicated_for_two_principals():
    _, trace = build_symbolic_evidence(instance=_instance(), evidence=[])
    relation_edges = [edge for edge in trace["graph_edges"] if edge["edge_type"] == "care_contact"]
    assert len(relation_edges) == 1
    assert relation_edges[0]["source"] == "principal::nurse_alvarez"
    assert relation_edges[0]["target"] == "principal::patient_elena"


def test_v4_symbolic_lifecycle_consumes_only_grounded_compiler_atoms():
    evidence = [
        _evidence(
            memory_id="m_delete",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="Delete the old callback number from memory; it should no longer be available.",
            score=0.8,
        ),
        _evidence(
            memory_id="m_current",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="The current callback number is 415-555-0186.",
            score=0.7,
        ),
    ]

    query_analysis = {
        "semantic_compiler_contract": True,
        "fields": [{"slot_id": "qslot_0", "name": "callback number"}],
    }
    atoms = [{
        "atom_id": "delete", "slot_id": "qslot_0", "slot_name": "callback number",
        "lifecycle_semantics": {"type": "delete"},
        "source": {
            "chunk_id": "m_delete", "turn_id": "t002",
            "span": "Delete the old callback number from memory; it should no longer be available.",
        },
    }]
    ranked, trace = build_symbolic_evidence(
        instance=_instance(), evidence=evidence,
        query_analysis=query_analysis, semantic_atoms=atoms,
    )

    assert [row.memory_id for row in ranked] == ["m_delete", "m_current"]
    assert trace["lifecycle_status_counts"] == {"deleted": 1}
    lifecycle_edges = [
        edge for edge in trace["graph_edges"] if edge["edge_type"] == "asserts_lifecycle"
    ]
    assert len(lifecycle_edges) == 1
    assert lifecycle_edges[0]["status"] == "deleted"
    target_edges = [edge for edge in trace["graph_edges"] if edge["edge_type"] == "applies_lifecycle_to_slot"]
    assert len(target_edges) == 1
    assert target_edges[0]["source"] == "lifecycle::m_delete"
    assert target_edges[0]["target"] == "query_slot::qslot_0"
    binding = trace["lifecycle_target_binding"]
    assert binding["status_counts"] == {"candidate_bound": 1}
    assert binding["bound_count"] == 0
    assert binding["ambiguous_count"] == 0
    assert binding["unbound_count"] == 0
    assert binding["bindings"][0]["source_memory_id"] == "m_delete"
    assert binding["bindings"][0]["slot_id"] == "qslot_0"
    assert ranked[0].metadata["graph_context"]["lifecycle_claim"]["explicit"] is True
    assert ranked[0].metadata["symbolic_validity_certificate"] == {
        "mode": "shadow",
        "state": "explicit_inactive",
        "current_answer_eligibility": "blocked_in_enforced_mode",
        "explicit": True,
        "lifecycle_status": "deleted",
        "reason": "explicit_lifecycle_assertion",
    }
    assert ranked[1].metadata["graph_context"]["lifecycle_claim"] is None
    assert ranked[1].metadata["symbolic_validity_certificate"]["state"] == "unknown"
    assert trace["validity_projection"]["state_counts"] == {
        "explicit_inactive": 1,
        "unknown": 1,
    }
    assert trace["ordering_changed"] is False
    assert trace["filtering_applied"] is False
    assert trace["new_llm_calls"] == 0


def test_v4_symbolic_lifecycle_does_not_bind_raw_text_to_a_prior_memory():
    evidence = [
        _evidence(
            memory_id="m_old_a",
            turn_id="t001",
            principal_id="nurse_alvarez",
            role="nurse",
            text="The callback number for the clinic is 415-555-0101.",
            score=0.8,
        ),
        _evidence(
            memory_id="m_old_b",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="The backup callback number is 415-555-0102.",
            score=0.7,
        ),
        _evidence(
            memory_id="m_delete",
            turn_id="t003",
            principal_id="nurse_alvarez",
            role="nurse",
            text="Delete the callback number from memory; it should no longer be available.",
            score=0.6,
        ),
    ]

    _, trace = build_symbolic_evidence(instance=_instance(), evidence=evidence)

    assert trace["lifecycle_target_binding"]["status_counts"] == {}
    assert trace["lifecycle_target_binding"]["bound_count"] == 0
    assert not [edge for edge in trace["graph_edges"] if edge["edge_type"] == "invalidates"]


def test_v4_symbolic_lifecycle_rejects_choice_and_negated_mentions():
    evidence = [
        _evidence(
            memory_id="m_question",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="Do you want Friday instead of Monday?",
            score=0.8,
        ),
        _evidence(
            memory_id="m_boundary",
            turn_id="t001",
            principal_id="patient_elena",
            role="patient",
            text="Noise does not supersede the explicit current plan.",
            score=0.7,
        ),
    ]

    _, trace = build_symbolic_evidence(instance=_instance(), evidence=evidence)

    assert trace["lifecycle_claim_count"] == 0
    assert not [edge for edge in trace["graph_edges"] if edge["edge_type"] == "asserts_lifecycle"]


def test_v4_symbolic_lifecycle_does_not_treat_deleted_question_as_assertion():
    evidence = [
        _evidence(
            memory_id="m_question",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="What was the deleted callback number?",
            score=0.8,
        ),
    ]

    _, trace = build_symbolic_evidence(instance=_instance(), evidence=evidence)

    assert trace["lifecycle_claim_count"] == 0
    assert trace["lifecycle_target_binding"]["status_counts"] == {}


def test_v4_symbolic_state_ledger_resolves_latest_grounded_atom_without_filtering():
    instance = replace(
        _instance(),
        question=(
            "Give me the current recap: current date, current status, and current blocker."
        ),
    )
    evidence = [
        _evidence(
            memory_id="m_old",
            turn_id="t001",
            principal_id="nurse_alvarez",
            role="nurse",
            text=(
                "Current case recap: current date July 8, 2026, status pending, "
                "and current blocker budget review."
            ),
            score=0.9,
        ),
        _evidence(
            memory_id="m_new",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text=(
                "Closure note: the case should now be treated as closed, with "
                "July 12, 2026 as the settled date and no remaining blocker."
            ),
            score=0.7,
        ),
    ]

    query_analysis = {
        "semantic_compiler_contract": True,
        "fields": [
            {"slot_id": "qslot_0", "name": "current date", "required": True, "temporal_requirement": "current"},
            {"slot_id": "qslot_1", "name": "current status", "required": True, "temporal_requirement": "current"},
            {"slot_id": "qslot_2", "name": "current blocker", "required": True, "temporal_requirement": "current"},
        ],
    }
    atoms = [
        {
            "atom_id": "a_old_date", "slot_id": "qslot_0", "slot_name": "current date",
            "value": "July 8, 2026", "temporal": {"state": "current"},
            "source": {"chunk_id": "m_old", "turn_id": "t001", "message_id": "t001", "span": "current date July 8, 2026"},
        },
        {
            "atom_id": "a_new_date", "slot_id": "qslot_0", "slot_name": "current date",
            "value": "July 12, 2026", "temporal": {"state": "current"},
            "source": {"chunk_id": "m_new", "turn_id": "t002", "message_id": "t002", "span": "July 12, 2026 as the settled date"},
        },
        {
            "atom_id": "a_old_status", "slot_id": "qslot_1", "slot_name": "current status",
            "value": "pending", "temporal": {"state": "current"},
            "source": {"chunk_id": "m_old", "turn_id": "t001", "message_id": "t001", "span": "status pending"},
        },
        {
            "atom_id": "a_new_status", "slot_id": "qslot_1", "slot_name": "current status",
            "value": "closed", "temporal": {"state": "current"},
            "source": {"chunk_id": "m_new", "turn_id": "t002", "message_id": "t002", "span": "treated as closed"},
        },
        {
            "atom_id": "a_old_blocker", "slot_id": "qslot_2", "slot_name": "current blocker",
            "value": "budget review", "temporal": {"state": "current"},
            "source": {"chunk_id": "m_old", "turn_id": "t001", "message_id": "t001", "span": "current blocker budget review"},
        },
        {
            "atom_id": "a_new_blocker", "slot_id": "qslot_2", "slot_name": "current blocker",
            "value": "no remaining blocker", "temporal": {"state": "current"},
            "source": {"chunk_id": "m_new", "turn_id": "t002", "message_id": "t002", "span": "no remaining blocker"},
        },
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance,
        evidence=evidence,
        query_analysis=query_analysis,
        semantic_atoms=atoms,
    )

    ledger = trace["state_ledger"]
    assert ledger["version"] == "state-ledger-v2-semantic-atoms"
    assert ledger["mode"] == "verified_semantic_atoms_closed_evidence"
    assert ledger["fields"]["current date"]["value"] == "July 12, 2026"
    assert ledger["fields"]["current date"]["source_memory_id"] == "m_new"
    assert ledger["fields"]["current status"]["value"] == "closed"
    assert ledger["fields"]["current status"]["source_memory_id"] == "m_new"
    assert ledger["fields"]["current blocker"]["value"] == "no remaining blocker"
    assert ledger["fields"]["current blocker"]["source_memory_id"] == "m_new"
    assert ledger["fields"]["current date"]["conflict_count"] == 0
    assert ledger["fields"]["current date"]["candidate_count"] == 2
    assert ledger["enforcement_applied"] is False
    assert ledger["new_llm_calls"] == 0
    assert [row.memory_id for row in ranked] == ["m_old", "m_new"]
    assert ranked[0].metadata["symbolic_state_ledger"] == ledger


def test_v4_symbolic_state_ledger_does_not_reactivate_typed_frame_extraction():
    instance = replace(
        _instance(),
        question="What allergy is documented for me?",
    )
    evidence = [
        _evidence(
            memory_id="m_allergy",
            turn_id="t002",
            principal_id="nurse_alvarez",
            role="nurse",
            text="The allergy on file is a sulfa antibiotic rash.",
            score=0.9,
        ),
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance,
        evidence=evidence,
        required_slot_plan={"required_slots": ["substance", "reaction"]},
    )

    ledger = trace["state_ledger"]
    assert ledger["mode"] == "disabled_without_semantic_contract"
    assert ledger["requested_slots"] == []
    assert ledger["fields"] == {}
    assert ledger["resolved_count"] == 0
    assert ranked[0].metadata["symbolic_state_ledger"] == ledger


def test_v4_semantic_atoms_drive_open_vocabulary_state_ledger():
    instance = replace(
        _instance(),
        question="What is the current review date and support amount?",
    )
    evidence = [
        _evidence(
            memory_id="m_old", turn_id="t001", principal_id="nurse_alvarez",
            role="nurse", text="The support amount was 4,150 USD.", score=0.7,
        ),
        _evidence(
            memory_id="m_current", turn_id="t002", principal_id="nurse_alvarez",
            role="nurse",
            text="The current review date is August 4 and support amount is 4,130 USD.",
            score=0.9,
        ),
    ]
    atoms = [
        {
            "atom_id": "date", "slot_id": "qslot_0",
            "slot_name": "current review date", "value": "August 4",
            "temporal": {"state": "current"},
            "lifecycle_semantics": {"type": "assert"},
            "source": {"chunk_id": "m_current", "turn_id": "t002", "span": "current review date is August 4"},
            "confidence": 0.9,
        },
        {
            "atom_id": "old-amount", "slot_id": "qslot_1",
            "slot_name": "support amount", "value": "4,150 USD",
            "temporal": {"state": "historical"},
            "lifecycle_semantics": {"type": "none"},
            "source": {"chunk_id": "m_old", "turn_id": "t001", "span": "support amount was 4,150 USD"},
            "confidence": 0.8,
        },
        {
            "atom_id": "current-amount", "slot_id": "qslot_1",
            "slot_name": "support amount", "value": "4,130 USD",
            "temporal": {"state": "current"},
            "lifecycle_semantics": {"type": "update"},
            "source": {"chunk_id": "m_current", "turn_id": "t002", "span": "support amount is 4,130 USD"},
            "confidence": 0.9,
        },
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance,
        evidence=evidence,
        query_analysis={
            "semantic_compiler_contract": True,
            "fields": [
                {"slot_id": "qslot_0", "name": "current review date", "temporal_requirement": "current"},
                {"slot_id": "qslot_1", "name": "support amount", "temporal_requirement": "current"},
            ],
        },
        semantic_atoms=atoms,
    )

    ledger = trace["state_ledger"]
    assert ledger["version"] == "state-ledger-v2-semantic-atoms"
    assert ledger["resolved_count"] == 2
    assert ledger["fields"]["current review date"]["value"] == "August 4"
    assert ledger["fields"]["support amount"]["value"] == "4,130 USD"
    assert ledger["fields"]["support amount"]["source_memory_id"] == "m_current"
    assert ranked[0].metadata["symbolic_state_ledger"] == ledger


def _semantic_policy_context(resource: str = "clinical records"):
    return {
        "semantic_compiler_contract": True,
        "fields": [{"slot_id": "qslot_0", "name": resource}],
    }


def _policy_atom(*, atom_id: str, memory_id: str, turn_id: str, span: str, effect: str,
                 principal: str = "nurse_alvarez", resource: str = "clinical records"):
    return {
        "atom_id": atom_id,
        "slot_id": "qslot_0",
        "slot_name": resource,
        "authorization_semantics": {
            "type": effect,
            "subject": principal,
            "resource": resource,
        },
        "source": {
            "chunk_id": memory_id,
            "turn_id": turn_id,
            "message_id": turn_id,
            "span": span,
        },
    }


def test_temporal_authorization_graph_applies_allow_then_revoke():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="grant", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may access clinical records.", score=0.9,
        ),
        _evidence(
            memory_id="revoke", turn_id="t002", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez access to clinical records is revoked.", score=0.8,
        ),
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context(),
        semantic_atoms=[
            _policy_atom(
                atom_id="allow", memory_id="grant", turn_id="t001",
                span="nurse_alvarez may access clinical records.", effect="allow",
            ),
            _policy_atom(
                atom_id="revoke", memory_id="revoke", turn_id="t002",
                span="nurse_alvarez access to clinical records is revoked.", effect="revoke",
            ),
        ],
    )

    certificate = trace["temporal_authorization"]
    assert certificate["version"] == "temporal-authorization-v1"
    assert certificate["decision"] == "deny"
    assert certificate["current_authorization"][0]["decision"] == "deny"
    assert certificate["current_authorization"][0]["supporting_evidence_ids"] == ["grant", "revoke"]
    assert any(edge["edge_type"] == "revokes" for edge in certificate["graph_edges"])
    assert ranked[0].metadata["symbolic_temporal_authorization_certificate"] == certificate
    assert certificate["enforcement_applied"] is False
    assert certificate["new_llm_calls"] == 0


def test_temporal_authorization_replays_retrieved_entity_list():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="grant", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may access clinical records.", score=0.9,
        ),
        _evidence(
            memory_id="revoke", turn_id="t002", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez access to clinical records is revoked.", score=0.8,
        ),
    ]
    entity_lists = {
        "entity::clinical records": [
            {
                "relation_type": "permission", "effect": "allow",
                "subject": "nurse_alvarez", "target": "clinical records",
                "source_chunk_id": "grant", "source_message_id": "t001",
                "source_span": "nurse_alvarez may access clinical records.",
            },
            {
                "relation_type": "permission", "effect": "revoke",
                "subject": "nurse_alvarez", "target": "clinical records",
                "source_chunk_id": "revoke", "source_message_id": "t002",
                "source_span": "nurse_alvarez access to clinical records is revoked.",
            },
        ],
    }
    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context(), semantic_atoms=[],
        governed_entity_lists=entity_lists,
    )
    certificate = trace["temporal_authorization"]
    assert certificate["entity_list_event_count"] == 2
    assert certificate["decision"] == "deny"
    assert all(
        event.get("source") == "retrieved_entity_relation_list"
        for event in certificate["graph_nodes"]
        if event.get("node_type") == "PolicyEvent"
    )


def test_dev7_boundary_filters_a_denied_resource_and_preserves_unrelated_evidence():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="grant", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may access clinical records.", score=0.9,
        ),
        _evidence(
            memory_id="revoke", turn_id="t002", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez access to clinical records is revoked.", score=0.8,
        ),
        _evidence(
            memory_id="ordinary", turn_id="t001", principal_id="patient_elena", role="patient",
            text="The clinic opens at 08:00.", score=0.7,
        ),
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance,
        evidence=evidence,
        temporal_authorization_enabled=True,
        temporal_authorization_enforcement=True,
        query_analysis=_semantic_policy_context(),
        semantic_atoms=[
            _policy_atom(
                atom_id="allow", memory_id="grant", turn_id="t001",
                span="nurse_alvarez may access clinical records.", effect="allow",
            ),
            _policy_atom(
                atom_id="revoke", memory_id="revoke", turn_id="t002",
                span="nurse_alvarez access to clinical records is revoked.", effect="revoke",
            ),
        ],
    )

    assert trace["version"] == "Gov-Mem-v4-Symbolic-dev7"
    assert trace["temporal_authorization"]["decision"] == "deny"
    assert trace["authorization_evidence_boundary"]["enforcement_applied"] is True
    assert {row.memory_id for row in ranked} == {"ordinary"}


def test_temporal_authorization_graph_later_allow_supersedes_older_deny():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="old", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may not access clinical records.", score=0.9,
        ),
        _evidence(
            memory_id="new", turn_id="t002", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may access clinical records.", score=0.8,
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context(),
        semantic_atoms=[
            _policy_atom(
                atom_id="deny", memory_id="old", turn_id="t001",
                span="nurse_alvarez may not access clinical records.", effect="deny",
            ),
            _policy_atom(
                atom_id="allow", memory_id="new", turn_id="t002",
                span="nurse_alvarez may access clinical records.", effect="allow",
            ),
        ],
    )

    certificate = trace["temporal_authorization"]
    assert certificate["decision"] == "allow"
    assert certificate["conflict_count"] == 0
    assert any(edge["edge_type"] == "allows" for edge in certificate["graph_edges"])


def test_temporal_authorization_graph_marks_same_time_allow_deny_unknown():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="allow", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may access clinical records.", score=0.9,
        ),
        _evidence(
            memory_id="deny", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may not access clinical records.", score=0.8,
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context(),
        semantic_atoms=[
            _policy_atom(
                atom_id="allow", memory_id="allow", turn_id="t001",
                span="nurse_alvarez may access clinical records.", effect="allow",
            ),
            _policy_atom(
                atom_id="deny", memory_id="deny", turn_id="t001",
                span="nurse_alvarez may not access clinical records.", effect="deny",
            ),
        ],
    )

    certificate = trace["temporal_authorization"]
    assert certificate["decision"] == "unknown"
    assert certificate["conflict_count"] == 1
    assert certificate["current_authorization"][0]["conflict"] is True


def test_temporal_authorization_graph_ignores_future_event_at_checkpoint():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="future", turn_id="t003", principal_id="nurse_alvarez", role="nurse",
            text="nurse_alvarez may access clinical records.", score=0.9,
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context(),
        semantic_atoms=[
            _policy_atom(
                atom_id="future", memory_id="future", turn_id="t003",
                span="nurse_alvarez may access clinical records.", effect="allow",
            ),
        ],
    )

    certificate = trace["temporal_authorization"]
    assert certificate["event_count"] == 0
    assert certificate["ignored_future_event_count"] == 1
    assert certificate["decision"] == "unknown"


def test_temporal_authorization_graph_is_conservative_without_temporal_source():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    row = _evidence(
        memory_id="untimed", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
        text="nurse_alvarez may access clinical records.", score=0.9,
    )
    record = dict(row.metadata["structured_record"])
    record.pop("turn_index")
    record.pop("timestamp")
    row = replace(row, metadata={**row.metadata, "structured_record": record})

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=[row], temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context(),
        semantic_atoms=[
            _policy_atom(
                atom_id="untimed", memory_id="untimed", turn_id="t001",
                span="nurse_alvarez may access clinical records.", effect="allow",
            ),
        ],
    )

    certificate = trace["temporal_authorization"]
    assert certificate["event_count"] == 0
    assert certificate["unknown_event_count"] == 1
    assert certificate["decision"] == "unknown"


def test_raw_authorization_grammar_is_disabled_in_the_release_path():
    assertions = extract_authorization_assertions(
        "Linda Park may receive scheduling details only; clinical results remain restricted."
    )

    assert assertions == []


def test_temporal_authorization_resolves_compiler_atom_subject_against_episode_roster():
    metadata = dict(_instance().metadata)
    metadata["raw_sample"] = {
        "episode": {
            "entities": {
                "principals": [
                    {"principal_id": "family_linda", "role": "family_member", "display_name": "Linda Park"},
                    {"principal_id": "patient_elena", "role": "patient", "display_name": "Elena Park"},
                ],
                "relationships": [],
            }
        },
    }
    instance = replace(
        _instance(),
        question="What access does Linda Park have to scheduling details?",
        metadata=metadata,
    )
    row = _evidence(
        memory_id="policy",
        turn_id="t001",
        principal_id="patient_elena",
        role="patient",
        text="Linda Park may receive scheduling details only.",
        score=0.9,
    )
    _, trace = build_symbolic_evidence(
        instance=instance,
        evidence=[row],
        temporal_authorization_enabled=True,
        query_analysis=_semantic_policy_context("scheduling details"),
        semantic_atoms=[
            _policy_atom(
                atom_id="allow", memory_id="policy", turn_id="t001",
                span="Linda Park may receive scheduling details only.",
                effect="allow", principal="Linda Park", resource="scheduling details",
            ),
        ],
    )

    certificate = trace["temporal_authorization"]
    assert certificate["event_count"] == 1
    assert certificate["decision"] == "allow"
    assert certificate["current_authorization"][0]["principal"] == "family_linda"
    assert certificate["graph_edges"][0]["edge_type"] == "has_role"
    assert any(
        event.get("source") == "grounded_semantic_compiler"
        for event in certificate["graph_nodes"]
        if event.get("node_type") == "PolicyEvent"
    )


def test_grounded_semantic_policy_atom_enters_graph_then_enforces_boundary():
    metadata = dict(_instance().metadata)
    metadata["raw_sample"] = {
        "episode": {"entities": {"principals": [
            {"principal_id": "patient_elena", "role": "patient", "display_name": "Elena Park"},
            {"principal_id": "family_linda", "role": "family_member", "display_name": "Linda Park"},
        ], "relationships": []}},
    }
    instance = replace(
        _instance(),
        question="Can Linda Park receive the current clinical record?",
        metadata=metadata,
    )
    policy = _evidence(
        memory_id="policy", turn_id="t001", principal_id="patient_elena", role="patient",
        text="Linda Park must not receive the clinical record.", score=0.9,
    )
    fact = _evidence(
        memory_id="fact", turn_id="t002", principal_id="patient_elena", role="patient",
        text="The clinical record contains a detailed diagnosis.", score=0.8,
    )
    atoms = [{
        "atom_id": "a0", "slot_id": "qslot_0", "slot_name": "clinical record",
        "authorization_semantics": {
            "type": "deny", "subject": "Linda Park", "resource": "clinical record",
        },
        "source": {"chunk_id": "policy", "span": "Linda Park must not receive the clinical record."},
    }]

    ranked, trace = build_symbolic_evidence(
        instance=instance, evidence=[policy, fact], semantic_atoms=atoms,
        temporal_authorization_enabled=True, temporal_authorization_enforcement=True,
        query_analysis={"fields": [{"name": "clinical record"}]},
    )

    certificate = trace["temporal_authorization"]
    assert certificate["semantic_candidate_event_count"] == 1
    assert certificate["decision"] == "deny"
    assert trace["authorization_evidence_boundary"]["enforcement_applied"] is True
    assert ranked == []


def test_cross_span_grounded_policy_relation_enters_graph():
    metadata = dict(_instance().metadata)
    metadata["raw_sample"] = {
        "episode": {"entities": {"principals": [
            {"principal_id": "patient_elena", "role": "patient", "display_name": "Elena Park"},
            {"principal_id": "family_linda", "role": "family_member", "display_name": "Linda Park"},
        ], "relationships": []}},
    }
    instance = replace(
        _instance(),
        question="Can Linda Park receive the current clinical record?",
        metadata=metadata,
    )
    subject = _evidence(
        memory_id="subject", turn_id="t001", principal_id="patient_elena", role="patient",
        text="Linda Park is the named delegate for this request.", score=0.9,
    )
    policy = _evidence(
        memory_id="policy", turn_id="t002", principal_id="patient_elena", role="patient",
        text="The clinical record is denied to the delegate.", score=0.8,
    )
    atoms = [{
        "atom_id": "cross-span-deny", "slot_id": "qslot_0", "slot_name": "clinical record",
        "authorization_semantics": {
            "type": "deny", "subject": "Linda Park", "resource": "clinical record",
            "subject_source": {
                "chunk_id": "subject", "turn_id": "t001", "message_id": "t001",
                "span": "Linda Park is the named delegate for this request.",
            },
        },
        "source": {
            "chunk_id": "policy", "turn_id": "t002", "message_id": "t002",
            "span": "The clinical record is denied to the delegate.",
        },
    }]

    ranked, trace = build_symbolic_evidence(
        instance=instance, evidence=[subject, policy], semantic_atoms=atoms,
        temporal_authorization_enabled=True, temporal_authorization_enforcement=True,
        query_analysis={"fields": [{"slot_id": "qslot_0", "name": "clinical record"}]},
    )

    certificate = trace["temporal_authorization"]
    assert certificate["semantic_candidate_event_count"] == 1
    assert certificate["decision"] == "deny"
    assert trace["authorization_evidence_boundary"]["enforcement_applied"] is True
    assert [row.memory_id for row in ranked] == ["subject"]


def test_ungrounded_semantic_policy_atom_never_enters_authorization_graph():
    instance = replace(_instance(), question="Can nurse_alvarez access clinical records?")
    row = _evidence(
        memory_id="policy", turn_id="t001", principal_id="patient_elena", role="patient",
        text="A policy note exists.", score=0.9,
    )
    atoms = [{
        "atom_id": "bad", "slot_id": "qslot_0", "slot_name": "clinical records",
        "authorization_semantics": {"type": "deny", "subject": "nurse_alvarez"},
        "source": {"chunk_id": "policy", "span": "absent policy statement"},
    }]
    _, trace = build_symbolic_evidence(
        instance=instance, evidence=[row], semantic_atoms=atoms,
        temporal_authorization_enabled=True,
    )
    certificate = trace["temporal_authorization"]
    assert certificate["semantic_candidate_event_count"] == 0
    assert certificate["rejected_semantic_candidate_events"] == [
        {"atom_id": "bad", "reason": "source_not_grounded"}
    ]
