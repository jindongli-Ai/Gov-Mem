from dataclasses import replace

from gov_mem.backbones.symbolic_evidence import (
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


def test_dev4_policy_certificate_denies_sensitive_scope_for_logistics_only_policy():
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
    assert certificate["decision"] == "deny"
    assert certificate["scope_decisions"]["clinical"] == "deny"
    assert certificate["enforcement_applied"] is False
    assert ranked[0].metadata["symbolic_policy_certificate"] == certificate
    assert ranked[0].metadata["symbolic_policy_facts"]


def test_dev4_policy_certificate_uses_latest_explicit_policy_fact():
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

    assert trace["policy_consistency"]["decision"] == "allow"
    assert trace["policy_consistency"]["supporting_evidence_ids"] == ["new_policy"]


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


def test_v4_symbolic_lifecycle_requires_explicit_language_and_is_annotation_only():
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

    ranked, trace = build_symbolic_evidence(instance=_instance(), evidence=evidence)

    assert [row.memory_id for row in ranked] == ["m_delete", "m_current"]
    assert trace["lifecycle_status_counts"] == {"deleted": 1}
    lifecycle_edges = [
        edge for edge in trace["graph_edges"] if edge["edge_type"] == "asserts_lifecycle"
    ]
    assert len(lifecycle_edges) == 1
    assert lifecycle_edges[0]["status"] == "deleted"
    target_edges = [edge for edge in trace["graph_edges"] if edge["edge_type"] == "invalidates"]
    assert len(target_edges) == 1
    assert target_edges[0]["source"] == "lifecycle::m_delete"
    assert target_edges[0]["target"] == "evidence::m_current"
    binding = trace["lifecycle_target_binding"]
    assert binding["status_counts"] == {"bound": 1}
    assert binding["bound_count"] == 1
    assert binding["ambiguous_count"] == 0
    assert binding["unbound_count"] == 0
    assert binding["bindings"][0]["source_memory_id"] == "m_delete"
    assert binding["bindings"][0]["target_memory_id"] == "m_current"
    assert binding["bindings"][0]["target_turn_id"] == "t001"
    assert binding["bindings"][0]["edge_type"] == "invalidates"
    assert binding["bindings"][0]["inference"] == "explicit_lifecycle_prior_overlap"
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


def test_v4_symbolic_lifecycle_binding_stays_ambiguous_without_unique_target():
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

    assert trace["lifecycle_target_binding"]["status_counts"] == {"ambiguous": 1}
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


def test_v4_symbolic_state_ledger_resolves_latest_retrieved_claim_without_filtering():
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

    ranked, trace = build_symbolic_evidence(instance=instance, evidence=evidence)

    ledger = trace["state_ledger"]
    assert ledger["version"] == "state-ledger-v1"
    assert ledger["mode"] == "retrieved_evidence_only"
    assert ledger["fields"]["target_date"]["value"] == "July 12, 2026"
    assert ledger["fields"]["target_date"]["source_memory_id"] == "m_new"
    assert ledger["fields"]["status"]["value"] == "closed"
    assert ledger["fields"]["status"]["source_memory_id"] == "m_new"
    assert ledger["fields"]["blocker"]["value"] == "no remaining blocker"
    assert ledger["fields"]["blocker"]["source_memory_id"] == "m_new"
    assert ledger["fields"]["target_date"]["conflict_count"] == 1
    assert ledger["enforcement_applied"] is False
    assert ledger["new_llm_calls"] == 0
    assert [row.memory_id for row in ranked] == ["m_old", "m_new"]
    assert ranked[0].metadata["symbolic_state_ledger"] == ledger


def test_v4_symbolic_state_ledger_reuses_clinical_plan_and_typed_frame_slots():
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
    assert ledger["requested_slots"] == ["substance", "reaction"]
    assert ledger["fields"]["substance"]["value"] == "sulfa antibiotic"
    assert ledger["fields"]["reaction"]["value"] == "rash"
    assert ledger["resolved_count"] == 2
    assert ranked[0].metadata["symbolic_state_ledger"] == ledger


def _auth_event(effect: str, principal: str = "nurse_alvarez", resource: str = "clinical records"):
    return {
        "effect": effect,
        "principal_id": principal,
        "resource": resource,
    }


def test_temporal_authorization_graph_applies_allow_then_revoke():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="grant", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="A policy grants access.", score=0.9,
            extra_metadata={"authorization_events": [_auth_event("allow")]},
        ),
        _evidence(
            memory_id="revoke", turn_id="t002", principal_id="nurse_alvarez", role="nurse",
            text="A policy revokes access.", score=0.8,
            extra_metadata={"authorization_events": [_auth_event("revoke")]},
        ),
    ]

    ranked, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
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


def test_temporal_authorization_graph_later_allow_supersedes_older_deny():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="old", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="A policy denies access.", score=0.9,
            extra_metadata={"authorization_events": [_auth_event("deny")]},
        ),
        _evidence(
            memory_id="new", turn_id="t002", principal_id="nurse_alvarez", role="nurse",
            text="A new policy allows access.", score=0.8,
            extra_metadata={"authorization_events": [{**_auth_event("allow"), "event_kind": "supersedes"}]},
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
    )

    certificate = trace["temporal_authorization"]
    assert certificate["decision"] == "allow"
    assert certificate["conflict_count"] == 0
    assert any(edge["edge_type"] == "supersedes" for edge in certificate["graph_edges"])


def test_temporal_authorization_graph_marks_same_time_allow_deny_unknown():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    evidence = [
        _evidence(
            memory_id="allow", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="A policy allows access.", score=0.9,
            extra_metadata={"authorization_events": [_auth_event("allow")]},
        ),
        _evidence(
            memory_id="deny", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
            text="A policy denies access.", score=0.8,
            extra_metadata={"authorization_events": [_auth_event("deny")]},
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
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
            text="A future policy allows access.", score=0.9,
            extra_metadata={"authorization_events": [_auth_event("allow")]},
        ),
    ]

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=evidence, temporal_authorization_enabled=True,
    )

    certificate = trace["temporal_authorization"]
    assert certificate["event_count"] == 0
    assert certificate["ignored_future_event_count"] == 1
    assert certificate["decision"] == "unknown"


def test_temporal_authorization_graph_is_conservative_without_temporal_source():
    instance = replace(_instance(), question="What access does nurse_alvarez have to clinical records?")
    row = _evidence(
        memory_id="untimed", turn_id="t001", principal_id="nurse_alvarez", role="nurse",
        text="A policy allows access.", score=0.9,
        extra_metadata={"authorization_events": [_auth_event("allow")]},
    )
    record = dict(row.metadata["structured_record"])
    record.pop("turn_index")
    record.pop("timestamp")
    row = replace(row, metadata={**row.metadata, "structured_record": record})

    _, trace = build_symbolic_evidence(
        instance=instance, evidence=[row], temporal_authorization_enabled=True,
    )

    certificate = trace["temporal_authorization"]
    assert certificate["event_count"] == 0
    assert certificate["unknown_event_count"] == 1
    assert certificate["decision"] == "unknown"


def test_authorization_assertion_is_structured_without_domain_vocabulary():
    assertions = extract_authorization_assertions(
        "Linda Park may receive scheduling details only; clinical results remain restricted."
    )

    assert assertions
    assert assertions[0]["effect"] == "allow"
    assert assertions[0]["subject"] == "Linda Park"
    assert "scheduling details" in assertions[0]["resource"]
    assert assertions[0]["source"] == "ingestion_text_grammar"
    assert assertions[0]["source_span"][0] == 0


def test_temporal_authorization_consumes_ingestion_annotation_and_resolves_roster_name():
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
    record = dict(row.metadata["structured_record"])
    record["authorization_assertions"] = extract_authorization_assertions(
        record["text"]
    )
    row = replace(row, metadata={**row.metadata, "structured_record": record})

    _, trace = build_symbolic_evidence(
        instance=instance,
        evidence=[row],
        temporal_authorization_enabled=True,
    )

    certificate = trace["temporal_authorization"]
    assert certificate["event_count"] == 1
    assert certificate["decision"] == "allow"
    assert certificate["current_authorization"][0]["principal"] == "family_linda"
    assert certificate["graph_edges"][0]["edge_type"] == "has_role"
    assert any(
        event.get("source") == "ingestion_text_grammar"
        for event in certificate["graph_nodes"]
        if event.get("node_type") == "PolicyEvent"
    )
