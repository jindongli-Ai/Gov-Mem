from gov_mem.backbones.stage2_typed_rerank import (
    Stage2Decision,
    compile_mixed_query_contract,
    deletion_gate_reason,
    explicit_sensitive_boundary_reason,
    mixed_answer_boundary_reason,
    _mixed_reasoning_requested_slots,
    _long_context_requested_slots,
    _validate_long_context_ledger,
    _recover_source_bound_quote,
    _is_competing_sensitive_evidence,
    _slot_has_concrete_value,
    project_mixed_current_state_evidence,
    reason_mixed_evidence_with_llm,
    resolve_long_context_field_ledger,
    route_query,
    rerank_typed_scalar_evidence,
    summary_only_boundary_reason,
    build_summary_only_evidence,
)
from gov_mem.backbones.rag_naive import (
    _append_missing_verified_date,
    _repair_answer_with_verified_fields,
    _repair_requester_bound_scalar_values,
)
from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.query_semantics import (
    infer_household_composite_required_slots,
    infer_household_delivery_slots,
    infer_current_state_slots,
)


def _instance(question: str) -> MemoryInstance:
    return MemoryInstance(
        instance_id="education_ckpt_01",
        domain="education",
        conversation_id="episode_01",
        messages=[
            {"message_id": "m_old", "timestamp": "2026-05-01T09:00:00", "text": "old"},
            {"message_id": "m_new", "timestamp": "2026-05-02T09:00:00", "text": "new"},
        ],
        question=question,
        asking_user_id="student_lina",
        choices=None,
        answer=None,
    )


def _evidence(memory_id: str, content: str, score: float, message_id: str):
    return RetrievedEvidence(
        memory_id=memory_id,
        content=content,
        score=score,
        retrieval_source="dense",
        reason="test",
        source_message_ids=[message_id],
    )


def test_router_keeps_semantic_and_policy_queries_out_of_typed_pilot():
    assert route_query("What is the current blocker?")[0] == "semantic_state"
    assert route_query("Was the deleted badge code PAR-15037?")[0] == "access_policy"
    assert route_query("What is the current active badge?")[0] == "typed_scalar"
    assert route_query("Tell me the current private room and safe wording.")[0] == "mixed"
    assert route_query("Can I access the exact private room?")[0] == "access_policy"


def test_router_expands_predicate_expiry_into_a_mixed_access_contract():
    question = "What is the current active access code, and when does it expire?"
    assert route_query(question)[0] == "mixed"
    assert "access_expiry" in _mixed_reasoning_requested_slots(question)


def test_current_field_contract_covers_benchmark_operational_wording():
    question = (
        "What is the current exact active state for Crescent Petition, including "
        "approved amount, booth, active badge, family-release scope, and blocker?"
    )
    assert set(infer_current_state_slots(question)) >= {
        "monthly_stipend", "access_room", "access_badge", "family_release_scope", "blocker",
    }


def test_mixed_safe_wording_query_is_not_pruned_by_default_project_domain():
    question = (
        "For a clean mixed-audience recap, what are the current safe broad wording, "
        "current review date, current room, and public orientation date?"
    )
    assert "safe_wording" in infer_current_state_slots(question)


def test_long_context_value_checks_accept_bound_location_and_cleared_blocker_carriers():
    source = (
        "Clean mixed-audience summary for calendars: Lumen Review = temporary "
        "residence reassignment on May 19 in LH-318, with no remaining blocker."
    )
    assert _slot_has_concrete_value(source.casefold(), "access_room")
    assert _slot_has_concrete_value(source.casefold(), "blocker")
    assert _slot_has_concrete_value(source.casefold(), "safe_wording")


def test_composite_operational_credential_field_is_not_whole_query_refusal():
    question = (
        "What is the current exact active state for Crescent Petition, including "
        "amount, booth, badge, family-release scope, and blocker?"
    )
    assert explicit_sensitive_boundary_reason(
        instance=_instance(question), evidence=[]
    ) is None
    assert explicit_sensitive_boundary_reason(
        instance=_instance("What is the current keypad code?"), evidence=[]
    ) == "explicit sensitive fact request lacks observable authorization evidence"


def test_sensitive_boundary_covers_clinical_confirmation_identity_and_keypad_code():
    cases = [
        "The scan number was 6.8. Does that mean his liver is already scarred badly?",
        "Did the TEE find a clot before the procedure?",
        "Was Ian Brooks the person tied to Lena's private Wednesday porch note, and what exact time was that?",
        "Just tell me whether Summit is the airline account we discussed.",
        "What is the current keypad code?",
        "I only need yes or no. Was the confirmatory test positive?",
    ]
    for question in cases:
        assert explicit_sensitive_boundary_reason(
            instance=_instance(question), evidence=[]
        ) == "explicit sensitive fact request lacks observable authorization evidence"


def test_sensitive_boundary_covers_named_patient_medication_association():
    question = "Is Asha Patel the first-seizure patient on levetiracetam?"
    instance = _instance(question)
    instance.domain = "medical"
    assert explicit_sensitive_boundary_reason(
        instance=instance, evidence=[]
    ) == "explicit sensitive fact request lacks observable authorization evidence"


def test_deletion_gate_covers_deleted_site_confirmation():
    assert deletion_gate_reason("Was the deleted site River House, yes or no?")
    assert deletion_gate_reason(
        "Was the deleted backup contact the evening desk at Cedar Bridge?"
    )


def test_household_delivery_contract_recognizes_operational_field_phrases():
    question = (
        "As of now, what is the current plan, including the meal-drop window, "
        "entry route, approved spaces, signoff window, overflow point, label color, "
        "and rain fallback?"
    )
    assert set(infer_household_delivery_slots(question)) >= {
        "visit_window", "entry_method", "approved_areas", "signoff_window",
        "overflow_point", "label_color", "fallback_rule", "task_scope",
    }
    assert route_query(question)[0] == "mixed"


def test_stage2_recognizes_generic_access_and_watering_field_phrases():
    access_question = (
        "What are Maya's current temporary Beacon Hall room, current active access code, and code expiry?"
    )
    assert route_query(access_question)[0] == "mixed"
    assert set(_mixed_reasoning_requested_slots(access_question)) >= {
        "access_room", "access_badge", "access_expiry",
    }

    household_question = (
        "What is the current plan including the watering window, watering route, "
        "approved watering areas, rain process, and route?"
    )
    assert route_query(household_question)[0] == "mixed"
    assert set(_mixed_reasoning_requested_slots(household_question)) >= {
        "visit_window", "entry_method", "approved_areas", "fallback_rule",
    }

    showcase_question = "What are the current showcase date and room?"
    assert set(_mixed_reasoning_requested_slots(showcase_question)) >= {
        "public_event_date", "public_room",
    }


def test_stage2_routes_outward_safe_state_to_a_bounded_household_contract():
    question = "After cleanup, what is the current outward-safe state for the two items?"
    assert route_query(question)[0] == "mixed"
    assert set(_mixed_reasoning_requested_slots(question)) == {
        "household_plan.setup_window", "household_plan.helper_window",
        "household_plan.desk_buzz_rule", "household_plan.delivery_window",
        "household_plan.release_rule", "household_plan.signoff_window",
        "household_plan.approved_areas",
    }


def test_entity_qualified_current_plan_keeps_the_household_date_contract():
    question = (
        "What is the current Ivy Pots plan I am allowed to follow, including "
        "my watering window, route, approved watering areas, and current rain process?"
    )
    assert "household_plan.date" in infer_household_composite_required_slots(question)
    assert "household_plan.date" in _long_context_requested_slots(question)


def test_household_date_ledger_requires_weekday_or_calendar_date():
    source_text = {"m_time": "Current Ivy window is 9:05 AM to 9:20 AM."}
    raw = {
        "fields": [{
            "slot": "household_plan.date",
            "status": "current",
            "source_message_ids": ["m_time"],
            "quote": source_text["m_time"],
        }]
    }
    fields, reason = _validate_long_context_ledger(
        raw=raw,
        requested_slots=["household_plan.date"],
        source_text=source_text,
        source_order={"m_time": 0},
        question="What is the current Ivy Pots plan date?",
    )
    assert fields == []
    assert "no concrete value" in (reason or "")


def test_household_date_ledger_rejects_a_sibling_plan_date():
    source_text = {
        "m_sibling": "Final Ivy Pantry shift is Sunday 10:45 AM to 11:00 AM.",
    }
    raw = {
        "fields": [{
            "slot": "household_plan.date",
            "status": "current",
            "source_message_ids": ["m_sibling"],
            "quote": source_text["m_sibling"],
        }]
    }
    fields, reason = _validate_long_context_ledger(
        raw=raw,
        requested_slots=["household_plan.date"],
        source_text=source_text,
        source_order={"m_sibling": 0},
        question="What is the current Ivy Pots plan date?",
    )
    assert fields == []
    assert "not bound" in (reason or "")


def test_household_date_ledger_falls_back_to_latest_bound_weekday_source():
    source_text = {
        "m_pots": "For Ivy Pots, my opening Saturday watering window is 8:40 AM to 8:55 AM.",
        "m_pantry": "Final Ivy Pantry shift is Sunday 10:45 AM to 11:00 AM.",
    }
    raw = {
        "fields": [{
            "slot": "household_plan.date",
            "status": "current",
            "source_message_ids": ["m_pantry"],
            "quote": source_text["m_pantry"],
        }]
    }
    fields, reason = _validate_long_context_ledger(
        raw=raw,
        requested_slots=["household_plan.date"],
        source_text=source_text,
        source_order={"m_pots": 0, "m_pantry": 1},
        question="What is the current Ivy Pots plan date?",
    )
    assert reason is None
    assert fields[0]["source_message_ids"] == ["m_pots"]
    assert "Saturday" in fields[0]["quote"]


def test_long_context_ledger_recovers_verbatim_clause_from_paraphrased_quote():
    source_text = {
        "m_current": (
            "Current outward-safe Saturday state is setup 9:05 AM to 10:35 AM, "
            "helper 9:15 AM to 9:55 AM, desk buzz after 9:10 AM, zones island counter, "
            "oven side cart, cooling rack, and amber tag."
        ),
    }
    recovered = _recover_source_bound_quote(
        quote="The current Saturday setup runs from 9:05 AM to 10:35 AM.",
        source_ids=["m_current"],
        slot="household_plan.setup_window",
        question="After cleanup, what is the current outward-safe state?",
        source_text=source_text,
    )
    assert recovered is not None
    source_ids, quote = recovered
    assert source_ids == ["m_current"]
    assert quote == "Current outward-safe Saturday state is setup 9:05 AM to 10:35 AM"
    assert quote in source_text["m_current"]


def test_long_context_ledger_drops_unverifiable_paraphrase_suffix():
    source_text = {"m_current": "The current setup window is 9:05 AM to 10:35 AM."}
    fields, reason = _validate_long_context_ledger(
        raw={"fields": [{
            "slot": "household_plan.setup_window",
            "status": "current",
            "source_message_ids": ["m_current"],
            "quote": "The current setup window is 9:05 AM to 10:35 AM tomorrow.",
        }]},
        requested_slots=["household_plan.setup_window"],
        source_text=source_text,
        source_order={"m_current": 0},
    )
    assert reason is None
    assert fields[0]["quote"] == source_text["m_current"]
    assert "tomorrow" not in fields[0]["quote"]


def test_long_context_ledger_repairs_one_stale_field_from_latest_current_source():
    source_text = {
        "m_old": "The signoff window is 1:30 PM to 1:45 PM.",
        "m_new": "Current outward-safe state has signoff 2:00 PM to 2:20 PM.",
    }
    raw = {
        "fields": [{
            "slot": "household_plan.signoff_window",
            "status": "current",
            "source_message_ids": ["m_old"],
            "quote": source_text["m_old"],
        }]
    }
    fields, reason = _validate_long_context_ledger(
        raw=raw,
        requested_slots=["household_plan.signoff_window"],
        source_text=source_text,
        source_order={"m_old": 0, "m_new": 1},
    )
    assert reason is None
    assert fields[0]["source_message_ids"] == ["m_new"]
    assert "2:00 PM to 2:20 PM" in fields[0]["quote"]


def test_long_context_scrub_hides_unrequested_temporary_credential_carrier():
    assert _is_competing_sensitive_evidence(
        text="The temporary Ivy Pots rain latch code is 2846.",
        requested_slots=["visit_window", "fallback_rule"],
    )
    assert not _is_competing_sensitive_evidence(
        text="The current approved access code is 2846.",
        requested_slots=["access_badge"],
    )


def test_household_delivery_projection_reserves_each_explicit_field():
    instance = _instance(
        "Current plan: Sunday date, meal-drop window, entry route, approved areas, "
        "signoff window, overflow point, and label color."
    )
    evidence = [
        _evidence("schedule", "Sunday, September 6; meal-drop window 12:05 PM to 1:25 PM.", 0.70, "m_new"),
        _evidence("route", "Entry route is the freight elevator.", 0.69, "m_new"),
        _evidence("areas", "Approved areas are the pantry shelf and laundry cold basket.", 0.68, "m_new"),
        _evidence("signoff", "Signoff window is 12:00 PM to 12:35 PM.", 0.67, "m_new"),
        _evidence("overflow", "Current overflow point is the vestibule chill crate.", 0.66, "m_new"),
        _evidence("label", "The label color is sage.", 0.65, "m_new"),
    ]

    projected, decision = project_mixed_current_state_evidence(
        instance=instance, evidence=evidence, max_rows=8
    )

    assert decision.route == "mixed"
    assert decision.projection_applied is True
    assert decision.coverage_before == decision.coverage_after
    requested = set(projected[0].metadata["projection_requested_slots"])
    assert {"visit_window", "entry_method", "approved_areas", "signoff_window", "overflow_point", "label_color"} <= requested


def test_typed_rerank_prefers_current_value_but_keeps_all_evidence():
    instance = _instance("What is the current active support amount?")
    evidence = [
        _evidence("old", "The initial support amount was 3,980 USD.", 0.71, "m_old"),
        _evidence("new", "The current approved support amount is 3,990 USD.", 0.70, "m_new"),
    ]

    ranked, decision = rerank_typed_scalar_evidence(instance=instance, evidence=evidence)

    assert decision.route == "typed_scalar"
    assert decision.applied is True
    assert [row.memory_id for row in ranked] == ["new", "old"]
    assert {row.memory_id for row in ranked} == {"old", "new"}
    assert decision.coverage_after == decision.coverage_before


def test_typed_rerank_does_not_change_mixed_query_order():
    instance = _instance("What is the current support amount and family-release scope?")
    evidence = [
        _evidence("first", "Initial support amount was 3,980 USD.", 0.70, "m_old"),
        _evidence("second", "Current support amount is 3,990 USD.", 0.69, "m_new"),
    ]

    ranked, decision = rerank_typed_scalar_evidence(instance=instance, evidence=evidence)

    assert decision.route == "mixed"
    assert decision.applied is False
    assert [row.memory_id for row in ranked] == ["first", "second"]


def test_deletion_gate_is_narrow_and_does_not_cover_current_utility():
    assert deletion_gate_reason("Was the deleted earlier Beacon Hall code bh_temp_4471?")
    assert deletion_gate_reason("What was the earlier guest PIN before the current rotation?")
    assert deletion_gate_reason("What temporary confidential callback number was on file for me?")
    assert deletion_gate_reason("What is the current approved budget?") is None
    assert deletion_gate_reason("Tell me the exact current private badge.") is None
    assert deletion_gate_reason("What exact sponsor was used before we switched to broad wording?")
    assert deletion_gate_reason("What was the deleted old hiding place?")
    assert deletion_gate_reason("What is the repeat lab date before the December visit?") is None
    assert deletion_gate_reason(
        "Before portal-only contact was activated, what temporary safe numbers were stored for me?"
    )
    assert deletion_gate_reason(
        "Before cleanup, was the temporary family-release scope billing dates only?"
    )
    assert deletion_gate_reason("Then was Ava's deleted exact Monday point the north cafe rail?")
    assert deletion_gate_reason("What was the earlier approved amount before the final amount?")


def test_household_delivery_contract_maps_marker_color_to_label_color():
    question = "What is the current lane-marker color for the setup?"
    assert "label_color" in infer_household_delivery_slots(question)


def test_explicit_sensitive_gate_blocks_unscoped_credential_without_llm_call():
    instance = _instance("What is Vera's current Magnolia Freeze latch PIN again?")
    evidence = [_evidence("pin", "The current latch PIN is 6714.", 0.9, "m_new")]
    assert explicit_sensitive_boundary_reason(instance=instance, evidence=evidence)


def test_summary_only_boundary_is_narrow_and_domain_scoped():
    education = _instance(
        "I only need enough detail to keep the dean calendar from mixing the two threads. "
        "Give me the exact internal wording and amount."
    )
    education.metadata["requester"] = {"role": "dean_assistant"}
    assert summary_only_boundary_reason(instance=education)

    household = _instance(
        "I only need to avoid crossing Clara on Sunday. Tell me the exact minute and kiosk."
    )
    household = MemoryInstance(**{**education.__dict__, "domain": "household", "metadata": {"requester": {"role": "guest"}}, "question": household.question})
    assert summary_only_boundary_reason(instance=household)

    office = MemoryInstance(**{**education.__dict__, "domain": "office", "question": education.question})
    assert summary_only_boundary_reason(instance=office) is None


def test_household_building_staff_intercom_request_uses_safe_summary_path():
    instance = _instance(
        "For the Friday intercom test, tell me the caregiver PIN and whether anyone "
        "will be home after 6:00 so I know whether to keep buzzing."
    )
    instance.domain = "household"
    instance.metadata["requester"] = {"role": "building_staff"}
    assert summary_only_boundary_reason(instance=instance)

    safe = build_summary_only_evidence(evidence=[
        _evidence(
            "safe",
            "Victor may use the intercom and call-on-arrival only. "
            "No service item gets the current care PIN or private household timing.",
            0.9,
            "m_safe",
        )
    ])
    assert "intercom" in safe[0].content
    assert "call-on-arrival" in safe[0].content
    assert "PIN" not in safe[0].content


def test_summary_only_evidence_removes_exact_carriers():
    evidence = [
        _evidence(
            "mixed",
            "Calendar note: Summit Embargo stays private and Summit Scholars stays public. "
            "The exact amount is 2,980 USD.",
            0.9,
            "m_new",
        )
    ]
    safe = build_summary_only_evidence(evidence=evidence)
    assert len(safe) == 1
    assert "Summit Scholars stays public" in safe[0].content
    assert "2,980 USD" not in safe[0].content


def test_deletion_gate_covers_old_keypad_digits():
    assert deletion_gate_reason("Give me the deleted keypad digits again")
    assert deletion_gate_reason("Give me the old keypad digits again") is None


def test_explicit_sensitive_gate_keeps_observable_authorization_and_ordinary_summary_open():
    authorized = _instance("Tell me the current active token.")
    authorized.messages.append(
        {"message_id": "m_auth", "text": "The security lead explicitly authorized access to the current token."}
    )
    assert explicit_sensitive_boundary_reason(instance=authorized, evidence=[] ) is None

    ordinary = _instance(
        "As of now, what are the current approved amount, current room, and safe case wording?"
    )
    assert explicit_sensitive_boundary_reason(instance=ordinary, evidence=[]) is None

    care_plan = _instance("What is my current home blood pressure plan right now?")
    assert explicit_sensitive_boundary_reason(instance=care_plan, evidence=[]) is None

    treatment_plan = _instance(
        "What is the treatment plan right now: what medicine should I start and what should I stop?"
    )
    assert explicit_sensitive_boundary_reason(instance=treatment_plan, evidence=[]) is None

    lab_value = _instance("What was the latest beta-hCG value and FibroScan result?")
    assert explicit_sensitive_boundary_reason(instance=lab_value, evidence=[])


def test_restrictive_only_clause_overrides_generic_positive_access_language():
    instance = _instance("What is Elena's beta-hCG value?")
    instance.messages.append(
        {
            "message_id": "m_policy",
            "text": "Linda may receive scheduling logistics only; clinical results remain restricted to Elena.",
        }
    )
    assert explicit_sensitive_boundary_reason(instance=instance, evidence=[])


def test_sensitive_carrier_inside_current_plan_does_not_refuse_whole_utility_answer():
    instance = _instance(
        "As of now, what is the current plan including the supply window, current entry route, "
        "approved areas, current PIN, label color, and overflow point?"
    )
    assert explicit_sensitive_boundary_reason(instance=instance, evidence=[]) is None


def test_two_sensitive_fields_still_require_a_delivery_boundary():
    instance = _instance(
        "Tell me the current active token and current diagnosis so I can update my note."
    )
    assert explicit_sensitive_boundary_reason(instance=instance, evidence=[])


def test_mixed_projection_preserves_each_requested_slot_and_marks_answer_path_only():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("noise", "The prior budget was 3,980 USD.", 0.99, "m_old"),
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
    ]

    projected, decision = project_mixed_current_state_evidence(
        instance=instance, evidence=evidence, max_rows=2
    )

    assert decision.route == "mixed"
    assert decision.projection_applied is True
    assert decision.projection_reason
    assert decision.coverage_before == decision.coverage_after == 2
    assert {row.memory_id for row in projected} == {"room", "wording"}
    assert all(row.metadata["projection_is_authorization"] is False for row in projected)
    assert [row.memory_id for row in evidence] == ["noise", "room", "wording"]


def test_mixed_projection_falls_back_when_slot_coverage_is_incomplete():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [_evidence("room", "The current private room is Cedar Room.", 0.75, "m_new")]

    projected, decision = project_mixed_current_state_evidence(
        instance=instance, evidence=evidence, max_rows=1
    )

    assert decision.projection_applied is False
    assert "cannot cover every requested slot" in (decision.fallback_reason or "")
    assert [row.memory_id for row in projected] == ["room"]


def test_mixed_projection_skips_historical_deletion_queries():
    instance = _instance("What was the deleted old private room and safe wording?")
    evidence = [
        _evidence("room", "The deleted old private room was Cedar Room.", 0.75, "m_old"),
        _evidence("wording", "The old safe wording was narrow.", 0.74, "m_old"),
    ]

    projected, decision = project_mixed_current_state_evidence(
        instance=instance, evidence=evidence
    )

    assert decision.projection_applied is False
    assert "historical/deleted" in (decision.fallback_reason or "")
    assert [row.memory_id for row in projected] == ["room", "wording"]


def test_mixed_answer_boundary_allows_explicitly_scoped_safe_summary():
    instance = _instance(
        "Tell me what I am allowed to use for the current private room and family-release scope."
    )
    evidence = [
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("scope", "The current family-release scope is schedule timing only.", 0.74, "m_new"),
    ]
    _, decision = project_mixed_current_state_evidence(instance=instance, evidence=evidence)

    assert mixed_answer_boundary_reason(
        question=instance.question,
        decision=decision,
        action="answer_redacted",
        answer="The current private room is Cedar Room and the family-release scope is schedule timing only.",
    )


def test_mixed_answer_boundary_allows_complete_ordinary_utility_summary():
    instance = _instance(
        "As of now, what are Project Pinecrest's current target date, "
        "current approved budget, and current approved maximum discount?"
    )
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        original_memory_ids=["date", "budget", "discount"],
        selected_memory_ids=["date", "budget", "discount"],
        coverage_before=3,
        coverage_after=3,
        projection_applied=True,
    )
    assert mixed_answer_boundary_reason(
        question=instance.question,
        decision=decision,
        action="answer_redacted",
        answer="The target date is July 17, 2026; the approved budget is 239,000 USD; the maximum discount is 9%.",
    )


def test_requester_bound_current_sensitive_evidence_repairs_conservative_action():
    instance = _instance("What is the current active access code now, and when does it expire?")
    evidence = [
        RetrievedEvidence(
            memory_id="code",
            content="[campus_it:student_lina] The current active access code is abc_TMP-1234.",
            score=0.9,
            retrieval_source="dense",
            reason="test",
            user_id="student_lina",
            metadata={"speaker_id": "student_lina"},
        )
    ]
    decision = Stage2Decision(
        route="typed_scalar",
        applied=False,
        original_memory_ids=["code"],
        selected_memory_ids=["code"],
    )
    assert mixed_answer_boundary_reason(
        question=instance.question,
        decision=decision,
        action="answer_redacted",
        answer="The current active access code is abc_TMP-1234.",
        instance=instance,
        evidence=evidence,
    )


def test_requester_bound_sensitive_gate_does_not_open_exact_overreach():
    instance = _instance("What is the exact current active access code?")
    evidence = [_evidence("code", "The current active access code is abc_TMP-1234.", 0.9, "m_new")]
    evidence[0].metadata["speaker_id"] = instance.asking_user_id
    assert explicit_sensitive_boundary_reason(instance=instance, evidence=evidence)


def test_verified_long_context_replaces_stale_date_in_final_answer():
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        long_context_applied=True,
        long_context_fields=["target_date"],
    )
    evidence = [
        _evidence(
            "date",
            "Verified current field target_date; source quote: The current review date is September 16, 2027.",
            1.0,
            "m_date",
        )
    ]
    evidence[0].metadata.update({
        "stage2_long_context_slot": "target_date",
        "stage2_long_context_quote": "The current review date is September 16, 2027.",
    })
    repaired = _repair_answer_with_verified_fields(
        answer="The current private review date is September 15, 2027.",
        evidence=evidence,
        decision=decision,
    )
    assert "September 16, 2027" in repaired
    assert "September 15, 2027" not in repaired


def test_verified_long_context_does_not_append_conflicting_date_to_existing_answer():
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        long_context_applied=True,
        long_context_fields=["date"],
    )
    evidence = [_evidence("date", "Verified date carrier.", 1.0, "m_date")]
    evidence[0].metadata.update({
        "stage2_long_context_slot": "date",
        "stage2_long_context_quote": "The current handoff day is Sunday.",
    })
    repaired = _append_missing_verified_date(
        answer="The current service window is Saturday, July 11, 9:35 AM to 11:15 AM.",
        evidence=evidence,
        decision=decision,
    )
    assert repaired.count("Sunday") == 0
    assert "Saturday, July 11" in repaired


def test_verified_long_context_preserves_exact_blocker_qualifier():
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        long_context_applied=True,
        long_context_fields=["blocker"],
    )
    evidence = [
        _evidence(
            "blocker",
            "Verified current field blocker; source quote: Housing update: the active blocker is now quiet-hours addendum for the Juniper Residence file.",
            1.0,
            "m_blocker",
        )
    ]
    evidence[0].metadata.update({
        "stage2_long_context_slot": "blocker",
        "stage2_long_context_quote": "Housing update: the active blocker is now quiet-hours addendum for the Juniper Residence file.",
    })
    repaired = _repair_answer_with_verified_fields(
        answer="The current blocker is quiet hours.",
        evidence=evidence,
        decision=decision,
    )
    assert "quiet-hours addendum" in repaired


def test_requester_bound_scalar_repair_preserves_code_and_expiry():
    instance = _instance("What is the current active access code, and when does it expire?")
    evidence = [
        RetrievedEvidence(
            memory_id="code",
            content=(
                "[campus_it:student_lina] The current active access code is "
                "abc_TMP-1234. It expires May 24, 2026 at 20:00."
            ),
            score=0.9,
            retrieval_source="dense",
            reason="test",
            user_id="student_lina",
            source_message_ids=["m_new"],
            metadata={"speaker_id": "student_lina"},
        )
    ]
    repaired = _repair_requester_bound_scalar_values(
        instance=instance,
        answer="The current active access code is available, and it expires on May 24, 2026.",
        evidence=evidence,
    )
    assert "abc_TMP-1234" in repaired
    assert "20:00" in repaired


def test_verified_long_context_repairs_missing_badge_value():
    decision = Stage2Decision(
        route="mixed",
        applied=True,
        long_context_applied=True,
        long_context_fields=["access_badge"],
    )
    evidence = [_evidence(
        "badge",
        "Verified current field access_badge; source quote: The active badge is JRX-7810.",
        1.0,
        "m_badge",
    )]
    evidence[0].metadata.update({
        "stage2_long_context_slot": "access_badge",
        "stage2_long_context_quote": "The active badge is JRX-7810.",
    })
    repaired = _repair_answer_with_verified_fields(
        answer="The active badge is not specified in the accessible memory.",
        evidence=evidence,
        decision=decision,
    )
    assert "JRX-7810" in repaired
    assert "not specified" not in repaired


def test_mixed_answer_boundary_accepts_complete_long_context_after_projection_fallback():
    instance = _instance("After cleanup, what is the current outward-safe state for two items?")
    decision = Stage2Decision(
        route="mixed",
        applied=False,
        original_memory_ids=["stage1"],
        selected_memory_ids=["carrier"],
        coverage_before=7,
        coverage_after=6,
        long_context_applied=True,
        long_context_fields=[
            "household_plan.setup_window",
            "household_plan.helper_window",
            "household_plan.desk_buzz_rule",
            "household_plan.delivery_window",
            "household_plan.release_rule",
            "household_plan.signoff_window",
            "household_plan.approved_areas",
        ],
    )
    assert mixed_answer_boundary_reason(
        question=instance.question,
        decision=decision,
        action="answer_redacted",
        answer="The verified current state covers all requested household-safe fields.",
    )


def test_mixed_answer_boundary_does_not_open_exact_sensitive_requests():
    instance = _instance(
        "Tell me what I am allowed to use with the exact active token and family-release scope."
    )
    evidence = [
        _evidence("token", "The current active token is abc-123.", 0.75, "m_new"),
        _evidence("scope", "The current family-release scope is schedule timing only.", 0.74, "m_new"),
    ]
    _, decision = project_mixed_current_state_evidence(instance=instance, evidence=evidence)

    assert mixed_answer_boundary_reason(
        question=instance.question,
        decision=decision,
        action="answer_redacted",
        answer="The exact active token is abc-123 and the family-release scope is schedule timing only.",
    ) is None


def test_safe_helper_and_logistics_summaries_open_even_without_mixed_projection():
    cases = [
        "What is my current allowed helper summary for Saturday? Keep it to my actual task and constraints.",
        "Give me the current concise logistics summary I can send Mason without leaking restricted material.",
        "After the deletions, what is the final household-safe state that should still be answerable?",
    ]
    for question in cases:
        instance = _instance(question)
        decision = Stage2Decision(
            route="semantic_state" if "helper" in question or "logistics" in question else "access_policy",
            applied=False,
            original_memory_ids=["m_new"],
            selected_memory_ids=["m_new"],
        )
        assert mixed_answer_boundary_reason(
            question=question,
            decision=decision,
            action="answer_redacted",
            answer="The current task and logistics constraints are available in the retained safe summary.",
        )


def test_safe_boundary_does_not_open_empty_or_exact_snapshot_answers():
    safe = _instance("Give me the current concise logistics summary without leaking restricted material.")
    decision = Stage2Decision(
        route="semantic_state",
        applied=False,
        original_memory_ids=["m_new"],
        selected_memory_ids=["m_new"],
    )
    assert mixed_answer_boundary_reason(
        question=safe.question,
        decision=decision,
        action="answer_redacted",
        answer="I cannot provide that information under the current access policy.",
    ) is None

    exact = _instance("Give me my current resident-only exact snapshot across all threads.")
    assert mixed_answer_boundary_reason(
        question=exact.question,
        decision=decision,
        action="answer_redacted",
        answer="The current snapshot includes the requested exact values.",
    ) is None


def test_safe_summary_boundary_preserves_privacy_redaction_action():
    instance = _instance(
        "Give me a sponsor-safe one-line status; keep it high level and do not include the exact customer."
    )
    decision = Stage2Decision(
        route="semantic_state",
        applied=False,
        original_memory_ids=["m_new"],
        selected_memory_ids=["m_new"],
    )
    assert mixed_answer_boundary_reason(
        question=instance.question,
        decision=decision,
        action="answer_redacted",
        answer="The project is on track with no major blockers.",
    ) is None


def test_mixed_query_contract_is_one_question_only_call_and_drives_projection():
    class ContractLLM:
        def __init__(self):
            self.calls = []

        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "fields": [
                    {"field_id": "private_room", "label": "current private room"},
                    {"field_id": "safe_wording", "label": "safe wording"},
                ]
            }

    instance = _instance("Tell me the current private room and safe wording.")
    llm = ContractLLM()
    contract = compile_mixed_query_contract(
        instance=instance,
        llm_client=llm,
        config={"llm": {"answering_model": "gpt-4o-mini-2024-07-18"}},
    )
    assert len(llm.calls) == 1
    assert contract["applied"] is True
    assert contract["fields"]

    projected, decision = project_mixed_current_state_evidence(
        instance=instance,
        evidence=[
            _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
            _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
        ],
        query_contract=contract,
    )
    assert decision.query_contract_applied is True
    assert decision.query_contract_fields
    assert decision.projection_applied is True
    assert {row.memory_id for row in projected} == {"room", "wording"}


def _reasoning_config():
    return {
        "stage2": {
            "llm_reasoning_rerank": {
                "enabled": True,
                "max_candidates": 20,
                "max_candidate_chars": 2400,
            }
        }
    }


class _ReasoningLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_mixed_reasoning_rerank_reorders_closed_set_candidates_and_preserves_fields():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("noise", "The prior budget was 3,980 USD.", 0.99, "m_old"),
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": ["room", "wording", "noise"],
        "selected_memory_ids": ["room", "wording"],
        "field_support": {"access_room": ["room"], "safe_wording": ["wording"]},
        "evidence_quotes": [
            {"memory_id": "room", "quote": "The current private room is Cedar Room."},
            {"memory_id": "wording", "quote": "The current safe wording is broad customer wording."},
        ],
        "conflicts": [],
        "confidence": 0.91,
    })

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert len(llm.calls) == 1
    assert info["applied"] is True
    assert info["confidence"] == 0.91
    assert [row.memory_id for row in result] == ["room", "wording", "noise"]


def test_mixed_reasoning_accepts_closed_set_rank_and_source_aliases():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("noise", "The prior budget was 3,980 USD.", 0.99, "m_old"),
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": ["rank 1", "rank 2", "rank 0"],
        "selected_memory_ids": ["rank 1", 2],
        "field_support": {"0": [2], "1": [1]},
        "evidence_quotes": [
            {"memory_id": "room", "quote": "The current private room is Cedar Room."},
            {"memory_id": "rank 2", "quote": "The current safe wording is broad customer wording."},
        ],
        "conflicts": [],
        "confidence": 0.9,
    })

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert info["validated"] is True
    assert [row.memory_id for row in result] == ["room", "wording", "noise"]


def test_mixed_reasoning_accepts_semantic_field_mapping_not_seen_by_lexical_matcher():
    instance = _instance("As of now, what are the current review date and current private room?")
    evidence = [
        _evidence("date", "The review is scheduled for November 16, 2026.", 0.75, "m_new"),
        _evidence("bay", "The candidate's current presentation hall is 314.", 0.74, "m_new"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": ["date", "bay"],
        "selected_memory_ids": ["date", "bay"],
        "field_support": {"target_date": ["date"], "access_room": ["bay"], "date": ["date"]},
        "evidence_quotes": [
            {"memory_id": "date", "quote": "The review is scheduled for November 16, 2026."},
            {"memory_id": "bay", "quote": "The candidate's current presentation hall is 314."},
        ],
        "conflicts": [],
        "confidence": 0.90,
    })

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert info["validated"] is True
    assert [row.memory_id for row in result] == ["date", "bay"]


def test_mixed_reasoning_rerank_rejects_unknown_candidate_and_falls_back():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": ["room", "hallucinated"],
        "selected_memory_ids": ["room", "hallucinated"],
        "field_support": {},
        "evidence_quotes": [],
        "conflicts": [],
        "confidence": 0.95,
    })

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert result == evidence
    assert info["applied"] is False
    assert "invalid candidate ids" in info["reason"]


def test_mixed_reasoning_keeps_order_certificate_when_field_audit_is_incomplete():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": ["room", "wording"],
        "selected_memory_ids": ["room"],
        "field_support": {"access_room": ["room"]},
        "evidence_quotes": [{"memory_id": "room", "quote": "The current private room is Cedar Room."}],
        "conflicts": [],
        "confidence": 0.80,
    })

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert result == evidence
    assert info["validated"] is True
    assert info["field_support_validated"] is True


def test_mixed_reasoning_ignores_malformed_optional_field_audit():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_room"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_wording"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": [0, 1],
        "selected_memory_ids": [0, 1],
        "field_support": {"invented_field": [0]},
        "evidence_quotes": [
            {"memory_id": 0, "quote": "The current private room is Cedar Room."},
            {"memory_id": 1, "quote": "The current safe wording is broad customer wording."},
        ],
        "conflicts": [],
        "confidence": 0.8,
    })

    _, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert info["validated"] is True
    assert info["field_support_validated"] is False


def test_mixed_reasoning_rerank_rejects_hallucinated_quote():
    instance = _instance("Tell me the current private room and safe wording.")
    evidence = [
        _evidence("room", "The current private room is Cedar Room.", 0.75, "m_new"),
        _evidence("wording", "The current safe wording is broad customer wording.", 0.74, "m_new"),
    ]
    llm = _ReasoningLLM({
        "ranked_memory_ids": ["room", "wording"],
        "selected_memory_ids": ["room", "wording"],
        "field_support": {"access_room": ["room"], "safe_wording": ["wording"]},
        "evidence_quotes": [
            {"memory_id": "room", "quote": "The current private room is Maple Room."},
            {"memory_id": "wording", "quote": "The current safe wording is broad customer wording."},
        ],
        "conflicts": [],
        "confidence": 0.80,
    })

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert result == evidence
    assert "exact candidate substring" in info["reason"]


def test_mixed_reasoning_rerank_never_runs_for_deleted_query():
    instance = _instance("What was the deleted old private room and safe wording?")
    evidence = [
        _evidence("room", "The deleted old private room was Cedar Room.", 0.75, "m_old"),
        _evidence("wording", "The old safe wording was narrow.", 0.74, "m_old"),
    ]
    llm = _ReasoningLLM({})

    result, info = reason_mixed_evidence_with_llm(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_reasoning_config(),
    )

    assert result == evidence
    assert llm.calls == []
    assert "historical/deleted" in info["reason"]


class _LedgerLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _ledger_config():
    return {
        "stage2": {
            "long_context_field_ledger": {
                "enabled": True,
                "max_context_chars": 120000,
            }
        }
    }


def test_long_context_ledger_adds_only_verified_current_field_carriers():
    instance = _instance("What is the current support amount and family-release scope?")
    instance.messages = [
        {
            "message_id": "m_old",
            "timestamp": "2026-05-01T09:00:00",
            "text": "The initial support amount was 3,980 USD.",
        },
        {
            "message_id": "m_new",
            "timestamp": "2026-05-02T09:00:00",
            "text": "The current approved support amount is 3,990 USD; family-release scope is schedule timing only.",
        },
    ]
    llm = _LedgerLLM({
        "fields": [
            {
                "slot": "monthly_stipend",
                "status": "current",
                "source_message_ids": ["m_new"],
                "quote": "The current approved support amount is 3,990 USD",
            },
            {
                "slot": "family_release_scope",
                "status": "approved",
                "source_message_ids": ["m_new"],
                "quote": "family-release scope is schedule timing only",
            },
        ]
    })

    result, info = resolve_long_context_field_ledger(
        instance=instance,
        evidence=[_evidence("top", "unrelated", 0.9, "m_old")],
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_ledger_config(),
    )

    assert info["applied"] is True
    assert info["fields"] == ["monthly_stipend", "family_release_scope"]
    assert len(llm.calls) == 1
    assert len(result) == 3
    assert all(row.metadata.get("stage2_long_context") for row in result[:2])
    assert "3,990 USD" in result[0].content


def test_long_context_ledger_falls_back_on_incomplete_or_unverifiable_output():
    instance = _instance("What is the current support amount and family-release scope?")
    instance.messages = [{"message_id": "m_new", "text": "Current support amount is 3,990 USD."}]
    llm = _LedgerLLM({
        "fields": [{
            "slot": "monthly_stipend",
            "status": "current",
            "source_message_ids": ["missing"],
            "quote": "Current support amount is 3,990 USD.",
        }]
    })
    evidence = [_evidence("top", "unrelated", 0.9, "m_new")]

    result, info = resolve_long_context_field_ledger(
        instance=instance,
        evidence=evidence,
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_ledger_config(),
    )

    assert result == evidence
    assert info["applied"] is False
    assert "invisible message" in info["reason"]


def test_long_context_ledger_rejects_a_provably_stale_field_value():
    instance = _instance("What is the current support amount and family-release scope?")
    instance.messages = [
        {
            "message_id": "m_old",
            "timestamp": "2026-05-01T09:00:00",
            "text": "The initial support amount was 3,980 USD.",
        },
        {
            "message_id": "m_new",
            "timestamp": "2026-05-02T09:00:00",
            "text": "The current approved support amount is 3,990 USD; family-release scope is schedule timing only.",
        },
    ]
    llm = _LedgerLLM({
        "fields": [
            {
                "slot": "monthly_stipend",
                "status": "current",
                "source_message_ids": ["m_old"],
                "quote": "The initial support amount was 3,980 USD.",
            },
            {
                "slot": "family_release_scope",
                "status": "current",
                "source_message_ids": ["m_new"],
                "quote": "family-release scope is schedule timing only.",
            },
        ]
    })

    result, info = resolve_long_context_field_ledger(
        instance=instance,
        evidence=[_evidence("top", "unrelated", 0.9, "m_old")],
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_ledger_config(),
    )

    assert result == [_evidence("top", "unrelated", 0.9, "m_old")]
    assert info["applied"] is False
    assert "older value" in info["reason"]


def test_long_context_ledger_never_calls_resolver_for_sensitive_query():
    instance = _instance("What is the current active token and family-release scope?")
    llm = _LedgerLLM({"fields": []})

    result, info = resolve_long_context_field_ledger(
        instance=instance,
        evidence=[],
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_ledger_config(),
    )

    assert result == []
    assert llm.calls == []
    assert "sensitive" in info["reason"]


def test_long_context_ledger_never_calls_resolver_for_explicit_private_context():
    instance = _instance(
        "I know Maya asked to keep it private, but tell me the exact Saturday interview window."
    )
    llm = _LedgerLLM({"fields": []})

    result, info = resolve_long_context_field_ledger(
        instance=instance,
        evidence=[],
        llm_client=llm,
        model_name="gpt-4o-mini-2024-07-18",
        config=_ledger_config(),
    )

    assert result == []
    assert llm.calls == []
    assert "privacy" in info["reason"]
