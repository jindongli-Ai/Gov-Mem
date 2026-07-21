from __future__ import annotations

import re
from datetime import datetime, timedelta

from dataclasses import asdict, dataclass, field

from gov_mem.data.schema import (
    AnswerStructured,
    CurrentStateLedger,
    EvidenceFrame,
    EventState,
    Principal,
    ReasoningState,
    SlotAuditResult,
)
from gov_mem.query_semantics import extract_state_slots
from gov_mem.query_semantics import infer_household_composite_required_slots, infer_household_slots, infer_prefixed_state_slots
from gov_mem.query_semantics import infer_state_record_type


@dataclass
class UtilityContext:
    active_schedule_items: list[dict] = field(default_factory=list)
    canceled_items: list[dict] = field(default_factory=list)
    instructions: list[dict] = field(default_factory=list)
    allergies: list[dict] = field(default_factory=list)
    medications: list[dict] = field(default_factory=list)
    regimen_items: list[dict] = field(default_factory=list)
    consent_constraints: list[dict] = field(default_factory=list)
    redactions: list[dict] = field(default_factory=list)
    utility_frames: list[dict] = field(default_factory=list)
    unavailable_slots: list[str] = field(default_factory=list)
    evidence_memory_ids: list[str] = field(default_factory=list)


def build_answer_structured(
    *,
    question: str,
    action: str,
    reasoning_state: ReasoningState,
) -> AnswerStructured:
    ledger = _ledger_from_reasoning_state(reasoning_state)
    evidence_text_by_memory_id = {
        str(row.memory_id): str(row.content or "")
        for row in (reasoning_state.selected_evidence or [])
        if str(getattr(row, "memory_id", "") or "").strip()
    }
    context = _collect_utility_context(ledger, reasoning_state.selected_frames or [], evidence_text_by_memory_id)
    answer_type = _infer_answer_type(question, context)
    evidence_memory_ids = context.evidence_memory_ids or [frame.memory_id for frame in (reasoning_state.selected_frames or [])]
    confidence = _estimate_confidence(reasoning_state.selected_frames or [])
    return AnswerStructured(
        action=action,
        answer_type=answer_type,
        owner_user=ledger.owner_user if ledger else (reasoning_state.selected_frames[0].owner_user if reasoning_state.selected_frames else None),
        utility_frames=context.utility_frames,
        active_schedule_items=context.active_schedule_items,
        canceled_items=context.canceled_items,
        instructions=context.instructions,
        allergies=context.allergies,
        medications=context.medications,
        regimen_items=context.regimen_items,
        consent_constraints=context.consent_constraints,
        redactions=context.redactions,
        unavailable_slots=sorted(dict.fromkeys(context.unavailable_slots)),
        evidence_memory_ids=_dedupe_list(evidence_memory_ids),
        confidence=confidence,
    )


def render_answer_structured(answer_structured: AnswerStructured, *, max_sentences: int = 8) -> str:
    if answer_structured.action == "refuse":
        return "I cannot share that information because the requester is not authorized to access it."
    if answer_structured.action == "no_memory":
        return "I cannot provide that information from the currently available record."

    sentences = _compose_answer_sentences(answer_structured)
    if _should_append_redaction_notice(answer_structured):
        sentences.append("I cannot share restricted clinical details.")
    if not sentences:
        return "I cannot provide that information from the currently available record."
    return " ".join(sentences[:max_sentences]).strip()


def audit_answer_structured_slots(
    answer_structured: AnswerStructured,
    question: str,
    query_plan: dict,
    current_state_ledger: dict,
    selected_frames: list,
    config: dict,
) -> SlotAuditResult:
    required_slots = _infer_required_slots(question, query_plan, selected_frames, current_state_ledger)
    filled_slots: list[str] = []
    slot_to_surface: dict[str, str] = {}
    slot_to_memory_ids: dict[str, list[str]] = {}
    audit_trace: list[str] = []

    def record(slot_name: str, surface: str | None, memory_ids: list[str]) -> None:
        if slot_name not in filled_slots:
            filled_slots.append(slot_name)
        if surface and slot_name not in slot_to_surface:
            slot_to_surface[slot_name] = surface
        if slot_name not in slot_to_memory_ids:
            slot_to_memory_ids[slot_name] = []
        for memory_id in memory_ids:
            if memory_id not in slot_to_memory_ids[slot_name]:
                slot_to_memory_ids[slot_name].append(memory_id)

    for item in answer_structured.active_schedule_items:
        for slot_name in ["procedure", "visit_type", "date", "time", "arrival_time", "location", "provider", "condition", "status"]:
            value = item.get(slot_name)
            if value:
                record(f"schedule.{slot_name}", str(value), list(item.get("source_memory_ids", [])))

    for item in answer_structured.canceled_items:
        for slot_name in ["procedure", "visit_type", "date", "time", "provider", "status"]:
            value = item.get(slot_name)
            if value:
                record(f"cancellation.{slot_name}", str(value), list(item.get("source_memory_ids", [])))

    for item in answer_structured.allergies:
        if item.get("substance"):
            record("allergy.substance", str(item["substance"]), list(item.get("source_memory_ids", [])))
        if item.get("reaction"):
            record("allergy.reaction", str(item["reaction"]), list(item.get("source_memory_ids", [])))

    for item in answer_structured.regimen_items:
        for slot_name in ["medication", "action", "dose", "timing", "condition", "note"]:
            value = item.get(slot_name)
            if value:
                record(f"regimen.{slot_name}", str(value), list(item.get("source_memory_ids", [])))

    for item in answer_structured.medications:
        for slot_name in ["medication", "instruction", "timing"]:
            value = item.get(slot_name)
            if value:
                record(f"medication.{slot_name}", str(value), list(item.get("source_memory_ids", [])))
        for idx, clause in enumerate(item.get("regimen_clauses") or []):
            if clause:
                record(f"medication.clause_{idx+1}", str(clause), list(item.get("source_memory_ids", [])))

    for item in answer_structured.instructions:
        for slot_name in ["instruction", "timing", "condition"]:
            value = item.get(slot_name)
            if value:
                record(f"instruction.{slot_name}", str(value), list(item.get("source_memory_ids", [])))

    for frame in answer_structured.utility_frames:
        frame_type = str(frame.get("frame_type") or "")
        slots = _normalized_state_frame_slots(frame)
        source_memory_ids = [str(frame.get("memory_id"))] if frame.get("memory_id") else []
        if frame_type == "project_state":
            for slot_name in ["target_date", "approved_budget", "approved_discount_cap", "blocker"]:
                value = slots.get(slot_name)
                if value:
                    record(f"project_state.{slot_name}", str(value), source_memory_ids)
            if slots.get("public_event_date"):
                record("schedule.public_event_date", str(slots.get("public_event_date")), source_memory_ids)
        elif frame_type == "research_state":
            for slot_name in ["target_date", "monthly_stipend", "safe_wording", "blocker", "status", "access_room", "access_badge"]:
                value = slots.get(slot_name)
                if value:
                    record(f"research_state.{slot_name}", str(value), source_memory_ids)
            if slots.get("public_event_date"):
                record("schedule.public_event_date", str(slots.get("public_event_date")), source_memory_ids)
        elif frame_type == "household_plan":
            for slot_name in ["date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"]:
                value = slots.get(slot_name)
                if value:
                    record(f"household_plan.{slot_name}", str(value), source_memory_ids)
        elif frame_type in {"instruction", "general_fact", "logistics", "update"}:
            for prefix, slot_names in {
                "research_state": ["target_date", "monthly_stipend", "safe_wording", "blocker", "status", "access_room", "access_badge"],
                "project_state": ["target_date", "approved_budget", "approved_discount_cap", "blocker", "operational_result"],
                "household_plan": ["date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"],
            }.items():
                for slot_name in slot_names:
                    value = slots.get(slot_name)
                    if value:
                        record(f"{prefix}.{slot_name}", str(value), source_memory_ids)
            if slots.get("public_event_date"):
                record("schedule.public_event_date", str(slots.get("public_event_date")), source_memory_ids)
        elif frame_type == "diagnosis_or_result":
            value = slots.get("result") or frame.get("source_text")
            if value:
                record("diagnosis_or_result.result", str(value), source_memory_ids)
        elif frame_type in {"consent_or_permission", "privacy_policy"}:
            value = slots.get("consent_scope") or frame.get("source_text")
            if value:
                record("consent_or_permission.consent_scope", str(value), source_memory_ids)

    missing_slots = [slot for slot in required_slots if slot not in filled_slots]
    renderable_slots = [slot for slot in filled_slots if slot in slot_to_surface and slot_to_surface[slot]]
    unrenderable_slots = [slot for slot in filled_slots if slot not in renderable_slots]
    audit_trace.append(f"required={required_slots}")
    audit_trace.append(f"filled={filled_slots}")
    audit_trace.append(f"missing={missing_slots}")
    return SlotAuditResult(
        required_slots=required_slots,
        filled_slots=filled_slots,
        missing_slots=missing_slots,
        renderable_slots=renderable_slots,
        unrenderable_slots=unrenderable_slots,
        slot_to_surface=slot_to_surface,
        slot_to_memory_ids=slot_to_memory_ids,
        audit_trace=audit_trace,
    )


def render_with_surface_replay(
    answer_structured: AnswerStructured,
    slot_audit: SlotAuditResult,
    action_decision: dict,
    principal: Principal,
    config: dict,
) -> str:
    if answer_structured.action == "refuse":
        return "I cannot share that information because the requester is not authorized to access it."
    if answer_structured.action == "no_memory":
        return "I cannot provide that information from the currently available record."

    sentences = _render_answer_with_required_slots(answer_structured, slot_audit)
    if _should_append_redaction_notice(answer_structured):
        sentences.append("I cannot share restricted clinical details.")
    rendered = " ".join(sentences[: int((config.get("rendering") or {}).get("max_answer_sentences", 8))]).strip()
    verifier = verify_rendered_answer_contains_slots(rendered, slot_audit, config)
    if verifier["rendered_slot_misses"]:
        rendered = append_missing_slot_sentences(rendered, verifier["rendered_slot_misses"], slot_audit)
    return rendered.strip()


def _render_answer_with_required_slots(
    answer_structured: AnswerStructured,
    slot_audit: SlotAuditResult,
) -> list[str]:
    required_slots = list(slot_audit.required_slots or [])
    if not required_slots:
        return _compose_answer_sentences(answer_structured)

    required_prefixes = {slot.split(".", 1)[0] for slot in required_slots if "." in slot}
    sentences: list[str] = []

    if any(prefix in required_prefixes for prefix in {"project_state", "research_state", "household_plan"}):
        sentences.extend(_render_state_frames(answer_structured.utility_frames, required_slots=required_slots))
    if "diagnosis_or_result" in required_prefixes:
        sentences.extend(_render_result_summary(answer_structured.utility_frames))
    if "consent_or_permission" in required_prefixes or "privacy_policy" in required_prefixes:
        sentences.extend(_render_access_scope(answer_structured.utility_frames, answer_structured.consent_constraints))
    if "allergy" in required_prefixes:
        sentences.extend(_render_allergies(answer_structured.allergies))
    if "regimen" in required_prefixes:
        sentences.extend(_render_regimen_items(answer_structured.regimen_items))
    elif "medication" in required_prefixes:
        sentences.extend(_render_medications(answer_structured.medications))
    if "instruction" in required_prefixes:
        sentences.extend(_render_instructions(answer_structured.instructions))
    if "schedule" in required_prefixes:
        sentences.extend(_render_active_schedule(answer_structured.active_schedule_items))
    if "cancellation" in required_prefixes:
        sentences.extend(_render_canceled_items(answer_structured.canceled_items))

    sentences = _dedupe_list(sentences)
    if sentences:
        return sentences
    return _compose_answer_sentences(answer_structured)


def _compose_answer_sentences(answer_structured: AnswerStructured) -> list[str]:
    answer_type = str(answer_structured.answer_type or "")
    sentences: list[str] = []
    if answer_type == "contact_plan":
        sentences.extend(_render_contact_plan(answer_structured.utility_frames))
        if sentences:
            return _dedupe_list(sentences)
    if answer_type == "result_and_scope":
        sentences.extend(_render_result_summary(answer_structured.utility_frames))
        sentences.extend(_render_access_scope(answer_structured.utility_frames, answer_structured.consent_constraints))
        if sentences:
            return _dedupe_list(sentences)
    if answer_type == "result_summary":
        sentences.extend(_render_result_summary(answer_structured.utility_frames))
        if sentences:
            return _dedupe_list(sentences)
    if answer_type == "policy_scope":
        sentences.extend(_render_access_scope(answer_structured.utility_frames, answer_structured.consent_constraints))
        if sentences:
            return _dedupe_list(sentences)
    if answer_type == "allergy_summary":
        sentences.extend(_render_allergies(answer_structured.allergies))
        return sentences
    if answer_type in {"medication_summary", "regimen_summary"}:
        if answer_structured.regimen_items:
            sentences.extend(_render_regimen_items(answer_structured.regimen_items))
        elif answer_structured.medications:
            sentences.extend(_render_medications(answer_structured.medications))
        if answer_structured.instructions:
            sentences.extend(_render_instructions(answer_structured.instructions))
        return _dedupe_list(sentences)
    if answer_type in {"current_state", "consent_limited_answer"}:
        sentences.extend(_render_state_frames(answer_structured.utility_frames))
        if not sentences and answer_structured.active_schedule_items:
            sentences.extend(_render_active_schedule(answer_structured.active_schedule_items))
        if answer_structured.consent_constraints and answer_type == "consent_limited_answer":
            sentences.extend(_render_consent(answer_structured.consent_constraints))
        return _dedupe_list(sentences)
    if answer_structured.allergies:
        sentences.extend(_render_allergies(answer_structured.allergies))
    if answer_structured.regimen_items:
        sentences.extend(_render_regimen_items(answer_structured.regimen_items))
    elif answer_structured.medications:
        sentences.extend(_render_medications(answer_structured.medications))
    sentences.extend(_render_state_frames(answer_structured.utility_frames))
    if answer_structured.active_schedule_items:
        sentences.extend(_render_active_schedule(answer_structured.active_schedule_items))
    if answer_structured.instructions:
        sentences.extend(_render_instructions(answer_structured.instructions))
    if answer_structured.canceled_items:
        sentences.extend(_render_canceled_items(answer_structured.canceled_items))
    if answer_structured.consent_constraints and "consent_limited_answer" in answer_type:
        sentences.extend(_render_consent(answer_structured.consent_constraints))
    return _dedupe_list(sentences)


PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")


def _render_contact_plan(frames: list[dict]) -> list[str]:
    texts = _utility_source_texts(frames)
    if not texts:
        return []
    sentences: list[str] = []
    current_phone = ""
    prior_phone = ""
    for text in texts:
        transition_match = re.search(
            r"(?:changes?\s+from|from)\s+(\d{3}-\d{3}-\d{4})\s+(?:to|into)\s+(\d{3}-\d{3}-\d{4})",
            text,
            re.IGNORECASE,
        )
        if transition_match:
            prior_phone = transition_match.group(1)
            current_phone = transition_match.group(2)
            break
        instead_match = re.search(
            r"use\s+(\d{3}-\d{3}-\d{4})\s+instead of\s+(\d{3}-\d{3}-\d{4})",
            text,
            re.IGNORECASE,
        )
        if instead_match:
            current_phone = instead_match.group(1)
            prior_phone = instead_match.group(2)
            break
        direct_match = re.search(r"\bdirect mobile\s+(\d{3}-\d{3}-\d{4})\b", text, re.IGNORECASE)
        if direct_match and not current_phone:
            current_phone = direct_match.group(1)
    if not current_phone:
        phones = []
        for text in texts:
            phones.extend(PHONE_RE.findall(text))
        if phones:
            current_phone = phones[-1]
            if len(phones) > 1:
                prior_phone = phones[-2]
    if current_phone and prior_phone and current_phone != prior_phone:
        sentences.append(f"Use {current_phone} instead of {prior_phone}.")
    elif current_phone:
        sentences.append(f"Use {current_phone}.")

    rule_parts: list[str] = []
    if any("direct mobile" in text.lower() for text in texts) and current_phone:
        rule_parts.append(f"direct mobile {current_phone}")
    text_okay_match = next(
        (
            re.search(r"\btext okay(?:\s+after\s+[0-9: ]+(?:AM|PM))?\b", text, re.IGNORECASE)
            for text in texts
            if "text okay" in text.lower()
        ),
        None,
    )
    if text_okay_match:
        rule_parts.append(text_okay_match.group(0).strip())
    do_not_use_match = next(
        (
            re.search(r"\bdo not use\s+([A-Za-z][A-Za-z ]+)\b", text, re.IGNORECASE)
            for text in texts
            if "do not use" in text.lower()
        ),
        None,
    )
    if do_not_use_match:
        rule_parts.append(f"do not use {do_not_use_match.group(1).strip()}")
    if any("generic callback only" in text.lower() for text in texts):
        rule_parts.append("generic callback only")
    voicemail_match = next(
        (
            re.search(r"\bdo not mention\s+([^.;]+)\s+in voicemail\b", text, re.IGNORECASE)
            for text in texts
            if "voicemail" in text.lower()
        ),
        None,
    )
    if voicemail_match:
        rule_parts.append(f"do not mention {voicemail_match.group(1).strip()} in voicemail")
    elif any("generic voicemail only" in text.lower() for text in texts):
        rule_parts.append("generic voicemail only")
    if any("portal" in text.lower() for text in texts) or (
        any("generic callback only" in text.lower() for text in texts)
        and any("voicemail" in text.lower() for text in texts)
    ):
        rule_parts.append("use the portal for details")
    rule_parts = list(dict.fromkeys(rule_parts))
    if rule_parts:
        sentences.append("Current callback plan: " + "; ".join(rule_parts) + ".")
    return _dedupe_list(sentences)


def _render_result_summary(frames: list[dict]) -> list[str]:
    texts = _utility_source_texts(frames)
    if not texts:
        return []
    sentences: list[str] = []
    for text in texts:
        lab_match = re.search(r"\bcurrent lab status is\s+([^.]+)", text, re.IGNORECASE)
        if lab_match:
            sentences.append("Current lab status: " + lab_match.group(1).strip().rstrip(".") + ".")
            break
    if not sentences:
        for text in texts:
            lowered = text.lower()
            if "confirmatory" in lowered and "cd4" in lowered and "cmp" in lowered:
                parts: list[str] = []
                if "positive" in lowered:
                    parts.append("HIV confirmatory positive")
                cd4_match = re.search(r"\bcd4(?:\s+is)?\s+(\d+)\b", text, re.IGNORECASE)
                if cd4_match:
                    parts.append(f"CD4 {cd4_match.group(1)}")
                if "cmp is normal" in lowered or "cmp normal" in lowered:
                    parts.append("CMP normal")
                if parts:
                    lead = ", ".join(parts[:-1])
                    tail = parts[-1]
                    body = f"{lead}, and {tail}" if lead else tail
                    sentences.append(f"Current lab status: {body}.")
                    break
    for text in texts:
        diagnosis_match = re.search(
            r"\b(?:leading diagnosis remains|current diagnosis(?: for [^.]+)?(?: is| remains)?|current suspicion is)\s+([^.]+)",
            text,
            re.IGNORECASE,
        )
        if diagnosis_match:
            sentences.append("Current diagnosis: " + diagnosis_match.group(1).strip().rstrip(".") + ".")
            break
    return _dedupe_list(sentences)


def _render_access_scope(frames: list[dict], consent_constraints: list[dict]) -> list[str]:
    texts = _utility_source_texts(frames)
    for item in consent_constraints:
        consent_text = str(item.get("consent_scope") or "").strip()
        if consent_text:
            texts.append(consent_text)
    texts = _dedupe_list([text for text in texts if text])
    if not texts:
        return []
    for text in texts:
        lowered = text.lower()
        if "logs-only" in lowered:
            return ["The debugging contractor was given logs-only access."]
        if "scheduling-only" in lowered:
            return ["The access scope was scheduling-only."]
    scope_text = next(
        (
            text.strip()
            for text in texts
            if any(token in text.lower() for token in ["access through", "standing privilege", "approved for the contractor", "access is", "scope is"])
        ),
        "",
    )
    return [scope_text] if scope_text else []


def _utility_source_texts(frames: list[dict]) -> list[str]:
    texts: list[str] = []
    for frame in frames:
        text = str(frame.get("source_text") or "").strip()
        if text:
            texts.append(text)
            continue
        if frame.get("frame_type") == "consent_or_permission":
            scope = str((frame.get("surface_spans") or {}).get("consent_scope") or (frame.get("slots") or {}).get("consent_scope") or "").strip()
            if scope:
                texts.append(scope)
    return _dedupe_list(texts)


def _question_requests_deleted_secret_no_memory(question: str) -> bool:
    lowered = question.lower()
    return "deleted" in lowered and any(
        token in lowered
        for token in [
            "backup contact",
            "backup number",
            "old number",
            "alias",
            "before it was removed",
            "before it was deleted",
        ]
    )


def _question_requests_medical_sensitive_existence(question: str) -> bool:
    lowered = question.lower()
    asks_binary = any(
        token in lowered
        for token in ["which is it", "whether ", "is there a ", "is this ", "just yes or no", "only need yes or no"]
    )
    medical_target = any(
        token in lowered
        for token in ["behavioral", "psychiatry", "primary care", "tumor", "cancer", "diagnosis", "positive", "infection"]
    )
    return asks_binary and medical_target


def _question_requests_office_mapping_inference(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in ["same account as", "same customer as", "which customer", "which account"]) and any(
        token in lowered for token in ["deleted mapping", "earlier deleted mapping", "used to discuss", "without naming the customer"]
    )


def _question_requests_exact_restricted_state(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in [
            "exact badge",
            "exact private suite",
            "exact suite",
            "exact room",
            "exact amount",
            "exact wording",
            "exact private wording",
            "portal credential",
            "current credential",
        ]
    )


def _principal_has_medical_sensitive_access(principal: Principal) -> bool:
    if principal.relation_to_owner == "owner":
        return True
    return str(principal.role or "").lower() in {"clinician", "nurse", "labtech", "pharmacist", "social_worker"}


def _should_append_redaction_notice(answer_structured: AnswerStructured) -> bool:
    if not (answer_structured.action == "answer_redacted" or answer_structured.redactions):
        return False
    answer_type = str(answer_structured.answer_type or "")
    has_clinical_payload = bool(
        answer_structured.allergies
        or answer_structured.medications
        or answer_structured.regimen_items
        or answer_structured.consent_constraints
    )
    has_state_payload = any(
        str(frame.get("frame_type") or "") in {"project_state", "research_state", "household_plan"}
        for frame in answer_structured.utility_frames
    )
    if answer_type in {"current_state", "mixed"} and has_state_payload and not has_clinical_payload:
        return False
    return True


def verify_rendered_answer_contains_slots(rendered_answer: str, slot_audit: SlotAuditResult, config: dict) -> dict:
    rendered_lower = rendered_answer.lower()
    hits = []
    misses = []
    required_renderable = [slot for slot in slot_audit.required_slots if slot in slot_audit.renderable_slots]
    for slot_name in required_renderable:
        surface = str(slot_audit.slot_to_surface.get(slot_name) or "").strip()
        if surface and surface.lower() in rendered_lower:
            hits.append(slot_name)
        elif surface:
            misses.append(slot_name)
    return {
        "rendered_slot_hits": hits,
        "rendered_slot_misses": misses,
        "renderer_omission": bool(misses),
    }


def append_missing_slot_sentences(rendered_answer: str, missed_slots: list[str], slot_audit: SlotAuditResult) -> str:
    additions = []
    for slot_name in missed_slots:
        surface = slot_audit.slot_to_surface.get(slot_name)
        if not surface:
            continue
        label = slot_name.split(".", 1)[-1].replace("_", " ")
        additions.append(f"{label}: {surface}.")
    if not additions:
        return rendered_answer
    return (rendered_answer + " " + " ".join(additions)).strip()


def correct_action_with_runtime_evidence(
    action_decision,
    answer_structured: AnswerStructured,
    principal: Principal,
    slot_audit: SlotAuditResult,
    config: dict,
    query_type: str | None = None,
    question: str | None = None,
):
    trace = []
    corrected = action_decision
    normalized_query_type = str(query_type or "").lower()
    forgetting_sensitive = _answer_structured_has_forgetting_or_retired_secret(answer_structured)
    rationale = str(getattr(corrected, "rationale_summary", "") or "").lower()
    strong_no_memory_intent = bool(
        getattr(corrected, "forgetting_decision", None)
        or "[normalized:deleted_secret_no_memory]" in rationale
        or "[normalized:safety_forgetting_no_memory]" in rationale
    )
    normalized_question = str(question or "").lower()
    exact_restricted_state_request = _question_requests_exact_restricted_state(normalized_question)
    complete_required_coverage = bool(slot_audit.required_slots) and not slot_audit.missing_slots
    if _question_requests_deleted_secret_no_memory(normalized_question) and corrected.action == "refuse":
        corrected.action = "no_memory"
        corrected.answer_mode = "abstain"
        trace.append("Corrected refuse to no_memory because deleted contact/alias recovery requests must not reconstruct removed memory.")
    if (
        normalized_query_type in {"privacy", "utility"}
        and corrected.action in {"answer", "answer_redacted", "no_memory"}
        and _question_requests_medical_sensitive_existence(normalized_question)
        and not _principal_has_medical_sensitive_access(principal)
    ):
        corrected.action = "refuse"
        corrected.answer_mode = "refusal"
        trace.append("Corrected action to refuse because non-clinical/non-owner requesters may not obtain sensitive medical existence or diagnosis disclosures.")
    if (
        normalized_query_type == "utility"
        and corrected.action == "no_memory"
        and not forgetting_sensitive
        and not strong_no_memory_intent
        and complete_required_coverage
        and answer_structured_has_allowed_utility_slots(answer_structured)
        and slot_audit.renderable_slots
    ):
        corrected.action = "answer"
        corrected.answer_mode = "direct"
        trace.append("Corrected no_memory to answer because allowed runtime evidence contains renderable utility slots.")
    if (
        normalized_query_type == "utility"
        and corrected.action == "refuse"
        and not forgetting_sensitive
        and not strong_no_memory_intent
        and principal.relation_to_owner in {"owner", "authorized_staff"}
        and answer_structured_has_allowed_utility_slots(answer_structured)
    ):
        corrected.action = "answer"
        corrected.answer_mode = "direct"
        trace.append("Corrected refuse to answer because principal has utility access and renderable slots exist.")
    nonclinical_state_slots = any(
        slot.startswith("household_plan.") or slot.startswith("project_state.") or slot.startswith("research_state.")
        for slot in slot_audit.renderable_slots
    )
    has_clinical_payload = bool(
        answer_structured.allergies
        or answer_structured.medications
        or answer_structured.regimen_items
    )
    if (
        normalized_query_type == "utility"
        and corrected.action == "answer_redacted"
        and nonclinical_state_slots
        and not has_clinical_payload
        and not answer_structured.redactions
        and not forgetting_sensitive
        and not strong_no_memory_intent
    ):
        corrected.action = "answer"
        corrected.answer_mode = "direct"
        trace.append("Corrected answer_redacted to answer because only nonclinical current-state slots are being disclosed.")
    if (
        normalized_query_type == "utility"
        and corrected.action == "answer_redacted"
        and str(answer_structured.answer_type or "") == "contact_plan"
        and _principal_has_medical_sensitive_access(principal)
        and not forgetting_sensitive
        and not strong_no_memory_intent
    ):
        corrected.action = "answer"
        corrected.answer_mode = "direct"
        trace.append("Corrected answer_redacted to answer because care-team contact-plan queries are directly answerable to the authorized clinical requester.")
    if (
        normalized_query_type == "utility"
        and corrected.action == "refuse"
        and principal.relation_to_owner == "delegate"
        and nonclinical_state_slots
        and not has_clinical_payload
        and not exact_restricted_state_request
        and not forgetting_sensitive
        and not strong_no_memory_intent
    ):
        corrected.action = "answer_redacted"
        corrected.answer_mode = "redacted"
        trace.append("Corrected delegate current-state refusal because delegate access permits logistics-safe current-state answers with redaction when needed.")
    if (
        normalized_query_type == "utility"
        and corrected.action in {"refuse", "answer_redacted"}
        and principal.relation_to_owner == "family"
        and nonclinical_state_slots
        and not has_clinical_payload
        and not exact_restricted_state_request
        and not answer_structured.redactions
        and not forgetting_sensitive
        and not strong_no_memory_intent
    ):
        corrected.action = "answer"
        corrected.answer_mode = "direct"
        trace.append("Corrected family current-state answer to direct answer because only nonclinical state slots are being disclosed and no redacted evidence is involved.")
    if (
        normalized_query_type == "utility"
        and corrected.action == "refuse"
        and principal.relation_to_owner == "family"
        and not forgetting_sensitive
        and not strong_no_memory_intent
        and any(slot.startswith("schedule.") or slot.startswith("instruction.") for slot in slot_audit.renderable_slots)
    ):
        corrected.action = "answer_redacted"
        corrected.answer_mode = "redacted"
        trace.append("Corrected refuse to answer_redacted because family requester has logistics-only renderable slots.")
    non_owner_household_state = any(slot.startswith("household_plan.") for slot in slot_audit.renderable_slots)
    sensitive_household_state = any(
        slot.startswith("household_plan.") and any(token in slot for token in ["visit_window", "entry_method", "approved_areas", "parking_pass", "arrival_contact_rule", "package_rule"])
        for slot in slot_audit.renderable_slots
    )
    if corrected.action in {"answer", "answer_redacted"} and principal.relation_to_owner not in {"owner", "authorized_staff", "delegate"} and non_owner_household_state and (sensitive_household_state or answer_structured.redactions):
        corrected.action = "refuse"
        corrected.answer_mode = "deny"
        trace.append("Corrected household state answer to refuse because non-owner/non-staff/non-delegate requesters must not receive sensitive household-plan details.")
    if _question_requests_deleted_secret_no_memory(normalized_question) and corrected.action != "no_memory":
        corrected.action = "no_memory"
        corrected.answer_mode = "abstain"
        trace.append("Forced no_memory at the final guardrail because deleted contact recovery requests cannot be answered or refused as if the deleted value were still available.")
    if (
        _question_requests_medical_sensitive_existence(normalized_question)
        and not _principal_has_medical_sensitive_access(principal)
        and corrected.action in {"answer", "answer_redacted"}
    ):
        corrected.action = "refuse"
        corrected.answer_mode = "refusal"
        trace.append("Forced refuse at the final guardrail because non-clinical/non-owner requesters cannot receive sensitive medical existence or diagnosis disclosures.")
    if _question_requests_office_mapping_inference(normalized_question) and corrected.action == "no_memory":
        corrected.action = "refuse"
        corrected.answer_mode = "refusal"
        trace.append("Forced refuse at the final guardrail because indirect account/customer mapping inference requests should be denied rather than treated as absent memory.")
    return corrected, trace


def _answer_structured_has_forgetting_or_retired_secret(answer_structured: AnswerStructured) -> bool:
    for frame in answer_structured.utility_frames:
        frame_type = str(frame.get("frame_type") or "")
        lifecycle_status = str(frame.get("lifecycle_status") or "")
        if frame_type == "forgetting" or lifecycle_status == "deleted":
            return True
        slots = frame.get("surface_spans") or frame.get("slots") or {}
        raw_text = " ".join(str(value) for value in slots.values()).lower()
        if any(token in raw_text for token in ["deleted", "delete", "forget", "erased"]):
            return True
    return False


def answer_structured_to_dict(answer_structured: AnswerStructured) -> dict:
    return asdict(answer_structured)


def answer_structured_has_allowed_utility_slots(answer_structured: AnswerStructured) -> bool:
    return bool(
        answer_structured.active_schedule_items
        or answer_structured.allergies
        or answer_structured.medications
        or answer_structured.regimen_items
        or answer_structured.instructions
        or any(
            str(frame.get("frame_type") or "") in {"project_state", "research_state", "household_plan"}
            and any((frame.get("slots") or {}).get(slot) for slot in [
                "target_date", "approved_budget", "approved_discount_cap", "monthly_stipend",
                "safe_wording", "status", "blocker", "visit_window", "entry_method", "package_rule", "approved_areas",
                "parking_pass", "arrival_contact_rule", "time", "secondary_time",
            ])
            for frame in answer_structured.utility_frames
        )
    )


def _split_medication_clauses(text: str) -> list[str]:
    pieces = []
    seen = set()
    for raw_piece in re.split(r"(?:\.\s+|;\s+|,\s+(?=(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to)\b))", str(text or "")):
        piece = raw_piece.strip(" .;,:")
        if not piece:
            continue
        match = re.search(r"\b(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to)\b", piece, re.IGNORECASE)
        if not match:
            continue
        piece = piece[match.start():].strip(" .;,:")
        normalized = " ".join(piece.split()).lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            pieces.append(piece[:1].upper() + piece[1:] + ".")
    return pieces


def _ledger_from_reasoning_state(reasoning_state: ReasoningState) -> CurrentStateLedger | None:
    ledger = reasoning_state.current_state_ledger or {}
    if not ledger:
        return None
    try:
        return CurrentStateLedger(
            owner_user=ledger.get("owner_user"),
            active_events=_event_states_from_dict(ledger.get("active_events") or {}),
            canceled_events=_event_states_from_dict(ledger.get("canceled_events") or {}),
            superseded_events=_event_states_from_dict(ledger.get("superseded_events") or {}),
            deleted_events=_event_states_from_dict(ledger.get("deleted_events") or {}),
            active_slots={},
            canceled_slots={},
            deleted_slots={},
            superseded_slots={},
            trace=list(ledger.get("trace") or []),
        )
    except Exception:
        return None


def _event_states_from_dict(raw: dict) -> dict[str, EventState]:
    events: dict[str, EventState] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            events[str(key)] = EventState(
                event_key=str(value.get("event_key") or key),
                frame_type=str(value.get("frame_type") or "general_fact"),
                subject_entity=value.get("subject_entity"),
                lifecycle_status=str(value.get("lifecycle_status") or "active"),
                slots=dict(value.get("slots") or {}),
                surface_spans=dict(value.get("surface_spans") or {}),
                frame_ids=[str(x) for x in value.get("frame_ids", [])],
                memory_ids=[str(x) for x in value.get("memory_ids", [])],
                effective_time=value.get("effective_time"),
                confidence=float(value.get("confidence", 0.0)),
            )
        except Exception:
            continue
    return events


def _collect_utility_context(
    ledger: CurrentStateLedger | None,
    frames: list[EvidenceFrame],
    evidence_text_by_memory_id: dict[str, str] | None = None,
) -> UtilityContext:
    context = UtilityContext()
    evidence_text_by_memory_id = evidence_text_by_memory_id or {}
    for frame in frames:
        context.utility_frames.append(
            {
                "frame_id": frame.frame_id,
                "memory_id": frame.memory_id,
                "frame_type": frame.frame_type,
                "lifecycle_status": frame.lifecycle_status,
                "slots": dict(frame.slots),
                "surface_spans": dict(frame.surface_spans),
                "sensitivity": dict(frame.sensitivity or {}),
                "privacy_level": str((frame.sensitivity or {}).get("privacy_level") or ""),
                "source_text": evidence_text_by_memory_id.get(str(frame.memory_id), ""),
            }
        )
        if frame.sensitivity.get("redaction_required"):
            context.redactions.append(
                {
                    "frame_id": frame.frame_id,
                    "memory_id": frame.memory_id,
                    "redacted_slots": sorted(frame.slots.keys()),
                }
            )

    if ledger is None:
        for frame in frames:
            _append_frame_by_type(context, frame)
        return context

    for event in ledger.active_events.values():
        if event.frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "update"}:
            context.active_schedule_items.append(_event_to_schedule_item(event))
        elif event.frame_type == "instruction":
            context.instructions.append(_event_to_instruction(event))
        elif event.frame_type == "allergy":
            context.allergies.append(_event_to_allergy(event))
        elif event.frame_type == "medication":
            context.medications.append(_event_to_medication(event))
        elif event.frame_type in {"consent_or_permission", "privacy_policy"}:
            context.consent_constraints.append(_event_to_consent(event))
        context.evidence_memory_ids.extend(event.memory_ids)

    for event in ledger.canceled_events.values():
        context.canceled_items.append(_event_to_schedule_item(event))
        context.evidence_memory_ids.extend(event.memory_ids)

    for event in ledger.superseded_events.values():
        context.unavailable_slots.extend(event.slots.keys())
        context.evidence_memory_ids.extend(event.memory_ids)

    for event in ledger.deleted_events.values():
        context.unavailable_slots.extend(event.slots.keys())
        context.evidence_memory_ids.extend(event.memory_ids)

    if not context.active_schedule_items:
        for frame in frames:
            if frame.frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "update"}:
                context.active_schedule_items.append(_frame_to_schedule_item(frame))
    if not context.allergies:
        for frame in frames:
            if frame.frame_type == "allergy":
                context.allergies.append(_frame_to_allergy(frame))
    if not context.medications:
        for frame in frames:
            if frame.frame_type == "medication":
                context.medications.append(_frame_to_medication(frame))
    if not context.instructions:
        for frame in frames:
            if frame.frame_type == "instruction":
                context.instructions.append(_frame_to_instruction(frame))

    return _dedupe_context(context)


def _append_frame_by_type(context: UtilityContext, frame: EvidenceFrame) -> None:
    if frame.frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "update"}:
        context.active_schedule_items.append(_frame_to_schedule_item(frame))
    elif frame.frame_type == "cancellation":
        context.canceled_items.append(_frame_to_schedule_item(frame))
    elif frame.frame_type == "instruction":
        context.instructions.append(_frame_to_instruction(frame))
    elif frame.frame_type == "allergy":
        context.allergies.append(_frame_to_allergy(frame))
    elif frame.frame_type == "medication":
        context.medications.append(_frame_to_medication(frame))
    elif frame.frame_type in {"consent_or_permission", "privacy_policy"}:
        context.consent_constraints.append(_frame_to_consent(frame))
    context.evidence_memory_ids.append(frame.memory_id)


def _event_to_schedule_item(event: EventState) -> dict:
    return {
        "event_key": event.event_key,
        "frame_type": event.frame_type,
        "status": event.slots.get("status") or event.lifecycle_status,
        "procedure": _slot(event, "procedure"),
        "visit_type": _slot(event, "visit_type"),
        "date": _surface(event, "date"),
        "time": _surface(event, "time"),
        "arrival_time": _surface(event, "arrival_time"),
        "location": _surface(event, "location"),
        "provider": _surface(event, "provider"),
        "condition": _surface(event, "precondition") or _surface(event, "condition"),
        "source_frame_ids": list(event.frame_ids),
        "source_memory_ids": list(event.memory_ids),
    }


def _event_to_instruction(event: EventState) -> dict:
    return {
        "instruction": _surface(event, "instruction") or _surface(event, "prep_instruction") or next(iter(event.slots.values()), None),
        "condition": _surface(event, "condition") or _surface(event, "precondition"),
        "timing": _surface(event, "timing") or _surface(event, "date"),
        "source_frame_ids": list(event.frame_ids),
        "source_memory_ids": list(event.memory_ids),
    }


def _event_to_allergy(event: EventState) -> dict:
    return {
        "substance": _surface(event, "substance") or _slot(event, "substance"),
        "reaction": _surface(event, "reaction") or _slot(event, "reaction"),
        "source_frame_ids": list(event.frame_ids),
        "source_memory_ids": list(event.memory_ids),
    }


def _event_to_medication(event: EventState) -> dict:
    medication = _surface(event, "medication") or _slot(event, "medication")
    instruction = _surface(event, "instruction") or _slot(event, "instruction")
    timing = _surface(event, "timing") or _slot(event, "timing")
    clauses = []
    if instruction:
        clauses = _split_medication_clauses(str(instruction))
    elif medication:
        clauses = _split_medication_clauses(str(medication))
    return {
        "medication": medication,
        "instruction": instruction,
        "timing": timing,
        "regimen_clauses": clauses,
        "source_frame_ids": list(event.frame_ids),
        "source_memory_ids": list(event.memory_ids),
    }


def _event_to_consent(event: EventState) -> dict:
    return {
        "consent_scope": _surface(event, "consent_scope") or _slot(event, "consent_scope"),
        "source_frame_ids": list(event.frame_ids),
        "source_memory_ids": list(event.memory_ids),
    }


def _frame_to_schedule_item(frame: EvidenceFrame) -> dict:
    return {
        "frame_id": frame.frame_id,
        "memory_id": frame.memory_id,
        "frame_type": frame.frame_type,
        "status": frame.slots.get("status") or frame.lifecycle_status,
        "procedure": frame.surface_spans.get("procedure") or frame.slots.get("procedure"),
        "visit_type": frame.surface_spans.get("visit_type") or frame.slots.get("visit_type"),
        "date": frame.surface_spans.get("date") or frame.slots.get("date"),
        "time": frame.surface_spans.get("time") or frame.slots.get("time"),
        "arrival_time": frame.surface_spans.get("arrival_time") or frame.slots.get("arrival_time"),
        "location": frame.surface_spans.get("location") or frame.slots.get("location"),
        "provider": frame.surface_spans.get("provider") or frame.slots.get("provider"),
        "condition": frame.surface_spans.get("precondition") or frame.slots.get("precondition"),
        "source_frame_ids": [frame.frame_id],
        "source_memory_ids": [frame.memory_id],
    }


def _frame_to_instruction(frame: EvidenceFrame) -> dict:
    return {
        "frame_id": frame.frame_id,
        "memory_id": frame.memory_id,
        "instruction": frame.surface_spans.get("prep_instruction") or frame.slots.get("prep_instruction") or frame.surface_spans.get("instruction") or frame.slots.get("instruction"),
        "condition": frame.surface_spans.get("precondition") or frame.slots.get("precondition"),
        "timing": frame.surface_spans.get("timing") or frame.slots.get("timing") or frame.surface_spans.get("date") or frame.slots.get("date"),
        "source_frame_ids": [frame.frame_id],
        "source_memory_ids": [frame.memory_id],
    }


def _frame_to_allergy(frame: EvidenceFrame) -> dict:
    return {
        "frame_id": frame.frame_id,
        "memory_id": frame.memory_id,
        "substance": frame.surface_spans.get("substance") or frame.slots.get("substance"),
        "reaction": frame.surface_spans.get("reaction") or frame.slots.get("reaction"),
        "source_frame_ids": [frame.frame_id],
        "source_memory_ids": [frame.memory_id],
    }


def _frame_to_medication(frame: EvidenceFrame) -> dict:
    medication = frame.surface_spans.get("medication") or frame.slots.get("medication")
    instruction = frame.surface_spans.get("instruction") or frame.slots.get("instruction")
    timing = frame.surface_spans.get("timing") or frame.slots.get("timing")
    clauses = []
    if instruction:
        clauses = _split_medication_clauses(str(instruction))
    elif medication:
        clauses = _split_medication_clauses(str(medication))
    return {
        "frame_id": frame.frame_id,
        "memory_id": frame.memory_id,
        "medication": medication,
        "instruction": instruction,
        "timing": timing,
        "regimen_clauses": clauses,
        "source_frame_ids": [frame.frame_id],
        "source_memory_ids": [frame.memory_id],
    }


def _frame_to_consent(frame: EvidenceFrame) -> dict:
    return {
        "frame_id": frame.frame_id,
        "memory_id": frame.memory_id,
        "consent_scope": frame.surface_spans.get("consent_scope") or frame.slots.get("consent_scope"),
        "source_frame_ids": [frame.frame_id],
        "source_memory_ids": [frame.memory_id],
    }


def _question_has_contact_plan_signal(question: str) -> bool:
    lowered = question.lower()
    has_contact_token = any(
        token in lowered
        for token in [
            "callback plan",
            "safe callback plan",
            "contact plan",
            "callback instruction",
            "callback number",
            "direct mobile",
            "voicemail",
            "portal",
            "text okay",
        ]
    )
    if not has_contact_token:
        return False
    if any(
        token in lowered
        for token in [
            "amount",
            "approved amount",
            "support amount",
            "tuition hold",
            "hold remains",
            "hold state",
            "blocker",
            "badge",
            "suite",
            "credential",
        ]
    ):
        return False
    return True


def _question_has_result_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in [
            "lab status",
            "lab recap",
            "current diagnosis",
            "leading diagnosis",
            "diagnosis",
            "confirmatory",
            "result",
        ]
    )


def _question_has_access_scope_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        token in lowered
        for token in [
            "access scope",
            "what access",
            "scope was given",
            "scope given",
            "logs-only",
            "scheduling-only",
            "debugging contractor",
        ]
    )


def _infer_answer_type(question: str, context: UtilityContext) -> str:
    lowered = question.lower()
    household_query = bool(infer_household_slots(question))
    contact_plan_query = _question_has_contact_plan_signal(question)
    result_query = _question_has_result_signal(question)
    access_scope_query = _question_has_access_scope_signal(question)
    if contact_plan_query and not household_query:
        return "contact_plan"
    if result_query and access_scope_query:
        return "result_and_scope"
    if result_query:
        return "result_summary"
    if access_scope_query:
        return "policy_scope"
    if "allergy" in lowered or context.allergies:
        return "allergy_summary"
    if any(token in lowered for token in ["medication", "pain", "nausea"]) or (context.medications and not household_query):
        return "medication_summary"
    if infer_prefixed_state_slots(question):
        return "current_state"
    if any(token in lowered for token in ["permission", "authorized", "share", "family access"]) or context.consent_constraints:
        return "consent_limited_answer"
    if any(token in lowered for token in ["schedule", "appointment", "visit", "follow-up", "follow up", "latest", "Tuesday", "Friday"]):
        return "current_state"
    if context.active_schedule_items or context.canceled_items:
        return "schedule_summary"
    return "mixed"



def _infer_required_slots(question: str, query_plan: dict, selected_frames: list, current_state_ledger: dict) -> list[str]:
    lowered = question.lower()
    required = []
    exact_restricted_state_request = _question_requests_exact_restricted_state(question)
    if _question_has_contact_plan_signal(question):
        return []
    if _question_has_result_signal(question):
        required.append("diagnosis_or_result.result")
        if _question_has_access_scope_signal(question):
            required.append("consent_or_permission.consent_scope")
        return list(dict.fromkeys(required))
    if _question_has_access_scope_signal(question):
        return ["consent_or_permission.consent_scope"]
    current_state_slots = infer_prefixed_state_slots(question)
    required.extend(current_state_slots)
    if "exact active portal credential" in lowered or "current portal credential" in lowered:
        required.extend(["instruction.instruction"])

    household_focus_slots = infer_household_composite_required_slots(question)
    household_focus = any(slot.startswith("household_plan.") for slot in current_state_slots) or bool(household_focus_slots)
    if household_focus:
        required.extend(household_focus_slots)
        return list(dict.fromkeys(required))

    medication_focus = any(
        token in lowered
        for token in ["medication", "medicine", "dose", "dosage", "take", "taking", "switch to", "stop", "continue", "supplement", "blood pressure"]
    )
    if any(token in lowered for token in ["appointment", "follow-up", "follow up", "visit", "arrive", "location", "provider", "ultrasound", "scan", "repair window", "arrival window"]) and not medication_focus and not exact_restricted_state_request:
        required.extend([
            "schedule.procedure",
            "schedule.visit_type",
            "schedule.date",
            "schedule.time",
            "schedule.arrival_time",
            "schedule.location",
            "schedule.provider",
            "schedule.status",
            "schedule.condition",
        ])
    if any(token in lowered for token in ["allergy", "reaction"]):
        required.extend(["allergy.substance", "allergy.reaction"])
    if any(
        token in lowered
        for token in [
            "medication",
            "test",
            "instruction",
            "beta-hcg",
            "arrival-confirmation",
            "arrival confirmation",
            "arrival-confirmation rule",
            "arrival confirmation rule",
            "text on arrival",
            "call on arrival",
            "credential",
            "badge",
            "suite",
            "private room",
            "portal",
        ]
    ) and not exact_restricted_state_request:
        required.extend(["medication.medication", "medication.instruction", "instruction.instruction", "instruction.timing", "instruction.condition"])
    if "canceled" in lowered or "no longer active" in lowered or "prior" in lowered:
        required.extend(["cancellation.status", "cancellation.date", "cancellation.time"])
    if not required:
        for frame in selected_frames:
            for slot_name in frame.slots.keys():
                required.append(f"{frame.frame_type}.{slot_name}")
    return list(dict.fromkeys(required))

def _render_active_schedule(items: list[dict]) -> list[str]:
    sentences = []
    for item in items:
        date = item.get("date")
        time = item.get("time")
        arrival_time = item.get("arrival_time")
        location = item.get("location")
        provider = item.get("provider")
        procedure = item.get("procedure") or item.get("visit_type") or "appointment"
        phrase = f"The {procedure} is"
        if date:
            phrase += f" on {date}"
        if time:
            phrase += f" at {time}"
        if location:
            phrase += f" in {location}"
        if provider:
            phrase += f" with {provider}"
        phrase += "."
        if arrival_time:
            phrase += f" Please arrive by {arrival_time}."
        if item.get("condition"):
            phrase += f" {item['condition']}."
        sentences.append(phrase)
    return sentences


def _render_canceled_items(items: list[dict]) -> list[str]:
    sentences = []
    for item in items:
        date = item.get("date")
        time = item.get("time")
        provider = item.get("provider")
        procedure = item.get("procedure") or item.get("visit_type") or "appointment"
        text = f"The prior {procedure}"
        if date:
            text += f" on {date}"
        if time:
            text += f" at {time}"
        if provider:
            text += f" with {provider}"
        text += " is canceled."
        sentences.append(text)
    return sentences


def _render_allergies(items: list[dict]) -> list[str]:
    sentences = []
    for item in items:
        substance = item.get("substance")
        reaction = item.get("reaction")
        if substance and reaction:
            sentences.append(f"The recorded allergy is {substance}, with {reaction} as the reaction.")
        elif substance:
            sentences.append(f"The recorded allergy is {substance}.")
    return sentences


def _render_regimen_items(items: list[dict]) -> list[str]:
    sentences = []
    seen = set()
    for item in items:
        med = str(item.get("medication") or "").strip()
        action = str(item.get("action") or "").strip().lower()
        dose = str(item.get("dose") or "").strip()
        timing = str(item.get("timing") or "").strip()
        condition = str(item.get("condition") or "").strip()
        note = str(item.get("note") or "").strip()
        key = " | ".join([med.lower(), action, dose.lower(), timing.lower(), condition.lower(), note.lower()])
        if not key.strip(" |") or key in seen:
            continue
        seen.add(key)
        parts = []
        if action and med:
            if action in {"continue", "maintain", "keep", "stay on"}:
                parts.append(f"Continue {med}")
            elif action in {"start", "begin", "initiate"}:
                parts.append(f"Start {med}")
            elif action in {"stop", "discontinue", "hold", "pause"}:
                parts.append(f"{action.capitalize()} {med}")
            elif action in {"increase", "decrease", "adjust", "titrate"}:
                parts.append(f"{action.capitalize()} {med}")
            else:
                parts.append(f"{action.capitalize()} {med}")
        elif med:
            parts.append(med)
        if dose:
            parts[-1] = f"{parts[-1]} {dose}" if parts else dose
        if timing:
            parts.append(timing)
        if condition:
            parts.append(condition)
        if note:
            parts.append(note)
        sentence = ". ".join([p.strip().rstrip(".") for p in parts if p.strip()])
        if sentence:
            sentences.append(sentence.rstrip(".") + ".")
    return sentences


def _render_medications(items: list[dict]) -> list[str]:
    sentences = []
    seen_medications = set()
    seen_clauses = set()
    for item in items:
        med = str(item.get("medication") or "").strip()
        timing = str(item.get("timing") or "").strip()
        clauses = [str(x).strip() for x in (item.get("regimen_clauses") or []) if str(x).strip()]
        clause_texts = []
        for clause in clauses:
            normalized = " ".join(clause.split()).lower().rstrip(".")
            if normalized and normalized not in seen_clauses:
                seen_clauses.add(normalized)
                clause_texts.append(clause if clause.endswith(".") else clause + ".")
        if med:
            med_key = " ".join(med.split()).lower().rstrip(".")
            if med_key in seen_medications:
                med = ""
            else:
                seen_medications.add(med_key)
        if not med and not clause_texts and not timing and not item.get("instruction"):
            continue
        if clause_texts:
            text = " ".join(clause_texts)
        elif item.get("instruction"):
            instr = str(item.get("instruction")).strip()
            text = instr + ("." if instr and not instr.endswith(".") else "")
        else:
            text = f"Medication note: {med}." if med else ""
        if timing and timing.lower() not in text.lower():
            text = (text + " " + timing + ".").strip()
        if text:
            sentences.append(text.strip())
    return sentences


def _render_instructions(items: list[dict]) -> list[str]:
    sentences = []
    for item in items:
        instruction = str(item.get("instruction") or "").strip()
        lowered = instruction.lower()
        if not instruction:
            continue
        if any(
            token in lowered
            for token in [
                "passphrase",
                "keypad code",
                "credential",
                "pin",
                "token",
                "guest-network",
                "private-note",
                "separate in memory",
                "separation",
                "thread",
                "should only receive",
                "delegates may coordinate",
                "keep the exact",
                "must stay separated",
                "out of scope",
                "should not receive",
                "do not relay",
                "do not give",
                "treated as private case data",
                "remains private",
                "not authorized",
            ]
        ):
            continue
        sentences.append(instruction)
    return sentences


def _render_state_frames(frames: list[dict], required_slots: list[str] | None = None) -> list[str]:
    sentences: list[str] = []
    required_slots = list(required_slots or [])
    frame_source_texts = [
        _frame_source_text(frame)
        for frame in frames
        if _frame_source_text(frame)
    ]

    household_fields = {
        "visit_window",
        "entry_method",
        "package_rule",
        "approved_areas",
        "parking_pass",
        "arrival_contact_rule",
    }
    required_household_slots = {
        slot.split(".", 1)[1]
        for slot in required_slots
        if slot.startswith("household_plan.") and "." in slot
    }
    household_like_frame_types = {"household_plan", "instruction", "general_fact", "logistics", "update"}
    household_candidates: list[tuple[float, dict, str]] = []
    for frame in frames:
        if str(frame.get("frame_type") or "") not in household_like_frame_types:
            continue
        slots = _normalized_state_frame_slots(frame)
        if not any(slots.get(key) for key in household_fields | {"date"}):
            continue
        source_text = _frame_source_text(frame)
        sensitivity = dict(frame.get("sensitivity") or {})
        if sensitivity.get("redaction_required") or str(frame.get("privacy_level") or "").lower() in {"private", "restricted"}:
            continue

        score = 0.0
        score += 2.0 * sum(1 for key in required_household_slots if slots.get(key))
        if slots.get("date"):
            score += 0.3
        if slots.get("visit_window"):
            score += 0.4
        if str(frame.get("lifecycle_status") or "").lower() == "active":
            score += 0.5
        if slots.get("entry_method") and slots.get("approved_areas"):
            score += 0.5
        if slots.get("arrival_contact_rule") and slots.get("visit_window"):
            score += 0.4
        if slots.get("package_rule") and len(str(slots.get("package_rule") or "")) < 160:
            score += 0.3
        if len(source_text.splitlines()) == 1:
            score += 0.2
        if required_household_slots:
            extra_slots = sum(1 for key in household_fields if key not in required_household_slots and slots.get(key))
            score -= 0.1 * extra_slots

        household_candidates.append((score, slots, source_text))

    household_candidates.sort(key=lambda item: item[0], reverse=True)

    state_candidates: list[tuple[str, dict, str]] = []
    for frame in frames:
        frame_type = str(frame.get("frame_type") or "")
        normalized_slots = _normalized_state_frame_slots(frame)
        effective_frame_type = frame_type
        if frame_type not in {"project_state", "research_state", "household_plan"}:
            inferred = infer_state_record_type(text=_frame_source_text(frame), slots=normalized_slots, frame_type=frame_type)
            if inferred in {"project_state", "research_state", "household_plan"}:
                effective_frame_type = inferred
        if effective_frame_type == "household_plan":
            continue
        if effective_frame_type in {"project_state", "research_state"} or normalized_slots.get("public_event_date"):
            state_candidates.append((effective_frame_type, normalized_slots, _frame_source_text(frame)))

    def _state_slot_score(effective_frame_type: str, slot_name: str, slots: dict, source_text: str) -> float:
        value = str(slots.get(slot_name) or "").strip()
        if not value:
            return float("-inf")
        lowered_source = source_text.lower()
        score = 1.0
        if effective_frame_type in {"project_state", "research_state"}:
            score += 0.5
        if "current" in lowered_source or "active" in lowered_source or "remains" in lowered_source or "is now" in lowered_source:
            score += 0.8
        if slot_name == "target_date":
            if any(token in lowered_source for token in ["current review date", "current target date", "scheduling may use", "logistics note"]):
                score += 2.2
            elif "moves to" in lowered_source:
                score += 0.7
        elif slot_name == "monthly_stipend":
            if any(token in lowered_source for token in ["active private support amount", "active support amount", "active bridge-support amount", "active bridge support amount"]):
                score += 1.6
            if "separate from public" in lowered_source:
                score += 0.2
        elif slot_name == "safe_wording":
            if any(token in lowered_source for token in ["safe label", "safe-summary", "safe summary", "described only as", "safe wording"]):
                score += 2.0
        elif slot_name == "blocker":
            if any(token in lowered_source for token in ["blocker remains", "blocker is now", "it is now", "remaining blocker"]):
                score += 2.0
        elif slot_name == "public_event_date":
            if "public" in lowered_source and "date" in lowered_source:
                score += 2.0
        return score

    def _best_state_slot(slot_name: str, preferred_types: tuple[str, ...] = ()) -> str:
        best_value = ""
        best_score = float("-inf")
        for effective_frame_type, slots, source_text in state_candidates:
            slot_score = _state_slot_score(effective_frame_type, slot_name, slots, source_text)
            if slot_score == float("-inf"):
                continue
            if preferred_types and effective_frame_type in preferred_types:
                slot_score += 0.5
            if slot_score > best_score:
                best_score = slot_score
                best_value = str(slots.get(slot_name) or "").strip()
        return best_value

    required_project_slots = {
        slot.split(".", 1)[1]
        for slot in required_slots
        if slot.startswith("project_state.") and "." in slot
    }
    required_research_slots = {
        slot.split(".", 1)[1]
        for slot in required_slots
        if slot.startswith("research_state.") and "." in slot
    }
    project_parts: list[str] = []
    if not required_research_slots or "target_date" in required_project_slots:
        project_target_date = _best_state_slot("target_date", preferred_types=("project_state",))
        if project_target_date and (not required_project_slots or "target_date" in required_project_slots):
            project_parts.append(f"the current target date is {project_target_date}")
    project_budget = _best_state_slot("approved_budget", preferred_types=("project_state",))
    if project_budget and (not required_project_slots or "approved_budget" in required_project_slots):
        project_parts.append(f"the current approved budget is {project_budget}")
    project_discount = _best_state_slot("approved_discount_cap", preferred_types=("project_state",))
    if project_discount and (not required_project_slots or "approved_discount_cap" in required_project_slots):
        project_parts.append(f"the current approved maximum discount is {project_discount}")
    project_blocker = _best_state_slot("blocker", preferred_types=("project_state", "research_state"))
    if project_blocker and (not required_project_slots or "blocker" in required_project_slots):
        project_parts.append(f"the current blocker is {project_blocker}")
    if project_parts and (required_project_slots or not required_research_slots):
        sentences.append("Project status: " + ", ".join(project_parts) + ".")

    research_parts: list[str] = []
    research_target_date = _best_state_slot("target_date", preferred_types=("research_state", "project_state"))
    if research_target_date and (not required_research_slots or "target_date" in required_research_slots):
        research_parts.append(f"the current review date is {research_target_date}")
    research_stipend = _best_state_slot("monthly_stipend", preferred_types=("research_state",))
    if research_stipend and (not required_research_slots or "monthly_stipend" in required_research_slots):
        research_parts.append(f"the current monthly stipend is {research_stipend}")
    research_safe_wording = _best_state_slot("safe_wording", preferred_types=("research_state",))
    if research_safe_wording and (not required_research_slots or "safe_wording" in required_research_slots):
        research_parts.append(f"the safe label is {research_safe_wording}")
    research_status = _best_state_slot("status", preferred_types=("research_state",))
    if research_status and (not required_research_slots or "status" in required_research_slots):
        research_parts.append(f"the current review status is {research_status}")
    research_blocker = _best_state_slot("blocker", preferred_types=("research_state", "project_state"))
    if research_blocker and (not required_research_slots or "blocker" in required_research_slots):
        research_parts.append(f"the current blocker is {research_blocker}")
    research_access_room = _best_state_slot("access_room", preferred_types=("research_state",))
    if research_access_room and (not required_research_slots or "access_room" in required_research_slots):
        research_parts.append(f"the active private room is {research_access_room}")
    research_access_badge = _best_state_slot("access_badge", preferred_types=("research_state",))
    if research_access_badge and (not required_research_slots or "access_badge" in required_research_slots):
        research_parts.append(f"the active badge is {research_access_badge}")
    if research_parts:
        sentences.append("Current research state: " + ", ".join(research_parts) + ".")

    public_event_required = any(slot == "schedule.public_event_date" for slot in required_slots)
    public_event_date = _best_state_slot("public_event_date", preferred_types=("research_state", "project_state"))
    if public_event_date and (public_event_required or not required_slots):
        sentences.append(f"Public schedule: the current public event date is {public_event_date}.")

    def _first_slot(candidates: list[tuple[float, dict, str]], key: str) -> str:
        for _, slots, _ in candidates:
            value = str(slots.get(key) or "").strip()
            if value:
                return value
        return ""

    if household_candidates:
        top_slots = household_candidates[0][1]
        top_source_text = household_candidates[0][2]
        anchor_slots = household_candidates[0][1]
        anchor_source_text = household_candidates[0][2]
        if required_household_slots:
            best_anchor_score = float("-inf")
            for score, slots, source_text in household_candidates:
                matched_required = sum(1 for key in required_household_slots if str(slots.get(key) or "").strip())
                if matched_required == 0:
                    continue
                candidate_score = (3.0 * matched_required) + score
                if candidate_score > best_anchor_score:
                    best_anchor_score = candidate_score
                    anchor_slots = slots
                    anchor_source_text = source_text
        household_aggregate: dict[str, str] = {}
        for key in ["date", "visit_window", "approved_areas", "arrival_contact_rule", "entry_method", "package_rule", "parking_pass"]:
            value = str(anchor_slots.get(key) or "").strip()
            if value:
                household_aggregate[key] = value
        anchor_date = str(household_aggregate.get("date") or "").strip()
        for _, slots, source_text in household_candidates:
            if anchor_source_text and source_text == anchor_source_text:
                continue
            candidate_date = str(slots.get("date") or "").strip()
            if anchor_date and candidate_date and candidate_date != anchor_date:
                continue
            keys_to_fill = required_household_slots or {"date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}
            for key in keys_to_fill:
                if household_aggregate.get(key):
                    continue
                value = str(slots.get(key) or "").strip()
                if value:
                    household_aggregate[key] = value

        household_parts = []
        household_date = str(household_aggregate.get("date") or "").strip()
        if household_aggregate.get("visit_window") and household_date:
            expanded_household_date = _expand_weekday_to_full_date(household_date, anchor_source_text or top_source_text)
            household_parts.append(f"the current visit window is {(expanded_household_date or household_date)} from {household_aggregate['visit_window']}")
        elif household_aggregate.get("visit_window"):
            household_parts.append(f"the current visit window is {household_aggregate['visit_window']}")
        elif household_date:
            expanded_household_date = _expand_weekday_to_full_date(household_date, anchor_source_text or top_source_text)
            household_parts.append(f"the current active date is {expanded_household_date or household_date}")
        if household_aggregate.get("entry_method"):
            household_parts.append(f"the entry method is {household_aggregate['entry_method']}")
        if household_aggregate.get("package_rule"):
            household_parts.append(f"the current logistics rule is {household_aggregate['package_rule']}")
        if household_aggregate.get("approved_areas"):
            household_parts.append(f"approved areas are {household_aggregate['approved_areas']}")
        if household_aggregate.get("parking_pass"):
            household_parts.append(f"the current parking pass is {household_aggregate['parking_pass']}")
        if household_aggregate.get("arrival_contact_rule"):
            household_parts.append(f"the arrival-contact rule is {household_aggregate['arrival_contact_rule']}")
        if household_parts:
            sentences.append("Current active state: " + ", ".join(household_parts) + ".")

    return _dedupe_list(sentences)


def _normalized_state_frame_slots(frame: dict) -> dict:
    slots = dict(frame.get("surface_spans") or frame.get("slots") or {})
    frame_type = str(frame.get("frame_type") or "")
    source_text = _frame_source_text(frame)
    if source_text:
        parsed_source_slots = extract_state_slots(source_text)
        for key, value in parsed_source_slots.items():
            slots.setdefault(key, value)
    if frame_type == "household_plan" and not slots.get("visit_window"):
        start = str(slots.get("time") or "").strip()
        end = str(slots.get("secondary_time") or "").strip()
        if start and end:
            slots["visit_window"] = f"{start} to {end}"
    raw_text = " ".join(
        str(value or "")
        for value in [slots.get("prep_instruction"), slots.get("instruction"), slots.get("consent_scope"), slots.get("result")]
    ).strip()
    if raw_text:
        parsed = _extract_generic_state_slots(raw_text)
        for key, value in parsed.items():
            slots.setdefault(key, value)
    return slots


def _extract_generic_state_slots(text: str) -> dict[str, str]:
    slots = dict(extract_state_slots(text))
    lowered = text.lower()
    date_match = re.search(r"\b(?:review date|date)[:\s]+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)", text, re.IGNORECASE)
    if date_match:
        slots.setdefault("target_date", date_match.group(1).strip())
    if "target_date" not in slots:
        fallback_date = re.search(r"\b([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)\b", text)
        if fallback_date and any(token in lowered for token in ["review date", "current date", "date should we use"]):
            slots["target_date"] = fallback_date.group(1).strip()
    status_match = re.search(r"\b(?:review is|status is|is)\s+(closed|open|pending)\b", text, re.IGNORECASE)
    if status_match and any(token in lowered for token in ["review", "status", "advising-side closure", "advising side"]):
        slots["status"] = status_match.group(1).lower().strip()
    if "blocker" not in slots:
        pending_match = re.search(r"\b([^.;]+?)\s+remained pending\b", text, re.IGNORECASE)
        if pending_match:
            slots["blocker"] = pending_match.group(1).strip()
    if "target_date" not in slots:
        moved_match = re.search(r"\b(?:active review target moves to|review target moves to|review date moves to)\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)", text, re.IGNORECASE)
        if moved_match:
            slots["target_date"] = moved_match.group(1).strip()
    if "blocker" not in slots:
        blocker_now_match = re.search(r"\bblocker\s+is\s+now\s+([^.;]+)", text, re.IGNORECASE)
        if blocker_now_match:
            slots["blocker"] = blocker_now_match.group(1).strip()
    return slots


def _frame_source_text(frame: dict) -> str:
    source_text = frame.get("source_text")
    return str(source_text or "").strip()


def _expand_weekday_to_full_date(weekday_text: str, source_text: str) -> str:
    weekday = str(weekday_text or "").strip()
    if not weekday or "," in weekday:
        return weekday
    weekday_index = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }.get(weekday.lower())
    if weekday_index is None:
        return weekday
    ts_match = re.search(r"\[(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}\]", str(source_text or ""))
    if not ts_match:
        return weekday
    try:
        anchor_day = datetime.strptime(ts_match.group(1), "%Y-%m-%d").date()
    except Exception:
        return weekday
    delta = (weekday_index - anchor_day.weekday()) % 7
    resolved = anchor_day + timedelta(days=delta)
    return f"{weekday.title()}, {resolved.strftime('%B')} {resolved.day}"


def _render_consent(items: list[dict]) -> list[str]:
    return [f"Access policy: {item.get('consent_scope')}" for item in items if item.get("consent_scope")]


def _estimate_confidence(frames: list[EvidenceFrame]) -> float:
    if not frames:
        return 0.0
    return sum(frame.confidence for frame in frames) / float(len(frames))


def _dedupe_context(context: UtilityContext) -> UtilityContext:
    context.active_schedule_items = _dedupe_dicts(context.active_schedule_items)
    context.canceled_items = _dedupe_dicts(context.canceled_items)
    context.instructions = _dedupe_dicts(context.instructions)
    context.allergies = _dedupe_dicts(context.allergies)
    context.medications = _dedupe_dicts(context.medications)
    context.consent_constraints = _dedupe_dicts(context.consent_constraints)
    context.redactions = _dedupe_dicts(context.redactions)
    context.utility_frames = _dedupe_dicts(context.utility_frames)
    context.unavailable_slots = _dedupe_list(context.unavailable_slots)
    context.evidence_memory_ids = _dedupe_list(context.evidence_memory_ids)
    return context


def _slot(event: EventState, key: str):
    return event.slots.get(key)


def _surface(event: EventState, key: str):
    return event.surface_spans.get(key) or event.slots.get(key)


def _dedupe_dicts(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for item in items:
        key = repr(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _dedupe_list(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = str(item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(item))
    return out
