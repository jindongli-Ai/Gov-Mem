from __future__ import annotations

import re

from gov_mem.backbones.coverage_packer import PackedUtilityEvidence
from gov_mem.backbones.need_spec import AnswerNeedSpec
from gov_mem.backbones.utility_records import (
    canonicalize_medication_surface,
    extract_medication_clauses,
    normalize_medication_clause_surface,
)


def _extract_medication_entities(text: str) -> set[str]:
    lowered = canonicalize_medication_surface(text).lower()
    entities: set[str] = set()
    for clause in extract_medication_clauses(lowered) or [lowered]:
        clause = clause.lower()
        match = re.search(
            r"(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to|take)\s+"
            r"([a-z][a-z0-9_./'-]*(?:\s+[a-z][a-z0-9_./'-]*){0,4})",
            clause,
        )
        if not match:
            continue
        candidate = re.split(
            r"\s+(?:at|with|for|before|after|until|every|once|twice|daily|nightly|as needed)\b",
            match.group(1),
            maxsplit=1,
        )[0].strip(" .,:;")
        candidate = re.sub(r"\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|tablets?|capsules?)\b.*$", "", candidate).strip()
        if candidate:
            entities.add(candidate)
    return entities


def _infer_carryover_medication_clauses(text: str, covered_medication_entities: set[str]) -> list[str]:
    cleaned = canonicalize_medication_surface(text)
    if not cleaned:
        return []
    clauses: list[str] = []
    pieces = re.split(r",\s+|\s+and\s+", cleaned)
    for piece in pieces:
        candidate = piece.strip(" .;")
        if not candidate:
            continue
        if not re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg)\b", candidate, re.IGNORECASE):
            continue
        med_entities = _extract_medication_entities(candidate)
        if not med_entities:
            continue
        if med_entities & covered_medication_entities:
            continue
        if any(token in candidate.lower() for token in ["twice daily", "every morning", "daily", "nightly", "once daily"]):
            clauses.append(f"Continue {candidate}.")
    return clauses



def render_canonical_answer(
    packed: PackedUtilityEvidence,
    answer_need: AnswerNeedSpec,
    action_decision: dict,
    principal,
    config: dict,
) -> str:
    parts: list[str] = []
    logistics_only = answer_need.minimality_policy == "logistics_only"
    if logistics_only:
        parts.append("I can share logistics only.")
    medication_clauses: list[str] = []
    seen_medication_clauses: set[str] = set()
    covered_medication_entities: set[str] = set()
    for record in packed.selected_records:
        if record.record_type == "medication_status":
            clauses = extract_medication_clauses(record.evidence_line)
            if clauses:
                for clause in clauses:
                    normalized = canonicalize_medication_surface(clause).lower()
                    if not normalized or normalized in seen_medication_clauses:
                        continue
                    seen_medication_clauses.add(normalized)
                    covered_medication_entities.update(_extract_medication_entities(clause))
                    medication_clauses.append(clause)
            else:
                for clause in _infer_carryover_medication_clauses(record.evidence_line, covered_medication_entities):
                    normalized = canonicalize_medication_surface(clause).lower()
                    if not normalized or normalized in seen_medication_clauses:
                        continue
                    seen_medication_clauses.add(normalized)
                    covered_medication_entities.update(_extract_medication_entities(clause))
                    medication_clauses.append(clause)
            continue
        sentence = _render_record(record, answer_need)
        if sentence:
            parts.append(sentence)
    parts = medication_clauses + parts
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _slot_requested(answer_need: AnswerNeedSpec, slot_name: str) -> bool:
    required = set(answer_need.required_slot_groups or [])
    optional = set(answer_need.optional_slot_groups or [])
    if not required and not optional:
        return True
    return slot_name in required or slot_name in optional


def _render_record(record, answer_need: AnswerNeedSpec) -> str:
    slots = record.surface_spans or record.slots
    if record.record_type == "active_schedule":
        procedure = slots.get("procedure") or slots.get("visit_type") or "appointment"
        public_event_date = slots.get("public_event_date")
        requested_public_event_date = _slot_requested(answer_need, "public_event_date")
        date = public_event_date if requested_public_event_date and public_event_date else slots.get("date") or public_event_date
        time = slots.get("time")
        location = slots.get("location")
        arrival = slots.get("arrival_time")
        provider = slots.get("provider")
        if requested_public_event_date and public_event_date and not any(
            _slot_requested(answer_need, slot_name)
            for slot_name in ["date", "time", "location", "arrival_time", "provider"]
        ):
            public_label = "orientation" if str(procedure).lower() == "orientation" else "event"
            return f"The public {public_label} date is {public_event_date}."
        sentence = f"The {procedure} is"
        if date:
            sentence += f" on {date}"
        if time:
            sentence += f" at {time}"
        if location:
            sentence += f" in {location}"
        sentence += "."
        if arrival:
            sentence += f" Please arrive by {arrival}."
        if provider:
            sentence += f" The provider is {provider}."
        return sentence
    if record.record_type == "canceled_schedule":
        procedure = slots.get("procedure") or slots.get("visit_type") or "appointment"
        date = slots.get("date")
        time = slots.get("time")
        sentence = f"The prior {procedure}"
        if date:
            sentence += f" on {date}"
        if time:
            sentence += f" at {time}"
        sentence += " is canceled."
        return sentence
    if record.record_type == "rescheduled_schedule":
        procedure = slots.get("procedure") or slots.get("visit_type") or "appointment"
        date = slots.get("date")
        time = slots.get("time")
        location = slots.get("location")
        sentence = f"The {procedure} was rescheduled"
        if date:
            sentence += f" to {date}"
        if time:
            sentence += f" at {time}"
        if location:
            sentence += f" in {location}"
        sentence += "."
        return sentence
    if record.record_type == "instruction":
        return slots.get("instruction") or record.evidence_line
    if record.record_type == "return_precaution":
        return record.evidence_line
    if record.record_type == "contact_method":
        if slots.get("portal") and slots.get("backup_contact"):
            return f"Use {slots.get('portal')} for contact. The backup contact is {slots.get('backup_contact')}."
        if slots.get("portal"):
            return f"Use {slots.get('portal')} for contact."
        if slots.get("backup_contact"):
            return f"The backup contact is {slots.get('backup_contact')}."
        return record.evidence_line
    if record.record_type == "allergy":
        substance = slots.get("allergy_substance")
        reaction = slots.get("allergy_reaction")
        if substance and reaction:
            return f"The recorded allergy is {substance}, with {reaction} as the reaction."
        return record.evidence_line
    if record.record_type == "medication_status":
        return _render_medication_status_record(record.evidence_line)
    if record.record_type == "policy_permission":
        if slots.get("policy_scope"):
            return f"The allowed sharing scope is {slots.get('policy_scope')}."
        return record.evidence_line
    if record.record_type == "project_state":
        parts = []
        if slots.get("target_date") and _slot_requested(answer_need, "target_date"):
            parts.append(f"the current target date is {slots.get('target_date')}")
        if slots.get("approved_budget") and _slot_requested(answer_need, "approved_budget"):
            parts.append(f"the current approved budget is {slots.get('approved_budget')}")
        if slots.get("approved_discount_cap") and _slot_requested(answer_need, "approved_discount_cap"):
            parts.append(f"the current approved maximum discount is {slots.get('approved_discount_cap')}")
        if slots.get("blocker") and _slot_requested(answer_need, "blocker"):
            parts.append(f"the current blocker is {slots.get('blocker')}")
        return "Project status: " + ", ".join(parts) + "." if parts else record.evidence_line
    if record.record_type == "research_state":
        parts = []
        if slots.get("target_date") and _slot_requested(answer_need, "target_date"):
            parts.append(f"the current dry-run date is {slots.get('target_date')}")
        if slots.get("monthly_stipend") and _slot_requested(answer_need, "monthly_stipend"):
            parts.append(f"the current monthly stipend is {slots.get('monthly_stipend')}")
        if slots.get("safe_wording") and _slot_requested(answer_need, "safe_wording"):
            parts.append(f"the safe sponsor wording is {slots.get('safe_wording')}")
        if slots.get("blocker") and _slot_requested(answer_need, "blocker"):
            parts.append(f"the current blocker is {slots.get('blocker')}")
        return "Current research state: " + ", ".join(parts) + "." if parts else record.evidence_line
    if record.record_type == "household_plan":
        parts = []
        if slots.get("date") and slots.get("visit_window") and (_slot_requested(answer_need, "date") or _slot_requested(answer_need, "visit_window")):
            parts.append(f"the current visit window is {slots.get('date')} from {slots.get('visit_window')}")
        elif slots.get("visit_window") and _slot_requested(answer_need, "visit_window"):
            parts.append(f"the current visit window is {slots.get('visit_window')}")
        if slots.get("entry_method") and _slot_requested(answer_need, "entry_method"):
            parts.append(f"the entry method is {slots.get('entry_method')}")
        if slots.get("package_rule") and _slot_requested(answer_need, "package_rule"):
            parts.append(_render_household_package_rule_sentence(str(slots.get("package_rule") or "")))
        approved_areas_requested = _slot_requested(answer_need, "approved_areas")
        if slots.get("approved_areas") and approved_areas_requested:
            parts.append(_render_household_scope_sentence(str(slots.get("approved_areas") or "")))
        if slots.get("parking_pass") and _slot_requested(answer_need, "parking_pass"):
            parts.append(f"the current parking pass is {slots.get('parking_pass')}")
        if slots.get("arrival_contact_rule") and _slot_requested(answer_need, "arrival_contact_rule"):
            parts.append(f"the arrival-contact rule is {slots.get('arrival_contact_rule')}")
        return "Current household plan: " + ", ".join(parts) + "." if parts else record.evidence_line
    return record.evidence_line


def _render_medication_status_record(text: str) -> str:
    return normalize_medication_clause_surface(text)


def _render_household_package_rule(value: str) -> str:
    cleaned = " ".join(str(value or "").split()).strip(" .;")
    if not cleaned:
        return cleaned
    return cleaned


def _render_household_package_rule_sentence(value: str) -> str:
    cleaned = _render_household_package_rule(value)
    if not cleaned:
        return cleaned
    return f"the current logistics rule is {cleaned}"


def _render_household_scope_sentence(value: str) -> str:
    cleaned = " ".join(str(value or "").split()).strip(" .;")
    if not cleaned:
        return cleaned
    if ";" in cleaned and "out of scope" in cleaned.lower():
        allowed, blocked = [part.strip() for part in cleaned.split(";", 1)]
        return f"approved areas are {allowed}, and {blocked}"
    return f"approved areas are {cleaned}"
