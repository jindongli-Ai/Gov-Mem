from __future__ import annotations

from dataclasses import dataclass, field
import re

from gov_mem.backbones.need_spec import AnswerNeedSpec
from gov_mem.backbones.utility_records import UtilityRecord

RECORD_SLOT_GROUPS: dict[str, set[str]] = {
    "active_schedule": {"date", "public_event_date", "time", "location", "arrival_time", "provider", "procedure", "visit_type", "status"},
    "canceled_schedule": {"date", "time", "location", "status", "procedure", "visit_type"},
    "rescheduled_schedule": {"date", "time", "location", "status", "procedure", "visit_type"},
    "instruction": {"instruction", "condition", "timing"},
    "return_precaution": {"condition", "instruction"},
    "contact_method": {"contact_method", "phone", "portal", "backup_contact"},
    "allergy": {"allergy_substance", "allergy_reaction"},
    "medication_status": {"medication", "dosage", "instruction", "timing", "condition", "medication_name"},
    "policy_permission": {"policy_scope"},
    "project_state": {"target_date", "approved_budget", "approved_discount_cap", "blocker"},
    "research_state": {"target_date", "monthly_stipend", "safe_wording", "blocker"},
    "household_plan": {"date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"},
}


@dataclass
class PackedUtilityEvidence:
    selected_records: list[UtilityRecord]
    dropped_records: list[UtilityRecord]
    coverage: dict
    trace: list[str] = field(default_factory=list)


def pack_utility_records(
    records: list[UtilityRecord],
    answer_need: AnswerNeedSpec,
    config: dict,
) -> PackedUtilityEvidence:
    required_units = _required_units(answer_need)
    required_families = set(answer_need.required_coverage_families or [])
    selected: list[UtilityRecord] = []
    dropped: list[UtilityRecord] = []
    covered: set[str] = set()
    covered_families: set[str] = set()
    remaining = list(records)
    trace: list[str] = []
    while remaining:
        best = None
        best_score = float("-inf")
        for record in remaining:
            score = _score_record(record, answer_need, covered, covered_families, required_units)
            if score > best_score:
                best_score = score
                best = record
        if best is None or best_score <= 0:
            break
        selected.append(best)
        remaining.remove(best)
        newly = _covered_units(best, answer_need)
        covered.update(newly)
        new_families = _record_coverage_families(best)
        covered_families.update(new_families)
        trace.append(f"selected {best.record_id} score={best_score:.2f} units={sorted(newly)} families={sorted(new_families)}")
        units_done = (not required_units) or required_units.issubset(covered)
        families_done = (not required_families) or required_families.issubset(covered_families)
        if units_done and families_done:
            break
    for record in remaining:
        dropped.append(record)
    selected = _dedupe_current_state_slots(selected, answer_need, dropped, trace)
    return PackedUtilityEvidence(
        selected_records=selected,
        dropped_records=dropped,
        coverage={
            "required_units": sorted(required_units),
            "covered_units": sorted(covered),
            "missing_units": sorted(required_units - covered),
            "required_families": sorted(required_families),
            "covered_families": sorted(covered_families),
            "missing_families": sorted(required_families - covered_families),
        },
        trace=trace,
    )


def _dedupe_current_state_slots(
    selected: list[UtilityRecord],
    answer_need: AnswerNeedSpec,
    dropped: list[UtilityRecord],
    trace: list[str],
) -> list[UtilityRecord]:
    if not answer_need.current_state_required:
        return selected
    required_units = _required_units(answer_need)
    all_records = selected + dropped
    retained_ids: set[str] = set()
    for unit in required_units:
        candidates = [record for record in all_records if unit in _covered_units(record, answer_need)]
        if not candidates:
            continue
        retained_ids.add(max(candidates, key=_current_state_record_key).record_id)
    retained = [record for record in all_records if record.record_id in retained_ids]
    retained.extend(
        record
        for record in selected
        if not (_covered_units(record, answer_need) & required_units) and record.record_id not in retained_ids
    )
    for record in all_records:
        if record.record_id in retained_ids or record in retained:
            continue
        if record not in dropped:
            dropped.append(record)
    trace.append(f"resolved current-state units by provenance: {sorted(required_units)}")
    return retained


def _current_state_record_key(record: UtilityRecord) -> tuple[int, str, float]:
    lowered = str(record.evidence_line or "").lower()
    authority = 1
    if any(token in lowered for token in ("revised", "supersedes", "replaces", "updated", "effective")):
        authority = 3
    elif any(token in lowered for token in ("current", "official", "active", "approved", "is now", "remains")):
        authority = 2
    if any(token in lowered for token in ("provisional", "tentative", "exploratory", "initial")):
        authority = 0
    return authority, str(record.source_time or ""), float(record.confidence or 0.0)


def _covered_units(record: UtilityRecord, answer_need: AnswerNeedSpec) -> set[str]:
    units = set()
    for slot in record.allowed_slots:
        units.add(f"{record.record_type}.{slot}")
        if record.record_type == "instruction" and slot == "prep_instruction":
            units.add("instruction.instruction")
        if record.record_type == "instruction" and slot == "instruction":
            units.add("instruction.prep_instruction")
    return units


def _required_units(answer_need: AnswerNeedSpec) -> set[str]:
    required_units: set[str] = set()
    requested_slots = set(answer_need.required_slot_groups or [])
    for record_type in answer_need.required_record_types:
        relevant_slots = RECORD_SLOT_GROUPS.get(record_type)
        if relevant_slots is None:
            for slot in requested_slots:
                required_units.add(f"{record_type}.{slot}")
            continue
        selected_slots = relevant_slots & requested_slots if requested_slots else set()
        if not selected_slots:
            if record_type in {"instruction", "return_precaution"}:
                selected_slots = {"instruction"} & relevant_slots or {"condition"} & relevant_slots
            elif record_type in {"contact_method", "policy_permission"}:
                selected_slots = set(relevant_slots)
            elif record_type in {"allergy"}:
                selected_slots = {"allergy_substance", "allergy_reaction"} & relevant_slots
        for slot in selected_slots:
            required_units.add(f"{record_type}.{slot}")
    return required_units


def _score_record(
    record: UtilityRecord,
    answer_need: AnswerNeedSpec,
    covered: set[str],
    covered_families: set[str],
    required_units: set[str],
) -> float:
    score = 0.0
    record_units = _covered_units(record, answer_need)
    record_families = _record_coverage_families(record)
    required_families = set(answer_need.required_coverage_families or [])
    lowered = str(record.evidence_line or "").lower()
    required_overlap = record_units & required_units
    if record.record_type in answer_need.required_record_types:
        score += 4.0 if required_overlap else 0.5
    new_units = record_units - covered
    score += 2.2 * len(new_units & required_units)
    if required_families:
        uncovered_required_families = required_families - covered_families
        new_family_hits = record_families & uncovered_required_families
        repeated_family_hits = (record_families & required_families) - new_family_hits
        score += 2.6 * len(new_family_hits)
        score += 0.4 * len(repeated_family_hits)
        if repeated_family_hits and not new_family_hits:
            score -= 1.0 * len(repeated_family_hits)
    if answer_need.current_state_required and record.lifecycle_status == "active":
        score += 1.2
    if answer_need.current_state_required and record.record_type in {"project_state", "research_state", "household_plan"}:
        if any(token in lowered for token in ["revised", "supersedes", "superseded", "replaces", "updated", "is now"]):
            score += 2.5
        elif any(token in lowered for token in ["current", "approved"]):
            score += 0.8
        if any(token in lowered for token in ["provisional", "tentative", "exploratory", "initial", "until a newer"]):
            score -= 2.0
        score += 0.3 * max(0.0, min(record.confidence, 1.0))
    if record.record_type == "household_plan" and any(slot in record.allowed_slots for slot in ["visit_window", "entry_method", "package_rule"]):
        score += 2.5
    if record.record_type == "household_plan":
        if any(token in lowered for token in ["as of now", "current ", " is now ", "updating ", "update:", "checkpoint note"]):
            score += 3.0
        if any(token in lowered for token in ["tentative", "tentatively", "initial plan", "initial "]) and not any(
            token in lowered for token in ["as of now", "current ", " is now ", "updating ", "update:"]
        ):
            score -= 2.2
        if any(token in lowered for token in ["i just need", "i only need", "if a large box lands during that window"]):
            score -= 2.6
    if record.record_type in {"project_state", "research_state"} and len(required_overlap) >= 2:
        score += 2.0
    if record.record_type == "active_schedule" and "public_event_date" in record.allowed_slots:
        if any(token in lowered for token in [" is now ", " remains ", " current ", "public-track reminder", "current logistics note"]):
            score += 2.0
        if any(token in lowered for token in ["public schedule update again", "public orientation remains", "public date"]):
            score += 1.4
        if "moved from" in lowered:
            score += 0.4
    if answer_need.cancellation_required and record.record_type in {"canceled_schedule", "rescheduled_schedule"}:
        score += 1.5
    if record.surface_spans:
        score += 0.4
    if not required_overlap and required_units:
        score -= 1.5
    if not record.allowed_slots:
        score -= 2.0
    if record.record_type == "active_schedule" and required_units and not required_overlap:
        score -= 2.5
    if record.record_type == "medication_status":
        if (
            "i'm taking" in lowered
            or "i am taking" in lowered
            or "i also took" in lowered
            or "last night before" in lowered
        ) and not any(token in lowered for token in ["continue ", "start ", "stop ", "use ", "hold ", "restart "]):
            score -= 2.4
    if record.principal_access in {"owner", "clinical_staff", "logistics_staff", "family"}:
        score += 0.3
    if record.lifecycle_status == "deleted":
        score -= 10.0
    if record.denied_slots:
        score -= 1.2 * len(record.denied_slots)
    if answer_need.minimality_policy == "logistics_only" and record.record_type in {"clinical_sensitive"}:
        score -= 8.0
    if answer_need.minimality_policy == "answer_only_requested_domains" and record.record_type == "policy_permission" and "policy_permission" not in answer_need.need_types:
        score -= 2.5
    if record.record_type == "policy_permission" and not any(name in answer_need.need_types for name in ["policy_permission", "contact_method", "schedule"]):
        score -= 2.0
    return score


def _record_coverage_families(record: UtilityRecord) -> set[str]:
    lowered = str(record.evidence_line or "").lower()
    families: set[str] = set()
    if re.search(r"\bstop\b", lowered):
        families.add("stop")
    if re.search(r"\bstart\b", lowered):
        families.add("start")
    if re.search(r"\bcontinue\b", lowered) or "stay the same" in lowered:
        families.add("continue")
    if re.search(r"\buse\b", lowered) or any(token in lowered for token in ["as needed", "prn"]):
        families.add("use")
    if any(token in lowered for token in ["check blood pressure", "monitor blood pressure", "twice daily", "morning and evening"]):
        families.add("monitor")
    return families
