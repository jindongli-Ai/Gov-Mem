from __future__ import annotations

import re

from gov_mem.data.schema import QueryPlan, RetrievedEvidence
from gov_mem.governance_runtime.access import is_policy_frame
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.query_semantics import infer_current_state_slots, infer_household_slots


def apply_not_filter(plan: QueryPlan, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    if "NOT" not in plan.reasoning_ops:
        return evidence
    filtered: list[RetrievedEvidence] = []
    for row in evidence:
        if plan.target_users and not _matches_target_user_or_entity(plan, row):
            continue
        filtered.append(row)
    return filtered


def apply_temporal_order(plan: QueryPlan, evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    if "temporal_order" not in plan.reasoning_ops:
        return evidence
    ranked = sorted(evidence, key=lambda row: (_temporal_priority(row), row.time or ""), reverse=True)
    return _dedupe_superseded_schedule_rows(ranked)


def detect_conflicts(evidence: list[RetrievedEvidence]) -> list[dict]:
    conflicts: list[dict] = []
    seen_by_key: dict[tuple[str | None, str | None], RetrievedEvidence] = {}
    for row in evidence:
        key = (row.user_id, row.memory_type)
        previous = seen_by_key.get(key)
        if previous and previous.content != row.content:
            conflicts.append(
                {
                    "memory_id_a": previous.memory_id,
                    "memory_id_b": row.memory_id,
                    "user_id": row.user_id,
                    "memory_type": row.memory_type,
                }
            )
        else:
            seen_by_key[key] = row
    return conflicts


def apply_compare(evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    return evidence


def _matches_target_user_or_entity(plan: QueryPlan, row: RetrievedEvidence) -> bool:
    target_users = {
        str(value).strip().lower()
        for value in (plan.target_users or [])
        if str(value or "").strip()
    }
    if not target_users:
        return True

    row_user = str(getattr(row, "user_id", "") or "").strip().lower()
    if row_user and row_user in target_users:
        return True

    row_entities = [str(value or "").strip().lower() for value in (getattr(row, "entities", None) or [])]
    haystack = " ".join(
        [
            str(getattr(row, "content", "") or "").lower(),
            " ".join(row_entities),
            str(getattr(row, "memory_type", "") or "").lower(),
            str(getattr(row, "scope", "") or "").lower(),
        ]
    )
    normalized_haystack = haystack.replace("_", " ")
    for target in target_users:
        normalized_target = target.replace("_", " ")
        if normalized_target and normalized_target in normalized_haystack:
            return True
        target_tokens = [token for token in normalized_target.split() if len(token) >= 3]
        if target_tokens and all(token in normalized_haystack for token in target_tokens):
            return True
    return False


def build_required_slot_plan(question: str, query_plan: QueryPlan, action_decision=None) -> dict:
    lowered = (question or "").lower()
    target_frame_types = ["general_fact"]
    required_slots: list[str] = []
    optional_slots: list[str] = []
    temporal_policy = "current_state"
    mixed = False
    domains: list[str] = []

    semantic_spec = dict(getattr(query_plan, "semantic_spec", {}) or {})
    certifiable_needs = semantic_spec.get("certifiable_needs")
    certified_attributes = [
        str(item.get("attribute") or item.get("need_id") or "")
        for item in certifiable_needs
        if isinstance(item, dict) and str(item.get("attribute") or item.get("need_id") or "")
    ] if isinstance(certifiable_needs, list) else []
    open_attributes = [
        str(attribute)
        for attribute in (certified_attributes or list(semantic_spec.get("requested_attributes") or []))
        if str(attribute)
    ]
    requested_attributes = certified_attributes or list(semantic_spec.get("requested_attributes") or [])
    semantic_slots = [
        str(slot)
        for slot in (requested_attributes or list(semantic_spec.get("requested_slots") or []))
        if str(slot)
    ]
    current_slot_names = {
        "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
        "monthly_stipend", "safe_wording", "blocker", "access_room", "access_badge",
        "operational_result",
        "contract_structure", "selected_vendor", "family_release_scope",
        "public_room", "coordination_label", "access_token",
    }
    household_slot_names = {
        "date", "visit_window", "entry_method", "package_rule", "approved_areas",
        "parking_pass", "arrival_contact_rule",
    }
    security_slot_names = {
        "phone", "contact_method", "backup_contact", "consent_scope", "policy_scope",
    }
    requested_current_slots = [slot for slot in semantic_slots if slot in current_slot_names]
    if not requested_current_slots:
        requested_current_slots = infer_current_state_slots(question)
    clinical_plan = _infer_clinical_plan_requirements(lowered)
    household_slots = [slot for slot in semantic_slots if slot in household_slot_names]
    if not household_slots:
        household_slots = infer_household_slots(question)
    security_slots = [slot for slot in semantic_slots if slot in security_slot_names]
    office_incident_scope_query = (
        any(token in lowered for token in ["current diagnosis", "leading diagnosis", "diagnosis for the", "diagnosis"])
        and any(token in lowered for token in ["access scope", "debugging contractor", "logs-only", "what access"])
    )
    if security_slots:
        domains.append("security_contact")
        target_frame_types = ["contact_method", "instruction", "consent_or_permission", "privacy_policy", "general_fact"]
        required_slots.extend(security_slots)
        optional_slots.extend(["provider", "status"])
    elif office_incident_scope_query:
        domains.extend(["incident_state", "policy_scope"])
        target_frame_types = ["project_state", "diagnosis_or_result", "consent_or_permission", "general_fact", "update"]
        required_slots.extend(["operational_result", "consent_scope"])
        optional_slots.extend(["status"])
    if requested_current_slots:
        domains.append("current_state")
        target_frame_types = _target_frame_types_for_current_slots(requested_current_slots)
        required_slots.extend(requested_current_slots)
        optional_slots.extend(["date", "instruction"])
    elif household_slots:
        domains.append("household_plan")
        target_frame_types = ["household_plan", "general_fact", "logistics", "update"]
        optional_slots.extend(["date", "visit_window", "entry_method", "parking_pass", "arrival_contact_rule", "approved_areas", "package_rule", "location", "status"])
        required_slots.extend(household_slots or ["date", "visit_window"])
    elif clinical_plan:
        domains.extend(clinical_plan.get("domains", []))
        target_frame_types = list(clinical_plan.get("target_frame_types", target_frame_types))
        required_slots.extend(clinical_plan.get("required_slots", []))
        optional_slots.extend(clinical_plan.get("optional_slots", []))
    elif any(token in lowered for token in ["appointment", "schedule", "visit", "follow-up", "follow up", "imaging", "ultrasound"]):
        domains.append("schedule")
        target_frame_types = ["test_or_imaging", "clinic_visit", "instruction", "cancellation", "update", "appointment", "logistics"]
        required_slots.extend(["date", "time", "arrival_time", "location", "provider", "procedure", "visit_type", "status", "condition"])
        optional_slots.extend(["precondition", "replacement_event", "canceled_event"])
    elif any(token in lowered for token in ["allergy", "reaction"]):
        domains.append("allergy")
        target_frame_types = ["allergy", "test_or_imaging", "clinic_visit"]
        required_slots.extend(["substance", "reaction"])
        optional_slots.extend(["date", "time", "arrival_time", "location", "provider", "procedure"])
    elif any(token in lowered for token in ["medication", "pain", "nausea"]):
        domains.append("medication")
        target_frame_types = ["medication", "instruction"]
        required_slots.extend(["medication", "instruction", "timing"])
        optional_slots.extend(["prep_instruction"])
    elif any(token in lowered for token in ["permission", "authorized", "family-access", "family access", "share"]):
        domains.append("policy")
        target_frame_types = ["consent_or_permission", "privacy_policy", "logistics"]
        required_slots.extend(["consent_scope", "status"])
        optional_slots.extend(["date", "time", "location", "procedure"])
    elif any(token in lowered for token in ["callback", "voicemail", "portal"]):
        domains.append("logistics")
        target_frame_types = ["logistics", "instruction"]
        required_slots.extend(["prep_instruction"])
    else:
        if query_plan.target_entities:
            required_slots.extend(["status"])
            optional_slots.extend(["date", "time", "location"])

    if len(domains) > 1:
        mixed = True
    if any(token in lowered for token in ["and", "both", ",", "mixed", "also"]):
        mixed = True
    if mixed:
        target_frame_types = list(dict.fromkeys(target_frame_types + ["allergy", "medication", "test_or_imaging", "clinic_visit", "instruction", "cancellation", "logistics"]))

    required_slots = list(dict.fromkeys(required_slots))
    required_slots = list(dict.fromkeys(required_slots + open_attributes))
    optional_slots = list(dict.fromkeys(optional_slots))

    return {
        "target_frame_types": target_frame_types,
        "required_slots": required_slots,
        "optional_slots": optional_slots,
        "target_entities": list(dict.fromkeys(str(value).strip() for value in (query_plan.target_entities or []) if str(value).strip())),
        "temporal_policy": temporal_policy,
        "mixed": mixed,
        "domains": domains,
    }


def _temporal_priority(row: RetrievedEvidence) -> tuple[int, int, int]:
    lifecycle = str(getattr(row, "lifecycle_status", "") or "").lower()
    metadata = getattr(row, "metadata", {}) or {}
    content = str(getattr(row, "content", "") or "").lower()

    active_rank = 0
    if lifecycle == "active":
        active_rank = 4
    elif lifecycle == "canceled":
        active_rank = 3
    elif lifecycle in {"superseded", "deleted"}:
        active_rank = 1
    elif lifecycle:
        active_rank = 2

    current_rank = 0
    if any(token in content for token in ["current", "official", "updated", "supersedes", "replaces"]):
        current_rank += 2
    if bool(metadata.get("is_current")):
        current_rank += 2
    if bool(metadata.get("superseded_by")) or bool(metadata.get("is_superseded")):
        current_rank -= 2

    score_rank = int(min(max(float(getattr(row, "score", 0.0) or 0.0), 0.0), 1.0) * 1000)
    return active_rank, current_rank, score_rank


def _dedupe_superseded_schedule_rows(evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    selected: list[RetrievedEvidence] = []
    seen_schedule_keys: set[tuple[str, str, str, str, str]] = set()
    for row in evidence:
        try:
            frame = compile_evidence_frame(row)
        except Exception:
            frame = None
        if frame is None or getattr(frame, "frame_type", "") not in {"test_or_imaging", "clinic_visit", "appointment"}:
            selected.append(row)
            continue
        slots = getattr(frame, "slots", {}) or {}
        key = (
            str(getattr(row, "user_id", "") or ""),
            str(slots.get("procedure") or ""),
            str(slots.get("visit_type") or ""),
            str(slots.get("date") or ""),
            str(slots.get("time") or slots.get("arrival_time") or ""),
        )
        if key in seen_schedule_keys:
            continue
        seen_schedule_keys.add(key)
        selected.append(row)
    return selected
def _infer_clinical_plan_requirements(lowered: str) -> dict:
    current_cues = [
        "current",
        "currently",
        "right now",
        "now",
        "latest",
        "updated",
        "still needs",
        "from here",
        "as of now",
    ]
    regimen_cues = [
        "treatment plan",
        "regimen",
        "medication plan",
        "which medications",
        "what medications",
        "what should i take",
        "what should i use",
        "home blood pressure plan",
        "blood pressure plan",
    ]
    prep_cues = [
        "prep plan",
        "preparation",
        "prep",
        "before the procedure",
        "before procedure",
        "before biopsy",
        "biopsy-prep",
        "pre-op",
        "pre op",
    ]
    followup_cues = [
        "follow-up plan",
        "follow up plan",
        "next step",
        "what happens next",
        "return precaution",
        "return precautions",
    ]
    schedule_entities = ["biopsy", "procedure", "scan", "ultrasound", "visit", "appointment", "injection", "follow-up", "follow up"]
    asks_current_plan = any(token in lowered for token in current_cues) and any(token in lowered for token in ["plan", "treatment", "prep", "follow-up", "follow up", "regimen"])
    asks_regimen = any(token in lowered for token in regimen_cues)
    asks_prep = any(token in lowered for token in prep_cues)
    asks_followup = any(token in lowered for token in followup_cues)
    if not (asks_current_plan or asks_regimen or asks_prep or asks_followup):
        return {}

    domains: list[str] = []
    target_frame_types: list[str] = []
    required_slots: list[str] = []
    optional_slots: list[str] = []

    if asks_regimen:
        domains.append("medication_plan")
        target_frame_types.extend(["medication", "instruction"])
        required_slots.extend(["medication", "instruction"])
        optional_slots.extend(["timing", "condition", "prep_instruction"])

    if asks_prep:
        domains.append("procedure_prep")
        target_frame_types.extend(["instruction", "test_or_imaging", "clinic_visit", "appointment", "logistics"])
        required_slots.append("prep_instruction")
        optional_slots.extend(["instruction", "condition", "date", "time", "arrival_time", "location", "provider", "procedure"])

    if asks_followup:
        domains.append("followup_plan")
        target_frame_types.extend(["instruction", "clinic_visit", "test_or_imaging", "appointment"])
        required_slots.extend(["instruction", "condition"])
        optional_slots.extend(["date", "time", "location", "provider", "procedure", "status"])

    if asks_current_plan and not (asks_regimen or asks_prep or asks_followup):
        target_frame_types.extend(["instruction", "medication"])
        required_slots.extend(["instruction"])
        optional_slots.extend(["condition", "timing", "medication"])

    if any(token in lowered for token in schedule_entities):
        optional_slots.extend(["date", "time", "arrival_time", "location", "provider", "procedure", "visit_type", "status"])

    return {
        "domains": list(dict.fromkeys(domains or ["clinical_plan"])),
        "target_frame_types": list(dict.fromkeys(target_frame_types or ["instruction", "medication"])),
        "required_slots": list(dict.fromkeys(required_slots)),
        "optional_slots": list(dict.fromkeys(optional_slots)),
    }


def _target_frame_types_for_current_slots(requested_current_slots: list[str]) -> list[str]:
    requested = set(requested_current_slots or [])
    frame_types: list[str] = []
    if requested & {"approved_budget", "approved_discount_cap"}:
        frame_types.append("project_state")
    if requested & {"contract_structure", "selected_vendor", "coordination_label", "access_token"}:
        frame_types.append("project_state")
    if requested & {"monthly_stipend", "safe_wording"}:
        frame_types.append("research_state")
    if requested & {"family_release_scope", "public_room"}:
        frame_types.append("research_state")
    if "public_event_date" in requested:
        frame_types.extend(["appointment", "update", "general_fact", "logistics"])
    if requested & {"date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}:
        frame_types.extend(["household_plan", "general_fact", "logistics", "update"])
    if "target_date" in requested:
        if requested & {"approved_budget", "approved_discount_cap"}:
            frame_types.append("project_state")
        elif requested & {"monthly_stipend", "safe_wording"}:
            frame_types.append("research_state")
        elif requested & {"date", "visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}:
            frame_types.extend(["household_plan", "general_fact", "logistics", "update"])
        else:
            frame_types.extend(["project_state", "research_state"])
    if "blocker" in requested and not frame_types:
        frame_types.extend(["project_state", "research_state"])
    if not frame_types:
        frame_types.append("general_fact")
    return list(dict.fromkeys(frame_types))


def _is_current_state_plan(required_slot_plan: dict) -> bool:
    if "current_state" in set(required_slot_plan.get("domains") or []):
        return True
    target_frame_types = set(required_slot_plan.get("target_frame_types") or [])
    return bool(target_frame_types & {"project_state", "research_state", "household_plan"})


def select_evidence_by_slot_coverage(
    *,
    frames,
    evidence: list[RetrievedEvidence],
    required_slot_plan: dict,
    requester,
    config: dict,
) -> tuple[list[RetrievedEvidence], list, dict]:
    max_selected = int((config.get("reasoning") or {}).get("max_selected_frames", 12))
    required_slots = list(required_slot_plan.get("required_slots", []))
    covered_slots: set[str] = set()
    covered_required_keys: set[str] = set()
    selected_frames = []
    selected_evidence: list[RetrievedEvidence] = []
    remaining = list(zip(frames, evidence))
    latest_slot_times = _latest_aligned_slot_times(remaining, required_slots, required_slot_plan)
    while remaining and len(selected_frames) < max_selected:
        best_idx = None
        best_gain = float("-inf")
        for idx, (frame, row) in enumerate(remaining):
            gain = _frame_selection_gain(
                frame, row, required_slots, covered_required_keys, required_slot_plan, latest_slot_times
            )
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None or best_gain <= 0:
            break
        frame, row = remaining.pop(best_idx)
        selected_frames.append(frame)
        selected_evidence.append(row)
        covered_slots.update(_covered_slot_labels(frame, row, required_slot_plan))
        covered_required_keys.update(_covered_required_slot_keys(frame, row, required_slot_plan))
        if _coverage_ratio(required_slots, covered_required_keys) >= float((config.get("reasoning") or {}).get("min_slot_coverage", 0.70)):
            if not bool((config.get("reasoning") or {}).get("preserve_complementary_evidence", True)):
                break

    if not selected_frames and remaining:
        frame, row = remaining.pop(0)
        selected_frames.append(frame)
        selected_evidence.append(row)
        covered_slots.update(_covered_slot_labels(frame, row, required_slot_plan))
        covered_required_keys.update(_covered_required_slot_keys(frame, row, required_slot_plan))

    missing_slots = [slot for slot in required_slots if slot not in covered_required_keys]
    return selected_evidence, selected_frames, {
        "required_slots": required_slots,
        "covered_slots": sorted(covered_slots),
        "missing_slots": missing_slots,
        "coverage_ratio": _coverage_ratio(required_slots, covered_required_keys),
    }


def _frame_selection_gain(
    frame,
    row: RetrievedEvidence,
    required_slots: list[str],
    covered_required_keys: set[str],
    required_slot_plan: dict,
    latest_slot_times: dict[str, str] | None = None,
) -> float:
    gain = 0.0
    missing = {slot for slot in required_slots if slot not in covered_required_keys}
    coverage_quality = _slot_coverage_quality(frame, row, required_slot_plan)
    for slot in missing:
        quality = coverage_quality.get(slot)
        if quality is not None:
            gain += 1.25 * quality
    if frame.frame_type in set(required_slot_plan.get("target_frame_types", [])):
        gain += 0.5
    if _is_current_state_plan(required_slot_plan):
        gain += _current_state_authority_bonus(
            frame, row, missing, required_slot_plan, latest_slot_times or {}
        )
        if bool((row.metadata or {}).get("redaction_required") or (row.metadata or {}).get("requires_redaction")):
            gain -= 1.8
    if required_slot_plan.get("mixed") and frame.frame_type in {"allergy", "medication", "test_or_imaging", "clinic_visit", "instruction", "cancellation", "logistics", "project_state", "research_state", "household_plan"}:
        gain += 0.4
    if is_policy_frame(frame):
        gain -= 0.6
    if frame.lifecycle_status == "active":
        gain += 0.3
    if frame.lifecycle_status == "canceled":
        gain += 0.15
    if frame.effective_time:
        gain += 0.1
    gain += min(float(row.score), 1.0)
    if frame.lifecycle_status in {"deleted", "superseded"}:
        gain -= 2.0
    return gain


def _covered_required_slot_keys(frame, row: RetrievedEvidence, required_slot_plan: dict) -> set[str]:
    coverage_quality = _slot_coverage_quality(frame, row, required_slot_plan)
    return {slot for slot, quality in coverage_quality.items() if quality >= 0.55}


def _covered_slot_labels(frame, row: RetrievedEvidence, required_slot_plan: dict) -> set[str]:
    labels = {
        f"{frame.frame_type}.{slot}"
        for slot in frame.slots.keys()
        if slot in set(required_slot_plan.get("required_slots") or [])
    }
    for slot in _covered_required_slot_keys(frame, row, required_slot_plan):
        if slot not in frame.slots:
            labels.add(f"{frame.frame_type}.{slot}:semantic")
    return labels


def _slot_coverage_quality(frame, row: RetrievedEvidence, required_slot_plan: dict) -> dict[str, float]:
    required_slots = list(required_slot_plan.get("required_slots") or [])
    if not required_slots:
        return {}
    frame_slots = dict(getattr(frame, "slots", {}) or {})
    lowered = str(getattr(row, "content", "") or "").lower()
    quality: dict[str, float] = {}
    for slot in required_slots:
        candidates = _slot_equivalent_candidates(slot, frame_slots)
        if not candidates:
            continue
        best = 0.0
        for candidate_slot, value in candidates:
            candidate_quality = _slot_value_quality(
                required_slot=slot,
                frame_type=str(getattr(frame, "frame_type", "") or ""),
                candidate_slot=candidate_slot,
                value=value,
                lowered_text=lowered,
            )
            best = max(best, candidate_quality)
        if best > 0:
            quality[slot] = best
    return quality


def _slot_equivalent_candidates(required_slot: str, frame_slots: dict[str, object]) -> list[tuple[str, str]]:
    equivalents = {
        "prep_instruction": ["prep_instruction", "instruction"],
        "instruction": ["instruction", "prep_instruction"],
        "medication": ["medication", "instruction"],
        "blocker": ["blocker", "status"],
        "arrival_contact_rule": ["arrival_contact_rule", "instruction"],
    }
    candidate_keys = equivalents.get(required_slot, [required_slot])
    out: list[tuple[str, str]] = []
    for key in candidate_keys:
        value = str(frame_slots.get(key) or "").strip()
        if value:
            out.append((key, value))
    return out


def _slot_value_quality(
    *,
    required_slot: str,
    frame_type: str,
    candidate_slot: str,
    value: str,
    lowered_text: str,
) -> float:
    if required_slot == "arrival_contact_rule":
        lowered = value.lower()
        if candidate_slot == "arrival_contact_rule":
            return 1.0
        if candidate_slot == "instruction" and any(
            token in lowered
            for token in [
                "text on arrival",
                "call on arrival",
                "arrival confirmation",
                "arrival-contact",
                "arrival contact",
                "buzz from the front desk",
                "contact on arrival",
                "check in on arrival",
            ]
        ):
            return 0.92
        return 0.0
    if required_slot != "prep_instruction":
        return 1.0
    lowered = value.lower()
    strong_operational_patterns = [
        r"\bcontinue\b",
        r"\bstop\b",
        r"\bhold\b",
        r"\bavoid\b",
        r"\barrive\b",
        r"\btake\b",
        r"\brestart\b",
        r"\bskip\b",
        r"\bbefore\b",
        r"nothing by mouth",
        r"after midnight",
        r"check-?in",
        r"\bconsent\b",
    ]
    weak_markers = [
        "intake questions",
        "vitals",
        "move quickly",
        "prep instructions",
        "callback plan",
        "scheduled quickly",
    ]
    operational_hit_flags = {
        "continue": bool(re.search(r"\bcontinue\b", lowered)),
        "stop_or_hold": bool(re.search(r"\b(stop|hold|avoid|skip|restart)\b", lowered)),
        "take": bool(re.search(r"\btake\b", lowered)),
        "timing": bool(re.search(r"\b(before|after)\b", lowered) or "after midnight" in lowered or "day before" in lowered),
        "arrival": bool(re.search(r"\barrive\b", lowered) or re.search(r"check-?in", lowered) or re.search(r"\bconsent\b", lowered)),
        "fasting": "nothing by mouth" in lowered,
    }
    operational_hits = sum(1 for matched in operational_hit_flags.values() if matched)
    has_strong_operational_signal = operational_hits > 0
    if candidate_slot == "prep_instruction" and has_strong_operational_signal:
        quality = 0.82
        quality += 0.12 * min(operational_hits, 4)
        if operational_hit_flags["arrival"]:
            quality += 0.16
        if operational_hit_flags["timing"]:
            quality += 0.08
        return quality
    if candidate_slot == "instruction" and has_strong_operational_signal:
        quality = 0.68 if frame_type == "medication" else 0.56
        quality += 0.11 * min(operational_hits, 4)
        if operational_hit_flags["arrival"]:
            quality += 0.14
        if operational_hit_flags["timing"]:
            quality += 0.07
        return quality
    if candidate_slot == "prep_instruction" and any(marker in lowered for marker in weak_markers):
        return 0.25
    if candidate_slot == "instruction" and any(marker in lowered for marker in weak_markers):
        return 0.2
    if re.search(r"\b(continue|stop|hold|avoid|arrive|take|restart|skip)\b", lowered_text):
        return 1.0
    return 0.85 if candidate_slot == "prep_instruction" else 0.6


def _current_state_authority_bonus(
    frame,
    row: RetrievedEvidence,
    missing_slots: set[str],
    required_slot_plan: dict,
    latest_slot_times: dict[str, str],
) -> float:
    lowered = str(row.content or "").lower()
    bonus = 0.0
    if frame.frame_type in {"project_state", "research_state", "household_plan"}:
        if any(token in lowered for token in ["official", "confirmed", "approved", "current", "latest", "summary"]):
            bonus += 2.2
        if any(token in lowered for token in ["supersedes", "superseded", "replaces the earlier", "that replaces the earlier", "revised approved", "is now"]):
            bonus += 1.8
        if any(token in lowered for token in ["updated", "effective", "current target", "current entry", "arrival instruction"]):
            bonus += 1.5
        if any(token in lowered for token in ["provisional", "interim", "until a newer decision supersedes it", "for now"]) and not any(
            token in lowered for token in ["official", "finance-confirmed", "current right now", "supersedes", "revised approved"]
        ):
            bonus -= 1.8
    alignment = _entity_alignment(frame, row, required_slot_plan.get("target_entities") or [])
    bonus += 1.4 * alignment
    row_time = _normalized_timestamp(row.time or frame.effective_time)
    covered_missing = set(_slot_coverage_quality(frame, row, required_slot_plan)) & missing_slots
    for slot in covered_missing:
        latest_time = latest_slot_times.get(slot)
        if row_time and latest_time:
            bonus += 1.25 if row_time == latest_time else -0.75
    return bonus


def _latest_aligned_slot_times(candidates, required_slots: list[str], required_slot_plan: dict) -> dict[str, str]:
    latest: dict[str, str] = {}
    target_entities = required_slot_plan.get("target_entities") or []
    aligned = [pair for pair in candidates if _entity_alignment(pair[0], pair[1], target_entities) >= 0]
    pool = aligned or list(candidates)
    for frame, row in pool:
        if str(getattr(frame, "lifecycle_status", "") or "active").lower() != "active":
            continue
        timestamp = _normalized_timestamp(row.time or frame.effective_time)
        if not timestamp:
            continue
        covered = set(_slot_coverage_quality(frame, row, required_slot_plan)) & set(required_slots)
        for slot in covered:
            if timestamp > latest.get(slot, ""):
                latest[slot] = timestamp
    return latest


def _entity_alignment(frame, row: RetrievedEvidence, target_entities: list[str]) -> float:
    target_sets = [_entity_tokens(value) for value in target_entities]
    target_sets = [tokens for tokens in target_sets if tokens]
    if not target_sets:
        return 0.0
    candidate_values = list(getattr(row, "entities", None) or []) + [
        getattr(frame, "subject_entity", None), getattr(row, "content", None)
    ]
    candidate_tokens = set().union(*(_entity_tokens(value) for value in candidate_values))
    return 1.0 if any(tokens <= candidate_tokens for tokens in target_sets) else -1.0


def _entity_tokens(value) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _normalized_timestamp(value) -> str:
    # ISO-like timestamps compare correctly after retaining all date/time digits.
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _coverage_ratio(required_slots: list[str], covered_slots: set[str]) -> float:
    if not required_slots:
        return 1.0
    return len([slot for slot in required_slots if any(slot == covered.split(".", 1)[-1] for covered in covered_slots)]) / float(len(required_slots))
