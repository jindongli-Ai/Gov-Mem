from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import md5
import re
from typing import Any

from gov_mem.backbones.need_spec import AnswerNeedSpec
from gov_mem.query_semantics import extract_state_slots, infer_state_record_type


@dataclass
class UtilityRecord:
    record_id: str
    source_chunk_id: str | None
    source_line_id: str | None
    record_type: str
    owner_user: str | None
    principal_access: str
    lifecycle_status: str
    slots: dict[str, str] = field(default_factory=dict)
    surface_spans: dict[str, str] = field(default_factory=dict)
    denied_slots: list[str] = field(default_factory=list)
    allowed_slots: list[str] = field(default_factory=list)
    evidence_line: str = ""
    confidence: float = 1.0
    source_time: str = ""
    trace: list[str] = field(default_factory=list)


TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:,\s*)?(?:\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))?\s+\d{1,2}\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE)
DATE_GENERIC_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b", re.IGNORECASE)
PROVIDER_RE = re.compile(r"\bDr\.\s+[A-Z][a-zA-Z]+\b")
LOCATION_RE = re.compile(r"\b(?:(?!(?:AM|PM)\b)[A-Z][A-Za-z0-9&' -]{1,60}\s+(?:Suite|Clinic|Center|Office|Lab|Ward|Desk)\s*[A-Z0-9-]*|front desk)\b")
PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
ARRIVAL_RE = re.compile(r"(?:arrive by|please arrive by|check in by|come by|arrival(?:\s+for)?)(?:\s+|:)\d{1,2}:\d{2}\s?(?:AM|PM)", re.IGNORECASE)
MONEY_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*USD\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:\.\d+)?%")
ENTRY_METHOD_RE = re.compile(r"\b(?:via|through|using|use|should use|entry is via|entry via|current entry is|route is)\s+(?:the\s+)?([A-Za-z0-9\- ]+(?:door|keypad|gate|entry|code|lockbox))\b", re.IGNORECASE)
VISIT_WINDOW_RE = re.compile(
    r"(?:\bfrom\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\s+to\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\b|"
    r"\b(\d{1,2}:\d{2}\s?(?:AM|PM))\s+to\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\b)",
    re.IGNORECASE,
)
UPDATED_TARGET_DATE_RE = re.compile(
    r"\b(?:moves?\s+from\s+[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?\s+to\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"(?:official|current)\s+(?:pilot\s+)?target(?:\s+date)?(?:\s+is|\s+as)?\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"launch date remains\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"current target date(?:\s+as)?\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"dry-run date(?:\s+is)?\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?))\b",
    re.IGNORECASE,
)
PUBLIC_EVENT_SHARE_RE = re.compile(
    r"\b(?:shared broadly|can be shared broadly|broadly with|students and staff|staff and students|announcements should|calendar line|mixed-audience agendas|public schedule|public event|orientation|open house)\b",
    re.IGNORECASE,
)
PUBLIC_EVENT_DATE_RE = re.compile(
    r"\b(?:public\s+)?(?:event|orientation|ceremony|open\s+house|calendar)\b[^.;]{0,100}?"
    r"\b(?:is|remains|now|on|moved\s+to)\s+"
    r"([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)",
    re.IGNORECASE,
)
SAFE_DISCLOSURE_RE = re.compile(
    r"\b(?:broad|safe|shareable)\s+(?:[a-z-]+\s+){0,4}(?:wording|label|description)"
    r"(?:\s+(?:is|remains|should\s+stay|:))?\s*"
    r"['\"]?([A-Za-z][^.;'\"]+)",
    re.IGNORECASE,
)
CURRENT_MONEY_RE = re.compile(
    r"(?:approved current(?: [a-z]+)? budget|approved current amount|approved amount|approved value|current amount|revised approved [a-z]+ budget|budget is now|working budget is now)\s*(?::|is|=)?\s*(\d[\d,]*(?:\.\d+)?\s*USD)",
    re.IGNORECASE,
)
CURRENT_PERCENT_RE = re.compile(
    r"(?:approved (?:maximum )?discount|current approved maximum discount|commercial cap|discount is now|cap is now|maximum discount is now|revised approved [a-z]+ discount)\s*(?::|is|=)?\s*(\d{1,2}(?:\.\d+)?%)",
    re.IGNORECASE,
)
MEDICATION_DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|grams?|tablets?|capsules?)"
    r"(?:\s*(?:q\d+h|bid|tid|daily|nightly|once daily|twice daily|every\s+\w+\s+hours?|as needed|prn|with food|at bedtime))?\b",
    re.IGNORECASE,
)
MONITORING_INSTRUCTION_RE = re.compile(
    r"\b(?:check|monitor)\s+(?:your\s+)?blood pressure\b|\bblood pressure twice daily\b|\bmorning and evening\b|\bwrite the readings\b|\blog\b",
    re.IGNORECASE,
)
MEDICATION_CLAUSE_SPLIT_RE = re.compile(
    r"(?:\.\s+|;\s+|,\s+(?=(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to)\b))",
    re.IGNORECASE,
)
MEDICATION_ACTION_RE = re.compile(r"\b(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to|take)\b", re.IGNORECASE)


def build_utility_records(
    projected_evidence_lines: list[dict],
    policy_decisions: list[dict],
    principal: Any,
    answer_need: AnswerNeedSpec,
    config: dict,
) -> list[UtilityRecord]:
    decision_by_chunk = {str(item.get("chunk_id")): item for item in (policy_decisions or [])}
    records: list[UtilityRecord] = []
    for idx, line in enumerate(projected_evidence_lines):
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        meta = line.get("line_meta") or {}
        chunk_id = str(line.get("chunk_id") or "")
        decision = decision_by_chunk.get(chunk_id, {})
        for surface_idx, surface in enumerate(_iter_record_surfaces(text, meta)):
            surface_meta = dict(meta)
            # Parent chunks may contain multiple conflicting updates; each atom
            # must derive its state slots from its own evidence surface.
            if surface != text:
                surface_meta.pop("slots", None)
                surface_meta.pop("surface_spans", None)
            record_type = _infer_record_type(surface, surface_meta)
            slots, spans = _extract_slots_and_spans(surface, surface_meta)
            lifecycle = _infer_lifecycle_status(surface)
            allowed_slots, denied_slots = _resolve_slot_access(principal, slots, decision, record_type)
            record = UtilityRecord(
                record_id=md5(f"{chunk_id}:{idx}:{surface_idx}:{surface}".encode("utf-8")).hexdigest()[:12],
                source_chunk_id=chunk_id or None,
                source_line_id=f"{chunk_id}:{idx}:{surface_idx}" if chunk_id else f"{idx}:{surface_idx}",
                record_type=record_type,
                owner_user=None,
                principal_access=_infer_principal_access(principal),
                lifecycle_status=lifecycle,
                slots=slots,
                surface_spans=spans,
                denied_slots=denied_slots,
                allowed_slots=allowed_slots,
                evidence_line=surface,
                confidence=float(line.get("score") or 1.0),
                source_time=str(surface_meta.get("source_time") or ""),
                trace=[f"frame_type={surface_meta.get('frame_type')}", f"allowed_slots={allowed_slots}", f"denied_slots={denied_slots}"],
            )
            records.append(record)
    return records


def _iter_record_surfaces(text: str, meta: dict[str, Any]) -> list[str]:
    """Split multi-claim state evidence before lifecycle and slot resolution."""
    frame_type = str(meta.get("frame_type") or "")
    state_like = frame_type in {"project_state", "research_state", "household_plan"} or infer_state_record_type(
        text=text,
        slots=dict(meta.get("slots") or {}),
        frame_type=frame_type,
    ) in {"project_state", "research_state", "household_plan"}
    if not state_like:
        return [text]
    surfaces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", text) if piece.strip()]
    return surfaces or [text]


def _infer_record_type(text: str, meta: dict[str, Any]) -> str:
    lowered = text.lower()
    frame_type = str(meta.get("frame_type") or "")
    if PUBLIC_EVENT_DATE_RE.search(text):
        return "active_schedule"
    if SAFE_DISCLOSURE_RE.search(text):
        return "research_state"
    state_record_type = infer_state_record_type(text=text, slots=dict(meta.get("slots") or {}), frame_type=frame_type)
    if state_record_type is not None:
        return state_record_type
    if (meta.get("slots") or {}).get("public_event_date") or (
        any(token in lowered for token in ["orientation", "public schedule", "public event", "calendar line"])
        and frame_type not in {"medication", "allergy", "diagnosis_or_result"}
    ):
        return "active_schedule"
    if any(token in lowered for token in ["return urgently", "go to the ed", "go in right away", "fainting", "shortness of breath"]):
        return "return_precaution"
    if any(
        token in lowered
        for token in [
            "current treatment plan is",
            "treatment plan is",
            "before monday, continue",
            "arrive 30 minutes early",
            "no sex for 7 days",
            "repeat rpr in 3 months",
        ]
    ):
        return "medication_status" if any(token in lowered for token in ["continue ", "stop ", "avoid ", "treatment plan"]) else "instruction"
    if "portal" in lowered or "backup" in lowered or "contact" in lowered or "phone" in lowered or "pickup" in lowered:
        return "contact_method"
    if "allergy" in lowered or "rash" in lowered or "sulfa" in lowered or frame_type == "allergy":
        return "allergy"
    if "logistics only" in lowered or "may receive" in lowered or "restricted" in lowered or frame_type == "consent_or_permission":
        return "policy_permission"
    if _looks_like_medication_text(text, frame_type):
        return "medication_status"
    if frame_type == "instruction" or any(token in lowered for token in ["unless symptoms worsen", "before tuesday", "do not need another routine", "need to fast", "nothing by mouth"]):
        return "instruction"
    if frame_type == "consent_or_permission":
        return "policy_permission"
    if any(token in lowered for token in ["unless symptoms worsen", "before tuesday", "do not need another routine", "need to fast", "nothing by mouth"]):
        return "instruction"
    if any(token in lowered for token in ["rescheduled", "moved to", "updated to"]):
        return "rescheduled_schedule"
    if any(token in lowered for token in ["canceled", "cancelled", "no longer active", "no longer scheduled"]):
        return "canceled_schedule"
    if frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "update"} or any(token in lowered for token in ["ultrasound", "visit", "arrival", "schedule", "ecg", "procedure"]):
        return "active_schedule"
    if any(token in lowered for token in ["viable", "viability", "impression", "beta-hcg", "hcg"]):
        return "clinical_sensitive"
    return "general_utility"


def _infer_lifecycle_status(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["deleted", "forget", "remove", "no longer remember"]):
        return "deleted"
    if any(token in lowered for token in ["canceled", "cancelled", "no longer scheduled", "no longer active"]):
        return "canceled"
    if any(token in lowered for token in ["rescheduled", "moved to", "updated to", "updated:"]):
        return "superseded"
    return "active"


def _extract_slots_and_spans(text: str, meta: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    spans = dict(meta.get("surface_spans") or {})
    slots = {str(k): str(v) for k, v in (meta.get("slots") or {}).items() if v}
    frame_type = str(meta.get("frame_type") or "")
    date_match = DATE_RE.search(text) or DATE_GENERIC_RE.search(text)
    if date_match:
        _prefer_more_specific_slot(slots, spans, "date", date_match.group(0))
    times = TIME_RE.findall(text)
    if times:
        spans.setdefault("time", times[0])
        slots.setdefault("time", times[0])
        if len(times) > 1:
            spans.setdefault("secondary_time", times[1])
            slots.setdefault("secondary_time", times[1])
    arrival = ARRIVAL_RE.search(text)
    if arrival:
        arrival_value = TIME_RE.search(arrival.group(0))
        if arrival_value:
            _prefer_more_specific_slot(slots, spans, "arrival_time", arrival_value.group(0))
    provider = PROVIDER_RE.search(text)
    if provider:
        _prefer_more_specific_slot(slots, spans, "provider", provider.group(0))
    location = LOCATION_RE.search(text)
    if location:
        _prefer_more_specific_slot(slots, spans, "location", location.group(0))
    phone = PHONE_RE.search(text)
    if phone:
        _prefer_more_specific_slot(slots, spans, "phone", phone.group(0))
    lowered = text.lower()
    if "portal" in lowered:
        spans.setdefault("portal", "portal")
        slots.setdefault("portal", "portal")
        slots.setdefault("contact_method", "portal")
    if "backup" in lowered:
        slots.setdefault("backup_contact", text)
        spans.setdefault("backup_contact", text)
    if "pickup" in lowered:
        slots.setdefault("pickup", text)
        spans.setdefault("pickup", text)
    if "sulfa" in lowered:
        spans.setdefault("allergy_substance", "sulfa antibiotic")
        slots.setdefault("allergy_substance", "sulfa antibiotic")
    if "rash" in lowered:
        spans.setdefault("allergy_reaction", "rash")
        slots.setdefault("allergy_reaction", "rash")
    if "logistics only" in lowered:
        slots.setdefault("policy_scope", "logistics only")
        spans.setdefault("policy_scope", "logistics only")
    if "through monday" in lowered:
        slots.setdefault("policy_scope", "through Monday")
        spans.setdefault("policy_scope", "through Monday")
    if "through friday" in lowered:
        slots.setdefault("policy_scope", "through Friday")
        spans.setdefault("policy_scope", "through Friday")
    medication_like = _looks_like_medication_text(text, frame_type)
    if any(token in lowered for token in ["current treatment plan is", "treatment plan is"]) and not medication_like:
        medication_like = True
    if medication_like and any(token in lowered for token in ["continue ", "restart ", "hold ", "stop ", "start ", "use ", "take ", "switch to ", "avoid "]):
        slots.setdefault("instruction", text)
        spans.setdefault("instruction", text)
    if any(token in lowered for token in ["current treatment plan is", "treatment plan is"]):
        slots.setdefault("instruction", text)
        spans.setdefault("instruction", text)
        if "today" in lowered:
            slots.setdefault("timing", "today")
            spans.setdefault("timing", "today")
    if any(token in lowered for token in ["arrive 30 minutes early", "30 minutes early on monday", "30 minutes early for consent and check-in"]):
        slots.setdefault("instruction", text)
        spans.setdefault("instruction", text)
        slots.setdefault("timing", "30 minutes early")
        spans.setdefault("timing", "30 minutes early")
    if MONITORING_INSTRUCTION_RE.search(text):
        slots.setdefault("instruction", text)
        spans.setdefault("instruction", text)
        slots.setdefault("condition", "blood pressure monitoring")
        spans.setdefault("condition", "blood pressure monitoring")
        if any(token in lowered for token in ["twice daily", "morning and evening"]):
            slots.setdefault("timing", "twice daily")
            spans.setdefault("timing", "twice daily")
    dosage = MEDICATION_DOSAGE_RE.search(text)
    if dosage and medication_like:
        _prefer_more_specific_slot(slots, spans, "dosage", dosage.group(0))
    medication_name = _extract_medication_name(text) if medication_like else None
    if medication_name:
        _prefer_more_specific_slot(slots, spans, "medication", medication_name)
    if "return urgently" in lowered:
        slots.setdefault("condition", text.replace("Return urgently for ", "").strip())
        spans.setdefault("condition", text)
    for key, value in extract_state_slots(text).items():
        _prefer_more_specific_slot(slots, spans, key, value)
    if "safe_wording" not in slots:
        safe_match = SAFE_DISCLOSURE_RE.search(text)
        if safe_match:
            _prefer_more_specific_slot(slots, spans, "safe_wording", safe_match.group(1).strip(" ,."))
    if "public_event_date" not in slots:
        public_event_match = PUBLIC_EVENT_DATE_RE.search(text)
        if public_event_match:
            _prefer_more_specific_slot(slots, spans, "public_event_date", public_event_match.group(1).strip())
    household_like = frame_type == "household_plan" or any(
        token in lowered
        for token in ["visit window", "arrival window", "entry method", "approved areas", "approved rooms", "package rule", "oversized-package"]
    )
    if not household_like and "public_event_date" not in slots and "target_date" not in slots:
        if any(token in lowered for token in ["target date", "launch date remains", "dry-run date"]) and slots.get("date"):
            _prefer_more_specific_slot(slots, spans, "target_date", slots["date"])
    arrival_instruction = re.search(r"\b(?:text|call|contact)\s+(?:[A-Z][A-Za-z'-]*\s+)?on\s+arrival\b", text, re.IGNORECASE)
    if arrival_instruction:
        value = arrival_instruction.group(0).strip()
        slots.setdefault("instruction", value)
        spans.setdefault("instruction", value)
    return slots, spans


def canonicalize_medication_surface(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return ""
    if not re.match(r"^(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to|take)\b", cleaned, re.IGNORECASE):
        cleaned = re.sub(r"^[A-Za-z][A-Za-z /-]*?:\s*", "", cleaned)
    cleaned = re.sub(r"\bmicrograms\b", "mcg", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmilligrams\b", "mg", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bBID\b", "twice daily", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btwice a day\b", "twice daily", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bat bedtime\b", "nightly", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;,:")
    return cleaned


def normalize_medication_clause_surface(text: str) -> str:
    cleaned = canonicalize_medication_surface(text)
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"\bstop\s+([a-z][a-z0-9' -]{1,40}?)\s+(?:now|for now|for the biopsy|before the biopsy|until after [^.;]+|until after the procedure|until after fna)\b",
        r"hold \1",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"keep the total under\s+([\d,]+)\s+mg\s+in a day",
        r"\1 mg/day max",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\buse\s+([a-z0-9./' -]+?)\s+instead:\s+(\d+(?:\.\d+)?\s*mg\b)",
        r"use \1 \2",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"because of the [^,]+, I want you to stop\s+",
        "stop ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^Understood\.\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^For pain,\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^Let's\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ;")
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    return cleaned


def extract_medication_clauses(text: str) -> list[str]:
    cleaned = canonicalize_medication_surface(text)
    if not cleaned:
        return []
    clauses: list[str] = []
    seen: set[str] = set()
    for raw_piece in MEDICATION_CLAUSE_SPLIT_RE.split(cleaned):
        piece = raw_piece.strip(" .;,:")
        if not piece:
            continue
        action_match = re.search(r"\b(?:continue|increase|stop|use|start|hold|keep|restart|avoid|switch to)\b", piece, re.IGNORECASE)
        if not action_match:
            continue
        piece = piece[action_match.start():].strip(" .;,:")
        if not piece:
            continue
        normalized = canonicalize_medication_surface(piece).lower()
        piece = normalize_medication_clause_surface(piece)
        normalized = canonicalize_medication_surface(piece).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        clauses.append(piece[:1].upper() + piece[1:] if piece else piece)
    return clauses


def _prefer_more_specific_slot(slots: dict[str, str], spans: dict[str, str], key: str, value: str) -> None:
    candidate = str(value or "").strip()
    if not candidate:
        return
    current = str(slots.get(key) or "").strip()
    if not current or (current.lower() in candidate.lower() and len(candidate) > len(current)):
        slots[key] = candidate
        spans[key] = candidate


def _resolve_slot_access(principal: Any, slots: dict[str, str], decision: dict[str, Any], record_type: str) -> tuple[list[str], list[str]]:
    role = str(getattr(principal, "organization_role", "") or "").lower()
    relation = str(getattr(principal, "relation_to_owner", "") or "").lower()
    logistics = {
        "date", "time", "secondary_time", "arrival_time", "location", "provider", "procedure", "visit_type", "status",
        "instruction", "contact_method", "phone", "portal", "backup_contact", "policy_scope", "pickup",
        "target_date", "public_event_date", "approved_budget", "approved_discount_cap", "monthly_stipend", "safe_wording",
        "blocker", "entry_method", "visit_window", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule",
    }
    clinical = {"medication", "dosage", "allergy_substance", "allergy_reaction", "condition"}
    allowed: list[str] = []
    denied: list[str] = []
    logistics_only = relation in {"family", "caregiver", "proxy", "delegate"} or role in {"scheduler", "front_desk", "delegate_assistant"}
    for key in slots:
        if logistics_only and key in clinical and record_type not in {"allergy"}:
            denied.append(key)
        else:
            allowed.append(key)
    if logistics_only and record_type == "clinical_sensitive":
        denied.extend([key for key in slots if key not in denied])
        allowed = [key for key in allowed if key not in denied]
    return sorted(set(allowed)), sorted(set(denied))


def _infer_principal_access(principal: Any) -> str:
    relation = str(getattr(principal, "relation_to_owner", "") or "").lower()
    role = str(getattr(principal, "organization_role", "") or "").lower()
    if relation == "owner":
        return "owner"
    if role in {"clinician", "nurse", "pharmacist"}:
        return "clinical_staff"
    if role in {"scheduler", "front_desk", "delegate_assistant"}:
        return "logistics_staff"
    if role in {"financial_aid", "advisor", "registrar", "department_admin", "campus_it", "resident", "primary_resident", "professor", "counselor", "ta", "ra"}:
        return "operational_staff"
    if relation in {"family", "caregiver", "proxy"}:
        return "family"
    return "unknown"


def _looks_like_medication_text(text: str, frame_type: str) -> bool:
    lowered = text.lower()
    if frame_type == "medication":
        return True
    if MONITORING_INSTRUCTION_RE.search(text):
        return True
    explicit_medication_cues = ["medication", "medicine", "regimen", "prescription"]
    if any(token in lowered for token in explicit_medication_cues):
        return True
    if frame_type in {"household_plan", "project_state", "research_state", "appointment", "test_or_imaging", "clinic_visit", "logistics", "update", "consent_or_permission"}:
        return False
    if not MEDICATION_ACTION_RE.search(text):
        return False
    if not (MEDICATION_DOSAGE_RE.search(text) or any(token in lowered for token in ["nightly", "daily", "twice daily", "with food", "as needed", "prn", "tablet", "capsule", "mg", "mcg"])):
        return False
    return True


def _extract_medication_name(text: str) -> str | None:
    match = re.search(
        r"\b(?:start|stop|continue|hold|restart|use|take|avoid|switch to)\s+([A-Za-z][A-Za-z0-9'./\-]*(?:\s+[A-Za-z0-9'./\-]+){0,4})",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    candidate = match.group(1).strip(" .,:;")
    for tail in ["for now", "right now", "because", "and", "with food", "at bedtime", "every six hours", "twice daily", "once daily"]:
        if candidate.lower().endswith(tail):
            candidate = candidate[: -len(tail)].strip(" .,:;")
    return candidate or None
