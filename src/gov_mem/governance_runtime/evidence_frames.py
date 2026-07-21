from __future__ import annotations

import re
from dataclasses import asdict
from hashlib import md5

from gov_mem.data.schema import EvidenceFrame, RetrievedEvidence
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.llm.prompts import build_answering_user_prompt
from gov_mem.query_semantics import extract_state_slots


FRAME_TYPES = (
    "appointment",
    "test_or_imaging",
    "clinic_visit",
    "medication",
    "allergy",
    "instruction",
    "consent_or_permission",
    "logistics",
    "cancellation",
    "update",
    "forgetting",
    "privacy_policy",
    "diagnosis_or_result",
    "general_fact",
    "project_state",
    "research_state",
    "household_plan",
)

DATE_RE = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"(?:,\s*)?(?:\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))?"
    r"\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
DATE_GENERIC_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b",
    re.IGNORECASE,
)
WEEKDAY_ONLY_RE = re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE)
DATE_RELATIVE_RE = re.compile(r"\b(?:today|tomorrow|next Tuesday|next Friday|this Tuesday|this Friday)\b", re.IGNORECASE)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", re.IGNORECASE)
ARRIVAL_RE = re.compile(r"(?:arrive by|arrival(?: time)?(?: is)?|please arrive by|check in by|come by|arrive at)\s+(\d{1,2}:\d{2}\s?(?:AM|PM))", re.IGNORECASE)
LOCATION_RE = re.compile(r"\b(?:(?!(?:AM|PM)\b)[A-Z][A-Za-z0-9&' -]{1,60}\s+(?:Suite|Clinic|Center|Office|Lab|Ward|Desk)\s*[A-Z0-9-]*|front desk)\b")
PROVIDER_RE = re.compile(r"\bDr\.\s+[A-Z][a-zA-Z]+\b")
ALLERGY_RE = re.compile(r"\b(allerg(?:y|ic)\s+to|reaction to|allergy documented:)\s*([A-Za-z0-9\-\s]+)", re.IGNORECASE)
REACTION_RE = re.compile(r"\b(rash|hives|anaphylaxis|nausea|vomiting|itching|swelling)\b", re.IGNORECASE)
MONEY_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*USD\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"\b\d{1,2}(?:\.\d+)?%")
CURRENT_CUE_RE = re.compile(
    r"\b(?:current|official current|revised approved|current cap to use|supersedes?|progression|moves? from|is now|latest|remains|replaces?|updated to|later approved|only approved amount)\b",
    re.IGNORECASE,
)
ACCESS_ARTIFACT_TERM_RE = re.compile(
    r"\b(?:credential|credentials|passphrase|token|badge|keypad|door code|access key|access note|portal credential|guest-network|guest network|release phrase|passcode|(?:qr|barcode|one-time)\s+(?:code|credential|token))\b",
    re.IGNORECASE,
)
ACCESS_ARTIFACT_VALIDITY_RE = re.compile(
    r"\b(?:active through|active until|valid through|valid until|expires?|expired|expiration|expiry|until revoked|until replaced|credential rotation|access note supersedes)\b",
    re.IGNORECASE,
)
PUBLIC_EVENT_SHARE_RE = re.compile(
    r"\b(?:shared broadly|can be shared broadly|broadly with|students and staff|staff and students|announcements should|calendar line|mixed-audience agendas|public schedule|public event|orientation|open house)\b",
    re.IGNORECASE,
)
BLOCKER_RE = re.compile(
    r"\b(?:current\s+blocker\s+is|remaining\s+blocker(?:\s+is)?|remaining\s+blockers?\s+are|one\s+open\s+blocker:\s*|blocker:\s*)([^.;]+)",
    re.IGNORECASE,
)
PENDING_BLOCKER_RE = re.compile(r"\b([^.;]+?)\s+remained pending\b", re.IGNORECASE)
NO_HOLD_RE = re.compile(r"\bno enrollment hold\b", re.IGNORECASE)
NO_REMAINING_BLOCKER_RE = re.compile(r"\b(?:no remaining(?:\s+[a-z]+)?\s+blockers?|no live blocker|no blocker remains|blockers? (?:are )?cleared)\b", re.IGNORECASE)
ENTRY_METHOD_RE = re.compile(
    r"\b(?:via|through|using|use|should use|entry is via|entry via|current entry is|entry method is|approved entrance is|route is)\s+(?:the\s+)?([A-Za-z0-9\- ]+(?:door|keypad|gate|entry|code|lockbox))\b",
    re.IGNORECASE,
)
VISIT_WINDOW_RE = re.compile(
    r"(?:\bfrom\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\s+to\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\b|"
    r"\bbetween\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\s+and\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\b|"
    r"\b(\d{1,2}:\d{2}\s?(?:AM|PM))\s+to\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\b|"
    r"\b(\d{1,2}:\d{2}\s?(?:AM|PM))\s*[\-\u2013\u2014]\s*(\d{1,2}:\d{2}\s?(?:AM|PM))\b)",
    re.IGNORECASE,
)
PACKAGE_RULE_RE = re.compile(
    r"\b(?:large-package rule|oversized[- ]package rule|package note|if a package is too large[^.]*|oversized deliveries[^.]*)\b[^.]*",
    re.IGNORECASE,
)
PARKING_PASS_RE = re.compile(r"\b(?:visitor\s+parking\s+pass(?:\s+is)?(?:\s+currently)?|parking\s+pass(?:\s+is)?(?:\s+currently)?|visitor\s+pass)\s+([A-Z]\d+)\b", re.IGNORECASE)
ARRIVAL_CONTACT_RE = re.compile(r"\b([A-Z][a-z]+)\s+should\s+text\s+([A-Z][a-z]+)\s+on\s+arrival\b", re.IGNORECASE)
ARRIVAL_CONTACT_GENERIC_RE = re.compile(
    r"\b(?:(?:text|call|contact|check in with)\s+[A-Z][A-Za-z'-]*(?:\s+(?:on\s+arrival|from\s+the\s+(?:lobby|desk)))?|"
    r"[A-Z][A-Za-z'-]*\s+(?:should\s+)?(?:text|call|contact|meet)\s+[^.;]+?(?:on\s+arrival|at\s+the\s+(?:lobby|desk)))\b",
    re.IGNORECASE,
)
APPROVED_AREAS_RE = re.compile(
    r"\bapproved areas(?: are|:)\s+([^.;]+)",
    re.IGNORECASE,
)
LIMITED_AREAS_RE = re.compile(
    r"\b(?:access|work|activity)\s+(?:is\s+)?(?:limited\s+to|allowed\s+in)\s+([^.;]+)",
    re.IGNORECASE,
)
LIMITED_ACCESS_AREAS_RE = re.compile(
    r"\b(?:access\s+limited\s+to|limited\s+to)\s+([^.;]+)",
    re.IGNORECASE,
)
SAFE_WORDING_RE = re.compile(
    r"\b(?:safe external wording remains|safe external wording should stay|broad sponsor wording|safe sponsor wording|safe description should stay|safe title(?: should stay)?|broad safe title|active outward title remains|only broad safe wording)\s+['\"]?([^'\"]+)['\"]?",
    re.IGNORECASE,
)
SAFE_STATUS_WORDING_RE = re.compile(
    r"\b(?:housing wording should stay broad|safe wording should stay broad|only the broad phrase|broad room[- ]reassignment wording(?:\s+is)?|room[- ]reassignment wording(?:\s+is)?)\s*(?::|\bis\b|\bshould stay\b)\s*['\"]?([A-Za-z][^.;'\"]+?)['\"]?(?:[.;]|$)",
    re.IGNORECASE,
)
SAFE_BROAD_EXPLANATION_RE = re.compile(
    r"\b(?:only broad explanation(?: outside the support chain)? is|broad replies should continue to use only|broad safe recap:\s*|current safe recap:\s*)([A-Za-z][^.;]+)",
    re.IGNORECASE,
)
UPDATED_TARGET_DATE_RE = re.compile(
    r"\b(?:moves?\s+from\s+[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?\s+to\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"moved\s+to\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"(?:official|current)\s+(?:pilot\s+)?target(?:\s+date)?(?:\s+is|\s+as)?\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"launch date remains\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"current target date(?:\s+as)?\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"dry-run date(?:\s+is)?\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"review date moved to\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"with current date\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)|"
    r"treat\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)\s+as the current target)\b",
    re.IGNORECASE,
)
HOLD_STATUS_RE = re.compile(
    r"\b(?:tuition hold(?: is)?(?: now)?\s+(cleared|active|still active)|"
    r"no tuition hold|"
    r"hold removal|"
    r"hold is in final clearance|"
    r"no enrollment hold)\b",
    re.IGNORECASE,
)
TEST_OR_IMAGING_RE = re.compile(r"\b(?:ultrasound|imaging|scan|x-ray|mri|beta-hcg|lab draw|blood draw)\b", re.IGNORECASE)
MED_DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|micrograms?|grams?|tablets?|capsules?)"
    r"(?:\s*(?:q\d+h|bid|tid|daily|nightly|once daily|twice daily|every\s+\w+\s+hours?|as needed|prn|with food|at bedtime))?\b",
    re.IGNORECASE,
)
MEDICATION_ACTION_RE = re.compile(r"\b(?:start|stop|continue|hold|restart|use|take|avoid|switch to)\b", re.IGNORECASE)
MONITORING_INSTRUCTION_RE = re.compile(
    r"\b(?:check|monitor)\s+(?:your\s+)?blood pressure\b|\bblood pressure twice daily\b|\bmorning and evening\b|\bwrite the readings\b|\blog\b",
    re.IGNORECASE,
)


def compile_evidence_frames(evidence: list[RetrievedEvidence]) -> list[EvidenceFrame]:
    return [compile_evidence_frame(row) for row in evidence]


def compile_evidence_frame(row: RetrievedEvidence) -> EvidenceFrame:
    content = row.content or ""
    frame_type = _infer_frame_type(content, row)
    slots = _extract_slots(content, frame_type)
    frame_id = md5(f"{row.memory_id}:{frame_type}:{content}".encode("utf-8")).hexdigest()[:12]
    sensitivity = {
        "privacy_level": (row.metadata or {}).get("privacy_level"),
        "redaction_required": bool((row.metadata or {}).get("redaction_required") or (row.metadata or {}).get("requires_redaction")),
        "sensitive_entities": list((row.metadata or {}).get("sensitive_entities") or []),
    }
    access_scope = {
        "authorized_users": list((row.metadata or {}).get("authorized_users") or []),
        "forbidden_users": list((row.metadata or {}).get("forbidden_users") or []),
        "access_scope": (row.metadata or {}).get("access_scope"),
    }
    semantic_tags = dict((row.metadata or {}).get("semantic_tags") or {})
    event_identity = dict(semantic_tags.get("event_identity") or {})
    provenance_attributes = dict(semantic_tags.get("attributes") or {})
    semantic_surface_values = dict(semantic_tags.get("surface_values") or {})
    state_delta = dict(semantic_tags.get("state_delta") or {})
    changed_fields = state_delta.get("changed_fields")
    if isinstance(changed_fields, dict) and changed_fields:
        # changed_fields is the certified post-turn active state. Other
        # attributes remain provenance and may contain prior/from values.
        semantic_attributes = {
            str(key): value
            for key, value in changed_fields.items()
            if value not in (None, "", [])
        }
    else:
        semantic_attributes = provenance_attributes
    for key, value in semantic_attributes.items():
        if value not in (None, "", []):
            # Keep normalized values in semantic_attributes for alignment, but
            # render from an exact grounded source span when one is certified.
            slots.setdefault(key, semantic_surface_values.get(key) or value)
    grounded_surface_spans = _extract_surface_spans(content, slots)
    for key in semantic_attributes:
        surface_value = semantic_surface_values.get(key)
        if surface_value not in (None, "", []):
            grounded_surface_spans[key] = surface_value
    return EvidenceFrame(
        frame_id=frame_id,
        memory_id=row.memory_id,
        source_message_ids=list(row.source_message_ids),
        frame_type=frame_type,
        owner_user=row.user_id,
        subject_entity=str(event_identity.get("entity_key") or _infer_subject_entity(frame_type, slots, row)),
        lifecycle_status=str((row.metadata or {}).get("memory_status") or "active"),
        effective_time=row.time,
        event_time=slots.get("date") or row.time,
        slots=slots,
        access_scope=access_scope,
        sensitivity=sensitivity,
        surface_spans=grounded_surface_spans,
        confidence=float(row.score),
        discourse_act=str(semantic_tags.get("discourse_act") or "unknown"),
        assertion_confidence=float(semantic_tags.get("assertion_confidence") or 0.0),
        event_identity=event_identity,
        state_delta=state_delta,
        semantic_attributes=semantic_attributes,
        provenance={
            "memory_id": row.memory_id,
            "source_message_ids": list(row.source_message_ids),
            "retrieval_source": row.retrieval_source,
        },
    )


def normalize_frames_with_llm(
    memory_item,
    initial_frames: list[EvidenceFrame],
    llm_client: LLMClient | None,
    config: dict | None,
) -> list[EvidenceFrame]:
    if llm_client is None or not llm_client.is_available():
        return initial_frames
    cfg = config or {}
    runtime_cfg = cfg.get("governance_runtime") or {}
    if not bool(runtime_cfg.get("use_llm_frame_normalizer", True)):
        return initial_frames
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(cfg, "memory_ingestion"),
            system_prompt="Extract typed event frames from the memory text. Preserve exact surface strings for dates, times, locations, providers, procedures, medications, allergies, instructions, cancellations, and updates. Do not infer hidden labels. Do not answer the user question.",
            user_prompt=build_answering_user_prompt(
                question=str(getattr(memory_item, "content", "") or ""),
                asking_user_id=None,
                choices=None,
                selected_evidence=[{
                    "memory_id": getattr(memory_item, "memory_id", None),
                    "content": getattr(memory_item, "content", ""),
                }],
                reasoning_trace=[f"Initial frames: {len(initial_frames)}"],
                conclusion_hint="Frame normalization only.",
                skill_text="",
                retrieved_lessons=[],
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return initial_frames
    if not isinstance(raw, dict):
        return initial_frames
    extra_frames = []
    for idx, item in enumerate(raw.get("frames", []) if isinstance(raw.get("frames"), list) else []):
        try:
            extra_frames.append(
                EvidenceFrame(
                    frame_id=str(item.get("frame_id") or f"llm_{idx}"),
                    memory_id=str(getattr(memory_item, "memory_id", "")),
                    source_message_ids=list(getattr(memory_item, "source_message_ids", []) or []),
                    frame_type=str(item.get("frame_type") or "general_fact"),
                    owner_user=getattr(memory_item, "user_id", None),
                    subject_entity=str(item.get("subject_entity") or None),
                    lifecycle_status=str(item.get("lifecycle_status") or "active"),
                    effective_time=item.get("effective_time"),
                    event_time=item.get("event_time"),
                    slots=dict(item.get("slots") or {}),
                    access_scope=dict(item.get("access_scope") or {}),
                    sensitivity=dict(item.get("sensitivity") or {}),
                    surface_spans=dict(item.get("surface_spans") or {}),
                    confidence=float(item.get("confidence", 0.5)),
                    discourse_act=str(item.get("discourse_act") or "unknown"),
                    assertion_confidence=float(item.get("assertion_confidence") or 0.0),
                    event_identity=dict(item.get("event_identity") or {}),
                    state_delta=dict(item.get("state_delta") or {}),
                    semantic_attributes=dict(item.get("semantic_attributes") or item.get("attributes") or {}),
                    provenance=dict(item.get("provenance") or {}),
                )
            )
        except Exception:
            continue
    return initial_frames + extra_frames


def frame_to_dict(frame: EvidenceFrame) -> dict:
    return asdict(frame)


def _infer_frame_type(content: str, row: RetrievedEvidence) -> str:
    lowered = content.lower()
    if any(token in lowered for token in ["forget", "delete", "remove", "clear temporary"]):
        return "forgetting"
    explicit_state_slots = extract_state_slots(content)
    household_slot_names = {"visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}
    if set(explicit_state_slots) & household_slot_names:
        return "household_plan"
    if set(explicit_state_slots) & {"target_date", "approved_budget", "approved_discount_cap", "monthly_stipend", "safe_wording", "blocker"}:
        return "research_state" if set(explicit_state_slots) & {"monthly_stipend", "safe_wording"} else "project_state"
    if _looks_like_support_state_summary(content):
        return "research_state"
    if any(token in lowered for token in ["canceled", "cancelled", "no longer scheduled"]):
        return "cancellation"
    if any(token in lowered for token in ["updated", "replaced", "instead of", "changed effective"]):
        return "update"
    if any(token in lowered for token in ["allergy", "allergic", "reaction to"]):
        return "allergy"
    if _looks_like_medication_frame_text(content):
        return "medication"
    if any(
        token in lowered
        for token in [
            "target date",
            "approved budget",
            "approved current",
            "current budget",
            "working budget",
            "discount cap",
            "commercial cap",
            "max discount",
            "current blocker",
            "launch date remains",
            "official pilot target",
            "moves from",
            "status anchor",
            "current atlas state",
            "monthly stipend",
            "safe external wording",
            "safe sponsor wording",
            "student-safe title",
            "safe title",
            "support figure",
            "approved visit window",
            "entry method",
            "oversized-package",
            "large-package rule",
            "approved areas",
            "revised approved",
            "budget is now",
            "discount is now",
            "cap is now",
            "current planned arrival window",
            "approved door",
            "side door",
            "text on arrival",
        ]
    ):
        if any(token in lowered for token in ["stipend", "dry-run", "irb", "microscope", "sponsor wording", "safe title", "student-safe title", "support figure", "atlas"]):
            return "research_state"
        if any(token in lowered for token in ["visit window", "entry method", "approved areas", "package rule", "keypad", "current planned arrival window", "approved door", "side door", "text on arrival"]):
            return "household_plan"
        return "project_state"
    if any(
        token in lowered
        for token in [
            "leading diagnosis remains",
            "current diagnosis",
            "current suspicion is",
            "working diagnosis",
            "incident cause",
            "root cause",
            "incorrect environment refresh",
            "assurance workers",
            "staging workers",
        ]
    ):
        return "project_state"
    if any(
        token in lowered
        for token in [
            "review date",
            "approved amount",
            "final approved amount",
            "support amount",
            "support figure",
            "safe title",
            "mixed-audience summaries",
            "outward scheduling-safe wording",
            "hold is active while",
        ]
    ):
        return "research_state"
    if TEST_OR_IMAGING_RE.search(content):
        return "test_or_imaging"
    if any(token in lowered for token in ["clinic visit", "follow-up", "follow up", "with dr."]):
        return "clinic_visit"
    if any(
        token in lowered
        for token in [
            "consent",
            "authorized",
            "access revoked",
            "do not share",
            "family access",
            "permission",
            "logs-only",
            "scheduling-only",
            "scope only",
            "scoped debugging window",
            "broad standing privilege",
        ]
    ):
        return "consent_or_permission"
    if any(token in lowered for token in ["voicemail", "callback", "portal", "parking", "arrive by", "suite", "front desk", "clinic logistics"]):
        return "logistics"
    if any(
        token in lowered
        for token in [
            "diagnosis",
            "result",
            "lab value",
            "lab status",
            "lab recap",
            "pregnancy",
            "beta-hcg result",
            "confirmatory positive",
            "cmp is normal",
            "cmp normal",
            "cd4",
            "viral load",
            "leading diagnosis remains",
            "current suspicion is",
            "current diagnosis",
        ]
    ):
        return "diagnosis_or_result"
    if MONITORING_INSTRUCTION_RE.search(content):
        return "instruction"
    if any(token in lowered for token in ["should", "must", "instruction", "unless symptoms worsen", "take ", "stop ", "avoid "]):
        return "instruction"
    if any(token in lowered for token in ["appointment", "scheduled", "schedule"]):
        return "appointment"
    return "general_fact"


def _extract_slots(content: str, frame_type: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    date_match = DATE_RE.search(content) or DATE_GENERIC_RE.search(content) or WEEKDAY_ONLY_RE.search(content) or DATE_RELATIVE_RE.search(content)
    if date_match:
        slots["date"] = _upgrade_partial_date_from_context(content, date_match.group(0).strip())
    times = TIME_RE.findall(content)
    if times:
        slots["time"] = times[0].strip()
        if len(times) > 1:
            slots.setdefault("secondary_time", times[1].strip())
    arrival_match = ARRIVAL_RE.search(content)
    if arrival_match:
        slots["arrival_time"] = arrival_match.group(1).strip()
    location_match = LOCATION_RE.search(content)
    if location_match:
        slots["location"] = location_match.group(0).strip()
    provider_match = PROVIDER_RE.search(content)
    if provider_match:
        slots["provider"] = provider_match.group(0).strip()

    lowered = content.lower()
    if frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "cancellation", "update"}:
        if "ultrasound" in lowered:
            slots["procedure"] = "ultrasound"
        elif "orientation" in lowered:
            slots["procedure"] = "orientation"
        elif "beta-hcg" in lowered:
            slots["procedure"] = "beta-hCG"
        elif "blood draw" in lowered:
            slots["procedure"] = "blood draw"
        elif "follow-up" in lowered or "follow up" in lowered:
            slots["procedure"] = "follow-up"
        elif "scan" in lowered:
            slots["procedure"] = "scan"
        if frame_type == "clinic_visit":
            slots["visit_type"] = "clinic visit"
        elif frame_type == "test_or_imaging":
            slots["visit_type"] = "test or imaging"
        elif frame_type == "appointment":
            slots["visit_type"] = "appointment"
        if "unless symptoms worsen" in lowered:
            slots["precondition"] = "unless symptoms worsen"
        if any(token in lowered for token in ["canceled", "cancelled", "no longer scheduled", "no longer active"]):
            slots["status"] = "canceled"
            slots["canceled_event"] = content.strip()
        if any(token in lowered for token in ["instead of", "replaced", "updated", "moved to", "rescheduled to"]):
            slots["status"] = "updated"
            slots["replacement_event"] = content.strip()
        public_event_date = extract_state_slots(content).get("public_event_date")
        if public_event_date:
            slots["public_event_date"] = public_event_date

    if frame_type == "allergy":
        allergy_match = ALLERGY_RE.search(content)
        if allergy_match:
            slots["substance"] = allergy_match.group(2).strip(" .")
        reaction_match = REACTION_RE.search(content)
        if reaction_match:
            slots["reaction"] = reaction_match.group(1).strip()
        slots["allergy"] = content.strip()

    if frame_type == "medication":
        slots["medication"] = content.strip()
        dosage_match = MED_DOSAGE_RE.search(content)
        if dosage_match:
            slots["dosage"] = dosage_match.group(0).strip()
        med_name = _extract_medication_name(content)
        if med_name:
            slots["medication_name"] = med_name
        if any(token in lowered for token in ["stop ", "avoid ", "take ", "use ", "continue ", "start "]):
            slots["instruction"] = content.strip()
        if "unless symptoms worsen" in lowered:
            slots["condition"] = "unless symptoms worsen"
        if "before tuesday" in lowered:
            slots["timing"] = "before Tuesday"
        elif any(token in lowered for token in ["nightly", "at bedtime", "each morning", "twice a day", "twice daily", "once daily", "daily", "with food", "every six hours", "q6h", "as needed", "prn"]):
            slots["timing"] = content.strip()

    if frame_type in {"consent_or_permission", "privacy_policy"}:
        slots["consent_scope"] = content.strip()

    if frame_type == "instruction":
        slots["prep_instruction"] = content.strip()
        if MONITORING_INSTRUCTION_RE.search(content):
            slots["instruction"] = content.strip()
            if "blood pressure" in lowered:
                slots["condition"] = "blood pressure monitoring"
            if any(token in lowered for token in ["twice daily", "morning and evening"]):
                slots["timing"] = "twice daily"

    if frame_type == "diagnosis_or_result":
        slots["result"] = content.strip()
    if frame_type == "logistics":
        if "clinic" in lowered:
            slots["visit_type"] = "clinic logistics"
        if "front desk" in lowered:
            slots["location"] = "front desk"
    if frame_type in {"project_state", "research_state", "household_plan", "general_fact"}:
        access_artifact_temporal = _looks_like_access_artifact_temporal_line(content)
        if access_artifact_temporal and slots.get("date"):
            slots["access_valid_until"] = slots["date"]
        shared_state_slots = extract_state_slots(content)
        for key, value in shared_state_slots.items():
            slots[key] = value
        for key, value in _extract_open_state_slots(content).items():
            slots[key] = value
        if shared_state_slots.get("approved_budget"):
            slots.setdefault("budget", shared_state_slots["approved_budget"])
        if shared_state_slots.get("approved_discount_cap"):
            slots.setdefault("discount_cap", shared_state_slots["approved_discount_cap"])
        if shared_state_slots.get("monthly_stipend"):
            slots.setdefault("stipend", shared_state_slots["monthly_stipend"])
        hold_status = _extract_hold_status(content)
        if hold_status and "status" not in slots:
            slots["status"] = hold_status
        if "blocker" not in slots:
            pending_blocker_match = PENDING_BLOCKER_RE.search(content)
            blocker_match = BLOCKER_RE.search(content)
            if pending_blocker_match:
                slots["blocker"] = pending_blocker_match.group(1).strip(" .")
            elif blocker_match:
                slots["blocker"] = blocker_match.group(1).strip(" .")
            elif "was cleared" in lowered and "proceeded" in lowered and "pending" not in lowered:
                # Keep a source-grounded status value instead of inventing a
                # shorter canonical phrase that may drop qualifiers.
                cleared_match = re.search(
                    r"\b[^.;]*?\b(?:was|were)\s+cleared\b",
                    content,
                    re.IGNORECASE,
                )
                slots["blocker"] = (
                    cleared_match.group(0).strip()
                    if cleared_match
                    else "cleared"
                )
        if any(
            token in lowered
            for token in [
                "leading diagnosis remains",
                "current diagnosis",
                "current suspicion is",
                "working diagnosis",
                "incident cause",
                "root cause",
            ]
        ):
            slots["operational_result"] = content.strip()
        if shared_state_slots.get("target_date"):
            slots["target_date"] = shared_state_slots["target_date"]
            if frame_type in {"update", "general_fact"} and any(token in lowered for token in ["current date", "current committee date", "updated to", "going forward"]):
                slots["date"] = shared_state_slots["target_date"]
        elif (
            not access_artifact_temporal
            and any(token in lowered for token in ["target date", "launch date remains", "dry-run date", "dry-run target", "stands at", "current target", "review date", "current date"])
        ):
            if "date" in slots:
                slots["target_date"] = slots["date"]
        if (
            "target_date" not in slots
            and "date" in slots
            and not access_artifact_temporal
            and any(token in lowered for token in ["closure", "review memo", "kickoff"])
        ):
            slots["target_date"] = slots["date"]
        if shared_state_slots.get("public_event_date"):
            slots["public_event_date"] = shared_state_slots["public_event_date"]
            slots.pop("target_date", None)
        _set_if_present(slots, "package_rule", _extract_package_rule(content))
        _set_if_present(slots, "arrival_contact_rule", _extract_arrival_contact_rule(content))

    if frame_type in {"household_plan", "general_fact", "update", "logistics", "diagnosis_or_result"}:
        if "visit_window" not in slots:
            visit_window_match = VISIT_WINDOW_RE.search(content)
            if visit_window_match:
                start = visit_window_match.group(1) or visit_window_match.group(3) or visit_window_match.group(5)
                end = visit_window_match.group(2) or visit_window_match.group(4) or visit_window_match.group(6)
                if start and end:
                    slots["visit_window"] = f"{start.strip()} to {end.strip()}"
        _set_if_present(slots, "entry_method", _extract_entry_method(content))
        _set_if_present(slots, "arrival_contact_rule", _extract_arrival_contact_rule(content))
        _set_if_present(slots, "approved_areas", _extract_approved_areas(content))
        if "visit_window" not in slots:
            visit_window_match = VISIT_WINDOW_RE.search(content)
            if visit_window_match:
                start = visit_window_match.group(1) or visit_window_match.group(3) or visit_window_match.group(5)
                end = visit_window_match.group(2) or visit_window_match.group(4) or visit_window_match.group(6)
                if start and end:
                    slots["visit_window"] = f"{start.strip()} to {end.strip()}"
    return slots


def _extract_entry_method(content: str) -> str | None:
    """Return an evidence-grounded access route without naming specific fixtures."""
    match = ENTRY_METHOD_RE.search(content)
    if match:
        return match.group(1).strip(" ,.;")
    match = re.search(
        r"\b(?:pick\s+up|collect|use|enter\s+with)\s+(?:the\s+)?([^.;]+?(?:card|pass|envelope|badge|key|code|buzz))\b",
        content,
        re.IGNORECASE,
    )
    return match.group(1).strip(" ,.;") if match else None


def _extract_arrival_contact_rule(content: str) -> str | None:
    if "resident confirmation required on arrival" in content.lower():
        return "resident confirmation required on arrival"
    match = ARRIVAL_CONTACT_RE.search(content) or ARRIVAL_CONTACT_GENERIC_RE.search(content)
    return match.group(0).strip(" ,.;") if match else None


def _extract_package_rule(content: str) -> str | None:
    match = PACKAGE_RULE_RE.search(content)
    if match:
        return match.group(0).strip(" ,.;")
    match = re.search(r"\b(?:move|place|leave|return)\s+[^.;]*(?:package|delivery|parcel)[^.;]*", content, re.IGNORECASE)
    return match.group(0).strip(" ,.;") if match else None


def _extract_approved_areas(content: str) -> str | None:
    match = APPROVED_AREAS_RE.search(content) or LIMITED_AREAS_RE.search(content)
    return match.group(1).strip(" ,.;") if match else None


def _set_if_present(slots: dict[str, str], name: str, value: str | None) -> None:
    if value and name not in slots:
        slots[name] = value


def _extract_open_state_slots(content: str) -> dict[str, str]:
    patterns = {
        "contract_structure": [
            r"\b((?:fixed|rolling|renewable)\s+[^.;]{0,60}?term(?:\s+with\s+[^.;]+)?)",
            r"\bcontract structure(?:\s+is|:)?\s+([^.;]+)",
        ],
        "selected_vendor": [
            r"\b([A-Z][A-Za-z0-9&-]+)\s+is\s+(?:now\s+)?the\s+selected\s+[^.;]*vendor\b",
            r"\bselected\s+[^.;]*vendor(?:\s+is|:)?\s+([A-Z][A-Za-z0-9&-]+)",
        ],
        "family_release_scope": [
            r"\bfamily[- ]release scope(?:\s+is|:)?\s+([^.;]+)",
            r"\bfamily may receive\s+([^.;]+)",
        ],
        "public_room": [
            r"\bpublic\s+(?:mentors\s+|meeting\s+)?room(?:\s+is|:)?\s+([A-Z][A-Za-z0-9&' -]*\d+)",
        ],
        "coordination_label": [
            r"\bcoordination label(?:\s+is|:)?\s+([A-Za-z0-9_-]+)",
        ],
        "access_token": [
            r"\b(?:active|current)\s+(?:staging\s+)?token(?:\s+is|:)?\s+([A-Za-z0-9_-]{8,})",
        ],
    }
    extracted: dict[str, str] = {}
    for slot, slot_patterns in patterns.items():
        for pattern in slot_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                extracted[slot] = match.group(1).strip(" ,.;")
                break
    return extracted


def _extract_latest_target_date(content: str) -> str | None:
    if _looks_like_access_artifact_temporal_line(content):
        return None
    return extract_state_slots(content).get("target_date")


def _has_current_cue(content: str) -> bool:
    return bool(CURRENT_CUE_RE.search(content) or '->' in content)


def _looks_like_access_artifact_temporal_line(content: str) -> bool:
    lowered = str(content or "").lower()
    if not ACCESS_ARTIFACT_TERM_RE.search(content):
        return False
    if ACCESS_ARTIFACT_VALIDITY_RE.search(content):
        return True
    return "access note" in lowered and "supersede" in lowered


def _looks_like_public_event_context(content: str) -> bool:
    lowered = str(content or "").lower()
    if any(
        token in lowered
        for token in [
            "public note",
            "public schedule",
            "public event",
            "orientation",
            "calendar line",
            "announcements should",
            "event times",
        ]
    ):
        return True
    return "public" in lowered and bool(PUBLIC_EVENT_SHARE_RE.search(content))


def _extract_current_money_amount(content: str) -> str | None:
    directional_patterns = [
        r"(?:approved current(?: [a-z]+)? budget|approved current amount|approved amount|approved value|current amount|revised approved [a-z]+ budget|budget is now|working budget is now)\s*(?::|is|=)?\s*(\d[\d,]*(?:\.\d+)?\s*USD)",
        r"(\d[\d,]*(?:\.\d+)?\s*USD)\s*(?:after removing|after excluding|after trimming|after removing contingency)",
        r"(?:supersedes?|replaces?)\s+(?:the\s+)?earlier\s+(\d[\d,]*(?:\.\d+)?\s*USD)",
    ]
    preferred_patterns = directional_patterns[:2]
    for pattern in preferred_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    explicit_patterns = [
        r"(?:approved current amount is|approved amount is|approved value is|current amount is|later approved amount is|approved at|only approved amount\s+is|only approved amount)\s+(\d[\d,]*(?:\.\d+)?\s*USD)",
        r"(\d[\d,]*(?:\.\d+)?\s*USD)\s*(?:,?\s*(?:which|that)\s+(?:replaces|supersedes))",
        r"replaced by an approved amount of\s+(\d[\d,]*(?:\.\d+)?\s*USD)",
        r"updated to approved\s+(\d[\d,]*(?:\.\d+)?\s*USD)",
        r"only approved\s+(\d[\d,]*(?:\.\d+)?\s*USD)\s+remains",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    matches = [m.group(0).strip() for m in MONEY_RE.finditer(content)]
    if not matches:
        return None
    if _has_current_cue(content):
        return matches[-1]
    return matches[0]


def _extract_public_event_date(content: str) -> str | None:
    return extract_state_slots(content).get("public_event_date")


def _upgrade_partial_date_from_context(content: str, date_value: str) -> str:
    value = str(date_value or "").strip()
    if not value:
        return value
    if re.search(r",\s*\d{4}\b", value):
        return value
    match = re.search(rf"{re.escape(value)}(?:,\s*(\d{{4}}))", content, flags=re.IGNORECASE)
    if match:
        return f"{value}, {match.group(1)}"
    return value


def _extract_hold_status(content: str) -> str | None:
    lowered = content.lower()
    if "no tuition hold" in lowered or "hold is now cleared" in lowered or "tuition hold cleared" in lowered:
        return "tuition hold cleared"
    if "tuition hold still active" in lowered or "temporary tuition hold" in lowered:
        return "tuition hold active"
    if "hold is in final clearance" in lowered:
        return "tuition hold in final clearance"
    match = HOLD_STATUS_RE.search(content)
    if match:
        group = str(match.group(1) or "").strip().lower()
        if group == "cleared":
            return "tuition hold cleared"
        if group:
            return f"tuition hold {group}"
    return None


def _extract_current_percent(content: str) -> str | None:
    directional_patterns = [
        r"(?:approved (?:maximum )?discount|current approved maximum discount|commercial cap|discount is now|cap is now|maximum discount is now|revised approved [a-z]+ discount)\s*(?::|is|=)?\s*(\d{1,2}(?:\.\d+)?%)",
        r"(\d{1,2}(?:\.\d+)?%)\s*(?:after removing|after excluding|after trimming)",
    ]
    for pattern in directional_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    matches = [m.group(0).strip() for m in PERCENT_RE.finditer(content)]
    if not matches:
        return None
    if _has_current_cue(content):
        return matches[-1]
    return matches[0]


def _extract_safe_wording(content: str) -> str | None:
    candidate = extract_state_slots(content).get("safe_wording")
    if candidate:
        return candidate
    summary_match = re.search(
        r"\b(?:broad|safe)\s+(?:housing\s+)?wording\s+([A-Za-z][^,.;]+)",
        content,
        re.IGNORECASE,
    )
    if summary_match:
        return _clean_safe_wording_candidate(summary_match.group(1))
    return None


def _clean_safe_wording_candidate(value: str) -> str | None:
    candidate = str(value or "").strip(" \t\n\r\"'“”‘’,:;.-")
    if not candidate:
        return None
    for splitter in [" but not ", " do not ", " rather than ", " instead of "]:
        if splitter in candidate.lower():
            candidate = re.split(splitter, candidate, flags=re.IGNORECASE)[0].strip(" \t\n\r\"'“”‘’,:;.-")
            break
    if not candidate or candidate.startswith(","):
        return None
    if re.search(r"\b\d[\d,]*(?:\.\d+)?\s*USD\b", candidate, re.IGNORECASE):
        return None
    if len(candidate.split()) > 14:
        return None
    return candidate or None


def _extract_surface_spans(content: str, slots: dict[str, str]) -> dict[str, str]:
    spans: dict[str, str] = {}
    for slot_name, slot_value in slots.items():
        if not slot_value:
            continue
        match = re.search(re.escape(str(slot_value)), content, flags=re.IGNORECASE)
        spans[slot_name] = match.group(0) if match else str(slot_value)
    return spans


def _infer_subject_entity(frame_type: str, slots: dict[str, str], row: RetrievedEvidence) -> str | None:
    if "procedure" in slots:
        return slots["procedure"]
    if "provider" in slots:
        return slots["provider"]
    if frame_type == "allergy" and "substance" in slots:
        return slots["substance"]
    for entity in row.entities or []:
        if not _looks_like_slot_value_entity(entity):
            return entity
    return None


def _looks_like_slot_value_entity(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if MONEY_RE.fullmatch(text) or PERCENT_RE.fullmatch(text):
        return True
    if re.fullmatch(r"[\d\s,.:/\-]+", text):
        return True
    if re.fullmatch(
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+\d{1,2}(?:,\s*\d{4})?",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


def _looks_like_medication_frame_text(content: str) -> bool:
    lowered = content.lower()
    if any(token in lowered for token in ["medication", "medicine", "prescription", "regimen", "drug therapy"]):
        return True
    if MEDICATION_ACTION_RE.search(content):
        if MED_DOSAGE_RE.search(content):
            return True
        if any(
            token in lowered
            for token in ["daily", "nightly", "bedtime", "with food", "as needed", "prn", "tablet", "capsule", "mcg", "milligram"]
        ):
            return True
    return False


def _looks_like_support_state_summary(content: str) -> bool:
    lowered = content.lower()
    has_current_cue = any(
        token in lowered
        for token in [
            "current",
            "active",
            "current-state recap",
            "current state at this point",
            "financial-aid recap",
            "authorized recap",
            "safe recap",
        ]
    )
    has_support_signal = any(
        token in lowered
        for token in [
            "grant",
            "scholarship",
            "aid",
            "support amount",
            "approved amount",
            "tuition hold",
            "review memo",
            "broad wording",
            "safe wording",
            "orientation",
        ]
    )
    has_structured_payload = bool(
        _extract_current_money_amount(content)
        or _extract_hold_status(content)
        or _extract_latest_target_date(content)
        or SAFE_STATUS_WORDING_RE.search(content)
        or SAFE_BROAD_EXPLANATION_RE.search(content)
    )
    return has_current_cue and has_support_signal and has_structured_payload


def _looks_like_household_state_summary(content: str) -> bool:
    household_slots = extract_state_slots(content)
    return any(
        household_slots.get(key)
        for key in ["visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"]
    )


def _extract_medication_name(content: str) -> str | None:
    match = re.search(
        r"\b(?:start|stop|continue|hold|restart|use|take|avoid|switch to)\s+([A-Za-z][A-Za-z0-9'./\\-]*(?:\s+[A-Za-z0-9'./\\-]+){0,4})",
        content,
        re.IGNORECASE,
    )
    if not match:
        return None
    candidate = match.group(1).strip(" .,:;")
    for tail in ["for now", "right now", "because", "and", "with food", "at bedtime", "every six hours", "twice daily", "once daily"]:
        if candidate.lower().endswith(tail):
            candidate = candidate[: -len(tail)].strip(" .,:;")
    return candidate or None
