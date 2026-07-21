from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from gov_mem.query_semantics import (
    infer_current_state_domain as shared_infer_current_state_domain,
    infer_current_state_slots as shared_infer_current_state_slots,
    infer_household_composite_required_slots as shared_infer_household_composite_required_slots,
    infer_household_slots as shared_infer_household_slots,
)


@dataclass
class AnswerNeedSpec:
    need_types: list[str] = field(default_factory=list)
    required_record_types: list[str] = field(default_factory=list)
    required_slot_groups: list[str] = field(default_factory=list)
    optional_slot_groups: list[str] = field(default_factory=list)
    required_coverage_families: list[str] = field(default_factory=list)
    current_state_required: bool = False
    cancellation_required: bool = False
    policy_context_required: bool = False
    minimality_policy: str = "answer_only_requested_domains"
    trace: list[str] = field(default_factory=list)


def build_answer_need_spec(
    question: str,
    requester_context: dict,
    projected_evidence_lines: list[dict],
    semantic_intent: dict | None,
    config: dict,
) -> AnswerNeedSpec:
    lowered = str(question or "").lower()
    semantic_intent = semantic_intent or {}
    semantic_spec = dict(semantic_intent.get("semantic_spec") or {})
    semantic_slots = [str(slot) for slot in list(semantic_spec.get("requested_slots") or []) if str(slot)]
    semantic_available = bool(semantic_slots) or any(
        str(semantic_spec.get(key) or "unspecified") != "unspecified"
        for key in ["temporal_scope", "disclosure_scope", "state_domain", "request_shape"]
    )
    spec = AnswerNeedSpec()

    def add_need(name: str) -> None:
        if name not in spec.need_types:
            spec.need_types.append(name)

    def add_record(name: str) -> None:
        if name not in spec.required_record_types:
            spec.required_record_types.append(name)

    def add_slot(name: str) -> None:
        if name not in spec.required_slot_groups:
            spec.required_slot_groups.append(name)

    def add_optional(name: str) -> None:
        if name not in spec.optional_slot_groups:
            spec.optional_slot_groups.append(name)

    def add_family(name: str) -> None:
        if name not in spec.required_coverage_families:
            spec.required_coverage_families.append(name)

    household_slot_names = {
        "date", "time", "location", "visit_window", "entry_method", "package_rule",
        "approved_areas", "parking_pass", "arrival_contact_rule",
    }
    current_slot_names = {
        "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
        "monthly_stipend", "safe_wording", "blocker", "access_room", "access_badge",
        "operational_result",
    }
    requested_household_slots = [slot for slot in semantic_slots if slot in household_slot_names]
    if not semantic_available:
        requested_household_slots = _infer_household_slot_groups(lowered)
    requested_household_composite_slots = [
        str(slot).split(".", 1)[-1]
        for slot in shared_infer_household_composite_required_slots(question)
        if str(slot)
    ]
    household_plan_request = bool(requested_household_slots or requested_household_composite_slots)
    current_state_slots = [slot for slot in semantic_slots if slot in current_slot_names]
    if not semantic_available:
        current_state_slots = _infer_current_state_slot_groups(lowered)
    current_state_request = bool(current_state_slots)

    if current_state_request:
        state_domain = str(semantic_spec.get("state_domain") or "unspecified")
        if state_domain == "unspecified":
            state_domain = shared_infer_current_state_domain(question)
        project_state_slots = [slot for slot in current_state_slots if slot in {"target_date", "approved_budget", "approved_discount_cap", "blocker"}]
        research_state_slots = [slot for slot in current_state_slots if slot in {"target_date", "monthly_stipend", "safe_wording", "blocker"}]
        public_state_slots = [slot for slot in current_state_slots if slot in {"public_event_date"}]
        if project_state_slots and state_domain == "project":
            add_need("project_state")
            add_record("project_state")
            for key in project_state_slots:
                add_slot(key)
        if research_state_slots and state_domain == "research":
            add_need("research_state")
            add_record("research_state")
            for key in research_state_slots:
                add_slot(key)
        if public_state_slots:
            add_need("schedule")
            add_record("active_schedule")
            for key in public_state_slots:
                add_slot(key)

    if household_plan_request:
        add_need("household_plan")
        add_record("household_plan")
        household_slots = list(dict.fromkeys(requested_household_slots + requested_household_composite_slots))
        if not household_slots:
            household_slots = ["date", "visit_window"]
        if "approved_areas" in household_slots and "date" in household_slots and "visit_window" not in household_slots:
            household_slots.append("visit_window")
        if "approved_areas" in household_slots and any(
            token in lowered for token in ["logistics-only summary", "logistics only summary", "safely tell", "safe summary"]
        ):
            for key in ["parking_pass", "package_rule"]:
                if key not in household_slots:
                    household_slots.append(key)
        if spec.current_state_required:
            for key in ["date", "visit_window"]:
                if key not in household_slots:
                    household_slots.append(key)
        for key in household_slots:
            add_slot(key)
        if "approved_areas" not in household_slots:
            add_optional("approved_areas")

    if not household_plan_request and any(token in lowered for token in ["schedule", "appointment", "visit", "ultrasound", "follow-up", "follow up", "when", "where", "arrive"]):
        add_need("schedule")
        add_record("active_schedule")
        for key in ["date", "time", "location", "arrival_time", "provider", "procedure"]:
            add_slot(key)

    if str(semantic_spec.get("temporal_scope") or "") == "current" or (
        not semantic_available and any(token in lowered for token in ["current", "updated", "latest", "now", "still", "as of now", "currently"])
    ):
        spec.current_state_required = True
        add_need("current_plan")
    asks_current_clinical_plan = spec.current_state_required and any(
        token in lowered
        for token in [
            "treatment plan",
            "regimen",
            "medication plan",
            "biopsy-prep",
            "prep plan",
            "preparation",
            "what happens next",
            "follow-up plan",
            "follow up plan",
        ]
    )

    if any(token in lowered for token in ["cancel", "canceled", "cancelled", "rescheduled", "moved", "prior", "previous", "no longer"]):
        spec.cancellation_required = True
        add_need("cancellation")
        add_record("canceled_schedule")
        add_record("rescheduled_schedule")
        add_slot("status")

    if any(token in lowered for token in ["allergy", "allergic", "reaction", "rash"]):
        add_need("allergy")
        add_record("allergy")
        add_slot("allergy_substance")
        add_slot("allergy_reaction")

    regimen_family_hints = _infer_regimen_coverage_families(lowered, semantic_intent=semantic_intent)
    medication_request = bool(semantic_intent.get("current_regimen_request")) or _is_medical_regimen_request(lowered)
    if medication_request and _can_require_medication(lowered, current_state_request=current_state_request, household_plan_request=household_plan_request):
        add_need("medication")
        add_record("medication_status")
        for key in ["medication", "dosage", "instruction"]:
            add_slot(key)
        for family in regimen_family_hints:
            add_family(family)

    if asks_current_clinical_plan and not spec.required_record_types:
        if any(token in lowered for token in ["biopsy-prep", "prep plan", "preparation", "prep"]):
            add_need("instruction")
            add_record("instruction")
            add_slot("instruction")
            add_optional("condition")
            add_optional("date")
            add_optional("time")
            add_optional("location")
            add_optional("provider")
            add_optional("procedure")
            if _has_medication_prep_evidence(projected_evidence_lines):
                add_need("medication")
                add_record("medication_status")
                add_optional("medication")
                add_optional("dosage")
                add_optional("timing")
        elif any(token in lowered for token in ["treatment plan", "regimen", "medication plan"]):
            add_need("medication")
            add_record("medication_status")
            add_record("instruction")
            add_slot("medication")
            add_slot("instruction")
            add_optional("dosage")
            add_optional("condition")
            for family in regimen_family_hints:
                add_family(family)
        elif any(token in lowered for token in ["follow-up plan", "follow up plan", "what happens next", "next step"]):
            add_need("instruction")
            add_record("instruction")
            add_record("active_schedule")
            add_slot("instruction")
            add_optional("condition")
            add_optional("date")
            add_optional("time")
            add_optional("location")
            add_optional("provider")
            add_optional("procedure")

    if bool(semantic_intent.get("monitoring_plan_request")) or any(token in lowered for token in ["before", "still needs", "instruction", "unless", "need to", "precaution", "warning sign"]) or any(
        token in lowered for token in ["check blood pressure", "monitor blood pressure", "twice daily", "morning and evening"]
    ):
        add_need("instruction")
        add_record("instruction")
        for key in ["instruction", "condition"]:
            add_slot(key)

    if any(token in lowered for token in ["return precaution", "return urgently", "reinforce", "go to the ed", "go in right away"]):
        add_need("return_precaution")
        add_record("return_precaution")
        add_slot("condition")

    if any(token in lowered for token in ["call", "phone", "portal", "contact", "message", "pickup", "backup"]):
        add_need("contact_method")
        add_record("contact_method")
        for key in ["contact_method", "phone", "portal", "backup_contact"]:
            add_slot(key)

    if any(token in lowered for token in ["access setting", "family-access setting", "permission", "permissions", "logistics-only", "logistics only", "authorized"]):
        add_need("policy_permission")
        add_record("policy_permission")
        add_slot("policy_scope")
        spec.policy_context_required = True

    if len(spec.need_types) > 1:
        add_need("mixed")

    role = str((requester_context or {}).get("organization_role") or "").lower()
    relation = str((requester_context or {}).get("relation_to_owner") or "").lower()
    if relation in {"family", "caregiver", "proxy", "delegate"} or role in {"scheduler", "front_desk", "delegate_assistant"}:
        spec.minimality_policy = "logistics_only"
    elif role in {"clinician", "nurse", "pharmacist"} or relation == "owner":
        spec.minimality_policy = "clinical_allowed"

    if spec.policy_context_required and any(name in spec.need_types for name in ["schedule", "contact_method"]):
        spec.minimality_policy = "include_policy_plus_utility"
    if spec.current_state_required and spec.cancellation_required:
        spec.minimality_policy = "include_current_and_canceled"

    if any(line.get("line_meta", {}).get("frame_type") == "allergy" for line in projected_evidence_lines):
        add_optional("allergy_substance")
        add_optional("allergy_reaction")

    spec.trace.append(f"need_types={spec.need_types}")
    spec.trace.append(f"required_record_types={spec.required_record_types}")
    spec.trace.append(f"required_slot_groups={spec.required_slot_groups}")
    spec.trace.append(f"semantic_intent={semantic_intent}")
    spec.trace.append(f"semantic_spec={semantic_spec}")
    spec.trace.append(f"required_coverage_families={spec.required_coverage_families}")
    spec.trace.append(f"minimality_policy={spec.minimality_policy}")
    return spec


def _has_medication_prep_evidence(projected_evidence_lines: list[dict[str, Any]]) -> bool:
    for line in projected_evidence_lines:
        text = str(line.get("text") or "").lower()
        line_meta = dict(line.get("line_meta") or {})
        frame_type = str(line_meta.get("frame_type") or "")
        if frame_type == "medication":
            return True
        if not any(token in text for token in ["continue ", "stop ", "hold ", "avoid ", "take ", "restart ", "skip "]):
            continue
        if any(token in text for token in ["mg", "mcg", "tablet", "capsule", "ibuprofen", "levothyroxine", "apixaban", "metoprolol", "lisinopril", "biotin", "omeprazole", "magnesium"]):
            return True
    return False


def _is_medical_regimen_request(lowered: str) -> bool:
    medication_anchor_patterns = [
        r"\bmedication(?:s)?\b",
        r"\bmedicine(?:s)?\b",
        r"\bdose\b",
        r"\bdosage\b",
        r"\btreatment plan\b",
        r"\bregimen\b",
        r"\bapixaban\b",
        r"\bmetoprolol\b",
        r"\blisinopril\b",
        r"\bibuprofen\b",
        r"\bmagnesium\b",
        r"\bacetaminophen\b",
    ]
    if any(re.search(pattern, lowered) for pattern in medication_anchor_patterns):
        return True
    action_patterns = [
        r"\bstart\b",
        r"\bstop\b",
        r"\bcontinue\b",
        r"\bhold\b",
        r"\brestart\b",
        r"\buse\b",
        r"\btake\b",
        r"\bstay the same\b",
        r"\bkeep taking\b",
    ]
    current_patterns = [r"\bcurrent\b", r"\bright now\b", r"\bnow\b"]
    clinical_context_patterns = [
        r"\bfor pain\b",
        r"\bfor nausea\b",
        r"\bblood pressure\b",
        r"\bsymptoms?\b",
        r"\bmed(?:ication|icine)s?\b",
        r"\bdos(?:e|age)\b",
        r"\bprescription\b",
        r"\btablet\b",
        r"\bmg\b",
    ]
    return (
        any(re.search(pattern, lowered) for pattern in action_patterns)
        and any(re.search(pattern, lowered) for pattern in current_patterns)
        and any(re.search(pattern, lowered) for pattern in clinical_context_patterns)
    )


def _can_require_medication(lowered: str, *, current_state_request: bool, household_plan_request: bool) -> bool:
    if _has_clinical_regimen_context(lowered):
        return True
    if household_plan_request:
        return False
    if current_state_request and _infer_current_state_domain(lowered) in {"research", "project"}:
        return False
    return any(token in lowered for token in ["medication", "medicine", "regimen", "prescription"])


def _has_clinical_regimen_context(lowered: str) -> bool:
    clinical_tokens = [
        "medication",
        "medicine",
        "regimen",
        "dose",
        "dosage",
        "prescription",
        "for pain",
        "for nausea",
        "symptom",
        "symptoms",
        "blood pressure",
        "tablet",
        "capsule",
        "mg",
        "mcg",
    ]
    return any(token in lowered for token in clinical_tokens)


def _infer_regimen_coverage_families(lowered: str, semantic_intent: dict[str, Any] | None = None) -> list[str]:
    semantic_intent = semantic_intent or {}
    families: list[str] = []
    if any(token in lowered for token in ["stop", "what should i stop"]):
        families.append("stop")
    if any(token in lowered for token in ["start", "what should i start"]):
        families.append("start")
    if _question_explicitly_requests_continue_regimen(lowered):
        families.append("continue")
    if re.search(r"\buse\b", lowered) or "what should i use" in lowered or "what can i take" in lowered:
        families.append("use")
        if _question_implies_start_add_regimen(lowered):
            families.append("start")
    if any(token in lowered for token in ["monitor", "check", "blood pressure plan", "home blood pressure plan"]):
        families.append("monitor")
    if bool(semantic_intent.get("start_or_add_regimen_request")) and "start" not in families:
        families.append("start")
    if _should_require_continue_regimen_family(lowered, semantic_intent=semantic_intent, current_families=families) and "continue" not in families:
        families.append("continue")
    if bool(semantic_intent.get("monitoring_plan_request")) and "monitor" not in families:
        families.append("monitor")
    return families


def _question_explicitly_requests_continue_regimen(lowered: str) -> bool:
    return any(
        token in lowered
        for token in [
            "continue",
            "stay the same",
            "regular medicines stay the same",
            "keep taking",
            "keep using",
        ]
    )


def _should_require_continue_regimen_family(
    lowered: str,
    *,
    semantic_intent: dict[str, Any],
    current_families: list[str],
) -> bool:
    if _question_explicitly_requests_continue_regimen(lowered):
        return True
    if not bool(semantic_intent.get("continue_regimen_request")):
        return False
    if any(family in current_families for family in ["start", "stop", "use", "monitor"]):
        return False
    return any(
        token in lowered
        for token in [
            "current medications",
            "currently active medication",
            "currently active medications",
            "what medications are currently active",
            "which medications are currently active",
            "medication plan",
            "treatment plan",
            "regimen",
        ]
    )


def _question_implies_start_add_regimen(lowered: str) -> bool:
    if re.search(r"\b(?:what|which)\b.*\b(?:medication|medications|medicine|medicines)\b.*\b(?:use|take)\b.*\bfor\b", lowered):
        return True
    if re.search(r"\b(?:what|which)\b.*\b(?:use|take)\b.*\bfor\b", lowered) and any(
        token in lowered for token in ["pain", "nausea", "symptom", "symptoms", "treatment", "relief"]
    ):
        return True
    return False


def _infer_current_state_slot_groups(lowered_question: str) -> list[str]:
    return shared_infer_current_state_slots(lowered_question)


def _infer_current_state_domain(lowered_question: str) -> str:
    return shared_infer_current_state_domain(lowered_question)


def _infer_household_slot_groups(lowered_question: str) -> list[str]:
    return shared_infer_household_slots(lowered_question)
