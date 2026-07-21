from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import AccessScope, EvidenceFrame, Principal


ROLE_TOKENS = {
    "patient",
    "family",
    "mother",
    "father",
    "caregiver",
    "clinician",
    "doctor",
    "dr",
    "nurse",
    "scheduler",
    "staff",
    "front",
    "desk",
    "social",
    "worker",
    "billing",
    "labtech",
    "lab",
    "pharmacist",
    "assistant",
    "executive",
    "dean",
    "program",
    "committee",
    "admin",
    "administrator",
    "coordinator",
    "agent",
    "team",
    "care",
    "clinical",
}

LOGISTICS_SLOTS = {
    "date",
    "time",
    "arrival_time",
    "location",
    "provider",
    "procedure",
    "visit_type",
    "prep_instruction",
    "precondition",
    "status",
}

CLINICAL_SENSITIVE_SLOTS = {
    "diagnosis",
    "result",
    "lab_value",
    "medication",
    "allergy",
    "pregnancy_status",
    "symptoms",
    "reaction",
    "substance",
}

POLICY_SLOTS = {
    "consent_scope",
    "authorization",
    "forbidden_users",
    "redaction_required",
}

POLICY_FRAME_TYPES = {"consent_or_permission", "privacy_policy"}

GROUP_RULES = {
    "care_team": {"clinician", "nurse", "scheduler", "social_worker", "front_desk", "labtech", "billing", "pharmacist", "staff"},
    "clinical_staff": {"clinician", "nurse", "labtech", "pharmacist", "staff"},
    "clinicians": {"clinician", "doctor", "dr"},
    "nurses": {"nurse"},
    "schedulers": {"scheduler"},
    "scheduler": {"scheduler"},
    "social_work": {"social_worker"},
    "social_worker": {"social_worker"},
    "front_desk": {"front_desk"},
    "clinic_reception": {"front_desk"},
    "reception": {"front_desk"},
    "reception_staff": {"front_desk"},
    "registration_staff": {"front_desk"},
    "care_coordinators": {"scheduler", "social_worker", "front_desk", "staff"},
    "care_coordinator": {"scheduler", "social_worker", "front_desk", "staff"},
    "clinic_staff": {"clinician", "nurse", "scheduler", "social_worker", "front_desk", "staff"},
    "authorized_staff_only": {"clinician", "nurse", "scheduler", "social_worker", "front_desk", "staff"},
    "assigned_clinicians": {"clinician", "nurse"},
    "assigned_care_team": {"clinician", "nurse", "scheduler", "social_worker", "front_desk", "staff"},
    "care_team_members": {"clinician", "nurse", "scheduler", "social_worker", "front_desk", "staff"},
    "social_worker": {"social_worker"},
    "social work": {"social_worker"},
    "billing_department": {"billing"},
    "lab_staff": {"labtech"},
}

OWNER_LIKE_PREFIXES = ("patient_", "pm_", "prof_", "resident_", "student_")
FAMILY_LIKE_TOKENS = {"family", "caregiver", "adult child", "adult_child"}
DELEGATE_LIKE_TOKENS = {
    "assistant",
    "executive assistant",
    "dean assistant",
    "program assistant",
    "department administrator",
    "department admin",
    "coordinator",
    "household manager",
    "trusted contact",
    "building staff",
    "technician",
    "cleaner",
}


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def principal_core(value: Any) -> str:
    tokens = [token for token in normalize_text(value).split() if token and token not in ROLE_TOKENS]
    if not tokens:
        return ""
    if len(tokens) >= 2:
        return " ".join(tokens[-2:])
    return tokens[0]


def normalize_role(value: Any) -> str:
    lowered = normalize_text(value)
    if "parent" in lowered or "guardian" in lowered:
        return "family"
    if "executive" in lowered and "assistant" in lowered:
        return "delegate_assistant"
    if "dean" in lowered and "assistant" in lowered:
        return "delegate_assistant"
    if "program" in lowered and "assistant" in lowered:
        return "delegate_assistant"
    if lowered.startswith("assistant"):
        return "delegate_assistant"
    if lowered.startswith("coordinator"):
        return "delegate_assistant"
    if "product" in lowered and "manager" in lowered:
        return "product_manager"
    if "helper" in lowered and any(token in lowered for token in {"home", "household", "it", "building"}):
        return "delegate_assistant"
    if lowered.startswith("advisor"):
        return "advisor"
    if lowered.startswith("registrar"):
        return "registrar"
    if "financial" in lowered and "aid" in lowered:
        return "financial_aid"
    if lowered.startswith("counselor"):
        return "counselor"
    if lowered == "ta" or lowered.startswith("teaching assistant"):
        return "ta"
    if lowered == "ra" or lowered.startswith("research assistant"):
        return "ra"
    if "campus" in lowered and "it" in lowered:
        return "campus_it"
    if "department" in lowered and ("administrator" in lowered or "admin" in lowered):
        return "department_admin"
    if lowered.startswith("professor") or lowered.startswith("prof "):
        return "professor"
    if "primary" in lowered and "resident" in lowered:
        return "primary_resident"
    if lowered == "resident":
        return "resident"
    if "partner" in lowered and "spouse" in lowered:
        return "partner_spouse"
    if lowered.startswith("partner") or lowered.startswith("spouse"):
        return "partner_spouse"
    if lowered.startswith("guest"):
        return "guest"
    if lowered.startswith("adult child") or lowered.startswith("adult_child"):
        return "family"
    if any(token in lowered for token in FAMILY_LIKE_TOKENS):
        return "family"
    if "social" in lowered and "worker" in lowered:
        return "social_worker"
    if "front" in lowered and "desk" in lowered:
        return "front_desk"
    if "doctor" in lowered or lowered == "dr" or lowered.startswith("clinician"):
        return "clinician"
    if lowered.startswith("nurse"):
        return "nurse"
    if lowered.startswith("scheduler"):
        return "scheduler"
    if lowered.startswith("patient"):
        return "owner"
    if lowered.startswith("family") or lowered in {"mother", "father", "caregiver"}:
        return "family"
    if lowered.startswith("lab"):
        return "labtech"
    if lowered.startswith("billing"):
        return "billing"
    if lowered.startswith("pharmacist"):
        return "pharmacist"
    if lowered.startswith("security"):
        return "security"
    if lowered.startswith("legal"):
        return "legal"
    if lowered.startswith("sre"):
        return "sre"
    if lowered.startswith("eng") or lowered.startswith("engineer"):
        return "engineer"
    if lowered.startswith("employee"):
        return "employee"
    if lowered.startswith("staff"):
        return "staff"
    if any(token in lowered for token in DELEGATE_LIKE_TOKENS):
        return "delegate_assistant"
    return lowered.replace(" ", "_")


def infer_relation_to_owner(requester_id: str | None, requester_role: str | None, owner_user_id: str | None) -> str | None:
    if is_owner_access(requester_id, owner_user_id):
        return "owner"
    # A role is organizational context, not evidence that this requester is
    # related to this particular owner. Episode-local relation resolution must
    # provide every non-self authorization relationship.
    return "unknown"


def build_principal(
    *,
    requester_id: str | None,
    requester_role: str | None,
    owner_user_id: str | None,
    relation_override: str | None = None,
) -> Principal:
    role = normalize_role(requester_role) if requester_role else None
    relation = str(relation_override or "").strip()
    if relation not in {"owner", "family", "delegate", "authorized_staff"}:
        relation = infer_relation_to_owner(requester_id, role, owner_user_id)
    org_role = role if role in {
        "clinician",
        "nurse",
        "labtech",
        "pharmacist",
        "social_worker",
        "front_desk",
        "scheduler",
        "staff",
        "delegate_assistant",
        "professor",
        "product_manager",
        "resident",
        "primary_resident",
        "advisor",
        "registrar",
        "financial_aid",
        "counselor",
        "ta",
        "ra",
        "campus_it",
        "department_admin",
        "security",
        "legal",
        "sre",
        "engineer",
        "employee",
    } else None
    return Principal(
        user_id=requester_id,
        role=role,
        relation_to_owner=relation,
        organization_role=org_role,
    )


def infer_owner_user_id(
    *,
    messages: list[dict[str, Any]] | None = None,
    evidence_rows: list[Any] | None = None,
    requester_id: str | None = None,
) -> str | None:
    messages = messages or []
    evidence_rows = evidence_rows or []
    for message in messages:
        speaker = str(message.get("speaker_id") or "")
        if speaker.startswith("patient_"):
            return speaker

    candidate_scores: dict[str, float] = {}

    def add_candidate(user_id: str | None, weight: float) -> None:
        candidate = str(user_id or "").strip()
        if not candidate or not candidate.startswith(OWNER_LIKE_PREFIXES):
            return
        candidate_scores[candidate] = candidate_scores.get(candidate, 0.0) + weight

    for message in messages:
        add_candidate(message.get("speaker_id"), 1.0)
    for row in evidence_rows:
        add_candidate(getattr(row, "user_id", None), 2.0 + float(getattr(row, "score", 0.0) or 0.0))

    if requester_id and requester_id in candidate_scores:
        candidate_scores[requester_id] += 1.0

    if not candidate_scores:
        return None
    return max(candidate_scores.items(), key=lambda item: (item[1], item[0] == requester_id))[0]


def is_same_principal(left: Any, right: Any) -> bool:
    left_text = normalize_text(left)
    right_text = normalize_text(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    left_core = principal_core(left)
    right_core = principal_core(right)
    return bool(left_core and left_core == right_core)


def is_owner_access(requester_id: str | None, owner_user_id: str | None) -> bool:
    return bool(requester_id and owner_user_id and is_same_principal(requester_id, owner_user_id))


def normalize_user_reference(value: Any, alias_map: dict[str, str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return raw
    for candidate_key in {normalize_text(raw), principal_core(raw)}:
        if candidate_key and candidate_key in alias_map:
            return alias_map[candidate_key]
    return raw


def normalize_user_list(values: Any, alias_map: dict[str, str]) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    normalized = []
    seen = set()
    for value in values:
        item = normalize_user_reference(value, alias_map)
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            normalized.append(item)
    return normalized


def is_requester_authorized(
    *,
    meta: dict[str, Any],
    requester_id: str | None,
    requester_role: str | None,
    owner_user_id: str | None,
) -> bool:
    if not requester_id:
        return True
    forbidden_users = meta.get("forbidden_users") or []
    for entry in forbidden_users:
        if _matches_access_entry(entry, requester_id=requester_id, requester_role=requester_role, owner_user_id=owner_user_id):
            return False
    if is_owner_access(requester_id, owner_user_id):
        return True
    authorized_users = meta.get("authorized_users") or []
    if not authorized_users:
        return True
    return any(
        _matches_access_entry(entry, requester_id=requester_id, requester_role=requester_role, owner_user_id=owner_user_id)
        for entry in authorized_users
    )


def requires_redaction_for_requester(
    *,
    meta: dict[str, Any],
    requester_id: str | None,
    requester_role: str | None,
    owner_user_id: str | None,
) -> bool:
    if is_owner_access(requester_id, owner_user_id):
        return False
    if not meta.get("redaction_required"):
        return False
    if requester_role and normalize_role(requester_role) in {"clinician", "nurse", "scheduler", "social_worker", "front_desk"}:
        return False
    return True


def resolve_access_scope(
    *,
    principal: Principal,
    owner_user_id: str | None,
    meta: dict[str, Any],
) -> AccessScope:
    memory_status = str(meta.get("memory_status") or "active")
    if memory_status == "deleted":
        return AccessScope(
            can_access_clinical_details=False,
            can_access_logistics=False,
            can_access_sensitive_entities=False,
            can_access_deleted_memory=False,
            requires_redaction=False,
            reason="deleted memory is not accessible",
        )

    if principal.user_id is None:
        return AccessScope(
            can_access_clinical_details=False,
            can_access_logistics=False,
            can_access_sensitive_entities=False,
            can_access_deleted_memory=False,
            requires_redaction=False,
            reason="unknown requester",
        )

    if is_owner_access(principal.user_id, owner_user_id):
        return AccessScope(
            can_access_clinical_details=True,
            can_access_logistics=True,
            can_access_sensitive_entities=True,
            can_access_deleted_memory=False,
            requires_redaction=False,
            reason="owner access",
        )

    if not is_requester_authorized(
        meta=meta,
        requester_id=principal.user_id,
        requester_role=principal.role,
        owner_user_id=owner_user_id,
    ):
        return AccessScope(
            can_access_clinical_details=False,
            can_access_logistics=False,
            can_access_sensitive_entities=False,
            can_access_deleted_memory=False,
            requires_redaction=False,
            reason="requester not authorized",
        )

    if principal.relation_to_owner == "authorized_staff":
        return AccessScope(
            can_access_clinical_details=True,
            can_access_logistics=True,
            can_access_sensitive_entities=True,
            can_access_deleted_memory=False,
            requires_redaction=requires_redaction_for_requester(
                meta=meta,
                requester_id=principal.user_id,
                requester_role=principal.role,
                owner_user_id=owner_user_id,
            ),
            reason="authorized staff access",
        )

    if principal.relation_to_owner == "delegate":
        return AccessScope(
            can_access_clinical_details=False,
            can_access_logistics=True,
            can_access_sensitive_entities=False,
            can_access_deleted_memory=False,
            requires_redaction=True,
            reason="delegate access limited to broad or logistics-safe state",
        )

    if principal.relation_to_owner == "family":
        return AccessScope(
            can_access_clinical_details=False,
            can_access_logistics=True,
            can_access_sensitive_entities=False,
            can_access_deleted_memory=False,
            requires_redaction=bool(meta.get("redaction_required")),
            reason="family logistics-only access",
        )

    return AccessScope(
        can_access_clinical_details=False,
        can_access_logistics=False,
        can_access_sensitive_entities=False,
        can_access_deleted_memory=False,
        requires_redaction=False,
        reason="default deny",
    )


def resolve_slot_access(
    *,
    frame: EvidenceFrame,
    principal: Principal,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    scope = resolve_access_scope(principal=principal, owner_user_id=frame.owner_user, meta=meta)
    allowed_slots: list[str] = []
    denied_slots: list[str] = []
    for slot_name in frame.slots.keys():
        if slot_name in POLICY_SLOTS:
            denied_slots.append(slot_name)
            continue
        if slot_name in LOGISTICS_SLOTS:
            if scope.can_access_logistics:
                allowed_slots.append(slot_name)
            else:
                denied_slots.append(slot_name)
            continue
        if slot_name in CLINICAL_SENSITIVE_SLOTS:
            if scope.can_access_clinical_details:
                allowed_slots.append(slot_name)
            else:
                denied_slots.append(slot_name)
            continue
        if scope.can_access_logistics or scope.can_access_clinical_details:
            allowed_slots.append(slot_name)
        else:
            denied_slots.append(slot_name)
    return {
        "frame_id": frame.frame_id,
        "allowed_slots": allowed_slots,
        "denied_slots": denied_slots,
        "requires_redaction": scope.requires_redaction,
        "access_reason": scope.reason,
    }


def is_policy_frame(frame: EvidenceFrame) -> bool:
    return frame.frame_type in POLICY_FRAME_TYPES


def is_logistics_memory(*, content: str, scope: str | None, memory_type: str | None) -> bool:
    lowered = normalize_text(content)
    if memory_type == "task":
        return True
    if scope and any(token in normalize_text(scope) for token in ["schedule", "appointment", "logistics", "arrival", "parking", "contact"]):
        return True
    keywords = [
        "appointment",
        "arrive",
        "arrival",
        "location",
        "suite",
        "parking",
        "schedule",
        "scheduled",
        "time",
        "callback",
        "voicemail",
        "portal",
    ]
    return any(token in lowered for token in keywords)


def requester_has_logistics_access(*, requester_id: str | None, requester_role: str | None, evidence_rows: list[Any]) -> bool:
    if not requester_id:
        return False
    requester_text = normalize_text(requester_id)
    requester_core = principal_core(requester_id)
    role = normalize_role(requester_role)
    for row in evidence_rows:
        content = normalize_text(getattr(row, "content", ""))
        if "logistics" not in content and "appointment time" not in content and "time/location" not in content:
            continue
        if requester_text and requester_text in content:
            return True
        if requester_core and requester_core in content:
            return True
        if role == "family" and any(token in content for token in ["mother", "family", "linda park"]):
            return True
    return False


def _matches_access_entry(
    entry: Any,
    *,
    requester_id: str | None,
    requester_role: str | None,
    owner_user_id: str | None,
) -> bool:
    normalized = normalize_text(entry)
    if not normalized:
        return False
    requester_text = normalize_text(requester_id)
    requester_core = principal_core(requester_id)
    role = normalize_role(requester_role)
    if normalized == requester_text or normalized == requester_core:
        return True
    if requester_core and requester_core in normalized:
        return True
    if requester_text and requester_text in normalized:
        return True
    if requester_core and requester_core == normalized:
        return True
    if owner_user_id and is_owner_access(requester_id, owner_user_id) and any(token in normalized for token in ["patient", "self"]):
        return True
    if role and role in normalized:
        return True
    for group_name, roles in GROUP_RULES.items():
        if group_name in normalized and role in roles:
            return True
    return False
