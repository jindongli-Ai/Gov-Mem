from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import AccessScope, EvidenceFrame, Principal


POLICY_FRAME_TYPES = {"consent_or_permission", "privacy_policy"}


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def principal_core(value: Any) -> str:
    # Principal identifiers are runtime-visible data. Do not remove pieces
    # using a predefined role vocabulary; preserve the normalized identifier.
    tokens = [token for token in normalize_text(value).split() if token]
    if not tokens:
        return ""
    if len(tokens) >= 2:
        return " ".join(tokens[-2:])
    return tokens[0]


def normalize_role(value: Any) -> str:
    # Role labels are episode-observable values. Canonical relation semantics
    # are supplied by the symbolic relationship graph; this helper only
    # normalizes the observed label and never maps through a fixed ontology.
    return normalize_text(value).replace(" ", "_")


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
    org_role = role
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
        message_role = normalize_text(message.get("speaker_role") or message.get("role"))
        if message.get("is_owner") is True or message_role in {"owner", "patient"}:
            return speaker or None

    candidate_scores: dict[str, float] = {}

    def add_candidate(user_id: str | None, weight: float) -> None:
        candidate = str(user_id or "").strip()
        if not candidate:
            return
        candidate_scores[candidate] = candidate_scores.get(candidate, 0.0) + weight

    for message in messages:
        message_role = normalize_text(message.get("speaker_role") or message.get("role"))
        add_candidate(message.get("speaker_id"), 3.0 if message_role in {"owner", "patient"} else 1.0)
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
    semantic_attributes = dict(getattr(frame, "semantic_attributes", {}) or {})
    for slot_name in frame.slots.keys():
        attribute = semantic_attributes.get(slot_name)
        sensitivity = {}
        if isinstance(attribute, dict):
            sensitivity = dict(
                attribute.get("sensitivity_semantics")
                or attribute.get("sensitivity")
                or {}
            )
        protected = str(sensitivity.get("type") or "").casefold() in {
            "restricted", "private", "confidential"
        }
        if scope.can_access_clinical_details or scope.can_access_sensitive_entities:
            allowed_slots.append(slot_name)
        elif scope.can_access_logistics and not protected:
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
    """Classify an operational carrier from structured metadata only.

    The v4 paper path must not rediscover semantic categories from a built-in
    natural-language keyword list. Ingestion/semantic compilation supplies the
    memory type and scope; raw content is intentionally ignored here.
    """
    del content
    if memory_type == "task":
        return True
    return normalize_text(scope) in {
        "schedule", "appointment", "logistics", "arrival", "parking", "contact",
    }


def requester_has_logistics_access(*, requester_id: str | None, requester_role: str | None, evidence_rows: list[Any]) -> bool:
    del requester_id, requester_role
    return any(
        bool((getattr(row, "metadata", {}) or {}).get("safe_operational_carrier"))
        for row in evidence_rows
    )


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
    # Group membership must be materialized by the runtime episode graph.
    # There is no built-in group-to-role vocabulary here.
    return False
