from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from gov_mem.data.schema import MemoryInstance
from gov_mem.memory.amem_memory import AtomicMemory


@dataclass
class GovernedMemoryAtom:
    atom_id: str
    atom_type: str
    text: str
    slots: dict[str, Any]
    owner_id: str | None
    subject_id: str | None
    speaker_id: str | None
    source_turn: int | None
    timestamp: str | None
    lifecycle: str
    sensitivity: str
    access_scope: list[str]
    related_entities: list[str]
    provenance: dict[str, Any]
    confidence: float
    discourse_act: str = "unknown"
    assertion_confidence: float = 0.0
    event_identity: dict[str, Any] = field(default_factory=dict)
    state_delta: dict[str, Any] = field(default_factory=dict)
    semantic_attributes: dict[str, Any] = field(default_factory=dict)


def governed_atom_to_dict(atom: GovernedMemoryAtom) -> dict[str, Any]:
    return asdict(atom)


def adapt_atomic_memories_to_governed_atoms(
    *,
    instance: MemoryInstance,
    atomic_memories: list[AtomicMemory],
    information_owner_by_message_id: dict[str, str] | None = None,
) -> list[GovernedMemoryAtom]:
    message_index_by_id: dict[str, int] = {}
    message_by_id: dict[str, dict[str, Any]] = {}
    for index, message in enumerate(instance.messages):
        message_id = str(message.get("message_id") or "")
        if not message_id:
            continue
        message_index_by_id[message_id] = index
        message_by_id[message_id] = message

    atoms: list[GovernedMemoryAtom] = []
    for memory in atomic_memories:
        source_message = _resolve_source_message(memory, message_by_id)
        source_turn = _resolve_source_turn(memory, message_index_by_id)
        semantic_tags = dict((memory.access_tags or {}).get("semantic_tags") or {})
        evidence_span = str(semantic_tags.get("evidence_span") or "").strip()
        # A verified complete span replaces a heuristic fragment as the fact
        # atom text, so every certified typed value is traceable to one
        # continuous source statement.
        text = evidence_span or str(memory.content or "").strip()
        semantic_attributes = dict(semantic_tags.get("attributes") or {})
        state_delta = dict(semantic_tags.get("state_delta") or {})
        changed_fields = state_delta.get("changed_fields")
        if isinstance(changed_fields, dict):
            for key, value in changed_fields.items():
                if value not in (None, "", []):
                    semantic_attributes.setdefault(str(key), value)
        slots = dict(memory.slots or {})
        if evidence_span:
            slots = {
                key: value
                for key, value in slots.items()
                if str(value).strip() and str(value).lower() in evidence_span.lower()
            }
        for key, value in semantic_attributes.items():
            if value not in (None, "", []):
                slots.setdefault(str(key), value)
        related_entities = _dedupe([*(memory.entities or []), *(str(v) for v in slots.values() if _looks_like_entity_value(v))])
        event_identity = dict(semantic_tags.get("event_identity") or {})
        subject_id = str(event_identity.get("entity_key") or "").strip() or _infer_subject_id(memory, related_entities)
        speaker_id = str((source_message or {}).get("speaker_id") or memory.owner_user or "") or None
        information_owner_id = _resolve_information_owner_id(
            memory,
            information_owner_by_message_id=information_owner_by_message_id,
        )
        provenance = {
            "instance_id": instance.instance_id,
            "source_memory_id": memory.memory_id,
            "source_message_ids": list(memory.source_message_ids or []),
            "source_message_text": (source_message or {}).get("text"),
            "source_speaker_id": speaker_id,
            "information_owner_id": information_owner_id,
            "frame_type": str((memory.access_tags or {}).get("frame_type") or memory.memory_type),
            "surface_spans": {
                **dict((memory.access_tags or {}).get("surface_spans") or {}),
                **dict(semantic_tags.get("surface_values") or {}),
            },
            "evidence_span": evidence_span or None,
            "grounded_claims": list(semantic_tags.get("claims") or []),
            "extraction_source": (memory.access_tags or {}).get("extraction_source"),
        }
        lifecycle = _structured_lifecycle(memory, text=text, semantic_tags=semantic_tags)
        if _is_mixed_current_state_deletion(memory, semantic_tags):
            # A source can update an active record while recording deletion of
            # a different prior value. Preserve only the active fields here;
            # the deletion event remains represented by lifecycle evidence,
            # but must not erase the current state as a whole.
            lifecycle = "active"
            slots = {
                key: value for key, value in slots.items()
                if not _is_deletion_marker(key, value)
            }
            semantic_attributes = {
                key: value for key, value in semantic_attributes.items()
                if not _is_deletion_marker(key, value)
            }
        base_kwargs = {
            "text": text,
            "slots": slots,
            "owner_id": information_owner_id,
            "subject_id": subject_id,
            "speaker_id": speaker_id,
            "source_turn": source_turn,
            "timestamp": memory.timestamp,
            "lifecycle": lifecycle,
            "sensitivity": _infer_sensitivity(memory, text=text, lifecycle=lifecycle),
            "access_scope": _infer_access_scope(memory, text=text),
            "related_entities": related_entities,
            "provenance": provenance,
            "confidence": float(memory.confidence),
            "discourse_act": str(semantic_tags.get("discourse_act") or "unknown"),
            "assertion_confidence": float(semantic_tags.get("assertion_confidence") or 0.0),
            "event_identity": dict(semantic_tags.get("event_identity") or {}),
            "state_delta": state_delta,
            "semantic_attributes": semantic_attributes,
        }
        primary_type = _infer_primary_atom_type(memory, text=text, lifecycle=lifecycle)
        atoms.append(
            GovernedMemoryAtom(
                atom_id=f"{memory.memory_id}::{primary_type}",
                atom_type=primary_type,
                **base_kwargs,
            )
        )
        for derived_type in _infer_derived_atom_types(
            memory, text=text, primary_type=primary_type, lifecycle=lifecycle,
        ):
            atoms.append(
                GovernedMemoryAtom(
                    atom_id=f"{memory.memory_id}::{derived_type}",
                    atom_type=derived_type,
                    **base_kwargs,
                )
            )
    return _dedupe_atoms(atoms)


def _resolve_source_message(memory: AtomicMemory, message_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for message_id in memory.source_message_ids or []:
        resolved = message_by_id.get(str(message_id))
        if resolved is not None:
            return resolved
    return None


def _resolve_information_owner_id(
    memory: AtomicMemory,
    *,
    information_owner_by_message_id: dict[str, str] | None,
) -> str | None:
    """Never use a reporting speaker as a protected-information owner."""
    if information_owner_by_message_id is None:
        return memory.owner_user
    owners = {
        str(information_owner_by_message_id.get(str(message_id)) or "").strip()
        for message_id in list(memory.source_message_ids or [])
        if str(information_owner_by_message_id.get(str(message_id)) or "").strip()
    }
    return next(iter(owners)) if len(owners) == 1 else None


def _resolve_source_turn(memory: AtomicMemory, message_index_by_id: dict[str, int]) -> int | None:
    turns = [message_index_by_id[str(message_id)] for message_id in memory.source_message_ids or [] if str(message_id) in message_index_by_id]
    if not turns:
        return None
    return min(turns)


def _infer_subject_id(memory: AtomicMemory, related_entities: list[str]) -> str | None:
    for entity in related_entities:
        entity_norm = str(entity).strip()
        if not entity_norm:
            continue
        if entity_norm == str(memory.owner_user or "").strip():
            continue
        return entity_norm
    return memory.owner_user


def _normalize_lifecycle(lifecycle_status: str | None, *, text: str) -> str:
    lowered = text.lower()
    raw = str(lifecycle_status or "").strip().lower()
    if raw in {"active", "canceled", "superseded", "deleted", "historical"}:
        return raw
    if _looks_like_deletion_text(lowered):
        return "deleted"
    if _looks_like_supersession_text(lowered):
        return "superseded"
    if any(token in lowered for token in ["canceled", "cancelled", "no longer active"]):
        return "canceled"
    if any(token in lowered for token in ["old ", "earlier ", "previous ", "historical"]):
        return "historical"
    return "active"


def _infer_governed_lifecycle(memory: AtomicMemory, *, text: str) -> str:
    lifecycle = _normalize_lifecycle(memory.lifecycle_status, text=text)
    if memory.memory_type == "forgetting" and lifecycle == "active":
        return "deleted"
    if memory.memory_type == "update" and lifecycle == "active" and _looks_like_supersession_text(text.lower()):
        return "superseded"
    return lifecycle


def _structured_lifecycle(
    memory: AtomicMemory,
    *,
    text: str,
    semantic_tags: dict[str, Any],
) -> str:
    lifecycle = _infer_governed_lifecycle(memory, text=text)
    discourse_act = str(semantic_tags.get("discourse_act") or "unknown").lower()
    state_delta = semantic_tags.get("state_delta")
    operation = str(state_delta.get("operation") or "none").lower() if isinstance(state_delta, dict) else "none"
    is_active_operation = (
        discourse_act in {"assertion", "update", "confirmation"}
        and operation in {"create", "set", "add", "supersede"}
    )
    if lifecycle == "superseded" and is_active_operation:
        return "active"
    if lifecycle == "deleted" and is_active_operation and _has_grounded_active_change(semantic_tags):
        return "active"
    return lifecycle


def _has_grounded_active_change(semantic_tags: dict[str, Any]) -> bool:
    """Require a record-local, non-deletion value before restoring its lifecycle."""
    state_delta = semantic_tags.get("state_delta")
    changed = state_delta.get("changed_fields") if isinstance(state_delta, dict) else None
    return isinstance(changed, dict) and any(
        value not in (None, "", []) and not _is_deletion_marker(key, value)
        for key, value in changed.items()
    )


def _is_mixed_current_state_deletion(memory: AtomicMemory, semantic_tags: dict[str, Any]) -> bool:
    state_delta = semantic_tags.get("state_delta")
    changed = state_delta.get("changed_fields") if isinstance(state_delta, dict) else None
    return (
        _infer_governed_lifecycle(memory, text=str(memory.content or "")) == "deleted"
        and str(semantic_tags.get("discourse_act") or "").lower() in {"assertion", "update", "confirmation"}
        and str(state_delta.get("operation") or "").lower() in {"set", "add", "create", "supersede"}
        and isinstance(changed, dict)
        and any(not _is_deletion_marker(key, value) for key, value in changed.items())
    )


def _is_deletion_marker(key: object, value: object) -> bool:
    key_text = str(key or "").lower()
    value_text = str(value or "").strip().lower()
    return key_text.endswith(("_deleted", "_deletion", "_removed", "_unavailable")) or value_text in {
        "deleted", "removed", "unavailable",
    }


def _infer_sensitivity(memory: AtomicMemory, *, text: str, lifecycle: str | None = None) -> str:
    lowered = text.lower()
    lifecycle = lifecycle or _infer_governed_lifecycle(memory, text=text)
    if lifecycle == "deleted":
        return "deleted"
    if any(token in lowered for token in ["private", "confidential", "do not share", "not authorized", "restricted"]):
        return "restricted"
    if memory.memory_type in {"medication", "allergy", "instruction", "return_precaution"}:
        return "medical_private"
    if memory.memory_type in {"policy_permission", "contact_method"}:
        return "restricted"
    if memory.memory_type in {"appointment", "clinic_visit", "test_or_imaging", "logistics"}:
        return "private"
    return "private"


def _infer_access_scope(memory: AtomicMemory, *, text: str) -> list[str]:
    lowered = text.lower()
    scopes: list[str] = []
    tag_text = jsonish_dump(memory.access_tags)
    combined = f"{lowered} {tag_text}".lower()
    scope_patterns = {
        "self": ["patient", "owner", "self only", "for me"],
        "family": ["family", "mother", "father", "spouse", "partner", "caregiver", "mom", "dad"],
        "clinician": ["clinician", "doctor", "nurse", "pharmacist"],
        "collaborator": ["collaborator", "coordinator", "assistant", "scheduler", "front desk"],
        "researcher": ["research", "study", "pi", "scholar", "grant"],
        "external": ["external", "vendor", "sponsor", "customer"],
    }
    for scope, tokens in scope_patterns.items():
        if any(_has_phrase(combined, token) for token in tokens):
            scopes.append(scope)
    if not scopes:
        scopes.append("self")
    return _dedupe(scopes)


def _infer_primary_atom_type(
    memory: AtomicMemory, *, text: str, lifecycle: str | None = None,
) -> str:
    lowered = text.lower()
    lifecycle = lifecycle or _infer_governed_lifecycle(memory, text=text)
    if lifecycle == "deleted":
        return "deletion_atom"
    if lifecycle == "superseded":
        return "supersession_atom"
    # A current record can carry a policy-like phrase alongside concrete
    # values. Keep its fact/event representation as the primary atom so Stage
    # 2 can adjudicate the record; a separate derived policy atom still
    # captures the governance statement below. Classifying the whole record
    # as policy would otherwise erase every answerable SlotNode before the
    # LLM-mediated reranker sees it.
    if any(value not in (None, "", []) for value in (memory.slots or {}).values()):
        if _looks_like_event_atom(memory, lowered):
            return "event_atom"
        return "fact_atom"
    if _looks_like_permission_text(lowered):
        return "permission_atom"
    if _looks_like_policy_text(lowered):
        return "policy_atom"
    if _looks_like_event_atom(memory, lowered):
        return "event_atom"
    if _looks_like_relation_text(lowered):
        return "relation_atom"
    if _looks_like_role_text(lowered):
        return "role_atom"
    return "fact_atom"


def _infer_derived_atom_types(
    memory: AtomicMemory,
    *,
    text: str,
    primary_type: str,
    lifecycle: str | None = None,
) -> list[str]:
    lowered = text.lower()
    derived: list[str] = []
    if _looks_like_permission_text(lowered) and primary_type != "permission_atom":
        derived.append("permission_atom")
    if _looks_like_policy_text(lowered) and primary_type != "policy_atom":
        derived.append("policy_atom")
    if _looks_like_relation_text(lowered) and primary_type != "relation_atom":
        derived.append("relation_atom")
    if _looks_like_role_text(lowered) and primary_type != "role_atom":
        derived.append("role_atom")
    lifecycle = lifecycle or _infer_governed_lifecycle(memory, text=text)
    if lifecycle == "deleted" and primary_type != "deletion_atom":
        derived.append("deletion_atom")
    if lifecycle == "superseded" and primary_type != "supersession_atom":
        derived.append("supersession_atom")
    return derived


def _looks_like_event_atom(memory: AtomicMemory, lowered: str) -> bool:
    slot_names = set((memory.slots or {}).keys())
    if slot_names & {"date", "time", "arrival_time", "location", "provider", "procedure", "visit_window"}:
        return True
    return memory.memory_type in {"appointment", "clinic_visit", "test_or_imaging", "logistics"}


def _looks_like_permission_text(lowered: str) -> bool:
    return any(
        _has_phrase(lowered, token)
        for token in [
            "authorized",
            "permission",
            "consent",
            "may receive",
            "can hear",
            "allowed to",
            "logistics only",
            "share with",
            "do not share",
        ]
    )


def _looks_like_policy_text(lowered: str) -> bool:
    return any(
        _has_phrase(lowered, token)
        for token in [
            "must",
            "should only",
            "policy",
            "protocol",
            "only if",
            "unless",
            "do not",
            "without naming",
            "generic wording",
            "portal only",
            "is shareable",
            "shareable only",
            "only the current",
            "not exact",
            "keep private",
            "keep my current private",
            "separate from public",
            "cannot share",
            "without assignment",
            "across services",
        ]
    )


def _looks_like_relation_text(lowered: str) -> bool:
    return any(
        _has_phrase(lowered, token)
        for token in [
            "my mom",
            "my mother",
            "my father",
            "my husband",
            "my wife",
            "my partner",
            "caregiver",
            "delegate",
            "driving her",
            "driving him",
            "family",
        ]
    )


def _looks_like_role_text(lowered: str) -> bool:
    return any(
        _has_phrase(lowered, token)
        for token in [
            "nurse",
            "doctor",
            "clinician",
            "reception",
            "front desk",
            "scheduler",
            "research coordinator",
            "pi",
            "assistant",
        ]
    )


def _looks_like_entity_value(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(re.search(r"[A-Za-z]", text)) and len(text.split()) <= 8


def _has_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(str(phrase).strip().lower())
    if not escaped:
        return False
    return re.search(rf"\b{escaped}\b", text) is not None


def _looks_like_deletion_text(lowered: str) -> bool:
    return any(
        _has_phrase(lowered, token)
        for token in [
            "delete",
            "deleted",
            "remove from memory",
            "removed from",
            "must not be reconstructed",
            "retired",
            "treated as deleted",
            "unavailable going forward",
        ]
    )


def _looks_like_supersession_text(lowered: str) -> bool:
    return any(
        _has_phrase(lowered, token)
        for token in [
            "supersede",
            "supersedes",
            "superseded",
            "replace",
            "replaces",
            "replaced",
            "updated to",
            "updated:",
            "is now",
            "moves from",
            "current approved",
            "current target",
            "current ceiling",
            "official pilot target",
            "interim",
            "placeholder",
            "stale draft",
            "instead of",
        ]
    )


def _dedupe_atoms(atoms: list[GovernedMemoryAtom]) -> list[GovernedMemoryAtom]:
    seen: set[tuple[str, str, str, str | None]] = set()
    out: list[GovernedMemoryAtom] = []
    for atom in atoms:
        key = (atom.atom_type, atom.text, atom.speaker_id or "", atom.timestamp)
        if key in seen:
            continue
        seen.add(key)
        out.append(atom)
    return out


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        norm = str(value).strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def jsonish_dump(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    parts: list[str] = []
    for key, value in payload.items():
        parts.append(f"{key}={value}")
    return " ".join(parts)
