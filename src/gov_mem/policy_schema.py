"""Structured schemas for Stateful Policy Reasoning.

This module is deliberately independent from the legacy retrieval/reranking
stack.  Policy reasoning operates on observable state and provenance; it does
not consume benchmark labels or scorer metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ABSTAIN = "ABSTAIN"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    FORGET = "FORGET"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"
    FORGOTTEN = "forgotten"
    INACCESSIBLE = "inaccessible"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class OperationKind(str, Enum):
    GRANT = "grant"
    REVOKE = "revoke"
    UPDATE = "update"
    DELETE = "delete"
    FORGET = "forget"
    SHARE = "share"
    ACCESS = "access"


@dataclass(frozen=True)
class Provenance:
    source_message_ids: tuple[str, ...] = ()
    timestamp: str | None = None
    turn_index: int | None = None
    evidence_text: str = ""


@dataclass(frozen=True)
class PrincipalState:
    principal_id: str
    role: str | None = None
    display_name: str | None = None
    relations: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    aliases: tuple[str, ...] = ()
    entity_type: str = "person"


@dataclass(frozen=True)
class MemoryItemState:
    memory_id: str
    owner: str | None
    subject: str | None
    scope: str | None
    content_ref: str
    created_at: str | None
    status: MemoryStatus = MemoryStatus.ACTIVE
    provenance: Provenance = field(default_factory=Provenance)
    topics: tuple[str, ...] = ()
    subject_principal_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PermissionState:
    policy_id: str
    grantor: str | None
    grantee: str | None
    operation: str
    target_memory_ids: tuple[str, ...] = ()
    target_subject: str | None = None
    scope: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    revoked: bool = False
    effect: str = "allow"
    specificity: int = 0
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True)
class OperationState:
    operation_id: str
    kind: OperationKind
    actor: str | None
    target_memory_ids: tuple[str, ...] = ()
    target_subject: str | None = None
    scope: str | None = None
    value: str | None = None
    effective_at: str | None = None
    provenance: Provenance = field(default_factory=Provenance)


@dataclass(frozen=True)
class PolicyState:
    principals: tuple[PrincipalState, ...]
    memory_items: tuple[MemoryItemState, ...]
    ownership_relations: tuple[dict[str, Any], ...]
    permission_relations: tuple[PermissionState, ...]
    delegation_relations: tuple[PermissionState, ...]
    revocation_relations: tuple[PermissionState, ...]
    memory_status: dict[str, MemoryStatus]
    operation_history: tuple[OperationState, ...]
    temporal_constraints: tuple[dict[str, Any], ...]
    scope_constraints: tuple[dict[str, Any], ...]
    provenance: tuple[Provenance, ...]
    as_of_turn_id: str | None = None


@dataclass(frozen=True)
class QueryIntent:
    requester: str | None
    target_subject: str | None
    target_scope: str | None
    requested_operation: str
    target_memory_ids: tuple[str, ...] = ()
    mentioned_entities: tuple[str, ...] = ()
    requested_topics: tuple[str, ...] = ()
    requested_attributes: tuple[str, ...] = ()
    sensitivity_topics: tuple[str, ...] = ()
    disclosure_mode: str = "unknown"
    confidence: float = 0.0
    uncertainty: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = ()
    mentioned_principal_ids: tuple[str, ...] = ()
    identity_ambiguous: bool = False
    identity_resolution_reason: str | None = None


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    requester: str | None
    target_subject: str | None
    requested_operation: str
    allowed_memory_ids: tuple[str, ...] = ()
    blocked_memory_ids: tuple[str, ...] = ()
    applicable_policy_ids: tuple[str, ...] = ()
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    transition_trace: tuple[str, ...] = ()
    uncertainty: float = 0.0
    abstain_reason: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    action: PolicyAction
    answer_text: str
    accessed_memory_ids: tuple[str, ...] = ()
    blocked_memory_ids: tuple[str, ...] = ()
    state_changes: tuple[dict[str, Any], ...] = ()
    audit_trace: tuple[str, ...] = ()
    delivery_action: str = "answer"


def schema_to_dict(value: Any) -> Any:
    """Serialize enums/dataclasses without leaking hidden evaluation fields."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: schema_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): schema_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [schema_to_dict(item) for item in value]
    return value
