from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class MemoryInstance:
    instance_id: str
    domain: str | None
    conversation_id: str | None
    messages: list[dict]
    question: str
    asking_user_id: str | None
    choices: list[str] | None
    answer: str | bool | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    memory_id: str
    instance_id: str
    user_id: str | None
    scope: str
    content: str
    memory_type: str
    entities: list[str]
    time: str | None
    source_message_ids: list[str]
    confidence: float
    privacy_level: str | None
    tags: list[str]
    memory_status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    query_type: str
    target_users: list[str]
    target_entities: list[str]
    required_memory_types: list[str]
    symbolic_filters: dict[str, Any]
    dense_queries: list[str]
    reasoning_ops: list[str]
    semantic_spec: dict[str, Any] = field(default_factory=dict)
    # Runtime-only telemetry explains whether semantic routing used the LLM contract.
    planning_trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedEvidence:
    memory_id: str
    content: str
    score: float
    retrieval_source: str
    reason: str
    user_id: str | None = None
    memory_type: str | None = None
    scope: str | None = None
    entities: list[str] = field(default_factory=list)
    time: str | None = None
    source_message_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Principal:
    user_id: str | None
    role: str | None
    relation_to_owner: str | None
    organization_role: str | None


@dataclass
class AccessScope:
    can_access_clinical_details: bool
    can_access_logistics: bool
    can_access_sensitive_entities: bool
    can_access_deleted_memory: bool
    requires_redaction: bool
    reason: str


@dataclass
class EvidenceFrame:
    frame_id: str
    memory_id: str
    source_message_ids: list[str]
    frame_type: str
    owner_user: str | None
    subject_entity: str | None
    lifecycle_status: str
    effective_time: str | None
    event_time: str | None
    slots: dict[str, Any]
    access_scope: dict[str, Any]
    sensitivity: dict[str, Any]
    surface_spans: dict[str, str]
    confidence: float
    discourse_act: str = "unknown"
    assertion_confidence: float = 0.0
    event_identity: dict[str, Any] = field(default_factory=dict)
    state_delta: dict[str, Any] = field(default_factory=dict)
    semantic_attributes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSlot:
    key: str
    value: str
    frame_ids: list[str]
    memory_ids: list[str]
    lifecycle_status: str
    effective_time: str | None
    confidence: float


@dataclass
class EventState:
    event_key: str
    frame_type: str
    subject_entity: str | None
    lifecycle_status: str
    slots: dict[str, Any]
    surface_spans: dict[str, str]
    frame_ids: list[str]
    memory_ids: list[str]
    effective_time: str | None
    confidence: float


@dataclass
class CurrentStateLedger:
    owner_user: str | None
    active_events: dict[str, EventState]
    canceled_events: dict[str, EventState]
    superseded_events: dict[str, EventState]
    deleted_events: dict[str, EventState]
    active_slots: dict[str, StateSlot]
    canceled_slots: dict[str, StateSlot]
    deleted_slots: dict[str, StateSlot]
    superseded_slots: dict[str, StateSlot]
    trace: list[str]


@dataclass
class ReasoningState:
    selected_evidence: list[RetrievedEvidence]
    reasoning_trace: list[str]
    conflicts: list[dict[str, Any]]
    conclusion_hint: str | None
    selected_frames: list[EvidenceFrame] = field(default_factory=list)
    current_state_ledger: dict[str, Any] = field(default_factory=dict)
    required_slot_plan: dict[str, Any] = field(default_factory=dict)
    slot_coverage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerStructured:
    action: str
    answer_type: str
    owner_user: str | None
    utility_frames: list[dict[str, Any]]
    active_schedule_items: list[dict[str, Any]]
    canceled_items: list[dict[str, Any]]
    instructions: list[dict[str, Any]]
    allergies: list[dict[str, Any]]
    medications: list[dict[str, Any]]
    regimen_items: list[dict[str, Any]]
    consent_constraints: list[dict[str, Any]]
    redactions: list[dict[str, Any]]
    unavailable_slots: list[str]
    evidence_memory_ids: list[str]
    confidence: float


@dataclass
class SlotAuditResult:
    required_slots: list[str]
    filled_slots: list[str]
    missing_slots: list[str]
    renderable_slots: list[str]
    unrenderable_slots: list[str]
    slot_to_surface: dict[str, str]
    slot_to_memory_ids: dict[str, list[str]]
    audit_trace: list[str]


@dataclass
class AnswerResult:
    prediction: str | bool | None
    answer_text: str
    used_memory_ids: list[str]
    reasoning_summary: str
    action: str = "answer"
    answer_structured: dict[str, Any] = field(default_factory=dict)
    redacted_memory_ids: list[str] = field(default_factory=list)
    refused_memory_ids: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceItem:
    experience_id: str
    dataset_name: str
    instance_id: str
    query_type: str
    success: bool
    failure_type: str | None
    lesson: str
    suggested_skill_update: str | None
    related_memory_ids: list[str]
    domain: str | None = None


@dataclass
class Skill:
    skill_id: str
    name: str
    stage: str
    instruction: str
    version: int
    source: str


@dataclass
class CaseResult:
    instance_id: str
    question: str
    gold_answer: str | bool | None
    prediction: str | bool | None
    correct: bool
    query_type: str | None
    used_memory_ids: list[str]
    failure_type: str | None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernedActionDecision:
    action: str
    answer_mode: str
    privacy_decision: str
    forgetting_decision: str | None
    evidence_memory_ids: list[str]
    rationale_summary: str


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value
