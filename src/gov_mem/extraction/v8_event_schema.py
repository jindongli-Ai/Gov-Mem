"""Typed contracts for Gov-Mem v8 scene and governance extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


V8_EVENT_SCHEMA_VERSION = "govmem-v8-event-schema-1"


@dataclass(frozen=True)
class V8SourceSpan:
    turn_id: str
    span: str


@dataclass
class V8Scene:
    scene_id: str
    scene_type: str
    canonical_name: str
    participant_principal_ids: list[str] = field(default_factory=list)
    status: str = "active"
    source_spans: list[V8SourceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class V8Entity:
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    scene_id: str | None = None
    source_spans: list[V8SourceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class V8Fact:
    fact_id: str
    scene_id: str | None
    subject_id: str | None
    field_name: str
    value: str
    temporal_state: str = "unknown"
    lifecycle: str = "assert"
    source_spans: list[V8SourceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class V8Relation:
    relation_id: str
    relation_type: str
    source_id: str
    target_id: str
    scene_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    source_spans: list[V8SourceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class V8GovernanceEvent:
    event_id: str
    event_type: str
    effect: str
    issuer_principal_id: str | None
    grantee_principal_id: str | None
    action: str
    resource_id: str
    resource_surface: str
    scene_id: str | None = None
    included_scopes: list[str] = field(default_factory=list)
    excluded_scopes: list[str] = field(default_factory=list)
    condition: str | None = None
    valid_from_turn_id: str | None = None
    valid_until: str | None = None
    source_spans: list[V8SourceSpan] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class V8ExtractionBatch:
    scenes: list[V8Scene] = field(default_factory=list)
    entities: list[V8Entity] = field(default_factory=list)
    facts: list[V8Fact] = field(default_factory=list)
    relations: list[V8Relation] = field(default_factory=list)
    governance_events: list[V8GovernanceEvent] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
