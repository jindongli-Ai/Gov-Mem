"""Append-only, checkpoint-safe event storage for Gov-Mem v8."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gov_mem.extraction.v8_event_schema import V8ExtractionBatch


@dataclass
class V8EventStore:
    conversation_id: str
    schema_version: str = "govmem-v8-event-store-2"
    memory_mode: str = "full"
    processed_turn_ids: list[str] = field(default_factory=list)
    scenes: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    governance_events: list[dict[str, Any]] = field(default_factory=list)
    extraction_audits: list[dict[str, Any]] = field(default_factory=list)

    def append(self, batch: V8ExtractionBatch, *, processed_turn_ids: list[str]) -> None:
        self._extend_unique(self.scenes, [asdict(item) for item in batch.scenes], "scene_id")
        self._extend_unique(self.entities, [asdict(item) for item in batch.entities], "entity_id")
        self._extend_unique(self.facts, [asdict(item) for item in batch.facts], "fact_id")
        self._extend_unique(self.relations, [asdict(item) for item in batch.relations], "relation_id")
        self._extend_unique(
            self.governance_events,
            [asdict(item) for item in batch.governance_events],
            "event_id",
        )
        # Backward-compatible bridge for early v8 batches that represented a
        # scene as an entity before the explicit scene contract was added.
        scene_rows = []
        for entity in batch.entities:
            if entity.entity_type == "scene":
                scene_rows.append({
                    "scene_id": entity.entity_id,
                    "canonical_name": entity.canonical_name,
                    "aliases": list(entity.aliases),
                    "source_spans": [asdict(span) for span in entity.source_spans],
                })
        self._extend_unique(self.scenes, scene_rows, "scene_id")
        for turn_id in processed_turn_ids:
            if turn_id and turn_id not in self.processed_turn_ids:
                self.processed_turn_ids.append(turn_id)
        self.extraction_audits.append(dict(batch.audit))

    @staticmethod
    def _extend_unique(target: list[dict[str, Any]], rows: list[dict[str, Any]], key: str) -> None:
        # Preserve observations of the same ID at different checkpoints. Merging
        # in storage would attach future participants/spans to an earlier scene.
        known = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in target}
        for row in rows:
            signature = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if row.get(key) and signature not in known:
                target.append(row)
                known.add(signature)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, *, conversation_id: str, memory_mode: str = "full") -> "V8EventStore":
        if not path.exists():
            return cls(conversation_id=conversation_id, memory_mode=memory_mode)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Invalid V8 event cache: {path}") from exc
        if str(payload.get("conversation_id") or "") != conversation_id:
            raise ValueError("V8 event cache conversation mismatch")
        if payload.get("schema_version") != "govmem-v8-event-store-2":
            raise ValueError("V8 implementation changed; use a fresh output directory")
        if payload.get("memory_mode", "full") != memory_mode:
            raise ValueError("V8 memory mode changed; use a fresh output directory")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in payload.items() if key in allowed})
