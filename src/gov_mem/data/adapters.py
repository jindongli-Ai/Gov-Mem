from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gov_mem.data.schema import MemoryInstance
from gov_mem.data.timestamps import normalize_message_timestamp
from gov_mem.governance_runtime.leakage_guard import strip_hidden_eval_fields
from gov_mem.utils.io import read_json, read_jsonl


@dataclass
class DatasetBundle:
    dataset_name: str
    instances: list[MemoryInstance]
    metadata: dict[str, Any]


class BaseDatasetAdapter:
    def load(
        self,
        data_path: str | Path,
        max_instances: int | None = None,
        start_index: int = 0,
        checkpoint_ids: set[str] | None = None,
    ) -> DatasetBundle:
        raise NotImplementedError


class CheckpointBenchmarkAdapter(BaseDatasetAdapter):
    def __init__(self, *, use_asking_user_id: bool = True):
        self.use_asking_user_id = use_asking_user_id

    def load(
        self,
        data_path: str | Path,
        max_instances: int | None = None,
        start_index: int = 0,
        checkpoint_ids: set[str] | None = None,
    ) -> DatasetBundle:
        data_dir = Path(data_path)
        if data_dir.is_file():
            data_dir = data_dir.parent
        episodes_path = data_dir / "episodes.jsonl"
        checkpoints_path = data_dir / "checkpoints.jsonl"
        if not episodes_path.exists() or not checkpoints_path.exists():
            raise FileNotFoundError(
                f"Checkpoint benchmark adapter expected episodes.jsonl and checkpoints.jsonl under {data_dir}"
            )

        domain = data_dir.name
        episodes = {row["episode_id"]: row for row in read_jsonl(episodes_path)}
        checkpoints = read_jsonl(checkpoints_path)
        instances: list[MemoryInstance] = []

        for checkpoint_idx, checkpoint in enumerate(checkpoints):
            checkpoint_id = str(checkpoint.get("checkpoint_id") or "")
            if checkpoint_ids is not None and checkpoint_id not in checkpoint_ids:
                continue
            if checkpoint_idx < start_index:
                continue
            episode = episodes[checkpoint["episode_id"]]
            raw_sample = {
                "episode": episode,
                "checkpoint": checkpoint,
                "domain": domain,
            }
            instance = self.build_observable_instance(raw_sample)
            instances.append(instance)
            if max_instances is not None and len(instances) >= max_instances:
                break

        return DatasetBundle(
            dataset_name="checkpoint_benchmark",
            instances=instances,
            metadata={
                "domain": domain,
                "data_dir": str(data_dir.resolve()),
                "supports_official_benchmark": True,
                "start_index": start_index,
                "max_instances": max_instances,
                "total_checkpoints": len(checkpoints),
                "filtered_checkpoint_count": len(instances) if checkpoint_ids is not None else None,
            },
        )

    def build_observable_instance(self, raw_sample) -> MemoryInstance:
        episode = raw_sample["episode"]
        checkpoint = raw_sample["checkpoint"]
        domain = raw_sample["domain"]
        visible_messages = self._visible_messages_until_checkpoint(
            turns=episode.get("turns", []),
            as_of_turn_id=str(checkpoint["as_of_turn_id"]),
        )
        asking_user_id = None
        if self.use_asking_user_id:
            asking_user_id = ((checkpoint.get("asker") or {}).get("principal_id"))

        requester = checkpoint.get("asker") or {}
        raw_episode = {
            "episode_id": episode.get("episode_id"),
            "domain": episode.get("domain"),
            "entities": episode.get("entities"),
        }
        observable_metadata = {
            "conversation_id": str(checkpoint["episode_id"]),
            "domain": domain,
            "as_of_turn_id": str(checkpoint["as_of_turn_id"]),
            "requester": {
                "principal_id": requester.get("principal_id"),
                "role": requester.get("role"),
            },
        }
        metadata = {
            "dataset_name": "checkpoint_benchmark",
            "domain": domain,
            "conversation_id": str(checkpoint["episode_id"]),
            "runtime_profile": {
                "use_action_decision": True,
                "use_structured_answering": True,
                "use_text_answer_evaluation": True,
                "apply_checkpoint_updates": True,
            },
            "benchmark_profile": {
                "benchmark_name": "checkpoint_benchmark",
                "official_eval_family": "checkpoint_benchmark",
            },
            "requester": {
                "principal_id": requester.get("principal_id"),
                "role": requester.get("role"),
            },
            "authorized_user": requester.get("principal_id"),
            "observable_metadata": strip_hidden_eval_fields(observable_metadata),
            "raw_sample": {
                "episode": strip_hidden_eval_fields(raw_episode),
                "checkpoint_observable": strip_hidden_eval_fields({
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "episode_id": checkpoint.get("episode_id"),
                    "as_of_turn_id": checkpoint.get("as_of_turn_id"),
                    "asker": checkpoint.get("asker"),
                    "query_text": checkpoint.get("query_text"),
                }),
            },
            "evaluation": {
                "checkpoint_id": str(checkpoint["checkpoint_id"]),
                "query_type": checkpoint.get("query_type"),
                "attack_type": checkpoint.get("attack_type"),
                "expected_action": checkpoint.get("expected_action"),
                "judge_spec": checkpoint.get("judge_spec"),
                "leak_targets": checkpoint.get("leak_targets"),
            },
            "observable": {
                "as_of_turn_id": str(checkpoint["as_of_turn_id"]),
                "asker_role": requester.get("role"),
            },
        }
        return MemoryInstance(
            instance_id=str(checkpoint["checkpoint_id"]),
            domain=domain,
            conversation_id=str(checkpoint["episode_id"]),
            messages=visible_messages,
            question=str(checkpoint.get("query_text") or ""),
            asking_user_id=asking_user_id,
            choices=None,
            answer=None,
            metadata=metadata,
        )

    @staticmethod
    def _visible_messages_until_checkpoint(turns: list[dict], as_of_turn_id: str) -> list[dict]:
        target = str(as_of_turn_id or "").strip()
        if not target:
            raise ValueError("Checkpoint is missing as_of_turn_id; refusing to expose episode history.")

        turn_ids = [str(turn.get("turn_id") or "") for turn in turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("Episode contains duplicate turn_id values; refusing unsafe time truncation.")
        if target not in turn_ids:
            raise ValueError(
                f"as_of_turn_id={target!r} was not found in the episode; "
                "refusing to expose the full episode as a fallback."
            )

        visible: list[dict] = []
        for turn in turns:
            normalized_turn = normalize_message_timestamp(dict(turn))
            speaker = turn.get("speaker") or {}
            source_turn = dict(normalized_turn)
            visible.append(
                {
                    # Retain the checkpoint-visible GateMem turn, changing
                    # only timestamp precision so future source fields remain
                    # available without natural-language reconstruction.
                    "source_turn": source_turn,
                    "turn_id": str(turn.get("turn_id")),
                    "message_id": str(turn.get("turn_id")),
                    "speaker_id": speaker.get("principal_id"),
                    "speaker_role": speaker.get("role"),
                    "text": str(turn.get("text") or ""),
                    "timestamp": normalized_turn.get("timestamp"),
                    "turn_kind": turn.get("turn_kind"),
                }
            )
            if str(turn.get("turn_id")) == target:
                break
        return visible


class GenericJsonAdapter(BaseDatasetAdapter):
    def __init__(self, *, field_map: dict[str, str] | None = None, use_asking_user_id: bool = True):
        self.field_map = field_map or {}
        self.use_asking_user_id = use_asking_user_id

    def load(
        self,
        data_path: str | Path,
        max_instances: int | None = None,
        start_index: int = 0,
        checkpoint_ids: set[str] | None = None,
    ) -> DatasetBundle:
        path = Path(data_path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows = read_jsonl(path)
        elif suffix == ".json":
            raw = read_json(path)
            rows = raw if isinstance(raw, list) else raw.get("data", [])
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        else:
            raise ValueError(f"Unsupported data file type for generic adapter: {path}")

        instances: list[MemoryInstance] = []
        for idx, row in enumerate(rows):
            row_id = str(row.get(self._field("instance_id"), idx))
            if checkpoint_ids is not None and row_id not in checkpoint_ids:
                continue
            if idx < start_index:
                continue
            row = dict(row)
            messages = [
                normalize_message_timestamp(dict(message)) if isinstance(message, dict) else message
                for message in list(row.get(self._field("messages"), []))
            ]
            instance = MemoryInstance(
                instance_id=row_id,
                domain=row.get(self._field("domain")),
                conversation_id=row.get(self._field("conversation_id")),
                messages=messages,
                question=str(row.get(self._field("question"), "")),
                asking_user_id=(
                    row.get(self._field("asking_user_id")) if self.use_asking_user_id else None
                ),
                choices=row.get(self._field("choices")),
                answer=row.get(self._field("answer")),
                metadata=row.get(self._field("metadata"), {}) or {},
            )
            instances.append(instance)
            if max_instances is not None and len(instances) >= max_instances:
                break

        return DatasetBundle(
            dataset_name="generic",
            instances=instances,
            metadata={
                "data_path": str(path.resolve()),
                "format": suffix.lstrip("."),
                "start_index": start_index,
                "max_instances": max_instances,
                "total_rows": len(rows),
                "filtered_checkpoint_count": len(instances) if checkpoint_ids is not None else None,
            },
        )

    def _field(self, name: str) -> str:
        return self.field_map.get(name, name)


def build_dataset_adapter(
    dataset_name: str,
    *,
    use_asking_user_id: bool = True,
    field_map: dict[str, str] | None = None,
) -> BaseDatasetAdapter:
    key = dataset_name.strip().lower()
    if key in {"checkpoint_benchmark", "checkpoint-benchmark"}:
        return CheckpointBenchmarkAdapter(use_asking_user_id=use_asking_user_id)
    return GenericJsonAdapter(field_map=field_map, use_asking_user_id=use_asking_user_id)
