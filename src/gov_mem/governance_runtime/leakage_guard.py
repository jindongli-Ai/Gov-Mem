from __future__ import annotations

from dataclasses import replace
from typing import Any

from gov_mem.data.schema import MemoryInstance


HIDDEN_EVAL_KEYS = {
    "answer",
    "gold_answer",
    "gold_answers",
    "gold_evidence",
    "oracle_memory",
    "oracle_evidence",
    "gold_rationale",
    "rationale",
    "expected_action",
    "judge_spec",
    "leak_targets",
    "scorer_output",
    "query_type",
    "attack_type",
}


def hidden_eval_keys() -> set[str]:
    return set(HIDDEN_EVAL_KEYS)


def strip_hidden_eval_fields(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if str(key) in HIDDEN_EVAL_KEYS:
                continue
            cleaned[str(key)] = strip_hidden_eval_fields(value)
        return cleaned
    if isinstance(payload, list):
        return [strip_hidden_eval_fields(item) for item in payload]
    return payload


def contains_hidden_eval_fields(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in HIDDEN_EVAL_KEYS:
                return True
            if contains_hidden_eval_fields(value):
                return True
    elif isinstance(payload, list):
        return any(contains_hidden_eval_fields(item) for item in payload)
    return False


def hidden_eval_field_paths(payload: Any, *, _path: str = "payload") -> list[str]:
    """Return paths of evaluator-only keys found in a runtime payload."""
    paths: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            path = f"{_path}.{key_text}"
            if key_text in HIDDEN_EVAL_KEYS:
                paths.append(path)
            paths.extend(hidden_eval_field_paths(value, _path=path))
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            paths.extend(hidden_eval_field_paths(item, _path=f"{_path}[{index}]"))
    return paths


def assert_runtime_payload_safe(payload: Any, *, context: str) -> None:
    paths = hidden_eval_field_paths(payload)
    if paths:
        raise ValueError(
            f"Runtime payload leakage detected in {context} at {', '.join(paths[:8])}. "
            "Hidden evaluation fields must not enter runtime prompts."
        )


def runtime_instance_view(instance: MemoryInstance) -> MemoryInstance:
    """Create the runtime view while leaving evaluator metadata post-hoc only.

    The evaluator still receives the original adapter instance after the
    backbone finishes.  No answer, expected action, query category, judge
    specification, or leak target can enter a runtime backbone through this
    view.
    """
    cleaned_metadata = strip_hidden_eval_fields(dict(instance.metadata or {}))
    runtime = replace(instance, answer=None, metadata=cleaned_metadata)
    assert_runtime_payload_safe(runtime.metadata, context="runtime_instance_metadata")
    return runtime
