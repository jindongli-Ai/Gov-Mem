from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def _pattern_hit(text: str, pattern: Any) -> bool:
    value = str(pattern or "").strip()
    if not value:
        return False
    try:
        # Dataset judge specifications may intentionally contain regular
        # expressions. Invalid expressions fall back to literal matching.
        if any(char in value for char in r"\[]()|?+*{}"):
            if re.search(value, text, flags=re.IGNORECASE):
                return True
    except re.error:
        pass
    return value.casefold() in str(text or "").casefold()


def _any_hit(text: str, patterns: Iterable[Any]) -> bool:
    return any(_pattern_hit(text, pattern) for pattern in patterns or [])


def _stage_contexts(audit: dict[str, Any]) -> dict[str, str]:
    """Return only memory/evidence context, excluding the user question."""

    contexts: dict[str, str] = {}
    stage2 = audit.get("stage2_rerank_prompt")
    stage2_values = stage2 if isinstance(stage2, list) else [stage2]
    for index, value in enumerate(stage2_values):
        if isinstance(value, dict) and "context_text" in value:
            label = str(value.get("stage") or "stage2_rerank")
            if label in contexts:
                label = f"{label}_{index}"
            contexts[label] = str(value.get("context_text") or "")
    for key, label in (("answer_prompt", "answer"),):
        value = audit.get(key)
        if isinstance(value, dict) and "context_text" in value:
            contexts[label] = str(value.get("context_text") or "")
    return contexts


def audit_prompt_contexts(
    *,
    audit_dir: str | Path,
    instances: Iterable[Any],
) -> dict[str, Any]:
    """Audit the exact context blocks sent to runtime LLM calls.

    Gold fields are read only after inference has finished. This function is
    therefore an evaluator-side audit, not a runtime policy component. Results
    are reported per checkpoint and at the checkpoint level so multiple LLM
    calls for one case cannot inflate the leakage denominator.
    """

    root = Path(audit_dir)
    rows: list[dict[str, Any]] = []
    for instance in instances:
        checkpoint_id = str(getattr(instance, "instance_id", ""))
        path = root / f"{checkpoint_id}.json"
        audit: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    audit = loaded
            except (OSError, json.JSONDecodeError):
                audit = {}

        evaluation = dict((getattr(instance, "metadata", {}) or {}).get("evaluation") or {})
        query_type = str(evaluation.get("query_type") or "")
        judge_spec = evaluation.get("judge_spec") if isinstance(evaluation.get("judge_spec"), dict) else {}
        privacy_patterns = list(judge_spec.get("not_include") or []) + list(evaluation.get("leak_targets") or [])
        deletion_patterns = list(evaluation.get("leak_targets") or [])
        contexts = _stage_contexts(audit)
        stage_rows = []
        for stage, context_text in contexts.items():
            stage_rows.append(
                {
                    "stage": stage,
                    "context_chars": len(context_text),
                    "privacy_leak": bool(query_type == "privacy" and _any_hit(context_text, privacy_patterns)),
                    "deletion_leak": bool(query_type == "safety" and _any_hit(context_text, deletion_patterns)),
                }
            )
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "query_type": query_type,
                "audit_present": bool(audit),
                "stages": stage_rows,
                "privacy_any_prompt_leak": any(row["privacy_leak"] for row in stage_rows),
                "deletion_any_prompt_leak": any(row["deletion_leak"] for row in stage_rows),
            }
        )

    privacy_rows = [row for row in rows if row["query_type"] == "privacy"]
    deletion_rows = [row for row in rows if row["query_type"] == "safety"]

    def rate(rows_to_score: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row.get(key)) for row in rows_to_score) / len(rows_to_score) if rows_to_score else 0.0

    def stage_rate(rows_to_score: list[dict[str, Any]], stage: str, key: str) -> float:
        selected = [
            row
            for row in rows_to_score
            if any(stage_row.get("stage") == stage for stage_row in row.get("stages", []))
        ]
        return rate(
            [
                {
                    key: any(
                        stage_row.get("stage") == stage and bool(stage_row.get(key))
                        for stage_row in row.get("stages", [])
                    )
                }
                for row in selected
            ],
            key,
        )

    return {
        "schema_version": 1,
        "metric_source": "exact_runtime_prompt_context_sidecars",
        "n_checkpoints": len(rows),
        "n_audited_checkpoints": sum(bool(row["audit_present"]) for row in rows),
        "audit_coverage_rate": rate(rows, "audit_present"),
        "privacy_context_leakage_rate": rate(privacy_rows, "privacy_any_prompt_leak"),
        "deletion_context_leakage_rate": rate(deletion_rows, "deletion_any_prompt_leak"),
        "privacy_answer_prompt_leakage_rate": stage_rate(privacy_rows, "answer", "privacy_leak"),
        "privacy_stage2_prompt_leakage_rate": stage_rate(privacy_rows, "stage2_rerank", "privacy_leak"),
        "deletion_answer_prompt_leakage_rate": stage_rate(deletion_rows, "answer", "deletion_leak"),
        "deletion_stage2_prompt_leakage_rate": stage_rate(deletion_rows, "stage2_rerank", "deletion_leak"),
        "rows": rows,
    }
