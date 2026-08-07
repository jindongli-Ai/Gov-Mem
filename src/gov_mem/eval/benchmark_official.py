from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from gov_mem.data.checkpoint_benchmark import DOMAINS, PROJECT_ROOT

ALLOWED_ACTIONS = {"answer", "answer_redacted", "refuse", "no_memory"}


def _discover_official_benchmark_root() -> Path:
    third_party_root = PROJECT_ROOT / "third_party"
    if not third_party_root.exists():
        raise FileNotFoundError("Missing third_party directory for official benchmark scorer discovery.")
    for score_script in third_party_root.rglob("score_predictions.py"):
        if score_script.parent.name == "scripts" and score_script.parent.parent.name == "bench":
            return score_script.parents[2]
    raise FileNotFoundError("Could not discover official benchmark scorer root under third_party/.")


OFFICIAL_BENCHMARK_ROOT = _discover_official_benchmark_root()
OFFICIAL_SCORE_SCRIPT = OFFICIAL_BENCHMARK_ROOT / "bench" / "scripts" / "score_predictions.py"


def dataset_dir_for_domain(domain: str, *, data_root: Path | None = None) -> Path:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported checkpoint benchmark domain: {domain!r}. Expected one of {DOMAINS}.")
    if data_root is None:
        from gov_mem.data.checkpoint_benchmark import DATASET_ROOT

        base_dir = DATASET_ROOT
    else:
        base_dir = data_root
    path = base_dir / domain
    if not path.exists():
        raise FileNotFoundError(f"Missing local checkpoint benchmark domain directory: {path}")
    return path


def _normalize_prediction_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "checkpoint_id" not in row:
        raise ValueError("Prediction row is missing checkpoint_id.")

    if "output" in row:
        output = row["output"]
        if not isinstance(output, dict):
            raise ValueError("Prediction row field 'output' must be a JSON object.")
        normalized = dict(row)
    else:
        normalized = {
            "checkpoint_id": row["checkpoint_id"],
            "output": {
                "action": row.get("action", ""),
                "answer": row.get("answer", ""),
                "answer_structured": row.get("answer_structured") or {},
                "used_record_ids": row.get("used_record_ids") or [],
            },
        }
        extras = {
            k: v
            for k, v in row.items()
            if k not in {"checkpoint_id", "action", "answer", "answer_structured", "used_record_ids"}
        }
        # memory_audit is part of the official evaluation contract, not debug
        # metadata: the scorer needs it to inspect the actual answer context.
        memory_audit = extras.pop("memory_audit", None)
        if isinstance(memory_audit, dict):
            normalized["output"]["memory_audit"] = memory_audit
        if extras:
            normalized["output"]["debug_external"] = extras

    output = normalized["output"]
    action = str(output.get("action") or "")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(
            f"Prediction row {normalized['checkpoint_id']!r} has invalid action {action!r}. "
            f"Expected one of {sorted(ALLOWED_ACTIONS)}."
        )

    output.setdefault("answer", "")
    output.setdefault("answer_structured", {})
    output.setdefault("used_record_ids", [])
    return normalized


def load_and_validate_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    rows: List[Dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in predictions file {predictions_path} at line {line_number}"
                ) from exc
            rows.append(_normalize_prediction_row(obj))

    checkpoint_ids = [str(row["checkpoint_id"]) for row in rows]
    duplicates = _duplicates(checkpoint_ids)
    if duplicates:
        raise ValueError(f"Duplicate checkpoint_id values found: {duplicates[:10]}")

    return rows


def _duplicates(values: Iterable[str]) -> List[str]:
    seen = set()
    duplicates: List[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _prediction_context_audit_available(row: Dict[str, Any]) -> bool:
    output = row.get("output") if isinstance(row.get("output"), dict) else row
    audit = output.get("memory_audit") if isinstance(output, dict) else None
    prompt_context = audit.get("prompt_context") if isinstance(audit, dict) else None
    return isinstance(prompt_context, dict) and "text" in prompt_context


def _annotate_context_audit_coverage(*, predictions: List[Dict[str, Any]], out_dir: Path) -> None:
    """Record whether official context leakage metrics have full evidence.

    The vendored scorer keeps backward-compatible zero-valued fields for old
    prediction files. A Gov-Mem result must distinguish a real zero from a
    missing audit field, so the wrapper annotates coverage and marks context
    leakage as unknown when coverage is incomplete.
    """
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    total = len(predictions)
    audited = sum(1 for row in predictions if _prediction_context_audit_available(row))
    coverage = float(audited) / float(total) if total else 0.0
    summary["n_context_audited"] = audited
    summary["context_audit_coverage_rate"] = coverage
    summary["context_audit_status"] = "complete" if coverage >= 1.0 else "incomplete_or_unknown"
    if coverage < 1.0:
        for key in (
            "privacy_context_leakage_rate",
            "deletion_context_leakage_rate",
            "privacy_e2e_leakage_rate",
            "deletion_e2e_leakage_rate",
            "compliance_utility_e2e_score",
        ):
            if key in summary:
                summary[key] = None
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_official_scorer(
    *,
    domain: str,
    data_dir: Path | None = None,
    predictions_path: Path,
    out_dir: Path,
    use_llm_judge: bool = True,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    judge_temperature: float | None = None,
    judge_max_output_tokens: int | None = None,
    judge_api_base: str | None = None,
    judge_api_key_env: str | None = None,
    judge_api_key: str | None = None,
    judge_concurrency: int | None = None,
    resume_judge: bool = False,
    # GateMem paper main tables report judge-derived U/A/F without the
    # optional post-hoc action gate. Keep the stricter gate opt-in.
    gate_by_action: bool = False,
) -> subprocess.CompletedProcess[str]:
    resolved_data_dir = dataset_dir_for_domain(domain, data_root=data_dir.parent if data_dir is not None and data_dir.name == domain else data_dir)
    if not OFFICIAL_SCORE_SCRIPT.exists():
        raise FileNotFoundError(f"Official benchmark scoring script not found: {OFFICIAL_SCORE_SCRIPT}")

    cmd = [
        sys.executable,
        str(OFFICIAL_SCORE_SCRIPT),
        "--data_dir",
        str(resolved_data_dir),
        "--predictions",
        str(predictions_path),
        "--out_dir",
        str(out_dir),
    ]
    if gate_by_action:
        cmd.append("--gate_by_action")

    if use_llm_judge:
        cmd.append("--use_llm_judge")
        if judge_provider:
            cmd.extend(["--judge_provider", judge_provider])
        if judge_model:
            cmd.extend(["--judge_model", judge_model])
        if judge_temperature is not None:
            cmd.extend(["--judge_temperature", str(judge_temperature)])
        if judge_max_output_tokens is not None:
            cmd.extend(["--judge_max_output_tokens", str(judge_max_output_tokens)])
        if judge_api_base:
            cmd.extend(["--judge_api_base", judge_api_base])
        if judge_api_key_env:
            cmd.extend(["--judge_api_key_env", judge_api_key_env])
        if judge_concurrency is not None:
            cmd.extend(["--judge_concurrency", str(judge_concurrency)])
        if resume_judge:
            cmd.append("--resume_judge")

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Let the scorer use the system Python environment so it can resolve native deps like numpy.
    pythonpath_parts = [str(OFFICIAL_BENCHMARK_ROOT)]
    existing_pythonpath = env.get('PYTHONPATH')
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env['PYTHONPATH'] = os.pathsep.join(pythonpath_parts)
    if judge_api_key:
        env[judge_api_key_env or "YUNWU_API_KEY"] = judge_api_key

    result = subprocess.run(cmd, check=True, text=True, env=env)
    # Some command-construction smoke tests intentionally use a placeholder
    # predictions path. Audit coverage is only meaningful for a real file.
    if predictions_path.exists():
        _annotate_context_audit_coverage(
            predictions=load_and_validate_predictions(predictions_path),
            out_dir=out_dir,
        )
    return result
