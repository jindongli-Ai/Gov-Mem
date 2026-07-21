from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.utils.config import load_yaml_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_domain_summary(run_dir: Path, domain: str, suite_row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    summary_path = run_dir / domain / "official_eval" / "checkpoint_benchmark" / domain / "summary.json"
    if summary_path.exists():
        return _read_json(summary_path), str(summary_path)
    return dict(suite_row.get("summary") or {}), ""


def _aggregate_summary(run_dir: Path, suite_summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    per_domain: dict[str, Any] = {}
    total_checkpoints = 0
    total_action = 0.0
    total_utility_scored = 0
    total_utility_correct = 0.0
    total_privacy_scored = 0
    total_privacy_leaks = 0.0
    total_deletion_scored = 0
    total_deletion_leaks = 0.0

    for domain, row in (suite_summary.get("domains") or {}).items():
        row = dict(row or {})
        summary, resolved_summary_path = _load_domain_summary(run_dir, str(domain), row)
        checkpoints = int(summary.get("n_checkpoints") or row.get("n_entries") or 0)
        utility_scored = int(summary.get("n_utility_scored") or 0)
        privacy_scored = int(summary.get("n_privacy_scored") or 0)
        deletion_scored = int(summary.get("n_safety_scored") or 0)

        action_accuracy = float(summary.get("action_accuracy") or 0.0)
        utility_accuracy = float(summary.get("utility_accuracy") or 0.0)
        privacy_leakage_rate = float(summary.get("privacy_leakage_rate") or 0.0)
        deletion_leakage_rate = float(summary.get("deletion_leakage_rate") or 0.0)
        mgs = utility_accuracy * (1.0 - privacy_leakage_rate) * (1.0 - deletion_leakage_rate)

        per_domain[str(domain)] = {
            "checkpoints": checkpoints,
            "U": utility_accuracy,
            "A": privacy_leakage_rate,
            "F": deletion_leakage_rate,
            "MGS": mgs,
            "action_accuracy": action_accuracy,
            "summary_path": resolved_summary_path,
        }

        total_checkpoints += checkpoints
        total_action += action_accuracy * checkpoints
        total_utility_scored += utility_scored
        total_utility_correct += utility_accuracy * utility_scored
        total_privacy_scored += privacy_scored
        total_privacy_leaks += privacy_leakage_rate * privacy_scored
        total_deletion_scored += deletion_scored
        total_deletion_leaks += deletion_leakage_rate * deletion_scored

    overall_u = (total_utility_correct / total_utility_scored) if total_utility_scored else 0.0
    overall_a = (total_privacy_leaks / total_privacy_scored) if total_privacy_scored else 0.0
    overall_f = (total_deletion_leaks / total_deletion_scored) if total_deletion_scored else 0.0
    overall = {
        "MGS": overall_u * (1.0 - overall_a) * (1.0 - overall_f),
        "U": overall_u,
        "A": overall_a,
        "F": overall_f,
        "action_accuracy": (total_action / total_checkpoints) if total_checkpoints else 0.0,
    }
    return overall, per_domain


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Gov-Mem official_score.json from a suite run directory.")
    parser.add_argument("--run_dir", required=True, help="Suite run directory containing suite_summary.json")
    parser.add_argument("--config", required=True, help="Config used for the run")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: <run_dir>/official_score.json")
    parser.add_argument("--run_name", default=None, help="Optional run name override")
    parser.add_argument("--suite_manifest", default=None, help="Frozen manifest used for this run.")
    parser.add_argument("--llm_judge", action="store_true", help="Mark metrics as produced by the official LLM judge.")
    parser.add_argument("--tag", default="generated_from_suite_summary")
    parser.add_argument("--eval_type", default="official-compatible cross-domain evaluation")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve()
    output_path = Path(args.output).resolve() if args.output else (run_dir / "official_score.json")

    suite_summary_path = run_dir / "suite_summary.json"
    if not suite_summary_path.exists():
        raise FileNotFoundError(f"Missing suite summary: {suite_summary_path}")

    suite_summary = _read_json(suite_summary_path)
    config = load_yaml_config(config_path)
    overall, per_domain = _aggregate_summary(run_dir, suite_summary)

    runner_cfg = dict(config.get("runner") or {})
    llm_cfg = dict(config.get("llm") or {})
    experiment_mode = str(runner_cfg.get("experiment_mode") or (config.get("experiment") or {}).get("mode") or "unknown")
    checkpoints = sum(int(dict(row).get("checkpoints") or 0) for row in per_domain.values())
    payload = {
        "run_name": args.run_name or run_dir.name,
        "tag": args.tag,
        "eval_type": args.eval_type,
        "scope": {
            "suite_manifest": args.suite_manifest or runner_cfg.get("suite_manifest"),
            "result_dir": str(run_dir),
            "llm_judge": bool(args.llm_judge),
            "runtime_api": llm_cfg.get("provider"),
            "base_model": llm_cfg.get("base_model"),
            "experiment_mode": experiment_mode,
            "checkpoints": checkpoints,
        },
        "overall": overall,
        "per_domain": per_domain,
        "per_backbone": {
            experiment_mode: {
                **overall,
                "checkpoints": checkpoints,
            }
        },
        "sources": {
            "suite_summary": str(suite_summary_path),
            "config": str(config_path),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"official_score={output_path}")


if __name__ == "__main__":
    main()
