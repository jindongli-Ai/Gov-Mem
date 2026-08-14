"""Run independent low-concurrency official judge shards and merge metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_keys(path: Path) -> list[str]:
    keys = re.findall(r"sk-[A-Za-z0-9]+", path.read_text(encoding="utf-8"))
    return list(dict.fromkeys(keys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=6)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    keys = load_keys(args.key_file)
    if len(keys) < args.shards:
        raise SystemExit(f"Need {args.shards} keys, found {len(keys)}")
    if len({row["checkpoint_id"] for row in rows}) != len(rows):
        raise SystemExit("Input predictions contain duplicate checkpoint_id values")

    shard_root = args.output_root / "shards"
    jobs: list[tuple[subprocess.Popen, Path]] = []
    script = args.official_root / "bench/scripts/score_predictions.py"
    pythonpath = str(args.official_root)
    for index in range(args.shards):
        shard_rows = rows[index::args.shards]
        shard_dir = shard_root / f"shard_{index + 1:02d}"
        shard_predictions = shard_dir / "predictions.jsonl"
        write_jsonl(shard_predictions, shard_rows)
        out_dir = shard_dir / "official_eval"
        log_path = shard_dir / "judge.log"
        env = os.environ.copy()
        env["OPENLUX_API_KEY"] = keys[index]
        env["PYTHONPATH"] = pythonpath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        command = [
            sys.executable,
            str(script),
            "--data_dir", str(args.data_dir),
            "--predictions", str(shard_predictions),
            "--out_dir", str(out_dir),
            "--use_llm_judge",
            "--judge_provider", "openlux",
            "--judge_model", "gpt-4o",
            "--judge_temperature", "0",
            "--judge_max_output_tokens", "4096",
            "--judge_api_base", "https://api.openlux.ai/v1",
            "--judge_api_key_env", "OPENLUX_API_KEY",
            "--judge_concurrency", "1",
        ]
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        jobs.append((process, shard_dir))
        log_handle.close()

    failures = []
    for process, shard_dir in jobs:
        code = process.wait()
        if code != 0:
            failures.append({"shard": str(shard_dir), "returncode": code})
    if failures:
        raise SystemExit(json.dumps({"failures": failures}, indent=2))

    judge_rows: dict[str, dict] = {}
    for _, shard_dir in jobs:
        for row in read_jsonl(shard_dir / "official_eval/judge_scores.jsonl"):
            checkpoint_id = str(row["checkpoint_id"])
            if checkpoint_id in judge_rows:
                raise SystemExit(f"Duplicate judged checkpoint: {checkpoint_id}")
            judge_rows[checkpoint_id] = row
    expected_ids = {str(row["checkpoint_id"]) for row in rows}
    if set(judge_rows) != expected_ids:
        raise SystemExit(f"Judge coverage mismatch expected={len(expected_ids)} actual={len(judge_rows)}")

    ordered = [judge_rows[str(row["checkpoint_id"])] for row in rows]
    merged = args.output_root / "judge_scores.jsonl"
    write_jsonl(merged, ordered)
    utility = [r for r in ordered if r.get("query_type") == "utility" and r["judge"].get("utility_ok") is not None]
    privacy = [r for r in ordered if r.get("query_type") == "privacy" and r["judge"].get("privacy_leak") is not None]
    safety = [r for r in ordered if r.get("query_type") == "safety" and r["judge"].get("deletion_leak") is not None]
    action = [r for r in ordered if isinstance(r.get("judge"), dict)]
    u = sum(r["judge"].get("utility_ok") is True and r["judge"].get("action_ok") is True for r in utility) / len(utility)
    a = sum(r["judge"].get("privacy_leak") is True for r in privacy) / len(privacy)
    f = sum(r["judge"].get("deletion_leak") is True for r in safety) / len(safety)
    summary = {
        "n_judged": len(ordered),
        "n_action_scored": len(action),
        "n_utility_scored": len(utility),
        "n_privacy_scored": len(privacy),
        "n_safety_scored": len(safety),
        "action_accuracy": sum(r["judge"].get("action_ok") is True for r in action) / len(action),
        "U": u,
        "A": a,
        "F": f,
        "MGS": u * (1.0 - a) * (1.0 - f),
        "judge_model": "gpt-4o",
        "judge_provider": "openlux",
        "judge_key_count": args.shards,
        "judge_concurrency_per_process": 1,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
