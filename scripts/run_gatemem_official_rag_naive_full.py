from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = ROOT / "third_party" / "GateMem-official"
RUN_EVAL = OFFICIAL_ROOT / "bench" / "scripts" / "run_eval.py"
DOMAINS = ("medical", "office", "education", "household")
EXPECTED_CHECKPOINTS = {
    "medical": 579,
    "office": 547,
    "education": 540,
    "household": 552,
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gatemem_suite import _discover_api_keys


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_complete(run_root: Path, domain: str) -> bool:
    summary_path = run_root / domain / "summary.json"
    if not summary_path.exists():
        return False
    summary = _read_json(summary_path)
    return int(summary.get("n_checkpoints") or 0) == EXPECTED_CHECKPOINTS[domain]


def _split_keys(keys: list[str]) -> dict[str, list[str]]:
    return {
        domain: keys[index::len(DOMAINS)]
        for index, domain in enumerate(DOMAINS)
    }


def _build_score(run_root: Path, config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text())
    per_domain: dict[str, dict[str, float | int]] = {}
    for domain in DOMAINS:
        summary = _read_json(run_root / domain / "summary.json")
        checkpoints = int(summary.get("n_checkpoints") or 0)
        if checkpoints != EXPECTED_CHECKPOINTS[domain]:
            raise RuntimeError(
                f"Incomplete {domain} summary: {checkpoints}/{EXPECTED_CHECKPOINTS[domain]}"
            )
        u = float(summary.get("utility_accuracy") or 0.0)
        a = float(summary.get("privacy_leakage_rate") or 0.0)
        f = float(summary.get("deletion_leakage_rate") or 0.0)
        per_domain[domain] = {
            "checkpoints": checkpoints,
            "U": u,
            "A": a,
            "F": f,
            "MGS": u * (1.0 - a) * (1.0 - f),
        }

    average = {
        metric: sum(float(row[metric]) for row in per_domain.values()) / len(per_domain)
        for metric in ("U", "A", "F", "MGS")
    }
    payload = {
        "run_name": run_root.name,
        "protocol": "GateMem official RAG-Naive protocol, unified OpenLux reproduction",
        "scope": {
            "agent": "rag_naive",
            "base_model": config['llm_model'],
            "embedding_model": "text-embedding-3-small",
            "judge_model": "gpt-4o",
            "gate_by_action": False,
            "checkpoints": sum(EXPECTED_CHECKPOINTS.values()),
            "temperature": config['temperature'],
            "config": str(config_path),
        },
        "per_domain": per_domain,
        "four_domain_average": average,
    }
    output_path = run_root / "official_score.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GateMem's official RAG-Naive baseline on all four domains."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "gatemem_official_rag_naive_openlux_gemini25flashlite.yaml"),
    )
    parser.add_argument(
        "--run_dir",
        default=str(ROOT / "experiments" / "runs" / "gatemem_official_rag_naive_gemini25flashlite_full_20260917"),
    )
    parser.add_argument("--max_api_keys", type=int, default=30)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    run_root = Path(args.run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    keys = _discover_api_keys(provider="openlux")[: max(1, args.max_api_keys)]
    if len(keys) < len(DOMAINS):
        raise RuntimeError(f"At least four OpenLux keys are required; found {len(keys)}")
    key_shards = _split_keys(keys)

    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for domain in DOMAINS:
        if _domain_complete(run_root, domain):
            print(f"{domain}: already complete", flush=True)
            continue

        domain_dir = run_root / domain
        prediction_path = domain_dir / "predictions.jsonl"
        shard = key_shards[domain]
        env = os.environ.copy()
        env["OPENLUX_API_KEY"] = shard[0]
        env["OPENLUX_API_KEYS"] = ",".join(shard)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [
            sys.executable,
            str(RUN_EVAL),
            "--config",
            str(config_path),
            "--data_dir",
            str(OFFICIAL_ROOT / "bench" / "data" / domain),
            "--out_dir",
            str(run_root),
            "--run_name",
            domain,
            "--episode_concurrency",
            str(len(shard)),
            "--judge_concurrency",
            str(len(shard)),
            "--log_file",
            "run.log",
            "--no_progress",
        ]
        if prediction_path.exists():
            cmd.append("--resume")
        stdout_path = run_root / f"{domain}.stdout.log"
        stdout_handle = stdout_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((domain, process, stdout_handle))
        print(
            f"{domain}: pid={process.pid} workers={len(shard)} resume={prediction_path.exists()}",
            flush=True,
        )

    failures: list[str] = []
    for domain, process, stdout_handle in processes:
        return_code = process.wait()
        stdout_handle.close()
        print(f"{domain}: exit={return_code}", flush=True)
        if return_code != 0 or not _domain_complete(run_root, domain):
            failures.append(domain)

    if failures:
        raise RuntimeError("Incomplete domains: " + ", ".join(failures))

    score_path = _build_score(run_root, config_path)
    print(f"official_score={score_path}", flush=True)


if __name__ == "__main__":
    main()
