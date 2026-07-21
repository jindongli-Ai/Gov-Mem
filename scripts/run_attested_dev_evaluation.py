#!/usr/bin/env python3
"""Run one declared development evaluation with provenance checks.

This runner exists to keep the only scorer-informed workflow separate from
blind evaluation. It never modifies prompts, skills, or runtime configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.evolution.dev_guard import load_dev_attestation
from gov_mem.utils.config import load_yaml_config


def _validate_manifest(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = dict(payload.get("selection_policy") or {})
    required_false = (
        "uses_answer_fields",
        "uses_query_type_fields",
        "uses_attack_type_fields",
        "uses_evidence_fields",
        "uses_scorer_fields",
    )
    if not payload.get("entries") or any(policy.get(key) is not False for key in required_false):
        raise ValueError("Development manifest must be an ID-only manifest with all label-use flags false")
    allowed_entry_keys = {"domain", "checkpoint_id", "episode_id"}
    if any(set(dict(entry)) != allowed_entry_keys for entry in payload["entries"]):
        raise ValueError("Development manifest entries may contain only domain, checkpoint_id, and episode_id")


def _load_yunwu_environment() -> dict[str, str]:
    """Reuse the repository's documented key source without persisting keys."""
    environment = os.environ.copy()
    if environment.get("YUNWU_API_KEY") or environment.get("YUNWU_API_KEYS"):
        return environment
    readme = ROOT / "README_API_Yunwu.md"
    if not readme.exists():
        return environment
    keys = re.findall(r"sk-[A-Za-z0-9_-]+", readme.read_text(encoding="utf-8"))
    if keys:
        environment["YUNWU_API_KEYS"] = ",".join(keys[:4])
        environment["YUNWU_API_KEY"] = keys[0]
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an attested development-only Gov-Mem evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev_attestation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--experiment_mode", default="rag_policy_amem")
    parser.add_argument("--parallel_domains", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Strictly resume a compatible interrupted attested run.")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    attestation = load_dev_attestation(args.dev_attestation)
    if str(manifest.relative_to(ROOT)) != str(attestation.get("manifest") or ""):
        raise ValueError("Attestation manifest does not match --manifest")
    if str(output_dir) not in {str(Path(path).resolve()) for path in attestation["source_runs"]}:
        raise ValueError("--output_dir must be predeclared in dev_attestation.source_runs")
    _validate_manifest(manifest)
    child_env = _load_yunwu_environment()
    if not (child_env.get("YUNWU_API_KEY") or child_env.get("YUNWU_API_KEYS")):
        raise RuntimeError("No LLM credential available; refusing an invalid heuristic-fallback development score")

    suite_cmd = [
        sys.executable, str(ROOT / "scripts" / "run_gatemem_suite.py"),
        "--suite_manifest", str(manifest),
        "--output_dir", str(output_dir),
        "--config", str(config_path),
        "--experiment_mode", args.experiment_mode,
        "--parallel_domains", str(max(1, args.parallel_domains)),
    ]
    if args.resume:
        suite_cmd.append("--resume")
    subprocess.run(suite_cmd, cwd=str(ROOT), check=True, env=child_env)
    run_config = load_yaml_config(config_path)
    if bool(run_config.get("require_attested_evidence_span", False)) and bool(
        run_config.get("enable_graph_typed_slot_realization", False)
    ):
        audit_cmd = [
            sys.executable, str(ROOT / "scripts" / "audit_attested_provenance_run.py"),
            "--run_dir", str(output_dir),
            "--output_path", str(output_dir / "attested_provenance_audit.json"),
        ]
        subprocess.run(audit_cmd, cwd=str(ROOT), check=True, env=child_env)
    score_cmd = [
        sys.executable, str(ROOT / "scripts" / "build_official_score.py"),
        "--run_dir", str(output_dir),
        "--config", str(Path(args.config).resolve()),
        "--run_name", output_dir.name,
        "--tag", "attested_dev_only",
        "--eval_type", "development-only structural diagnosis evaluation",
    ]
    subprocess.run(score_cmd, cwd=str(ROOT), check=True, env=child_env)
    print(f"attested_dev_run={output_dir}")
    print(f"official_score={output_dir / 'official_score.json'}")
    print("Next: run scripts/diagnose_dev_failures.py with the same attestation; do not enable generated updates automatically.")


if __name__ == "__main__":
    main()
