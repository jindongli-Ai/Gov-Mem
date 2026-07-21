from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def _wait_for_pid(pid: int, log_path: Path) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(30)
    summary = ROOT / "outputs/dev_evolution/round0_core_no_adaptation/suite_summary.json"
    if not summary.exists():
        raise RuntimeError(f"round 0 ended without suite summary; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue structural Gov-Mem evolution after dev round 0.")
    parser.add_argument("--round0_pid", type=int, required=True)
    args = parser.parse_args()

    run_dir = "outputs/dev_evolution/round0_core_no_adaptation"
    attestation = "configs/adaptation/dev_episode_complete_seed42.attestation.json"
    artifact_root = "outputs/dev_evolution/artifacts"
    _wait_for_pid(args.round0_pid, ROOT / "outputs/dev_evolution/round0_core_no_adaptation.log")

    _run(
        "-m", "gov_mem.experience.experience_memory",
        "--run_dir", run_dir,
        "--dev_attestation", attestation,
        "--output_dir", f"{artifact_root}/experience",
    )
    _run(
        "-m", "gov_mem.experience.failure_summarizer",
        "--run_dir", run_dir,
        "--dev_attestation", attestation,
        "--output_path", f"{artifact_root}/experience/failure_patterns.jsonl",
    )
    _run(
        "-m", "gov_mem.skills.skill_library",
        "--pattern_path", f"{artifact_root}/experience/failure_patterns.jsonl",
        "--output_path", f"{artifact_root}/skills/governance_skill_library.jsonl",
    )
    _run(
        "scripts/build_dev_updates.py",
        "--pattern_path", f"{artifact_root}/experience/failure_patterns.jsonl",
        "--skill_path", f"{artifact_root}/skills/governance_skill_library.jsonl",
        "--dev_attestation", attestation,
        "--output_dir", f"{artifact_root}/evolution",
    )
    _run("scripts/run_structural_evolution_smoke.py")
    _run("scripts/run_semantic_invariance_smoke.py")
    _run(
        "scripts/run_gatemem_suite.py",
        "--suite_manifest", "experiments/gatemem_suites/gatemem_dev_episode_complete_seed42.json",
        "--output_dir", "outputs/dev_evolution/round1_structural_adaptation",
        "--config", "configs/gov_mem_structural_evolution_round1.yaml",
        "--experiment_mode", "rag_policy_amem",
        "--parallel_domains", "4",
    )


if __name__ == "__main__":
    main()
