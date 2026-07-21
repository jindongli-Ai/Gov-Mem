#!/usr/bin/env python3
"""Create an attested, post-hoc structural diagnosis table for dev failures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.evolution.dev_guard import load_dev_attestation
from gov_mem.experience.failure_case import DevFailureCaseBuilder
from gov_mem.utils.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose dev failures from runtime traces only.")
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--dev_attestation", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    attestation = load_dev_attestation(args.dev_attestation)
    declared = {str(Path(path).resolve()) for path in attestation["source_runs"]}
    requested = {str(Path(path).resolve()) for path in args.run_dir}
    if requested - declared:
        raise ValueError("Every run_dir must be declared in the development attestation")

    cases = []
    for run_dir in args.run_dir:
        cases.extend(DevFailureCaseBuilder().build_from_run(run_dir=run_dir))
    rows = []
    for case in cases:
        rows.append({
            "case_id": case.case_id,
            "failure_type": case.failure_type,
            "structural_diagnosis": case.structural_diagnosis,
            "provenance": {"dev_attestation": attestation, **case.provenance},
        })
    write_jsonl(args.output_path, rows)
    print(f"diagnosed_failures={len(rows)}")
    print(f"output_path={Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
