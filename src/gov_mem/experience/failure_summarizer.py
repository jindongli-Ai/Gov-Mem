from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from gov_mem.experience.failure_case import DevFailureCaseBuilder, FailureCase
from gov_mem.experience.pattern_inducer import FailurePattern, PatternInducer
from gov_mem.evolution.dev_guard import load_dev_attestation, require_matching_attestation
from gov_mem.utils.io import read_jsonl, write_jsonl


class FailureSummarizer:
    def summarize(
        self,
        *,
        failure_cases: list[FailureCase],
        min_support: int = 2,
    ) -> list[FailurePattern]:
        return PatternInducer().induce(
            failure_cases=failure_cases,
            min_support=min_support,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dev-only Gov-Mem failures into generalized failure patterns.")
    parser.add_argument("--run_dir", action="append", default=[], help="Development run directory. Can be passed multiple times.")
    parser.add_argument("--dev_attestation", required=True, help="Development-only provenance attestation JSON.")
    parser.add_argument("--failure_case_path", action="append", default=[], help="Existing failure_cases.jsonl path. Can be passed multiple times.")
    parser.add_argument("--output_path", default="outputs/experience/failure_patterns.jsonl", help="Path to write failure patterns.")
    parser.add_argument("--backbone", default="rag_policy_amem", help="Backbone name for newly built failure cases.")
    parser.add_argument("--min_support", type=int, default=2, help="Minimum support count for a generalized pattern.")
    args = parser.parse_args()

    attestation = load_dev_attestation(args.dev_attestation)
    declared_runs = {str(Path(run).resolve()) for run in attestation["source_runs"]}
    if {str(Path(run).resolve()) for run in args.run_dir} - declared_runs:
        raise ValueError("Each run_dir must be declared in the development attestation")

    failure_cases: list[FailureCase] = []
    if args.run_dir:
        builder = DevFailureCaseBuilder()
        for run_dir in args.run_dir:
            failure_cases.extend(builder.build_from_run(run_dir=run_dir, backbone=args.backbone))
    for path_str in args.failure_case_path:
        rows = read_jsonl(path_str)
        require_matching_attestation(artifacts=rows, attestation=attestation)
        for row in rows:
            failure_cases.append(FailureCase(**row))

    deduped: list[FailureCase] = []
    seen: set[str] = set()
    for case in failure_cases:
        if case.case_id in seen:
            continue
        seen.add(case.case_id)
        deduped.append(case)

    patterns = FailureSummarizer().summarize(
        failure_cases=deduped,
        min_support=args.min_support,
    )
    for pattern in patterns:
        pattern.provenance["dev_attestation"] = attestation
    write_jsonl(Path(args.output_path), [asdict(item) for item in patterns])
    print(f"failure_cases={len(deduped)}")
    print(f"patterns={len(patterns)}")
    print(f"output_path={Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
