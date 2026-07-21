from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | None:
    left = _as_float(candidate.get(key))
    right = _as_float(baseline.get(key))
    if left is None or right is None:
        return None
    return left - right


def _format_delta(delta: float | None) -> str:
    if delta is None:
        return "n/a"
    return f"{delta:+.4f}"


def _print_section(title: str) -> None:
    print(f"\n{title}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gov-Mem official score artifacts.")
    parser.add_argument("--baseline", required=True, help="Path to the frozen v0 official_score.json")
    parser.add_argument("--candidate", required=True, help="Path to the new version official_score.json")
    parser.add_argument(
        "--max_mgs_drop_points",
        type=float,
        default=1.0,
        help="Maximum allowed MGS drop in percentage points before rejection. Default: 1.0",
    )
    args = parser.parse_args()

    baseline = _load_json(Path(args.baseline).resolve())
    candidate = _load_json(Path(args.candidate).resolve())
    tolerance = float(args.max_mgs_drop_points) / 100.0

    base_overall = dict(baseline.get("overall") or {})
    cand_overall = dict(candidate.get("overall") or {})
    overall_mgs_delta = _metric_delta(cand_overall, base_overall, "MGS")
    accepted = overall_mgs_delta is not None and overall_mgs_delta >= -tolerance

    _print_section("Overall")
    print(f"baseline_name: {baseline.get('run_name')}")
    print(f"candidate_name: {candidate.get('run_name')}")
    for key in ["MGS", "U", "A", "F", "action_accuracy"]:
        print(
            f"{key}: baseline={base_overall.get(key)} candidate={cand_overall.get(key)} delta={_format_delta(_metric_delta(cand_overall, base_overall, key))}"
        )

    _print_section("Per-Domain")
    domain_names = sorted(set((baseline.get("per_domain") or {}).keys()) | set((candidate.get("per_domain") or {}).keys()))
    for domain in domain_names:
        base_row = dict((baseline.get("per_domain") or {}).get(domain) or {})
        cand_row = dict((candidate.get("per_domain") or {}).get(domain) or {})
        print(
            f"{domain}: baseline_MGS={base_row.get('MGS')} candidate_MGS={cand_row.get('MGS')} delta={_format_delta(_metric_delta(cand_row, base_row, 'MGS'))}"
        )

    _print_section("Per-Backbone")
    backbone_names = sorted(set((baseline.get("per_backbone") or {}).keys()) | set((candidate.get("per_backbone") or {}).keys()))
    for backbone in backbone_names:
        base_row = dict((baseline.get("per_backbone") or {}).get(backbone) or {})
        cand_row = dict((candidate.get("per_backbone") or {}).get(backbone) or {})
        print(
            f"{backbone}: baseline_MGS={base_row.get('MGS')} candidate_MGS={cand_row.get('MGS')} delta={_format_delta(_metric_delta(cand_row, base_row, 'MGS'))}"
        )

    _print_section("Decision")
    print(f"max_allowed_mgs_drop_points: {args.max_mgs_drop_points:.2f}")
    print(f"observed_mgs_delta: {_format_delta(overall_mgs_delta)}")
    print(f"accepted_as_default: {accepted}")
    if not accepted:
        print("result: candidate exceeds the allowed MGS drop threshold and should not become the default version.")
        sys.exit(2)


if __name__ == "__main__":
    main()
