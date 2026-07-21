#!/usr/bin/env python3
"""Join dev-only official outcomes with Gov-Mem runtime traces.

This is a post-hoc diagnostic tool. It must never be imported by runtime code:
official labels explain *where* observed failures occur, while fixes must be
derived from repeated structural boundaries rather than checkpoint wording.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DOMAINS = ("medical", "office", "education", "household")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--score_root",
        default=None,
        help="Optional root holding <domain>/scores.jsonl; defaults to each domain official_eval output.",
    )
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    score_root = Path(args.score_root) if args.score_root else None
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        scores_path = _score_path(run_dir=run_dir, score_root=score_root, domain=domain)
        for score in _read_jsonl(scores_path):
            checkpoint_id = str(score.get("checkpoint_id") or "")
            trace = _read_json(run_dir / domain / "debug_cases" / "checkpoint_benchmark" / f"{checkpoint_id}.json")
            rows.append(_atlas_row(domain=domain, score=score, trace=trace))

    payload = {
        "artifact_scope": "development_only_post_hoc_diagnosis",
        "run_dir": str(run_dir.resolve()),
        "score_root": str(score_root.resolve()) if score_root else "per-domain official_eval",
        "case_count": len(rows),
        "mechanism_counts": dict(sorted(Counter(row["mechanism"] for row in rows).items())),
        "by_domain": _summarize_by_domain(rows),
        "cases": rows,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    print(f"cases={len(rows)}")
    print(f"output_path={output_path.resolve()}")


def _atlas_row(*, domain: str, score: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    certificate = dict(trace.get("graph_authorization_certificate") or {})
    alignment = dict(certificate.get("semantic_alignment") or {})
    coverage = dict(trace.get("slot_coverage") or {})
    dual = dict(trace.get("dual_channel_retrieval") or {})
    utility = dict(dual.get("utility_evidence") or {})
    governance = dict(dual.get("governance_evidence") or {})
    decision = dict(trace.get("action_decision") or {})
    aux = dict(score.get("aux") or {})
    ground_truth = {
        "query_type": score.get("query_type"),
        "expected_action": score.get("expected_action"),
        "utility_required_count": aux.get("include_needed"),
        "utility_hit_count": aux.get("include_hits"),
        "privacy_leak": score.get("privacy_e2e_leak"),
        "deletion_leak": score.get("deletion_e2e_leak"),
    }
    behavior = {
        "predicted_action": score.get("pred_action") or decision.get("action"),
        "action_correct": score.get("action_correct"),
        "utility_correct": score.get("utility_correct"),
        "over_refusal": score.get("over_refusal"),
        "answer": str(trace.get("final_prediction") or ""),
    }
    runtime = {
        "semantic_contract_certifiable": bool(
            ((trace.get("query_plan") or {}).get("semantic_spec") or {}).get("attribute_bindings_valid")
        ),
        "semantic_alignment_reason": alignment.get("reason"),
        "unresolved_attributes": list(alignment.get("unresolved_attributes") or []),
        "utility_selected_atom_count": len(utility.get("selected_atom_ids") or []),
        "governance_selected_policy_count": len(governance.get("selected_policy_atom_ids") or []),
        "slot_coverage_missing": list(coverage.get("missing_slots") or []),
        "certificate_authorized": bool(certificate.get("authorized")),
        "certificate_reason": str(certificate.get("reason") or ""),
    }
    return {
        "checkpoint_id": str(score.get("checkpoint_id") or ""),
        "domain": domain,
        "ground_truth": ground_truth,
        "gov_mem_behavior": behavior,
        "runtime_boundary": runtime,
        "mechanism": _mechanism(ground_truth=ground_truth, behavior=behavior, runtime=runtime),
    }


def _mechanism(*, ground_truth: dict[str, Any], behavior: dict[str, Any], runtime: dict[str, Any]) -> str:
    if ground_truth.get("privacy_leak") is True:
        return "privacy_leakage"
    if ground_truth.get("deletion_leak") is True:
        return "deletion_reconstruction"
    if ground_truth.get("query_type") != "utility":
        return "non_utility_action_mismatch" if not behavior.get("action_correct") else "non_utility_success"
    if behavior.get("utility_correct") is True:
        return "utility_success"
    if ground_truth.get("expected_action") == "answer" and behavior.get("predicted_action") in {"no_memory", "refuse"}:
        return "utility_action_collapse"
    if not runtime.get("semantic_contract_certifiable"):
        return "semantic_contract_not_certifiable"
    if runtime.get("semantic_alignment_reason") != "all_requested_attributes_aligned":
        return "semantic_alignment_miss"
    if not runtime.get("certificate_authorized"):
        return "certificate_path_miss"
    if runtime.get("slot_coverage_missing"):
        return "typed_coverage_miss"
    return "realization_or_renderer_miss"


def _summarize_by_domain(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[row["domain"]][row["mechanism"]] += 1
    return {domain: dict(sorted(counts.items())) for domain, counts in sorted(grouped.items())}


def _score_path(*, run_dir: Path, score_root: Path | None, domain: str) -> Path:
    if score_root is not None:
        direct = score_root / domain / "scores.jsonl"
        nested = score_root / "checkpoint_benchmark" / domain / "scores.jsonl"
        return direct if direct.exists() else nested
    return run_dir / domain / "official_eval" / "checkpoint_benchmark" / domain / "scores.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


if __name__ == "__main__":
    main()
