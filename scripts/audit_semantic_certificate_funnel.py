#!/usr/bin/env python3
"""Summarize evidence-to-certificate loss from Gov-Mem runtime traces.

The audit is deliberately structural: it never reads question text or applies
domain vocabulary.  It may be run on a completed dev run to identify whether
utility loss occurs before retrieval, semantic alignment, authorization, or
final realization.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Gov-Mem suite output directory")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cases = []
    for path in sorted(run_dir.glob("*/debug_cases/*/*.json")):
        payload = _read_json(path)
        if payload:
            cases.append(_summarize_case(path=path, payload=payload))

    report = {
        "run_dir": str(run_dir.resolve()),
        "case_count": len(cases),
        "overall": _aggregate(cases),
        "by_domain": _grouped(cases, "domain"),
        "by_query_type": _grouped(cases, "query_type"),
        "cases": cases,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    print(f"cases={len(cases)}")
    print(f"output_path={output_path.resolve()}")


def _summarize_case(*, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    plan = dict(payload.get("query_plan") or {})
    semantic_spec = dict(plan.get("semantic_spec") or {})
    requested = _requested_attributes(semantic_spec)
    coverage = dict(payload.get("slot_coverage") or {})
    dual = dict(payload.get("dual_channel_retrieval") or {})
    utility = dict(dual.get("utility_evidence") or {})
    governance = dict(dual.get("governance_evidence") or {})
    certificate = dict(payload.get("graph_authorization_certificate") or {})
    realization = dict(payload.get("final_realization") or {})
    score = dict(payload.get("official_score") or {})
    decision = dict(payload.get("action_decision") or {})
    # <run_dir>/<domain>/debug_cases/<dataset>/<instance>.json
    domain = path.parents[2].name

    return {
        "case_id": str(payload.get("instance_id") or path.stem),
        "domain": domain,
        "query_type": str(score.get("query_type") or plan.get("query_type") or "unknown"),
        "requested_attribute_count": len(requested),
        "semantic_contract_certifiable": bool(semantic_spec.get("attribute_bindings_valid")),
        "selected_evidence_count": len(payload.get("selected_evidence") or []),
        "coverage_missing_count": len(coverage.get("missing_slots") or []),
        "utility_selected_atom_count": len(utility.get("selected_atom_ids") or []),
        "governance_selected_policy_count": len(governance.get("selected_policy_atom_ids") or []),
        "certificate_authorized": bool(certificate.get("authorized")),
        "certificate_reason": _reason_family(certificate.get("reason")),
        "realization_available": bool(realization),
        "predicted_action": str(score.get("pred_action") or decision.get("action") or "unknown"),
        "official_utility_correct": score.get("utility_correct"),
        "official_action_correct": score.get("action_correct"),
    }


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(cases)
    counts = Counter()
    for case in cases:
        if case["requested_attribute_count"]:
            counts["has_semantic_request"] += 1
        if case["selected_evidence_count"]:
            counts["has_selected_evidence"] += 1
        if not case["coverage_missing_count"]:
            counts["slot_coverage_complete"] += 1
        if case["utility_selected_atom_count"]:
            counts["has_utility_atoms"] += 1
        if case["governance_selected_policy_count"]:
            counts["has_governance_policy"] += 1
        if case["certificate_authorized"]:
            counts["certificate_authorized"] += 1
        if case["realization_available"]:
            counts["realization_available"] += 1
        if case["official_utility_correct"] is True:
            counts["utility_correct"] += 1
        if case["official_action_correct"] is True:
            counts["action_correct"] += 1
    return {
        "cases": total,
        "stage_counts": dict(sorted(counts.items())),
        "stage_rates": {
            key: value / total if total else 0.0
            for key, value in sorted(counts.items())
        },
        "certificate_reason_counts": dict(sorted(Counter(
            case["certificate_reason"] for case in cases
        ).items())),
        "action_counts": dict(sorted(Counter(case["predicted_action"] for case in cases).items())),
    }


def _grouped(cases: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case.get(key) or "unknown")].append(case)
    return {name: _aggregate(rows) for name, rows in sorted(groups.items())}


def _requested_attributes(semantic_spec: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(
        str(value).strip()
        for value in (
            semantic_spec.get("requested_attributes")
            or semantic_spec.get("requested_slots")
            or []
        )
        if str(value).strip()
    ))


def _reason_family(reason: object) -> str:
    text = str(reason or "missing_certificate")
    return text.split(":", 1)[0]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
