"""Descriptive paired diagnostics; never infer module causality from two runs.

Reads saved official judgments and frozen artifacts only. No model calls and no
runtime/gold feedback. Component decomposition is arithmetic, not an ablation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed_rows(path):
    records = rows(path)
    indexed = {r["checkpoint_id"]: r for r in records}
    if len(indexed) != len(records):
        raise ValueError(f"Duplicate checkpoint in {path}")
    return indexed


def mgs(components):
    u, a, f = components
    return u * (1 - a) * (1 - f)


def component_decomposition(before, after):
    """Average all six replacement orders to allocate metric interactions."""
    contributions = [0.0, 0.0, 0.0]
    for order in permutations(range(3)):
        current = list(before)
        for index in order:
            previous = mgs(current)
            current[index] = after[index]
            contributions[index] += (mgs(current) - previous) / 6
    assert abs(sum(contributions) - (mgs(after) - mgs(before))) < 1e-12
    return dict(zip(("utility", "privacy", "deletion"), contributions))


def is_error(row):
    return row.get("execution_status") == "error" or row.get("action") == "error"


def component_label(row):
    judge, kind = row["judge"], row["query_type"]
    field = {"utility": "utility_ok", "privacy": "privacy_leak", "safety": "deletion_leak"}[kind]
    label = judge.get(field)
    if label is None:
        return "unknown"
    if kind == "utility":
        return "good" if label is True and judge.get("action_ok") is True else "bad"
    return "bad" if label is True else "good"


def paired_domain(old_dir, new_dir):
    old_preds = indexed_rows(old_dir / "predictions.jsonl")
    new_preds = indexed_rows(new_dir / "predictions.jsonl")
    old_j = indexed_rows(old_dir / "official_eval/judge_scores.jsonl")
    new_j = indexed_rows(new_dir / "official_eval/judge_scores.jsonl")
    assert old_preds.keys() == new_preds.keys() == old_j.keys() == new_j.keys()
    assert all(r["judge"]["parse_ok"] for r in [*old_j.values(), *new_j.values()])
    cohorts = {}
    for checkpoint in sorted(old_preds):
        a, b = old_preds[checkpoint], new_preds[checkpoint]
        ea, eb = is_error(a), is_error(b)
        cohort = ("both_error" if eb else "recovered") if ea else ("new_error" if eb else "both_success")
        data = cohorts.setdefault(cohort, {"checkpoints": 0, "by_type": {}})
        data["checkpoints"] += 1
        kind = old_j[checkpoint]["query_type"]
        assert kind == new_j[checkpoint]["query_type"]
        by_type = data["by_type"].setdefault(kind, {"n": 0, "old_good": 0, "new_good": 0, "transitions": {}})
        old_label, new_label = component_label(old_j[checkpoint]), component_label(new_j[checkpoint])
        by_type["n"] += 1
        by_type["old_good"] += old_label == "good"
        by_type["new_good"] += new_label == "good"
        key = f"{old_label}->{new_label}"
        by_type["transitions"][key] = by_type["transitions"].get(key, 0) + 1
    return cohorts


def source_selection_audit(root):
    counts = Counter()
    for path in (root / "govmem").glob("*/*/predictions/checkpoint_benchmark/*.json"):
        raw = read(path).get("raw_response", {})
        ledger = raw.get("v8_claim_ledger")
        if not ledger:
            continue
        counts["normal_checkpoints"] += 1
        selected = [c for c in ledger["claims"] if str(c.get("source_memory_id", "")).startswith("v8_visible_source::")]
        counts["claims_from_graph_or_ingestion_only"] += len(selected)
        counts["checkpoints_selecting_graph_or_ingestion_only"] += bool(selected)
        safe_ids = set(raw.get("v8_safe_evidence", {}).get("released_claim_ids", []))
        released = [c for c in selected if c["claim_id"] in safe_ids]
        counts["source_only_claims_reaching_stage3"] += len(released)
        counts["checkpoints_with_source_only_claims_reaching_stage3"] += bool(released)
        counts["blocked_claims"] += sum(c["delivery"] == "block" for c in ledger["claims"])
        counts["symbolic_marked_claims"] += sum(bool(c.get("symbolic_veto_reasons")) for c in ledger["claims"])
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True, help="Old scored root")
    parser.add_argument("--new", type=Path, required=True, help="New scored root")
    parser.add_argument("--old_runtime", type=Path, required=True)
    parser.add_argument("--new_runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = read(args.old / "paired_metrics.json")["systems"]["govmem"]
    new = read(args.new / "paired_metrics.json")["systems"]["govmem"]
    assert read(args.old_runtime / "manifest.json") == read(args.new_runtime / "manifest.json")
    domains, totals, shared = {}, Counter(), {}
    for domain, after in new["per_domain"].items():
        before = old["per_domain"][domain]
        component = component_decomposition([before[k] for k in ("U", "A", "F")], [after[k] for k in ("U", "A", "F")])
        cohorts = paired_domain(args.old / "govmem" / domain, args.new / "govmem" / domain)
        domains[domain] = {"old_UAF": {k: before[k] for k in ("U", "A", "F")},
            "new_UAF": {k: after[k] for k in ("U", "A", "F")},
            "old_MGS": before["MGS"], "new_MGS": after["MGS"],
            "metric_component_decomposition": component, "execution_cohorts": cohorts}
        for name, cohort in cohorts.items():
            totals[name] += cohort["checkpoints"]
        for kind, d in cohorts.get("both_success", {}).get("by_type", {}).items():
            s = shared.setdefault(kind, Counter())
            for k in ("n", "old_good", "new_good"):
                s[k] += d[k]
            s.update(d["transitions"])
    old_src, new_src = args.old_runtime / "runtime_snapshot", args.new_runtime / "runtime_snapshot"
    files = set(p.relative_to(old_src) for p in old_src.rglob("*.py")) | set(p.relative_to(new_src) for p in new_src.rglob("*.py"))
    changed = [str(p) for p in sorted(files) if not (old_src/p).exists() or not (new_src/p).exists()
               or (old_src/p).read_bytes() != (new_src/p).read_bytes()]
    result = {"interpretation": "Descriptive metric/cohort diagnostics only; not module-level causal attribution. Shared-success subsets are diagnostic, not replacement performance scores.",
        "old_runtime_sha256": read(args.old_runtime / "execution_identity.json")["runtime_sha256"],
        "new_runtime_sha256": read(args.new_runtime / "execution_identity.json")["runtime_sha256"],
        "changed_runtime_files": changed,
        "mean_MGS_gain": new["four_domain_mean_MGS"] - old["four_domain_mean_MGS"],
        "mean_metric_component_decomposition": {k: sum(d["metric_component_decomposition"][k] for d in domains.values()) / len(domains)
                                                for k in ("utility", "privacy", "deletion")},
        "execution_cohort_totals": dict(totals), "both_success_component_counts": {k: dict(v) for k, v in shared.items()},
        "new_source_selection_audit": source_selection_audit(args.new_runtime), "domains": domains}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "domains"}, indent=2))


if __name__ == "__main__":
    main()
