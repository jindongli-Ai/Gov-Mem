#!/usr/bin/env python3
"""Build a balanced ID-only held-out 200 checkpoint manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"
DOMAINS = ("medical", "office", "education", "household")
EXCLUDED_MANIFESTS = (
    "experiments/gatemem_suites/gatemem_smoke40.json",
    "experiments/gatemem_suites/stateful_policy_generalization_200_seed20260727.json",
)


def _rank(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _load_entries(domain: str) -> list[dict[str, str]]:
    rows = []
    for line in (DATA_ROOT / domain / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        checkpoint_id = str(raw.get("checkpoint_id") or "")
        episode_id = str(raw.get("episode_id") or "")
        if checkpoint_id and episode_id:
            rows.append({"domain": domain, "checkpoint_id": checkpoint_id, "episode_id": episode_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--per-domain", type=int, default=50)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Additional manifest paths whose checkpoint IDs must be excluded.",
    )
    args = parser.parse_args()

    excluded: set[str] = set()
    excluded_manifests = [*EXCLUDED_MANIFESTS, *args.exclude_manifest]
    for manifest_path in excluded_manifests:
        payload = json.loads((ROOT / manifest_path).read_text(encoding="utf-8"))
        excluded.update(str(row["checkpoint_id"]) for row in payload.get("entries", []))

    entries: list[dict[str, str]] = []
    selected_counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    for domain in DOMAINS:
        candidates = [row for row in _load_entries(domain) if row["checkpoint_id"] not in excluded]
        candidates.sort(key=lambda row: _rank(args.seed, domain, row["checkpoint_id"]))
        selected = candidates[: args.per_domain]
        if len(selected) != args.per_domain:
            raise ValueError(f"{domain}: only {len(selected)} held-out rows available")
        entries.extend(selected)
        selected_counts[domain] = len(selected)
        available_counts[domain] = len(candidates)

    selected_ids = {row["checkpoint_id"] for row in entries}
    assert not selected_ids.intersection(excluded)
    payload = {
        "suite_name": f"rag_naive_v3_long_context_heldout_{len(entries)}_seed{args.seed}",
        "dataset_name": "checkpoint_benchmark",
        "version": 1,
        "purpose": "Balanced blind held-out evaluation; excludes the paired smoke40 and prior 200 manifests.",
        "selection_policy": {
            "unit": "checkpoint_id_only_deterministic_hash",
            "seed": args.seed,
            "per_domain": args.per_domain,
            "domains": list(DOMAINS),
            "available_after_exclusion": available_counts,
            "selected_counts": selected_counts,
            "excluded_manifests": excluded_manifests,
            "excluded_checkpoint_count": len(excluded),
            "uses_answer_fields": False,
            "uses_query_type_fields": False,
            "uses_attack_type_fields": False,
            "uses_evidence_fields": False,
            "uses_scorer_fields": False,
            "selected_checkpoint_sha256": hashlib.sha256(
                "\n".join(sorted(selected_ids)).encode("utf-8")
            ).hexdigest(),
        },
        "entries": entries,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output, "entries=", len(entries), "excluded=", len(excluded), "available=", available_counts)


if __name__ == "__main__":
    main()
