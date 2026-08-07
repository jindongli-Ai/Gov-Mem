#!/usr/bin/env python3
"""Build a balanced, previously unused 40-checkpoint manifest."""

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
    "experiments/gatemem_suites/rag_naive_v3_long_context_heldout_200_seed20260729.json",
)


def _rank(seed: int, domain: str, checkpoint_id: str) -> str:
    return hashlib.sha256(f"{seed}|{domain}|{checkpoint_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--per-domain", type=int, default=10)
    args = parser.parse_args()

    excluded: set[str] = set()
    for manifest_name in EXCLUDED_MANIFESTS:
        payload = json.loads((ROOT / manifest_name).read_text(encoding="utf-8"))
        excluded.update(str(row["checkpoint_id"]) for row in payload.get("entries", []))

    entries: list[dict[str, str]] = []
    available_counts: dict[str, int] = {}
    for domain in DOMAINS:
        rows = []
        for line in (DATA_ROOT / domain / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            checkpoint_id = str(raw.get("checkpoint_id") or "")
            episode_id = str(raw.get("episode_id") or "")
            if checkpoint_id and episode_id and checkpoint_id not in excluded:
                rows.append({"domain": domain, "checkpoint_id": checkpoint_id, "episode_id": episode_id})
        rows.sort(key=lambda row: _rank(args.seed, domain, row["checkpoint_id"]))
        available_counts[domain] = len(rows)
        entries.extend(rows[: args.per_domain])

    expected = len(DOMAINS) * args.per_domain
    if len(entries) != expected:
        raise ValueError(f"expected {expected} entries, got {len(entries)}")
    selected_ids = {row["checkpoint_id"] for row in entries}
    payload = {
        "suite_name": f"rag_naive_v3_reasoning_rerank_heldout_{len(entries)}_seed{args.seed}",
        "dataset_name": "checkpoint_benchmark",
        "version": 1,
        "purpose": "Balanced blind held-out validation for Stage 2B candidate reasoning rerank.",
        "selection_policy": {
            "unit": "checkpoint_id_only_deterministic_hash",
            "seed": args.seed,
            "per_domain": args.per_domain,
            "domains": list(DOMAINS),
            "available_after_exclusion": available_counts,
            "excluded_manifests": list(EXCLUDED_MANIFESTS),
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
    print(output, "entries=", len(entries), "excluded=", len(excluded))


if __name__ == "__main__":
    main()
