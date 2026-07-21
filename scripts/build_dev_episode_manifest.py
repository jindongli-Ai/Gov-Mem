#!/usr/bin/env python3
"""Create an ID-only, complete-episode development partition.

The selector intentionally touches only domain, checkpoint_id, and episode_id.
It never reads answer, query category, attack category, evidence, or scorer data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"
DOMAINS = ("medical", "office", "education", "household")


def _stable_rank(*, seed: int, domain: str, episode_id: str) -> str:
    payload = f"{seed}|{domain}|{episode_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_id_rows(domain: str) -> list[dict[str, str]]:
    path = DATA_ROOT / domain / "checkpoints.jsonl"
    rows: list[dict[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(raw_line)
        checkpoint_id = str(raw.get("checkpoint_id") or "")
        episode_id = str(raw.get("episode_id") or "")
        if checkpoint_id and episode_id:
            rows.append({"checkpoint_id": checkpoint_id, "episode_id": episode_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ID-only complete-episode development manifest.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes_per_domain", type=int, default=1)
    args = parser.parse_args()
    if args.episodes_per_domain < 1:
        raise ValueError("episodes_per_domain must be positive")

    entries: list[dict[str, str]] = []
    selected_episodes: dict[str, list[str]] = {}
    source_identity: list[str] = []
    for domain in DOMAINS:
        rows = _load_id_rows(domain)
        episodes = sorted({row["episode_id"] for row in rows})
        selected = sorted(
            episodes,
            key=lambda episode_id: _stable_rank(seed=args.seed, domain=domain, episode_id=episode_id),
        )[:args.episodes_per_domain]
        selected_episodes[domain] = selected
        for row in rows:
            source_identity.append(f"{domain}:{row['checkpoint_id']}:{row['episode_id']}")
            if row["episode_id"] in selected:
                entries.append({"domain": domain, **row})

    source_hash = hashlib.sha256("\n".join(sorted(source_identity)).encode("utf-8")).hexdigest()
    payload = {
        "suite_name": f"gov_mem_dev_episode_hashseed{args.seed}",
        "dataset_name": "checkpoint_benchmark",
        "version": 1,
        "purpose": "development-only analysis; selected without labels or scorer data",
        "selection_policy": {
            "unit": "complete_episode",
            "hash": "sha256(seed|domain|episode_id)",
            "seed": args.seed,
            "episodes_per_domain": args.episodes_per_domain,
            "uses_answer_fields": False,
            "uses_query_type_fields": False,
            "uses_attack_type_fields": False,
            "uses_evidence_fields": False,
            "uses_scorer_fields": False,
            "dataset_identity_sha256": source_hash,
            "selected_episodes": selected_episodes,
        },
        "entries": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{output} entries={len(entries)} dataset_identity_sha256={source_hash}")


if __name__ == "__main__":
    main()
