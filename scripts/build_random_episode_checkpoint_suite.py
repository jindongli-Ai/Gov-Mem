#!/usr/bin/env python3
"""Create an ID-only deterministic-random episode/checkpoint suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"
DOMAINS = ("medical", "office", "education", "household")


def _rank(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _load_ids(domain: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in (DATA_ROOT / domain / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        checkpoint_id = str(raw.get("checkpoint_id") or "")
        episode_id = str(raw.get("episode_id") or "")
        if checkpoint_id and episode_id:
            rows.append({"checkpoint_id": checkpoint_id, "episode_id": episode_id})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ID-only random-episode checkpoint probe.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoints_per_domain", type=int, default=8)
    parser.add_argument(
        "--episodes_per_domain",
        type=int,
        default=1,
        help="Number of deterministic-random episodes to select in each domain.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="Domains to include in the ID-only suite.",
    )
    args = parser.parse_args()
    if args.checkpoints_per_domain < 1:
        raise ValueError("checkpoints_per_domain must be positive")
    if args.episodes_per_domain < 1:
        raise ValueError("episodes_per_domain must be positive")

    entries: list[dict[str, str]] = []
    selected_episodes: dict[str, list[str]] = {}
    available_counts: dict[str, dict[str, int]] = {}
    source_identity: list[str] = []
    for domain in args.domains:
        rows = _load_ids(domain)
        source_identity.extend(f"{domain}:{row['checkpoint_id']}:{row['episode_id']}" for row in rows)
        episodes = sorted({row["episode_id"] for row in rows})
        chosen_episode_ids = sorted(
            episodes,
            key=lambda value: _rank(args.seed, domain, "episode", value),
        )[:args.episodes_per_domain]
        selected_episodes[domain] = chosen_episode_ids
        available_counts[domain] = {}
        for episode_id in chosen_episode_ids:
            episode_rows = [row for row in rows if row["episode_id"] == episode_id]
            available_counts[domain][episode_id] = len(episode_rows)
            selected = sorted(
                episode_rows,
                key=lambda row: _rank(args.seed, domain, episode_id, "checkpoint", row["checkpoint_id"]),
            )[:args.checkpoints_per_domain]
            entries.extend({"domain": domain, **row} for row in selected)

    payload = {
        "suite_name": f"gov_mem_random_episode_checkpoint_probe_hashseed{args.seed}",
        "dataset_name": "checkpoint_benchmark",
        "version": 1,
        "purpose": "ID-only cross-domain diagnostic probe; not a full-benchmark result.",
        "selection_policy": {
            "unit": "random_episodes_then_random_checkpoints",
            "hash": "sha256(seed|domain|episode|episode_id) then sha256(seed|domain|episode_id|checkpoint|checkpoint_id)",
            "seed": args.seed,
            "domains": list(args.domains),
            "episodes_per_domain": args.episodes_per_domain,
            "checkpoints_per_domain": args.checkpoints_per_domain,
            "available_checkpoints_in_selected_episode": available_counts,
            "selected_episodes": selected_episodes,
            "uses_answer_fields": False,
            "uses_query_type_fields": False,
            "uses_attack_type_fields": False,
            "uses_evidence_fields": False,
            "uses_scorer_fields": False,
            "dataset_identity_sha256": hashlib.sha256(
                "\n".join(sorted(source_identity)).encode("utf-8")
            ).hexdigest(),
        },
        "entries": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{output} entries={len(entries)} selected_episodes={selected_episodes}")


if __name__ == "__main__":
    main()
