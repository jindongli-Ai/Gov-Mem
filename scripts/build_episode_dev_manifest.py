from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOMAINS = ("medical", "office", "education", "household")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an episode-complete deterministic development partition.")
    parser.add_argument("--data_root", default="dataset/GateMem/gatemem/data")
    parser.add_argument("--output_path", default="experiments/gatemem_suites/gatemem_dev_episode_complete_seed42.json")
    parser.add_argument("--episodes_per_domain", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries: list[dict] = []
    selected_episodes: dict[str, list[str]] = {}
    for domain in DOMAINS:
        checkpoints = _read_jsonl(Path(args.data_root) / domain / "checkpoints.jsonl")
        episode_ids = sorted({str(row["episode_id"]) for row in checkpoints})
        ranked = sorted(
            episode_ids,
            key=lambda episode_id: hashlib.sha256(f"gov-mem-dev:{args.seed}:{domain}:{episode_id}".encode()).hexdigest(),
        )
        selected = ranked[: args.episodes_per_domain]
        selected_episodes[domain] = selected
        selected_set = set(selected)
        for row in checkpoints:
            if str(row["episode_id"]) not in selected_set:
                continue
            entries.append({
                "domain": domain,
                "checkpoint_id": str(row["checkpoint_id"]),
                "episode_id": str(row["episode_id"]),
                "query_type": str(row.get("query_type") or ""),
                "attack_type": str(row.get("attack_type") or "none"),
            })

    payload = {
        "suite_name": f"gatemem_dev_episode_complete_seed{args.seed}",
        "dataset_name": "gatemem",
        "version": 1,
        "purpose": "development-only experience induction and skill evolution",
        "selection_policy": {
            "unit": "complete_episode",
            "seed": args.seed,
            "hash": "sha256",
            "episodes_per_domain": args.episodes_per_domain,
            "selected_episodes": selected_episodes,
            "blind_test_exclusion_required": True,
        },
        "entries": entries,
    }
    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"episodes={sum(len(value) for value in selected_episodes.values())} checkpoints={len(entries)}")
    print(f"output_path={output.resolve()}")


if __name__ == "__main__":
    main()
