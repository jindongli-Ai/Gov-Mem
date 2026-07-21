from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"
DOMAINS = ("medical", "office", "education", "household")


def _load_rows(domain: str) -> list[dict]:
    path = DATA_ROOT / domain / "checkpoints.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _sample_episode_checkpoints(
    rows: list[dict],
    *,
    episode_count: int,
    checkpoints_per_episode: int,
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """Sample independent episodes, then checkpoints inside each episode."""
    by_episode: dict[str, list[dict]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if episode_id:
            by_episode.setdefault(episode_id, []).append(row)
    episode_ids = list(by_episode)
    rng.shuffle(episode_ids)
    selected_episode_ids = episode_ids[:episode_count]
    selected: list[dict] = []
    for episode_id in selected_episode_ids:
        checkpoints = list(by_episode[episode_id])
        rng.shuffle(checkpoints)
        selected.extend(checkpoints[:checkpoints_per_episode])
    return selected, selected_episode_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed random cross-domain evaluation manifest without reading answers or scorer fields."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--episodes_per_domain", type=int, default=2)
    parser.add_argument("--checkpoints_per_episode", type=int, default=3)
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAINS,
        default=list(DOMAINS),
        help="One or more domains for a fixed ID-only probe.",
    )
    args = parser.parse_args()
    if args.episodes_per_domain < 1 or args.checkpoints_per_episode < 1:
        raise ValueError("episodes_per_domain and checkpoints_per_episode must be positive")

    entries: list[dict] = []
    selected_episodes: dict[str, list[str]] = {}
    available_episode_counts: dict[str, int] = {}
    source_identity: list[str] = []
    domains = tuple(args.domains)
    for domain_index, domain in enumerate(domains):
        rows = _load_rows(domain)
        source_identity.extend(
            f"{domain}:{row.get('checkpoint_id')}:{row.get('episode_id')}"
            for row in rows
        )
        rng = random.Random(args.seed + 1009 * domain_index)
        sampled, episode_ids = _sample_episode_checkpoints(
            rows,
            episode_count=args.episodes_per_domain,
            checkpoints_per_episode=args.checkpoints_per_episode,
            rng=rng,
        )
        selected_episodes[domain] = episode_ids
        available_episode_counts[domain] = len({str(row.get("episode_id") or "") for row in rows})
        entries.extend(
            {
                "domain": domain,
                "checkpoint_id": str(row["checkpoint_id"]),
                "episode_id": str(row["episode_id"]),
            }
            for row in sampled
        )
    payload = {
        "suite_name": f"random_{'_'.join(domains)}_{args.seed}",
        "dataset_name": "checkpoint_benchmark",
        "version": 1,
        "purpose": "Fixed random blind regression over the declared domains.",
        "selection_policy": {
            "seed": args.seed,
            "episodes_per_domain": args.episodes_per_domain,
            "checkpoints_per_episode": args.checkpoints_per_episode,
            "domains": list(domains),
            "selected_episodes": selected_episodes,
            "available_episode_counts": available_episode_counts,
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
