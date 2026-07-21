from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "dataset" / "GateMem" / "gatemem" / "data"
SUITE_ROOT = ROOT / "experiments" / "gatemem_suites"
DOMAINS = ["medical", "office", "education", "household"]


def _load_rows(domain: str) -> list[dict]:
    path = DATA_ROOT / domain / "checkpoints.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _pick_utility(rows: list[dict], count: int) -> list[dict]:
    selected: list[dict] = []
    used_ids: set[str] = set()
    used_episodes: set[str] = set()
    utility_rows = [row for row in rows if row.get("query_type") == "utility"]

    for row in utility_rows:
        episode_id = str(row["episode_id"])
        checkpoint_id = str(row["checkpoint_id"])
        if episode_id in used_episodes:
            continue
        selected.append(row)
        used_ids.add(checkpoint_id)
        used_episodes.add(episode_id)
        if len(selected) >= count:
            return selected

    for row in utility_rows:
        checkpoint_id = str(row["checkpoint_id"])
        if checkpoint_id in used_ids:
            continue
        selected.append(row)
        used_ids.add(checkpoint_id)
        if len(selected) >= count:
            return selected
    return selected


def _pick_attack_rows(rows: list[dict], *, query_type: str, count: int) -> list[dict]:
    query_rows = [row for row in rows if row.get("query_type") == query_type]
    by_attack: dict[str, list[dict]] = defaultdict(list)
    for row in query_rows:
        by_attack[str(row.get("attack_type") or "none")].append(row)

    selected: list[dict] = []
    used_ids: set[str] = set()
    used_episodes: set[str] = set()
    attack_types = list(by_attack.keys())

    # First pass: maximize subtype coverage with distinct episodes.
    for attack_type in attack_types:
        for row in by_attack[attack_type]:
            checkpoint_id = str(row["checkpoint_id"])
            episode_id = str(row["episode_id"])
            if checkpoint_id in used_ids or episode_id in used_episodes:
                continue
            selected.append(row)
            used_ids.add(checkpoint_id)
            used_episodes.add(episode_id)
            break
        if len(selected) >= count:
            return selected

    # Second pass: fill missing subtypes even if episodes repeat.
    for attack_type in attack_types:
        if len(selected) >= count:
            return selected
        for row in by_attack[attack_type]:
            checkpoint_id = str(row["checkpoint_id"])
            if checkpoint_id in used_ids:
                continue
            selected.append(row)
            used_ids.add(checkpoint_id)
            break

    # Final pass: round-robin by attack type to reach target count.
    while len(selected) < count:
        progressed = False
        for attack_type in attack_types:
            for row in by_attack[attack_type]:
                checkpoint_id = str(row["checkpoint_id"])
                if checkpoint_id in used_ids:
                    continue
                selected.append(row)
                used_ids.add(checkpoint_id)
                progressed = True
                break
            if len(selected) >= count:
                return selected
        if not progressed:
            break
    return selected


def _to_entry(domain: str, row: dict) -> dict:
    return {
        "domain": domain,
        "checkpoint_id": str(row["checkpoint_id"]),
        "episode_id": str(row["episode_id"]),
        "query_type": str(row.get("query_type") or ""),
        "attack_type": str(row.get("attack_type") or "none"),
    }


def _build_suite(
    *,
    suite_name: str,
    version: int,
    purpose: str,
    per_domain_counts: dict[str, int],
) -> dict:
    entries: list[dict] = []
    for domain in DOMAINS:
        rows = _load_rows(domain)
        utility_rows = _pick_utility(rows, per_domain_counts["utility"])
        privacy_rows = _pick_attack_rows(rows, query_type="privacy", count=per_domain_counts["privacy"])
        safety_rows = _pick_attack_rows(rows, query_type="safety", count=per_domain_counts["safety"])
        domain_entries = [_to_entry(domain, row) for row in utility_rows + privacy_rows + safety_rows]
        entries.extend(domain_entries)

    return {
        "suite_name": suite_name,
        "dataset_name": "gatemem",
        "version": version,
        "purpose": purpose,
        "selection_policy": {
            "domains": DOMAINS,
            "per_domain_counts": per_domain_counts,
            "prefer_distinct_episodes": True,
            "prefer_distinct_attack_types_within_query_type": True,
            "selection_order": ["utility", "privacy", "safety"],
        },
        "entries": entries,
    }


def _write_suite(payload: dict) -> Path:
    SUITE_ROOT.mkdir(parents=True, exist_ok=True)
    path = SUITE_ROOT / f"{payload['suite_name']}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    suites = [
        _build_suite(
            suite_name="gatemem_smoke40",
            version=1,
            purpose="cross-domain smoke regression after framework-level changes",
            per_domain_counts={"utility": 4, "privacy": 3, "safety": 3},
        ),
        _build_suite(
            suite_name="gatemem_regression120",
            version=1,
            purpose="cross-domain formal regression after significant framework upgrades",
            per_domain_counts={"utility": 10, "privacy": 10, "safety": 10},
        ),
    ]
    for payload in suites:
        path = _write_suite(payload)
        print(f"{path} entries={len(payload['entries'])}")


if __name__ == "__main__":
    main()
