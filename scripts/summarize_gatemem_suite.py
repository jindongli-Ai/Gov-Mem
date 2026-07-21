from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a GateMem fixed suite manifest.")
    parser.add_argument("--suite_manifest", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.suite_manifest).read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_domain[str(entry["domain"])].append(entry)

    print(f"suite_name: {payload.get('suite_name')}")
    print(f"version: {payload.get('version')}")
    print(f"total_entries: {len(entries)}")
    print()

    for domain in sorted(by_domain):
        rows = by_domain[domain]
        q = Counter(str(r.get("query_type")) for r in rows)
        a = Counter(str(r.get("attack_type") or "none") for r in rows)
        eps = {str(r.get("episode_id")) for r in rows}
        print(f"[{domain}]")
        print(f"  n={len(rows)}")
        print(f"  query_type={dict(q)}")
        print(f"  attack_type={dict(a)}")
        print(f"  distinct_episodes={len(eps)}")
        print()


if __name__ == "__main__":
    main()
