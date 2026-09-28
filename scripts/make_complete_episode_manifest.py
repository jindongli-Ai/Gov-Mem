#!/usr/bin/env python3
"""Create a checkpoint manifest containing one complete episode in source order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--episode_id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_ids = []
    with args.checkpoints.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("episode_id") or "") == args.episode_id:
                checkpoint_ids.append(str(row["checkpoint_id"]))
    if not checkpoint_ids:
        raise SystemExit(f"Episode not found: {args.episode_id}")

    payload = {
        "schema_version": "govmem-complete-episode-manifest-1",
        "selection_unit": "complete_episode",
        "episode_id": args.episode_id,
        "checkpoint_count": len(checkpoint_ids),
        "checkpoint_ids": checkpoint_ids,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
