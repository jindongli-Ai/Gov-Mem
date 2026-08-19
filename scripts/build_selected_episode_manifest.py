#!/usr/bin/env python3
"""Build an ID-only manifest for explicitly selected complete episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=str(ROOT / "dataset" / "GateMem" / "gatemem" / "data"))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--episode_id", action="append", required=True)
    parser.add_argument(
        "--suite_name",
        default=None,
        help="Explicit experiment suite name for auditability.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    requested = list(dict.fromkeys(str(value) for value in args.episode_id))
    rows = [
        json.loads(line)
        for line in (data_root / args.domain / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    available = {str(row.get("episode_id") or "") for row in rows}
    missing = [episode_id for episode_id in requested if episode_id not in available]
    if missing:
        raise SystemExit(f"Unknown episode IDs: {missing}")

    entries = [
        {
            "domain": args.domain,
            "checkpoint_id": str(row["checkpoint_id"]),
            "episode_id": str(row["episode_id"]),
        }
        for row in rows
        if str(row.get("episode_id") or "") in requested
    ]
    identity = "\n".join(
        f"{item['domain']}:{item['checkpoint_id']}:{item['episode_id']}"
        for item in entries
    )
    payload = {
        "suite_name": args.suite_name or f"govmem_v4_symbolic_{args.domain}_selected",
        "dataset_name": "checkpoint_benchmark",
        "version": 1,
        "purpose": "Complete Medical episode validation selected by episode ID only.",
        "selection_policy": {
            "unit": "complete_episode",
            "domain": args.domain,
            "selected_episodes": requested,
            "uses_answer_fields": False,
            "uses_query_type_fields": False,
            "uses_attack_type_fields": False,
            "uses_evidence_fields": False,
            "uses_scorer_fields": False,
            "selected_checkpoint_identity_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        },
        "entries": entries,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"episodes={len(requested)} checkpoints={len(entries)} output={output}")


if __name__ == "__main__":
    main()
