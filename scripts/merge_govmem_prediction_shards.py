"""Merge completed Gov-Mem episode shards without rerunning model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = {str(row["checkpoint_id"]): row for row in read_jsonl(args.checkpoints)}
    selected: dict[str, tuple[dict, Path]] = {}
    duplicates: dict[str, list[str]] = {}
    for source_root in args.source:
        paths = sorted(source_root.glob("**/predictions/checkpoint_benchmark/predictions.jsonl"))
        for path in paths:
            for row in read_jsonl(path):
                checkpoint_id = str(row["checkpoint_id"])
                if checkpoint_id not in expected:
                    raise ValueError(f"Unexpected checkpoint_id {checkpoint_id} in {path}")
                if checkpoint_id in selected:
                    duplicates.setdefault(checkpoint_id, []).append(str(path))
                    continue
                selected[checkpoint_id] = (row, path)

    missing = sorted(set(expected) - set(selected))
    if missing:
        raise ValueError(f"Missing {len(missing)} checkpoint IDs; first={missing[:5]}")

    rows = [selected[checkpoint_id][0] for checkpoint_id in expected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "expected": len(expected),
        "selected": len(selected),
        "duplicate_checkpoint_ids": len(duplicates),
        "duplicate_rows_ignored": sum(len(paths) for paths in duplicates.values()),
        "sources": [str(path) for path in args.source],
        "embedding_selection": "text-embedding-3-large for all selected rows",
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
