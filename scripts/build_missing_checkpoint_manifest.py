"""Build a manifest containing only checkpoint IDs absent from completed shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = _read_jsonl(args.checkpoints)
    completed: set[str] = set()
    for completed_root in args.completed_root:
        prediction_paths = set(completed_root.glob("runs/*/predictions/checkpoint_benchmark/predictions.jsonl"))
        # Support both the historical `runs/*` layout and the local runtime
        # layout, where episode shards live directly below output/medical.
        prediction_paths.update(completed_root.glob("**/predictions/checkpoint_benchmark/predictions.jsonl"))
        for prediction_path in sorted(prediction_paths):
            for row in _read_jsonl(prediction_path):
                completed.add(str(row["checkpoint_id"]))

    missing = [
        {**row, "domain": str(row.get("domain") or "medical")}
        for row in expected
        if str(row["checkpoint_id"]) not in completed
    ]
    payload = {
        "suite_name": "govmem_symbolic_gpt4omini_large_medical_missing",
        "version": 1,
        "entries": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"expected={len(expected)} completed={len(completed)} missing={len(missing)}")


if __name__ == "__main__":
    main()
