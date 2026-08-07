from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.eval.benchmark_official import load_and_validate_predictions, run_official_scorer
from gov_mem.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge GateMem shard predictions and run official scorer.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument(
        "--gate_by_action",
        action="store_true",
        help="Enable the optional strict post-hoc action gate; disabled for GateMem paper-compatible metrics.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    merged_pred_dir = ensure_dir(output_dir / "predictions" / "gatemem")
    merged_pred_path = merged_pred_dir / "predictions.jsonl"

    rows_by_id: dict[str, dict] = {}
    for shard_dir_str in args.shard_dirs:
        shard_dir = Path(shard_dir_str)
        pred_path = shard_dir / "predictions" / "gatemem" / "predictions.jsonl"
        rows = load_and_validate_predictions(pred_path)
        for row in rows:
            checkpoint_id = str(row["checkpoint_id"])
            if checkpoint_id in rows_by_id:
                raise ValueError(f"Duplicate checkpoint_id across shards: {checkpoint_id}")
            rows_by_id[checkpoint_id] = row

    merged_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    with merged_pred_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    official_out_dir = output_dir / "official_eval" / "gatemem" / args.domain
    run_official_scorer(
        domain=args.domain,
        predictions_path=merged_pred_path,
        out_dir=official_out_dir,
        gate_by_action=bool(args.gate_by_action),
    )


if __name__ == "__main__":
    main()
