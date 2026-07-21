from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.eval.gatemem_official import load_and_validate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Gov-Mem predictions against GateMem official format.")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    args = parser.parse_args()

    predictions_path = Path(args.predictions).resolve()
    rows = load_and_validate_predictions(predictions_path)

    summary = {
        "predictions": str(predictions_path),
        "num_rows": len(rows),
        "status": "ok",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
