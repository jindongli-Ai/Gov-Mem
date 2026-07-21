from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize round-wise evolved-skill score artifacts.")
    parser.add_argument("--evolution_dir", default="outputs/evolution", help="Directory containing round_*_score.json")
    args = parser.parse_args()

    root = Path(args.evolution_dir)
    rows = []
    for round_path in sorted(root.glob("round_*_score.json")):
        row = read_json(round_path)
        rows.append(row)
    table_path = root / "evolution_curve_readback.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round_index", "U", "A", "F", "OR", "MGS", "source"])
        for row in rows:
            writer.writerow([row.get("round_index"), row.get("U"), row.get("A"), row.get("F"), row.get("OR"), row.get("MGS"), row.get("source")])
    print(f"rounds={len(rows)}")
    print(f"table_path={table_path.resolve()}")


if __name__ == "__main__":
    main()
