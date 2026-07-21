#!/usr/bin/env python3
"""Fail closed when a run's typed graph evidence lacks source attestation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.evolution.attested_provenance_audit import audit_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit attested typed provenance in a Gov-Mem run directory.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_path", default=None)
    args = parser.parse_args()
    output = Path(args.output_path) if args.output_path else Path(args.run_dir) / "attested_provenance_audit.json"
    result = audit_run(run_dir=args.run_dir, output_path=output)
    print(f"attested_realized_atoms={result['attested_realized_atom_count']}/{result['realized_atom_count']}")
    print(f"audit_path={output.resolve()}")
    if not result["passed"]:
        raise SystemExit(f"attested_provenance_violations={len(result['violations'])}")


if __name__ == "__main__":
    main()
