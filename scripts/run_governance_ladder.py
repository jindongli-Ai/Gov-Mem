"""Run the Gov-Mem evaluation ladder without promoting unverified changes.

The ladder keeps fast structural checks and targeted official-score regressions
separate from per-domain and full-suite confirmation runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PHASES = {
    "static": ["python", "scripts/run_structural_evolution_smoke.py"],
    "targeted": ["python", "scripts/run_gatemem_suite.py", "--manifest", "experiments/gatemem_suites/round7_slot_family_certificate.json"],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["static", "targeted"])
    parser.add_argument("--output-root", default="outputs/dev_evolution/governance_ladder")
    parser.add_argument("--config", default="configs/gov_mem_lightweight_adaptation.yaml")
    args = parser.parse_args()
    command = list(PHASES[args.phase])
    if args.phase == "targeted":
        command.extend(["--output-root", str(Path(args.output_root) / "targeted"), "--config", args.config])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
