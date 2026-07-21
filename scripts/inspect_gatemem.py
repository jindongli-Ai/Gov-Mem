from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.data.gatemem import DATASET_ROOT, dataset_overview, iter_checkpoints, iter_episodes


def main() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"GateMem dataset directory not found: {DATASET_ROOT}")

    overview = dataset_overview()
    print("GateMem dataset root:")
    print(DATASET_ROOT)
    print()
    print("Domain counts:")
    print(json.dumps(overview, indent=2, ensure_ascii=False))
    print()

    sample_episode = next(iter_episodes("education"))
    sample_checkpoint = next(iter_checkpoints("education"))

    print("Sample education episode keys:")
    print(sorted(sample_episode.keys()))
    print()
    print("Sample education checkpoint keys:")
    print(sorted(sample_checkpoint.keys()))


if __name__ == "__main__":
    main()
