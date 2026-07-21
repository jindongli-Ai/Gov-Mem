from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.evolution.self_evolving_loop import SelfEvolvingLoop
from gov_mem.utils.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a dev-only self-evolving loop summary from existing round artifacts.")
    parser.add_argument("--round0", required=True, help="Round 0 suite_summary.json or official summary.json")
    parser.add_argument("--round1", required=True, help="Round 1 suite_summary.json or official summary.json")
    parser.add_argument("--round2", required=True, help="Round 2 suite_summary.json or official summary.json")
    parser.add_argument("--round3", required=True, help="Round 3 suite_summary.json or official summary.json")
    parser.add_argument("--skill_path", required=True, help="Frozen governance_skill_library.jsonl")
    parser.add_argument("--rule_path", required=True, help="Frozen rule_patches.jsonl")
    parser.add_argument("--output_dir", default="outputs/evolution", help="Output directory")
    args = parser.parse_args()

    loop = SelfEvolvingLoop()
    round_scores = [
        _score_from_path(loop, args.round0, 0),
        _score_from_path(loop, args.round1, 1),
        _score_from_path(loop, args.round2, 2),
        _score_from_path(loop, args.round3, 3),
    ]
    final_skills = read_jsonl(args.skill_path)
    final_rules = read_jsonl(args.rule_path)
    loop.materialize(
        round_scores=round_scores,
        final_skills=final_skills,
        final_rules=final_rules,
        output_dir=args.output_dir,
    )
    print(f"rounds={len(round_scores)}")
    print(f"final_skills={len(final_skills)}")
    print(f"final_rules={len(final_rules)}")
    print(f"output_dir={Path(args.output_dir).resolve()}")


def _score_from_path(loop: SelfEvolvingLoop, path: str, round_index: int):
    resolved = Path(path)
    if resolved.name == "suite_summary.json":
        return loop.score_from_suite_summary(resolved, round_index=round_index)
    return loop.score_from_official_summary(resolved, round_index=round_index)


if __name__ == "__main__":
    main()
