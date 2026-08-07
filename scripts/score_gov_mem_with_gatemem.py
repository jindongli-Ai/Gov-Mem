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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score Gov-Mem predictions with the official GateMem scorer."
    )
    parser.add_argument("--domain", required=True, choices=["education", "household", "medical", "office"])
    parser.add_argument("--predictions", required=True, help="Path to Gov-Mem predictions.jsonl")
    parser.add_argument("--out_dir", required=True, help="Directory for official GateMem scoring outputs")
    parser.add_argument("--use_llm_judge", action="store_true", default=True, help="Enable the official LLM-as-a-judge stage")
    parser.add_argument("--judge_provider", default="yunwu")
    parser.add_argument("--judge_model", default="gpt-4o")
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_max_output_tokens", type=int, default=4096)
    parser.add_argument("--judge_api_base", default="https://yunwu.ai/v1")
    parser.add_argument("--judge_api_key_env", default="YUNWU_API_KEY")
    parser.add_argument("--judge_concurrency", type=int, default=None)
    parser.add_argument(
        "--gate_by_action",
        action="store_true",
        help="Enable the optional strict post-hoc action gate. Disabled by default for GateMem paper-compatible metrics.",
    )
    args = parser.parse_args()

    predictions_path = Path(args.predictions).resolve()
    out_dir = Path(args.out_dir).resolve()

    rows = load_and_validate_predictions(predictions_path)
    result = {
        "validated_predictions": len(rows),
        "predictions": str(predictions_path),
        "domain": args.domain,
        "out_dir": str(out_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    run_official_scorer(
        domain=args.domain,
        predictions_path=predictions_path,
        out_dir=out_dir,
        use_llm_judge=bool(args.use_llm_judge),
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_max_output_tokens=args.judge_max_output_tokens,
        judge_api_base=args.judge_api_base,
        judge_api_key_env=args.judge_api_key_env,
        judge_concurrency=args.judge_concurrency,
        gate_by_action=bool(args.gate_by_action),
    )


if __name__ == "__main__":
    main()
