"""Replay saved runtime prompts offline; never call a model or read gold labels."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from gov_mem.governance_runtime.v8_prompt_input import application_policy, pack_prompt, unpack_prompt
from gov_mem.utils.config import load_yaml_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/govmem_v8_late_governance_gemini25flashlite.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    counts = defaultdict(lambda: defaultdict(int))
    identities = hashlib.sha256()
    for path in sorted(args.predictions_root.glob("*/*/predictions/checkpoint_benchmark/*.json")):
        # Read only saved runtime prompt; neither labels nor answers inform changes.
        prediction = json.loads(path.read_text())
        domain = path.relative_to(args.predictions_root).parts[0]
        counts[domain]["prediction_files"] += 1
        prompt = prediction.get("raw_response", {}).get("prompt_audit", {}).get("stage2_rerank_prompt", {}).get("user_prompt")
        if not prompt:
            counts[domain]["missing_runtime_prompt"] += 1
            continue
        payload = json.loads(prompt)
        identities.update(str(path.relative_to(args.predictions_root)).encode() + b"\0" + prompt.encode())
        counts[domain]["runtime_prompts"] += 1
        counts[domain]["historical_chars"] += len(prompt)
        counts[domain]["original_compact_chars"] += len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        payload["access_policy"] = application_policy(config, domain)
        packed, stats = pack_prompt(payload)
        assert unpack_prompt(packed) == payload, f"Lossy input: {path}"
        counts[domain]["with_policy_unpacked_chars"] += stats["original_chars"]
        counts[domain]["with_policy_packed_chars"] += stats["packed_chars"]
        counts[domain]["roundtrip_passed"] += 1
    if not counts:
        raise ValueError("No saved predictions found")
    totals = {key: sum(values.get(key, 0) for values in counts.values())
              for key in set().union(*(set(v) for v in counts.values()))}
    result = {"kind": "offline_exact_input_roundtrip", "paid_api_calls": 0,
              "runtime_prompts_sha256": identities.hexdigest(),
              "domains": dict(counts), "total": totals,
              "limitations": [
                  "Only available saved runtime prompts are replayed; missing/error prompts are counted.",
                  "Character counts exclude system prompts, outputs and repair calls; they are not token or cost estimates.",
                  "Lossless decoding proves evidence preservation, not LLM comprehension or improved MGS.",
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
