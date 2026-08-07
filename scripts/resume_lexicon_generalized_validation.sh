#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

while ! curl -sS --connect-timeout 5 --max-time 10 https://yunwu.ai/v1/models >/dev/null 2>&1; do
  sleep 30
done

exec python scripts/run_gatemem_suite.py \
  --suite_manifest experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json \
  --output_dir outputs/2026-07-30-rag_naive_v3_stage2_generalization_40_seed20260731_lexicon_generalized_v1 \
  --config configs/rag_naive_v3_gpt4omini_reasoning_rerank_long_context_enabled.yaml \
  --experiment_mode rag_naive_v3_typed_rerank \
  --data_root dataset/GateMem/gatemem/data \
  --parallel_domains 1 \
  --parallel_episodes 1 \
  --resume
