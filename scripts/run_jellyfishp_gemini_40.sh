#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
OUT=outputs/2026-08-01-rag_naive_v3_jellyfishp_gemini_3_flash_40

exec python scripts/run_gatemem_suite.py \
  --suite_manifest experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json \
  --output_dir "$OUT" \
  --config configs/rag_naive_v3_jellyfishp_gemini_3_flash.yaml \
  --experiment_mode rag_naive_v3_typed_rerank \
  --data_root dataset/GateMem/gatemem/data \
  --parallel_domains 4 \
  --parallel_episodes 6 \
  --skip_official_eval
