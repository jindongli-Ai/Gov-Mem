#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
OUT=outputs/2026-08-01-rag_naive_v3_jellyfishp_deepseek_v4_flash_40

# Wait for the actual chat model, not only /models: the latter can stay healthy
# while the upstream channel is returning 500/429 responses. Probe the whole
# configured pool so one unhealthy key does not hold back the experiment.
while ! python - <<'PY'
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import requests

keys = list(dict.fromkeys(re.findall(r"sk-[A-Za-z0-9_-]+", Path("README_API_jellyfishp.md").read_text(encoding="utf-8"))))

def probe(key: str) -> bool:
    try:
        response = requests.post(
            "https://newapi.medu.chat/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-v4-flash",
                "temperature": 0,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "Reply with OK."}],
            },
            timeout=(6, 20),
        )
        return response.ok
    except requests.RequestException:
        return False

with ThreadPoolExecutor(max_workers=max(1, len(keys))) as pool:
    if any(future.result() for future in as_completed([pool.submit(probe, key) for key in keys])):
        raise SystemExit(0)
raise SystemExit(1)
PY
do
  sleep 30
done

exec python scripts/run_gatemem_suite.py \
  --suite_manifest experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json \
  --output_dir "$OUT" \
  --config configs/rag_naive_v3_jellyfishp_deepseek_v4_flash.yaml \
  --experiment_mode rag_naive_v3_typed_rerank \
  --data_root dataset/GateMem/gatemem/data \
  --parallel_domains 4 \
  --parallel_episodes 6 \
  --skip_official_eval
