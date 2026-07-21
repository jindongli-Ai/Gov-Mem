#!/usr/bin/env bash
set -euo pipefail
cd /data_nvme/user/jli/codes/2027_Gov-Mem
PY=/data_nvme/user/jli/codes/miniconda3/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YUNWU_API_KEY="$($PY -c "import re; print(re.findall(r'sk-[A-Za-z0-9_-]+', open('README_API_Yunwu.md').read())[6])")"
MANIFEST=experiments/gatemem_suites/round50_ingestion_item_probe1.json
$PY - <<'PY'
import json
from pathlib import Path
Path("experiments/gatemem_suites/round50_ingestion_item_probe1.json").write_text(json.dumps({
    "suite_name": "round50_ingestion_item_probe1",
    "dataset_name": "checkpoint_benchmark",
    "version": 1,
    "entries": [{"domain": "medical", "checkpoint_id": "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_21"}],
}, indent=2) + "\n")
PY
OUT=outputs/dev_evolution/round50_ingestion_item_probe1
mkdir -p "$OUT"
nohup "$PY" scripts/run_gatemem_suite.py --suite_manifest "$MANIFEST" --output_dir "$OUT" \
  --config configs/gov_mem_lightweight_adaptation.yaml --experiment_mode rag_policy_amem \
  --data_root dataset/GateMem/gatemem/data --parallel_domains 1 > "$OUT/launcher.log" 2>&1 &
echo $!
