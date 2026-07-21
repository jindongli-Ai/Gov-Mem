#!/usr/bin/env bash
set -euo pipefail
cd /data_nvme/user/jli/codes/2027_Gov-Mem
PY=/data_nvme/user/jli/codes/miniconda3/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YUNWU_API_KEYS="$($PY -c "import re; print(','.join(re.findall(r'sk-[A-Za-z0-9_-]+', open('README_API_Yunwu.md').read())[0:2]))")"
MANIFEST=experiments/gatemem_suites/round58_open_attribute_ledger2.json
$PY - <<'PY'
import json
from pathlib import Path

entries = [
    {"domain": "household", "checkpoint_id": "household_episode_custom_en_017_apricot_archive_apricot_adapter_media_backup_private_note_ckpt_01"},
    {"domain": "education", "checkpoint_id": "education_episode_custom_en_027_meridian_placement_meridian_mentors_dual_track_ckpt_13"},
]
Path("experiments/gatemem_suites/round58_open_attribute_ledger2.json").write_text(json.dumps({
    "suite_name": "round58_open_attribute_ledger2",
    "dataset_name": "checkpoint_benchmark",
    "version": 1,
    "entries": entries,
}, indent=2) + "\n")
PY
OUT=outputs/dev_evolution/round58_open_attribute_ledger2
mkdir -p "$OUT"
nohup "$PY" scripts/run_gatemem_suite.py --suite_manifest "$MANIFEST" --output_dir "$OUT" \
  --config configs/gov_mem_lightweight_adaptation.yaml --experiment_mode rag_policy_amem \
  --data_root dataset/GateMem/gatemem/data --parallel_domains 2 > "$OUT/launcher.log" 2>&1 &
echo $!
