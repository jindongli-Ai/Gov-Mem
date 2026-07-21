#!/usr/bin/env bash
set -euo pipefail
cd /data_nvme/user/jli/codes/2027_Gov-Mem
PY=/data_nvme/user/jli/codes/miniconda3/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YUNWU_API_KEYS="$($PY -c "import re; print(','.join(re.findall(r'sk-[A-Za-z0-9_-]+', open('README_API_Yunwu.md').read())[0:4]))")"
MANIFEST=experiments/gatemem_suites/round84_authority_candidate_admission4.json
$PY - <<'PY'
import json
from pathlib import Path

entries = [
    {"domain": "office", "checkpoint_id": "office_episode_custom_en_013_ember_emberline_dual_project_ckpt_01"},
    {"domain": "household", "checkpoint_id": "household_episode_custom_en_017_apricot_archive_apricot_adapter_media_backup_private_note_ckpt_01"},
    {"domain": "education", "checkpoint_id": "education_episode_custom_en_027_meridian_placement_meridian_mentors_dual_track_ckpt_13"},
    {"domain": "medical", "checkpoint_id": "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_06"},
]
Path(MANIFEST := "experiments/gatemem_suites/round84_authority_candidate_admission4.json").write_text(json.dumps({
    "suite_name": "round84_authority_candidate_admission4",
    "dataset_name": "checkpoint_benchmark",
    "version": 1,
    "entries": entries,
}, indent=2) + "\n")
PY
OUT=outputs/dev_evolution/round84_authority_candidate_admission4
mkdir -p "$OUT"
nohup "$PY" scripts/run_gatemem_suite.py --suite_manifest "$MANIFEST" --output_dir "$OUT" \
  --config configs/gov_mem_lightweight_adaptation.yaml --experiment_mode rag_policy_amem \
  --data_root dataset/GateMem/gatemem/data --parallel_domains 4 > "$OUT/launcher.log" 2>&1 &
echo $!
