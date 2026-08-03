# Gov-Mem

This workspace uses GateMem as the default benchmark dataset for Gov-Mem.

## Dataset

The raw GateMem dataset is stored under `dataset/GateMem`.

Raw dataset files are treated as read-only:

- Do not edit files under `dataset/GateMem/`
- Put any derived artifacts in new directories outside the raw dataset tree

## Local Utilities

- `src/gov_mem/data/gatemem.py`: GateMem loader utilities
- `scripts/inspect_gatemem.py`: quick dataset sanity check
- `scripts/score_gov_mem_with_gatemem.py`: score Gov-Mem predictions with the official GateMem evaluator
- `scripts/validate_gov_mem_predictions.py`: validate prediction-file format before scoring

## Official Evaluation

The official GateMem evaluation toolkit is vendored at:

- `third_party/GateMem-official`

Gov-Mem should use the official GateMem scorer for reported benchmark results.
This keeps evaluation aligned with the GateMem paper and its released metric
definitions.

## Latest Frozen Framework Performance

The current frozen Gov-Mem v3 framework uses RAG-Naive Retrieval in Stage 1 and
typed, constrained reasoning reranking with a long-context field ledger in Stage
2. The table below reports the strict combined 800-checkpoint evaluation: 50
checkpoints from the earlier 200-case run plus 150 new checkpoints per domain.
The framework code and all experiment settings were unchanged across the two
runs.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 200 | 64.38% | 34.43% | 4.55% | 40.30% | 76.50% | 13.70% |
| Office | 200 | 59.65% | 8.96% | 2.63% | 52.88% | 79.00% | 22.81% |
| Education | 200 | 43.75% | 19.44% | 6.25% | 33.04% | 74.50% | 10.94% |
| Household | 200 | 49.12% | 23.68% | 2.99% | 36.37% | 65.00% | 21.05% |
| **Four-domain average MGS** | **800** | | | | **40.65%** | | |

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the
arithmetic mean of the four domain MGS values. `A` is answer-level privacy
leakage, `F` is answer-level deletion leakage, `Action` is action accuracy, and
`OR` is over-refusal rate among utility cases.

| Evaluation | Four-domain average MGS |
|---|---:|
| Earlier 200-case slice | 56.04% |
| New 600-case slice | 35.13% |
| Strict combined 800-case result | **40.65%** |

Configuration for the reported result:

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official GateMem judge provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: typed reasoning rerank, long-context field ledger, source-bound safe wording

The detailed privacy/deletion diagnostics and Overleaf-ready LaTeX tables are
available in [`experiments/result/2026-08-03_rag_naive_v3_stage2_generalization_800_overleaf_tables.tex`](experiments/result/2026-08-03_rag_naive_v3_stage2_generalization_800_overleaf_tables.tex).

## Quick Start

```bash
python3 scripts/inspect_gatemem.py
```

Validate a predictions file:

```bash
python3 scripts/validate_gov_mem_predictions.py \
  --predictions outputs/gov_mem_medical/predictions.jsonl
```

Score predictions with the official GateMem scorer:

```bash
python3 scripts/score_gov_mem_with_gatemem.py \
  --domain medical \
  --predictions outputs/gov_mem_medical/predictions.jsonl \
  --out_dir outputs/gov_mem_medical_scored
```

## Environment Note

The official GateMem scorer is used as-is from `third_party/GateMem-official`.
Its Python dependencies are not bundled into this repo automatically.
For this workspace, a local scorer dependency bundle has already been installed at:

- `third_party/python_deps/gatemem_eval`

The Gov-Mem wrapper scripts automatically add that directory to `PYTHONPATH`
when invoking the official GateMem scorer.

At minimum, the official scorer may require packages such as:

- `numpy`
- `tqdm`
- `PyYAML`
- `requests`
- `scikit-learn`

If you enable the official LLM judge or retrieval-heavy baselines, the full
`third_party/GateMem-official/requirements.txt` stack may also be needed.

## Gov-Mem Pipeline

Main entry:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_medical_debug \
  --config configs/govmem_default.yaml \
  --max_instances 10 \
  --stage all
```

## Base LLM Configuration

Gov-Mem now treats the experiment-time backbone model as a configurable `base LLM`
rather than hard-coding a specific model into the framework.

The main config interface is:

```yaml
llm:
  base_model: gpt-5.4-nano-2026-03-17
  role_models: {}
```

- `base_model`: the default LLM used by the framework for query planning, action decision, and other base-model-controlled stages
- `role_models`: optional per-role overrides such as `memory_ingestion`, `query_planning`, `action_decision`, or `answering`

Example: use one base LLM for the whole framework

```yaml
llm:
  base_model: DeepSeek-V4-Flash
  role_models: {}
```

Example: keep one base LLM but override answering only

```yaml
llm:
  base_model: Qwen3.5-plus
  role_models:
    answering: gpt-5
```

Ready-to-run example configs are provided under `configs/`:

- `configs/govmem_gpt5_nano.yaml`
- `configs/govmem_gpt5.yaml`
- `configs/govmem_deepseek_v4_flash.yaml`
- `configs/govmem_qwen35_plus.yaml`

Example runs:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_gpt5_nano_medical \
  --config configs/govmem_gpt5_nano.yaml \
  --experiment_mode rag_policy_amem \
  --max_instances 30 \
  --stage all
```

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_deepseek_v4_flash_medical \
  --config configs/govmem_deepseek_v4_flash.yaml \
  --experiment_mode rag_policy_amem \
  --max_instances 30 \
  --stage all
```

You can also override the base LLM at runtime before an experiment without editing
the YAML file:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_runtime_override \
  --config configs/govmem_default.yaml \
  --experiment_mode rag_policy_amem \
  --base_model DeepSeek-V4-Flash \
  --llm_provider yunwu \
  --llm_api_base https://yunwu.ai/v1 \
  --llm_api_key_env YUNWU_API_KEY \
  --max_instances 10 \
  --stage all
```

Optional role-specific overrides are also supported:

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_role_override \
  --config configs/govmem_default.yaml \
  --experiment_mode rag_policy_amem \
  --base_model Qwen3.5-plus \
  --role_model action_decision=gpt-5 \
  --role_model answering=gpt-5 \
  --max_instances 10 \
  --stage all
```

Each run writes the fully resolved LLM settings to:

- `outputs/.../run_metadata.json`
- `outputs/.../debug_cases/<dataset>/<instance>.json`

Supported stages:

```bash
python3 run_govmem.py --stage all
python3 run_govmem.py --stage ingest
python3 run_govmem.py --stage retrieve
python3 run_govmem.py --stage answer
python3 run_govmem.py --stage evaluate
```

Main Gov-Mem outputs:

- `outputs/.../memory/<dataset>/<instance>/memory_items.jsonl`
- `outputs/.../query_plan/<dataset>/<instance>.json`
- `outputs/.../retrieval/<dataset>/<instance>.json`
- `outputs/.../reasoning/<dataset>/<instance>.json`
- `outputs/.../predictions/<dataset>/<instance>.json`
- `outputs/.../predictions/<dataset>/predictions.jsonl`
- `outputs/.../experience/<dataset>/experience_bank.jsonl`
- `outputs/.../eval/<dataset>/metrics.json`
- `outputs/.../eval/<dataset>/case_results.jsonl`

If the dataset is GateMem, the runner also emits official scorer outputs under:

- `outputs/.../official_eval/gatemem/<domain>/summary.json`

## API Environment

Gov-Mem now distinguishes explicitly between real LLM mode and heuristic fallback mode.

Default provider is `yunwu`, using:

- `YUNWU_API_KEY`
- `YUNWU_BASE_URL`

Default Yunwu base URL:

```bash
export YUNWU_BASE_URL="https://yunwu.ai/v1"
```

If a valid API key is detected, logs will show:

```text
[Gov-Mem] Real LLM mode enabled: provider=..., model=...
```

If no valid API key is detected and fallback is allowed, logs will show:

```text
[Gov-Mem Warning] No valid LLM API key detected. Falling back to heuristic mode. Accuracy may be invalid.
```
