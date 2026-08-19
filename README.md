# Gov-Mem

This workspace uses GateMem as the default benchmark dataset for Gov-Mem.

## Current Research Snapshot (2026-08-19)

This section records the current paper-facing Gov-Mem framework and its latest
full-benchmark results. Treat the commit containing this snapshot as a
recovery point: later framework changes should be committed separately, so a
regression can be diagnosed or the previous version can be restored from Git
history.

### Active development version (2026-08-19)

The current development snapshot is **Gov-Mem-v4-Symbolic-dev7**. It is based on the frozen
`rag_naive_v3_typed_rerank` framework (`Gov-Mem-v3.0`) and preserves the
complete checkpoint-visible GateMem turn as structured retrieval provenance:
`turn_id`, timestamp, principal, role, turn kind, original text, checkpoint,
and the source turn object. Stage 2 receives this data as valid JSON and does
not need to re-extract these fields from prose.

The current implementation line is **Gov-Mem-v4-Symbolic-dev7**,
selected by `experiment.mode: govmem_v4_symbolic`. It retains v3 retrieval and
adds lightweight typed Symbolic annotations: principal-role consistency, an
Evidence-Principal-Entity relation graph, explicit lifecycle-event assertions,
and a source-bound current-state ledger. Dev7 adds an authorization-aware
evidence boundary and a deterministic claim-level provenance verifier. The
ledger records requested slots,
candidate values, source memory/turn provenance, and conflicts from retrieved
evidence only. It does not recover hidden transcript fields or make permission
decisions; dev7 may filter evidence only when the explicit authorization
boundary denies it. It does not add LLM calls. Lifecycle assertions
recognize only explicit language such as `deleted`, `revoked`, `superseded`, or
`updated from ... to ...`; they do not infer state from ordinary words such as
`current` or `latest`.

Dev7's claim-level provenance verifier is now connected to the actual
`govmem_v4_symbolic` RAG-Naive path. The answering model may return a
source-bound `claim_contract` in the same JSON response; the adapter maps
GateMem source-message IDs to retrieved chunk IDs only within the current
checkpoint, then records the deterministic audit in the official prediction's
`memory_audit` and in `answer_grounding.policy_privacy_verifier`. A missing
contract is recorded as `claim_provenance_not_applicable`; no synthetic source
is created and no additional LLM call is made. The current RAG-Naive adapter is
shadow-only, so malformed contracts are audited but do not suppress an answer;
the stateful executor continues to use the full fail-closed field-state
projection contract.
The first real integration smoke found that the base model's optional claim
contracts are not yet reliable enough for hard enforcement; this diagnostic is
recorded in
[`2026-08-19 claim-provenance smoke`](experiments/result/2026-08-19_Gov-Mem-v4-Symbolic-dev7_claim_provenance_smoke.md)
and is not a paper performance result.
The final paper-facing name is
**Gov-Mem-v4-Symbolic**, which will be frozen incrementally after each Symbolic
step passes regression and benchmark checks. The complete naming and
promotion record is in
[`VERSION_LOG.md`](VERSION_LOG.md).

The dev3 state-ledger increment was validated on one complete episode per
domain: Medical 27, Office 32, Education 18, and Household 24 checkpoints
(101 total). The run used OpenLux, `gpt-4o-mini`,
`text-embedding-3-small`, a 30-key pool with four concurrently leased keys,
and four episode workers.
All 101 prompt audits are complete; 46 state-relevant prompts contain the
retrieved-evidence-only ledger, while certificate and target-binding shadow
fields occur in zero prompts. Candidate ordering and filtering were unchanged
and the added LLM-call count was zero. This is an engineering validation, not a
2,218-checkpoint paper result and must not be mixed into the frozen full-
benchmark U/A/F/MGS table.

The official GateMem judge completed all 101 cases for this state-ledger run:

| Domain | Checkpoints | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 27 | 60.00% | 55.56% | 0.00% | 26.67% | 59.26% |
| Office | 32 | 55.56% | 0.00% | 0.00% | 55.56% | 78.13% |
| Education | 18 | 83.33% | 16.67% | 0.00% | 69.44% | 88.89% |
| Household | 24 | 75.00% | 12.50% | 0.00% | 65.63% | 75.00% |
| **All scored cases** | **101** | **66.67%** | **21.21%** | **0.00%** | **52.53%** | **74.26%** |

The all-cases row is weighted by the number of GateMem query types (33 utility,
33 privacy, and 35 safety cases). The four-domain arithmetic mean of domain MGS
is 54.32%. Relative to the preceding lifecycle smoke on the same 101
checkpoints, this run is a positive engineering signal (`U` +3.03 points,
`A` -3.03 points, weighted `MGS` +4.32 points), not a causal ablation because
the memory-model temperature is 0.2 and the comparison is a single stochastic
run. The full diagnostic record is in
[`experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-smoke4_openlux_gpt4omini.md`](experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-smoke4_openlux_gpt4omini.md).

The validity-shadow audit adds deterministic `EvidenceValidityCertificate`
records for explicit lifecycle states. The target-binding v2 increment also
links a lifecycle claim to an earlier retrieved target only when the match is
unique and temporally valid; ambiguous and unbound cases are retained as
explicit audit states. These records remain internal diagnostic metadata and
are deliberately excluded from Stage 2 and answer prompts, so shadow mode
does not change the LLM-visible evidence contract. Across the validated
101-checkpoint run, all answer and Stage 2 prompt audits contained zero
certificate and target-binding fields while internal records were retained.
The official judge completed all 101 cases:

| Domain | Checkpoints | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Medical | 27 | 70.00% | 66.67% | 0.00% | 23.33% |
| Office | 32 | 55.56% | 0.00% | 0.00% | 55.56% |
| Education | 18 | 0.00% | 16.67% | 0.00% | 0.00% |
| Household | 24 | 62.50% | 12.50% | 0.00% | 54.69% |
| **All scored cases** | **101** | **51.52%** | **24.24%** | **0.00%** | **39.03%** |

This is an engineering diagnostic rather than a paper performance result:
it uses one episode per domain, Education has only six utility cases, and the
memory-model temperature is 0.2. It must not be mixed into the frozen
2,218-checkpoint typed-rerank table. Enforcement is not enabled. The complete
score is recorded in
`outputs/2026-08-14_Gov-Mem-v4-Symbolic-dev2-target-binding-shadow-v2_gpt4omini_embedding3small_smoke4/official_score.json`.

### Canonical symbolic experiment method

The canonical symbolic method is **Gov-Mem-Symbolic**, selected by the
`govmem_symbolic` experiment mode. It is the neuro-symbolic governance track
with typed memory, policy/state ledgers, governed evidence, and slot-level
authorization. The historical `rag_policy_amem` mode is retained only as a
backward-compatible alias for reproducing old runs.

The separate frozen benchmark track is `rag_naive_v3_typed_rerank`:

1. GateMem-compatible Stage 1 RAG-Naive retrieval over visible dialogue turns,
   using the raw query and `top_k=20`.
2. Stage 2 typed, constrained evidence reranking over retrieved evidence only.
   The reranker binds query fields to typed source evidence, resolves current
   versus stale/deleted values, and preserves field-level answer boundaries.
3. Source-bound answer realization with explicit deletion and sensitive-field
   boundaries.

The formal frozen benchmark protocol disables the complete-transcript/
long-context ledger, gold feedback, and runtime experience updates. Its results
must not be labeled as Gov-Mem-Symbolic results.

### Latest frozen typed-rerank full-benchmark performance

The following table is the frozen `rag_naive_v3_typed_rerank` track, not the
Gov-Mem-Symbolic track. A new Symbolic full-benchmark table must be generated
under `govmem_symbolic` after the unified Symbolic configuration is frozen.

These results cover all 2,218 GateMem checkpoints: Medical 579, Office 547,
Education 540, and Household 552. All seven runs use the same checkpoint
manifest, Stage 1 retrieval, embedding model (`text-embedding-3-small`), Stage
2 configuration, and official evaluator (`gpt-4o`, temperature 0.0). Only the
base LLM changes. The official GateMem scorer uses
`gate_by_action=false`.

`U` is utility accuracy, `A` is answer-level access-control/privacy leakage,
and `F` is answer-level active-forgetting/deletion leakage. The headline score
is `MGS = U * (1 - A) * (1 - F)`, averaged across the four domain MGS values.
Action accuracy and over-refusal are supplementary metrics.

| Frozen typed-rerank base LLM | Medical MGS | Office MGS | Education MGS | Household MGS | Four-domain avg. MGS |
|---|---:|---:|---:|---:|---:|
| GPT-4o-mini | 32.89% | 43.35% | 25.66% | 34.39% | **34.07%** |
| GPT-5-mini | 42.35% | 65.26% | 27.73% | 43.09% | **44.61%** |
| GPT-5.4 | **56.88%** | 62.61% | 23.14% | 38.21% | **45.21%** |
| GPT-5.4-mini | 47.56% | 54.43% | 24.53% | 38.07% | **41.15%** |
| Gemini-2.5-Flash-Lite | 49.17% | **63.82%** | **30.43%** | 37.31% | **45.18%** |
| DeepSeek-V4-Flash | **58.89%** | 61.56% | 26.88% | **47.47%** | **48.70%** |
| Llama-3.3-70B-Instruct | 27.78% | 34.92% | 14.71% | 20.70% | 24.53% |

DeepSeek-V4-Flash is currently the best of these seven tested base LLMs by
average MGS, with an absolute improvement of 14.63 percentage points over
GPT-4o-mini. It also gives the best Medical and Household domain MGS in this
comparison.
The full set is important: smaller 40-, 200-, or 800-checkpoint diagnostics
are not interchangeable with these results.

GPT-5.4-mini, DeepSeek-V4-Flash, and Llama-3.3-70B-Instruct are included in
the main table as full-benchmark model comparisons. Their complete
domain-level metrics and protocols are documented in the linked result
artifacts below.

### Safety and interpretation boundaries

The official MGS is an answer-level benchmark metric. It is not equivalent to
zero exposure of restricted evidence in intermediate prompts. The independent
context audit for the strict runs reports non-zero privacy/deletion context
exposure in particular for Medical and Household. Therefore this repository
does not claim that the current v3 system has zero end-to-end leakage.

The latest frozen typed-rerank results are not a complete four-model, full-set
paired comparison against locally regenerated plain RAG-Naive predictions.
GateMem's released GPT-5.4 RAG-Naive reference Utility values are Medical
64.8%, Office 74.0%, Education 32.8%, and Household 51.1%. These are Utility
references only and should not be presented as MGS comparisons. A future
paper table must report the exact baseline protocol, checkpoint manifest,
model, evaluator, and whether the comparison is on U, A, F, or MGS.

### Main result artifacts

- [GPT-5.4 strict full result](experiments/result/2026-08-05-22-53-20_Gov-Mem_v3_full_all_2218_openlux_gpt54_strict.md)
- [GPT-5-mini strict full result](experiments/result/2026-08-05-23-57-05_Gov-Mem_v3_full_all_2218_openlux_gpt5mini_strict.md)
- [Gemini-2.5-Flash-Lite strict full result](experiments/result/2026-08-06-01-41-13_Gov-Mem_v3_full_all_2218_openlux_gemini25flashlite_strict.md)
- [GPT-4o-mini strict full result](experiments/result/2026-08-05_Gov-Mem_v3_paper_compatible_2218_openlux_gpt4omini_strict.md)
- [GPT-5.4-mini strict full result](experiments/result/2026-08-05_Gov-Mem_v3_full_all_2218_openlux_gpt54mini_strict.md)
- [DeepSeek-V4-Flash strict full result (2026-08-11)](experiments/result/2026-08-11_Gov-Mem_v3_full_all_2218_openlux_deepseekv4flash_strict.md)
- [Llama-3.3-70B-Instruct strict full result (2026-08-11)](experiments/result/2026-08-11_Gov-Mem_v3_full_all_2218_openlux_llama33_70b_instruct_strict.md)
- [Full U/A/F/MGS Markdown summary table (2026-08-11)](experiments/result/2026-08-11_Gov-Mem_v3_full_all_2218_performance_summary.md)
- [Current implementation and contribution reconstruction](report.md)

When reporting a new result, record the commit, config, model/provider,
manifest, evaluator, and whether long-context or gold-derived feedback was
enabled. Commit improvements as new history points instead of overwriting the
current snapshot.

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

## Historical Frozen Framework Diagnostic

The current frozen Gov-Mem v3 framework uses RAG-Naive Retrieval in Stage 1 and
typed, constrained reasoning reranking over retrieved evidence in Stage 2. The
table below preserves an earlier strict combined 800-checkpoint diagnostic for
traceability. That run used the long-context field ledger and therefore is not
directly comparable to the GateMem paper main table or to the formal protocol
below.
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

Configuration for this historical diagnostic:

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official GateMem judge provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: typed reasoning rerank plus long-context field ledger, source-bound safe wording
- This result must be labeled as a strict/ablation diagnostic, not as a GateMem paper-compatible main result

Formal GateMem-compatible Gov-Mem evaluations use the official scorer with
`gate_by_action=false`, memory-system temperature `0.2`, judge temperature
`0.0`, `text-embedding-3-small`, turn-level RAG-Naive retrieval with
`top_k=20`, and the retrieved-evidence-only Stage 2 configuration
`configs/rag_naive_v3_openlux_gpt4omini_embedding3small_pure.yaml`. Context-audit
rates are reported separately because they are not part of GateMem's paper MGS.

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

Formal Gov-Mem evaluations use retrieved evidence only in Stage 2. In
particular, `long_context_field_ledger.enabled` must remain `false`: that
optional component reads the complete visible checkpoint transcript and is
reserved for explicitly labeled Long-Context ablations. The formal Gemini
configuration is `configs/rag_naive_v3_openlux_gemini25flashlite_embedding3small_pure.yaml`.

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
  --experiment_mode govmem_symbolic \
  --max_instances 30 \
  --stage all
```

```bash
python3 run_govmem.py \
  --dataset_name gatemem \
  --data_path dataset/GateMem/gatemem/data/medical \
  --output_dir outputs/govmem_deepseek_v4_flash_medical \
  --config configs/govmem_deepseek_v4_flash.yaml \
  --experiment_mode govmem_symbolic \
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
  --experiment_mode govmem_symbolic \
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
  --experiment_mode govmem_symbolic \
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
