# Gov-Mem v3 Pure Evaluation with GPT-4o-mini: 200 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: constrained LLM rerank plus deterministic governance checks
- Long-context field ledger: **disabled**
- Manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_200_seed20260731.json`
- Checkpoints: 200 total, 50 per domain
- MGS: `U * (1 - A) * (1 - F)`, followed by the arithmetic mean across four domains

The episode pipeline ran in parallel with one leased OpenLux key per active
episode. The initial parallel official scoring pass stalled on an Office judge
request; the episode outputs were retained and Office was rescored serially.
All final domain summaries contain 50 judged checkpoints with `gpt-4o` and zero
judge parse failures.

## Result

| Domain | Checkpoints | U | A | F | MGS | Action |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 63.16% | 28.57% | 0.00% | 45.11% | 82.00% |
| Office | 50 | 70.00% | 5.56% | 0.00% | 66.11% | 80.00% |
| Education | 50 | 33.33% | 0.00% | 0.00% | 33.33% | 66.00% |
| Household | 50 | 53.85% | 20.83% | 0.00% | 42.63% | 62.00% |
| **Four-domain average** | **200** | | | | **46.80%** | |

## Comparison with the 40-Checkpoint Run

| Run | Checkpoints | Average MGS |
|---|---:|---:|
| Pure Gov-Mem, same configuration | 40 | 47.50% |
| Pure Gov-Mem, expanded validation | 200 | **46.80%** |
| Difference | +160 | **-0.70 pp** |

The close 40/200 averages suggest the 40-case result was not substantially
optimistic. The larger sample reveals that the main remaining risk is not
deletion leakage (`F=0` in every domain), but answer-level privacy leakage in
Medical and Household. Education remains the main Utility weakness.

## Artifacts

- Output directory: `outputs/2026-08-04_rag_naive_v3_pure_gpt4omini_generalization200/`
- Config: `configs/rag_naive_v3_openlux_gpt4omini_embedding3small_pure.yaml`
- Official summaries: `*/official_eval/checkpoint_benchmark/*/summary.json`
