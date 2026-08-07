# Gov-Mem v3 Full Evaluation with GPT-5.4-mini: 2218 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-5.4-mini` (undated, as requested)
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: unchanged frozen Gov-Mem v3 typed rerank and answer-redacted confirmation boundary
- Full manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Execution: all 2218 available checkpoints; four domains and episode shards ran in parallel with isolated OpenLux keys

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the arithmetic mean of the four domain MGS values. `A` is answer-level privacy leakage and `F` is answer-level deletion/staleness leakage. Action and OR are reported for completeness and are not multiplied into MGS.

## Full Result

| Domain | Checkpoints | U | A | F | MGS | Action | OR | Privacy context | Deletion context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 66.67% | 30.21% | 6.21% | 43.64% | 74.61% | 10.95% | 0.00% | 0.00% |
| Office | 547 | 55.84% | 2.34% | 1.35% | 53.80% | 82.63% | 17.53% | 0.00% | 0.00% |
| Education | 540 | 35.00% | 13.89% | 6.67% | 28.13% | 72.59% | 18.33% | 0.00% | 0.00% |
| Household | 552 | 45.65% | 14.13% | 1.63% | 38.56% | 70.47% | 15.76% | 0.00% | 0.00% |
| **Four-domain average** | **2218** | | | | **41.03%** | | | | |

## Comparison Across Memory-System Base LLMs

All three runs use the same 2218-checkpoint manifest, frozen framework, `text-embedding-3-small`, and official `gpt-4o` evaluator.

| Domain | GPT-4o-mini | GPT-5.4-nano | GPT-5.4-mini | Mini delta vs 4o-mini |
|---|---:|---:|---:|---:|
| Medical | 37.18% | 37.28% | 43.64% | +6.46 pp |
| Office | 53.32% | 55.99% | 53.80% | +0.48 pp |
| Education | 31.31% | 29.26% | 28.13% | -3.18 pp |
| Household | 34.32% | 30.29% | 38.56% | +4.24 pp |
| **Four-domain average** | **39.03%** | **38.21%** | **41.03%** | **+2.00 pp** |

## Interpretation

- The undated `gpt-5.4-mini` is the strongest of the three tested memory-system base LLMs on the full 2218-checkpoint population: `41.03%` average MGS.
- It improves Medical and Household substantially and slightly improves Office relative to GPT-4o-mini, but Education remains the main bottleneck and drops by 3.18 percentage points.
- The gain is not caused by retrieval changes: Stage 1, embedding, framework code, and official evaluator were held fixed. Context-level privacy/deletion exposure remained 0% in every domain.
- The remaining loss is concentrated in final answer utility and answer-level leakage, especially Education.
- No framework code was changed for this comparison; only the memory-system base model was changed.

## Artifacts

- Output directory: `outputs/2026-08-04-rag_naive_v3_full_all_2218_openlux_gpt54mini_v1/`
- Config: `configs/rag_naive_v3_openlux_gpt54mini_embedding3small_stage2_on.yaml`
- Per-domain official summaries: `official_eval/checkpoint_benchmark/<domain>/summary.json`
