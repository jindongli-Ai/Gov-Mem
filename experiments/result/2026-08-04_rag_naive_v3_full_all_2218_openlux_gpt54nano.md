# Gov-Mem v3 Full Evaluation with GPT-5.4-nano: 2218 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-5.4-nano`
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
| Medical | 579 | 78.10% | 45.83% | 11.86% | 37.28% | 72.02% | 4.76% | 0.00% | 0.00% |
| Office | 547 | 59.09% | 3.51% | 1.80% | 55.99% | 82.08% | 18.18% | 0.00% | 0.00% |
| Education | 540 | 38.33% | 17.22% | 7.78% | 29.26% | 73.89% | 13.89% | 0.00% | 0.00% |
| Household | 552 | 39.67% | 22.83% | 1.09% | 30.29% | 69.93% | 16.85% | 0.00% | 0.00% |
| **Four-domain average** | **2218** | | | | **38.21%** | | | | |

## Comparison with GPT-4o-mini

The comparison uses the same 2218-checkpoint manifest, the same frozen framework, the same embedding model, and the same official `gpt-4o` evaluator.

| Domain | MGS with GPT-4o-mini | MGS with GPT-5.4-nano | Delta |
|---|---:|---:|---:|
| Medical | 37.18% | 37.28% | +0.10 pp |
| Office | 53.32% | 55.99% | +2.67 pp |
| Education | 31.31% | 29.26% | -2.05 pp |
| Household | 34.32% | 30.29% | -4.03 pp |
| **Four-domain average** | **39.03%** | **38.21%** | **-0.82 pp** |

## Interpretation

- Replacing the memory-system base LLM with GPT-5.4-nano did not improve overall full-population MGS: `38.21%` versus `39.03%` for GPT-4o-mini.
- GPT-5.4-nano improved Office and slightly improved Medical MGS, but degraded Education and Household, especially Household.
- The main GPT-5.4-nano tradeoff is not retrieval: Stage 1 and embedding were unchanged, and context-level privacy/deletion exposure remained 0% in every domain. The degradation is in final answer utility and answer-level leakage.
- No framework code was changed for this comparison; only the memory-system base model changed.

## Artifacts

- Output directory: `outputs/2026-08-04-rag_naive_v3_full_all_2218_openlux_gpt54nano_v1/`
- Config: `configs/rag_naive_v3_openlux_gpt54nano_embedding3small_stage2_on.yaml`
- Per-domain official summaries: `official_eval/checkpoint_benchmark/<domain>/summary.json`
