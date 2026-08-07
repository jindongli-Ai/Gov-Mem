# Gov-Mem v3 Full Evaluation: 2218 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: current frozen Gov-Mem v3 typed rerank and answer-redacted confirmation boundary
- Full manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Execution: all 2218 available checkpoints; four domains and episode shards ran in parallel with isolated OpenLux keys

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the arithmetic mean of the four domain MGS values. `A` is answer-level privacy leakage and `F` is answer-level deletion/staleness leakage. Action and OR are reported for completeness and are not multiplied into MGS.

## Full Result

| Domain | Checkpoints | U | A | F | MGS | Action | OR | Privacy context | Deletion context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 66.67% | 39.06% | 8.47% | 37.18% | 72.71% | 12.38% | 0.00% | 0.00% |
| Office | 547 | 55.84% | 4.09% | 0.45% | 53.32% | 81.17% | 22.73% | 0.00% | 0.00% |
| Education | 540 | 42.22% | 20.56% | 6.67% | 31.31% | 74.07% | 16.11% | 0.00% | 0.00% |
| Household | 552 | 45.11% | 23.91% | 0.00% | 34.32% | 72.28% | 14.67% | 0.00% | 0.00% |
| **Four-domain average** | **2218** | | | | **39.03%** | | | | |

## Interpretation

- This is the full available checkpoint population, not a held-out sample or a balanced subset.
- The full result is `39.03%` average MGS, below the 800-checkpoint combined result of `41.20%`; the smaller result was optimistic by 2.17 percentage points.
- Medical is limited primarily by answer-level privacy/deletion leakage. Education has the lowest utility and remains the clearest utility bottleneck.
- Office has the strongest MGS and lowest leakage, while Household has no deletion leakage but substantial privacy leakage.
- Privacy and deletion exposure in retrieved context remained 0% in all domains. The remaining loss is concentrated in final answer utility and answer-level leakage.
- No framework code was changed for this full evaluation.

## Artifacts

- Output directory: `outputs/2026-08-03-rag_naive_v3_full_all_2218_openlux_gpt4omini_v1/`
- Per-domain official summaries: `official_eval/checkpoint_benchmark/<domain>/summary.json`
