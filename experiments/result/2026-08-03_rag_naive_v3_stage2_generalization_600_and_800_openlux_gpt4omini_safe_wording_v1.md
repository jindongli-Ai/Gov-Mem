# Gov-Mem v3 Frozen-Version Generalization: 600 and Combined 800 Checkpoints

## Summary

The current Gov-Mem framework was frozen before this experiment. No framework code, Stage 1 retrieval, Stage 2 logic, configuration, or model setting was changed between the earlier 200-case run and this 600-case run.

The new 600 checkpoints contain 150 checkpoints per domain and have zero checkpoint overlap with the earlier 200-case manifest. The 600-case result is reported separately, and the strict 800-case result is computed by combining the 50 earlier and 150 new checkpoints per domain, then rerunning the official scorer on all 200 checkpoints in each domain.

MGS is computed per domain as `U * (1 - A) * (1 - F)`, then the four domain MGS values are averaged.

## 600-Checkpoint Result

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 150 | 59.26% | 38.30% | 8.16% | 33.58% | 75.33% | 12.96% |
| Office | 150 | 53.19% | 12.24% | 1.85% | 45.81% | 77.33% | 27.66% |
| Education | 150 | 41.30% | 21.15% | 7.69% | 30.06% | 76.67% | 10.87% |
| Household | 150 | 40.91% | 21.15% | 3.70% | 31.06% | 66.00% | 25.00% |
| **Four-domain average MGS** | **600** | | | | **35.13%** | | |

## Strict Combined 800-Checkpoint Result

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 200 | 64.38% | 34.43% | 4.55% | 40.30% | 76.50% | 13.70% |
| Office | 200 | 59.65% | 8.96% | 2.63% | 52.88% | 79.00% | 22.81% |
| Education | 200 | 43.75% | 19.44% | 6.25% | 33.04% | 74.50% | 10.94% |
| Household | 200 | 49.12% | 23.68% | 2.99% | 36.37% | 65.00% | 21.05% |
| **Four-domain average MGS** | **800** | | | | **40.65%** | | |

The combined 800 result is not the arithmetic mean of the 200-case and 600-case total scores. It is recomputed from all 200 checkpoints in each domain using the official scorer.

## Configuration

- Memory-system provider: OpenLux
- Memory-system base LLM: `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider: OpenLux
- Stage 1 embedding model: `text-embedding-3-small`
- Official evaluation provider: OpenLux
- Official evaluation LLM: `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: typed reasoning rerank with long-context field ledger and source-bound safe wording
- Memory execution: one isolated API key per episode, four parallel domains, three parallel episode workers per domain
- Official judging: GateMem official scorer, domains judged in parallel with judge concurrency 2

## Interpretation

- The earlier 200-case MGS of 56.04% is a strong but optimistic slice; the larger held-out sample lowers the estimate to 40.65%.
- The lower 600 score is not evidence of a code regression because the exact same frozen framework and configuration produced both runs.
- The dominant generalization weaknesses are utility loss and answer-level privacy leakage, especially Medical and Household. Context-level privacy and deletion leakage remained zero in the combined run.
- The framework should remain frozen while analyzing this 800-case failure distribution. Any future code change must be evaluated as a new version and must not be merged with this frozen 800-case result.

## Artifacts

- 600 manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_600_seed20260730.json`
- 600 outputs: `outputs/2026-08-03-rag_naive_v3_stage2_generalization_600_openlux_gpt4omini_safe_wording_v1/`
- Combined 800 outputs: `outputs/2026-08-03-rag_naive_v3_stage2_generalization_800_combined_openlux_gpt4omini_safe_wording_v1/`
