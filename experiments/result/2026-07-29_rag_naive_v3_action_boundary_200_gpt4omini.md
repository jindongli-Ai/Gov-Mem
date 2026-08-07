# Gov-Mem v3 Projection + Action Boundary: 200-Checkpoint Result

Date: 2026-07-29

## Experimental setup

This run uses the existing 200-checkpoint generalization manifest and the same
official evaluation protocol as the previous 200-checkpoint v3 run. Stage 1
RAG-Naive retrieval is unchanged. Stage 2 includes the typed router, narrow
deletion gate, bounded mixed current-state projection, and the new conservative
action boundary.

| Setting | Value |
|---|---|
| Checkpoints | 200, 50 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Official evaluation provider | Yunwu |
| Embedding model | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive, unchanged |
| Stage 2 | Typed rerank + deletion gate + mixed projection + action boundary |
| Parallelism | 4 domains, up to 8 episodes/domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |

## Domain metrics

`U` is effective utility from the official judge. `A` is answer-level privacy
leakage, `F` is answer-level deletion/staleness leakage, `OR` is utility
over-refusal, and `MGS = U * (1-A) * (1-F)`.

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 70.00% | 66.67% | 22.22% | 0.1815 | 25.00% |
| Office | 50 | 36.36% | 66.67% | 0.00% | 0.1212 | 54.55% |
| Education | 50 | 33.33% | 54.55% | 14.29% | 0.1299 | 44.44% |
| Household | 50 | 13.33% | 33.33% | 10.00% | 0.0800 | 53.33% |
| **Weighted overall** | **200** | **40.63%** | **54.72%** | **10.84%** | **0.1640** | **42.19%** |

## Comparison with previous 200-checkpoint v3

| Version | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Previous v3 typed rerank + deletion gate | 40.63% | 52.83% | 20.48% | 0.1524 | 43.75% |
| v3 + mixed projection + action boundary | 40.63% | 54.72% | 10.84% | 0.1640 | 42.19% |

The run improves F by 9.64 percentage points and MGS by 0.0116, while pooled
U is unchanged. A is 1.89 points worse, so this is not yet a clean release:
the safety gain came with a privacy regression and needs further diagnosis.
The domain picture is mixed: Education U rose by 5.55 points, Office F fell
to zero, and Household U stayed at 13.33%.

## Stage 2 telemetry

| Route | Cases | Projection applied | Action boundary applied |
|---|---:|---:|---:|
| `access_policy` | 96 | 0 | 0 |
| `mixed` | 31 | 9 | 1 |
| `semantic_state` | 41 | 0 | 0 |
| `typed_scalar` | 32 | 0 | 0 |
| **Total** | **200** | **9** | **1** |

The single action-boundary activation was the
explicitly scoped Saffron Supper request: “the current setup state I am
allowed to use.” Exact credential/token/badge/PIN/private-exact requests did
not activate the boundary.

## Decision

Keep the current changes as an experimental branch, but do not call it a final
release: pooled U did not improve and A regressed. The bounded projection is
still useful because it reduced stale/deleted answer leakage. A guarded
query-contract probe was then run on 40 checkpoints; the extra LLM call
reduced U from 81.25% to 75.0%, so it was rolled back from the active pipeline
and recorded separately. No retrieval or embedding change is justified by
this result.

Output:

- `outputs/2026-07-29-rag_naive_v3_action_boundary_200_gpt4omini`
