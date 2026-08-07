# Gov-Mem v3 Mixed Projection + Action Boundary: 40-Checkpoint Result

Date: 2026-07-29

## Experimental setup

This is a paired 40-checkpoint smoke evaluation using the same
`gatemem_smoke40.json` manifest as the previous v3 run. Stage 1 remains the
official-compatible RAG-Naive retrieval path. The change under test is limited
to Stage 2 mixed-query evidence projection and a conservative action boundary.

| Setting | Value |
|---|---|
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Official evaluation provider | Yunwu |
| Embedding model | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive, unchanged |
| Stage 2 | Typed rerank, narrow deletion gate, mixed current-state projection |
| Action boundary | Only complete explicitly scoped safe projections; exact credential-like requests excluded |
| Parallelism | 4 domains, up to 8 episodes/domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |

## Domain metrics

`U` is effective utility from the official judge. `A` is answer-level privacy
leakage, `F` is answer-level deletion/staleness leakage, `OR` is utility
over-refusal, and `MGS = U * (1-A) * (1-F)`.

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.0% | 66.7% | 0.0% | 0.1667 | 25.0% |
| Office | 10 | 75.0% | 0.0% | 0.0% | 0.7500 | 0.0% |
| Education | 10 | 100.0% | 0.0% | 0.0% | 1.0000 | 0.0% |
| Household | 10 | 100.0% | 33.3% | 0.0% | 0.6667 | 0.0% |
| **Weighted overall** | **40** | **81.25%** | **25.0%** | **0.0%** | **0.6094** | **6.25%** |

## Paired comparison

| Version | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Previous v3 typed rerank + deletion gate | 75.0% | 25.0% | 0.0% | 0.5625 | 6.25% |
| v3 + mixed projection | 81.25% | 25.0% | 0.0% | 0.6094 | 6.25% |
| v3 + projection + action boundary | 81.25% | 25.0% | 0.0% | 0.6094 | 6.25% |

The projection run changed the final action in one safety case from `refuse`
to the more precise `no_memory`; it did not increase F. The action boundary
itself triggered zero times in this 40-case sample, so no utility improvement
should be attributed to that rule yet. The 15 changed answer texts were mostly
ordering or wording changes caused by the projected context.

## Stage 2 telemetry

| Route | Cases | Projection applied |
|---|---:|---:|
| `mixed` | 11 | 7 |
| `typed_scalar` | 7 | 0 |
| `semantic_state` | 6 | 0 |
| `access_policy` | 16 | 0 |

Projection is bounded to at most 12 rows and falls back to the complete Stage
1 result if every requested slot cannot be represented. It annotates relevance
only; it does not grant authorization. Historical/deleted queries bypass it
and remain covered by the deletion gate.

## Decision

Keep Stage 1 frozen and retain mixed projection. The result is a positive but
small generalization signal: Household improved without A/F regression, while
Medical remained unchanged. The action boundary has not yet earned promotion
because it did not execute on this sample. The next validation should run the
same current code on the existing 200-checkpoint generalization manifest and
measure how often explicit safe-scope cases are normalized, especially in
Education and Household. Only after that should the query-contract LLM branch
be introduced.

Output:

- `outputs/2026-07-29-rag_naive_v3_action_boundary_40_gpt4omini`
