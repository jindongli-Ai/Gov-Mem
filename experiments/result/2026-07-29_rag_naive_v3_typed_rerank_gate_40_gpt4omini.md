# Gov-Mem v3 Typed Rerank + Deletion Gate: Paired 40-Checkpoint Result

Date: 2026-07-29

## Experimental setup

The baseline and v3 runs use the same `gatemem_smoke40.json` manifest and the
same 40 checkpoint IDs. This is a paired comparison; the older 40-checkpoint
result from `2026-07-29-rag_naive_official_40_gpt4omini_v2` used different
episodes and is not used as a direct delta baseline.

| Setting | Value |
|---|---|
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Official evaluation provider | Yunwu |
| Embedding model | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive, unchanged |
| Stage 2 | v3 typed router/rerank plus narrow deletion gate |
| Parallelism | 4 domains, up to 8 episodes/domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |

## Domain metrics

`A` is answer-level privacy leakage and `F` is answer-level deletion/staleness
leakage. Domain MGS is `U * (1-A) * (1-F)`.

| Domain | Baseline U | v3 U | Baseline A | v3 A | Baseline F | v3 F | Baseline MGS | v3 MGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 50.00% | 50.00% | 66.67% | 66.67% | 33.33% | 0.00% | 0.1111 | 0.1667 |
| Office | 75.00% | 75.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.7500 | 0.7500 |
| Education | 100.00% | 100.00% | 0.00% | 0.00% | 33.33% | 0.00% | 0.6667 | 1.0000 |
| Household | 75.00% | 75.00% | 33.33% | 33.33% | 33.33% | 0.00% | 0.3333 | 0.5000 |
| Weighted overall | 75.00% | 75.00% | 25.00% | 25.00% | 25.00% | 0.00% | 0.4219 | 0.5625 |

## Stage 2 telemetry

- Typed rerank reordered 1/40 cases. Multi-family queries were deferred to
  preserve cross-slot Utility.
- The deletion gate handled 10/40 explicit historical/deleted scalar-secret
  queries.
- The gate returned closed-set `no_memory` and did not send those retrieved
  records to the answer LLM.
- The existing v2 lexicon and slot aliases provide the routing vocabulary;
  v3 adds no parallel domain vocabulary.

## Conclusion

This incremental version meets the current constraint: Utility did not regress,
and F reached zero on this paired 40-checkpoint run. A did not improve, so the
next change should be a separate, narrow privacy authorization gate for
explicit private requests. Stage 1 retrieval and the current typed rerank
should remain frozen while that next change is evaluated.

Outputs:

- Baseline: `outputs/2026-07-29-rag_naive_baseline_paired_40_gpt4omini`
- v3: `outputs/2026-07-29-rag_naive_v3_typed_rerank_gate_40_gpt4omini`
