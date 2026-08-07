# Gov-Mem v3 Stage 2 Field Contract and Carrier Validation

Date: 2026-07-30

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 40, 10 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu for memory, embedding, and official judge calls |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, frozen |
| Stage 2 | Generic field aliases, bounded safe-summary contract, chronology guard, verified carriers first |
| Parallelism | 4 domain workers, 5 episode workers per domain |
| Yunwu key isolation | 30-key pool; one leased key per episode |
| Manifest | `rag_naive_v3_reasoning_rerank_heldout_40_seed20260730.json` |

`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Domain Metrics

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Education | 10 | 33.33% | 0.00% | 0.00% | 0.3333 | 0.00% |
| Household | 10 | 0.00% | 25.00% | 0.00% | 0.0000 | 0.00% |
| Medical | 10 | 50.00% | 33.33% | 0.00% | 0.3333 | 50.00% |
| Office | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% |
| **Overall domain mean** | **40** | | | | **0.4167** | |

## Findings

- Generic routing now recognizes Beacon room/access-code/expiry, Household watering window/route/areas/rain process, showcase date/room, and outward-safe state without entity-specific rules.
- Ivy Pots entered the mixed route and long-context ledger; the verified answer context preserved all four requested fields in the final v16 trace.
- Stage 1 retrieval was unchanged.
- The run did not improve the 40-case domain-mean MGS over the prior `0.4167`; Household remains the dominant utility gap.
- Long-context was applied to only `2/40` cases in this run, so it is not ready for default promotion. Household answer-level privacy leakage was `25%` in this run and requires regression analysis before any larger experiment.

## Artifacts

- Suite summary: `outputs/2026-07-30-rag_naive_v3_long_context_heldout_on_40_v16/suite_summary.json`
- Per-case traces: `outputs/2026-07-30-rag_naive_v3_long_context_heldout_on_40_v16/*/episodes/*/debug_cases/`
- Source: `src/gov_mem/query_semantics.py`, `src/gov_mem/backbones/stage2_typed_rerank.py`
