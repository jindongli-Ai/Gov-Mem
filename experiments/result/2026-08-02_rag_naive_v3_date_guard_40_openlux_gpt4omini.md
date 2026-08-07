# Gov-Mem v3 Date-Guard Recheck

Date: 2026-08-02

## Experiment Settings

| Setting | Value |
|---|---|
| Checkpoints | 40, 10 per domain |
| Memory system provider | OpenLux |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Embedding provider | OpenLux |
| Embedding model | `text-embedding-3-small` |
| Official evaluation provider | OpenLux |
| Official evaluation LLM | `gpt-4o` |
| Stage 1 | RAG-Naive retrieval, unchanged |
| Stage 2 | Current v3 rules, constrained reasoning rerank, verified long-context ledger |
| Change under test | Do not append a second weekday/date when the answer already contains one |
| Parallelism | 4 domains, 8 episode workers per domain |
| Key isolation | 30-key pool; one key leased per episode |
| Manifest | `rag_naive_v3_stage2_generalization_40_seed20260731.json` |

`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Results

| Domain | N | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Education | 10 | 66.67% | 0.00% | 0.00% | 66.67% | 0.00% | 90.00% |
| Household | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 0.00% | 80.00% |
| Medical | 10 | 50.00% | 66.67% | 0.00% | 16.67% | 25.00% | 60.00% |
| Office | 10 | 60.00% | 0.00% | 0.00% | 60.00% | 0.00% | 80.00% |
| **Overall domain mean** | **40** |  |  |  | **44.17%** |  |  |

## Interpretation

The code change was deliberately limited to final answer date completion. It
does not change retrieval, policy gates, candidate reranking, or long-context
selection. This run is a 40-checkpoint validation, not a replacement for the
held-out 200-checkpoint generalization result.

The next improvement should focus on Medical sensitive-field boundary
precision and Household utility recall separately. Stage 1 remains frozen.

Output: `outputs/2026-08-02-rag_naive_v3_date_guard_40_openlux_gpt4omini`
