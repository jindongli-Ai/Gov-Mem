# Gov-Mem v3 Full Evaluation with GPT-5.4-mini: 2218 Checkpoints

Protocol label: `rag_naive_v3_typed_rerank` frozen typed-rerank track; not
`govmem_symbolic`.

This report records the full evaluation of the current Gov-Mem framework. It
is a Gov-Mem result, not a RAG-Naive baseline result.

## Evaluation Protocol

| Item | Setting |
|---|---|
| Checkpoints | 2218 total: Medical 579, Office 547, Education 540, Household 552 |
| Memory-system provider / base LLM | OpenLux / `gpt-5.4-mini` |
| Memory-system temperature / output limit | `0.2` / `4096` |
| Stage 1 retrieval | Frozen GateMem-compatible RAG-Naive turn retrieval, raw query, top-20 |
| Stage 1 embedding | OpenLux `text-embedding-3-small` |
| Stage 2 | Gov-Mem typed constrained rerank over retrieved evidence only |
| Long-context transcript / ledger | Disabled |
| Gold feedback / experience bank | Disabled for the clean benchmark runtime |
| Official evaluator provider / LLM | OpenLux / `gpt-4o` |
| Official evaluator temperature / output limit | `0.0` / `4096` |
| `gate_by_action` | `false`, matching GateMem paper main protocol |
| Prediction completeness | 2218/2218 |
| Official judge completeness | 2218/2218 |
| Context-audit coverage | 2218/2218 (100%) |
| Parallel execution | 4 domains, up to 5 episode shards concurrently with isolated keys |
| Total wall time | Approximately 24 minutes 26 seconds, including recovery of one unavailable scorer key |

The memory-system base model is the only model changed relative to the
paper-compatible Gov-Mem v3 comparison. Stage 1, Stage 2, embedding model,
dataset manifest, and official evaluator remain fixed.

## Official Paper Metrics

`MGS = U * (1 - A) * (1 - F)`. The reported overall MGS is the arithmetic
mean of the four domain MGS values. `A` and `F` are the official answer-level
privacy and deletion/staleness leakage rates. Action accuracy and over-refusal
are reported for completeness and are not multiplied into MGS.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 70.48% | 27.60% | 6.78% | 47.56% | 76.86% | 7.14% |
| Office | 547 | 57.79% | 4.09% | 1.80% | 54.43% | 79.71% | 24.03% |
| Education | 540 | 31.67% | 14.44% | 9.44% | 24.53% | 71.48% | 18.33% |
| Household | 552 | 45.65% | 15.22% | 1.63% | 38.07% | 70.47% | 14.13% |
| **Four-domain average** | **2218** | | | | **41.15%** | | |

## Independent Context Audit

The official answer-level metrics must be distinguished from context exposure.
The runtime prompt audit covered every checkpoint. These context rates are
reported separately and are not substituted for the paper metrics above.

| Domain | Privacy context exposure | Deletion context exposure |
|---|---:|---:|
| Medical | 63.02% | 26.55% |
| Office | 4.68% | 3.60% |
| Education | 22.22% | 13.33% |
| Household | 45.65% | 7.07% |

The context audit is complete, but the non-zero exposure rates show that the
official 41.15% MGS should not be interpreted as zero context leakage.

## Artifacts

- Full suite summary: `outputs/2026-08-05_Gov-Mem_v3_full_all_2218_openlux_gpt54mini_v1/suite_summary.json`
- Per-domain official summaries: `outputs/2026-08-05_Gov-Mem_v3_full_all_2218_openlux_gpt54mini_v1/<domain>/official_eval/checkpoint_benchmark/<domain>/summary.json`
- Frozen manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Configuration: `configs/rag_naive_v3_openlux_gpt4omini_embedding3small_pure.yaml` with runtime override `--base_model gpt-5.4-mini`
