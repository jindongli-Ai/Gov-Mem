# Gov-Mem v3 Full Evaluation with Llama-3.3-70B-Instruct: 2,218 Checkpoints

Protocol label: `rag_naive_v3_typed_rerank` frozen typed-rerank track; not
`govmem_symbolic`.

This dated report records the real full-benchmark evaluation completed on
2026-08-11. It is a Gov-Mem result, not a RAG-Naive baseline result.

## Evaluation Protocol

| Item | Setting |
|---|---|
| Checkpoints | 2,218 total: Medical 579, Office 547, Education 540, Household 552 |
| Memory-system provider / base LLM | OpenLux / `llama-3.3-70b-instruct` |
| Memory-system temperature / output limit | `0.2` / `4096` |
| Stage 1 retrieval | Frozen GateMem-compatible RAG-Naive turn retrieval, raw query, top-20 |
| Stage 1 embedding | OpenLux `text-embedding-3-small` |
| Stage 2 | Gov-Mem typed constrained rerank over retrieved evidence only |
| Long-context transcript / ledger | Disabled |
| Gold feedback / experience bank | Disabled for the clean benchmark runtime |
| Official evaluator provider / LLM | OpenLux / `gpt-4o` |
| Official evaluator temperature / output limit | `0.0` / `4096` |
| `gate_by_action` | `false`, matching GateMem paper main protocol |
| Prediction completeness | 2,218/2,218; unique checkpoint IDs verified |
| Official judge completeness | 2,218/2,218; parse failure rate 0% |
| Context-audit coverage | 2,218/2,218 (100%) |
| Parallel execution | 4 domains, up to 30 episode shards concurrently, isolated OpenLux keys |
| Key-pool execution | 30-key pool for memory system, embeddings, and official evaluation |

The memory-system base model is the only model changed relative to the
paper-compatible Gov-Mem v3 comparison. Stage 1, Stage 2, embedding model,
dataset manifest, and official evaluator remain fixed. Prediction used strict
resume after transient OpenLux API failures; completed checkpoint outputs were
preserved and only incomplete checkpoints were reprocessed.

## Official Paper Metrics

`MGS = U * (1 - A) * (1 - F)`. The reported overall MGS is the arithmetic
mean of the four domain MGS values. `A` and `F` are the official answer-level
privacy and deletion/staleness leakage rates. Action accuracy and
over-refusal are reported for completeness and are not multiplied into MGS.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 39.05% | 18.75% | 12.43% | 27.78% | 58.55% | 43.81% |
| Office | 547 | 36.36% | 1.75% | 2.25% | 34.92% | 72.03% | 51.95% |
| Education | 540 | 17.22% | 11.11% | 3.89% | 14.71% | 56.48% | 56.67% |
| Household | 552 | 23.91% | 12.50% | 1.09% | 20.70% | 55.07% | 50.00% |
| **Four-domain average** | **2,218** | | | | **24.53%** | | |

## Independent Context Audit

| Domain | Privacy context exposure | Deletion context exposure |
|---|---:|---:|
| Medical | 63.02% | 26.55% |
| Office | 4.68% | 3.60% |
| Education | 22.22% | 13.33% |
| Household | 45.65% | 7.07% |

The context audit is complete, but the non-zero exposure rates show that the
official 24.53% MGS should not be interpreted as zero context leakage.

## Artifacts

- Full suite summary: `outputs/2026-08-11-rag_naive_v3_full_all_2218_openlux_llama33_70b_instruct_v1/suite_summary.json`
- Per-domain official summaries: `outputs/2026-08-11-rag_naive_v3_full_all_2218_openlux_llama33_70b_instruct_v1/<domain>/official_eval/checkpoint_benchmark/<domain>/summary.json`
- Frozen manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Configuration: `configs/rag_naive_v3_openlux_llama33_70b_instruct_embedding3small_pure.yaml`
