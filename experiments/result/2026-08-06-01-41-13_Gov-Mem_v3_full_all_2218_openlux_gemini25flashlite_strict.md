# Gov-Mem v3 Full Evaluation with Gemini-2.5-Flash-Lite: 2218 Checkpoints

Protocol label: `rag_naive_v3_typed_rerank` frozen typed-rerank track; not
`govmem_symbolic`.

This report records the full evaluation of the current Gov-Mem framework. It
is a Gov-Mem result, not a RAG-Naive baseline result.

## Evaluation Protocol

| Item | Setting |
|---|---|
| Checkpoints | 2218 total: Medical 579, Office 547, Education 540, Household 552 |
| Memory-system provider / base LLM | OpenLux / `gemini-2.5-flash-lite` |
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
| Parallel execution | 4 domains, up to 30 episode shards concurrently with isolated keys |
| Total wall time | Approximately 21 minutes 21 seconds, including official-score recovery |

The memory-system base model is the only model changed relative to the
paper-compatible Gov-Mem v3 comparison. Stage 1, Stage 2, embedding model,
dataset manifest, and official evaluator remain fixed.

The prediction stage used the local OpenLux key pool. During the initial
official evaluation, one key returned `503 No available channel` for `gpt-4o`;
the scoring stage was resumed with the remaining available keys. This affected
runtime only, not prediction content or evaluation protocol.

## Official Paper Metrics

`MGS = U * (1 - A) * (1 - F)`. The reported overall MGS is the arithmetic
mean of the four domain MGS values. `A` and `F` are the official answer-level
privacy and deletion/staleness leakage rates. Action accuracy and over-refusal
are reported for completeness and are not multiplied into MGS.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 71.90% | 27.08% | 6.21% | 49.17% | 79.27% | 10.48% |
| Office | 547 | 68.18% | 4.68% | 1.80% | 63.82% | 84.10% | 11.69% |
| Education | 540 | 38.33% | 14.44% | 7.22% | 30.43% | 75.37% | 10.56% |
| Household | 552 | 46.74% | 17.93% | 2.72% | 37.31% | 71.20% | 13.04% |
| **Four-domain average** | **2218** | | | | **45.18%** | | |

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
official 45.18% MGS should not be interpreted as zero context leakage.

## Artifacts

- Full suite summary: `outputs/2026-08-06-01-19-22_Gov-Mem_v3_full_all_2218_openlux_gemini25flashlite_v1/suite_summary.json`
- Per-domain official summaries: `outputs/2026-08-06-01-19-22_Gov-Mem_v3_full_all_2218_openlux_gemini25flashlite_v1/<domain>/official_eval/checkpoint_benchmark/<domain>/summary.json`
- Frozen manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Configuration: `configs/rag_naive_v3_openlux_gpt4omini_embedding3small_pure.yaml` with runtime override `--base_model gemini-2.5-flash-lite`
