# Gov-Mem v3 Long-Context Field Ledger: 200-Checkpoint Validation

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 200, 50 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive turn retrieval, unchanged |
| Stage 2 | Existing typed rerank, deletion gate, sensitive gate, mixed projection, bounded long-context field ledger |
| Parallelism | 4 domains, 5 episode workers per domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |
| Suite | `experiments/gatemem_suites/stateful_policy_generalization_200_seed20260727.json` |

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Domain Metrics

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 80.00% | 58.33% | 27.78% | 0.2407 | 15.00% |
| Office | 50 | 45.45% | 46.67% | 0.00% | 0.2424 | 36.36% |
| Education | 50 | 38.89% | 45.45% | 14.29% | 0.1818 | 44.44% |
| Household | 50 | 13.33% | 26.67% | 10.00% | 0.0880 | 26.67% |
| **Overall domain mean** | **200** | **44.42%** | **44.28%** | **13.01%** | **0.1882** | **30.62%** |

## Comparison With Previous 200-Checkpoint Run

| Version | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Previous field-contract v3 | 35.57% | 40.53% | 11.83% | 0.1572 | 41.74% |
| Long-context ledger v2 | 44.42% | 44.28% | 13.01% | 0.1882 | 30.62% |
| Change | **+8.85 pp** | **+3.75 pp** | **+1.18 pp** | **+0.0310** | **-11.12 pp** |

The long-context candidate improves aggregate Utility and MGS, especially in
Office and Household, but it regresses both safety metrics. Since A and F are
core release constraints, this is not a promotion-quality result.

## Long-Context Audit

- Applied to only 13/200 cases.
- 91 cases were not supported multi-field mixed queries.
- 64 historical/deleted queries, 20 explicit sensitive queries, and one
  explicit privacy/confidentiality query were excluded.
- Five malformed/non-verifiable quote results and several incomplete ledgers
  correctly fell back.
- All A/F leakage cases had `long_context_applied=false`.
- Privacy context leakage and deletion context leakage were both `0/200`.

This means long-context itself did not put prohibited evidence into the answer
context in this run. The observed safety regression is in the pre-existing
answer delivery path: privacy/deletion queries can still reach the answer LLM
with insufficiently strong action/gate handling. The long-context branch does
not solve that problem and should not be used to mask it.

## Decision

Reject this as the active default candidate and roll back the configuration
flag. `stage2.long_context_field_ledger.enabled` is now `false` in
`configs/rag_naive_v3_gpt4omini.yaml`; the implementation remains available
for later controlled experiments. Stage 1 remains unchanged.

The next improvement should target the shared answer-level privacy/deletion
boundary, not more long-context recall. It should first fix action/gate
handling on the concrete leaked cases, then repeat a small regression before
reconsidering long-context.

## Artifacts

- Suite summary: `outputs/2026-07-29-rag_naive_v3_long_context_ledger_v2_200_gpt4omini_retry30keys/suite_summary.json`
- Official domain summaries: `outputs/2026-07-29-rag_naive_v3_long_context_ledger_v2_200_gpt4omini_retry30keys/*/official_eval/checkpoint_benchmark/*/summary.json`
- Per-case Stage 2 audit: `outputs/2026-07-29-rag_naive_v3_long_context_ledger_v2_200_gpt4omini_retry30keys/*/episodes/*/debug_cases/checkpoint_benchmark/*.json`
