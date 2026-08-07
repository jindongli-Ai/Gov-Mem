# Gov-Mem v3 Long-Context Field Ledger: 40-Checkpoint Validation

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 40, 10 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive turn retrieval, unchanged |
| Stage 2 | Existing typed rerank, deletion gate, sensitive gate, mixed projection, bounded long-context field ledger |
| Long-context branch | 9/40 cases; one structured resolver call only for safe mixed current-state queries |
| Parallelism | 4 domains, 5 episode workers per domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |
| Suite | `experiments/gatemem_suites/gatemem_smoke40.json` |

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Domain Metrics

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 25.00% | 0.00% | 0.00% | 0.2500 | 25.00% |
| Office | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% |
| Education | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% |
| Household | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% |
| **Overall domain mean** | **40** | **81.25%** | **0.00%** | **0.00%** | **0.8125** | **6.25%** |

## Comparison With Previous 40-Checkpoint Candidate

| Version | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Previous field-contract v3 | 68.75% | 0.00% | 0.00% | 0.6875 | 6.25% |
| Long-context ledger v2 | 81.25% | 0.00% | 0.00% | 0.8125 | 6.25% |
| Change | **+12.50 pp** | **0.00 pp** | **0.00 pp** | **+0.1250** | **0.00 pp** |

The paired comparison is directional rather than a deterministic causal claim,
because the memory and judge LLM calls are repeated at temperature zero and
Yunwu responses can still vary. The positive signal is strongest in Office:
the long-context ledger recovered complete current target-date, budget, and
discount fields for three multi-field cases. Education and Household retained
their previous 40-case Utility rather than regressing.

## Long-Context Audit

- Applied to 9/40 cases.
- Excluded historical/deleted queries, explicit sensitive fields, policy or
  authorization queries, and explicit privacy/confidentiality cues.
- Every applied field had a real visible `message_id`, an exact source quote,
  and an allowed current-state status.
- Malformed, incomplete, stale-marked, or non-source-verifiable ledgers fell
  back to the existing Stage 2 evidence.
- Privacy answer leakage, deletion answer leakage, privacy context leakage,
  and deletion context leakage were all `0/40`.
- Stage 1 retrieval and its top-20 candidate set were not modified.

## Decision

Keep the long-context ledger as the current v3 candidate with the feature flag
enabled. Do not push this as a promoted GitHub release yet: the validation is
only 40 checkpoints and the Medical domain remains weak. The next validation
should use the existing 200-checkpoint manifest, with the same base/judge LLM
settings and per-episode key isolation. If the 200-case U regresses or any A/F
metric worsens, disable `stage2.long_context_field_ledger.enabled` and retain
the previous Stage 2 behavior.

## Artifacts

- Suite summary: `outputs/2026-07-29-rag_naive_v3_long_context_ledger_v2_40_gpt4omini/suite_summary.json`
- Official domain summaries: `outputs/2026-07-29-rag_naive_v3_long_context_ledger_v2_40_gpt4omini/*/official_eval/checkpoint_benchmark/*/summary.json`
- Per-case Stage 2 audit: `outputs/2026-07-29-rag_naive_v3_long_context_ledger_v2_40_gpt4omini/*/episodes/*/debug_cases/checkpoint_benchmark/*.json`
