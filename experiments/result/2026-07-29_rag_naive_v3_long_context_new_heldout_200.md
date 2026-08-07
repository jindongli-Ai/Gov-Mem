# Gov-Mem v3 Long-Context: New Held-Out 200-Checkpoint Comparison

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 200 new held-out checkpoints, 50 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu for all LLM and embedding calls |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive turn retrieval, unchanged |
| Stage 2 OFF | Existing typed rerank with long-context field ledger disabled |
| Stage 2 ON | Same pipeline with bounded long-context field ledger enabled |
| Parallelism | 4 domains, 5 episode workers per domain |
| Yunwu key isolation | 15 keys per arm, one leased key per episode; 30-key pool used across both arms |
| Manifest | `experiments/gatemem_suites/rag_naive_v3_long_context_heldout_200_seed20260729.json` |

The two arms used the exact same held-out manifest. `U` is official effective
utility accuracy, `A` is answer-level privacy leakage, `F` is answer-level
deletion/staleness leakage, `OR` is over-refusal, and
`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values, as required by the evaluation protocol.

## Domain Metrics

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Medical OFF | 50 | 71.43% | 63.16% | 17.65% | 0.2167 | 28.57% |
| Office OFF | 50 | 66.67% | 28.57% | 0.00% | 0.4762 | 22.22% |
| Education OFF | 50 | 9.09% | 37.50% | 8.33% | 0.0521 | 63.64% |
| Household OFF | 50 | 50.00% | 28.57% | 6.25% | 0.3348 | 25.00% |
| **Overall domain-mean OFF** | **200** | | | | **0.2700** | |
| Medical ON | 50 | 71.43% | 63.16% | 23.53% | 0.2012 | 28.57% |
| Office ON | 50 | 61.11% | 28.57% | 0.00% | 0.4365 | 22.22% |
| Education ON | 50 | 18.18% | 37.50% | 8.33% | 0.1042 | 54.55% |
| Household ON | 50 | 50.00% | 28.57% | 0.00% | 0.3571 | 20.00% |
| **Overall domain-mean ON** | **200** | | | | **0.2748** | |

Long-context ON versus OFF:

| Metric | OFF | ON | Delta |
|---|---:|---:|---:|
| Overall domain-mean MGS | 0.2700 | 0.2748 | +0.0048 |

The +0.0048 MGS change is small. A did not improve in any domain. F was
unchanged in aggregate (5 leaked checkpoints in each arm), while its domain
distribution changed: Medical increased from 3/17 to 4/17 and Household
decreased from 1/16 to 0/16. Office and Education were unchanged. Because the
two real-LLM runs make independent model calls, this small aggregate change is
not sufficient evidence that long-context caused a general improvement.

## Long-Context Audit

- OFF applied long-context to `0/200` cases, as configured.
- ON applied long-context to `18/200` cases.
- All ON applications were on the existing `mixed` route: Office `5`,
  Education `8`, Household `5`, Medical `0`.
- No ON case with `long_context_applied=true` had an official A or F leak.
- ON had 26 privacy leaks and 5 deletion leaks overall; OFF had the same
  aggregate counts.
- The held-out manifest has no overlap with the previously excluded 40-case
  smoke suite or the earlier 200-case stateful-policy suite.

The branch is therefore safety-contained in this run, but its coverage is too
narrow to address the broader Education and Household utility problem. The
large Education U increase is encouraging, but it is accompanied by a small
Office U decrease and does not establish a stable cross-domain gain from only
18 activations.

## Decision

Do not promote long-context ON as the default version yet. Keep the default
configuration with `stage2.long_context_field_ledger.enabled: false` and keep
the implementation available for the next controlled experiment. Stage 1
remains frozen.

The next change should focus on the shared Stage 2 answer-level privacy and
deletion gate, because A/F did not improve and the long-context branch did not
participate in any leaked case. After that gate change is validated on a small
regression set, re-test long-context on a targeted but previously unseen set of
mixed semantic queries to increase activation coverage without changing Stage 1.

## Artifacts

- OFF summary: `outputs/2026-07-29-rag_naive_v3_long_context_heldout_off_200/suite_summary.json`
- ON summary: `outputs/2026-07-29-rag_naive_v3_long_context_heldout_on_200/suite_summary.json`
- OFF config: `configs/rag_naive_v3_gpt4omini.yaml`
- ON config: `configs/rag_naive_v3_gpt4omini_long_context_enabled.yaml`
- Per-case audit: each output's `*/episodes/*/debug_cases/*/*.json`
