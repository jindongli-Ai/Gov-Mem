# Gov-Mem v3 Stage 2 Field-Contract Fix: New 600-Checkpoint Generalization

Date: 2026-07-30

## Experiment settings

| Setting | Value |
|---|---|
| Checkpoints | 600, 150 per domain |
| Manifest | `rag_naive_v3_stage2_generalization_600_seed20260730.json` |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, frozen |
| Stage 2 change | Mixed-field aliases, source-value validation, composite operational credential boundary |
| Parallelism | 4 domain workers, 8 episode workers per domain |
| Key isolation | 30-key pool; one leased key per episode |
| Previous v3 200/40 overlap | 0 |
| Previous long-context 200 overlap | 0 |

`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Results

| Domain | N | Utility U | Privacy A | Forgetting F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Education | 150 | 28.26% | 19.23% | 7.69% | 21.07% | 26.09% | 72.67% |
| Household | 150 | 40.91% | 25.00% | 3.70% | 29.55% | 20.45% | 66.00% |
| Medical | 150 | 59.26% | 44.68% | 10.20% | 29.44% | 11.11% | 73.33% |
| Office | 150 | 48.94% | 8.16% | 1.85% | 44.11% | 36.17% | 74.67% |
| **Overall domain mean** | **600** | | | | **31.04%** | | |

## Comparison with the previous 200

| Evaluation | Education MGS | Household MGS | Medical MGS | Office MGS | Overall MGS |
|---|---:|---:|---:|---:|---:|
| Previous v3 boundary-fix 200 | 26.39% | 47.12% | 53.76% | 63.11% | 47.59% |
| New held-out 600 | 21.07% | 29.55% | 29.44% | 44.11% | 31.04% |

The two evaluations are disjoint, so this is a generalization estimate rather
than a paired before/after delta. The 600 set contains a broader and harder
mix of multi-field utility requests and medical privacy/safety requests; the
47.59% result should therefore not be treated as the stable expected score.

## Stage 2 coverage

| Domain | Long-context ledger | Mixed projection | Policy/safety gate |
|---|---:|---:|---:|
| Education | 7 | 18 | 78 |
| Household | 20 | 26 | 76 |
| Medical | 9 | 6 | 49 |
| Office | 12 | 23 | 105 |
| **Total** | **48** | **73** | **308** |

## Failure analysis

1. Education and Household utility failures are dominated by incomplete
   realization of multi-field contracts. The answer often contains the date
   and one or two scalar fields but omits site, blocker, route, approved-area,
   or safe-wording details that are already present in retrieved evidence.

2. Medical utility failures show the same answer-completeness problem for
   schedules, medication lists, callback plans, and current-vs-canceled
   events. Medical A/F are also much higher on this broader sample, indicating
   that the sensitive boundary needs more precise field-level authorization
   handling rather than a broader global refusal.

3. Office utility failures include incomplete project-status bundles and
   over-refusal for diagnosis/token/coordination-label combinations. The
   current credential boundary is still too coarse for mixed operational
   bundles, while broad customer-safe summaries need a dedicated safe-summary
   path.

4. The local fix correctly covered previously missing query surfaces such as
   `safe broad wording`, `approved amount`, `current room/booth`, and
   `no remaining blocker`, and the regression suite reached 138 passing tests.
   However, the 600 run shows that parser coverage alone is insufficient:
   Stage 2 must next enforce complete field realization after the answer LLM,
   with deterministic source-grounded repair for omitted fields.

## Decision

Do not promote or push this field-contract-fix version as a performance win.
Keep Stage 1 frozen. The next improvement should be a bounded Stage 2C
answer-completeness verifier/repair step for ordinary multi-field utility
queries, followed by a targeted 40-checkpoint regression and then another
held-out generalization evaluation.
