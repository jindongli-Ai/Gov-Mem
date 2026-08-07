# Gov-Mem v3 Field-Complete Mixed Answer Contract: 40-Checkpoint Result

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 40, 10 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, unchanged |
| Stage 2 | Existing typed rerank, deletion gate, sensitive gate, mixed projection |
| Change under test | Bounded field-completeness cue for complete non-sensitive mixed projections |
| Parallelism | 4 domains, 4 episode workers per domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |
| Suite | `experiments/gatemem_suites/gatemem_smoke40.json` |

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`. Overall MGS below is the arithmetic mean of the
four domain MGS values.

## Domain Metrics

| Domain | N | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 25.00% | 0.00% | 0.00% | 0.2500 | 25.00% |
| Office | 10 | 50.00% | 0.00% | 0.00% | 0.5000 | 0.00% |
| Education | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% |
| Household | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% |
| **Overall domain mean** | **4 domains** | **68.75%** | **0.00%** | **0.00%** | **0.6875** | **6.25%** |

## Paired Runtime Observation

The preceding run used the same manifest and the same configured models, but
without the new answer cue. It produced Medical `U=50%`, Office `U=50%`,
Education `U=75%`, and Household `U=75%`, for a domain-mean MGS of `0.6250`.
The new run produced `0.6875`, with Household and Education at `U=100%`.
Because real Yunwu responses vary between repeated temperature-zero runs, this
is a positive validation signal, not yet a stable generalization claim.

## Stage 2 Observation

- Mixed projection was applied to 9/40 checkpoint queries.
- Seven mixed queries fell back because the retrieved evidence did not cover
  every requested field; the fallback retained the complete Stage 1 result.
- The Cedar Care utility answer retained the specific `side door keypad` and
  the `after 4:20 PM` condition after the answer cue was enabled.
- No A/F regression occurred: all four domains reported `A=0` and `F=0`.

## Decision

Keep this change as the current candidate, but do not promote it to a 200-
checkpoint claim yet. Medical utility fell in this repetition and the two 40-
case runs demonstrate evaluation variance. The next validation should repeat
the fixed 40 suite or use a paired multi-run check before expanding the
pipeline. Stage 1 Retrieval, sensitive permission gating, and the existing
fallback behavior remain unchanged.

## Artifacts

- Suite summary: `outputs/2026-07-29-rag_naive_v3_field_contract_40_gpt4omini/suite_summary.json`
- Predictions and official scores: `outputs/2026-07-29-rag_naive_v3_field_contract_40_gpt4omini/*/official_eval/checkpoint_benchmark/*/`
- Previous field-alias validation run: `outputs/2026-07-29-rag_naive_v3_household_fields_40_gpt4omini/suite_summary.json`
