# GateMem Cross-Domain Partial Evaluation

Timestamp: 2026-07-08 02:15:37

## Evaluation Setting

- Suite manifest:
  `experiments/gatemem_suites/gatemem_smoke40.json`
- Result directory:
  `outputs/gatemem_smoke40_20260708_0146_structured_v6`
- Config:
  `configs/govmem_gpt5_nano.yaml`
- Experiment mode:
  `rag_policy_amem`
- Runtime API:
  `yunwu`
- Memory-system base LLM:
  `provider=openai_compatible via yunwu; model=gpt-5.4-nano-2026-03-17`
- Official scorer:
  `third_party/GateMem-official/bench/scripts/score_predictions.py`
- LLM judge:
  `False`

Balanced subset composition:

- 4 domains
- 10 checkpoints per domain
- total:
  40 checkpoints

## Main Table

The following table follows the GateMem paper's main reporting semantics:

- `U` = Utility
- `A` = Access-Control Violation Rate
- `F` = Active-Forgetting Failure Rate
- `MGS = U * (1 - A) * (1 - F)`

| Domain | Checkpoints | U | A | F | MGS | Action Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Office | 10 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Education | 10 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Household | 10 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Overall (balanced subset) | 40 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |

## Supporting Regressions

- Current 4-case failure regression:
  `outputs/v4_failure_regression_v1_20260708_0001`
- Result:
  `4/4` passed
- Safety refusal regression:
  `outputs/smoke40_safety_refusal_regression_v1_20260708_0140`
- Result:
  official action `5/5` corrected to `no_memory`

## Source Files

- smoke40 suite summary:
  `outputs/gatemem_smoke40_20260708_0146_structured_v6/suite_summary.json`
- medical summary:
  `outputs/gatemem_smoke40_20260708_0146_structured_v6/medical/official_eval/gatemem/medical/summary.json`
- office summary:
  `outputs/gatemem_smoke40_20260708_0146_structured_v6/office/official_eval/gatemem/office/summary.json`
- education summary:
  `outputs/gatemem_smoke40_20260708_0146_structured_v6/education/official_eval/gatemem/education/summary.json`
- household summary:
  `outputs/gatemem_smoke40_20260708_0146_structured_v6/household/official_eval/gatemem/household/summary.json`

## Short Interpretation

- This round reaches a fully clean official-compatible `smoke40` result:
  all 40 checkpoints are correct under the official GateMem scoring pipeline.
- The gain is framework-level rather than case-level.
  The two effective abstractions in this round are:
  `coverage-constrained evidence organization` and `regime-aware action normalization`.
- Utility is now stable across all four domains, and the earlier cross-domain privacy failures are removed without introducing official leakage.
- The temporary regression introduced during iteration was also informative:
  a privacy-overreach refusal rule was initially too broad and incorrectly absorbed some safety reconstruction queries.
  After restricting that rule to non-safety regimes, the official safety actions returned to `no_memory`.

## Framework Change in This Round

1. Added coverage-constrained regimen selection.
   Medication answering no longer relies on generic high-score replay alone.
   The runtime now prioritizes requested regimen families such as `stop / use / start`, and can attach a current baseline continuation line when the query asks for the present regimen state.

2. Reduced over-triggered continuation requirements.
   A latent `continue_regimen_request` signal is no longer treated as mandatory when the user explicitly asks action-specific questions such as what to `use`, `start`, or `stop`.

3. Added regime-aware privacy and safety separation.
   Exact sensitive current-state overreach by non-owners is normalized to `refuse`, but deleted-memory reconstruction safety queries remain `no_memory` rather than being collapsed into privacy refusal.

4. Tightened typed-state rescue boundaries.
   Typed-state completion is prevented from rescuing non-owner exact sensitive current-state requests, which removes privacy leakage on education and household cases while preserving owner and authorized-staff utility answers.

## Methodological Note

- The local aggregate printed during runtime uses a safety-oriented `accuracy` notion that is not the same as the official GateMem action outcome.
- For result reporting, the authoritative files are the official per-domain `scores.jsonl`, per-domain `summary.json`, and the top-level `suite_summary.json`.
