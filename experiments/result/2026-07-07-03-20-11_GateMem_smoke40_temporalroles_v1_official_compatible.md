# GateMem Balanced Partial Evaluation

Timestamp: 2026-07-07 03:20:11

## Evaluation Setting

- Suite manifest:
  `experiments/gatemem_suites/gatemem_smoke40.json`
- Result directory:
  `outputs/gatemem_smoke40_20260707_025813_temporalroles_v1`
- Config:
  `configs/govmem_gpt5_nano.yaml`
- Experiment mode:
  `rag_policy_amem`
- Runtime API:
  `yunwu`
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

The following table follows the GateMem paper's reporting semantics:

- `U` = Utility
- `A` = Access-Control Violation Rate
- `F` = Active-Forgetting Failure Rate
- `MGS = U * (1 - A) * (1 - F)`

| Domain | Checkpoints | U | A | F | MGS | Action Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| Office | 10 | 0.50 | 0.00 | 0.00 | 0.50 | 1.00 |
| Education | 10 | 1.00 | 0.00 | 0.00 | 1.00 | 0.90 |
| Household | 10 | 0.75 | 0.00 | 0.00 | 0.75 | 0.90 |
| Overall (balanced subset) | 40 | 0.8125 | 0.00 | 0.00 | 0.8125 | 0.95 |

## Focused Regression Update

- Current-state focused regression:
  `outputs/current_state_bundle_regression_v15_20260707_025425_temporalroles_publicsplit`
- Focused result:
  Office `1/1`, Education `3/3`, total `4/4`
- Focused paper metrics:
  Office `MGS = 1.0`, Education `MGS = 1.0`

## Source Files

- smoke40 suite summary:
  `outputs/gatemem_smoke40_20260707_025813_temporalroles_v1/suite_summary.json`
- focused current-state suite summary:
  `outputs/current_state_bundle_regression_v15_20260707_025425_temporalroles_publicsplit/suite_summary.json`
- medical summary:
  `outputs/gatemem_smoke40_20260707_025813_temporalroles_v1/medical/official_eval/gatemem/medical/summary.json`
- office summary:
  `outputs/gatemem_smoke40_20260707_025813_temporalroles_v1/office/official_eval/gatemem/office/summary.json`
- education summary:
  `outputs/gatemem_smoke40_20260707_025813_temporalroles_v1/education/official_eval/gatemem/education/summary.json`
- household summary:
  `outputs/gatemem_smoke40_20260707_025813_temporalroles_v1/household/official_eval/gatemem/household/summary.json`

## Short Interpretation

- This round confirms a real framework-level gain on Education current-state utility, not a case-level patch.
- The key abstraction added in this round is temporal-role separation:
  access-artifact validity dates are separated from private case-state dates, and public-shareable event dates are separated from private case-state dates.
- The gain generalizes clearly on Education:
  the 10-case balanced subset rises from the earlier `U = 0.25` baseline run to `U = 1.00` in this round.
- Medical remains stable at `U = 1.00`.
- Office and Household remain the main bottlenecks for the next iteration.

## Framework Change in This Round

1. Added a generic `access artifact temporal` distinction.
   Dates from credentials, access windows, tokens, and validity statements no longer map to `target_date`.

2. Added a generic `public event date` distinction.
   Publicly shareable event dates no longer override private case-state dates in current-state answering.

3. Propagated the distinction through multiple runtime layers.
   The new temporal-role signal is consumed by frame extraction, metadata construction, family classification, current-state scoring, and utility-record construction.

## Next Problems Exposed

1. Office still underperforms on utility (`U = 0.50`).
   The remaining issue is likely not the same temporal-role confusion already fixed for Education. It more likely involves multi-revision current-state arbitration among budget, discount, and project-state lines.

2. Household utility is still moderate (`U = 0.75`).
   Current-state household-plan slot completion is not yet uniformly robust on the broader subset.

3. Education action accuracy is `0.90`.
   Utility is now full on this subset, but one action-side decision remains incorrect.

## Terminology Note

- This run is a cross-domain balanced partial evaluation on a 40-checkpoint subset.
- It is suitable for framework regression tracking and cross-domain error discovery.
- It is not a full-benchmark headline result.
