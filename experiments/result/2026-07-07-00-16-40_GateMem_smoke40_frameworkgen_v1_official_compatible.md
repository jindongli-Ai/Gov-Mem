# GateMem Balanced Partial Evaluation

Timestamp: 2026-07-07 00:16:40

## Evaluation Setting

- Suite manifest:
  `experiments/gatemem_suites/gatemem_smoke40.json`
- Result directory:
  `outputs/gatemem_smoke40_20260706_234805_frameworkgen_v1`
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
| Office | 10 | 0.75 | 0.00 | 0.00 | 0.75 | 1.00 |
| Education | 10 | 0.25 | 0.00 | 0.00 | 0.25 | 1.00 |
| Household | 10 | 0.75 | 0.00 | 0.00 | 0.75 | 0.90 |
| Overall (balanced subset) | 40 | 0.6875 | 0.00 | 0.00 | 0.6875 | 0.9750 |

## Source Files

- suite summary:
  `outputs/gatemem_smoke40_20260706_234805_frameworkgen_v1/suite_summary.json`
- medical summary:
  `outputs/gatemem_smoke40_20260706_234805_frameworkgen_v1/medical/official_eval/gatemem/medical/summary.json`
- office summary:
  `outputs/gatemem_smoke40_20260706_234805_frameworkgen_v1/office/official_eval/gatemem/office/summary.json`
- education summary:
  `outputs/gatemem_smoke40_20260706_234805_frameworkgen_v1/education/official_eval/gatemem/education/summary.json`
- household summary:
  `outputs/gatemem_smoke40_20260706_234805_frameworkgen_v1/household/official_eval/gatemem/household/summary.json`

## Short Interpretation

- The latest framework changes substantially improved medical utility while preserving zero privacy leakage on this balanced subset.
- The dominant remaining weakness is no longer privacy control. It is current-state utility generalization, especially in Education.
- Office and Household remain moderate:
  - both keep `A = 0.00`
  - both still lose utility on a limited number of current-state or access-governed utility cases

## Most Important Failure Patterns

1. Education current-state bundle selection still confuses stale, provisional, and truly current values.
2. Office current-state selection can still surface outdated target-date or discount values when multiple revisions exist.
3. Household still has at least one action-side failure despite strong privacy control, so action calibration is not uniformly solved across domains.

## Terminology Note

- This run is a cross-domain balanced partial evaluation on a 40-checkpoint subset.
- It is useful for framework regression tracking and error discovery.
- It is not a full-benchmark headline result.
