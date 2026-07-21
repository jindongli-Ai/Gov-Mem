# GateMem Full-Benchmark Result Template

Date: `YYYY-MM-DD`

## Evaluation Setting

- Run name:
  `TODO`
- Result directory:
  `TODO`
- Config:
  `TODO`
- Experiment mode:
  `TODO`
- Runtime API:
  `TODO`
- Official scorer:
  `third_party/GateMem-official/bench/scripts/score_predictions.py`
- LLM judge:
  `False` or `True`

## Main Table

The following table follows the GateMem paper's main reporting semantics:

- `U` = Utility
- `A` = Access-Control Violation Rate
- `F` = Active-Forgetting Failure Rate
- `MGS = U * (1 - A) * (1 - F)`

| Domain | Checkpoints | U | A | F | MGS | Action Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Medical | TODO | TODO | TODO | TODO | TODO | TODO |
| Office | TODO | TODO | TODO | TODO | TODO | TODO |
| Education | TODO | TODO | TODO | TODO | TODO | TODO |
| Household | TODO | TODO | TODO | TODO | TODO | TODO |
| Overall (full benchmark) | TODO | TODO | TODO | TODO | TODO | TODO |

## Metric Mapping

| Paper Metric | Official Summary Field |
|---|---|
| `U` | `utility_accuracy` |
| `A` | `privacy_leakage_rate` |
| `F` | `deletion_leakage_rate` |
| `MGS` | `compliance_utility_score` |

`Action Acc.` is a supplementary debugging metric. It is not one of the four main paper headline metrics.

## Source Files

- suite summary:
  `TODO`
- medical summary:
  `TODO`
- office summary:
  `TODO`
- education summary:
  `TODO`
- household summary:
  `TODO`

## Notes

- This file is intended for full-benchmark reporting rather than partial-subset reporting.
- If the run is not full benchmark, explicitly rename the report to `partial`, `subset`, or `cross-domain partial`.
- If `LLM judge = True`, note that the paper-style headline metrics are then judge-derived.

## Short Interpretation

- Main strengths:
  `TODO`
- Main weaknesses:
  `TODO`
- Most important follow-up directions:
  `TODO`
