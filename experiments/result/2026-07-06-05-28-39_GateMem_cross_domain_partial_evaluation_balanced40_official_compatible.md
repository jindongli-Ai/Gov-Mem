# GateMem Cross-Domain Partial Evaluation

Date: 2026-07-06

## Terminology

Here, "smoke" means a small-scale, fast regression evaluation used to quickly test whether the current framework is broadly functioning across multiple domains.

To avoid ambiguity, this run should be understood as:

- cross-domain partial evaluation
- balanced subset evaluation
- official-compatible small-scale check

It is not a full-benchmark run over all GateMem checkpoints, so its numbers are useful for diagnosis and regression tracking, but they should not be interpreted as final paper-level benchmark results.

## Evaluation Setting

- Suite manifest:
  `experiments/gatemem_suites/gatemem_smoke40.json`
- Result directory:
  `outputs/2026-07-06-01-28-28_gatemem_smoke40_v9_review`
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
- per domain:
  - 4 utility
  - 3 privacy
  - 3 safety

## Main Table

The following table follows the GateMem paper's main reporting semantics:

- `U` = Utility
- `A` = Access-Control Violation Rate
- `F` = Active-Forgetting Failure Rate
- `MGS = U * (1 - A) * (1 - F)`

| Domain | Checkpoints | U | A | F | MGS | Action Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 0.75 | 0.00 | 0.00 | 0.75 | 1.00 |
| Office | 10 | 0.50 | 0.00 | 0.00 | 0.50 | 1.00 |
| Education | 10 | 0.50 | 0.00 | 0.00 | 0.50 | 0.80 |
| Household | 10 | 0.25 | 0.00 | 0.00 | 0.25 | 0.90 |
| Overall (balanced subset) | 40 | 0.50 | 0.00 | 0.00 | 0.50 | 0.925 |

## Metric Mapping

Official file fields and paper metrics are aligned as follows:

| Paper Metric | Official Summary Field |
|---|---|
| `U` | `utility_accuracy` |
| `A` | `privacy_leakage_rate` |
| `F` | `deletion_leakage_rate` |
| `MGS` | `compliance_utility_score` |

`Action Acc.` is included here as a supplementary debugging metric. It is not one of the four main paper headline metrics.

## Result Sources

- suite summary:
  `outputs/2026-07-06-01-28-28_gatemem_smoke40_v9_review/suite_summary.json`
- medical:
  `outputs/2026-07-06-01-28-28_gatemem_smoke40_v9_review/medical/official_eval/gatemem/medical/summary.json`
- office:
  `outputs/2026-07-06-01-28-28_gatemem_smoke40_v9_review/office/official_eval/gatemem/office/summary.json`
- education:
  `outputs/2026-07-06-01-28-28_gatemem_smoke40_v9_review/education/official_eval/gatemem/education/summary.json`
- household:
  `outputs/2026-07-06-01-28-28_gatemem_smoke40_v9_review/household/official_eval/gatemem/household/summary.json`

## Short Interpretation

- On this balanced 40-checkpoint subset, the current Gov-Mem implementation shows relatively stable governance behavior on privacy and deletion.
- The main remaining weakness is authorized utility realization.
- The current domain difficulty ranking on this subset is:
  `Medical > Office = Education > Household`
- Since this is a partial subset rather than the full benchmark, these numbers should be used for iterative research comparison inside the project, not as final headline benchmark claims.
