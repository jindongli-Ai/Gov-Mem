# Result Reporting Convention

This directory stores experiment result summaries in Markdown format.

The goal is to keep terminology, table structure, and metric semantics consistent across runs.

## Terminology

### 1. Full Benchmark

A run over the complete intended evaluation set for all target domains.

Recommended label:

- `full benchmark`
- `full-benchmark evaluation`

Avoid calling this:

- `smoke`
- `quick check`

### 2. Partial Evaluation

A run over a subset of the benchmark, used for diagnosis, regression tracking, or rapid iteration.

Recommended label:

- `partial evaluation`
- `cross-domain partial evaluation`
- `balanced subset evaluation`

Use this when:

- only part of the checkpoint set is evaluated
- a domain-balanced subset is used
- the purpose is rapid research iteration rather than final reporting

### 3. Official-Compatible Evaluation

An evaluation whose scoring path is consistent with the official GateMem scorer.

In this project, that means the result is produced through:

- `third_party/GateMem-official/bench/scripts/score_predictions.py`

This term refers to the scoring protocol, not to the API provider used by Gov-Mem itself.

### 4. LLM Judge

The official GateMem scorer supports an optional LLM-as-a-judge mode.

Important distinction:

- `LLM judge = False`
  - metrics come from the official non-judge scoring path
- `LLM judge = True`
  - headline metrics become judge-derived

Any result file must explicitly state whether LLM judge was enabled.

## Required Table Semantics

All main result tables should follow the GateMem paper semantics:

- `U` = Utility
- `A` = Access-Control Violation Rate
- `F` = Active-Forgetting Failure Rate
- `MGS = U * (1 - A) * (1 - F)`

Official summary field mapping:

| Paper Metric | Official Summary Field |
|---|---|
| `U` | `utility_accuracy` |
| `A` | `privacy_leakage_rate` |
| `F` | `deletion_leakage_rate` |
| `MGS` | `compliance_utility_score` |

If `Action Acc.` is reported, it must be clearly marked as a supplementary debugging metric rather than a paper headline metric.

## Naming Convention

Recommended filenames:

- full benchmark:
  - `YYYY-MM-DD-HH-MM-SS_GateMem_full_benchmark_<tag>.md`
- partial evaluation:
  - `YYYY-MM-DD-HH-MM-SS_GateMem_cross_domain_partial_evaluation_<tag>.md`
  - `YYYY-MM-DD-HH-MM-SS_GateMem_<subset_name>_partial_evaluation_<tag>.md`
- framework recheck / targeted diagnosis:
  - `YYYY-MM-DD-HH-MM-SS_Gov-Mem_framework_recheck_<tag>.md`

Avoid ambiguous names such as:

- `smoke.md`
- `test_result.md`
- `quick_eval.md`

Required timestamp rule:

- the timestamp must be at the beginning of the filename
- the timestamp must use 24-hour format
- the timestamp should be precise to seconds
- recommended pattern:
  - `YYYY-MM-DD-HH-MM-SS`

## Minimum Required Sections

Each result file should contain:

1. evaluation setting
2. explicit statement of whether the run is full or partial
3. explicit statement of whether LLM judge is enabled
4. main table with `U / A / F / MGS`
5. source file locations
6. short interpretation

## Files Currently Present

- partial evaluation result:
  - `2026-07-06-05-28-39_GateMem_cross_domain_partial_evaluation_balanced40_official_compatible.md`
- full benchmark template:
  - `2026-07-06-05-28-41_GateMem_full_benchmark_result_template.md`
- framework recheck update:
  - `2026-07-06-05-28-40_Gov-Mem_framework_recheck_update.md`

## Practical Rule

If a run is not the complete benchmark, do not call it a full result.

If a run is small and fast, do not call it `smoke` in final reporting.

Use:

- `partial evaluation`

or, when needed:

- `official-compatible cross-domain partial evaluation`
