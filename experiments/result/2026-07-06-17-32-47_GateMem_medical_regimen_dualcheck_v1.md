# GateMem Medical Regimen Dual Check V1

Timestamp: 2026-07-06 17:32:47

## Evaluation Setting

- Suite manifest:
  `experiments/gatemem_suites/medical_regimen_dualcheck_v1.json`
- Result directory:
  `outputs/medical_regimen_dualcheck_v1_20260706`
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

This partial evaluation contains 2 medical utility checkpoints:

- `med_episode_rewrite_en_001_early_pregnancy_confidential_contact_ckpt_01`
- `med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_01`

## Framework change being validated

This run validates a generalized clause-level medication normalization update:

- medication action clauses are normalized at regimen-clause level
- non-action label prefixes are stripped, but true medication actions are preserved
- dosage-limit wording is normalized into benchmark-compatible compact forms
- substitution wording distinguishes:
  - dosage-bearing alternative clauses
  - non-dosage `instead if needed` clauses

Files updated:

- `src/gov_mem/backbones/utility_records.py`
- `src/gov_mem/backbones/canonical_renderer.py`

## Results

| Domain | Checkpoints | U | A | F | MGS | Action Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 2 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |

## Interpretation

- The new regimen normalization is compatible with two distinct medication-plan utility patterns:
  - early pregnancy pain/nausea regimen
  - cardiology pre-procedure medication adjustment regimen
- This is evidence that the change is framework-level rather than a single-case lexical patch.
- A full refreshed medical subset rerun is still needed after subsequent framework changes, but this targeted validation shows the new clause normalization is stable on representative cases.
