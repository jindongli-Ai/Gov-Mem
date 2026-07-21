# GateMem Gov-Mem Results

Timestamp: 2026-07-08 23:18:08

## Evaluation Setting

- Suite manifest: `experiments/gatemem_suites/gatemem_smoke40.json`
- Run directory: `/data_nvme/user/jli/codes/2027_Gov-Mem/outputs/gatemem_smoke40_2026-07-08-22-44-00_fix4`
- Runtime API: `yunwu`
- Base LLM: `gpt-5.4-nano-2026-03-17`
- Official scorer output root: `/data_nvme/user/jli/codes/2027_Gov-Mem/outputs/gatemem_smoke40_2026-07-08-22-44-00_fix4`
- Official scorer: `third_party/GateMem-official/bench/scripts/score_predictions.py`
- LLM judge: `False`

## Main Table

| Domain | Checkpoints | U | A | F | MGS | Action Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Medical | 10 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Office | 10 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Education | 10 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Household | 10 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| Overall | 40 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## Per-Domain Official Files

- Medical: `/data_nvme/user/jli/codes/2027_Gov-Mem/outputs/gatemem_smoke40_2026-07-08-22-44-00_fix4/medical/official_eval/gatemem/medical/summary.json`
- Office: `/data_nvme/user/jli/codes/2027_Gov-Mem/outputs/gatemem_smoke40_2026-07-08-22-44-00_fix4/office/official_eval/gatemem/office/summary.json`
- Education: `/data_nvme/user/jli/codes/2027_Gov-Mem/outputs/gatemem_smoke40_2026-07-08-22-44-00_fix4/education/official_eval/gatemem/education/summary.json`
- Household: `/data_nvme/user/jli/codes/2027_Gov-Mem/outputs/gatemem_smoke40_2026-07-08-22-44-00_fix4/household/official_eval/gatemem/household/summary.json`

## Notes

- This run validates the framework after the current-state family-scoped scoring revision that prevents cross-slot update evidence from incorrectly dominating target-date selection.
- Under the official GateMem-compatible reporting pipeline, all four smoke40 domains reached `action_accuracy=1.0`, `utility_accuracy=1.0`, zero privacy leakage, zero deletion leakage, and `MGS=1.0`.
