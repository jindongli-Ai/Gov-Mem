# Stateful Policy Reasoning: 60-Checkpoint Evaluation

- Manifest: `experiments/gatemem_suites/2026-07-27-05-40-01_stateful_policy_generalization_60.json`
- Output: `outputs/2026-07-27-05-40-01_stateful_policy_generalization_60/`
- Dataset: GateMem `checkpoint_benchmark`
- Method: Stateful Policy Reasoning Full
- Memory-system model: `gpt-4o-mini-2024-07-18`
- Embedding model: `text-embedding-3-large`
- Official judge: `gpt-4o` through Yunwu
- Scorer: GateMem official scorer
- Selection: 4 domains x 3 episodes/domain x 5 checkpoints/episode = 60 checkpoints
- Runtime: real Yunwu API; no mock results

## Official Results

`MGS = U * (1 - A) * (1 - F)`.

Here `A` is the official answer-level privacy leakage rate and `F` is the
official answer-level deletion/staleness leakage rate.

| Domain | Episodes | Checkpoints | Utility cases | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 3 | 15 | 5 | 0.600 | 0.200 | 0.000 | 0.480 | 0.800 |
| Office | 3 | 15 | 7 | 0.286 | 0.000 | 0.000 | 0.286 | 0.800 |
| Education | 3 | 15 | 4 | 0.500 | 0.200 | 0.333 | 0.267 | 0.533 |
| Household | 3 | 15 | 5 | 0.200 | 0.000 | 0.143 | 0.171 | 0.733 |
| **Overall** | **12** | **60** | **21** | **0.381** | **0.143** | **0.150** | **0.278** | **0.717** |

The Overall row is checkpoint-weighted for action accuracy and category-count
weighted for U/A/F. It is computed from the official domain summaries:
8/21 utility cases correct, 2/14 privacy cases leaked, and 3/20 deletion or
staleness cases leaked.

## Additional Safety Signals

The official summaries reported zero privacy-context leakage for all four
domains. End-to-end privacy leakage was `0.000` for Office, Education and
Household, and `0.200` for Medical. End-to-end deletion leakage was `0.000`
for Office, Education and Medical, and `0.143` for Household. These signals
are reported separately because the paper metrics above use the official
answer-level A/F fields.

## Interpretation

This is a real-api, official-scorer result, but it is still a small blind
generalization sample rather than a claim over all GateMem checkpoints.
Utility remains the dominant limiter of MGS: overall action accuracy is
`0.717`, while only `8/21` utility cases are correct. The largest immediate
weakness is Household (`U=0.200`), followed by Office (`U=0.286`) and
Education (`U=0.500`). Privacy/deletion protection is comparatively stronger,
although Education and Medical still contain answer-level privacy leakage and
Education contains deletion leakage.

## Reproduction

The exact manifest and per-domain official outputs are preserved under:

```text
outputs/2026-07-27-05-40-01_stateful_policy_generalization_60/
```

The authoritative per-domain files are:

```text
medical/official_eval/checkpoint_benchmark/medical/summary.json
office/official_eval/checkpoint_benchmark/office/summary.json
education/official_eval/checkpoint_benchmark/education/summary.json
household/official_eval/checkpoint_benchmark/household/summary.json
```

The final official scorer used the GateMem judge prompt and Yunwu provider;
all 60 checkpoints completed and produced judge scores.
