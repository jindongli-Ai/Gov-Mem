# Stateful Policy Location-Carrier Repair: 60-Checkpoint Validation

## Scope

- Manifest: `experiments/gatemem_suites/2026-07-27-05-40-01_stateful_policy_generalization_60.json`
- Output: `outputs/2026-07-28-stateful_policy_generalization_60_v21_location_carrier/`
- Runtime: real Yunwu API; 4 domains and 3 episodes in parallel
- Key isolation: 30-key pool, one leased key per episode
- Official judge: GateMem `gpt-4o` judge through Yunwu
- Change under test: preserve a concrete authorized location when a later
  route summary uses only a generic method label such as `keyed entry`

## Official Summary

| Domain | Checkpoints | Utility cases | U | A | F | OR | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Education | 15 | 4 | 0.250 | 0.200 | 0.167 | 0.000 | 0.167 | 0.667 |
| Household | 15 | 5 | 0.400 | 0.000 | 0.143 | 0.400 | 0.343 | 0.667 |
| Medical | 15 | 5 | 0.400 | 0.600 | 0.000 | 0.600 | 0.160 | 0.667 |
| Office | 15 | 7 | 0.286 | 0.000 | 0.000 | 0.143 | 0.286 | 0.933 |
| **Overall** | **60** | **21** | **0.333** | **0.286** | **0.080** | — | **0.219** | **0.733** |

Overall U/A/F are category-count weighted. The overall MGS is
`U * (1 - A) * (1 - F)`.

## Comparison

| Version | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|
| Stable reference 60 | 0.381 | 0.143 | 0.150 | 0.278 | 0.717 |
| v21 location-carrier 60 | 0.333 | 0.286 | 0.080 | 0.219 | 0.733 |
| Delta | -0.048 | +0.143 | -0.070 | -0.059 | +0.016 |

The run is not a release candidate. The location repair passed its targeted
Apricot validation (`utility=1.0`) and did not introduce a context leak, but
the broad sample still has independent over-refusal, action-contract, and
multi-field completeness failures. No GitHub push was made.

## Next Repair Boundary

The next change should remain bounded to two related mechanisms:

1. Normalize lifecycle/privacy delivery actions at the final policy boundary,
   including the distinction between `refuse`, `no_memory`, and
   `answer_redacted`.
2. Preserve an authorized operational field in mixed requests instead of
   converting the entire request into a refusal.

The location-carrier change remains enabled and is covered by the field
projection regression tests.
