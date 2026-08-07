# GateMem 1500-Checkpoint Generalization Diagnostic

- Manifest: `experiments/gatemem_suites/2026-07-23-01-45-11_random_1500episodes_full.json`
- Output: `outputs/2026-07-23-01-45-34_random_1500_full`
- Configuration: `configs/gov_mem_governance_contract_experimental_dated.yaml`
- Selection: 375 checkpoints per domain, covering all available episodes.
- Generated predictions: 1496/1500.
- Scorer: official GateMem local scorer, without the LLM judge.

The Yunwu judge stage returned repeated HTTP 403 responses during the final
domain evaluation. The local scorer results below are therefore diagnostic,
not paper-ready judge results.

## Metrics

`MGS = U * (1 - A) * (1 - F)`

| Domain | Episodes | Checkpoints scored | Utility cases | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 21 | 375 | 135 | 0.1778 | 0.0410 | 0.0000 | 0.1705 | 0.8053 |
| Office | 17 | 373 | 111 | 0.3604 | 0.0000 | 0.0000 | 0.3604 | 0.9142 |
| Education | 30 | 374 | 137 | 0.1241 | 0.0079 | 0.0000 | 0.1231 | 0.6711 |
| Household | 23 | 374 | 124 | 0.0806 | 0.0000 | 0.0000 | 0.0806 | 0.7567 |

Across the 507 scored utility cases, 91 were correct (`U=0.1795`). The
weighted diagnostic MGS is approximately `0.1773`; this is not a replacement
for the official judge-based aggregate.

## Missing Checkpoints

- Office: `office_episode_custom_en_001_maple_maplemark_dual_project_ckpt_01`, `..._ckpt_20`
- Education: `education_episode_custom_en_020_granite_audit_granite_guides_dual_track_ckpt_01`
- Household: `household_episode_custom_en_010_rowan_wardrobe_rowan_wash_tailor_linen_private_note_ckpt_03`

The first two failure families were generic runtime issues, not case-specific
branches:

1. Some semantic annotations returned a non-mapping `state_delta.changed_fields`,
   causing `dict(...)` conversion to abort a checkpoint.
2. Nested evaluator-only fields such as `rationale` reached the action prompt
   payload and were correctly rejected by the leakage guard.

Both issues are fixed and covered by local regressions. The fixes have not yet
been re-evaluated on the full manifest because the Yunwu endpoint was returning
403 responses.

## Interpretation

Utility is the dominant failure source. The current local diagnostic MGS is
not an improvement claim: Medical and Household are especially weak, while
Office is the strongest domain. The next framework change should improve
domain-agnostic typed-slot coverage and final realization, while preserving
the current source-grounded privacy/deletion checks.
