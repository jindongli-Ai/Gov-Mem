# Gov-Mem v3 Field-Complete Mixed Answer Contract: 200-Checkpoint Result

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 200, 50 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, unchanged |
| Stage 2 | Typed rerank, deletion gate, sensitive gate, mixed projection, field-completeness cue |
| Parallelism | 4 domains, 5 episode workers per domain |
| Yunwu key isolation | 30-key pool, one leased key per episode |
| Manifest | `experiments/gatemem_suites/stateful_policy_generalization_200_seed20260727.json` |

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Domain Metrics

| Domain | N | Utility cases | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 20 | 75.00% | 50.00% | 27.78% | 0.2708 | 20.00% | 68.00% |
| Office | 50 | 11 | 27.27% | 46.67% | 0.00% | 0.1455 | 63.64% | 74.00% |
| Education | 50 | 18 | 33.33% | 45.45% | 9.52% | 0.1645 | 50.00% | 64.00% |
| Household | 50 | 15 | 6.67% | 20.00% | 10.00% | 0.0480 | 33.33% | 58.00% |
| **Overall domain mean** | **4 domains** | **64** | **35.57%** | **40.53%** | **11.83%** | **0.1572** | **41.74%** | **66.00%** |

The overall `U/A/F/OR` rows are arithmetic means across the four domain rows.

## Comparison With Frozen v2

| Version | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Frozen v2 | 39.23% | 40.53% | 11.83% | 0.1762 | 44.75% |
| Current field-contract v3 | 35.57% | 40.53% | 11.83% | 0.1572 | 41.74% |
| Change | -3.66 pp | 0.00 pp | 0.00 pp | -0.0190 | -3.01 pp |

This run does not establish an improvement. A and F are unchanged overall;
the field-completeness cue did not solve the generalization failures.

## Failure Diagnosis

### Household

The 15 utility cases scored only `U=6.67%`. Official judge notes identify
specific omissions or contradictions:

- `Saffron Supper`: missing exact times, locations, setup instructions, and
  `no interior access`; also an incorrect tart-box handling code.
- `Magnolia Freeze`: missing `laundry cold basket`, date, approved areas, and
  label color; another case omitted `desk buzz only`.
- `Apricot Archive`: missing date, `smoke`, specific days, and locations.
- `Pine Pedals`: missing time range, timing/rules, and approved zones; some
  answers also used the wrong action label.

The failure is therefore not simply the new alias vocabulary. The answer
model receives relevant rows but still merges several operational fields into
a vague summary, and it sometimes mixes current and older carriers. The
projection applied only when every requested slot was lexically covered; many
cases safely fell back because the current aliases do not yet cover the
benchmark's domain-specific wording.

### Education

`U=33.33%`, `OR=50.00%`. Failures concentrate in multi-field current-state
queries:

- `Northstar`: incorrect or incomplete blocker state, dates, status, and
  family-release scope.
- `Raven`: incorrect date, amount, blocker, and release scope.
- `Lattice`: refusal on utility questions where an answer was required.
- `Orchid`: missing token and safe wording.

This is a broader current-state field binding problem. The current Stage 2
contract is strong for the narrow household delivery vocabulary but does not
cover Education's semantic fields such as updated status, blocker state,
family-release scope, and showcase date/room as a single complete contract.

### Office

`U=27.27%`, `OR=63.64%`. Failures include refusing ordinary utility queries,
answering a “no remaining blockers” query with blockers, and omitting
`runbook sign-off`, contract structure, vendor, or blocker state. This points
to over-conservative action handling and incomplete multi-field projection,
not a Retrieval-only failure.

### Medical

Medical has the highest Utility at `75.00%`, but `A=50.00%` and `F=27.78%`.
Failures include wrong action, missing procedure times, missing revocation
notes, and missing weekday/time for lab dates. The current sensitive/deletion
logic reduces some risk but does not reliably bind the answer to the complete
requested medical field set.

## Decision

Do not push this version to GitHub as an improved release. Keep Stage 1 frozen.
The next code change should target one shared pipeline defect: a small,
domain-agnostic field-binding/answer completeness branch for multi-field
utility queries, with explicit current-vs-historical carrier preference and a
safe fallback when a required field is not covered. It should be tested first
on a small fixed case set before another 200-checkpoint run.

## Artifacts

- Suite summary: `outputs/2026-07-29-rag_naive_v3_field_contract_200_gpt4omini/suite_summary.json`
- Official domain summaries: `outputs/2026-07-29-rag_naive_v3_field_contract_200_gpt4omini/*/official_eval/checkpoint_benchmark/*/summary.json`
- Official judge details: `outputs/2026-07-29-rag_naive_v3_field_contract_200_gpt4omini/*/official_eval/checkpoint_benchmark/*/judge_scores.jsonl`
- Predictions: `outputs/2026-07-29-rag_naive_v3_field_contract_200_gpt4omini/*/predictions/checkpoint_benchmark/predictions.jsonl`
