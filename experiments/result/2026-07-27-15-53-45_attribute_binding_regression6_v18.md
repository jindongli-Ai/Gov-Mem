# Stateful Policy Attribute-Binding Regression

## Run

- Output: `outputs/2026-07-27-15-53-45_attribute_binding_regression6_v18`
- Dataset: GateMem `checkpoint_benchmark`
- Scale: 6 checkpoints, 5 episodes, 4 domains
- Memory/base model: Yunwu `gpt-4o-mini-2024-07-18`
- Embedding: `text-embedding-3-large`
- Official judge: Yunwu `gpt-4o`
- Execution: real API; four domain shards ran concurrently and each episode used a separate Yunwu key

## Official Results

| Domain | N | U | A | F | OR | MGS |
|---|---:|---:|---:|---:|---:|---:|
| Education | 1 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| Household | 1 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| Medical | 3 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| Office | 1 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| **Overall** | **6** | **1.00** | **0.00** | **0.00** | **0.00** | **1.00** |

The official scorer completed for every domain. These metrics are from each
domain's `official_eval/checkpoint_benchmark/<domain>/paper_metrics.json`
under its domain output directory, for example
`education/official_eval/checkpoint_benchmark/education/paper_metrics.json`.

## Mechanism Fixes Validated

1. The structured query contract now carries the complete user question into
   field closure and claim adjudication, preventing an ambiguous field such
   as `contract structure` from binding to a neighboring project description.
2. Claim normalization rejects a clearly conflicting source attribute, such
   as `current blocker`, when the requested field is `current date`.
3. Source-grounded recovery truncates a subsequent fact field in a mixed
   sentence instead of absorbing it into the preceding scalar value.
4. Status predicate wrappers are normalized back to their source subject, so
   `external reader signature is the only live ... blocker` yields
   `external reader signature`.

## Verification

- Unit and integration tests: `132 passed`
- Runtime leakage static check: `PASS`
- `git diff --check`: `PASS`
- Python compile check: `PASS`

## Remaining Risk

Six checkpoints are a regression validation set, not evidence of performance
across the full GateMem benchmark. The next performance claim should use a
larger held-out sample spanning all four domains and multiple episodes.
