# Gov-Mem v3 Sensitive Delivery Gate: 200-Checkpoint Result

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
| Stage 2 | Typed rerank + deletion gate + mixed projection + bounded sensitive delivery gate |
| Parallelism | 4 domains, up to 8 episode workers/domain |
| Key isolation | 30-key pool, one leased key per episode |

## Official Metrics By Domain

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`.

| Domain | N | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 75.00% | 50.00% | 27.78% | 0.2708 | 20.00% | 68.00% |
| Office | 50 | 36.36% | 46.67% | 0.00% | 0.1939 | 54.55% | 76.00% |
| Education | 50 | 38.89% | 45.45% | 9.52% | 0.1919 | 44.44% | 66.00% |
| Household | 50 | 6.67% | 20.00% | 10.00% | 0.0480 | 60.00% | 52.00% |
| **Overall weighted** | **200** | **42.19%** | **39.62%** | **10.84%** | **0.2271** | **42.19%** | **65.50%** |

## Comparison With Previous 200

| Version | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Previous v3 action boundary | 40.62% | 54.72% | 10.84% | 0.1640 | 42.19% |
| v3 bounded sensitive gate | 42.19% | 39.62% | 10.84% | 0.2271 | 42.19% |
| Change | +1.57 pp | -15.10 pp | 0.00 pp | +0.0631 | 0.00 pp |

The improvement is real on this 200-checkpoint run, but the framework is still
not release quality. Household Utility remains the dominant failure surface,
and overall leakage remains high because many privacy queries do not match the
narrow lexical gate.

## Gate Analysis

The final gate triggered on 14/200 checkpoints. It blocks only explicit exact
or overreach-shaped sensitive requests, combinations of multiple sensitive
fields, and concrete clinical measurements. It deliberately leaves a PIN or
credential embedded in an otherwise ordinary current plan available to the
answer path.

This distinction corrected the first gate regression: the first 200 run
blocked 23/200 cases, including many valid Education and Household current
summaries. The final gate reduced that over-refusal while preserving the
privacy improvement.

## Next Priority

1. Household mixed current-state realization: construct a field-level delivery
   projection for window, route, areas, signoff, overflow, and labels. The
   projection must preserve ordinary fields while separately marking a PIN or
   exact private location as restricted.
2. Education and Office mixed summaries: reuse the existing v2 vocabulary and
   policy carriers to distinguish an authorized operational summary from a
   private-file or cross-role query. Do not add another broad keyword gate.
3. Deletion/general historical coverage: extend the semantic lifecycle branch
   for wording, location, relationship, and comparison queries instead of
   expanding the scalar deletion keyword list.
4. Only after these deterministic projections stabilize should an optional
   single LLM field-adjudication call be tested. The rejected query-contract
   probe shows that an extra general-purpose call can reduce Utility.

The next experiment should therefore be a Household-targeted projection smoke
test, followed by a held-out 40-checkpoint cross-domain regression. A new 200
checkpoint run is not justified until Household Utility improves without
raising A or F.
