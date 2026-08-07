# Gov-Mem v3 Stage 2B Reasoning Rerank: Validator and Prompt Ablation

Date: 2026-07-30

## Settings

| Setting | Value |
|---|---|
| Checkpoints | 40 held-out checkpoints, 10 per domain |
| Memory-system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu for all API calls |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, frozen |
| Parallelism | 4 domains, 5 episode workers per domain |
| Key isolation | One leased Yunwu key per episode; 30-key pool |
| Metric | `MGS = U * (1-A) * (1-F)`; overall is the mean of four domain MGS values |

## Domain Results

Percentages below are official `gpt-4o` judge results.

### Reasoning OFF

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Education | 0.00% | 0.00% | 0.00% | 0.00% |
| Household | 0.00% | 0.00% | 0.00% | 0.00% |
| Medical | 75.00% | 33.33% | 0.00% | 50.00% |
| Office | 0.00% | 0.00% | 0.00% | 0.00% |
| **Overall** |  |  |  | **12.50%** |

### Reasoning ON: validator alias adaptation (v3)

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Education | 33.33% | 0.00% | 0.00% | 33.33% |
| Household | 0.00% | 0.00% | 0.00% | 0.00% |
| Medical | 75.00% | 33.33% | 0.00% | 50.00% |
| Office | 0.00% | 0.00% | 0.00% | 0.00% |
| **Overall** |  |  |  | **20.83%** |

### Reasoning ON: compact-rank prompt (v4)

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Education | 33.33% | 0.00% | 0.00% | 33.33% |
| Household | 0.00% | 0.00% | 0.00% | 0.00% |
| Medical | 75.00% | 33.33% | 0.00% | 50.00% |
| Office | 0.00% | 0.00% | 0.00% | 0.00% |
| **Overall** |  |  |  | **20.83%** |

## Stage 2B Audit

| Run | Validated reasoning results | Actual candidate-order changes | Overall MGS |
|---|---:|---:|---:|
| v2 baseline prompt | 5 | 4 | 20.83% |
| v3 validator aliases | 4 | 3 | 20.83% |
| v4 compact-rank prompt | 3 | 3 | 20.83% |

The changes preserved A/F and did not lower the measured MGS, but they also did
not improve effective reasoning coverage. The remaining failures are mostly
field-support coverage and evidence-quote consistency. The compact-rank prompt
did not solve this; it merely changed the failure mix.

## Decision

Do not promote v3/v4 and do not push them as a new GitHub version. Do not run a
200-checkpoint promotion experiment for this treatment yet. Stage 1 remains
unchanged. The next improvement should be a small redesign of the Stage 2B
certificate contract, preferably separating:

1. LLM candidate ordering and conflict explanation.
2. Deterministic per-field coverage certificates derived from the already
   projected candidates.
3. Exact quote and closed-set validation.

That keeps the LLM responsible for semantic ordering while avoiding a brittle
requirement that the small model simultaneously invent correct field keys,
candidate references, and quote coverage for a 20-candidate response.

## Artifacts

- OFF: `outputs/2026-07-30-rag_naive_v3_reasoning_rerank_heldout_off_40_v2/suite_summary.json`
- v2 ON: `outputs/2026-07-30-rag_naive_v3_reasoning_rerank_heldout_on_40_v2/suite_summary.json`
- v3 ON: `outputs/2026-07-30-rag_naive_v3_reasoning_rerank_heldout_on_40_v3/suite_summary.json`
- v4 ON: `outputs/2026-07-30-rag_naive_v3_reasoning_rerank_heldout_on_40_v4/suite_summary.json`
- Manifest: `experiments/gatemem_suites/rag_naive_v3_reasoning_rerank_heldout_40_seed20260730.json`
