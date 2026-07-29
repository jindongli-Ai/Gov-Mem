# Gov-Mem v3 Stage 2 Boundary Fix: 200-Checkpoint Recheck

Date: 2026-07-30

## Experiment Settings

| Setting | Value |
|---|---|
| Checkpoints | 200, 50 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, frozen |
| Stage 2 | Typed routing, current-field long-context ledger, sensitive/deletion/existence boundaries, action-label correction |
| Parallelism | 4 domain workers, 8 episode workers per domain |
| Key isolation | 30-key pool; one leased key per episode |
| Manifest | `rag_naive_v3_stage2_generalization_200_seed20260731.json` |
| Output | `outputs/2026-07-30-rag_naive_v3_stage2_generalization_200_seed20260731_fix_boundaries_v2` |

`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values. `A` and `F` below are answer-level privacy and
deletion/staleness leakage rates from the official evaluation.

## Current Results

| Domain | N | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Education | 50 | 27.78% | 5.00% | 0.00% | 0.2639 | 27.78% | 62.00% |
| Household | 50 | 53.85% | 12.50% | 0.00% | 0.4712 | 23.08% | 58.00% |
| Medical | 50 | 68.42% | 21.43% | 0.00% | 0.5376 | 10.53% | 80.00% |
| Office | 50 | 70.00% | 5.56% | 4.55% | 0.6311 | 20.00% | 80.00% |
| **Overall domain mean** | **200** | | | | **0.4759** | | |

## Comparison With Previous 200

| Version | Education MGS | Household MGS | Medical MGS | Office MGS | Overall MGS |
|---|---:|---:|---:|---:|---:|
| Previous boundary version | 0.2500 | 0.4038 | 0.1194 | 0.5939 | 0.3418 |
| Boundary fix v1 intermediate | 0.2500 | 0.3814 | 0.4737 | 0.6611 | 0.4416 |
| Boundary fix v2 final recheck | 0.2639 | 0.4712 | 0.5376 | 0.6311 | 0.4759 |
| Change vs previous | +0.0139 | +0.0673 | +0.4182 | +0.0372 | +0.1341 |

## Stage 2 Coverage

| Signal | Cases |
|---|---:|
| Long-context field ledger applied | 17/200 |
| Mixed projection applied | 18/200 |
| Deterministic policy/safety gate applied | 120/200 |

## Interpretation

The same blind 200-case manifest was used before and after the change, so
the improvement is not caused by selecting easier checkpoints. The strongest
gains are the privacy boundaries: Household A fell from 25.00% to
12.50%, Education A fell from 10.00% to 5.00%, and Medical A fell from 78.57%
to 21.43%. The new existence boundary blocks confirmation of private-note
presence/deletion, while the clinical boundary blocks unscoped result and
diagnostic-interpretation disclosure.

This is not promoted as the final Gov-Mem v3 version yet. Education Utility
remains low at 27.78%, and Office F has a small regression to 4.55% in this
run. The next change should target Education and Household field realization
without changing Stage 1 or broadening the current sensitive gate globally.
