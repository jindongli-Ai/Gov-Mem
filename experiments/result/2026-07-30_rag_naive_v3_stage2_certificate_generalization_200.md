# Gov-Mem v3 Stage 2B Certificate Contract Generalization

Date: 2026-07-30

## Settings

| Setting | Value |
|---|---|
| Stage 1 | Official-compatible RAG-Naive retrieval, frozen |
| Stage 2 | Rule gates plus constrained LLM reasoning rerank and deterministic quote checks |
| Memory-system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu for all API calls |
| Embedding | `text-embedding-3-small` |
| Parallelism | 4 domains, 5 episode workers per domain |
| Key isolation | One leased Yunwu key per episode; 30-key pool |
| Metric | `MGS = U * (1-A) * (1-F)`; overall is the mean of four domain MGS values |

## 40-Checkpoint Diagnostic

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Education | 33.33% | 0.00% | 0.00% | 33.33% |
| Household | 0.00% | 0.00% | 0.00% | 0.00% |
| Medical | 75.00% | 33.33% | 0.00% | 50.00% |
| Office | 100.00% | 0.00% | 0.00% | 100.00% |
| **Overall** |  |  |  | **45.83%** |

Stage 2B produced 10 validated certificates and changed the candidate order
in 7 cases. The change was limited to making `field_support` optional audit
metadata; closed-set candidate references, selected-subset checks, and exact
source quotes remain mandatory.

## New 200-Checkpoint Held-Out Generalization

This manifest contains 50 previously unused checkpoints per domain and excludes
the earlier 40/200 manifests without using answer, evidence, or scorer fields.

| Domain | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Education | 27.78% | 5.00% | 0.00% | 26.39% |
| Household | 53.85% | 12.50% | 0.00% | 47.12% |
| Medical | 73.68% | 28.57% | 0.00% | 52.63% |
| Office | 70.00% | 5.56% | 4.55% | 63.11% |
| **Overall** |  |  |  | **47.31%** |

Stage 2B telemetry over the 200 checkpoints:

| Domain | Validated certificates | Actual order changes |
|---|---:|---:|
| Education | 8 | 8 |
| Household | 14 | 12 |
| Medical | 4 | 4 |
| Office | 8 | 8 |
| **Total** | **34** | **32** |

## Interpretation

The 200-checkpoint result confirms that the certificate change is not only an
Office-only rescue. Household reaches 47.12% MGS and Office reaches 63.11% on
the held-out set. Education remains the weakest domain at 26.39%, while Medical
has the strongest Utility but still carries higher access leakage.

The 200 result should be compared with the earlier 200+600 result as evidence
from different held-out manifests, not pooled as if it were the same sample.
The current code is therefore recorded as a new experimental history version;
Stage 1 remains unchanged.

## Artifacts

- 40-case summary: `outputs/2026-07-30-rag_naive_v3_reasoning_rerank_heldout_on_40_v6/suite_summary.json`
- 200-case summary: `outputs/2026-07-30-rag_naive_v3_stage2_generalization_200_cert_v1/suite_summary.json`
- 200 manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_200_seed20260731.json`
