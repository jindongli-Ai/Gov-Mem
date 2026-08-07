# Gov-Mem v3 Medical Boundary Fix

Date: 2026-08-02

## Change

This incremental Stage 2A change adds two general safety signals:

1. Treat an explicit deleted historical `contact` as a deletion-gated scalar.
2. In the `medical` domain, gate confirmation questions that bind a named
   person to a patient/medication/seizure/diagnosis association.

Ordinary treatment-plan questions remain on the normal answer path. Stage 1,
retrieval, LLM reranking, and long-context processing were unchanged.

## Experiment Settings

| Setting | Value |
|---|---|
| Checkpoints | 10 medical checkpoints |
| Memory system provider | OpenLux |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Embedding model | `text-embedding-3-small` |
| Official evaluation provider | OpenLux |
| Official evaluation LLM | `gpt-4o` |
| Parallelism | 1 domain, 3 episode workers |
| Key isolation | 30-key pool; one key leased per episode |

## Results

| Domain | N | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 0.00% | 0.00% | **50.00%** | 80.00% |

The corresponding pre-fix 10-case slice had `A=66.67%` and `MGS=16.67%`.
The comparison is directional because the official judge is stochastic, but
the two previously observed Medical privacy leaks were both blocked.

Output: `outputs/2026-08-02-rag_naive_v3_medical_boundary_10_openlux_gpt4omini_retry`
