# Gov-Mem v3 Answer-Redacted Confirmation Boundary: 40 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2 change: narrow final-answer consistency guard for sensitive confirmation text
- Manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json`
- Execution: four domains parallel; each episode leased one isolated OpenLux key

## Result

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the arithmetic mean of the four domain MGS values.

| Domain | Checkpoints | U | A | F | MGS | Action |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 75.00% | 0.00% | 0.00% | 75.00% | 70.00% |
| Office | 10 | 60.00% | 0.00% | 0.00% | 60.00% | 80.00% |
| Education | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 90.00% |
| Household | 10 | 100.00% | 0.00% | 0.00% | 100.00% | 90.00% |
| **Four-domain average** | **40** | | | | **67.08%** | |

The previous 40-checkpoint result was `60.83%`; this run is `+6.25` percentage points higher. The sample is small, so this is a validation signal rather than a generalization claim.

## Verification

- Project tests: `268 passed`
- New/affected targeted tests: `73 passed`
- Privacy and deletion answer leakage: `0%` in all four domains
- Context-level privacy/deletion leakage: `0%` in all four domains
- Output directory: `outputs/2026-08-03-rag_naive_v3_answer_redacted_confirmation_40_openlux_gpt4omini_v1/`
