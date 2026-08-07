# Gov-Mem v3 Answer-Redacted Confirmation Boundary: 200 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2 change: narrow final-answer consistency guard for sensitive confirmation text
- Manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_200_seed20260731.json`
- Execution: four domains parallel; each episode leased one isolated OpenLux key

## Result

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the arithmetic mean of the four domain MGS values.

| Domain | Checkpoints | U | A | F | MGS | Action | Privacy context | Deletion context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 73.68% | 28.57% | 0.00% | 52.63% | 78.00% | 0.00% | 0.00% |
| Office | 50 | 80.00% | 5.56% | 0.00% | 75.56% | 84.00% | 0.00% | 0.00% |
| Education | 50 | 55.56% | 5.00% | 0.00% | 52.78% | 68.00% | 0.00% | 0.00% |
| Household | 50 | 53.85% | 25.00% | 0.00% | 40.38% | 62.00% | 0.00% | 0.00% |
| **Four-domain average** | **200** | | | | **55.34%** | | | |

The previous comparable 200-case result was `56.04%`. This run is `-0.70` percentage points lower, so the change is not yet a demonstrated generalization improvement.

## Interpretation

- The final-answer guard did not introduce deletion leakage; all domain F values remained `0%`.
- Context-level privacy and deletion leakage remained `0%` in every domain.
- Medical and Household remain the main weaknesses because answer-level privacy leakage is `28.57%` and `25.00%`, respectively.
- The 40-case improvement was optimistic. The 200-case result supports keeping the change small, but not treating it as a major framework improvement.

## Artifacts

- Output directory: `outputs/2026-08-03-rag_naive_v3_answer_redacted_confirmation_200_openlux_gpt4omini_v1/`
- Official summaries: domain-specific `official_eval/checkpoint_benchmark/<domain>/summary.json`
