# Gov-Mem v3 Pure Retrieval-Evidence Evaluation: 40 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gemini-2.5-flash-lite`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: constrained LLM rerank plus deterministic governance checks
- Long-context field ledger: **disabled**
- Manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json`
- Checkpoints: 40 total, 10 per domain
- MGS: `U * (1 - A) * (1 - F)`, followed by the arithmetic mean across four domains

The first official scoring pass had one transient Household judge failure. The
episode outputs were retained and the official scoring pass was resumed. The
final four domain summaries below each contain 10 judged checkpoints with
`gpt-4o` and zero judge parse failures.

## Result

| Domain | Checkpoints | U | A | F | MGS | Action |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 0.00% | 0.00% | 50.00% | 80.00% |
| Office | 10 | 80.00% | 0.00% | 0.00% | 80.00% | 80.00% |
| Education | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 90.00% |
| Household | 10 | 66.67% | 0.00% | 0.00% | 66.67% | 90.00% |
| **Four-domain average** | **40** | | | | **57.50%** | |

## Interpretation

This is the first clean small-scale result after removing the complete-
transcript Long-Context path from the formal configuration. It is not directly
comparable to the previous Long-Context-enabled results except as an ablation
comparison. No privacy or deletion leakage was observed in this 40-checkpoint
sample, but Education utility remains the main weakness and requires broader
validation before drawing conclusions.

## Artifacts

- Output directory: `outputs/2026-08-04_rag_naive_v3_pure_gemini25flashlite_generalization40/`
- Config: `configs/rag_naive_v3_openlux_gemini25flashlite_embedding3small_pure.yaml`
- Official summaries: `*/official_eval/checkpoint_benchmark/*/summary.json`
