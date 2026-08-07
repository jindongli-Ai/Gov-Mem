# Gov-Mem v3 Pure Evaluation with GPT-4o-mini: 40 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: constrained LLM rerank plus deterministic governance checks
- Long-context field ledger: **disabled**
- Manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json`
- Checkpoints: 40 total, 10 per domain
- MGS: `U * (1 - A) * (1 - F)`, followed by the arithmetic mean across four domains

The first official scoring pass had a transient Office judge failure. Episode
outputs were retained and official scoring was resumed. The final four domain
summaries each contain 10 judged checkpoints with `gpt-4o` and zero judge parse
failures.

## Result

| Domain | Checkpoints | U | A | F | MGS | Action |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 0.00% | 0.00% | 50.00% | 80.00% |
| Office | 10 | 40.00% | 0.00% | 0.00% | 40.00% | 70.00% |
| Education | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 90.00% |
| Household | 10 | 66.67% | 0.00% | 0.00% | 66.67% | 90.00% |
| **Four-domain average** | **40** | | | | **47.50%** | |

## Interpretation

This is the clean small-scale result for the current retrieved-evidence-only
framework with `gpt-4o-mini` as the memory-system base LLM. Compared with the
same 40-checkpoint Gemini run (`57.50%`), the base model change lowers utility,
especially in Office and Education. The result does not indicate a change in
Stage 1 retrieval or governance leakage: answer-level `A` and `F` are both zero
in all four domains. Larger validation is required before treating the 40-case
difference as a general model ranking.

## Artifacts

- Output directory: `outputs/2026-08-04_rag_naive_v3_pure_gpt4omini_generalization40/`
- Config: `configs/rag_naive_v3_openlux_gpt4omini_embedding3small_pure.yaml`
- Official summaries: `*/official_eval/checkpoint_benchmark/*/summary.json`
