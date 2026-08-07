# Gov-Mem v3 Full Evaluation with Gemini 2.5 Flash Lite: 2218 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gemini-2.5-flash-lite`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2: unchanged frozen Gov-Mem v3 typed rerank and answer-redacted confirmation boundary
- Full manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Execution: all 2218 available checkpoints; four domains and episode shards ran in parallel with isolated OpenLux keys

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the arithmetic mean of the four domain MGS values. `A` is answer-level privacy leakage and `F` is answer-level deletion/staleness leakage. Action and OR are reported for completeness and are not multiplied into MGS.

## Full Result

| Domain | Checkpoints | U | A | F | MGS | Action | OR | Privacy context | Deletion context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 78.57% | 35.94% | 10.73% | 44.93% | 75.65% | 5.24% | 0.00% | 0.00% |
| Office | 547 | 65.58% | 5.85% | 1.35% | 60.91% | 84.28% | 11.04% | 0.00% | 0.00% |
| Education | 540 | 42.78% | 17.22% | 7.78% | 32.66% | 75.56% | 10.56% | 0.00% | 0.00% |
| Household | 552 | 51.09% | 21.74% | 3.26% | 38.68% | 69.93% | 16.85% | 0.00% | 0.00% |
| **Four-domain average** | **2218** | | | | **44.29%** | | | | |

## Comparison Across Memory-System Base LLMs

All runs use the same 2218-checkpoint manifest, frozen framework, `text-embedding-3-small`, and official `gpt-4o` evaluator.

| Domain | GPT-4o-mini | GPT-5.4-nano | GPT-5.4-mini | Gemini 2.5 Flash Lite |
|---|---:|---:|---:|---:|
| Medical | 37.18% | 37.28% | 43.64% | **44.93%** |
| Office | 53.32% | 55.99% | 53.80% | **60.91%** |
| Education | 31.31% | 29.26% | 28.13% | **32.66%** |
| Household | 34.32% | 30.29% | 38.56% | **38.68%** |
| **Four-domain average** | **39.03%** | **38.21%** | **41.03%** | **44.29%** |

## Interpretation

- Gemini 2.5 Flash Lite is currently the strongest tested memory-system base LLM on the full 2218-checkpoint population: `44.29%` average MGS.
- It improves all four domain MGS values relative to GPT-4o-mini, with the largest gains in Office and Medical.
- The gain is not caused by retrieval changes: Stage 1, embedding, framework code, and official evaluator were held fixed. Context-level privacy/deletion exposure remained 0% in every domain.
- Education remains the lowest-MGS domain, although it improves over all previous base-LLM runs. Household has improved utility but still has meaningful answer-level privacy/deletion leakage.
- No framework code was changed for this comparison; only the memory-system base model was changed.

## Artifacts

- Output directory: `outputs/2026-08-05-rag_naive_v3_full_all_2218_openlux_gemini25flashlite_v1/`
- Config: `configs/rag_naive_v3_openlux_gemini25flashlite_embedding3small_stage2_on.yaml`
- Per-domain official summaries: `official_eval/checkpoint_benchmark/<domain>/summary.json`
