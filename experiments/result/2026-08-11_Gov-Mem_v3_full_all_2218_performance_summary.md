# Frozen Gov-Mem v3 Typed-Rerank Full-Benchmark Performance Summary

Date: 2026-08-11
Benchmark: GateMem full benchmark, 2,218 checkpoints
Protocol: `rag_naive_v3_typed_rerank` (not `govmem_symbolic`)
Metric: `MGS = U * (1 - A) * (1 - F)`

| Task | Base LLM | N | U | A | F | MGS |
|---|---|---:|---:|---:|---:|---:|
| Medical | GPT-4o-mini | 579 | 64.29% | 43.75% | 9.04% | 32.89% |
| Medical | GPT-5-mini | 579 | 74.76% | 34.90% | 12.99% | 42.35% |
| Medical | GPT-5.4 | 579 | 78.10% | 21.88% | 6.78% | 56.88% |
| Medical | GPT-5.4-mini | 579 | 70.48% | 27.60% | 6.78% | 47.56% |
| Medical | Gemini-2.5-Flash-Lite | 579 | 71.90% | 27.08% | 6.21% | 49.17% |
| Medical | DeepSeek-V4-Flash | 579 | 81.90% | 22.40% | 7.34% | **58.89%** |
| Medical | Llama-3.3-70B-Instruct | 579 | 39.05% | 18.75% | 12.43% | 27.78% |
| Office | GPT-4o-mini | 547 | 46.10% | 4.68% | 1.35% | 43.35% |
| Office | GPT-5-mini | 547 | 67.53% | 2.92% | 0.45% | **65.26%** |
| Office | GPT-5.4 | 547 | 63.64% | 1.17% | 0.45% | 62.61% |
| Office | GPT-5.4-mini | 547 | 57.79% | 4.09% | 1.80% | 54.43% |
| Office | Gemini-2.5-Flash-Lite | 547 | 68.18% | 4.68% | 1.80% | 63.82% |
| Office | DeepSeek-V4-Flash | 547 | 64.29% | 2.92% | 1.35% | 61.56% |
| Office | Llama-3.3-70B-Instruct | 547 | 36.36% | 1.75% | 2.25% | 34.92% |
| Education | GPT-4o-mini | 540 | 33.89% | 18.89% | 6.67% | 25.66% |
| Education | GPT-5-mini | 540 | 35.56% | 13.33% | 10.00% | 27.73% |
| Education | GPT-5.4 | 540 | 27.22% | 9.44% | 6.11% | 23.14% |
| Education | GPT-5.4-mini | 540 | 31.67% | 14.44% | 9.44% | 24.53% |
| Education | Gemini-2.5-Flash-Lite | 540 | 38.33% | 14.44% | 7.22% | **30.43%** |
| Education | DeepSeek-V4-Flash | 540 | 32.78% | 10.00% | 8.89% | 26.88% |
| Education | Llama-3.3-70B-Instruct | 540 | 17.22% | 11.11% | 3.89% | 14.71% |
| Household | GPT-4o-mini | 552 | 42.93% | 19.02% | 1.09% | 34.39% |
| Household | GPT-5-mini | 552 | 55.98% | 21.74% | 1.63% | 43.09% |
| Household | GPT-5.4 | 552 | 45.65% | 16.30% | 0.00% | 38.21% |
| Household | GPT-5.4-mini | 552 | 45.65% | 15.22% | 1.63% | 38.07% |
| Household | Gemini-2.5-Flash-Lite | 552 | 46.74% | 17.93% | 2.72% | 37.31% |
| Household | DeepSeek-V4-Flash | 552 | 57.61% | 15.76% | 2.17% | **47.47%** |
| Household | Llama-3.3-70B-Instruct | 552 | 23.91% | 12.50% | 1.09% | 20.70% |

The four-domain average MGS values are: GPT-4o-mini 34.07%, GPT-5-mini
44.61%, GPT-5.4 45.21%, GPT-5.4-mini 41.15%, Gemini-2.5-Flash-Lite 45.18%,
DeepSeek-V4-Flash 48.70%, and Llama-3.3-70B-Instruct 24.53%.
