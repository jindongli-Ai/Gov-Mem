# Gov-Mem v3 Safe Summary Delivery Boundary: 40-Checkpoint Result

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 40, 10 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, unchanged |
| Stage 2 | Existing typed rerank, deletion gate, mixed projection, sensitive gate, plus bounded safe-summary delivery boundary |
| Parallelism | 4 domains, up to 8 episode workers per domain |
| Key isolation | 30-key pool, one leased key per episode |
| Suite | `experiments/gatemem_suites/gatemem_smoke40.json` |

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`.

## Official Metrics

| Domain | N | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 0.00% | 0.00% | 0.5000 | 25.00% | 90.00% |
| Office | 10 | 75.00% | 0.00% | 0.00% | 0.7500 | 0.00% | 90.00% |
| Education | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% | 100.00% |
| Household | 10 | 100.00% | 0.00% | 0.00% | 1.0000 | 0.00% | 100.00% |
| **Overall pooled** | **40** | **81.25%** | **0.00%** | **0.00%** | **0.8125** | **6.25%** | **95.00%** |

Overall denominators: 16 utility cases, 12 privacy cases, and 12 safety cases.

## Comparison With Previous 40

| Version | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Previous v2 | 75.00% | 16.67% | 0.00% | 0.6250 | 6.25% | 90.00% |
| Safe-summary boundary | 81.25% | 0.00% | 0.00% | 0.8125 | 6.25% | 95.00% |
| Change | +6.25 pp | -16.67 pp | 0.00 pp | +0.1875 | 0.00 pp | +5.00 pp |

## Interpretation

The change is limited to delivery action normalization. When the question
explicitly requests a helper, logistics, signoff-safe, or household-safe
summary and the answer contains substantive content, an unnecessary
`answer_redacted` is changed to `answer`. Exact snapshot/credential requests,
explicit policy or deletion requests, and empty refusal text remain unchanged.

The 40-case result supports advancing this small change to a held-out
200-checkpoint validation. It does not justify changing Stage 1 retrieval or
adding another general-purpose LLM reflection call.

## Artifacts

- Suite summary: `outputs/2026-07-29-rag_naive_v3_sensitive_gate_v5_40_gpt4omini/suite_summary.json`
- Source code: `src/gov_mem/backbones/stage2_typed_rerank.py`
- Direct-answer integration: `src/gov_mem/backbones/rag_naive.py`
