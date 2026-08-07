# Gov-Mem v3 Stage 2B/2C Combined Held-Out Validation

Date: 2026-07-30

## Settings

| Setting | Value |
|---|---|
| Checkpoints | 40 fresh held-out checkpoints, 10 per domain |
| Stage 1 | Official-compatible RAG-Naive retrieval, frozen |
| Stage 2B | Constrained base-LLM reasoning rerank enabled |
| Stage 2C | Long-context verified field ledger enabled |
| Memory-system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu for all API calls |
| Embedding | `text-embedding-3-small` |
| Parallelism | 4 domains, up to 6 episode workers per domain |
| Key isolation | 30-key pool, one leased key per episode |
| Manifest | `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_40_seed20260731.json` |
| Output | `outputs/2026-07-30-rag_naive_v3_stage2_generalization_40_seed20260731_combo_v2/` |

`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values. Overall U/A/F below are macro averages across domains and
are shown for completeness; they are not used to replace the required MGS
calculation.

## Results

| Domain | N | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Education | 10 | 66.67% | 0.00% | 0.00% | 66.67% |
| Household | 10 | 33.33% | 0.00% | 0.00% | 33.33% |
| Medical | 10 | 75.00% | 66.67% | 0.00% | 25.00% |
| Office | 10 | 40.00% | 0.00% | 0.00% | 40.00% |
| **Overall** | **40** | **53.75%** | **16.67%** | **0.00%** | **41.25%** |

## Interpretation

The combined switch is operational and preserves zero deletion leakage on this
held-out set, but it is not yet a promotion candidate. Household remains a
utility bottleneck, while Medical has a high access-leakage rate despite good
utility. The result indicates that enabling long-context field recovery for
all reasoning-rerank cases is too broad: the two Stage 2 mechanisms can interact
in ways that improve field completeness in some domains but weaken the hard
privacy boundary in others.

No Stage 1 change was made in this run. No code change should be selected from
this aggregate result alone. The next diagnostic should inspect the failed
Medical privacy cases and Household utility cases, then test one narrow branch
or gate on a small targeted set before another held-out run.
