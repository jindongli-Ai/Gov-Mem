# Gov-Mem v3 Safe Summary Delivery Boundary: 200-Checkpoint Validation

Date: 2026-07-29

## Experimental Settings

| Setting | Value |
|---|---|
| Checkpoints | 200, 50 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, unchanged |
| Stage 2 | Existing typed rerank, deletion gate, mixed projection, sensitive gate, plus bounded safe-summary delivery boundary |
| Parallelism | 4 domains, up to 8 episode workers per domain |
| Key isolation | 30-key pool, one leased key per episode |
| Manifest | `stateful_policy_generalization_200_seed20260727.json` |

`U` is effective Utility, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, `OR` is utility over-refusal, and
`MGS = U * (1-A) * (1-F)`.

## Official Metrics By Domain

| Domain | N | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 75.00% | 50.00% | 27.78% | 0.2708 | 20.00% | 68.00% |
| Office | 50 | 27.27% | 40.00% | 0.00% | 0.1636 | 63.64% | 74.00% |
| Education | 50 | 33.33% | 45.45% | 9.52% | 0.1645 | 50.00% | 64.00% |
| Household | 50 | 6.67% | 20.00% | 10.00% | 0.0480 | 33.33% | 60.00% |
| **Overall domain mean** | **4 domains** | **35.57%** | **38.86%** | **11.83%** | **0.1617** | **41.74%** | **66.50%** |

Overall values are arithmetic means across the four domain rows. MGS is also
the arithmetic mean of the four domain MGS values, as required by the project
reporting convention. All four domain prediction files and official judge
files contain exactly 50 rows.

## Comparison With Frozen 200

| Version | U | A | F | MGS | OR | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Frozen v3 sensitive gate v2 | 39.23% | 40.53% | 11.83% | 0.1762 | 44.75% | 65.50% |
| Safe-summary boundary v6 | 35.57% | 38.86% | 11.83% | 0.1617 | 41.74% | 66.50% |
| Change | -3.66 pp | -1.67 pp | 0.00 pp | -0.0144 | -3.01 pp | +1.00 pp |

## Interpretation

The delivery boundary triggered four times in this 200-checkpoint run, all
on explicit Household utility summaries. Household Utility remained 6.67%,
because the remaining failures are missing or incorrect logistics details,
not only the `answer_redacted` action. The boundary did not trigger on the
Office sponsor-safe privacy query after the privacy-shaped wording guard was
added.

The same 200-checkpoint manifest produced materially different results across
repeated real Yunwu runs, even with temperature zero. The corrected comparison
still shows the current run below the frozen version: Office MGS fell from
0.1939 to 0.1636 and Education MGS from 0.1919 to 0.1645, while Medical and
Household were unchanged. The safe-summary boundary triggered only on four
Household utility cases, so it cannot explain the Office/Education decline.
The implementation is therefore not promoted as a stable overall improvement;
the next step should first control evaluation variance and then address
Household content completeness. Stage 1 remains unchanged.

## Artifacts

- Suite summary: `outputs/2026-07-29-rag_naive_v3_sensitive_gate_v6_200_gpt4omini/suite_summary.json`
- 40-checkpoint result: `experiments/result/2026-07-29_rag_naive_v3_safe_summary_boundary_40_gpt4omini.md`
- Source code: `src/gov_mem/backbones/stage2_typed_rerank.py`
- Direct-answer integration: `src/gov_mem/backbones/rag_naive.py`
