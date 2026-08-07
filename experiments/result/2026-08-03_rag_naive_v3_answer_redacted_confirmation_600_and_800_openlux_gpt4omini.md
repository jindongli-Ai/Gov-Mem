# Gov-Mem v3 Answer-Redacted Confirmation Boundary: 600 and 800 Checkpoints

## Configuration

- Memory-system provider/model: OpenLux, `gpt-4o-mini-2024-07-18`
- Stage 1 embedding provider/model: OpenLux, `text-embedding-3-small`
- Official evaluation provider/model: OpenLux, `gpt-4o`
- Stage 1 retrieval: frozen RAG-Naive, `top_k=20`
- Stage 2 change: narrow final-answer consistency guard for sensitive confirmation text
- 600 manifest: `experiments/gatemem_suites/rag_naive_v3_stage2_generalization_600_seed20260730.json`
- 600 selection: 150 new checkpoints per domain; no overlap with the earlier 200-checkpoint manifest
- Execution: domains and episode shards ran in parallel with isolated OpenLux keys

MGS is computed per domain as `U * (1 - A) * (1 - F)`, followed by the arithmetic mean of the four domain MGS values. `A` is answer-level privacy leakage and `F` is answer-level deletion/staleness leakage. Action and OR are reported for completeness and are not multiplied into MGS.

## 600-Checkpoint Held-Out Result

| Domain | Checkpoints | U | A | F | MGS | Action | OR | Privacy context | Deletion context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 150 | 55.56% | 36.17% | 8.16% | 32.57% | 70.67% | 20.37% | 0.00% | 0.00% |
| Office | 150 | 59.57% | 10.20% | 1.85% | 52.50% | 79.33% | 21.28% | 0.00% | 0.00% |
| Education | 150 | 41.30% | 19.23% | 7.69% | 30.79% | 76.00% | 13.04% | 0.00% | 0.00% |
| Household | 150 | 47.73% | 26.92% | 0.00% | 34.88% | 70.00% | 15.91% | 0.00% | 0.00% |
| **Four-domain average** | **600** | | | | **37.69%** | | | | |

## Combined 200+600 Result

The earlier 200 and new 600 manifests are disjoint. Their prediction rows and completed official `gpt-4o` judge ledgers were merged by `checkpoint_id`, producing 200 checkpoints per domain. The combined score below was recomputed over all 200 rows per domain; it is not the arithmetic mean of the separate 200 and 600 MGS values.

| Domain | Checkpoints | U | A | F | MGS | Action | OR | Privacy context | Deletion context |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 200 | 60.27% | 34.43% | 6.06% | 37.13% | 72.50% | 19.18% | 0.00% | 0.00% |
| Office | 200 | 63.16% | 8.96% | 1.32% | 56.75% | 80.50% | 17.54% | 0.00% | 0.00% |
| Education | 200 | 43.75% | 15.28% | 6.25% | 34.75% | 74.00% | 12.50% | 0.00% | 0.00% |
| Household | 200 | 49.12% | 26.32% | 0.00% | 36.20% | 68.00% | 14.04% | 0.00% | 0.00% |
| **Four-domain average** | **800** | | | | **41.20%** | | | | |

## Interpretation

- The 600 held-out result is substantially below the earlier 200-checkpoint result; the small 40/200 result did not generalize to this larger blind sample.
- Medical and Education are the main weaknesses on the new 600 checkpoints. Medical has the highest privacy and deletion leakage; Education has the lowest utility.
- The combined 800 result is more representative than the earlier 200 result, but it also shows that the current framework should remain frozen while the failure cases are analyzed.
- Context-level privacy and deletion exposure remained 0% in every reported domain. The main loss is answer-level utility and answer-level leakage after final answer generation.

## Artifacts

- 600 output: `outputs/2026-08-03-rag_naive_v3_answer_redacted_confirmation_600_openlux_gpt4omini_v1/`
- Combined output: `outputs/2026-08-03-rag_naive_v3_answer_redacted_confirmation_800_combined_openlux_gpt4omini_v1/`
- Combined scoring reused the 200 and 600 official judge ledgers; no additional judge calls were needed for the merged table.
