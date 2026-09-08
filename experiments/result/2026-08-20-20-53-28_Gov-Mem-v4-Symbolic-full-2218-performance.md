# Gov-Mem-v4-Symbolic GateMem Full-Benchmark Performance

- Base LLM: OpenLux `gemini-2.5-flash-lite`
- Embedding: OpenLux `text-embedding-3-small`
- Official judge: OpenLux `gpt-4o`, temperature `0.0`
- Dataset: GateMem full benchmark, 2,218 checkpoints
- GateMem protocol: `gate_by_action=false`, gold feedback disabled, long-context ledger disabled
- MGS: `U * (1 - A) * (1 - F)`

| Domain | Checkpoints | U | A | F | MGS | Action Acc. | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 79.52% | 39.58% | 19.77% | 38.55% | 72.19% | 4.76% |
| Office | 547 | 69.48% | 5.26% | 3.60% | 63.45% | 83.91% | 11.04% |
| Education | 540 | 42.78% | 18.89% | 10.56% | 31.04% | 72.41% | 10.00% |
| Household | 552 | 54.35% | 27.72% | 4.35% | 37.58% | 70.47% | 12.50% |
| **Overall (pooled)** | **2,218** | **61.95%** | **23.38%** | **9.17%** | **43.11%** | **74.71%** | **9.34%** |

The overall U, A, F, Action Acc., and OR values are pooled by the corresponding
GateMem query counts. The overall MGS is computed from the pooled U/A/F values.
The four-domain arithmetic mean of domain MGS values is 42.65% and should not
be substituted for the pooled overall MGS above.

All 2,218 explanation records were present with `answer_unchanged=true` and
`scored_by_gatemem=false`; explanation fields are not included in U/A/F/MGS.
