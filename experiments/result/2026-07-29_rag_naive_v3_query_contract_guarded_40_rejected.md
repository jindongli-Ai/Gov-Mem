# Query-Contract Probe: Rejected 40-Checkpoint Variant

Date: 2026-07-29

This was a diagnostic probe, not a promoted framework version. It added one
question-only v2 `compile_query_contract` call for each mixed query and kept
the contract from adding slots when the rule contract already had two or more
slots. Stage 1 and the official evaluator were unchanged.

| Variant | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|
| Active projection + action boundary | 81.25% | 25.0% | 0.0% | 0.6094 | 6.25% |
| Query-contract guarded probe | 75.0% | 25.0% | 0.0% | 0.5625 | 6.25% |

The active pipeline has been restored to projection + action boundary. The
contract code and tests remain available, but it is not wired into
`RAGNaiveBackbone` until the extra-call variance is addressed with a more
stable paired protocol.

Output:

- `outputs/2026-07-29-rag_naive_v3_query_contract_guarded_40_gpt4omini`
