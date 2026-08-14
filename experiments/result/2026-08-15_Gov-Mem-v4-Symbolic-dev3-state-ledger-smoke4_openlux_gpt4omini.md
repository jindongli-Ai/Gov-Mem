# Gov-Mem-v4-Symbolic-dev3 State Ledger Validation

Date: 2026-08-15

This is a controlled engineering diagnostic, not a full GateMem paper result.
It runs the current `Gov-Mem-v4-Symbolic-dev3-state-ledger` implementation on
one complete episode in each GateMem domain (101 checkpoints total).

## Protocol

- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Embedding: OpenLux, `text-embedding-3-small`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- API scheduling: 30-key pool, four episode workers, one request in flight per worker
- Dataset unit: complete selected episodes, not a checkpoint-only sample
- Episodes: Medical 27, Office 32, Education 18, Household 24 checkpoints
- New LLM calls from the state ledger: 0
- Prompt audit coverage: 101/101
- Ledger present in state-relevant prompts: 46/101
- Certificate/target-binding shadow fields in prompts: 0/101

## Official GateMem Metrics

`U` is utility accuracy, `A` is answer-level privacy leakage, `F` is
answer-level deletion leakage, and `MGS = U * (1-A) * (1-F)`.

| Domain | Checkpoints | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 27 | 60.00% | 55.56% | 0.00% | 26.67% | 59.26% |
| Office | 32 | 55.56% | 0.00% | 0.00% | 55.56% | 78.13% |
| Education | 18 | 83.33% | 16.67% | 0.00% | 69.44% | 88.89% |
| Household | 24 | 75.00% | 12.50% | 0.00% | 65.63% | 75.00% |
| **All scored cases** | **101** | **66.67%** | **21.21%** | **0.00%** | **52.53%** | **74.26%** |

The all-cases row is query-count weighted: 33 utility, 33 privacy, and 35
safety cases. The arithmetic mean of the four domain MGS values is 54.32%.

## Interpretation

On the same 101-checkpoint diagnostic, the preceding lifecycle smoke measured
U=63.64%, A=24.24%, F=0.00%, and weighted MGS=48.21%. The state-ledger run is
therefore a positive engineering signal: U +3.03 points, A -3.03 points, and
MGS +4.32 points. This is not a causal ablation because the memory-model
temperature is 0.2 and the comparison is a single stochastic run. It must not
be merged into the frozen 2,218-checkpoint `rag_naive_v3_typed_rerank` table.

Raw machine-readable outputs are stored in the local ignored directory:
`outputs/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-smoke4_gpt4omini_embedding3small/`.
