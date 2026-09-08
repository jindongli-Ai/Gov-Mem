# Gov-Mem-v4-Symbolic-dev3 State Ledger: Medical Probe 2

Date: 2026-08-15

This is a diagnostic official GateMem run after the Medical typed-slot ledger
fix. It is not a paper performance result and must not be merged into the
frozen 2,218-checkpoint table.

## Protocol

- Dataset unit: 2 complete Medical episodes, 55 checkpoints
- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Embedding: OpenLux, `text-embedding-3-small`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- Scheduling: 2 episode workers, one request in flight per worker
- API key isolation: one leased key per worker; no worker-internal parallelism
- Added Symbolic LLM calls: 0
- Prompt audit coverage: 55/55
- Official judge parse failures: 0/55

## Official GateMem Metrics

`U` is effective utility accuracy, `A` is answer-level privacy leakage, `F` is
answer-level deletion leakage, and `MGS = U * (1-A) * (1-F)`.

| Domain | Checkpoints | U | A | F | MGS | Over-refusal |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 55 | 55.00% | 61.11% | 5.88% | 20.13% | 15.00% |

The matching frozen `rag_naive_v3_typed_rerank` output on the same 55
checkpoints recorded U=55.00%, A=55.56%, F=5.88%, and MGS=23.01%.
Because the memory-system calls are stochastic and were executed separately,
this is a diagnostic comparison, not a causal ablation.

## Ledger Coverage Audit

The ledger block was present in the official answer context for 35/55 cases.
After parsing the exact ledger payload:

| Query type | N | Non-empty requested slots | Resolved at least one slot |
|---|---:|---:|---:|
| Utility | 20 | 15 | 9 |
| Privacy | 18 | 5 | 4 |
| Safety | 17 | 0 | 0 |
| **All** | **55** | **20** | **13** |

The ledger therefore does not yet cover Medical safety queries. Empty ledgers
should not be filled by guessing fields, especially for deletion and access
queries; the clinical result contract needs a separate explicit definition.

## Regression Signal

Compared with the matching baseline output, action predictions were unchanged.
The Symbolic answer prompt was nevertheless larger by an average of about
3,897 characters per checkpoint. The ledger currently exposes all distinct
retrieved candidate values and conflict counts, which can make the answer
context substantially longer without changing the action boundary. This is a
clear follow-up target: compress the ledger to source-bound winning values and
compact conflict metadata, and add safety coverage only after its slot contract
is specified and tested.

Raw output:
`outputs/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-medical-probe2_gpt4omini/`.
