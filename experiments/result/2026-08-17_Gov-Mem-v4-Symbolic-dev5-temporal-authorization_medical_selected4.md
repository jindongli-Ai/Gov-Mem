# Gov-Mem-v4-Symbolic-dev5 Temporal Authorization: Four Medical Episodes

Date: 2026-08-17

This is a development diagnostic for the lightweight temporal authorization
state graph. It is not a paper-table result and must not be mixed into the
full four-domain performance table.

## Protocol

- Dataset unit: 4 complete Medical episodes, 108 checkpoints
- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Embedding: OpenLux, `text-embedding-3-small`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- Scheduling: 4 episode workers, one request in flight per worker
- Discovered OpenLux key pool: 10; one key isolated per worker
- Added Symbolic LLM calls: 0
- Temporal authorization graph: enabled, `enforcement=false`
- Official judge: 108/108 scored, parse failures 0/108
- Context audit coverage: 108/108

## Official GateMem Metrics

`U` is effective utility accuracy, `A` is answer-level privacy leakage, `F` is
answer-level deletion/staleness leakage, and `MGS = U * (1-A) * (1-F)`.

| System | Checkpoints | U | A | F | MGS | Action Acc. | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gov-Mem-v4-Symbolic-dev5 temporal authorization | 108 | 82.05% | 32.43% | 12.50% | **48.51%** | 72.22% | 10.26% |
| Earlier dev3 state-ledger diagnostic | 108 | 71.79% | 37.84% | 12.50% | 39.05% | not comparable | not recorded |
| Diagnostic delta | | +10.26 pp | -5.41 pp | +0.00 pp | **+9.46 pp** | | |

## Additional Safety Signals

| Signal | Rate |
|---|---:|
| Privacy context leakage | 51.35% |
| Deletion context leakage | 28.13% |
| Privacy end-to-end leakage | 51.35% |
| Deletion end-to-end leakage | 28.13% |
| Official judge parse failure | 0/108 |

The answer-level MGS signal is positive on this subset, but the context
leakage rates show that the current certificate is still a shadow reasoning
artifact rather than an enforced access-control boundary. The comparison is
not causal: the runs are separate stochastic executions, and this run also
experienced intermittent embedding timeouts with successful retry/resume.

## Architectural Audit

- Candidate count/order and retrieval filtering were unchanged.
- The temporal graph used typed `Principal`, `Role`, `Resource`, and
  `PolicyEvent` nodes with provenance-bound temporal keys.
- Same-time opposing authorization effects resolve to `unknown`; missing time
  or source provenance is conservative `unknown`; future events are ignored.
- The certificate was auxiliary context only: `enforcement_applied=false` and
  `new_llm_calls=0`.

## Interpretation

Keep dev5 as a validation candidate, not as a promoted final framework. Before
adding another enforcement mechanism, audit authorization-event coverage on
all four domains. The next defensible improvement is to preserve and consume
provider-neutral structured authorization attributes at ingestion, reducing
dependence on conservative natural-language parsing without introducing
dataset-specific branches or keyword lists.

## Artifacts

- Suite summary:
  `outputs/2026-08-16-govmem_v4_symbolic_dev5_temporal_authorization_medical_selected4_openlux_gpt4omini/suite_summary.json`
- Medical summary:
  `outputs/2026-08-16-govmem_v4_symbolic_dev5_temporal_authorization_medical_selected4_openlux_gpt4omini/medical/official_eval/checkpoint_benchmark/medical/summary.json`
- Paper metrics:
  `outputs/2026-08-16-govmem_v4_symbolic_dev5_temporal_authorization_medical_selected4_openlux_gpt4omini/medical/official_eval/checkpoint_benchmark/medical/paper_metrics.json`
- Suite manifest:
  `experiments/gatemem_suites/govmem_v4_symbolic_dev5_temporal_authorization_medical_selected4_20260816.json`
