# Gov-Mem-v4-Symbolic-dev6 Authorization Contract: Full Medical Evaluation

Date: 2026-08-17

This is a complete Medical-domain development validation result. It is not a
replacement for the frozen four-domain paper table.

## Protocol

- Dataset unit: all 21 Medical episodes, 579 checkpoints
- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Embedding provider/model: OpenLux, `text-embedding-3-small`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- Scheduling: 2 episode workers, one request in flight per worker
- OpenLux key pool discovered: 10; maximum active episode workers: 2
- Embedding request timeout: 90 seconds; retries remained enabled
- Runtime storage: local scratch; final artifacts published after completion
- Official judge coverage: 579/579; parse failures: 0/579
- Context audit coverage: 579/579

`MGS = U * (1-A) * (1-F)`.

## Official GateMem Metrics

| System | Checkpoints | U | A | F | MGS | OR |
|---|---:|---:|---:|---:|---:|---:|
| Gov-Mem-v4-Symbolic-dev6 authorization contract | 579 | 68.57% | 44.27% | 9.04% | **34.76%** | 10.95% |

Additional official values: action accuracy `70.64%`, utility accuracy before
action gating `70.00%`, privacy context leakage `63.02%`, and deletion context
leakage `26.55%`.

## Interpretation

The full Medical run is complete and internally consistent. The result should
be treated as an exploratory development signal: generation is stochastic and
this is not a paired causal ablation against another method. The embedding
model and timeout are recorded explicitly because `text-embedding-3-small`
had intermittent long responses during this run.

## Artifacts

- Suite manifest:
  `experiments/gatemem_suites/govmem_v4_symbolic_dev6_authorization_contract_medical_full579_20260817.json`
- Config:
  `configs/govmem_v4_symbolic_openlux_gpt4omini_embedding3small_dev6_authorization_contract.yaml`
- Output:
  `outputs/2026-08-17-Gov-Mem-v4-Symbolic-dev6-authorization-contract_medical_full579_embedding3small_timeout90/`
