# Gov-Mem-v4-Symbolic-dev6 Authorization Contract: Four Medical Episodes

Date: 2026-08-17

This is an exploratory framework-validation result. It is not a paper-table
result and must not be merged with the clean `text-embedding-3-small` table.

## Design Increment

Dev6 adds a provider-neutral ingestion contract for explicit authorization
assertions. A visible turn may carry a structured assertion with:

- `effect`: allow, deny, or revoke;
- original subject and resource wording;
- event kind and source span.

Principal resolution is deferred to the episode roster. If the contract is
present but the subject cannot be resolved, the assertion remains unknown and
the implementation does not perform a broader fallback parse. Retrieval order,
candidate filtering, enforcement, and LLM call count are unchanged.

## Protocol

- Dataset unit: 4 complete Medical episodes, 108 checkpoints
- Memory-system provider/model: OpenLux, `gpt-4o-mini`
- Embedding: OpenLux, `text-embedding-3-large`
- Official judge: OpenLux, `gpt-4o`, `gate_by_action=false`
- Scheduling: 4 episode workers, one request in flight per worker
- Official judge: 108/108 scored, parse failures 0/108
- Added Symbolic LLM calls: 0
- Temporal authorization graph: enabled, `enforcement=false`
- First small-embedding attempt: incomplete after repeated read timeouts; no score used

## Official GateMem Metrics

`MGS = U * (1 - A) * (1 - F)`.

| System | Checkpoints | U | A | F | MGS | Action Acc. | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gov-Mem-v4-Symbolic-dev6 authorization contract | 108 | 87.18% | 35.14% | 9.38% | **51.25%** | 75.00% | 5.13% |

## Coverage Audit

- Runtime prompt audit: 23 retrieved structured records carried 23
  `authorization_assertions`.
- Temporal certificates were present in 66 checkpoints; 11 had a positive
  applied event count, for 11 applied events total.
- Context privacy leakage: 54.05%.
- Context deletion leakage: 28.13%.
- The graph remains shadow-only and did not enforce access decisions.

## Interpretation

The result confirms that explicit authorization semantics survive the RAG
boundary as typed provenance, without a domain or episode-specific branch.
It does not establish a causal performance improvement: the comparison dev5
run used `text-embedding-3-small`, and LLM executions are stochastic. The
`51.25%` MGS is therefore a diagnostic signal only. The next evaluation should
use a fixed embedding model and paired checkpoints before any paper claim.

## Artifacts

- Suite manifest:
  `experiments/gatemem_suites/govmem_v4_symbolic_dev6_authorization_contract_medical_selected4_20260817.json`
- Config:
  `configs/govmem_v4_symbolic_openlux_gpt4omini_embedding3small_dev6_authorization_contract.yaml`
- Output:
  `outputs/2026-08-17-govmem_v4_symbolic_dev6_authorization_contract_medical_selected4_openlux_gpt4omini_embedding3large/`
