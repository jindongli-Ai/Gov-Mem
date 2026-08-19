# Gov-Mem-v4-Symbolic-dev7 Claim-Provenance Smoke

## Scope

- Date: 2026-08-19
- Framework: `Gov-Mem-v4-Symbolic-dev7`
- Benchmark path: actual `govmem_v4_symbolic` RAG-Naive path
- Dataset unit: one complete Medical episode, 28 checkpoints
- Base LLM: OpenLux `gpt-4o-mini`
- Embedding: OpenLux `text-embedding-3-small`
- Official judge: OpenLux `gpt-4o`
- Scheduling: one episode worker, one API key, judge concurrency 1
- Runtime storage: local `/tmp` scratch; completed output published afterward

## Purpose

This smoke validates that the newly integrated claim-level provenance verifier
is reached by the real RAG-Naive benchmark path and that its audit is exported
with each official prediction. It is an integration diagnostic, not a paper
performance result.

## Results

| Run | Claim enforcement | U | A | F | MGS | Action accuracy | Judge parse failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial integration | on | 0.00% | 33.33% | 0.00% | 0.00% | 53.57% | 0/28 |
| Final integration | non-intervention audit | 20.00% | 44.44% | 0.00% | 11.11% | 67.86% | 0/28 |

The initial hard-gate run is not a valid ablation: the model-generated claim
contract was incomplete or used paraphrases that were not literal source
spans, so the verifier correctly rejected some contracts but the adapter then
replaced the original answer with `no_memory`. This confounded contract
quality with benchmark answering behavior.

The final RAG-Naive configuration uses
`policy_verifier.claim_provenance_enforcement: false`. It performs a
non-intervention claim-level provenance audit: the original answer is preserved
while the verifier result is recorded. Across 28 predictions:

- 28/28 contained `policy_privacy_verifier` audit output.
- 17/28 had no model-emitted claim contract and were recorded as
  `claim_provenance_not_applicable`.
- 11/28 emitted a claim contract; 10 failed at least one source/value check.
- 18/28 audits passed, consisting of the 17 not-applicable audits plus one
  contract with no deterministic claims. Only one contract carried claims that
  passed all checks.
- No additional LLM call was introduced by the verifier.

## Decision

Keep the verifier integrated in non-intervention recording mode.
Do not report either smoke score in the paper and do not enable hard
enforcement on the free-form RAG-Naive path yet. The evidence shows that the
remaining issue is contract realization: the answering model must emit
complete, source-literal claim values and source bindings. The stateful
executor path continues to use its full field-state projection and fail-closed
verifier. A later enforcement experiment should be started only after a
contract-quality test on held-out episodes, followed by official GateMem
evaluation.

Artifacts:

- `experiments/smoke/2026-08-19_dev7_claim_provenance_integrated_medical_episode002`
- `experiments/smoke/2026-08-19_dev7_claim_provenance_shadow_medical_episode002`
- `experiments/gatemem_suites/govmem_v4_symbolic_dev7_claim_provenance_medical_episode002_20260819.json`

## Source-span completion validation

On 2026-08-19, the same complete Medical episode was rerun through the actual
pipeline with one OpenLux memory-system key, one episode worker, and one
official judge request in flight. All 28 predictions and all 28 official judge
records completed with zero judge parse failures. The official diagnostic
metrics were U 30.00%, A 44.44%, F 0.00%, and MGS 16.67%; these values are not
paper performance results.

The verifier audit was present in 28/28 predictions. The model supplied a
claim contract with fields in 9/28 predictions; the remaining 19 were recorded
as `claim_provenance_not_applicable`. No additional LLM call was introduced by
the verifier. This run validates pipeline integration and source bookkeeping,
not a causal performance gain. The complete artifact is
`experiments/smoke/2026-08-19_dev7_claim_provenance_source_completion_medical_episode002`.
