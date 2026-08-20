# Gov-Mem-v4-Symbolic-dev7 Full GateMem Evaluation

Date: 2026-08-20
Benchmark: GateMem full benchmark, 2,218 checkpoints
Framework: `Gov-Mem-v4-Symbolic-dev7`
Foundation: `rag_naive_v3_typed_rerank`
Experiment mode: `govmem_v4_symbolic`

## Protocol

| Item | Setting |
|---|---|
| Checkpoints | Medical 579, Office 547, Education 540, Household 552; 2,218 total |
| Memory-system provider / base LLM | OpenLux / `gpt-4o-mini` |
| Embedding | OpenLux / `text-embedding-3-small` |
| Stage 1 | GateMem-compatible RAG-Naive retrieval, raw query, `top_k=20` |
| Stage 2 | Typed constrained reranking over retrieved evidence only |
| Symbolic modules | Typed provenance, role consistency, evidence graph, lifecycle/state ledger, temporal authorization boundary |
| Claim-level explanation | Enabled; non-intervention; no extra LLM call; excluded from metrics |
| Experience / skill updates | Disabled |
| Gold feedback | Disabled; clean benchmark |
| Long-context ledger | Disabled |
| Official judge | OpenLux / `gpt-4o`, temperature 0, `gate_by_action=false` |
| Judge parse failures | 0% in all four domains |
| Context audit coverage | 100% in all four domains |

The official metrics use the GateMem paper protocol:

```text
U = effective utility accuracy
A = answer-level privacy leakage rate
F = answer-level deletion/staleness leakage rate
MGS = U * (1 - A) * (1 - F)
```

## Official Paper Metrics

| Domain | Checkpoints | U | A | F | MGS | Action accuracy | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 74.29% | 46.35% | 10.17% | **35.80%** | 70.47% | 10.48% |
| Office | 547 | 39.61% | 4.09% | 1.80% | **37.30%** | 75.69% | 39.61% |
| Education | 540 | 40.00% | 18.89% | 7.22% | **30.10%** | 71.85% | 17.78% |
| Household | 552 | 43.48% | 27.17% | 2.72% | **30.80%** | 70.11% | 19.02% |
| **Four-domain average / pooled U-A-F** | **2,218** | **49.72%** | **24.47%** | **5.53%** | **33.50%** | | |

The reported overall MGS is the arithmetic mean of the four domain MGS values.
The overall U, A, and F shown in the last row are checkpoint-count-weighted
pooled values and are shown separately to avoid mixing aggregation rules.
`OR` is over-refusal and is reported for completeness; it is not multiplied
into MGS.

## Explanation and Audit Coverage

All 2,218 official predictions contain the dev7 explanation record. Every record
has `answer_unchanged=true` and `scored_by_gatemem=false`. The record contains
the selected evidence references, Stage 2 route, role consistency result,
lifecycle and state-ledger summaries, temporal authorization decision, and
claim-level provenance status when a source-bound claim contract is available.
Missing claim contracts are recorded as incomplete; no source is synthesized.

The explanation is written into the official prediction's `memory_audit` and
does not alter the answer, action, context, or official metrics. The official
context audit is a separate measurement: it reports what evidence was exposed
to the answer model and must not be conflated with answer-level A/F.

## Artifacts

- Output directory: `outputs/govmem_v4_symbolic_dev7_full_all_2218_20260820`
- Manifest: `experiments/gatemem_suites/rag_naive_v3_full_all_2218_seed20260803.json`
- Config: `configs/govmem_v4_symbolic_openlux_gpt4omini_embedding3small_dev7_authorization_boundary.yaml`
- Framework log: `VERSION_LOG.md`

This is the first complete four-domain performance measurement for dev7. It is
not a causal ablation against the frozen v3 table because the framework and
runtime behavior changed; the paper should state the exact protocol and use a
paired ablation if it claims that a specific Symbolic component caused a gain.
