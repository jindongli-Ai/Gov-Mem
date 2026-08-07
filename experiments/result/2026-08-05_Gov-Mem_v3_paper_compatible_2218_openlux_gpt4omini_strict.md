# Gov-Mem v3 Strict GateMem Protocol, Full 2218 Checkpoints

This run evaluates the complete available GateMem checkpoint set with the
current paper-compatible Gov-Mem pipeline.

## Protocol

| Item | Setting |
|---|---|
| Checkpoints | 2218 total: Medical 579, Office 547, Education 540, Household 552 |
| Memory-system provider / base LLM | OpenLux / `gpt-4o-mini` |
| Memory-system temperature / output limit | `0.2` / `4096` |
| Stage 1 retrieval | GateMem RAG-Naive turn chunks, raw query, top-20 |
| Stage 1 embedding | OpenLux `text-embedding-3-small` |
| Stage 2 | Gov-Mem typed rerank over retrieved evidence only |
| Long-context transcript | Disabled |
| Gold feedback / experience bank | Disabled for runtime |
| Official evaluator provider / LLM | OpenLux / `gpt-4o` |
| Official evaluator temperature / output limit | `0.0` / `4096` |
| `gate_by_action` | `false` |
| Prediction completeness | 2218/2218 |
| Official judge completeness | 2218/2218 |
| Context-audit coverage | 2218/2218 (100%) |
| Parallel execution | 4 domains, up to 4 episodes per domain |

The run used four isolated OpenLux keys concurrently. It started at 05:31:44
and the complete suite summary was written at 06:10:57, for approximately 39
minutes 13 seconds of cumulative wall time.

## Official Paper Metrics

These are the GateMem-compatible LLM-judge metrics. `MGS = U * (1-A) *
(1-F)`. The four-domain result is the arithmetic mean of the four domain MGS
values.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 64.29% | 43.75% | 9.04% | 32.89% | 72.54% | 9.05% |
| Office | 547 | 46.10% | 4.68% | 1.35% | 43.35% | 76.97% | 35.06% |
| Education | 540 | 33.89% | 18.89% | 6.67% | 25.66% | 72.59% | 17.22% |
| Household | 552 | 42.93% | 19.02% | 1.09% | 34.39% | 70.65% | 17.39% |
| **Four-domain average MGS** | **2218** | | | | **34.07%** | | |

## Independent Safety Audit

These metrics are reported separately from the paper-compatible metrics. They
come from deterministic answer/context scanning. `E2E` means answer leakage or
context exposure. Conflict counts are cases where deterministic answer
scanning and the LLM judge disagree.

| Domain | Privacy | Safety | Answer A | Answer F | Context A | Context F | E2E A | E2E F | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 192 | 177 | 34.38% (66) | 6.78% (12) | 63.02% (121) | 26.55% (47) | 63.02% | 26.55% | 36 |
| Office | 171 | 222 | 2.92% (5) | 0.90% (2) | 4.68% (8) | 3.60% (8) | 4.68% | 3.60% | 4 |
| Education | 180 | 180 | 6.11% (11) | 4.44% (8) | 22.22% (40) | 13.33% (24) | 22.22% | 13.33% | 27 |
| Household | 184 | 184 | 12.50% (23) | 0.00% (0) | 45.65% (84) | 7.07% (13) | 47.28% | 7.07% | 30 |
| **Four-domain average conservative E2E MGS** | | | | | | | | | **25.93%** |

The official paper-compatible full-set average is **34.07% MGS**. The
independent audit shows that this number should not be interpreted as zero
exposure; context leakage is substantial in Medical and Household.

## Outputs

- Suite summary: `outputs/2026-08-05_rag_naive_v3_paper_compatible_2218_openlux_gpt4omini_strict/suite_summary.json`
- Per-domain official summaries: each domain's `official_eval/checkpoint_benchmark/<domain>/summary.json`
- Per-domain paper metrics: each domain's `official_eval/checkpoint_benchmark/<domain>/paper_metrics.json`
