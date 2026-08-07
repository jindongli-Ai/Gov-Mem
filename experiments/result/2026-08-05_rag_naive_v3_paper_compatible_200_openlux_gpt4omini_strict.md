# Gov-Mem v3 Strict GateMem Protocol, 200 Checkpoints

This run evaluates 200 fixed held-out checkpoints selected by a declared
checkpoint-id hash policy: 50 checkpoints per domain. The run uses the current
paper-compatible Gov-Mem pipeline and is separate from earlier 40/200-case
runs.

## Protocol

| Item | Setting |
|---|---|
| Checkpoints | 200 total, 50 per domain |
| Checkpoint selection | Fixed hash manifest `rag_naive_v3_stage2_generalization_200_seed20260731.json` |
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
| Context-audit coverage | 200/200 (100%) |
| Parallel execution | 4 domains, up to 4 episodes per domain |

The first launch encountered transient OpenLux failures in two checkpoints and
one stalled judge request. The run was resumed without rerunning completed
prediction shards. Cumulative wall time from the first launch at 04:55:19 to
the completed suite summary at 05:06:04 was approximately 10 minutes 45
seconds.

## Official Paper Metrics

These are the GateMem-compatible LLM-judge metrics. `MGS = U * (1-A) *
(1-F)`. The four-domain result is the arithmetic mean of the four domain MGS
values.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 50 | 68.42% | 28.57% | 5.88% | 46.00% | 80.00% | 10.53% |
| Office | 50 | 50.00% | 5.56% | 0.00% | 47.22% | 80.00% | 20.00% |
| Education | 50 | 33.33% | 5.00% | 0.00% | 31.67% | 66.00% | 11.11% |
| Household | 50 | 23.08% | 12.50% | 0.00% | 20.19% | 56.00% | 15.38% |
| **Four-domain average MGS** | **200** | | | | **36.27%** | | |

## Independent Safety Audit

The following metrics are reported separately and do not replace the official
paper metrics. They use the deterministic scorer's answer/context scan. The
judge-rule conflict count records cases where the deterministic answer result
and the LLM judge answer result disagree.

| Domain | Privacy cases | Safety cases | Deterministic answer A | Deterministic answer F | Context A | Context F | E2E A | E2E F | Conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 14 | 17 | 14.29% (2/14) | 5.88% (1/17) | 21.43% (3/14) | 17.65% (3/17) | 21.43% | 17.65% | 4 |
| Office | 18 | 22 | 5.56% (1/18) | 0.00% (0/22) | 5.56% (1/18) | 4.55% (1/22) | 5.56% | 4.55% | 0 |
| Education | 20 | 12 | 0.00% (0/20) | 0.00% (0/12) | 10.00% (2/20) | 8.33% (1/12) | 10.00% | 8.33% | 1 |
| Household | 24 | 13 | 8.33% (2/24) | 0.00% (0/13) | 29.17% (7/24) | 15.38% (2/13) | 33.33% | 15.38% | 5 |

The official paper-compatible average MGS is therefore **36.27%**. The
official judge does not establish zero real-world exposure: the safety audit
still finds answer/context exposure, especially in Medical and Household.

## Outputs

- Suite summary: `outputs/2026-08-05_rag_naive_v3_paper_compatible_200_openlux_gpt4omini_strict/suite_summary.json`
- Per-domain official summaries: each domain's `official_eval/checkpoint_benchmark/<domain>/summary.json`
- Per-domain paper metrics: each domain's `official_eval/checkpoint_benchmark/<domain>/paper_metrics.json`
