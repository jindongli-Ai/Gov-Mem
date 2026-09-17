# GateMem official RAG-Naive: Gemini 2.5 Flash Lite, full 2,218 checkpoints

Date: 2026-09-17
Run directory: `experiments/runs/gatemem_official_rag_naive_gemini25flashlite_full_20260917`

## Question

GateMem's paper table reports RAG-Naive with Gemini-2.5-Flash-Lite at domain
MGS values of 10.7%, 9.1%, 9.2%, and 11.6%, whose arithmetic mean is 10.15%
(reported as 10.2%). This run tests whether that result is reproduced by the
current live model endpoint.

## Protocol

- Vendored GateMem official `NaiveRAGAgent`; no Gov-Mem Stage 2 or symbolic components.
- All 2,218 checkpoints: Medical 579, Office 547, Education 540, Household 552.
- Answer model: OpenLux `gemini-2.5-flash-lite`, temperature 0.2.
- Retrieval: official turn chunks, embedding retrieval, top-k 20.
- Embedding model: OpenLux `text-embedding-3-small`.
- Official judge: OpenLux `gpt-4o`, temperature 0.0.
- `gate_by_action=false`, matching the GateMem main paper protocol.
- 30 OpenLux keys split 8/8/7/7 across the four domains.
- Prediction completeness: 2,218/2,218.
- Judge completeness: 2,218/2,218; unique checkpoint IDs: 2,218.
- Judge structured-output parse failures: 0.

## Results

`MGS = U * (1 - A) * (1 - F)`. The headline average is the arithmetic mean
of the four domain MGS values, matching the GateMem table convention.

| Domain | U | A | F | Reproduced MGS | GateMem paper MGS | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 75.71% | 44.27% | 26.55% | 30.99% | 10.70% | +20.29 pp |
| Office | 69.48% | 46.20% | 45.50% | 20.37% | 9.10% | +11.27 pp |
| Education | 39.44% | 19.44% | 41.67% | 18.54% | 9.20% | +9.34 pp |
| Household | 54.89% | 26.09% | 32.07% | 27.56% | 11.60% | +15.96 pp |
| **Four-domain average** | **59.88%** | **34.00%** | **36.45%** | **24.37%** | **10.15%** | **+14.22 pp** |

The current run does not reproduce the paper's 10.2% average MGS. Utility is
close to the paper average (59.88% versus 58.40%), while access-control and
forgetting leakage are substantially lower (A: 34.00% versus 49.38%; F:
36.45% versus 62.75%). These leakage differences explain almost all of the
MGS gap.

The result uses a live model alias through OpenLux. It establishes the current
endpoint behavior, but it cannot identify whether the difference comes from
model-version drift, provider behavior, or other unreleased details of the
paper's original execution. Exact reproduction would require the immutable
model snapshot and provider used for the paper run.

## Gov-Mem comparison

The existing full Gov-Mem-v4-Symbolic run with the same Gemini alias reports
25.63% four-domain average MGS. Against this reproduced official RAG-Naive
baseline (24.37%), the absolute gain is 1.26 percentage points. The 10.2%
paper-reported baseline must not be presented as if it were this current local
reproduction.

## Artifacts

- Aggregate: `experiments/runs/gatemem_official_rag_naive_gemini25flashlite_full_20260917/official_score.json`
- Per-domain official summaries: `<run_dir>/<domain>/summary.json`
- Predictions: `<run_dir>/<domain>/predictions.jsonl`
- Judge rows: `<run_dir>/<domain>/judge_scores.jsonl`
- Reproduction config: `configs/gatemem_official_rag_naive_openlux_gemini25flashlite.yaml`
