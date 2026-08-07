# Gov-Mem v3 Stage 2B Reasoning Rerank: Held-Out 40 Diagnostic

Date: 2026-07-29

## Settings

| Setting | Value |
|---|---|
| Checkpoints | 40 new held-out checkpoints, 10 per domain |
| Memory system base LLM | `gpt-4o-mini-2024-07-18` |
| Official evaluation LLM | `gpt-4o` |
| Provider | Yunwu |
| Embedding | `text-embedding-3-small` |
| Stage 1 | Official-compatible RAG-Naive retrieval, unchanged |
| OFF | Existing Stage 2; reasoning rerank disabled |
| ON | Stage 2B mixed-query reasoning rerank enabled; long-context disabled |
| Parallelism | 4 domains, 5 episode workers per domain |
| Key isolation | One leased Yunwu key per episode; suite discovered 30 keys |
| Manifest | `experiments/gatemem_suites/rag_naive_v3_reasoning_rerank_heldout_40_seed20260730.json` |

`MGS = U * (1-A) * (1-F)`. Overall MGS is the arithmetic mean of the four
domain MGS values.

## Results

### Reasoning OFF

| Domain | N | U | A | F | OR | MGS |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 33.33% | 33.33% | 50.00% | 0.2222 |
| Office | 10 | 0.00% | 80.00% | 0.00% | 100.00% | 0.0000 |
| Education | 10 | 0.00% | 0.00% | 40.00% | 33.33% | 0.0000 |
| Household | 10 | 0.00% | 0.00% | 25.00% | 50.00% | 0.0000 |
| **Overall** | **40** | | | | | **0.0556** |

### Reasoning ON

| Domain | N | U | A | F | OR | MGS |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 33.33% | 33.33% | 50.00% | 0.2222 |
| Office | 10 | 0.00% | 80.00% | 0.00% | 100.00% | 0.0000 |
| Education | 10 | 0.00% | 0.00% | 40.00% | 33.33% | 0.0000 |
| Household | 10 | 0.00% | 0.00% | 25.00% | 50.00% | 0.0000 |
| **Overall** | **40** | | | | | **0.0556** |

## Stage 2B Audit

- The 40-case manifest contained 6 `mixed` queries.
- Reasoning ON made 6 mixed-query calls.
- Only `1/6` passed Stage 2C validation.
- The one accepted result selected the same single candidate already preferred by
  the existing evidence order; it did not produce a meaningful rerank change.
- Four calls failed because the validator could not prove field coverage from
  lexical slot matching; one failed because the model returned invalid candidate
  ids.
- Deleted/historical and explicit sensitive cases were blocked before the
  reasoning call.
- The default config remains unchanged with reasoning rerank disabled.

## Decision

Do not run the 200-checkpoint promotion experiment yet. This run is a useful
implementation diagnostic, but it has almost no effective Stage 2B treatment
coverage. The next small change should make Stage 2C validate the LLM's
`field_support` mapping against the closed candidate set and exact source quotes,
instead of requiring the existing lexical slot matcher to independently infer
semantic coverage. This preserves the neuro-symbolic boundary:

- symbolic checks validate candidate ids, exact quotes, requested fields, and
  hard safety gates;
- the base LLM supplies semantic field-to-candidate reasoning.

After that narrow validator change, repeat the same 40-case diagnostic. Only if
reasoning activation and valid reranking increase without A/F regression should
we run a new held-out 200-case evaluation.

## Artifacts

- OFF summary: `outputs/2026-07-29-rag_naive_v3_reasoning_rerank_heldout_off_40/suite_summary.json`
- ON summary: `outputs/2026-07-29-rag_naive_v3_reasoning_rerank_heldout_on_40/suite_summary.json`
- Candidate config: `configs/rag_naive_v3_gpt4omini_reasoning_rerank_enabled.yaml`
- Default config: `configs/rag_naive_v3_gpt4omini.yaml`
