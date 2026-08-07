# Gov-Mem v3 Strict GateMem Protocol, 40 Checkpoints

This result uses the GateMem paper-compatible evaluation protocol. It is not
combined with the earlier long-context, action-gated, or incomplete diagnostic
runs.

## Protocol

| Item | Setting |
|---|---|
| Checkpoints | 40 total, 10 per domain |
| Memory-system provider / base LLM | OpenLux / `gpt-4o-mini` |
| Memory-system temperature / output limit | `0.2` / `4096` |
| Stage 1 retrieval | GateMem RAG-Naive turn chunks, raw query, top-20 |
| Stage 1 embedding | OpenLux `text-embedding-3-small` |
| Stage 2 | Gov-Mem typed rerank over retrieved evidence only |
| Long-context transcript | Disabled |
| Official evaluator provider / LLM | OpenLux / `gpt-4o` |
| Official evaluator temperature / output limit | `0.0` / `4096` |
| `gate_by_action` | `false` |
| Gold feedback / experience bank | Disabled for runtime |
| API execution | Four domains and episode shards parallelized; one leased key per episode |

## Paper Metrics

`MGS = U * (1 - A) * (1 - F)`. The four-domain average is the arithmetic mean
of the four domain MGS values.

| Domain | Checkpoints | U | A | F | MGS | Action | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 10 | 50.00% | 0.00% | 0.00% | 50.00% | 70.00% | 0.00% |
| Office | 10 | 40.00% | 0.00% | 0.00% | 40.00% | 70.00% | 20.00% |
| Education | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 90.00% | 0.00% |
| Household | 10 | 33.33% | 0.00% | 0.00% | 33.33% | 80.00% | 33.33% |
| **Four-domain average MGS** | **40** | | | | **39.17%** | | |

## Context Audit

Prompt-context audit coverage is 40/40 (100%). Context leakage is reported
separately from the paper MGS: Medical privacy/deletion context leakage is
33.33%/33.33%, Office 0.00%/0.00%, Education 0.00%/0.00%, and Household
50.00%/0.00%. These values do not replace the paper's answer-level A/F.

The earlier interrupted run is excluded because an embedding request returned
HTTP 503 and did not produce a complete prediction set. The official judge was
then resumed from its partial ledger without regenerating predictions.
