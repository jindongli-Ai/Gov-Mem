# Gov-Mem V8 Graph-Enabled Full GateMem Evaluation

**Run date:** 2026-10-05  
**Scope:** 91 complete episodes / 2,218 checkpoints  
**Memory and answering model:** Gemini-2.5-Flash-Lite, temperature 0  
**Embedding model:** `text-embedding-3-small`  
**Official judge:** GPT-4o, temperature 0, `gate_by_action=false`  
**Runtime:** `0e202f6e990639ebb7881a4c867b9f91aada15d9a40a7e8836ab989a8414aa36`  
**Output:** `outputs/v8_graph_full2218_20261005`  

This is the first complete 2,218-checkpoint run of the current
Graph-enabled V8 source. The configuration has
`memory_governed_slot_graph.enabled: true`. The graph remains an auxiliary
observer/audit channel; Dense RAG Top-20 remains the primary retrieval path.

## Official results

| Domain | U (%) | A (%) | F (%) | MGS (%) | Execution errors |
|---|---:|---:|---:|---:|---:|
| Medical | 88.10 | 34.38 | 19.21 | 46.71 | 2 |
| Office | 82.47 | 66.67 | 22.07 | 21.42 | 0 |
| Education | 68.33 | 29.44 | 11.67 | 42.59 | 0 |
| Household | 56.52 | 25.54 | 14.67 | 35.91 | 1 |
| **Four-domain arithmetic mean** | — | — | — | **36.66** | **3** |

The pessimistic four-domain mean MGS, counting execution errors in the
benchmark's pessimistic treatment, is **36.48%**. The exact machine-readable
result is `outputs/v8_graph_full2218_20261005/full_metrics.json`.

## Execution and cost audit

- All 2,218 predictions and all 2,218 official judge scores were produced.
- Three execution errors were retained in the prediction records: two in
  Medical and one in Household.
- No HTTP failures or missing usage records were reported for Gemini,
  embeddings, or GPT-4o judge requests.
- Gemini requests: 7,291; reported tokens: 35,375,257.
- Embedding requests: 3,119; reported tokens: 407,099.
- Judge requests: 2,218; reported tokens: 1,817,538.

## Relation to the historical 30.02% result

The earlier **30.02%** result is a completed full-benchmark result for the
historical frozen pre-graph V8 runtime. It is not a paired graph-off arm of
this current runtime. The current graph-enabled result above must therefore be
reported as a separate runtime result; the 6.64 percentage-point difference is
descriptive across frozen versions and is not, by itself, a causal graph
ablation estimate.

