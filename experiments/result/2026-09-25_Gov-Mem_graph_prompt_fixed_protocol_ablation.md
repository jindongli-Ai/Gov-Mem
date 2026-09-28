# Gov-Mem Graph Prompt Advisory: Fixed-Protocol Paired Ablation

This run uses the fixed manifest `v8_paired_random3_per_domain_seed20260918.json`
(12 complete episodes, 303 checkpoints) with the same latest runtime for both
arms. Graph-on and Graph-off used 12 distinct OpenLux keys each, with disjoint
key ranges and sequential history within each episode. Official scoring used
the unchanged GPT-4o judge with `gate_by_action=false`.

## Main result

| Arm | Medical MGS | Office MGS | Education MGS | Household MGS | Four-domain avg.MGS |
|---|---:|---:|---:|---:|---:|
| Graph-on | 8.83% | 13.16% | 17.28% | 34.38% | **18.41%** |
| Graph-off | 10.79% | 11.54% | 7.72% | 39.60% | **17.41%** |
| Graph-on minus Graph-off | -1.96 | +1.62 | +9.56 | -5.22 | **+1.00 pp** |

The graph prompt advisory therefore has a small positive average MGS effect in
this paired development ablation. The direction is not uniform by domain, so
this is evidence of a modest assistive signal, not a claim of universal graph
improvement. Both arms are below the paired Naive RAG reference (19.75%); the
graph ablation measures the incremental advisory effect, not overall Gov-Mem
quality.

## Execution and audit status

Both arms covered all 303 checkpoints. Neither arm had a transport/API error or
non-200 telemetry request. Each arm had one `audit_incomplete` checkpoint,
meaning the model response lacked enough grounding/contract information for a
complete governance audit. These cases were fail-closed and recorded separately;
they are not counted as execution errors. Official `execution_errors` is 0/303
for both arms.

Outputs:

- Graph-on: `outputs/v8_graph_prompt_random3_final_fixed_scored_20250925/paired_metrics.json`
- Graph-off: `outputs/v8_graph_prompt_random3_final_fixed_no_graph_scored_20250925/paired_metrics.json`
- Graph-on predictions: `outputs/v8_graph_prompt_random3_final_fixed_20250925`
- Graph-off predictions: `outputs/v8_graph_prompt_random3_final_fixed_no_graph_20250925`

This is a repeatedly exposed development sample, not an independent holdout.
