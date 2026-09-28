# Gov-Mem Governed Slot Graph Confirmation Ablation (Corrected)

> The first scoring attempt for this directory was invalid: both source
> configurations had `enabled: true`. It is retained as an audit artifact and
> is not a graph-on/off result. The valid experiment below uses the corrected
> graph-disabled configuration.

## Protocol

This is a paired confirmation run on the reserved 12-episode selection from
`v8_reserved_confirmation_episodes_20260919.json`: 306 checkpoints per arm,
four domains, three complete episodes per domain. The selection has historical
exposure and is not a pristine holdout. Graph-on and graph-off used the same
manifest and current runtime; graph-off removes the governed slot graph
advisory while retaining the rest of the Gov-Mem pipeline. Official scoring
used the GateMem GPT-4o judge with `gate_by_action=false`. MGS is the only
primary metric; U/A/F are diagnostic components.

## Results

| Arm | Medical MGS | Office MGS | Education MGS | Household MGS | Four-domain avg.MGS |
|---|---:|---:|---:|---:|---:|
| Graph-on | 34.98% | 14.14% | 2.40% | 17.36% | **17.22%** |
| Graph-off | 23.89% | 14.21% | 2.19% | 22.01% | **15.57%** |
| On - off | +11.09 pp | -0.07 pp | +0.21 pp | -4.65 pp | **+1.64 pp** |

Both arms covered all 306 predictions and all 303 official judge cases (100
utility, 100 privacy, 106 safety). No judge parse failures occurred. Graph-on
had one `audit_incomplete` record and graph-off had two, each caused by an
invalid answering response;
these were recorded as protocol/information failures, not network or API
transport failures. There were zero ReadTimeout, ConnectionError, non-200, or
process-level execution errors.

## Interpretation

The earlier 12-episode random paired development run measured Graph-on at
18.41% versus Graph-off at 17.41% (+1.00 pp). The corrected confirmation run
measures +1.64 pp. These two paired runs provide preliminary positive evidence,
but are still development/exposure data and do not establish a universal gain.
The result supports the intended lightweight role: the graph can provide audit
context in some episodes without replacing dense RAG. The paper should describe
Governed Slot Graph as an advisory permission-audit component whose findings
are injected into reasoning only when evidence is available; missing graph
evidence remains an audit gap.

The graph-on arm improved Medical and Office in this confirmation sample, but
lost in Education and Household. Since all selected episodes have historical
exposure, this is a confirmation/diagnostic result rather than an independent
generalization estimate. Any final full evaluation must use one frozen paired
protocol and report the four-domain arithmetic mean MGS without selecting
episodes by outcome.

## Reproduction

- Graph-on raw output: `outputs/v8_reserved_confirmation_graph_on_20250925`
- Valid graph-off raw output: `outputs/v8_reserved_confirmation_graph_off_true_20250925`
- Scored paired output: `outputs/v8_reserved_confirmation_graph_true_ablation_scored_20250925`
- Machine-readable metrics: `outputs/v8_reserved_confirmation_graph_true_ablation_scored_20250925/paired_metrics.json`
- Scoring entry point: `scripts/score_v8_paired_episode_suite.py`
