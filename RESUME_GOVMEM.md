# Gov-Mem Recovery Guide

This file is the short recovery entry point for the current project. It is
kept concise so that stale historical run instructions do not look active.

## Read First

1. `AGENTS.md` for storage, experiment, key, and provenance constraints.
2. `handoff.md` for the current research state and next decisions.
3. `README.md` for the current system summary and result index.
4. `VERSION_LOG.md` for immutable historical version identity.

## Current Canonical Pipeline

```text
observable memory prefix
  -> dense Naive-RAG Top-20
  -> optional Governed Slot Graph observer
  -> LLM permission/lifecycle and claim reasoning
  -> source and binding validation
  -> symbolic/neuro-symbolic critic
  -> safe evidence
  -> answer agent
```

The graph is advisory. It does not replace dense retrieval or directly grant
or deny access. It contributes grounded audit findings only when available.
Unknown or missing graph information is an audit gap, not an authorization
decision.

## Latest Valid Graph Ablation

The corrected paired confirmation run uses 12 complete episodes and 306
checkpoints per arm. It is historically exposed development/confirmation data,
not a pristine holdout.

| Arm | Four-domain average MGS |
|---|---:|
| Graph-on | 17.22% |
| Graph-off | 15.57% |
| Difference | +1.64 percentage points |

Raw outputs:

- `outputs/v8_reserved_confirmation_graph_on_20250925`
- `outputs/v8_reserved_confirmation_graph_off_true_20250925`
- `outputs/v8_reserved_confirmation_graph_true_ablation_scored_20250925`

Report:
`experiments/result/2026-09-25_Gov-Mem_confirmation_graph_prompt_ablation.md`

The earlier directory `v8_reserved_confirmation_graph_off_20250925` is not a
valid graph-off arm because its frozen configuration had graph enabled. Keep it
for audit; do not cite its score.

## Experiment Rules

- Use `/mnt/data_disk_2` for all code, caches, outputs, and temporary files.
- Use complete episodes and preserve sequential history within each episode.
- Verify actual distinct API-key and worker counts at launch; never print keys.
- Retry transport/API failures by complete episode, preserving the failed run.
- Treat information/protocol insufficiency separately from transport errors.
- Do not use confirmation episodes for training or teacher distillation.
- Use official GateMem scoring for U/A/F/MGS. The primary paper metric is the
  arithmetic mean of the four domain MGS values.
- Do not call historical development results pristine holdout or final
  2,218-query performance.

## Before A New Run

Check existing output status, manifest identity, runtime/config hashes, and
active processes. Never reset or clean the worktree, and never overwrite a
frozen output directory. Create a new output directory for every new protocol
variant.
