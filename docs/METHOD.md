# Gov-Mem: implementation reference

This document describes the checked-in implementation. The paper is under review.

The [framework-to-code correspondence](FRAMEWORK.md) maps Fig. 2 of the October 16 manuscript to the current execution path. Gov-Mem is neuro-symbolic: LLMs perform extraction and semantic adjudication, while deterministic source/binding checks, event-state projection, symbolic criticism and safe-evidence construction constrain evidence release. The Governed Slot Graph is enabled in the default Full configuration and feeds source-grounded advisory findings into the language reasoner.

## Canonical execution path

`run_govmem.py` → `GovMemRunner` in `src/gov_mem/pipeline.py` → `GovMemV8LateGovernanceBackbone.run_instance` in `src/gov_mem/backbones/govmem_v8_late_governance.py`.

Use `govmem_v8_late_governance` and `configs/govmem.yaml` for the current method. Historical modes remain for compatibility; they are not alternative descriptions of the current method. Runtime source files are copied without implementation changes from the research workspace.

1. Load a checkpoint's visible conversation prefix and requester; build the principal registry. Preserve sequential history within each episode.
2. Prepare the source-grounded event store and incremental ingestion. Shallow memory keeps raw turns and a compact policy/lifecycle ledger rather than a full world graph.
3. Embed visible turn chunks and retrieve dense Top-K evidence. Expand with bounded adjacent context (`v8_query.max_adjacent_context`). K is the initial dense retrieval size, not an upper bound on all subsequent context.
4. Run the enabled Governed Slot Graph channel and attach its observer audit. **The graph participates in Stage-2 reasoning through `governed_slot_graph_audit`**: source-grounded policy and deletion findings are advisory inputs to semantic claim review. Graph-derived factual candidates remain separate from dense evidence (`graph_selected=0`, `mode=observer_only`), as specified in Fig. 2. Graph silence does not bypass the remaining governance stages.
5. Generate a source-grounded claim ledger and semantic bindings. Joint extraction can append validated events and refresh projected state. Configuration permits bounded contract repair; malformed contracts can fail closed to `no_memory`.
6. Apply the symbolic governance critic against projected state. Language models perform semantic matching; the critic checks explicit bindings and state constraints. This does not imply unrestricted symbolic understanding of natural-language policies.
7. Build safe evidence from reviewed claims. Final answer generation receives released evidence plus sanitized blocked-slot metadata, the question, and requester information; denied values and source quotes are removed from that metadata. Refusal/no-memory paths differ from the supported-answer generation path.
8. Export answers, provenance, prompt-context audits, execution status and telemetry; optionally invoke the official GateMem scorer.

## Four modules

Paths below are relative to `src/gov_mem/`.

| Module | Implementation | Role |
|---|---|---|
| Governed Slot Graph | `memory/governed_slot_graph.py`, `memory/v8_slot_graph.py` | Incremental source-grounded slots and relations; observer findings in the current backbone |
| Event-State Projection | `governance_runtime/v8_event_store.py`, `governance_runtime/v8_state_projector.py` | Validated event storage and projected state for the visible prefix/requester |
| Symbolic Governance Critic | `governance_runtime/v8_symbolic_critic.py` | Review claim bindings against permissions, lifecycle state and graph context |
| Safe-Evidence Boundary | `governance_runtime/v8_safe_evidence.py`, `_answer_from_safe_evidence` in the backbone | Control reviewed content passed to final answer generation |

`extraction/` contains schemas, source validation and parsing. `governance_runtime/v8_claim_reasoner.py` contains Stage-2 prompts and validation. `governance_runtime/v8_shallow_memory.py` handles compact claims. `llm/client.py` implements transport and telemetry.

## Evaluation

Obtain [GateMem](https://github.com/rzhub/GateMem) separately. Four domains: Medical, Office, Education and Household. The full research benchmark has 91 episodes and 2,218 queries. These episodes have historical experimental exposure; they are not a pristine holdout.

U: utility accuracy; A: privacy leakage rate; F: deletion leakage rate. Within each domain, `MGS = U × (1 − A) × (1 − F)`. Average each metric equally across four domains. Avg.MGS is the mean of domain MGS values, **not** the product of macro-average U, A and F. Official scoring uses GPT-4o at temperature 0 with action gating disabled. Check checkpoint identities, label completeness, applicable judge fields and execution errors before reporting.

Numerical claims require archived configurations, runtime identities, predictions and scoring records. This repository does not publish the private experiment workspace. Source code alone does not establish statistical significance or the cause of differences between runs.

## Writing the paper

Use the canonical path and module locations above. Distinguish current execution from historical compatibility modules. Ground descriptions of graph candidate selection, answer context, contract repairs, fallbacks and call counts in the actual backbone. Call counts vary with ingestion, graph settings, repairs and safe evidence; avoid a fixed calls-per-query claim. Use separately verified experimental tables; do not invent results, citations, guarantees or ablation significance from source code.
