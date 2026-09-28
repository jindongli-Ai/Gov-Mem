# Gov-Mem V8 Architecture

This document describes the active `govmem_v8_late_governance` pipeline. Older
V4/V7 backbones remain in the repository for historical reproducibility, but
they are not the canonical Gov-Mem V8 path.

## One checkpoint

```text
observable memory prefix
  ├─ Dense Naive RAG Top-20 ──┐
  │                           ├─ evidence context
  └─ Governed Slot Graph ─────┘   (audit side channel)
          │
          └─ Stage-2 LLM permission/lifecycle reasoning
                 │
                 ├─ source grounding and event binding
                 ├─ visible-prefix state projection
                 └─ symbolic claim critic
                          │
                    safe evidence boundary
                          │
                    answer agent
```

## Graph input boundary

The Governed Slot Graph is built from **all chunks in the current observable
history prefix**, excluding the current question turn. It is not built from the
dense Top-20 results. The dense index and graph therefore consume the same
visible prefix through two separate paths:

1. Dense retrieval ranks the prefix and supplies the primary Top-20 evidence.
2. The graph builder incrementally extracts open-vocabulary slots, entities,
   relations, permission candidates, lifecycle candidates, and source spans
   from the visible prefix.
3. The graph matches the question against its episode-local structure and
   records candidate sources and audit findings.
4. Graph findings are advisory context for Stage 2. They do not authorize a
   release, deny an answer, or replace dense evidence. Missing or unparsed graph
   information is recorded as an audit gap.

In the current V8 backbone, graph candidates do not occupy Top-20 positions;
the graph is an observer and audit channel. The configuration and implementation
must be checked together when running an ablation.

## Governance stages

### Stage 1: recall

`DenseMemoryIndex` builds an embedding index over visible memory chunks and
retrieves `rag.naive_top_k` records, normally 20. Bounded adjacent context is
then added for continuity while preserving the original source text.

### Stage 2: language governance

The LLM receives the retrieved evidence, visible-prefix state, application
policy, graph audit context, requester identity, and lifecycle/permission
events. It proposes field-level claims and explicit event bindings. The model
does the open-ended semantic work: role and relationship interpretation,
resource/action/scene matching, consent scope, and partial disclosure.

### Symbolic/neuro-symbolic checks

The runtime validates exact source spans, principal IDs, relation endpoints,
event bindings, visible-prefix membership, lifecycle state, and temporal
projection. The symbolic critic applies only supported explicit constraints,
such as an active bound denial or deletion. Missing graph edges, unknown roles,
and incomplete parsing do not become automatic denials or grants.

### Safe evidence and answering

Only source-grounded claims that survive the critic reach the answer agent.
Blocked values, denied spans, raw mixed records, and free-text denial reasons
are excluded. The answer agent returns `answer`, `answer_redacted`, `refuse`, or
`no_memory` according to the safe claim set.

## Main implementation files

| Responsibility | File |
|---|---|
| Canonical V8 backbone | `src/gov_mem/backbones/govmem_v8_late_governance.py` |
| Graph adapter | `src/gov_mem/memory/v8_slot_graph.py` |
| Graph builder/retrieval | `src/gov_mem/memory/governed_slot_graph.py` |
| Event extraction and validation | `src/gov_mem/extraction/v8_event_extractor.py`, `v8_event_validator.py` |
| Event cache and visible state | `src/gov_mem/governance_runtime/v8_event_store.py`, `v8_state_projector.py` |
| Stage-2 claim reasoning | `src/gov_mem/governance_runtime/v8_claim_reasoner.py` |
| Safe evidence | `src/gov_mem/governance_runtime/v8_safe_evidence.py` |
| Symbolic critic | `src/gov_mem/governance_runtime/v8_symbolic_critic.py` |
| Canonical configuration | `configs/govmem_v8_late_governance_gemini25flashlite.yaml` |
| Main launcher | `run_govmem.py` |

## Reproducibility boundaries

- Only the observable prefix is available to retrieval, graph construction,
  reasoning, and answering.
- GateMem raw data and official scorer are kept read-only and are not modified
  by the runtime.
- Domain policy text is configured application metadata shared with the
  baseline protocol; it is not learned from test answers.
- Graph and symbolic failures are recorded separately from network/API
  transport failures.
- Complete-episode evaluation is required for formal comparisons.
