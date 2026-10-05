# Gov-Mem

**Gov-Mem** is a governance-aware neuro-symbolic memory framework for
multi-principal shared-memory agents. It treats memory authorization and
current-state validity as first-class concerns in retrieval and answering.

## Research problem

Conventional memory systems optimize what to store and retrieve. In shared
memory, semantic relevance does not establish that a memory may be disclosed:

```text
relevance ≠ authorization ≠ current-state validity
```

A memory can be relevant but requester-restricted, revoked, superseded, or
deleted. Gov-Mem preserves a strong RAG utility path and governs claims before
they reach the final answer model.

## Current V8 pipeline

```text
observable history prefix
        ├── Dense Naive RAG Top-20 (primary utility path)
        └── Governed Slot Graph (auxiliary governance observer)
                         ↓
        LLM semantic permission/lifecycle reasoning
                         ↓
        source and event binding validation
                         ↓
        symbolic/neuro-symbolic governance critic
                         ↓
        claim-level safe-evidence boundary
                         ↓
        answer / answer_redacted / refuse / no_memory
```

The episode-local Governed Slot Graph is incrementally extended from newly
observable history at each checkpoint. It is not built from Dense RAG Top-20.
The graph organizes source-grounded entities, slots, relations, permission
observations, and lifecycle events. In the canonical V8 runtime it is an
observer and audit channel: it does not directly grant permission, deny a
query, or displace Dense RAG evidence. Explicit graph findings are supplied to
the language governance reasoner as advisory signals.

The language model handles open-ended semantic questions such as permission
scope, requester-resource relationships, operational duties, and event
association. Deterministic checks validate source grounding, event bindings,
lifecycle state, and explicit active deny/revoke/delete conditions. Only
verified claims cross the safe-evidence boundary.

## Main contribution

Gov-Mem explicitly formulates **memory governance** as a problem distinct from
memory retrieval. The framework combines:

1. an incremental Governed Slot Graph for source-grounded governance evidence;
2. neuro-symbolic reasoning that separates semantic interpretation from
   deterministic verification; and
3. a claim-level safe-evidence boundary for partial, requester-aware release.

## Benchmark

Experiments use GateMem, a multi-domain benchmark for memory governance in
shared-memory agents:

- Medical: 579 checkpoints
- Office: 547 checkpoints
- Education: 540 checkpoints
- Household: 552 checkpoints
- Total: 2,218 checkpoints in 91 complete episodes

The primary paper metric is the arithmetic mean of the four domain-level MGS
values. Official scoring reports U (utility), A (access-control leakage), F
(post-deletion leakage), and MGS.

## Gemini-2.5-Flash-Lite results

The first complete Graph-enabled V8 run used Gemini-2.5-Flash-Lite for memory
processing and answering, `text-embedding-3-small` for retrieval, and GPT-4o as
the official judge. It completed all 2,218 checkpoints and obtained:

| Domain | U (%) | A (%) | F (%) | MGS (%) |
|---|---:|---:|---:|---:|
| Medical | 88.10 | 34.38 | 19.21 | 46.71 |
| Office | 82.47 | 66.67 | 22.07 | 21.42 |
| Education | 68.33 | 29.44 | 11.67 | 42.59 |
| Household | 56.52 | 25.54 | 14.67 | 35.91 |
| **Four-domain average** | — | — | — | **36.66** |

This result is the first full-benchmark result for the Graph-enabled V8
runtime. It retained three execution errors (two Medical, one Household); the
machine-readable output and full audit are in
[`outputs/v8_graph_full2218_20261005/full_metrics.json`](outputs/v8_graph_full2218_20261005/full_metrics.json)
and the dated report under [`experiments/result/`](experiments/result/).

The earlier 30.02% full-benchmark score is a historical frozen **pre-graph V8**
runtime. It should not be labeled as the current Graph-enabled result or as a
strict paired graph-off ablation.

## Repository guide

### Active implementation

- [`src/gov_mem/backbones/govmem_v8_late_governance.py`](src/gov_mem/backbones/govmem_v8_late_governance.py)
- [`src/gov_mem/memory/v8_slot_graph.py`](src/gov_mem/memory/v8_slot_graph.py)
- [`src/gov_mem/memory/governed_slot_graph.py`](src/gov_mem/memory/governed_slot_graph.py)
- [`src/gov_mem/governance_runtime/v8_claim_reasoner.py`](src/gov_mem/governance_runtime/v8_claim_reasoner.py)
- [`src/gov_mem/governance_runtime/v8_symbolic_critic.py`](src/gov_mem/governance_runtime/v8_symbolic_critic.py)
- [`configs/govmem_v8_late_governance_gemini25flashlite.yaml`](configs/govmem_v8_late_governance_gemini25flashlite.yaml)

### Documentation

- [`docs/GOVMEM_V8_ARCHITECTURE.md`](docs/GOVMEM_V8_ARCHITECTURE.md): code-grounded architecture
- [`docs/GOVMEM_V8_LATE_GOVERNANCE.md`](docs/GOVMEM_V8_LATE_GOVERNANCE.md): late-governance design
- [`docs/GOVMEM_V8_SLOT_GRAPH_INTEGRATION.md`](docs/GOVMEM_V8_SLOT_GRAPH_INTEGRATION.md): graph integration and boundaries
- [`README_GITHUB_PUSH.md`](README_GITHUB_PUSH.md): server-to-GitHub push instructions
- [`experiments/result/`](experiments/result/): dated experiment reports

### Tests

Run the focused V8 governance and graph tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q \
  tests/test_v8_slot_graph.py \
  tests/test_v8_late_governance_backbone.py \
  tests/test_v8_late_governance.py \
  tests/test_governed_slot_graph.py
```

### Full evaluation entry point

The frozen full-suite runner is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python scripts/run_v8_full_evaluation.py \
  --output outputs/<new-run-directory> \
  --stage all --workers 30
```

Use a new output directory for every protocol variant. The runner preserves
episode order, records execution errors, freezes runtime/config identities, and
uses the configured OpenLux key pool without printing credentials.

## Reproducibility and scope

All runtime decisions use only the observable history prefix at each
checkpoint. Future turns, gold answers, evaluator annotations, and scorer
metadata are excluded from inference. Historical reports remain available for
provenance; numbers from different frozen runtimes must not be mixed into one
causal ablation claim.

