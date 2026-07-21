# Gov-Mem Execution Roadmap

## Objective

Improve GateMem MGS without sacrificing privacy or deletion safety. All reported
results must use GateMem's official scorer and Yunwu-hosted LLM calls.

## Workstreams

1. Typed governance certificate: bind each requested slot to independent,
   lifecycle-valid evidence and a slot-family-matched policy decision.
2. Query-origin provenance: prevent the current request from self-confirming
   its own content during retrieval and realization.
3. Utility realization: retrieve complete multi-slot evidence, select the
   latest active version, and render only certified slots.
4. Policy compiler: convert explicit disclosure, assignment, and lifecycle
   constraints into policy atoms with cited support spans.
5. Lightweight adaptation: experience and skills provide audit reminders only;
   they never expand access or override governance certificates.
6. Evaluation ladder: unit/semantic checks -> targeted checkpoints -> one
   complete episode per domain -> fixed 101-checkpoint official suite.

## Current Status

- Query-origin evidence filtering: implemented.
- Typed phone/contact slot propagation: implemented.
- Deletion-before-reconstruction action constraint: implemented and passed its
  targeted official checkpoint.
- Policy-backed education blocker redaction: passed its targeted official
  checkpoint with zero leakage.
- Requested-slot-family redaction rule: implemented; pending paired education
  and medical official regression.
- Full 101-checkpoint evaluation: blocked until targeted regressions preserve
  zero leakage and utility does not regress.
