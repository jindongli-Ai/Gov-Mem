# Gov-Mem-v4-Symbolic-dev3 Medical Ledger Audit

Date: 2026-08-15

This is an offline audit of the previously saved 55-checkpoint Medical
holdout. It does not call OpenLux and does not change the official performance
table.

## Finding

Before this change, all 55 Medical ledgers had an empty `requested_slots`
list. The RAG retrieval already contained structured GateMem records, and the
existing required-slot planner could identify clinical plans, but the
Symbolic state ledger only consulted generic current-state and household
aliases.

## Minimal Fix

- Pass the existing `required_slot_plan` into the Symbolic evidence layer.
- Reuse the existing typed evidence-frame compiler for source-bound clinical
  fields such as allergy substance/reaction, medication, date, provider, and
  procedure.
- Extend the shared allergy parser to recognize GateMem phrasing such as
  `allergy on file is ...`.
- No new LLM calls, retrieval reordering, filtering, or enforcement were
  added.

## Offline Replay After Fix

| Measure | Count |
|---|---:|
| Medical checkpoints | 55 |
| Non-empty requested-slot ledgers | 27 |
| Ledgers with at least one resolved field | 15 |

The remaining empty cases include deleted/permission queries and general
clinical-summary wording without an existing slot contract. They should not
be filled with guessed fields. A separate, broader clinical-result contract
should only be added after its semantics are specified and tested.

## Scope

This audit is not an official GateMem evaluation and must not be used as a
paper performance result. The next validation step is a small official run on
2-4 Medical episodes using the normal low-concurrency protocol.
