# Stateful Policy Baseline Rollback and Framework Replan

## Decision

The recent per-field and role-capability changes are treated as a regression
line. The stable 60-checkpoint run remains the reference point:

| Version | N | U | A | F | MGS |
|---|---:|---:|---:|---:|---:|
| Stable Stateful Policy | 60 | 0.381 | 0.143 | 0.150 | 0.278 |
| Per-field Retrieval | 60 | 0.333 | 0.143 | 0.240 | 0.217 |
| Latest 80-checkpoint run | 80 | 0.419 | 0.481 | 0.182 | 0.178 |

The 80-checkpoint sample is not identical to the 60-checkpoint sample, so it
is not a strict paired comparison. It nevertheless confirms that the recent
pipeline has unstable privacy and answer-completeness behavior.

## Rollback Applied

1. `configs/gov_mem_stateful_policy_reasoning.yaml` now disables
   `role_capabilities_enabled`, matching the stable 60-checkpoint run.
2. The latest `lab_tech` role capability extension was removed from the
   baseline configuration.
3. Controlled retrieval now defaults to one `global_authorized_topk` candidate
   set after policy filtering.
4. The per-field retrieval union remains available only when an explicit
   `retrieval_strategy: per_field` or `field_local` is passed.
5. The existing state transition, explicit memory status, leakage guard, and
   policy verifier remain enabled.
6. The rollback does not reset the dirty worktree or delete the legacy
   modules and experiment artifacts.

The key invariant is that retrieval may recall evidence, but it must not
create multiple competing field-binding authorities. Field completeness will
be checked after the single authorized evidence set is formed.

## Verification

- `PYTHONPATH=src pytest -q tests`: `132 passed`
- `python -m compileall -q src run_govmem.py`: passed
- `python scripts/check_stateful_policy_runtime.py`: `PASS`
- `git diff --check`: passed

The initial bare `pytest -q` invocation was not used as a code result because
it omitted `PYTHONPATH=src` and collected unrelated `third_party` tests.

## New Framework Plan

### Phase 1: Restore a reproducible baseline

Run the original 60-checkpoint manifest with the rollback configuration and
confirm that the same pipeline is again near the `MGS=0.278` reference. Do
not use a new sample for this regression gate. The official judge and all API
settings must remain unchanged.

### Phase 2: Make one answer authority

Keep the pipeline boundary as:

`PolicyState -> PolicyDecision -> allowed evidence -> field projection -> answer`

The base LLM may parse query fields and propose source bindings. It may not
silently replace the selected field value. Python should only enforce:

- allowed-memory membership;
- lifecycle status;
- source provenance;
- value-type compatibility;
- temporal transition ordering;
- required-field coverage.

There must be one structured projection consumed by the final answer model.
No graph renderer, free-form renderer, fallback renderer, or second utility
bridge may compete with it in the same run.

### Phase 3: Fix Utility by mechanism

Prioritize the failure classes seen in the 60/80 runs:

1. Preserve all source-grounded candidates until temporal resolution.
2. Bind each field using `(subject, audience, temporal lane, attribute,
   value_type, provenance)`.
3. Resolve current/latest values deterministically after the LLM proposes
   semantic relations.
4. Require list cardinality and five-dimensional coverage before realization.
5. Let the answer model verbalize only the closed projection.
6. If realization omits a covered field, perform a bounded repair from the
   same projection; never reopen retrieval or infer a new value.

### Phase 4: Repair privacy without over-refusal

Apply authorization at field level for mixed requests. A restricted diagnosis,
credential, or private identity must not suppress an independently authorized
schedule, logistics fact, or public wording. Conversely, a safe projection
must not carry the source sentence that contains the blocked value.

The verifier should return a structured result per field:

`allowed | redact | deny | unknown`

It may narrow a projection, but it may not broaden the allowed memory set.

### Phase 5: Validation gates

1. Deterministic tests for state transition, typed binding, list completeness,
   blocked-memory isolation, and mixed disclosure.
2. Small real-API smoke tests covering one case per failure mechanism.
3. Re-run the same 60-checkpoint manifest.
4. Only if MGS is at least the stable baseline, evaluate a fresh held-out
   sample of at least 80 checkpoints.

Each change must have one hypothesis, one config switch, and one official
comparison table. No case-specific vocabulary, answer template, or checkpoint
name may be added to runtime logic.

## Current Status

The code is back on the stable retrieval and authorization configuration, but
the rollback has not yet been validated by a new official API run. The next
experiment should be the original 60-checkpoint baseline regression, not a
larger benchmark and not an ablation study.
