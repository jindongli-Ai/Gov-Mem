# Stateful Policy Safe-Delivery Binding v29

## Change boundary

This iteration kept one mechanism boundary: aggregate safe-summary evidence
binding.

- Safe-summary contracts now activate the public/current carrier recall path
  even when the policy intent scope is ordinary `logistics`.
- Current operational carriers such as arrival windows, entry methods, and
  approved zones are retained beyond dense top-k.
- A named delivery recipient is treated as a view-binding hint. Explicitly
  named sibling recipients and restricted surprise projections are excluded
  from the aggregate view.
- The same recipient boundary is applied after the recall-support union, so a
  later recall pass cannot reintroduce a sibling audience.
- Direct recipient-directed operational instructions remain eligible when
  they are safe setup constraints.

No authorization rule or policy decision was broadened.

## Verification

| Check | Result |
|---|---:|
| Stateful policy and field-projection tests | 170 passed |
| Runtime static check | PASS |
| Cross-domain repair probe v23 | 7 checkpoints; action accuracy 1.000; utility remained mixed |
| Mason safe-delivery v24-v29 | 1 checkpoint per run; action accuracy 1.000; utility 0.000 |
| Privacy/deletion leakage in v29 | 0 / 0 |

The most recent single-checkpoint answer was cleaner than the prior probe but
still incomplete: it retained the current supper window while treating a
request placeholder (`current dessert handoff method`) as a value and omitted
several required setup fields.

## Release decision

Not a release candidate. The stable 60-checkpoint reference remains
`MGS=0.278`; the last complete 60-checkpoint experimental version was
`MGS=0.219`. No 60-checkpoint run and no GitHub push were performed for v29.

## Remaining root cause

The aggregate field planner still promotes source-local requests and generic
method/rule placeholders into factual fields. The next bounded repair should
be a source-claim quality gate for aggregate planning: retain only a concrete
source predicate/value pair or a direct, recipient-scoped operational
instruction. It should be validated first on the safe-delivery probe and a
held-out cross-domain sample before any larger run.
