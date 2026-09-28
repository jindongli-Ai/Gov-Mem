# V8: raw memory plus a shallow neuro-symbolic policy ledger

The user requires symbolic/neuro-symbolic reasoning, but not a general knowledge
graph. The canonical V8 configuration now selects shallow memory with compact JSON model transport. This is an
architectural simplification of V8, not a new dev version.

## Selected pipeline

~~~text
visible conversation prefix
  -> original text memory bank
  -> dense Top-20 + bounded adjacent context
  -> Stage 2: language permission review + incremental policy/lifecycle edges
  -> deterministic prefix/time projection and exact event-bound symbolic veto
  -> approved original excerpts
  -> Stage 3 answering
~~~

Neural reasoning resolves natural-language scope, role/purpose, exceptions,
resource identity and which event applies to which evidence. Symbolic reasoning
preserves typed observations/provenance, projects permission transitions and
expiry deterministically, verifies exact source/binding identity, enforces
unambiguous bound denials and deletions, and removes restricted source spans. Both participate
in the runtime. There is no configuration switch to bypass the late critic.

## What is removed

Shallow ingestion does not ask the LLM to create scenes, entity nodes, ordinary
facts, participant rosters, or general relationships. Those model-output records
are rejected in shallow mode, not silently ignored. The old world-graph prompt
and broad ontology are absent from the shallow model request.

Ordinary facts stay in original memory text. Updates/deletions/cancellations can
be recorded as lifecycle EVENTs without reconstructing a second factual store.
Resources are local IDs with exact surface/source quotations; an optional
scene_id is a context label, not a scene node.

Only sanitized one-hop requester identity/assignment evidence is supplied.
It helps interpret public access rules and is not automatic authorization.
There is no multi-hop graph traversal or role-based hardcoded grant table.

## What remains mandatory

- Visible-prefix-only permission and lifecycle events, with exact citations.
- Principal -> resource/action/scope edges and issuer/time/condition fields.
- Deterministic effective permission state and its raw event history, both
  supplied to the language reasoner.
- Language-selected event binding followed by the existing narrow symbolic
  veto. Missing edges never automatically imply denial.
- The safe-evidence boundary before Stage 3.
- Gemini gemini-2.5-flash-lite at temperature zero for memory processing.

The current symbolic permission veto is deliberately narrow: active,
unconditional, whole-resource denials with exact resource/action/context binding
and no competing applicable allow. Ambiguous conditions/scopes and lifecycle
semantics remain language decisions with deterministic provenance/time checks.
Do not describe arbitrary language conditions as solved by a theorem prover.

## Preserve utility without copying every fact

The compact model protocol uses a candidate reference:

~~~json
{"slot":"visit","candidate_id":"candidate_0","keep":true,"bind":[]}
~~~

The model returns query_slots, claims, answer_action, and (when ingesting) a
flat events array. Python constructs the internal typed event collections.
The earlier TAB protocol remains compatibility-only: its first live integration
failed systematically and was stopped. This was missed by synthetic tests.

For a wholly safe relevant record, Python expands KEEP to the exact candidate
text, preserving numbers, negations, explanations, quotes and line breaks.
The LLM does not regenerate that text. A mixed record uses exact CLAIM excerpts
and separately grounded blocked spans; KEEP is not a default release for
unreviewed candidates. Source overlap redaction still applies after expansion.

The normal checkpoint still has two primary model calls (review + answering),
or one when all requested evidence is blocked. Cold-prefix ingestion and bounded
repair remain explicitly metered exceptions. Fewer required output records and
smaller schema were intended to lower overhead. The latest full run uses fewer
calls (522 versus 619), but more tokens (3,336,077 versus 3,164,624); cost is
still a limitation.

## Storage and compatibility

The canonical config uses v8_memory.mode=shallow, response_protocol=json and
candidate_id_style=source. A single-turn candidate uses its actual source turn
ID. Source quotations already supplied in graph/ingestion are also selectable
without a hidden bank lookup; a source with disjoint visible fragments cannot
be expanded into a fictitious whole turn. Grounded explicit denials need not
repeat an empty binding array. Released claims still require explicit bindings.

Valid event ingestion commits before query claim normalization. The query graph
folds older unconditional advisory updates under an identical issuer/grantee/
resource/action/context binding, while retaining every deletion, scoped or
conditional event and the raw append-only history. This is a graph view, not
physical deletion or a grant of access.
Internal serialization remains Python-generated JSON with a memory_mode marker.
A full-graph cache cannot silently be loaded as a shallow ledger; use a fresh
experiment output directory. Historical snapshots/results are unchanged.
Full mode remains for old direct callers/tests and explicit historical comparisons,
not as a second current model recommendation.

Stage 3 retains its existing API wrapper containing the textual answer.

## Evidence and limitations

The current matching frozen runtime completed all 12 episodes / 303 checkpoints:
**40.06% mean-domain MGS versus 19.75% baseline**, or **32.91%** with technical
failures penalized pessimistically. Seventeen execution errors remain. All four
domain MGS values exceed baseline, but Office utility is lower and Education
deletion leakage remains 50%. This reused development sample is not a holdout.
See the [current complete report](../experiments/result/2026-09-19_Gov-Mem-v8_source_projection_random3_per_domain_paired.md).
The real run contains eight independently matched symbolic release-to-block
changes, including six involving deletion. These are decision-path evidence,
not a causal ablation or proof of eight true prevented leaks.


Latest targeted lifecycle/source/flow tests: 79 passed. Repository test
suite: 463 passed with the same 13 previously documented legacy failures.

The synthetic two-checkpoint test first forwards an entire legitimate record
with no copied model value. At the next checkpoint it intentionally simulates
an LLM releasing a value bound to an explicit deny; the symbolic critic changes
that proposal to a block, and the answering model is not called. Separate tests
cover permission expiry, prior-resource grounding for a new revocation,
one-hop-only identity context, shallow prefill and incompatible caches.

These demonstrate that symbolic reasoning affects execution, not improved MGS.
The earlier frozen shallow run completed 12 episodes / 303 checkpoints: official mean
MGS 22.19% versus baseline 19.75%, but 47 execution errors reduce worst-case
MGS to 8.28%. This is not a stable win. Actual captures show three independent
symbolic release-to-block changes. Inference used 619 Gemini calls and
3,164,624 tokens, approximately 4.95 times baseline tokens.

Subsequent offline contract fixes recovered 73 of 413 captured responses with
zero regressions (330 passed at that point); this historical replay was not a
remeasurement of MGS. See the [complete report](../experiments/result/2026-09-19_Gov-Mem-v8_shallow_random3_per_domain_paired.md).
The earlier 4.37% V8 score belongs to the full-graph implementation.

Risks requiring complete-episode evaluation: relevant raw facts or assignments
can be missed by retrieval; KEEP requires the LLM to recognize a wholly safe
record; event extraction and semantic scope matching can still be wrong.
The desired evaluation should separately report utility, privacy/deletion leaks,
execution errors and actual provider usage, with the unchanged official baseline.

The [lifecycle/protocol follow-up](../experiments/result/2026-09-19_Gov-Mem-v8_lifecycle_protocol_followup.md)
records a complete Education pilot, its failed preliminary integration, all paid
usage, and further offline fixes. Pilot MGS is 55.56% (44.44% pessimistic), with
two errors; it is not a replacement for the four-domain comparison.
