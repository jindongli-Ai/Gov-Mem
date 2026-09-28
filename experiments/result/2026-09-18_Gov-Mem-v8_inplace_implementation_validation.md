# Gov-Mem V8 in-place implementation validation — 2026-09-18

This is an offline engineering report, not a GateMem benchmark result.
No paid memory-model, embedding, or judge API experiment was started for this
change. Historical Medical/Atlas/Beacon predictions and raw data were not edited.

## Implemented behavior

- One joint Stage-2 call for incremental graph extraction and language-based
  permission review; one final answer call when information is releasable.
- Dense Top-20 remains the main recall channel. Preserve complete candidate
  turns and useful verbatim excerpts/context claims, with bounded adjacent turns.
- One domain-independent, publicly inspectable structural ontology; no new
  benchmark-derived lexicon or fixed role-to-permission map.
- Append-only observations, prefix-safe scene/entity merging, relation and duty
  edges passed to the reasoner, scoped temporal permission projection.
- Removed lexical overlap, missing-edge, scene-membership, requester-authored
  auto-authorization, and same-subject propagation as permission shortcuts.
- Grounded language decisions plus narrow exact-binding symbolic enforcement;
  mask overlapping denied character spans without affecting independent fields.
- Correct action labels for complete scoped answers and actual partial release.
- V8 JSON attempts fixed at one; HTTP attempts bounded at two. Invalid model
  contracts remain visible failures. Default fail-fast saves telemetry and stops
  before spending through the rest of a broken episode.
- Cost audit includes shared extraction, cold-prefix prefill, actual HTTP retries,
  latency, token usage coverage and cumulative strict-resume totals.

The dev1 entry point is now a compatibility alias, not a frozen algorithm.
Use a fresh output directory; old event caches are deliberately incompatible.

## Checks

| Check | Result |
|---|---|
| Focused V8 + LLM protocol tests | **51 passed** |
| Project suite (`pytest tests`) | **376 passed, 13 failed** |
| Clean HEAD reproduction of the two failing legacy modules | **176 passed, same 13 failures** |
| Offline cost-audit fixture | Passed: 2 logical transport calls + 1 HTTP retry = 3 provider requests |
| `git diff --check` | Passed |

The 13 failures belong to existing `field_state_projection` and `stateful_policy`
tests. They were independently reproduced from `git archive HEAD` in a temporary
checkout. Those legacy tests and policy implementations were not changed.

Bare `pytest` initially attempted to collect a vendored SciPy test and failed
collection; use `pytest tests` to test this project's suite.

Synthetic coverage includes exact substring grounding (including punctuation),
ambiguous repeated values, source interval masking, actual requested-slot action
rendering, independent fields of the same subject, matching vs nonmatching
resource/action/scene, scoped/expired/conditional permissions, pending revoke,
whole-resource revoke, scene roster updates, subject-separated facts, cache reuse,
earlier-prefix replay, invalid extraction watermark, cold-prefix cost bounds,
JSON retry suppression, HTTP retry telemetry, and fail-fast behavior.

A complete four-checkpoint synthetic episode, using deterministic fake clients,
exercised full release → partial release → revocation → cached refusal. Its
logical chat call counts were **2, 2, 1, 1**. A replay of its earlier checkpoint
excluded cached future restrictions. The second answering prompt contained the
allowed room but not the restricted token. These assertions test orchestration
and information boundaries, not the LLM's semantic competence.

## What remains empirical

No claim is made that MGS exceeds RAG-Naive, real token cost has fallen, extraction
F1 is sufficient, or semantic leakage is zero. A joint request saves a separate
model call but may use more input/output tokens. Strict source/contract validation
may expose real-provider failures; measure completion rate and token usage along
with Utility/Privacy/Safety instead of silently counting failures as refusals.

Next empirical validation: two unseen complete episodes, each paired with RAG
under matching models/retrieval/evaluation; only then old-episode regression.
Formal U/A/F/MGS require the official scorer/protocol. Independent human
annotations remain necessary for semantic graph-extraction precision/recall/F1.
