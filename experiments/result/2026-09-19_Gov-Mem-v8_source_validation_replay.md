# V8 source-validation repair and offline replay — 2026-09-19

This report covers additional in-place V8 changes after the earlier correctness
follow-up. No model API was called. No new MGS or token-cost result is claimed.

## Observed failure

Stage 2 could see retrieved history and cached graph quotations, but joint
extraction validation only accepted ingestion.new_turns and context_turns.
A model citation could therefore refer to evidence actually supplied in the
same request and still fail validation. This caused technically valid citations
and redundant old observations to be treated as failed new ingestion.

The 28 terminal checkpoint errors in the frozen experiment comprise:

| Category | Checkpoints |
|---|---:|
| Lifecycle fact source/value grounding | 12 |
| JSON syntax | 7 |
| Claim grounding | 3 |
| Restriction grounding | 2 |
| Other governance event validation | 4 |

These are error-message categories, not semantic performance categories.
The earlier progress message saying 13 lifecycle-fact errors was incorrect;
the exact count is 12.

## Runtime fixes

- One shared source collector now supplies both claim validation and joint
  extraction with the exact excerpts actually sent to Stage 2: single-turn
  retrieved candidates, existing graph citations, and ingestion turns.
- It does not read the remaining conversation, future turns, labels, or answers.
  Graph excerpts remain separate; they cannot be joined into fabricated quotes.
  Multi-turn chunks without character attribution are not assigned wholesale to
  each source turn.
- A valid old observation is audited as context-only and not re-appended.
  New graph records still require at least one exactly grounded NEW_TURN source.
  If a record cites a new turn with a fabricated span, an old valid citation
  does not excuse it.
- For lifecycle records with one unique grounded excerpt, an empty/boolean value
  or a delete/cancel value absent from that excerpt is stored as the full literal
  excerpt. Model-generated values are discarded, not guessed. General unsupported
  update paraphrases still fail; multi-source ambiguous values still fail.
  This records the change statement; it does not authorize releasing a claim.
- Claims are deduplicated only after source offsets validate, and the key
  includes source identity, offsets, request status, and governance binding.
  Identical denied text in two sources must remain two protected spans.
  Raw saved responses contained ten such cross-candidate duplicate denial
  groups; that count does not mean ten observed final leaks.
- Stage 3, Gemini model selection, paid-evaluation outputs and raw datasets
  are unchanged.

## Fixed-response replay

The same 355 saved Stage-2 API responses, including repair attempts, were
processed by the frozen historical extractor and the revised extractor.
Both runs have capture SHA256:
e32e6a02eb3b3f9bd812bec61f3bcdf9b397618351426979a6e0638643ffb34c

| Replay outcome | Frozen V8 | Current V8 |
|---|---:|---:|
| Requests with valid extraction shape | 310 | 310 |
| Accepted extraction batches | 264 | 299 |
| Batches with critical extraction rejection | 46 | 11 |
| Unparseable responses | 13 | 13 |
| Requests without ingestion | 30 | 30 |
| Invalid events shape | 2 | 2 |

Per-file comparison: **35 resolved critical failures, zero newly failing
batches**. Some accepted batches contain only old/context observations; acceptance
does not imply additional graph facts or that the complete checkpoint succeeds.

Remaining failures include unknown subjects, missing grantee, unsupported
resource surfaces and invalid source quotes. They remain explicit failures.

This replay uses old JSON responses because those are the available captured
responses. It validates the common extraction validator and source-scope fix,
not Gemini's unmeasured performance with the current line protocol. It does not
rerun retrieval, reasoning, answering, or official judging, and does not simulate
the counterfactual cache evolution or calculate MGS.

Reproduction:

~~~bash
PYTHONPATH=outputs/v8_paired_random3_seed20260918_attempt8/runtime_snapshot/src \
python scripts/audit_v8_saved_extractions.py \
  --root outputs/v8_paired_random3_seed20260918_attempt8/govmem \
  --output experiments/result/2026-09-19_v8_extraction_replay_before.json

PYTHONPATH=src python scripts/audit_v8_saved_extractions.py \
  --root outputs/v8_paired_random3_seed20260918_attempt8/govmem \
  --source-scope prompt \
  --output experiments/result/2026-09-19_v8_extraction_replay_after.json
~~~

The artifacts contain validator source hashes and per-capture rejection/repair
audits. Original snapshots, captures, predictions and judges were not overwritten.

## Tests

Relevant suite: **91 passed**. Includes old-source/new-change joint extraction
through the complete backbone, future/unseen citation rejection, exact-source
lifecycle persistence, contradictory update rejection, and duplicate-source
redaction retaining allowed information.

Full repository: **416 passed, 13 failed**. The failures are the same legacy
field_state_projection/stateful_policy cases documented before these changes.
No new full-suite failure was introduced.
