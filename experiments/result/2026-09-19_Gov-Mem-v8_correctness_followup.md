# V8 correctness follow-up — 2026-09-19

The completed frozen paired sample remains V8 4.37% vs RAG-Naive 19.75% mean
official MGS. This follow-up changes the current V8 implementation; it does not
replace, rerun, or improve those measured results.

## Changes and evidence

1. **Missing public policy input.** The canonical YAML now contains the exact
   MEDICAL/OFFICE/EDUCATION/HOUSEHOLD_QUERY_POLICY strings from
   third_party/GateMem-official/bench/domains.py. An offline test compares all
   four strings verbatim (excluding YAML's final newline). The runtime accepts
   application-supplied policy and records its source/hash; it never imports
   dataset answers, query types, attack types, or scorer code. A policy citation
   can justify a language decision only when its exact text was supplied.
   Temporal scaffold attributes remain filtered: policy parity is improved,
   but full input parity is not claimed.

2. **Dropped lifecycle representation.** State projection previously exposed
   FACT deletion tombstones but not lifecycle EVENT history without a grantee.
   Inspection of the 12 frozen final episode stores found three lifecycle
   events, all delete events with no grantee. Current Stage 2 receives visible
   lifecycle events, including later updates, so the LLM can resolve their scope
   and chronology. A synthetic test covers earlier prefixes, deletion, later
   updates, and exact binding. This does not prove these three records caused
   any particular benchmark failure.

3. **Cross-scene permission conflict.** An allow in another scene formerly
   suppressed an explicitly bound deny with the same resource/action. Scene
   equality is now required for this conflict. Synthetic tests cover both
   different-scene denial and same-scene ambiguity; no new lexical veto added.

4. **Repeated extraction after claim failure.** Source/schema-valid,
   query-independent graph deltas are committed before claim validation.
   If claim grounding fails, bounded repair sees the refreshed graph without
   repeating ingestion; a fresh process also reuses the saved watermark.
   Invalid governance deltas still raise before commit. Both bounded repair and
   restart paths are tested. Actual provider savings have not been measured.

5. **Mixed request boundary.** A synthetic end-to-end case retains an allowed
   appointment fact, blocks a credential under an explicit supplied policy,
   and verifies the credential never enters the Stage-3 prompt.
   Stage 3 retains its existing JSON wrapper and text answer.

## Optional lossless input packing

The implementation can intern repeated long literal strings into a text pool,
with an offline inverse decoder. It is **disabled by default**: savings were
small and Gemini reference comprehension has not been measured.

Offline replay: 275 available saved normal Stage-2 prompts across all 12 selected
episodes. The 28 execution-error checkpoints lack these per-prediction prompt
files and are not part of this transport audit. This is not a new performance
evaluation and does not exclude errors from the original benchmark.

- All 275 round trips exactly preserved every field and string.
- Same inputs including public policy, compact JSON:
  5,549,899 characters unpacked vs 5,375,187 packed (**3.15% reduction**).
- Historical saved user prompts: 5,416,481 characters, before adding policy.
- These are user-prompt characters, excluding system instructions, outputs, and
  repair calls. They are neither token counts nor an API cost estimate.

Reproduction (no API calls):

~~~bash
PYTHONPATH=src python scripts/audit_v8_prompt_inputs.py \
  --predictions-root outputs/v8_paired_random3_seed20260918_attempt8/govmem \
  --output experiments/result/2026-09-19_v8_prompt_input_audit.json
~~~

## Validation and remaining limits

Relevant offline suite: 84 passed. It checks implementation contracts using
synthetic model responses; it cannot measure Gemini's semantic decisions.

Current memory model remains gemini-2.5-flash-lite, temperature 0; no paid API
requests were made. No additional version directory, benchmark lexicon, or
gold-driven rule was introduced. Frozen predictions and runtime snapshots were
not changed.

Outstanding: true MGS and provider cost of this revision, semantic extraction
recall, correct policy/claim matching, and safe utility retention. The existing
sample has been used during development and is not an untouched holdout.
A future authorized evaluation must still use all checkpoints of complete
episodes, retain execution errors, and report the official scorer's U/A/F/MGS
rather than infer performance from these offline tests.
