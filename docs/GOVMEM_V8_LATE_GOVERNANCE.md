# Current canonical architecture

The canonical V8 config now uses shallow neuro-symbolic memory, per the user's
requirement that symbolic reasoning remain mandatory while the graph stay small.
See [the current design](GOVMEM_V8_SHALLOW_MEMORY.md). The sections below document
earlier full-graph V8 behavior and fixes; they are not the shallow model's scores.

---

# Shared prompt-source validation (2026-09-19)

Claim validation and joint extraction now use the same exact source collector:
single-turn candidate text, graph source excerpts, and ingestion turns. New
records still require new-turn evidence. Cached graph excerpts remain separate;
mixed-turn candidates without offsets cannot establish a per-turn quote.

Lifecycle facts can retain their literal source statement instead of an empty,
boolean, or deleted/canceled value absent from a unique source. Unsupported
arbitrary update paraphrases and ambiguous multi-source values remain errors.
Claims are deduplicated after offset validation using source/governance identity,
so duplicate denied values in different candidates retain both protected spans.

The frozen-response replay reduces critical extraction failures from 46 to 11
among 310 valid-shape batches. This is not an end-to-end performance evaluation.
See experiments/result/2026-09-19_Gov-Mem-v8_source_validation_replay.md.

---

# Current correctness update (2026-09-19)

The canonical config supplies explicit application access policy. For GateMem
these are verbatim public rules used by the official baseline, kept in config
rather than a dataset-derived lexical classifier. Policy citations use reserved
source ID application_access_policy and must quote supplied text exactly.
No model-generated policy is accepted as a configured rule.

Graph context includes both FACT tombstones and visible lifecycle EVENT history.
Neither cancellation nor missing graph edges alone implies deletion or denial.
The scene participates in symbolic permission-conflict identity.

Validated graph deltas are committed independently of subsequent claim validation.
Consequently, claim repair reuses persisted state and omits already processed
ingestion. An invalid graph delta still prevents watermark advancement.
This changes failure recovery, not the safe-evidence boundary or Stage 3 interface.

Optional text-pool input packing is disabled by default after modest offline
savings. Public policy input remains enabled. No paid re-evaluation has occurred;
see the 2026-09-19 follow-up report for scope and validation limits.

---

# Current output protocol update (2026-09-18)

The current V8 config uses `v8_query.response_protocol: lines` following the
user's instruction. History extraction and Stage-2 model responses are plain text, one record per line, with
actual TAB separators and literal unescaped field values. SOURCE and
RESTRICTION lines cite exact visible text. EVENTS/ACTION/END markers detect
missing ingestion and incomplete responses; invalid text is still an audited
error, not an implicit release. Internal graph/cache JSON is program-generated.
Successful Stage-2 responses also have .txt sidecars under
`v8_text_responses/<dataset>/`. API transport remains standard JSON as required
by the provider; history extraction/Stage 2 do not ask the model to generate JSON syntax.

Every CLAIM requires `bind=NONE` or explicit known event IDs. `NONE` is an
explicit model judgment, not proof of authorization. Selected IDs are expanded
into canonical resource/action/scene bindings in Python and then passed to the
existing source-grounded symbolic critic. Unknown or inconsistent bindings fail.
This fixes the omitted-binding path seen in the completed experiment, without
claiming that the LLM will always select the correct policy. No lexical matching
or same-subject propagation was added. Stage 3 retains its original JSON API
wrapper containing the text answer and claim citations, and its existing
safe-evidence boundary. Newlines/tabs within source values require
narrower contiguous excerpts; the parser does not silently normalize source text.

Offline tests cover literal quotes/backslashes, truncation, malformed records,
source attachment, mandatory binding selection, unknown bindings, text prefill,
the unchanged Stage-3 interface and an end-to-end symbolic denial. The old JSON path remains
for explicit historical/test compatibility. No new paid run was started.
The old experiment (4.37% vs 19.75% MGS) is a different frozen JSON snapshot;
see `experiments/result/2026-09-18_Gov-Mem-v8_random3_per_domain_paired.md`.

---

# Gov-Mem V8: language-led late governance

User direction on 2026-09-18: modify the existing V8 implementation, without
creating dev2. The dev1 module/config remain compatibility entry points for the
active implementation. Existing benchmark outputs describe the old code only.

## Runtime contract

1. Keep turn-level dense Top-20 retrieval and bounded adjacent-turn context.
   By default no retrieved turn is truncated. Graph hints never replace RAG.
2. Load an episode-local append-only event cache. Project only observations
   whose complete source spans belong to the current visible prefix.
3. In one Stage-2 LLM request, extract a compact graph delta from newly visible
   turns and judge the current query's permissions. The delta is instructed to
   be query-independent, but shares the prompt with the question; this must be
   assessed in semantic extraction evaluation rather than assumed perfect.
4. Validate exact source substrings, closed principal IDs, relation endpoints,
   and typed duty records. Save validated events without advancing the cache
   on failed extraction. Reproject with the new delta before symbolic review.
5. The LLM identifies field-specific restrictions using raw evidence and graph
   state, citing actual visible sources. Missing graph edges do not mean denial.
   Scene membership, job title, kinship, or having repeated a value do not mean
   permission. Scope and operational duty require language reasoning.
6. The symbolic critic enforces an active unconditional deny only after the
   language model has matched the event and resource/action/scene binding.
   Conditional, conflicting, scoped, or expired states remain language decisions.
   No keyword-overlap veto, missing-edge veto, or same-subject propagation remains.
7. Release verbatim values or longer safe excerpts as claims, including explicit
   supporting-context claims. If an allowed excerpt contains a blocked span from
   the same source, mask that exact character interval. This does not propagate
   to another field merely because its text/subject/value happens to match.
8. Stage 3 receives only these claims and sanitized blocked requested-slot
   labels. It receives no original mixed records, source quotes, denied values,
   or free-text denial reasons. Derive `answer_redacted` from actual partial
   requested-field withholding, not scoped permission or summary labels.

The final context still depends on the LLM selecting sufficient safe excerpts;
this is not a guarantee of RAG-equivalent utility or zero semantic leakage.
Claim grounding is exact; interpretation of permission scope is not proved by
substring validation. Invalid contracts produce execution errors, not invented
refusal/no-memory predictions. Whole-episode completion checks prevent scoring
an incomplete output set as a complete experiment.

## Permission graph and temporal state

`V8EventStore` records observations of scenes, entities, facts, relations and
governance events, including repeated IDs with new sources. Projection merges
visible scene participants and aliases and incorporates `participates_in`
relations; future observations never contaminate an earlier roster. Extracted
relations and typed `operational_duty` edges are actually supplied to Stage 2.

Duty edges contain principal, resource, role, action, resource category, optional
subject/scene, and source spans. They inform language reasoning and are not a
fixed role-to-resource authorization table.

Fact state keys include both scene and subject. Permission state preserves
issuer, grantee, resource, action, scene, included/excluded scope, conditions,
and expiration. Whole-resource unconditional updates supersede prior scoped
states of the same issuer/resource/action. Effective-time and expiration
uncertainty is exposed to the LLM, not guessed from keywords. Permission event
history remains available for scope changes and conditional interpretation.

Cache schema is `govmem-v8-event-store-2`. Use a fresh output directory for this
revision. Corrupt or incompatible caches raise an error instead of silently
losing governance state. Model/config/source fingerprints must match on resume.

## Small public ontology

`src/gov_mem/extraction/v8_scene_schema.py` publishes one schema for all domains:
entity/relation types, allow/deny/revoke, lifecycle operators and duty attributes.
It contains no GateMem entity names, answer values, per-domain classifiers,
fixed role permissions, or benchmark-derived phrases. Concrete instances and
scope descriptions always come from the observable prefix. The old lexical
compatibility module remains empty; legacy v4 behavior is not rewritten.

T-Mem was inspected as a design reference for explicit structural categories,
prompt contracts and per-call cost accounting (commit
`dd9e1527bc75908485809580c9520af5a9a42879`). Its task-specific prompts, trigger
examples, thresholds, retrieval stages and code were not imported into V8.

## Calls and cost

- Normal checkpoint: one joint Stage-2 call plus one answering call.
- Entirely blocked/no-memory checkpoint: one Stage-2 call, no answering call.
- Same visible prefix: cached graph reused, no extraction-only call.
- Cold-prefix exception: over 64 new turns or 24,000 new-text characters invokes
  explicit prefill batches (32 turns each, at most four extra calls per checkpoint
  by default). Exceeding the budget raises a visible error, retaining completed
  cache batches for an explicit resume. No raw history is silently skipped.
- JSON parsing attempts: one in the V8 configs; HTTP attempts: at most two.
  Default `pipeline.fail_fast: true` stops the run at the first failed checkpoint
  after saving paid telemetry, instead of repeatedly paying through a bad run.
  Following live complete-episode protocol failures, `max_contract_repairs: 1`
  enables at most one extra Stage-2 generation with validator feedback. It is
  separately audited and counted; no gold feedback or unbounded retry loop.
  The paired evaluation enables `record_execution_errors`: terminal failures
  have `action=error`, an empty answer, and an error sidecar. Every checkpoint
  is attempted, errors are retained in official scoring, and a supplementary
  worst-case MGS treats errors as failures of their respective component.
- Model: `gemini-2.5-flash-lite`, temperature 0; embedding:
  `text-embedding-3-small`; embedding fallback disabled in the V8 configs.
- Record logical calls, actual provider requests (including HTTP retries),
  latency, prompt/completion tokens when returned, and response usage coverage.
  Preserve run totals on incomplete runs and accumulate them on strict resume.

Shared extraction can increase Stage-2 prompt/output size. Two calls is a
request-count design target, not evidence of lower token cost. Evaluate total
cost including ingestion, prefill, failures, retries and official judging
(judge cost reported separately). No dollar estimate is fabricated when
provider pricing/usage is unavailable.

## Validation and experiment protocol

Run local regressions:

```bash
PYTHONPATH=src pytest -q tests/test_v8_event_extractor.py \
  tests/test_v8_late_governance.py tests/test_v8_late_governance_backbone.py \
  tests/test_v8_state_projector.py tests/test_llm_client_protocol.py
```

Use `pytest tests`, not bare `pytest`, to exclude vendored dependencies' tests.
See the dated implementation validation report for results and known baseline
failures. Offline fake-provider checks establish orchestration and boundaries;
they do not measure LLM semantic quality, Utility, Privacy, Safety, or MGS.

A paid validation must use complete unseen episodes and a paired RAG-Naive
baseline with matching model, temperature, embedding, Top-20 and evaluator.
Do not pick individual checkpoints. Old Medical/Atlas/Beacon episodes are later
regressions, not development holdouts. Only the official protocol can produce
formal U/A/F/MGS results. Raw `dataset/GateMem/` remains read-only; no test gold
or future suffix enters extraction, permission reasoning or answering.

```bash
python scripts/audit_v8_extraction.py --output_dir <new-run-output>
```

The audit reports structure/calls, not extraction semantic F1. Episode-disjoint
human annotations are still required for that measurement. No paid GateMem
experiment was run for this implementation change.
