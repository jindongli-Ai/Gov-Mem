# Gov-Mem V8: four-domain complete-episode paired evaluation

Status: completed. Both systems attempted all 303 checkpoints across the same
12 complete episodes; all 606 official judge responses parsed. Current V8
underperforms the official baseline and is **not ready** as the intended method.

## Selection and isolation

The user authorized paid testing of three random **complete episodes** in each
of four domains. Selection was fixed before inference in
`experiments/manifests/v8_paired_random3_per_domain_seed20260918.json`.
Seed 20260918; domain-specific Python RNG; sample three from sorted eligible
IDs. Three earlier development episodes were excluded. No checkpoint type,
answer, gold evidence, judge label or performance was used for sampling.

| Domain | Episodes | Checkpoints |
|---|---:|---:|
| Medical | 3 | 81 |
| Office | 3 | 96 |
| Education | 3 | 54 |
| Household | 3 | 72 |
| Total | 12 | 303 |

Raw GateMem files remain read-only; source SHA256 hashes are in the manifest.
All checkpoints of the selected episodes are retained, including execution
errors. This is a 12-episode sample, not the full GateMem benchmark.

## Frozen implementation and matched models

Gov-Mem: existing V8, frozen runtime SHA256
`f4e68a14a19a75e8a4b94b090398e1aee9c2e7df510ce5da801fd280545d439d`.
Predictions: `outputs/v8_paired_random3_seed20260918_attempt8/govmem`.
Baseline: original vendored official **plain RAG-Naive**, not typed rerank v3.
Predictions: `outputs/v8_paired_random3_seed20260918_attempt2/rag_naive`.
Both use `gemini-2.5-flash-lite`, temperature 0, `text-embedding-3-small`,
turn retrieval Top-20, max output 4096. V8 additionally uses its documented
bounded adjacent-turn context and source-grounded graph; this is part of the
method, not an identical-prompt comparison. Baseline incremental ingestion
is the unmodified official implementation.

**Material input/prompt asymmetry:** the official RAG baseline supplies a
per-domain public GLOBAL ACCESS POLICY (`bench/domains.py`) and requester-
relevant episode-scaffold relationship attributes (`bench/agents/base.py`).
Current V8 deliberately removed domain-specific policy prompts and filters
relationship metadata to avoid future-summarizing access attributes. Thus
matching model, temperature and Top-20 does **not** make this a controlled
same-policy/same-metadata ablation. The official baseline's scaffold formatting
preserves attributes as supplied; prefix safety of any temporal access summaries
in those attributes was not established here. Scores are for these two actual
configurations, not proof that the governance architecture alone caused the gap.
Public task access rules are conceptually different from answer/entity lexicons
or gold labels. A future controlled comparison should align allowed policy and
as-of-safe relationship inputs on both sides without importing gold or future
summaries. The current official baseline was kept unchanged for transparency.

Final scoring: original official scorer, `gpt-4o` at temperature 0,
`gate_by_action=false`. Official U uses effective utility (utility_ok AND
action_ok), A is privacy leakage rate, F is deletion leakage rate, and
MGS = U × (1−A) × (1−F). Overall comparison uses the arithmetic mean of
four domain-level MGS values. Higher U/MGS and lower A/F are better.

## Integration failures and interpretation

Before freezing this evaluation, attempts 1–7 exposed JSON/grounding protocol
failures on the sampled Medical episodes. Changes used only runtime prompts,
responses and validator failures, never gold answers or judge scores. They
were generic source/contract fixes, not domain/entity vocabularies. All attempt
outputs and costs are retained. Nonetheless this sample was used for integration;
it must not be described as an untouched preregistered holdout or conclusive
paper evidence. No semantic code changes occurred after final snapshot freeze.

Normal V8 processing uses one joint graph/permission call and one answer call;
fully blocked requests omit the answer call. A contract failure permits **one**
additional Stage-2 generation with validator feedback. HTTP attempts are bounded
at two. An exhausted failure is exported as `action=error`, empty answer,
`execution_status=error`; it is never relabeled as a successful refusal or
removed from the denominator. Errors and successful checkpoint counts are
reported explicitly. Because empty error responses cannot leak, supplementary
**execution-error worst-case MGS** treats every failed utility checkpoint as
incorrect and every failed privacy/deletion checkpoint as a component failure.
This is an availability-aware sensitivity measure, not a replacement official
metric or a claim that the error actually leaked information.

A provider DNS address caused SSL failures in the early baseline attempt.
Metering now uses persistent connections, a bounded connection timeout, and
prefers a responsive address while retaining original hostname/TLS and DNS
fallbacks. Model prompts and official baseline algorithms were not changed.
Failed and resumed request costs are retained. The final V8 inference run reused
previously paid embeddings for identical texts/model from the integration
attempts; its final-run embedding cost is therefore a warm-cache measurement.
Do not compare it to a fresh baseline as cold-start end-to-end cost. Memory
chat calls were rerun for the frozen implementation. No monetary costs are invented
without verified provider pricing; report requests and provider token usage.

## Validation

Final source validation: `PYTHONPATH=src pytest -q tests --tb=no`:
386 passed; the same 13 pre-existing legacy failures in field_state_projection
and stateful_policy reproduced before this work. No new failing test.
Regression coverage includes bounded repair count, watermark preservation,
exact provenance, omitted unsupported releases, fatal malformed denials, and
retaining every attempted checkpoint as an explicit error when necessary.

## Results

| Domain | System | U ↑ | A ↓ | F ↓ | MGS ↑ | Execution errors |
|---|---|---:|---:|---:|---:|---:|
| medical | govmem | 82.76% | 92.59% | 100.00% | 0.00% | 3/81 |
| medical | rag_naive | 79.31% | 40.74% | 52.00% | 22.56% | 0/81 |
| office | govmem | 74.07% | 83.33% | 76.92% | 2.85% | 12/96 |
| office | rag_naive | 74.07% | 63.33% | 64.10% | 9.75% | 0/96 |
| education | govmem | 16.67% | 94.12% | 77.78% | 0.22% | 7/54 |
| education | rag_naive | 27.78% | 50.00% | 66.67% | 4.63% | 0/54 |
| household | govmem | 79.17% | 37.50% | 70.83% | 14.43% | 6/72 |
| household | rag_naive | 75.00% | 29.17% | 20.83% | 42.06% | 0/72 |

Four-domain mean official MGS: **Gov-Mem 4.37% vs RAG-Naive 19.75%**;
paired difference **−15.37 percentage points**. Every domain favors RAG-Naive.
Execution availability: **275/303 (90.76%)** for Gov-Mem, **303/303 (100%)**
for RAG-Naive. Conservative error/unknown-label worst-case mean MGS is
**2.12% vs 19.75%**. This supplementary measure does not treat error abstentions
as successful privacy/deletion behavior.

One Gov-Mem Education privacy judge returned `privacy_leak=null` despite valid
JSON (checkpoint `education_episode_custom_en_004_orchid_committee_orchid_commons_dual_track_ckpt_08`).
The unmodified official scorer excludes null component labels: Education A
uses 17/18 privacy labels for Gov-Mem, versus 18/18 for baseline. This missing
label is explicitly reported, not silently treated as safe. The supplementary
worst-case score counts it as a privacy failure. All 606 checkpoint judgments
are present; there were zero judge parse or HTTP failures. No paid rejudging
was done to choose a preferred result.

## Requests and provider-reported usage

| Work | Chat requests | Requests/checkpoint | Prompt tokens | Completion tokens | Total tokens |
|---|---:|---:|---:|---:|---:|
| Final V8, including contract failures/repairs | 618 | 2.040 | 3,312,265 | 188,008 | 3,500,275 |
| Official RAG-Naive | 303 | 1.000 | 584,060 | 55,771 | 639,831 |
| Extra V8 integration attempts 1–7 | 224 | — | 1,044,815 | 69,406 | 1,114,221 |
| Official gpt-4o judge, both systems | 606 | — | 486,492 | 31,664 | 518,156 |

Final V8 chat total tokens are **5.47× baseline**, despite 2.04× requests.
Stage 2: 355 requests, 3,330,376 tokens (~95.15% of V8 chat tokens).
Answering: 263 requests, 169,899 tokens. No prefill calls occurred. The 355
Stage-2 calls comprise 303 initial attempts and 52 repair generations: 25
repairs yielded normal predictions; 27 repaired attempts ultimately ended in errors.
Two errors occurred during answering after Stage 2; one followed a successful
Stage-2 repair. Of the 52 repairs, 26 exhausted Stage-2 validation and 26
passed it (25 ultimately yielded normal predictions). All memory-model
HTTP responses reported Gemini; no gpt-4o-mini was requested by this experiment.

Final V8 embeddings: 467 requests, 61,448 tokens, zero HTTP failures; this reused
warm cached embeddings. Extra V8 attempts: 106 embedding requests, 9,831 tokens.
Baseline embeddings: 2,989 requests including eight SSL failures, 94,257 reported
tokens. Baseline resumed ingestion is included. All successful API responses
included usage. Provider total_tokens is preserved as returned; for V8 it differs
by two tokens from the sum of prompt_tokens and completion_tokens. No dollars
are estimated, and failed unreported usage may still be billed by the provider.

The additional V8 debugging expense was real and did not produce a successful
performance improvement. It is reported separately, not hidden in a selectively
successful attempt. No further paid inference is running for this experiment.

Machine-readable artifacts:
- `outputs/v8_paired_random3_seed20260918_scored/paired_metrics.json`
- `outputs/v8_paired_random3_seed20260918_scored/http_costs.json`
- `outputs/v8_paired_random3_seed20260918_scored/runtime_audit.json`
- `outputs/v8_paired_random3_seed20260918_scored/official_source_identity.json`

## Implication for the next implementation

Do not present this snapshot as a successful Gov-Mem result. The next change
must restore actual event-to-claim bindings, supply legitimate public access
policy consistently without benchmark-derived answer lexicons, and preserve
relevant restriction context at the final boundary. Joint graph extraction and
permission classification also need a simpler reliable contract. Validate these
properties offline, then use a new independent complete-episode sample; do not
keep tuning and rescoring these same episodes. This report makes no claim that
these proposed changes have already improved performance.


## Reproduction

```bash
python scripts/run_v8_paired_episode_suite.py \
  --manifest experiments/manifests/v8_paired_random3_per_domain_seed20260918.json \
  --output outputs/v8_paired_random3_seed20260918_attempt8 \
  --system govmem --workers 3 --resume
python scripts/run_v8_paired_episode_suite.py \
  --manifest experiments/manifests/v8_paired_random3_per_domain_seed20260918.json \
  --output outputs/v8_paired_random3_seed20260918_attempt2 \
  --system rag_naive --workers 3 --key_offset 4 --resume
python scripts/score_v8_paired_episode_suite.py \
  --govmem_root outputs/v8_paired_random3_seed20260918_attempt8 \
  --baseline_root outputs/v8_paired_random3_seed20260918_attempt2 \
  --output outputs/v8_paired_random3_seed20260918_scored
```

Completed markers skip existing episodes. Existing incomplete runs require
explicit resume and matching frozen identity. Error rows count as attempted
checkpoints and are not automatically regenerated to cherry-pick successes.
A fresh reproduction must use a new output directory to avoid reusing results.
All credentials are loaded from configured environment/key discovery; none
are included in reports or telemetry.

## Diagnostic interpretation (frozen predictions only)

The complete Medical sample illustrates a failure of the governance boundary:
of 78 non-error predictions, 76 were `answer`, one was `refuse`, and one was
`answer_redacted`. The validated claim ledgers contained only five blocked
claims. No `bound_explicit_permission_veto` fired in this domain. Official
judgments identified disclosure of clinical information to logistics-only
requesters and recovery of explicitly deleted callback information. Source
exactness alone therefore did not establish authorization or lifecycle safety.

This is not evidence that symbolic reasoning should be removed. The implemented
hard veto requires an event ID plus exact resource/action/scene binding; these
bindings rarely occur in the compact model output. The graph existing on disk
is not proof that its constraints influenced the released claims. When Stage 2
erroneously releases a value, Stage 3 receives approved snippets without the
original restriction context, so the current boundary can perform worse than
plain RAG, whose answer model still sees restrictions among the retrieved turns.
These are observed failure paths; their relative contribution would require
new, independent controlled evaluations. No scoring feedback was used to alter
this frozen run.

The final runtime audit sharpens this diagnosis: across all 275 non-error
predictions, **no accepted claim had a nonempty resource_id, action, scene_id,
or permission_event_ids binding**. Consequently the explicit bound-permission
hard veto fired **zero times**. The 19 deletion-tombstone annotations confirmed
claims already blocked by the language reasoner; they did not substitute for
an independently active permission constraint. The compact output contract
made these binding fields optional, and the live model omitted them. Synthetic
tests supplying those fields establish code behavior when present, not that
the deployed model supplies them. This is an implementation readiness failure,
not a successful validation of the intended symbolic-governance design.

## Subsequent user-requested change (not part of these results)

After seeing the result, the user requested one information item per line
instead of model-generated JSON. Current workspace V8 now uses a plain-text
record protocol for history extraction/Stage 2, and mandatory explicit event selection
expanded to canonical bindings by Python. Internal storage remains program-generated
JSON; Stage 3 retains its original API wrapper containing the text answer.
This is only offline-tested; no
paid retest or improved MGS is claimed. This report, attempt8 runtime snapshot,
and all predictions/judgments retain the original evaluated implementation.
