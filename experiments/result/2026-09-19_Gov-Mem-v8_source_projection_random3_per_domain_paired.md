# V8 source projection: complete-episode paired development evaluation

**Current frozen V8 obtains 40.06% official mean-domain MGS versus 19.75% for
RAG-Naive (+20.31 percentage points).** All four domain MGS values exceed the
baseline. Penalizing execution errors and unknown judgments pessimistically
still gives V8 **32.91%**. This improves on the previous shallow V8's 22.19%
official / 8.28% pessimistic MGS, but **17 execution errors remain** and token
cost has not improved. This repeatedly used development sample is not a holdout
or a full GateMem benchmark claim.

## Scope, identity and completion

- Unchanged manifest: `experiments/manifests/v8_paired_random3_per_domain_seed20260918.json`.
- Four domains × three complete randomly selected episodes, 303 checkpoints
  per system. No checkpoint or error was removed; each has an official judgment.
- Current source matches measured runtime SHA256:
  `1346fd7a2492030d310809430553f9ce6ba9156ee87684d27539a82e71e1941c`.
- Inference: `outputs/v8_shallow_source_projection_random3_20260919`.
- Metrics, audit and scoring: `outputs/v8_shallow_source_projection_random3_20260919_scored`.
- Memory/answering: Gemini 2.5 Flash-Lite, temperature 0. Official judge:
  GPT-4o / OpenLux, temperature 0, `gate_by_action=false`.
- Reused baseline inference from `outputs/v8_paired_random3_seed20260918_attempt2/rag_naive`
  and official judgments from `outputs/v8_paired_random3_seed20260918_scored`.
  Exact prediction equality, official source hashes, unique complete judgments,
  parse success and judging settings were checked before reuse. Old judge
  telemetry was not copied as new paid usage.
- All dataset and frozen config hashes remain unchanged. Completion checks are
  in `completion_audit.json`; reuse provenance is in `reused_baseline_scores.json`.

The initial setup exhausted shared filesystem inodes during cache copying,
before any paid request. Only identical cache copies made during this follow-up
were verified and hard-linked to recover space; no code or experiment record
was removed. A Household episode later hit the three-consecutive-error circuit
breaker with 21/24 checkpoints recorded. Strict resume ran only the remaining
three under the identical snapshot; previous errors were retained. All 12
completion markers now exist.

## Official results

U is utility accuracy; A and F are privacy and deletion leakage rates (lower
is better). MGS = U × (1−A) × (1−F), averaged equally across the four domains.

| Domain | Checkpoints | V8 U | V8 A | V8 F | V8 MGS | RAG-Naive MGS | V8 errors | Worst-case V8 MGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Medical | 81 | 96.55% | 33.33% | 16.00% | 54.07% | 22.56% | 5 | 43.77% |
| Office | 96 | 62.96% | 56.67% | 46.15% | 14.69% | 9.75% | 6 | 10.66% |
| Education | 54 | 88.89% | 22.22% | 50.00% | 34.57% | 4.63% | 2 | 30.73% |
| Household | 72 | 75.00% | 20.83% | 4.17% | 56.90% | 42.06% | 4 | 46.48% |
| Mean / total | 303 | | | | **40.06%** | **19.75%** | **17** | **32.91%** |

Judge parse failures and applicable null labels are both zero. V8 has 286 normal
outputs and 17 execution errors (5.61%), versus baseline zero errors. The previous
shallow run had 47 errors (15.51%). The pessimistic supplement treats technical
failures as incorrect utility or leakage, rather than benefiting from an empty
or failed answer.

The improvement is not uniform across components. Office utility is **62.96%**
versus baseline **74.07%**; Education deletion leakage remains **50.00%**. Those
limitations must accompany the mean MGS improvement. Public access policies
match the official baseline; the previously documented temporal-scaffold
attribute asymmetry remains. No causal ablation isolates the symbolic component.

## What changed and actual symbolic execution

The raw text bank and RAG-first retrieval remain. Stage 2 uses the shallow
permission/lifecycle ledger, source-ID candidate references, exact already-visible
graph/ingestion excerpts and valid ingestion committed before claim normalization.
The query graph folds superseded unconditional advisory updates under exact
issuer/grantee/resource/action/context identity, keeping every deletion and the
raw append-only history. The symbolic critic now enforces both active bound
permission denials and bound deletions before Stage 3; future, conditional and
ambiguous scopes do not become unconditional vetoes.

`runtime_audit.json` records 354 Stage-2 responses including repairs, 573 validated
claims, and 42 event-bound claims across 29 normal checkpoints. There are two
bound permission vetoes and 17 bound deletion vetoes. Matching the captured
original proposals shows **eight independent release-to-block changes**, including
six involving deletion. These prove an active symbolic decision path, not eight
proven prevented leaks. One veto's original proposal could not be resolved and
was not counted as an independent intervention. Synthetic tests also verify
that future deletions, updates/cancellations and unrelated matching values are
not blocked by these rules.

## Actual usage: fewer calls, more tokens

| Inference on the same 303 checkpoints | Chat calls | Provider-reported chat tokens |
|---|---:|---:|
| RAG-Naive (reused) | 303 | 639,831 |
| Previous shallow V8 | 619 | 3,164,624 |
| Current V8 | **522** | **3,336,077** |

Current calls decrease 15.67%, but total tokens increase **5.42%** versus the
previous shallow implementation and remain **5.21× baseline**. Input tokens:
3,208,001; output tokens: 128,077. Preserve the provider total, which differs by
one token from its reported components. No monetary price is assumed.

New official scoring used **303 GPT-4o calls / 246,686 tokens** (230,897 prompt,
15,789 completion), with no recorded HTTP errors. All 522 captured inference
requests used Gemini Flash-Lite and returned HTTP 200. HTTP success does not
mean the model satisfied the runtime contract.

Earlier pilots in this follow-up are additional costs, not hidden in the full-run
total: incomplete pilot **20 Gemini calls / 109,508 tokens**; complete Education
pilot **29 Gemini calls / 219,409 tokens**, plus **18 GPT-4o judgments / 14,790
tokens**. The pilot is documented separately in
[the lifecycle/protocol follow-up](2026-09-19_Gov-Mem-v8_lifecycle_protocol_followup.md).
The older TAB diagnostic and previous full-run costs remain in their original
reports. Warm embedding cache provenance is recorded; no graph state was reused.
The whole follow-up (both pilots and this full run) totals **571 Gemini calls /
3,664,994 tokens**, plus **321 GPT-4o judging calls / 261,476 tokens**. See
`followup_total_usage.json`; these totals exclude the preceding full runs.

The offline 17.02% prompt-character reduction on a fixed pilot input did not
translate into lower full-run token usage. Actual captured graph-context text
grew from **819,790** to **2,447,963** characters across the old/new runs, despite
fewer Stage-2 responses. More surviving/repeated extracted state can grow the
graph enough to offset the query-view compaction. This remains a cost bottleneck;
see `2026-09-19_v8_full_prompt_breakdown.json`. Do not call the cost problem solved.

## Validation and remaining failures

Targeted lifecycle/source/flow tests: **79 passed**. Repository `tests` suite:
**463 passed, 13 existing legacy failures**, unchanged in identity.
`git diff --check` passes. Current source, frozen configurations, dataset hashes,
12 complete episodes and 303 unique predictions/judgments per system were checked.

The 17 errors comprise four structural JSON failures, one conflicting event ID,
one unknown event reference, one event-resource grounding failure, and ten
incomplete/inconsistent claim or event envelopes. No missing binding, invalid
source, unsupported deletion label or malformed JSON was silently turned into
a favorable refusal for scoring. Further reliability work must preserve that
boundary, and further cost work must be tested on actual evolving episodes.
