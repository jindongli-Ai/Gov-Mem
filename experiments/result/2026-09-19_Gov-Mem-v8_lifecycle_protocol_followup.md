# V8 lifecycle, source references and prompt growth follow-up

A complete 18-checkpoint Education pilot improves official single-episode MGS
from 5.56% to 55.56%, with two execution errors. It is a development diagnostic,
not a four-domain performance result. Current source includes further offline
fixes and was evaluated separately on the original fixed 12 episodes; see the
[completed paired report](2026-09-19_Gov-Mem-v8_source_projection_random3_per_domain_paired.md).

## Implementation

- Active, unconditional deletion events now veto exact bound claims before
  Stage 3. Source-grounded deleted classifications also reach the symbolic
  critic even when contradicted by a release proposal. Updates, cancellations,
  similar unbound values, conditional scopes and future events do not create
  deletion vetoes. Future starts stay pending even when expiry is specified.
- Valid query-independent ingestion commits before shallow claim normalization.
  A malformed claim no longer causes valid history extraction to be repeated.
  Invalid event sources still prevent watermark advancement.
- Single-turn candidate IDs use actual source turn IDs. Grounded explicit denials
  can omit empty binding arrays; released claims still require explicit bindings.
- Post-pilot, exact graph/ingestion source quotations already visible to Stage 2
  are also selectable, without hidden history access. One supplied excerpt must
  contain all fragments of a turn; disjoint snippets are never concatenated.
- Post-pilot, the query graph folds superseded unconditional advisory updates
  under an identical issuer/grantee/resource/action/context binding. All deletion,
  scoped/conditional/timed events and permissions remain. The append-only store
  and raw text are unchanged. Duplicate resource rows and empty optional fields
  are omitted without another model call.

## Paid pilots

Both pilots used the first Education episode in the fixed manifest, selected
before execution: `education_episode_custom_en_004_orchid_committee_orchid_commons_dual_track`.
Memory/answering used Gemini 2.5 Flash-Lite, temperature 0.

The first attempt, `outputs/v8_shallow_lifecycle_pilot_20260919`, stopped after
10/18 checkpoints with four errors. The circuit breaker exposed missing empty
bindings and candidate/source ID confusion. It was not scored. Frozen runtime:
`ac8f2d028bd0073cf92059a3c95ed583526897627703b855828717c91ccf3004`.

The second attempt, `outputs/v8_shallow_source_ids_pilot_20260919`, completed all
18 checkpoints including two errors. Eighteen GPT-4o official judgments parsed
successfully, with no applicable null labels and `gate_by_action=false`.
Frozen runtime: `925880f2729eefd1edb15c59a3960bc89eb2c178fb316538edf0c724ac4a4d32`.

| Same complete episode | U | A | F | Official MGS | Errors |
|---|---:|---:|---:|---:|---:|
| RAG-Naive, reused official judgments | 66.67% | 50.00% | 83.33% | 5.56% | 0 |
| Previous shallow V8 | 66.67% | 50.00% | 83.33% | 5.56% | 1 |
| Source-ID pilot V8 | 66.67% | 16.67% | 0.00% | 55.56% | 2 |

Counting failures pessimistically gives pilot MGS **44.44%**. There are only six
questions per component; this repeatedly used development episode establishes
neither generalization nor statistical significance. Official source hashes
were verified before comparing cached baseline results. Pilot metrics and
provenance: `outputs/v8_shallow_source_ids_pilot_20260919/pilot_scored/pilot_metrics.json`.

The pilot has three event-bound claims and two bound deletion vetoes; both
confirmed language blocks. They are not independent symbolic interventions.
Separately, the new deletion critic changes eight additional bound release
proposals to blocks when replaying the 256 normal checkpoints from the previous
full run. This is a deterministic decision audit, not eight proven prevented
leaks or a remeasurement of those answers.

## Usage and limitations

| Phase | Model | Calls | Prompt tokens | Completion tokens | Total |
|---|---|---:|---:|---:|---:|
| Incomplete pilot | Gemini Flash-Lite | 20 | 105,130 | 4,378 | 109,508 |
| Complete pilot | Gemini Flash-Lite | 29 | 209,962 | 9,447 | 219,409 |
| Official pilot judgments | GPT-4o | 18 | 13,846 | 944 | 14,790 |

Previously the same episode used 38 V8 calls / 195,781 tokens; baseline used
18 calls / 41,477 chat tokens. Fewer calls did **not** reduce tokens: pilot usage
rose 12.1%, to 5.29 times baseline. Repeated ordinary updates accumulating in
graph context contributed substantially. Warm exact-text embeddings were reused;
no graph cache was reused. All failed-attempt usage remains reported, and no
monetary prices are assumed.

Subsequent advisory projection reduces graph characters 73.38% and total prompt
characters 17.02% on the same 22 captured Stage-2 inputs while retaining all
deletions. These are offline character measurements, not actual token savings.
See `2026-09-19_v8_shallow_projection_size.json` and
`2026-09-19_v8_lifecycle_followup_http_costs.json`.

## Post-pilot offline checks

- Same 22 captures: 16 -> 20 pass, four recovered, zero regressions. Both final
  source-reference errors validate on their original supplied inputs. Remaining
  failed captures are malformed JSON and an unsupported deletion classification.
  These request counts include repairs; replay does not evolve memory or generate
  new answers and cannot establish a new MGS.
- Same earlier 413 captures: 334 pass / 79 fail, compared with 330 / 83 at the
  start of this follow-up.
- Targeted lifecycle/source/flow tests: 79 passed. Repository tests: **463 passed,
  13 existing legacy failures**, with the same failure identities.

Current source differs from the measured pilot. The next 12-episode runtime is
frozen at `1346fd7a2492030d310809430553f9ce6ba9156ee87684d27539a82e71e1941c`
in `outputs/v8_shallow_source_projection_random3_20260919`.
Its initial cache copy exhausted shared filesystem inodes before any paid
request. Exact duplicated caches created during this follow-up were verified
and hard-linked to recover space; no code or experiment records were removed.
