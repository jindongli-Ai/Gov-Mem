# Gov-Mem Version Log

This file is the version identity record for the paper codebase. A new
framework revision must add an entry here and must not overwrite the identity
of an earlier benchmark snapshot.

## Version Lineage

| Version | Canonical implementation | Meaning | Paper status |
|---|---|---|---|
| `Gov-Mem-v3.0` | `rag_naive_v3_typed_rerank` | Frozen RAG-Naive Stage 1 plus typed Stage 2 reranking and source-bound answer handling | Baseline/comparison track |
| `Gov-Mem-v3.1-StructuredProvenance` | `rag_naive_v3_typed_rerank` plus typed GateMem provenance preservation | Completed transition snapshot; restores structured turn metadata after retrieval without changing embedding input or retrieval ranking | Engineering recovery point |
| `Gov-Mem-v4-Symbolic-dev0` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | First minimal Symbolic step: typed principal-role consistency annotation | Previous development method |
| `Gov-Mem-v4-Symbolic-dev1` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | Lightweight Evidence-Principal provenance graph used as Stage 2 auxiliary context | Previous development method |
| `Gov-Mem-v4-Symbolic-dev2` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | Typed Principal-Entity/Resource relation graph from GateMem `entities.relationships`, used as Stage 2 auxiliary context | Current development method |
| `Gov-Mem-v4-Symbolic-dev3-state-ledger` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | Retrieved-evidence-only source-bound state ledger with typed slot provenance and conflict accounting | Current development method |
| `Gov-Mem-v4-Symbolic-dev5-temporal-authorization` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | Shadow temporal authorization state graph over Principal/Role/Resource/PolicyEvent nodes; provenance-grounded allow/deny/revoke/supersedes transitions | Validation candidate; not paper-frozen |
| `Gov-Mem-v4-Symbolic-dev6-authorization-contract` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | Ingestion-time, provider-neutral authorization assertions with source spans; retrieval consumes the typed contract before temporal graph resolution | Exploratory validation; not paper-frozen |
| `Gov-Mem-v4-Symbolic-dev7` | `rag_naive_v3_typed_rerank` plus `govmem_v4_symbolic` | Authorization-aware evidence boundary with deterministic claim-level provenance verification at final delivery | Current development method |
| `Gov-Mem-v4-Symbolic` | The frozen v4 line after dev0 regression and benchmark checks | RAG-Naive foundation plus deterministic/neuro-symbolic role, permission, temporal, and consistency reasoning | Paper method name |

## 2026-08-19: Gov-Mem-v4-Symbolic-dev7 claim-level provenance verifier

The current framework identity remains `Gov-Mem-v4-Symbolic-dev7`; this
increment does not introduce another version name. The final delivery gate now
performs a deterministic claim-level provenance audit over the existing field
state projection. For each supported field it checks that every selected value
is present in the delivered answer, has a source memory, stays within the
policy-approved memory set, and is supported by the source text (and any
provided source span). A deterministic claim attached to an `unknown`,
`conflict`, or `restricted` field fails closed. The audit is provider-neutral,
domain-neutral, uses no new LLM call, and is recorded in
`answer_grounding.policy_privacy_verifier.claim_provenance`.

This is a structural provenance boundary, not a free-form fact extractor: the
field-state projection remains the single authority for selection and temporal
resolution. The implementation is covered by the policy, answer projection,
and stateful projection regression tests. No benchmark result is promoted by
this code change; the next performance measurement should be run only after
the framework design is frozen.

On 2026-08-19 this verifier was also integrated into the actual
`govmem_v4_symbolic` RAG-Naive benchmark path. The path keeps the existing
Stage 1/Stage 2 retrieval and answer flow, asks the answering model for an
optional source-bound `claim_contract` in the same JSON response, resolves
GateMem source-message IDs only against the retrieved chunk rows, and records
the verifier audit in the official prediction's `memory_audit`. Missing claim
contracts are recorded as `claim_provenance_not_applicable`; they do not create
synthetic provenance or trigger a new model call. The adapter is a source
provenance boundary for this RAG-Naive path; the full policy-approved field
projection remains the authority in the stateful executor path. The RAG-Naive
adapter currently uses non-intervention claim-level provenance auditing
(`claim_provenance_enforcement: false`): contract failures remain visible in
the audit but do not replace an answer with `no_memory`. The audit is executed
on every answer path when the verifier is enabled; it is not a placeholder. If
a field cites several retrieved chunks, normalization keeps a source span for
each cited chunk, while the typed state ledger can only fill a missing binding
from the current retrieved evidence. This is deliberate because the first real
smoke showed that the base answer model does not yet emit complete source spans
consistently; enabling answer suppression before that contract quality is
validated would confound the framework test with an avoidable answer-
suppression artifact.

## 2026-08-17: Gov-Mem-v4-Symbolic-dev6 full Medical small-embedding validation

The complete Medical domain was evaluated across all 21 episodes and 579
checkpoints with OpenLux `gpt-4o-mini`, OpenLux
`text-embedding-3-small`, and the official OpenLux `gpt-4o` GateMem judge.
The run used two bounded episode workers, one request in flight per worker,
local scratch storage, and a 90-second embedding request timeout to tolerate
intermittent OpenLux latency. All 579 predictions and all 579 official judge
scores completed with zero judge parse failures and complete context auditing.

| System | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Gov-Mem-v4-Symbolic-dev6 authorization contract | 68.57% | 44.27% | 9.04% | **34.76%** |

This is a complete-domain exploratory development result, not a frozen
four-domain paper-table value or a paired causal ablation. Full record:
`experiments/result/2026-08-17_Gov-Mem-v4-Symbolic-dev6-authorization-contract_medical_full579_embedding3small.md`.

## 2026-08-16: Gov-Mem-v4-Symbolic-dev4 policy consistency diagnostic

The opt-in `symbolic.policy_consistency` layer was tested on two complete
Medical episodes (56 checkpoints) using OpenLux `gpt-4o-mini`,
`text-embedding-3-small`, and the official OpenLux `gpt-4o` judge. The run used
two episode workers with one request in flight per worker; judge coverage was
56/56 with zero parse failures.

| Scope | Dev3 MGS | Dev4 MGS | Delta |
|---|---:|---:|---:|
| Medical 002 cardiology | 11.11% | 6.67% | -4.44 pp |
| Medical 003 hepatitis C | 70.00% | 70.00% | +0.00 pp |
| Two-episode aggregate | 36.67% | 36.11% | -0.56 pp |

This is a small stochastic diagnostic, not a causal ablation. Since it did not
show a positive effect, dev4 is not promoted: the canonical development line
remains `Gov-Mem-v4-Symbolic-dev3-state-ledger`. The policy consistency code is
retained as an explicit opt-in experiment with enforcement disabled by default
for future redesign and testing. Detailed metrics are recorded in
`experiments/result/2026-08-16_Gov-Mem-v4-Symbolic-dev4-policy-consistency_medical_2_retry.md`.

## 2026-08-16: Gov-Mem-v4-Symbolic-dev5 temporal authorization state graph

The next lightweight Symbolic increment is an opt-in shadow graph over typed
`Principal`, `Role`, `Resource`, and `PolicyEvent` nodes. It consumes provider-
neutral structured authorization events when available and otherwise accepts
only conservative generic authorization sentence patterns. Events require a
retrieved source and a `turn_index` or timestamp; missing temporal provenance
is recorded as `unknown`. Events after the checkpoint are ignored, later
allow/deny/revoke events update the same principal-resource state, and
same-time opposing events produce `unknown` rather than an arbitrary winner.

The certificate is passed to Stage 2 only as auxiliary structured context.
Candidate order, retrieval filtering, answer authorization, and LLM call count
are unchanged; `enforcement=false` is mandatory in this development step.
Focused synthetic coverage is recorded in `tests/test_symbolic_evidence.py`.
The real-episode validation must be treated as a development diagnostic until
the complete four-domain framework rerun is finished.

## 2026-08-17: Gov-Mem-v4-Symbolic-dev5 four-episode Medical validation

Four complete Medical episodes (108 checkpoints) were evaluated with OpenLux
`gpt-4o-mini`, `text-embedding-3-small`, and the official OpenLux `gpt-4o`
judge. Four bounded episode workers were used, one request in flight per
worker. One transient embedding failure was recovered by strict resume; all
108 predictions and official judge cases completed, with zero judge parse
failures.

| System | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Dev5 temporal authorization | 82.05% | 32.43% | 12.50% | **48.51%** |
| Earlier dev3 diagnostic on the same four episodes | 71.79% | 37.84% | 12.50% | 39.05% |

This is a positive but non-causal subset signal. The runs are separate
stochastic executions. Context-level privacy and deletion leakage remained
51.35% and 28.13%, respectively, so the certificate remains shadow-only and
dev5 is not promoted to the paper-frozen method. The next review point is
authorization-event coverage and provider-neutral structured ingestion, not
another enforcement rule. Full record:
`experiments/result/2026-08-17_Gov-Mem-v4-Symbolic-dev5-temporal-authorization_medical_selected4.md`.

## 2026-08-17: Gov-Mem-v4-Symbolic-dev6 authorization contract validation

Dev6 adds only an ingestion-time, provider-neutral authorization contract.
Each visible turn preserves conservative `authorization_assertions` with an
effect, original subject/resource wording, and source span. Roster resolution
and temporal state updates happen after retrieval. If an assertion is present
but cannot be resolved, the implementation keeps it unknown instead of
falling back to a broader second text parse. Dense retrieval order, candidate
filtering, enforcement, and LLM call count are unchanged.

The first `text-embedding-3-small` attempt was incomplete because of repeated
OpenLux embedding read timeouts and produced no valid score. A separate
exploratory rerun used `text-embedding-3-large`; it completed 108/108
predictions and 108/108 official judge scores with zero parse failures. The
official metrics were:

| System | Embedding | U | A | F | MGS |
|---|---|---:|---:|---:|---:|
| Dev6 authorization contract | `text-embedding-3-large` | 87.18% | 35.14% | 9.38% | **51.25%** |

Compared with the separate dev5 run (`48.51%` MGS), this is `+2.74 pp`, but
the difference is not causal because the embedding model and stochastic LLM
execution changed. Runtime prompt audit found 23 retrieved structured records
carrying 23 authorization assertions; temporal certificates had positive
event counts in 11 of 66 certificate-bearing checkpoints. The certificate
therefore remains shadow-only, and dev6 is not paper-frozen. Full record:
`experiments/result/2026-08-17-Gov-Mem-v4-Symbolic-dev6-authorization-contract_medical_selected4_embedding3large.md`.

## 2026-08-16: Gov-Mem-v4-Symbolic-dev4 half-Medical validation

The opt-in policy consistency layer was evaluated on 11 additional complete
Medical episodes (302 checkpoints), bringing the current dev4 diagnostic
coverage to 13/21 Medical episodes when combined with the earlier 002/003
probe. The run used OpenLux `gpt-4o-mini`,
`text-embedding-3-small`, the official OpenLux `gpt-4o` judge, five global
episode workers, and one request in flight per worker. The key pool contained
10 keys; at most five were leased concurrently. Official scoring covered
302/302 checkpoints with zero parse failures.

| Scope | U | A | F | MGS |
|---|---:|---:|---:|---:|
| New 11-episode dev4 aggregate | 50.93% | 40.00% | 4.26% | 29.26% |
| Dev4 aggregate including prior 002/003 | 53.13% | 38.14% | 3.57% | 31.69% |

The result is heterogeneous and does not produce a positive aggregate signal;
the nine-episode overlap with the earlier dev3 diagnostics is approximately
33.96% MGS for dev4 versus 34.93% for dev3, with stochastic and embedding-run
differences. The promotion decision is unchanged: keep
`Gov-Mem-v4-Symbolic-dev3-state-ledger` as canonical, and retain policy
consistency only as an explicit opt-in design under revision. Detailed results
are in
`experiments/result/2026-08-16_Gov-Mem-v4-Symbolic-dev4-policy-consistency_medical_11.md`.

## 2026-08-16: Gov-Mem-v4-Symbolic-dev4 full Medical conclusion

The complete 21-episode Medical set was evaluated (579/579 checkpoints) with
the same OpenLux `gpt-4o-mini` memory model and
`text-embedding-3-small` embedding. Five bounded episode workers were used,
with one request in flight per worker. The official OpenLux `gpt-4o` judge
scored every checkpoint with zero parse failures.

| System | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Frozen `rag_naive_v3_typed_rerank` | 64.29% | 43.75% | 9.04% | 32.89% |
| Dev4 policy consistency | 55.71% | 33.33% | 5.65% | **35.04%** |
| Delta | -8.58 pp | -10.42 pp | -3.39 pp | **+2.15 pp** |

The earlier small-sample negative signals were not representative of the full
Medical distribution. On the official Medical MGS objective, policy
consistency is positive by 2.15 percentage points. The utility drop and the
privacy/deletion leakage reductions must both be reported; this is not a
causal ablation because the runs are separate stochastic executions. Dev4 is
now a positive Medical-domain candidate, but final promotion still requires a
complete four-domain rerun. Full record:
`experiments/result/2026-08-16_Gov-Mem-v4-Symbolic-dev4-policy-consistency_medical_full579.md`.

## 2026-08-16: Medical exploratory completion

The current canonical development identity remains
`Gov-Mem-v4-Symbolic-dev3-state-ledger`; the version name is unchanged. The
remaining 15 Medical episodes (416 checkpoints) completed official GateMem
evaluation with OpenLux `gpt-4o-mini` as the memory-system LLM and `gpt-4o` as
judge. Because `text-embedding-3-small` intermittently timed out, the run was
continued with `text-embedding-3-large` after the partial small-embedding
phase. This is explicitly an exploratory mixed-embedding result and is not a
paper-table or clean ablation result. See
`experiments/result/2026-08-16_Gov-Mem-v4-Symbolic-dev3-state-ledger_medical_remaining15_exploratory_mixed_embedding.md`.

## 2026-08-14: Structured Provenance Recovery Snapshot

**Canonical name:** `Gov-Mem-v3.1-StructuredProvenance`

**Short name:** `Gov-Mem-v3.1-SP`

**Base framework:** `rag_naive_v3_typed_rerank` (`Gov-Mem-v3.0`)

**Purpose:** preserve the complete checkpoint-visible GateMem turn as typed
retrieval provenance so Stage 2 can make decisions from the original fields
instead of re-extracting them from natural-language text.

**Implemented in this snapshot:**

- Preserve `turn_id`/`message_id`, `turn_index`, `timestamp`, `speaker.principal_id`, `speaker.role`, `turn_kind`, original text, checkpoint `as_of_turn_id`, and the source turn object.
- Carry the structured record through `RAGChunk`, `MemoryItem`, and `RetrievedEvidence`.
- Expose the record to Stage 2 as valid JSON under `[STRUCTURED_RECORD]`.
- Expose the same typed provenance to the Stage 2 LLM reranker.
- Keep the Stage 1 embedding input, `top_k`, and retrieval protocol unchanged.

**Not yet implemented in this version:**

- No role/permission graph has been added.
- No temporal state graph, supersedes graph, deletion graph, or deterministic
  consistency checker has been promoted into the method.
- No new full-benchmark performance result is attributed to this snapshot.

Therefore this snapshot must not be reported as `Gov-Mem-v4-Symbolic` and must
not be mixed into the frozen v3 performance table without a separately named
experiment.

## 2026-08-14: Gov-Mem-v4-Symbolic-dev0

**Canonical development name:** `Gov-Mem-v4-Symbolic-dev0`

**Experiment mode:** `govmem_v4_symbolic`

This version keeps the v3 Stage 1 dense retrieval and typed Stage 2, then adds
one episode-local Symbolic check over retrieved evidence: the typed
`speaker.principal_id` and `speaker.role` must agree with the GateMem principal
roster. The result is passed to Stage 2 as an auditable annotation. It keeps
every Stage 1 candidate, does not reorder evidence, makes no new LLM call, and
does not infer permission or lifecycle state.

The implementation is intentionally a development snapshot. It is not yet a
reported full-benchmark result and must not be mixed with the frozen v3 table.

## 2026-08-14: Gov-Mem-v4-Symbolic-dev1

**Canonical development name:** `Gov-Mem-v4-Symbolic-dev1`

This increment adds a small episode-local bipartite provenance graph:

- `Evidence` nodes represent retrieved turn chunks.
- `Principal` nodes represent the speakers found in the GateMem principal roster.
- `spoken_by` edges connect each evidence node to its speaker.
- `graph_context` is passed to Stage 2 as auxiliary evidence metadata.

This graph does not replace dense retrieval, reorder candidates, filter
candidates, infer permissions, or make additional LLM calls. The v3 Stage 1
embedding input, `top_k`, candidate order, and candidate count are unchanged.

## Promotion Rule

The `Gov-Mem-v4-Symbolic` paper version will be frozen only when the
Symbolic design is implemented on top of this provenance-preserving base and
the following are frozen and tested:

1. The principal-role consistency step passes regression tests without changing v3 retrieval behavior.
2. Permission constraints are represented explicitly and checked deterministically.
3. Temporal updates, supersession, revocation, and deletion are source-bound and consistency-checked.
4. Stage 2 consumes the typed evidence contract without recovering metadata from prose.
5. A new configuration, experiment output directory, and full protocol record identify the v4 run.

## 2026-08-14: Gov-Mem-v4-Symbolic lifecycle increment

The paper-facing method name remains `Gov-Mem-v4-Symbolic`. The next internal
development increment adds only explicit lifecycle assertions: deleted,
revoked/retired, superseded/replaced, and explicitly updated evidence are
represented as lifecycle-event nodes linked by `asserts_lifecycle` edges.
Ordinary words such as `current`, `latest`, or `change` are not treated as
lifecycle proof. The increment does not reorder or filter candidates, make
permission decisions, or add LLM calls.

## 2026-08-14: Gov-Mem-v4-Symbolic-dev2 validity shadow audit

The implementation adds an `EvidenceValidityCertificate` for explicit
deleted, revoked, superseded, and updated lifecycle assertions. In this
development snapshot the certificate is internal shadow metadata only: it is
stored in evidence traces and reasoning artifacts, but is not exposed to the
Stage 2 or answer-model prompts. No candidate filtering, reordering, or new
LLM call is performed, and `enforcement_applied` remains false.

Two complete one-episode-per-domain diagnostic runs covered 101 checkpoints
(Medical 27, Office 32, Education 18, Household 24) with OpenLux
`gpt-4o-mini`, `text-embedding-3-small`, and the official OpenLux `gpt-4o`
judge. The second run verified zero certificate occurrences in all 101 answer
prompts and Stage 2 prompts, while retaining 1,987 internal certificate
records. Official judge metrics are diagnostic only and must not be mixed
into the 2,218-checkpoint paper table:

| Run | Medical MGS | Office MGS | Education MGS | Household MGS | Four-domain mean MGS | Checkpoint-weighted overall MGS |
|---|---:|---:|---:|---:|---:|---:|
| Certificate visible in prompt (audit run) | 20.00% | 44.44% | 41.67% | 65.63% | 42.93% | 43.62% |
| Certificate internal-only (validated run) | 20.00% | 55.56% | 27.78% | 54.69% | 39.51% | 41.32% |

The domain differences are not treated as a causal ablation because the
memory-model temperature is 0.2 and OpenLux outputs are stochastic. The
current evidence supports only the architectural boundary: validity
certificates must remain internal until enforced semantics are implemented
and evaluated under a controlled comparison.

The suite runner also now rehydrates a completed published prediction shard
into local scratch during `--resume`, avoiding an empty-scratch resume error
and preventing unnecessary model calls.

## 2026-08-15: Gov-Mem-v4-Symbolic-dev2 target-binding shadow v2

This run extends the dev2 validity shadow audit with deterministic lifecycle
target binding. Explicit lifecycle claims are conservatively linked to a
retrieved earlier target only when the target is present, temporally valid,
and uniquely matched. Ties are marked `ambiguous` and unmatched claims are
marked `unbound`; no target is guessed. The binding remains shadow metadata:
it does not filter or reorder candidates, change permission decisions, add an
LLM call, or enable enforcement.

The complete one-episode-per-domain diagnostic used OpenLux `gpt-4o-mini` as
the memory base model, `text-embedding-3-small`, and the official OpenLux
`gpt-4o` GateMem judge. It covered 101 checkpoints (Medical 27, Office 32,
Education 18, Household 24), with four episode workers and one request in
flight per worker. Official scoring completed for all 101 checkpoints, and
the prompt audit covered all 101 checkpoints. No certificate or target-binding
field appeared in the Stage 2 or answer prompts.

| Domain | Checkpoints | U | A | F | MGS | Action accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Medical | 27 | 70.00% | 66.67% | 0.00% | 23.33% | 59.26% |
| Office | 32 | 55.56% | 0.00% | 0.00% | 55.56% | 78.13% |
| Education | 18 | 0.00% | 16.67% | 0.00% | 0.00% | 88.89% |
| Household | 24 | 62.50% | 12.50% | 0.00% | 54.69% | 75.00% |
| **All scored cases** | **101** | **51.52%** | **24.24%** | **0.00%** | **39.03%** | **74.26%** |

These metrics are an engineering diagnostic, not a causal ablation or a
paper result. The sample contains only one episode per domain, Education has
six utility cases with zero utility accuracy, and the memory-model
temperature is 0.2. The run must not be mixed into the frozen 2,218-checkpoint
typed-rerank table. The machine-readable score is
`outputs/2026-08-14_Gov-Mem-v4-Symbolic-dev2-target-binding-shadow-v2_gpt4omini_embedding3small_smoke4/official_score.json`.

## Experiment Naming

Use the version name in every new output directory and result artifact:

```text
YYYY-MM-DD_Gov-Mem-v3.1-SP_<base-llm>_<embedding>_<scope>
YYYY-MM-DD_Gov-Mem-v4-Symbolic-dev0_<base-llm>_<embedding>_<scope>
YYYY-MM-DD_Gov-Mem-v4-Symbolic-dev1_<base-llm>_<embedding>_<scope>
YYYY-MM-DD_Gov-Mem-v4-Symbolic_<base-llm>_<embedding>_<scope>
```

The existing frozen v3 results retain their original names and remain
recoverable through Git history and their backup tags.
## 2026-08-14 - Gov-Mem-v4-Symbolic-dev2 current snapshot

- Added an auxiliary typed principal-entity relation graph from GateMem's structured `entities.relationships`.
- Added typed relation context and conservative evidence-to-entity `about` edges.
- Added explicit lifecycle-event annotations for deleted, revoked/retired,
  superseded/replaced, and explicitly updated evidence.
- Preserved Stage 1 retrieval order, Stage 2 candidate set, permission behavior,
  and LLM call count. Lifecycle annotations are auxiliary only: they do not
  filter candidates, reorder candidates, make permission decisions, or call an
  LLM.
- Regression tests: 288 passed in four batches.
- Controlled OpenLux validation: one complete episode per domain, 101
  checkpoints total, a 30-key pool with four concurrently leased keys and
  four episode workers. All 101
  retrieval traces contain `evidence_principal_typed_relation_lifecycle`.
- This remains an engineering validation snapshot; no U/A/F/MGS performance
  claim is attributed to it yet.

## 2026-08-15 - Gov-Mem-v4-Symbolic-dev3 state-ledger validation

The paper-facing method name remains `Gov-Mem-v4-Symbolic`. This increment adds
`state-ledger-v1` as a lightweight auxiliary layer over retrieved structured
records. For query-requested state slots, it keeps the newest source-bound
candidate by `turn_index`, retains competing values and conflict counts, and
distinguishes public dates from subject current dates. It reads no hidden
transcript, makes no permission decision, does not alter retrieval ordering or
candidate filtering, and adds no LLM call.

The OpenLux validation used `gpt-4o-mini` for the memory system,
`text-embedding-3-small`, the official `gpt-4o` GateMem judge, 30 isolated API
keys, and four episode workers with one request in flight per worker. It covered
the complete selected episodes (101 checkpoints): Medical 27, Office 32,
Education 18, and Household 24. Official scoring and prompt auditing completed
for all cases. The weighted result was U=66.67%, A=21.21%, F=0.00%, and
MGS=52.53%. These numbers are a development diagnostic and are not part of the
frozen 2,218-checkpoint typed-rerank table.

The machine-readable output is ignored by Git under `outputs/`; the reproducible
result note is
`experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger-smoke4_openlux_gpt4omini.md`.

## 2026-08-15 - Gov-Mem-v4-Symbolic-dev3 paired eight-episode holdout

The current state-ledger implementation and the frozen
`rag_naive_v3_typed_rerank` baseline were evaluated on the same eight complete
episodes and 203 checkpoints with OpenLux `gpt-4o-mini`,
`text-embedding-3-small`, and the official `gpt-4o` judge. Both runs used four
episode workers, one leased local key per worker, and no worker-internal
parallelism. All official judge cases completed with zero parse failures.

| Run | U | A | F | Aggregate MGS | Four-domain mean MGS |
|---|---:|---:|---:|---:|---:|
| Frozen typed-rerank baseline | 42.42% | 27.27% | 7.04% | 28.68% | 25.07% |
| Gov-Mem-v4-Symbolic-dev3 | 45.45% | 28.79% | 5.63% | 30.55% | 30.17% |

The Symbolic version improves aggregate MGS by 1.86 percentage points and the
four-domain mean by 5.10 points. The effect is heterogeneous: Medical is
negative while Office, Education, and Household are positive. Because the
memory-model temperature is 0.2, this remains a matched engineering
diagnostic rather than a deterministic causal ablation. The full record is
`experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger_holdout8_paired_gpt4omini.md`.
This result must not be merged into the frozen 2,218-checkpoint paper table.

## 2026-08-15 - Timestamp precision normalization

All message timestamps entering the adapters and RAG chunk builders now use
explicit second precision in ISO-like form, such as
`2026-05-01T09:00:00`. Minute-only source timestamps receive `:00`; existing
seconds, fractional seconds, timezone suffixes, and source-turn fields are
preserved. Missing timestamps remain missing.

## 2026-08-15 - Medical state-ledger audit and typed-plan integration

The saved 55-checkpoint Medical holdout was audited offline. The root cause of
the empty Medical ledger was that `build_symbolic_evidence()` did not receive
the existing `required_slot_plan`; all 55 checkpoints therefore had an empty
`requested_slots` list even when retrieval had structured clinical evidence.
The Symbolic ledger now reuses the typed evidence-frame compiler for planned
clinical fields and recognizes GateMem's `allergy on file is ...` phrasing.
No LLM calls, retrieval ordering, filtering, or enforcement were added. The
offline replay produced 27 non-empty ledgers and 15 ledgers with at least one
resolved field. This is a diagnostic only; no paper performance table was
updated.

## 2026-08-15 - Medical state-ledger official probe 2

After the typed-plan integration, two complete Medical episodes (55
checkpoints) were rerun with OpenLux `gpt-4o-mini`,
`text-embedding-3-small`, and the official OpenLux `gpt-4o` GateMem judge.
The run used two episode workers, one leased API key per worker, no
worker-internal parallelism, zero added Symbolic LLM calls, complete prompt
auditing, and zero judge parse failures.

| Run | U | A | F | MGS | Checkpoints |
|---|---:|---:|---:|---:|---:|
| Gov-Mem-v4-Symbolic-dev3 state-ledger probe 2 | 55.00% | 61.11% | 5.88% | 20.13% | 55 |
| Matching frozen `rag_naive_v3_typed_rerank` output | 55.00% | 55.56% | 5.88% | 23.01% | 55 |

The exact official prompt audit found non-empty requested-slot ledgers in
20/55 cases and at least one resolved slot in 13/55 cases; no Medical safety
query used a non-empty ledger. The Symbolic run also increased the answer
prompt by about 3,897 characters per checkpoint on average, while action
predictions were unchanged. This is a diagnostic regression signal, not a
paper claim or a causal ablation. The result note is
`experiments/result/2026-08-15_Gov-Mem-v4-Symbolic-dev3-state-ledger_medical_probe2_openlux_gpt4omini.md`.

## 2026-08-16 - Four additional Medical episodes with rotated OpenLux keys

Four previously unvalidated complete Medical episodes (108 checkpoints) were
evaluated with the same `Gov-Mem-v4-Symbolic-dev3-state-ledger` method,
OpenLux `gpt-4o-mini`, `text-embedding-3-small`, and the official OpenLux
`gpt-4o` judge. The run used two episode workers, one request in flight per
worker, a separate key ordering, complete prompt auditing, and zero judge
parse failures.

| Episode | U | A | F | MGS |
|---|---:|---:|---:|---:|
| Medical 005 | 70.00% | 55.56% | 11.11% | 27.65% |
| Medical 009 | 88.89% | 11.11% | 12.50% | 69.14% |
| Medical 012 | 70.00% | 22.22% | 0.00% | 54.44% |
| Medical 020 | 60.00% | 60.00% | 28.57% | 17.14% |
| **Aggregate, 108 checkpoints** | **71.79%** | **37.84%** | **12.50%** | **39.05%** |

The result confirms substantial episode heterogeneity. The earlier low result
was not limited to episodes 001 and 013: episode 020 is also weak, while 009
and 012 are strong. Combining all six validated Medical episodes gives
U=66.10%, A=45.45%, F=10.20%, and MGS=32.38% over 163 checkpoints. These are
development diagnostics only and must not be merged into the frozen
2,218-checkpoint paper table. The detailed record is
`experiments/result/2026-08-16_Gov-Mem-v4-Symbolic-dev3-state-ledger_medical_selected4_openlux_gpt4omini.md`.
