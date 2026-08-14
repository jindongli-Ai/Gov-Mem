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
| `Gov-Mem-v4-Symbolic` | The frozen v4 line after dev0 regression and benchmark checks | RAG-Naive foundation plus deterministic/neuro-symbolic role, permission, temporal, and consistency reasoning | Paper method name |

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
