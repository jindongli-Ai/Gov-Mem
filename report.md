# Gov-Mem Current Pipeline Report

This is a code-grounded handoff document for writing the Gov-Mem ICLR paper.
It describes the implementation that is active in the repository on 2026-08-19.
It should be read together with the source files listed at the end. Do not
silently combine this pipeline with historical `rag_policy_amem` runs or with
disabled experimental modules.

## 1. Current identity

The current development version is:

```text
Name:             Gov-Mem-v4-Symbolic-dev7
Experiment mode:  govmem_v4_symbolic
Active backbone:  RAGNaiveBackbone
Foundation:       rag_naive_v3_typed_rerank
Config:           configs/govmem_v4_symbolic_openlux_gpt4omini_embedding3small_dev7_authorization_boundary.yaml
Date:             2026-08-19
```

The paper-facing method name is `Gov-Mem-v4-Symbolic`; `dev7` is the current
development snapshot. The method is deliberately incremental: it keeps the
strong and benchmark-compatible RAG-Naive retrieval and typed Stage-2 path,
then adds lightweight structured and Symbolic evidence reasoning.

Important implementation distinction:

- `experiment.mode: govmem_v4_symbolic` dispatches to
  `src/gov_mem/backbones/rag_naive.py`, class `RAGNaiveBackbone`.
- The old `govmem_symbolic` and `rag_policy_amem` modes dispatch to
  `src/gov_mem/backbones/rag_policy_amem.py`. They are historical alternative
  paths and are not the current dev7 pipeline.
- Therefore current dev7 must not be described as the `rag_policy_amem`
  implementation.

The core research claim is not that a graph replaces retrieval. The claim is
that a high-performing neural retrieval pipeline can preserve typed source
information and use small, source-grounded Symbolic structures to constrain
valid evidence, lifecycle interpretation, temporal authorization, and answer
provenance.

## 2. GateMem data and visibility contract

GateMem is a checkpointed conversational-memory benchmark. Each domain has:

```text
dataset/GateMem/gatemem/data/<domain>/episodes.jsonl
dataset/GateMem/gatemem/data/<domain>/checkpoints.jsonl
```

The four domains are `medical`, `office`, `education`, and `household`.
`episodes.jsonl` contains episode turns, speaker information, timestamps,
entities, and relationships. `checkpoints.jsonl` specifies a query, an asker,
and an `as_of_turn_id`.

For each checkpoint, `CheckpointBenchmarkAdapter`:

1. loads the episode and checkpoint;
2. finds the exact `as_of_turn_id` in the episode;
3. exposes only the prefix through that turn;
4. preserves the asking principal and observable requester role;
5. keeps benchmark answer labels, expected actions, leak targets, and judge
   specifications outside the model-visible runtime view.

The adapter refuses unsafe fallbacks. It errors if `as_of_turn_id` is missing,
unknown, or duplicated rather than exposing the entire episode. This matters
because the hidden future suffix must never become retrieval evidence.

At evaluation time, the scorer separately uses the checkpoint's expected action,
query type, utility target, privacy target, and deletion target. Those fields
are for scoring only. They are not allowed to authorize an answer or to become
evidence for the answering model.

## 3. End-to-end dev7 flow

```text
GateMem checkpoint
        |
        v
Observable episode prefix
        |
        v
Turn-preserving structured records and RAG chunks
        |
        v
Stage 1 dense retrieval: top 20 visible turns
        |
        v
Symbolic evidence annotation
  - role consistency
  - typed relation graph
  - explicit lifecycle claims
  - retrieved-evidence-only state ledger
  - temporal authorization graph
        |
        v
Authorization-aware evidence boundary
        |
        v
Stage 2 typed scalar reranking and bounded LLM evidence reranking
        |
        v
Deletion, sensitive-field, summary, and current-state boundaries
        |
        v
Answer model sees only the selected evidence and structured fields
        |
        v
Answer plus optional claim contract
        |
        v
Deterministic claim-level provenance verification and audit record
        |
        v
Official GateMem judge and U/A/F/MGS evaluation
```

There is no full-transcript answer prompt in the current paper-facing path.
The answer model receives the selected evidence set, typed provenance, and
Symbolic summaries/certificates that were built from that evidence. The model
does not receive hidden checkpoint evaluation fields.

## 4. Stage 0: preserving the source record

`_build_turn_chunks` creates one chunk per visible GateMem turn. The retrieval
text follows the compatible RAG-Naive form:

```text
[role:principal_id] original turn text
```

For example:

```text
[nurse:p_042] The medication review is scheduled for Saturday at 09:30.
```

The role and principal prefix is part of the compatible retrieval text. Other
important fields are not flattened into a prose prefix. They are stored as a
typed `structured_record` in chunk metadata:

```json
{
  "record_type": "message",
  "message_id": "turn_17",
  "turn_id": "turn_17",
  "turn_index": 16,
  "timestamp": "2026-05-12T09:30:00",
  "speaker": {
    "principal_id": "p_042",
    "role": "nurse"
  },
  "turn_kind": "message",
  "authorization_assertions": [],
  "text": "The medication review is scheduled for Saturday at 09:30.",
  "checkpoint": {
    "as_of_turn_id": "turn_24"
  },
  "source_turn": { "...": "original observable GateMem turn" }
}
```

The actual record retains the complete normalized source turn. Timestamps are
normalized with both date and time precision where the dataset provides them;
the pipeline must not reduce a timestamp to date-only text. Principal ID, role,
turn ID, source message ID, turn kind, and original text are therefore
recoverable after retrieval without asking an LLM to reconstruct them from a
natural-language chunk.

The structured metadata is not appended to the Stage-1 embedding text. This
preserves the RAG-Naive retrieval comparison. It is exposed to Stage 2 as
structured JSON after retrieval, where it can be checked deterministically.

## 5. Stage 1: compatible neural retrieval

The current dev7 configuration uses:

```text
Provider: OpenLux
Embedding model: text-embedding-3-small
Embedding API base: https://api.openlux.ai/v1
Top-k: 20
```

The visible turn chunks are embedded, the question is embedded once, and the
20 highest-scoring dense candidates become the initial evidence set. The
embedding model is configured independently from the memory-system base LLM.
The current base LLM is OpenLux `gpt-4o-mini`; the official GateMem judge is a
separate OpenLux `gpt-4o` process.

The initial evidence set is closed. Later processing may reorder, annotate, or
filter these candidates, but it must not silently retrieve a hidden turn or
invent a new source ID.

## 6. Symbolic evidence layer

The Symbolic layer is implemented in
`src/gov_mem/backbones/symbolic_evidence.py`. It operates on the retrieved
top-20 evidence, not on the hidden transcript. Its purpose is to make facts
that are easy to lose in prose explicit and checkable.

### 6.1 Typed relation graph

The graph is a lightweight, query-local evidence graph. It is not a second
large knowledge base and it is not used to replace dense retrieval.

Typical node types are:

```text
Evidence       one retrieved turn/chunk
Principal      a speaker or requester identity
Role           a role associated with a principal
Entity         a GateMem entity
Resource       a referenced object or resource
LifecycleEvent an explicit deletion/revocation/update event
PolicyEvent    an explicit allow/deny/revoke policy assertion
```

Typical edge types are:

```text
Evidence  --spoken_by-->       Principal
Evidence  --about-->           Entity or Resource
Principal --has_role-->        Role
PolicyEvent --allows/denies--> Principal, Role, or Resource
PolicyEvent --revokes-->       an earlier permission or evidence item
PolicyEvent --applies_to-->    a target
LifecycleEvent --supersedes--> an earlier evidence item
```

The graph is built from observable episode data and retrieved text with
conservative rules. For example, it checks a speaker's observed role against
the episode roster and records a conflict rather than silently correcting it.
Relationship objects are directional; the implementation does not manufacture
reverse edges merely because two principals appear in the same relationship.

The graph is therefore auxiliary Symbolic information: dense retrieval finds
candidate turns, while graph relations help check identity, scope, lifecycle,
and policy consistency among those candidates.

### 6.2 Principal-role consistency

For every retrieved record, the system compares the typed speaker fields with
the episode roster. It records states such as:

```text
consistent
conflict
missing_speaker_field
principal_not_in_roster
missing_record
```

This is a general consistency check. It is not a GateMem-domain-specific
if/else answer rule and it does not use the hidden answer label.

### 6.3 Explicit authorization assertions

Authorization assertions are extracted conservatively from source text and
retain source spans. The grammar is provider-neutral and recognizes explicit
allow/deny/revoke-style statements. Extraction is not itself authorization:
the assertion must be connected to a requester, role, target, and time-valid
evidence before it can affect the evidence boundary.

This distinction prevents a retrieved sentence that merely mentions a policy
from automatically granting access.

### 6.4 Lifecycle validity and target binding

The system recognizes explicit lifecycle language such as:

```text
deleted, revoked, superseded, updated, replaced
```

It does not infer a deletion or update merely because a sentence contains
ordinary words such as `current` or `latest`. An explicit lifecycle claim is
bound to an earlier retrieved target only when the target match is unique and
temporally valid. Ambiguous and unbound cases remain explicit audit states.

This is a source-grounded validity mechanism, not a hard-coded list of
GateMem questions.

### 6.5 Retrieved-evidence-only state ledger

For requested fields, the Symbolic layer builds a state ledger from the
retrieved evidence only. A ledger field can contain:

```text
requested slot
candidate values
selected value
source memory/chunk ID
source turn ID
source quote
conflicting candidates
missing/resolved status
```

The ledger helps preserve field-level provenance and resolve explicit conflicts.
It cannot recover a value that was not retrieved, and it cannot use the hidden
future transcript. The configuration keeps
`stage2.long_context_field_ledger.enabled: false`; the optional full visible
transcript ledger is not part of the current paper-facing protocol.

## 7. Temporal authorization and evidence boundary

Dev7 enables:

```yaml
symbolic:
  temporal_authorization:
    enabled: true
    enforcement: true
```

The temporal graph tracks explicit policy events over the observable timeline.
Its concepts include principal, role, resource, policy event, effective time,
allow, deny, revoke, and supersede. It produces a source-bound authorization
certificate describing the decision and the supporting event IDs.

When enforcement identifies evidence that is not valid for the current
requester, target, or temporal policy state, the authorization-aware evidence
boundary can remove that candidate before Stage 2. The boundary can narrow the
candidate set; it cannot promote blocked evidence or create an answer.

This is the main place where the graph affects the active dev7 pipeline. The
graph is not merely logged: its temporal authorization result can constrain
what evidence reaches later answer selection. The remaining graph annotations
and certificates also remain available for traceability.

## 8. Stage 2: typed reranking and safety boundaries

The Stage-2 implementation is in
`src/gov_mem/backbones/stage2_typed_rerank.py` and is retained from the strong
RAG-Naive v3 foundation.

For typed scalar queries, the deterministic score is:

```text
0.70 * dense retrieval score
+ 0.12 * query-anchor overlap
+ 0.10 * requested-family match
+ 0.06 * currentness signal
+ 0.02 * recency signal
```

This is a ranking signal, not a permission grant. The ranker keeps evidence
coverage and applies typed field matching so that a date, amount, identifier,
location, or other requested field is not casually substituted by a neighboring
field.

For mixed or difficult queries, the bounded LLM reranker receives at most 20
candidates and at most 2400 characters per candidate. It returns a validated
ordering/selection. The selected evidence is moved to the front while the
remaining candidates are retained when the contract permits. If the response
is missing or invalid, deterministic fallback projection is used.

Active delivery boundaries include:

- explicit deletion or forgetting requests;
- explicit sensitive-field requests;
- summary-only handling for broad requests where exact disclosure is unsafe;
- current-versus-stale evidence handling;
- multi-field and multi-entity separation;
- no-memory/refusal actions when the requested field is unavailable or blocked.

The current dev7 config has:

```yaml
stage2:
  llm_reasoning_rerank:
    enabled: true
    max_candidates: 20
    max_candidate_chars: 2400
  long_context_field_ledger:
    enabled: false
```

## 9. Answer realization

The answering model is the configured base answering model (`gpt-4o-mini` in
the current config). It sees the question plus the selected, structured
evidence. It does not see the full hidden episode and does not receive hidden
evaluation labels.

The answer schema supports:

```text
answer          direct source-grounded answer
answer_redacted safe subset or broad summary
refuse          policy blocks the requested disclosure
no_memory       requested value is unavailable, deleted, or unresolved
```

The answer includes used memory/chunk IDs and runtime metadata for evaluation.
For optional provenance auditing, the same answer JSON may contain a
`claim_contract` with requested fields, selected values, source IDs, and source
spans. The contract is an audit record; it is not additional answer content.

## 10. Claim-level provenance explanation

In this report, “explanation” means a post-answer, non-intervention record: the
system explains what evidence and Symbolic conditions support the delivered
answer. It does not change the answer and it is not used by the GateMem scorer.

Dev7 enables the verifier but intentionally sets:

```yaml
policy_verifier:
  enabled: true
  claim_provenance_enabled: true
  explanation_enabled: true
  claim_provenance_enforcement: false
  llm_enabled: true
  llm_advisory_only: true
```

The explanation module is called on the actual `govmem_v4_symbolic` RAG-Naive
answer path. It records, where a contract is available:

1. the answer's selected values;
2. whether cited source IDs belong to the Stage-2 selected evidence;
3. whether each source span occurs in the cited source text;
4. whether the source span supports the claimed field/value;
5. whether unsupported extra claims appear in the answer;
6. whether restricted or unknown fields are represented consistently.

GateMem source-message IDs are mapped to the internal chunk IDs only within the
current checkpoint. No synthetic source is created. If the base model omits a
claim contract, the explanation records `claim-level explanation incomplete`
rather than pretending that claim-level support was verified.

The explanation is non-intervention for the RAG-Naive dev7 path. A failed claim
check does not replace the original answer or turn it into a refusal. The
explanation records `answer_unchanged: true` and `scored_by_gatemem: false`.
This is deliberate: the temporal authorization boundary remains
intervention-capable earlier in the pipeline, while this module provides an
auditable explanation channel. The separate historical stateful executor has
its own fail-closed field-projection contract and must not be conflated with
this RAG-Naive path.

The explanation adds no extra LLM call in the active dev7 RAG-Naive path. It is
assembled from the selected evidence, Stage-2 decision, Symbolic trace, and
the deterministic claim audit. The optional model-produced claim contract is
only supplementary.

## 11. Why this is neuro-symbolic

The neural components handle semantic representation and recall:

```text
embedding retrieval
LLM-based bounded evidence reranking
LLM-based answer realization
```

The Symbolic components handle explicit, auditable structure:

```text
typed source metadata
principal-role consistency
entity/resource relations
explicit lifecycle transitions
temporal authorization transitions
state slots and conflicts
source IDs and source spans
deterministic claim verification
```

The division of labor is important. The graph and ledgers do not pretend to
understand every sentence perfectly; they constrain and verify claims using
information already recovered from the neural retrieval stage. The neural
model supplies semantic flexibility, while the Symbolic layer supplies
identity, time, lifecycle, authorization, and provenance checks that should
not be left to unconstrained prose generation.

## 12. Runtime and evaluation details

The current config uses:

```text
Memory base LLM:       OpenLux gpt-4o-mini
Embedding:             OpenLux text-embedding-3-small
Official judge:        OpenLux gpt-4o
Memory temperature:    0.2
Stage-1 top-k:         20
Official protocol:     gatemem_paper_main
Gold feedback:         disabled
Embedding fallback:    disabled in the dev7 config
```

The official judge is separate from the memory-system model. The metrics are
defined by the GateMem evaluation protocol:

```text
U  = utility accuracy
A  = privacy leakage rate
F  = deletion leakage rate
OR = over-refusal rate (reported separately)
MGS = U * (1 - A) * (1 - F)
```

Scores must be reported with the exact base LLM, embedding model, config,
dataset scope, checkpoint count, and official judge. A partial episode smoke
run is an engineering diagnostic, not a paper full-benchmark result. The
2026-08-19 claim-provenance smoke completed 28 Medical checkpoints with zero
judge parse failures and recorded the verifier in every prediction, but its
U/A/F/MGS values must not be presented as the final paper table.

For operational safety, experiments should use bounded episode workers and at
most one in-flight request per worker unless a separate capacity test is
approved. API keys should be leased, not used to create nested thread pools.
Long or frequent process scans can overload shared NFS; monitoring intervals
should be coarse enough to avoid turning status checks into filesystem scans.

## 13. Dev7 full-benchmark result (2026-08-20)

The framework was evaluated on all 2,218 GateMem checkpoints after the
dev7 claim-level explanation channel was integrated into the active
`govmem_v4_symbolic` RAG-Naive path. The run used OpenLux `gpt-4o-mini` as the
memory-system base LLM, OpenLux `text-embedding-3-small` for retrieval, and
OpenLux `gpt-4o` as the official GateMem judge. The official protocol used
`gate_by_action=false`; gold feedback, experience updates, skill updates, and
the long-context ledger were disabled.

| Domain | Checkpoints | U | A | F | MGS | Action accuracy | OR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 579 | 74.29% | 46.35% | 10.17% | **35.80%** | 70.47% | 10.48% |
| Office | 547 | 39.61% | 4.09% | 1.80% | **37.30%** | 75.69% | 39.61% |
| Education | 540 | 40.00% | 18.89% | 7.22% | **30.10%** | 71.85% | 17.78% |
| Household | 552 | 43.48% | 27.17% | 2.72% | **30.80%** | 70.11% | 19.02% |
| **Four-domain average / pooled U-A-F** | **2,218** | **49.72%** | **24.47%** | **5.53%** | **33.50%** | | |

The overall MGS is the arithmetic mean of the four domain MGS values. The
overall U/A/F values are checkpoint-count-weighted pooled values and are shown
separately to avoid mixing aggregation rules. `OR` is over-refusal and is not
multiplied into MGS. All four official judge jobs completed with zero parse
failures, and all 2,218 checkpoints have prompt-context audit records.

The explanation record is present in 2,218/2,218 official predictions. It is a
non-intervention artifact with `answer_unchanged=true` and
`scored_by_gatemem=false`; it records evidence references, Stage 2 routing,
Symbolic consistency/lifecycle/authorization facts, state-ledger summaries, and
claim-level provenance when a source-bound claim contract exists. It does not
change the answer, action, prompt context, or GateMem metric.

The complete dated record is
[`2026-08-20 dev7 full benchmark`](experiments/result/2026-08-20_Gov-Mem-v4-Symbolic-dev7_full_all_2218_openlux_gpt4omini.md).
This is a complete dev7 measurement, not a causal ablation against the frozen
v3 typed-rerank table. A causal claim for an individual Symbolic component
still requires a paired ablation under the same checkpoint and model protocol.

## 14. What a paper may claim

Defensible claims supported by the current implementation include:

- Gov-Mem preserves GateMem turn identity, role, timestamp, and source fields
  as typed provenance instead of requiring post-retrieval prose extraction.
- A query-local evidence graph represents principal, role, entity/resource,
  policy, lifecycle, and evidence relations.
- The framework performs explicit role consistency, lifecycle validity, and
  temporal authorization checks over retrieved evidence.
- The state ledger is closed over retrieved evidence and records field-level
  conflicts and provenance.
- Stage 2 and final claim verification restrict source IDs and source spans to
  the selected evidence set.
- The design combines neural retrieval/generation with deterministic
  structure-based checks while preserving the strong RAG-Naive backbone.

The following claims require additional controlled experiments and should not
be asserted from the current smoke artifacts alone:

- that every Symbolic component improves MGS;
- that the claim verifier improves answer accuracy while enforcement is false;
- that the graph alone causes a measured gain without an ablation;
- that dev7's full-benchmark result proves a causal gain for every Symbolic
  component without a paired ablation;
- that partial episode diagnostics are representative of all four domains.

Do not describe dataset-specific string rules as the main innovation. The
method should be framed as general typed provenance, evidence-closed state
resolution, temporal policy consistency, and auditable claim delivery.

## 15. Relevant files

```text
run_govmem.py
src/gov_mem/pipeline.py
src/gov_mem/data/adapters.py
src/gov_mem/backbones/rag_naive.py
src/gov_mem/backbones/symbolic_evidence.py
src/gov_mem/backbones/stage2_typed_rerank.py
src/gov_mem/policy_verifier.py
src/gov_mem/data/timestamps.py
configs/govmem_v4_symbolic_openlux_gpt4omini_embedding3small_dev7_authorization_boundary.yaml
README.md
VERSION_LOG.md
```

The version history in `VERSION_LOG.md` records how dev0 through dev7 were
promoted. The README contains the repository snapshot, benchmark tables, and
the distinction between diagnostic results and paper-compatible results.
