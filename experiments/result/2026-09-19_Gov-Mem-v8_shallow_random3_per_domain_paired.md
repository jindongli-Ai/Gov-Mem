# V8 shallow memory: complete-episode paired development evaluation

The frozen shallow implementation obtains **22.19% official mean-domain MGS**,
versus **19.75% for RAG-Naive** (+2.44 percentage points). This is **not a reliable
win**: V8 has 47 execution errors versus zero baseline errors. Counting execution
errors and applicable null judgments pessimistically gives V8 **8.28%**. Only
Medical exceeds the baseline. The prior frozen full-graph V8 obtained 4.37%.

## Scope and identity

- Manifest: `experiments/manifests/v8_paired_random3_per_domain_seed20260918.json`.
- Four domains, three randomly selected complete episodes each, 303 checkpoints
  per system; all 12 episodes completed, including failed checkpoints.
- Measured runtime SHA256: `14ed6a1de3f3d306bd7e7f1b145081b847c91cc205266f74b5af89b56c8a3242`.
- Inference and snapshot: `outputs/v8_shallow_random3_seed20260918_20260919_json`.
- Scores, cost telemetry and runtime audit:
  `outputs/v8_shallow_random3_seed20260918_20260919_json_scored`.
- Memory/answer model: Gemini 2.5 Flash-Lite, temperature 0. Official judge:
  GPT-4o, OpenLux, temperature 0, `gate_by_action=false`.
- Baseline predictions reused from `outputs/v8_paired_random3_seed20260918_attempt2/rag_naive`;
  official baseline judgments reused from `outputs/v8_paired_random3_seed20260918_scored/rag_naive`.
  Exact predictions, official source hashes, complete unique judgment coverage,
  parsing and judge settings were verified before reuse. Provenance is recorded
  in `reused_baseline_scores.json`; old paid telemetry was not copied.
- One Medical circuit breaker fired at its final checkpoint. Strict resume
  finalized metadata from all 27 existing predictions without rerunning them.

## Official results

All values below are percentages. U is utility accuracy; A and F are privacy
and deletion leakage rates (lower is better). MGS = U × (1−A) × (1−F), computed
per domain and then averaged equally across domains.

| Domain | Checkpoints | V8 U | V8 A | V8 F | V8 MGS | RAG-Naive MGS | V8 errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medical | 81 | 89.66 | 29.63 | 24.00 | 47.95 | 22.56 | 16 |
| Office | 96 | 66.67 | 65.52 | 71.79 | 6.48 | 9.75 | 11 |
| Education | 54 | 50.00 | 44.44 | 88.89 | 3.09 | 4.63 | 6 |
| Household | 72 | 66.67 | 37.50 | 25.00 | 31.25 | 42.06 | 14 |
| Mean / total | 303 | | | | **22.19** | **19.75** | **47** |

There are 303 prediction records and 303 official judgments per system. Judge
parse failures are zero. One Office privacy judgment has a null applicable
label: official aggregation excludes it; conservative aggregation counts it as
leakage. No failed checkpoint was removed. Pessimistic V8 domain MGS values are
17.93%, 2.79%, 0.93%, and 11.46% respectively (mean 8.28%).

This is a repeatedly used development sample, not an untouched holdout or the
full GateMem benchmark. Public access-policy input is aligned with the official
baseline; temporal-scaffold attribute asymmetry remains. These results do not
isolate the causal contribution of symbolic reasoning.

## Actual provider usage

| Phase | Model | Recorded requests | Prompt tokens | Completion tokens | Provider total |
|---|---|---:|---:|---:|---:|
| Completed JSON inference | Gemini 2.5 Flash-Lite | 619 | 3,078,797 | 85,825 | 3,164,624 |
| Fresh official judgments | GPT-4o | 303 | 230,449 | 15,810 | 246,259 |
| Aborted TAB diagnostic (separate) | Gemini 2.5 Flash-Lite | 78 | 635,716 | 24,974 | 660,690 |

All recorded requests returned HTTP 200; HTTP success does not establish valid
model output. Preserve provider total usage even though the completed inference
components differ from its reported total by two tokens. Inference tokens are
about **4.95×** the reused baseline's 639,831 tokens (303 calls), and about 9.6%
below the old V8's 3,500,275 tokens. Cost reduction is not yet sufficient.

Embeddings were warm cached: 2,274 exact-text cache entries were copied from
attempt8, with provenance saved; no graph state was reused. Baseline judging was
not paid again. No monetary estimate is made without provider prices. The
aborted diagnostic attempted 45 predictions, all errors, and completed zero
episodes; it was not scored. Interrupted in-flight requests may incur additional
unrecorded charges. The failed integration cost is not hidden in the new run.

## Symbolic reasoning in actual execution

`runtime_audit.json` records 413 Stage-2 responses including repairs, 449
validated claims, and 39 event-bound claims across 36 normal checkpoints.
The symbolic critic produced 19 permission vetoes: 16 confirmed existing
language blocks and **3 independently changed captured LLM release proposals to
blocks**. There were also three language deletion confirmations. These establish
an active symbolic decision path, not that three true leaks were prevented.
All 619 captured inference requests selected Gemini 2.5 Flash-Lite.

## Subsequent offline fixes: unmeasured implementation

Current source differs from the frozen runtime above. It preserves literal JSON
controls, accepts a literal quote without requiring a duplicate value, preserves
candidate-only explicit denials, separates visible source-turn references from
permission-event bindings, and permits grounded multi-policy denials without
inventing a canonical binding. Unknown sources, ungrounded restrictions,
structurally malformed output and incompatible released bindings still fail.
The prompt's conflicting `bind=NONE` instruction was removed from JSON mode.

Using the original supplied prompts and captured responses, contract validation
changed from **257 pass / 156 fail** to **330 pass / 83 fail** out of the same
413 responses: **73 recovered, zero newly failing**. Artifacts:
`2026-09-19_v8_shallow_contract_before.json` and
`2026-09-19_v8_shallow_contract_after.json` in this directory.
This replay includes repair requests and does not evolve the memory store,
generate answers or judge them. It is not 73 recovered checkpoints and does not
establish improved MGS. No paid rerun followed these offline changes.

Targeted contract, shallow-flow, binding and score-reuse tests: 32 passed.
Repository test suite (`PYTHONPATH=src pytest -q tests`): **438 passed, 13 failed**;
the failure set remains the previously documented legacy field-state/stateful
policy failures. Unscoped pytest also discovers vendored SciPy tests and stops
on a third-party collection import error; use the repository `tests` directory.
`git diff --check` passes.
Further work must address the remaining source/semantic failures and high
Office/Education deletion leakage before claiming stable performance or
launching another broad paid comparison.
