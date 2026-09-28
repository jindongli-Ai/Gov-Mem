# V8 Governed Slot Graph integration (2026-09-23)

The canonical V8 configuration now enables `memory_governed_slot_graph`.
Frozen experiment snapshots remain unchanged; previous V8 scores do not measure this implementation.

Flow:
1. Extract open-vocabulary slots, permission candidates and lifecycle candidates from observable history, excluding the current checkpoint question.
2. Incrementally build an episode-local directed MemoryGovernedSlotGraph. Preserve original source spans and append-only relations.
3. Match the question against entity/relation lists and recover original source chunks. Preserve the dense RAG ranking as the base, then give graph proposals at most four of the existing Stage-1 Top-K (20) positions; unused positions return to dense retrieval. When dense retrieval has fewer than Top-K hits, graph proposals may fill the remainder.
4. Existing adjacent context expansion and Stage-2 permission/lifecycle extraction operate on dense RAG evidence alone. The LLM validates source-grounded claims and binds events.
5. Existing deterministic symbolic critic checks exact bindings and active denials/deletions. The graph observer adds one narrow check: if a proposed release copies an exact deletion instruction from the same source turn, block that claim. All other graph findings are audit only. Only safe claims reach the answer model.

This is an audit graph plus the existing event-based symbolic enforcement layer, not a new general first-order theorem prover. Graph policy hints never authorize release or trigger a veto. Missing parsed roles, relationships or permissions mean **unknown**, never "allowed" or "denied". The exact deletion check can veto a claim but does not infer scope from a shared topic. Its net benefit remains to be measured. Existing adjacent/ingestion context means Top-K is not a cap on all Stage-2 visible text.

The V8 adapter deep-copies the last committed graph before extension. A failed extraction batch does not commit the graph or its processed-prefix watermark; the checkpoint keeps the dense RAG path and records the graph failure. Shorter or changed history discards the incompatible graph. Its cache is process-local; a fresh process rebuilds from the visible history rather than loading unchecked future state. The graph audit, explicit findings, and applied exact deletion vetoes are included in `answer_result.raw_response.governed_slot_graph`; findings do not enter Stage-2 evidence. Graph calls are included in logical call accounting, and provider telemetry remains authoritative for retries/failed requests.

## Paper-facing boundary and evaluation

The first paper studies **memory access governance** over a strong Naive RAG base. Governed Slot Graph and symbolic checks are assistive mechanisms; neither is assumed to parse every role, relation or permission. The graph candidate quota (default 4 of 20) is a fixed architectural choice, not a setting tuned per domain or checkpoint. Future evaluation must report graph extraction coverage, graph-only evidence contribution, veto precision/recall, malformed extraction and transport errors, latency, tokens, and U/A/F/MGS against the unchanged RAG base. Use the same whole episodes and visible history in ablations with graph retrieval or symbolic checks disabled. The four-episode diagnostics used during implementation are exposed development data and do not establish a performance gain.

No dataset-specific aliases or slot lists were added. The existing graph module's observed-token matching and generic lexical helpers are retained, not claimed to be a fully lexical-free ontology.

Validation: observer isolation tests cover unchanged Stage-2 evidence, explicit finding audit output, unchanged-prefix reuse, future-prefix rollback, and failed-extension atomicity. No paid experiment launched for this observer-only revision.
