"""Gov-Mem v8: RAG-first retrieval with late claim-level governance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from gov_mem.backbones.v4_support import (
    BackboneRunResult,
    RAGChunk,
    build_reasoning_state,
    chunk_to_memory_item,
    save_rag_chunks,
)
from gov_mem.data.schema import (
    AnswerResult,
    GovernedActionDecision,
    MemoryInstance,
    QueryPlan,
    RetrievedEvidence,
)
from gov_mem.data.timestamps import normalize_message_timestamp, normalize_timestamp
from gov_mem.extraction.v8_event_extractor import extract_v8_events, _turn_payload
from gov_mem.extraction.v8_event_validator import validate_v8_extraction
from gov_mem.extraction.v8_scene_schema import v8_scene_schema
from gov_mem.extraction.v8_principal_registry import build_v8_principal_registry
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.governance_runtime.v8_claim_reasoner import reason_v8_claims
from gov_mem.governance_runtime.v8_event_store import V8EventStore
from gov_mem.governance_runtime.v8_safe_evidence import build_v8_safe_evidence
from gov_mem.governance_runtime.v8_state_projector import project_v8_state
from gov_mem.governance_runtime.v8_symbolic_critic import criticize_v8_claims
from gov_mem.llm.client import LLMClient
from gov_mem.governance_runtime.v8_prompt_input import application_policy
from gov_mem.governance_runtime.v8_shallow_memory import (
    shallow_graph_context, shallow_resource_registry, ensure_shallow_events,
)
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.memory.v8_slot_graph import V8SlotGraphIndex


_RECOVERABLE_GOVERNANCE_MARKERS = (
    "grounding failed", "not_grounded", "requires explicit bind",
    "requires value and delivery", "unknown candidate", "unknown event",
    "without ingestion", "invalid v8 answer action", "invalid v8 stage-2",
    "jsondecodeerror", "extra data", "expecting ',' delimiter",
    "without a grounded claim", "incompatible resources/actions/scenes",
    "shallow ingestion requires an events array", "missing bind",
    "block requires a grounded, scope-specific restriction",
    "invalid v8 answering response",
)


def _recoverable_governance_gap(exc: BaseException) -> bool:
    """Return true for incomplete model contracts, never provider failures."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return False
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(marker in text for marker in _RECOVERABLE_GOVERNANCE_MARKERS)


V8_ANSWER_SYSTEM_PROMPT = """You are the final answering agent for Gov-Mem v8.
Return JSON only. Answer the question using only ALLOWED_CLAIMS. These claims
have already passed late governance. Never infer, reconstruct, confirm, or
mention a blocked value. BLOCKED_SLOTS contains names and reasons only; it has
no values. Preserve every useful requested field supported by allowed claims.
If some requested fields are blocked, answer the allowed portion concisely and
state only that the remaining field cannot be provided.
Preserve exact dates, numbers, identifiers, status phrases, and negations from
ALLOWED_CLAIMS; do not paraphrase an explicit "no X" status into weaker text.
For a negative status, include the compact phrase "no <status>" used by the
question when it is supported, followed by any more specific allowed wording.

Return {"action":"answer|answer_redacted|refuse|no_memory",
"answer":"...","used_claim_ids":["..."]}.
"""


def _critical_extraction_rejections(batch) -> list[dict[str, Any]]:
    """Invalid advisory facts/graph hints are omitted, never treated as policy.

    A missing or malformed governance/lifecycle record is different: it could
    hide an applicable restriction at a later checkpoint and remains fatal.
    """
    return [row for row in batch.rejected
            if row.get("reason") not in {"context_only_event", "context_only_record", "context_only_scene"}
            and (row.get("kind") == "governance_event"
                 or (row.get("kind") == "fact" and row.get("lifecycle", "assert") != "assert"))]


def _slot_graph_observer_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Summarize explicit graph warnings without changing RAG evidence."""
    graph = audit.get("graph") or {}
    policies = graph.get("policies") or []
    lifecycle = graph.get("lifecycle") or []
    findings = []
    for row in policies:
        effect = str(row.get("effect") or "").casefold()
        if effect in {"deny", "revoke", "require_permission"}:
            findings.append({"kind": "explicit_policy", "effect": effect,
                             "source_chunk_id": row.get("source_chunk_id"),
                             "source_message_id": row.get("source_message_id"),
                             "span": row.get("span")})
    for row in lifecycle:
        if str(row.get("lifecycle_type") or "").casefold() == "delete":
            findings.append({"kind": "explicit_deletion",
                             "source_chunk_id": row.get("source_chunk_id"),
                             "source_message_id": row.get("source_message_id"),
                             "span": row.get("span")})
    return {"mode": "observer_only", "findings": findings,
            "finding_count": len(findings),
            "effects_answer_evidence": False,
            "effects_authorization": False,
            "possible_exact_deletion_veto": True}


def _build_turn_chunks(instance: MemoryInstance) -> list[RAGChunk]:
    chunks = []
    for index, raw_message in enumerate(instance.messages):
        message = normalize_message_timestamp(raw_message)
        turn_id = str(message.get("turn_id") or message.get("message_id") or f"turn_{index}")
        speaker_id = str(message.get("speaker_id") or "unknown")
        speaker_role = str(message.get("speaker_role") or "")
        text = str(message.get("text") or "").strip()
        prefix = f"[{speaker_role}:{speaker_id}]" if speaker_role else f"[{speaker_id}]"
        chunks.append(RAGChunk(
            chunk_id=f"chunk_{index + 1:04d}_{turn_id}_{turn_id}",
            instance_id=instance.instance_id,
            text=f"{prefix} {text}".strip(),
            source_message_ids=[turn_id],
            speaker_ids=[speaker_id],
            timestamp_range=(normalize_timestamp(message.get("timestamp")), normalize_timestamp(message.get("timestamp"))),
            metadata={
                "chunk_type": "turn",
                "structured_record": {
                    "turn_id": turn_id,
                    "message_id": turn_id,
                    "turn_index": index,
                    "timestamp": normalize_timestamp(message.get("timestamp")),
                    "speaker": {"principal_id": speaker_id, "role": speaker_role},
                    "turn_kind": message.get("turn_kind"),
                    "text": text,
                    "checkpoint": {"as_of_turn_id": str((instance.metadata.get("observable") or {}).get("as_of_turn_id") or "")},
                },
            },
        ))
    return chunks


def _query_plan(instance: MemoryInstance) -> QueryPlan:
    return QueryPlan(
        query_type="utility",
        target_users=[instance.asking_user_id] if instance.asking_user_id else [],
        target_entities=[],
        required_memory_types=["chunk"],
        symbolic_filters={},
        dense_queries=[instance.question],
        reasoning_ops=["v8_late_claim_governance"],
        semantic_spec={"source": "v8_claim_reasoner"},
        planning_trace={"llm_calls": 0, "mode": "rag_first"},
    )


def _tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _graph_side_context(
    *, instance: MemoryInstance, store: V8EventStore,
    state_projection: dict[str, Any], static_relationships: list[dict[str, Any]],
    visible_turn_ids: list[str], memory_mode: str = "full",
) -> dict[str, Any]:
    if memory_mode == "shallow":
        return shallow_graph_context(state_projection, requester=instance.asking_user_id,
                                     identity_relations=static_relationships)
    query_tokens = _tokens(instance.question)

    def score(row: dict[str, Any]) -> tuple[int, int]:
        surface = " ".join(str(value) for value in row.values() if isinstance(value, (str, int, float)))
        spans = " ".join(
            str(item.get("span") or "") for item in row.get("source_spans") or [] if isinstance(item, dict)
        )
        return (len(query_tokens & _tokens(f"{surface} {spans}")), len(spans))

    facts = sorted(state_projection.get("current_facts") or [], key=score, reverse=True)
    relations = list(state_projection.get("relations") or [])
    return {
        "ontology": v8_scene_schema(),
        "as_of_turn_id": state_projection.get("as_of_turn_id"),
        "current_fact_hints": facts[:20],
        "requester_permissions": state_projection.get("requester_permissions") or [],
        "requester_permission_events": state_projection.get("requester_permission_events") or [],
        "lifecycle_tombstones": state_projection.get("lifecycle_tombstones") or [],
        "lifecycle_events": state_projection.get("lifecycle_events") or [],
        "identity_relations": static_relationships,
        "relations": relations,
        "duty_edges": state_projection.get("duty_edges") or [],
        "scene_hints": state_projection.get("current_scenes") or [],
        "entities": state_projection.get("current_entities") or [],
        "authority": "source_grounded_state_not_automatic_authorization",
    }


def _safe_claim_payload(safe_evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": row.metadata.get("claim_id"),
            "slot": row.metadata.get("slot"),
            "value": row.metadata.get("approved_value", row.content.split(": ", 1)[1] if ": " in row.content else row.content),
            "source_message_ids": row.source_message_ids,
            "source_time": row.metadata.get("source_time"),
            "temporal_state": row.metadata.get("temporal_state"),
            "delivery": row.metadata.get("delivery"),
            "symbolic_advisories": row.metadata.get("symbolic_advisories") or [],
        }
        for row in safe_evidence
    ]


def _requests_deleted_history(question: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", str(question or "").casefold()))
    return bool(re.search(
        r"\b(?:recover|restore|revive|old|older|earlier|previous|retired|deleted|before)\b",
        normalized,
    ))


def _expand_with_adjacent_context(
    raw_evidence: list[RetrievedEvidence],
    *,
    chunks: list[RAGChunk],
    max_extra: int,
) -> list[RetrievedEvidence]:
    """Complete local discourse around Top-K hits without another retrieval call."""

    if max_extra <= 0:
        return list(raw_evidence)
    index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    seen = {row.memory_id for row in raw_evidence}
    extra: list[RetrievedEvidence] = []
    # Immediate neighbors are often the other half of a question/answer,
    # exception, cancellation, or permission exchange. Preserve dense order
    # and append context so the Top-K result itself is unchanged.
    for row in raw_evidence:
        index = index_by_id.get(row.memory_id)
        if index is None:
            continue
        for neighbor_index in (index - 1, index + 1):
            if neighbor_index < 0 or neighbor_index >= len(chunks):
                continue
            chunk = chunks[neighbor_index]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            extra.append(RetrievedEvidence(
                memory_id=chunk.chunk_id,
                content=chunk.text,
                score=0.0,
                retrieval_source="dense_hit_adjacent_context",
                reason=f"Immediate discourse neighbor of {row.memory_id}",
                user_id=chunk.speaker_ids[0] if chunk.speaker_ids else None,
                memory_type="chunk",
                source_message_ids=list(chunk.source_message_ids),
                time=chunk.timestamp_range[0],
                metadata=dict(chunk.metadata or {}),
            ))
            if len(extra) >= max_extra:
                return [*raw_evidence, *extra]
    return [*raw_evidence, *extra]


def _answer_from_safe_evidence(
    *, instance: MemoryInstance, safe_evidence: list[RetrievedEvidence],
    safe_audit: dict[str, Any], llm_client: LLMClient, model_name: str,
    stage2_prompt_audit: dict[str, Any],
) -> AnswerResult:
    allowed = _safe_claim_payload(safe_evidence)
    blocked = list(safe_audit.get("blocked_claims") or [])
    if not allowed:
        deletion_blocked = any(
            any(
                reason in {"explicit_deleted_claim", "source_grounded_deletion_tombstone", "bound_lifecycle_deletion_veto"}
                for reason in item.get("symbolic_veto_reasons") or []
            )
            for item in blocked
        )
        if deletion_blocked and _requests_deleted_history(instance.question):
            text = "The requested information has been deleted and is not available."
            action = "no_memory"
        elif safe_audit.get("claim_ledger_action") == "refuse":
            text = "I cannot provide that information under the current access policy."
            action = "refuse"
        else:
            if deletion_blocked:
                text = "The requested information has been deleted and is not available."
                action = "no_memory"
            elif blocked:
                text = "I cannot provide that information under the current access policy."
                action = "refuse"
            else:
                text = "I do not have memory of that."
                action = "no_memory"
        return AnswerResult(
            prediction=text, answer_text=text, used_memory_ids=[], action=action,
            reasoning_summary="Gov-Mem v8 safe evidence view contains no releasable claim.",
            raw_response={
                "prompt_audit": {
                    "schema_version": 1, "audit_status": "no_runtime_answer_prompt",
                    "answer_prompt": None, "stage2_rerank_prompt": stage2_prompt_audit,
                }
            },
        )
    payload = {
        "question": instance.question,
        "requester": instance.metadata.get("requester") or {"principal_id": instance.asking_user_id},
        "allowed_claims": allowed,
        "blocked_slots": blocked,
    }
    assert_runtime_payload_safe(payload, context="v8_answering")
    user_prompt = json.dumps(payload, ensure_ascii=False)
    context_text = "\n".join(f"{item['slot']}: {item['value']}" for item in allowed)
    raw = llm_client.chat_json(
        model=model_name, system_prompt=V8_ANSWER_SYSTEM_PROMPT, user_prompt=user_prompt,
    )
    if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str) or not raw["answer"].strip():
        raise ValueError("Invalid V8 answering response; no prediction emitted")
    answer = raw["answer"].strip()
    # Derive action from requested-slot coverage, never from permission scope
    # labels or a model's choice to call a fully answered query redacted.
    action = "answer_redacted" if safe_audit.get("blocked_requested_count", len(blocked)) else "answer"
    allowed_claim_ids = {str(item["claim_id"]) for item in allowed}
    used_claim_ids = [
        str(value) for value in raw.get("used_claim_ids") or [] if str(value) in allowed_claim_ids
    ] or sorted(allowed_claim_ids)
    source_by_claim = {
        str(row.metadata.get("claim_id")): str(row.metadata.get("source_memory_id") or "")
        for row in safe_evidence
    }
    used_memory_ids = list(dict.fromkeys(
        source_by_claim[claim_id] for claim_id in used_claim_ids if source_by_claim.get(claim_id)
    ))
    return AnswerResult(
        prediction=answer,
        answer_text=answer,
        used_memory_ids=used_memory_ids,
        action=action,
        reasoning_summary="Gov-Mem v8 answer generated from the late-governed safe claim ledger.",
        raw_response={
            "v8_used_claim_ids": used_claim_ids,
            "prompt_audit": {
                "schema_version": 1,
                "audit_status": "runtime_answer_prompt",
                "answer_prompt": {
                    "system_prompt": V8_ANSWER_SYSTEM_PROMPT,
                    "user_prompt": user_prompt,
                    "context_text": context_text,
                },
                "stage2_rerank_prompt": stage2_prompt_audit,
            },
        },
    )


class GovMemV8LateGovernanceBackbone:
    def __init__(
        self, *, llm_client: LLMClient, embedding_client: LLMClient,
        config: dict[str, Any], output_dir: Path, dataset_name: str,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.memory_mode = (config.get("v8_memory") or {}).get("mode", "full")
        if self.memory_mode not in {"full", "shallow"}:
            raise ValueError("Unknown V8 memory mode")
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self._stores: dict[str, V8EventStore] = {}
        self._slot_graph = V8SlotGraphIndex()

    def _store(self, conversation_id: str) -> tuple[V8EventStore, Path]:
        path = self.output_dir / "v8_event_cache" / self.dataset_name / f"{conversation_id}.json"
        if conversation_id not in self._stores:
            self._stores[conversation_id] = V8EventStore.load(path, conversation_id=conversation_id, memory_mode=self.memory_mode)
        return self._stores[conversation_id], path

    def _prepare_ingestion(self, instance: MemoryInstance, registry):
        store, path = self._store(str(instance.conversation_id or instance.instance_id))
        visible = list(instance.messages)
        turn_ids = [str(m.get("turn_id") or m.get("message_id") or "") for m in visible]
        if not all(turn_ids) or len(set(turn_ids)) != len(turn_ids):
            raise ValueError("V8 requires unique visible turn IDs")
        pending = [m for m, t in zip(visible, turn_ids) if t not in set(store.processed_turn_ids)]
        new_count = len(pending)
        cfg = dict(self.config.get("v8_extraction") or {})
        inline_turns = max(1, int(cfg.get("inline_max_new_turns", 64)))
        inline_chars = max(1, int(cfg.get("inline_max_chars", 24000)))
        batch_size = max(1, int(cfg.get("max_new_turns_per_call", 32)))
        context_count = max(0, int(cfg.get("context_turns", 8)))
        audits = []
        prefill_gap = None

        def projection():
            return project_v8_state(store, visible_turn_ids=turn_ids,
                                    requester_principal_id=instance.asking_user_id, question=instance.question,
                                    checkpoint_time=visible[-1].get("timestamp") if visible else None)

        def context_for(messages):
            first = turn_ids.index(str(messages[0].get("turn_id") or messages[0].get("message_id")))
            return visible[max(0, first - context_count):first]

        # Large cold prefixes are explicitly metered exceptions, never a
        # silently truncated graph. The normal delta shares the Stage-2 call.
        while len(pending) > inline_turns or sum(len(str(m.get("text") or "")) for m in pending) > inline_chars:
            if len(audits) >= int(cfg.get("max_prefill_calls", 4)):
                raise ValueError("V8 cold-prefix budget exceeded; raise max_prefill_calls explicitly or resume cached progress")
            batch, chars = [], 0
            for message in pending[:batch_size]:
                size = len(str(message.get("text") or ""))
                if chars + size > inline_chars:
                    break
                batch.append(message)
                chars += size
            if not batch:
                raise ValueError("One visible turn exceeds the V8 text budget; increase inline_max_chars explicitly")
            state = projection()
            result = extract_v8_events(
                instance=instance, new_messages=batch, context_messages=context_for(batch),
                registry=registry, existing_scenes=state["current_scenes"], existing_entities=state["current_entities"],
                llm_client=self.llm_client, model_name=resolve_llm_model(self.config, "memory_ingestion"),
                response_protocol=(self.config.get("v8_query") or {}).get("response_protocol", "json"),
                memory_mode=self.memory_mode, resource_registry=shallow_resource_registry(state),
            )
            if result.audit.get("parse_failure") or _critical_extraction_rejections(result):
                prefill_gap = {
                    "status": "insufficient_evidence",
                    "reason": "prefill extraction returned incomplete or ungrounded records",
                    "rejected_reasons": result.audit.get("rejected_reasons", []),
                }
                # The optional graph/ledger channel cannot audit this prefix.
                # Keep the dense memory path alive and do not advance a partial
                # watermark that could be mistaken for a complete audit.
                pending = []
                break
            store.append(result, processed_turn_ids=[str(m.get("turn_id") or m.get("message_id")) for m in batch])
            store.save(path)
            audits.append(result.audit)
            pending = pending[len(batch):]
        state = projection()
        ingestion = None
        if pending:
            ingestion = {
                "domain_schema": v8_scene_schema(), "principal_registry": registry.candidate_payload(),
                "context_turns": [_turn_payload(m) for m in context_for(pending)],
                "new_turns": [_turn_payload(m) for m in pending],
                # Registries and static relationships are already in graph_context.
            }
        if ingestion is not None and self.memory_mode == "shallow":
            ingestion.pop("domain_schema", None)
        audit = {"memory_mode": self.memory_mode, "new_turn_count": new_count, "extraction_call_count": len(audits),
                 "joint_extraction": bool(pending), "batches": audits, "cache_path": str(path)}
        if prefill_gap:
            audit["audit_status"] = prefill_gap
        else:
            audit["audit_status"] = {"status": "complete"}
        return store, path, state, ingestion, audit

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        registry = build_v8_principal_registry(instance)
        telemetry_before = self.llm_client.telemetry_snapshot() if hasattr(self.llm_client, "telemetry_snapshot") else {}
        store, cache_path, state_projection, ingestion, extraction_audit = self._prepare_ingestion(instance, registry)
        chunks = _build_turn_chunks(instance)
        save_rag_chunks(self.output_dir, self.dataset_name, instance.instance_id, chunks)
        items = [chunk_to_memory_item(chunk) for chunk in chunks]
        index = DenseMemoryIndex.build(
            items=items,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            embedding_texts=[chunk.text for chunk in chunks],
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
        )
        top_k = int((self.config.get("rag") or {}).get("naive_top_k", 20))
        rows = index.query(
            query_texts=[instance.question], top_k=top_k,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
        )
        item_by_id = {item.memory_id: item for item in items}
        raw_evidence = [
            RetrievedEvidence(
                memory_id=item_by_id[memory_id].memory_id,
                content=item_by_id[memory_id].content,
                score=float(score), retrieval_source="dense",
                reason="v8 RAG-first Top-K retrieval",
                user_id=item_by_id[memory_id].user_id,
                memory_type="chunk", source_message_ids=item_by_id[memory_id].source_message_ids,
                time=item_by_id[memory_id].time,
                metadata=dict(item_by_id[memory_id].metadata or {}),
            )
            for memory_id, score in rows if memory_id in item_by_id
        ]
        dense_evidence = raw_evidence
        graph_audit = {"enabled": False, "llm_calls": 0}
        if (self.config.get("memory_governed_slot_graph") or {}).get("enabled", False):
            graph_rows, graph_audit = self._slot_graph.retrieve(
                conversation_id=str(instance.conversation_id or instance.instance_id),
                chunks=[c for c in chunks if str((instance.metadata.get("observable") or {}).get("as_of_turn_id") or "") not in c.source_message_ids],
                question=instance.question, llm_client=self.llm_client,
                model_name=resolve_llm_model(self.config, "memory_ingestion"), config=self.config)
            # The graph is an observer. It may report explicitly grounded
            # policy/deletion findings, but it never enters Stage-2 evidence.
            graph_audit["observer"] = _slot_graph_observer_audit(graph_audit)
            graph_audit["selection"] = {"mode": "observer_only", "dense_selected": len(dense_evidence),
                                         "graph_selected": 0, "graph_available": len(graph_rows),
                                         "dense_memory_ids": [row.memory_id for row in dense_evidence]}
        reasoning_evidence = _expand_with_adjacent_context(
            raw_evidence,
            chunks=chunks,
            max_extra=max(0, int((self.config.get("v8_query") or {}).get("max_adjacent_context", 12))),
        )
        visible_turn_ids = [
            str(item.get("turn_id") or item.get("message_id") or "") for item in instance.messages
        ]
        graph_context = _graph_side_context(
            instance=instance, store=store, state_projection=state_projection,
            static_relationships=registry.static_relationships,
            visible_turn_ids=visible_turn_ids, memory_mode=self.memory_mode,
        )
        def accept_events(events, sources):
            nonlocal ingestion, state_projection, graph_context
            fields = ("scenes", "entities", "facts", "relations", "governance_events")
            if not isinstance(events, dict) or any(not isinstance(events.get(k), list) for k in fields):
                raise ValueError("V8 joint extraction requires all five event arrays; watermark unchanged")
            if self.memory_mode == "shallow":
                ensure_shallow_events(events)
            extraction = validate_v8_extraction(
                events, turn_text=sources,
                registry=registry, allowed_output_turn_ids={m["turn_id"] for m in ingestion["new_turns"]},
                existing_entity_ids={e["entity_id"] for e in state_projection["current_entities"]}
                    | {r["scene_id"] for r in state_projection["current_scenes"]},
            )
            extraction.audit.update({"llm_calls": 0, "shared_stage2_call": True,
                                     "new_turn_count": len(ingestion["new_turns"])})
            # A malformed graph event is not a valid permission or veto. Keep
            # source-validated events and record the gap; the raw RAG evidence
            # remains available for the language governance pass.
            extraction.audit["audit_status"] = (
                "insufficient_evidence" if _critical_extraction_rejections(extraction)
                else "complete"
            )
            store.append(extraction, processed_turn_ids=[m["turn_id"] for m in ingestion["new_turns"]])
            store.save(cache_path)
            extraction_audit["batches"].append(extraction.audit)
            ingestion = None
            state_projection = project_v8_state(
                store, visible_turn_ids=visible_turn_ids, requester_principal_id=instance.asking_user_id,
                question=instance.question,
                checkpoint_time=instance.messages[-1].get("timestamp") if instance.messages else None,
            )
            graph_context = _graph_side_context(
                instance=instance, store=store, state_projection=state_projection,
                static_relationships=registry.static_relationships, visible_turn_ids=visible_turn_ids, memory_mode=self.memory_mode,
            )

        repair_feedback = None
        contract_failures = []
        contract_gap = None
        repair_limit = max(0, min(1, int((self.config.get("v8_query") or {}).get("max_contract_repairs", 0))))
        for contract_attempt in range(repair_limit + 1):
            try:
                claim_ledger = reason_v8_claims(
                    instance=instance, evidence=reasoning_evidence,
                    state_projection=state_projection, graph_context=graph_context,
                    graph_audit=graph_audit,
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "reasoning"),
                    max_candidate_chars=int((self.config.get("v8_query") or {}).get("max_candidate_chars", 0)),
                    ingestion=ingestion, repair_feedback=repair_feedback,
                    access_policy=application_policy(self.config, instance.domain),
                    compact_input=bool((self.config.get("v8_query") or {}).get("compact_input", False)),
                    candidate_id_style=(self.config.get("v8_query") or {}).get("candidate_id_style", "indexed"),
                    accept_events=accept_events, memory_mode=self.memory_mode,
                    response_protocol=(self.config.get("v8_query") or {}).get("response_protocol", "json"),
                )
                claim_ledger.pop("events", None)
                break
            except ValueError as exc:
                contract_failures.append(str(exc))
                if contract_attempt >= repair_limit:
                    if not _recoverable_governance_gap(exc):
                        raise
                    # No checked claim can be released from a malformed
                    # contract. Preserve the raw dense retrieval and the
                    # reason for the audit gap; never turn this into a permit.
                    contract_gap = {"status": "insufficient_evidence",
                                    "reason": str(exc), "attempts": len(contract_failures)}
                    claim_ledger = {"query_slots": [], "claims": [],
                                    "answer_action": "no_memory",
                                    "audit": {"validated": False},
                                    "prompt_audit": {"audit_status": "contract_gap"}}
                    break
                repair_feedback = str(exc)
        claim_ledger["audit"]["contract_failures"] = contract_failures
        claim_ledger["audit"]["contract_repair_calls"] = len(contract_failures)
        claim_ledger["audit"]["llm_calls"] = 1 + len(contract_failures)
        text_response_dir = None
        if claim_ledger.get("response_protocol") == "lines":
            text_response_dir = self.output_dir / "v8_text_responses" / self.dataset_name
            text_response_dir.mkdir(parents=True, exist_ok=True)
            text_id = str(instance.instance_id).replace("/", "_").replace("\\", "_")
            (text_response_dir / f"{text_id}.stage2.txt").write_text(claim_ledger["raw_text"], encoding="utf-8")
        # Also refresh relations/scenes for the symbolic verifier after the delta.
        graph_context = _graph_side_context(
            instance=instance, store=store, state_projection=state_projection,
            static_relationships=registry.static_relationships, visible_turn_ids=visible_turn_ids, memory_mode=self.memory_mode,
        )
        reviewed_ledger = criticize_v8_claims(
            claim_ledger,
            state_projection=state_projection,
            requester_principal_id=instance.asking_user_id,
            graph_context=graph_context,
        )
        safe_evidence, safe_audit = build_v8_safe_evidence(reviewed_ledger)
        answer_result = _answer_from_safe_evidence(
            instance=instance, safe_evidence=safe_evidence, safe_audit=safe_audit,
            llm_client=self.llm_client,
            model_name=resolve_llm_model(self.config, "answering"),
            stage2_prompt_audit=claim_ledger.get("prompt_audit") or {},
        )
        telemetry_after = self.llm_client.telemetry_snapshot() if hasattr(self.llm_client, "telemetry_snapshot") else {}
        chat_before = telemetry_before.get("chat/completions", {})
        chat_after = telemetry_after.get("chat/completions", {})
        cost = {key: chat_after.get(key, 0) - chat_before.get(key, 0)
                for key in set(chat_before) | set(chat_after)}
        cost.update({
            "logical_chat_calls": extraction_audit["extraction_call_count"] + graph_audit["llm_calls"] + 1 + len(contract_failures) + int(bool(safe_evidence)),
            "provider_requests": cost.get("calls", 0) + cost.get("retries", 0),
            "provider_telemetry_available": bool(telemetry_after),
            "prefill_calls": extraction_audit["extraction_call_count"],
            "stage2_prompt_chars": len(claim_ledger.get("prompt_audit", {}).get("user_prompt", "")),
        })
        answer_result.raw_response.update({
            "v8_cost": cost,
            "governed_slot_graph": graph_audit,
            "v8_claim_ledger": reviewed_ledger,
            "v8_safe_evidence": safe_audit,
            "governance_audit_status": contract_gap or {"status": "complete"},
            "v8_state_projection": state_projection,
            "event_ledger_summary": {
                "entity_count": len(store.entities), "fact_count": len(store.facts),
                "relation_count": len(store.relations),
                "governance_event_count": len(store.governance_events),
            },
        })
        plan = _query_plan(instance)
        reasoning_state = build_reasoning_state(
            safe_evidence,
            trace=[
                f"RAG-first retrieval retained {len(raw_evidence)} Top-K candidates and added "
                f"{len(reasoning_evidence) - len(raw_evidence)} adjacent context turns for Stage 2.",
                f"Late governance released {len(safe_evidence)} claims and blocked {safe_audit['blocked_claim_count']}.",
            ],
            current_state_ledger=state_projection,
            required_slot_plan={"query_slots": reviewed_ledger.get("query_slots") or []},
        )
        action_decision = GovernedActionDecision(
            action=answer_result.action,
            answer_mode="partial" if answer_result.action == "answer_redacted" else "direct" if answer_result.action == "answer" else "abstain",
            privacy_decision="late_claim_governance",
            forgetting_decision="explicit_veto_only",
            evidence_memory_ids=list(answer_result.used_memory_ids),
            rationale_summary=answer_result.reasoning_summary,
        )
        retrieval_result = {
            "retrieved_before_privacy_filter": raw_evidence,
            "retrieved_after_privacy_filter": safe_evidence,
            "filtered_evidence": safe_audit.get("blocked_claims") or [],
            "retrieval_candidates": [{"chunk_id": memory_id, "score": float(score)} for memory_id, score in rows],
            "rag_chunks": [chunk.__dict__ for chunk in chunks],
            "retrieval_backend": index.last_query_backend,
            "index_backend": index.backend,
            "embedding_model": str(self.config["embedding"]["model"]),
            "v8_dense_top_k_count": len(raw_evidence),
            "v8_adjacent_context_count": len(reasoning_evidence) - len(raw_evidence),
            "v8_reasoning_candidates": reasoning_evidence,
            "v8_extraction_audit": extraction_audit,
            "v8_state_projection": state_projection,
            "v8_graph_side_context": graph_context,
            "v8_cost": cost,
            "governed_slot_graph": graph_audit,
            "v8_claim_ledger": reviewed_ledger,
            "v8_safe_evidence": safe_audit,
        }
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload={
                "experiment_mode": "govmem_v8_late_governance",
                "v8_extraction_audit": extraction_audit,
                "v8_event_store": store.to_dict(),
                "v8_state_projection": state_projection,
                "v8_graph_side_context": graph_context,
                "v8_cost": cost,
            "governed_slot_graph": graph_audit,
                "v8_claim_ledger": reviewed_ledger,
                "v8_safe_evidence": safe_audit,
                "selected_evidence": [row.__dict__ for row in safe_evidence],
            },
        )


__all__ = ["GovMemV8LateGovernanceBackbone"]
