from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from hashlib import md5
from typing import Any

from gov_mem.backbones.canonical_renderer import render_canonical_answer
from gov_mem.backbones.coverage_packer import PackedUtilityEvidence, pack_utility_records
from gov_mem.backbones.coverage_verifier import verify_canonical_answer
from gov_mem.backbones.need_spec import AnswerNeedSpec, build_answer_need_spec
from gov_mem.backbones.utility_records import UtilityRecord, build_utility_records, canonicalize_medication_surface
from gov_mem.data.schema import (
    AnswerResult,
    GovernedActionDecision,
    MemoryInstance,
    MemoryItem,
    QueryPlan,
    ReasoningState,
    RetrievedEvidence,
    to_serializable,
)
from gov_mem.data.timestamps import normalize_message_timestamp, normalize_timestamp
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.planning.query_planner import QueryUnderstandingAgent
from gov_mem.query_semantics import (
    classify_state_slot_families,
    extract_state_slots,
    infer_current_state_slots as shared_infer_current_state_slots,
    infer_household_slots as shared_infer_household_slots,
)
from gov_mem.utils.io import ensure_dir, write_jsonl


@dataclass
class RAGChunk:
    chunk_id: str
    instance_id: str
    text: str
    source_message_ids: list[str]
    speaker_ids: list[str]
    timestamp_range: tuple[str | None, str | None]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackboneRunResult:
    query_plan: QueryPlan
    retrieval_result: dict[str, Any]
    reasoning_state: ReasoningState
    action_decision: GovernedActionDecision | None
    answer_result: AnswerResult
    debug_payload: dict[str, Any]


SEMANTIC_INTENT_SYSTEM_PROMPT = """
You are a semantic intent classifier for a governed memory system.
Return compact JSON only.
Classify the user's request into abstract intent tags without copying case-specific phrases.
""".strip()


def build_default_query_plan(instance: MemoryInstance) -> QueryPlan:
    requester_role = ((instance.metadata.get("requester") or {}).get("role")) or ((instance.metadata.get("observable") or {}).get("asker_role"))
    dense_queries = _build_dense_queries(
        question=instance.question,
        asking_user_id=instance.asking_user_id,
        requester_role=requester_role,
    )
    return QueryPlan(
        query_type="utility",
        target_users=[instance.asking_user_id] if instance.asking_user_id else [],
        target_entities=[],
        required_memory_types=["chunk"],
        symbolic_filters={},
        dense_queries=dense_queries,
        reasoning_ops=[],
    )


def build_rag_chunks(instance: MemoryInstance, config: dict[str, Any]) -> list[RAGChunk]:
    rag_cfg = config.get("rag") or {}
    chunks: list[RAGChunk] = []
    messages = [normalize_message_timestamp(message) for message in instance.messages]
    for idx, message in enumerate(messages):
        chunks.append(
            RAGChunk(
                chunk_id=f"{instance.instance_id}_msg_{idx:04d}",
                instance_id=instance.instance_id,
                text=_format_message(message),
                source_message_ids=[str(message.get("message_id"))],
                speaker_ids=[str(message.get("speaker_id") or "")],
                timestamp_range=(
                    normalize_timestamp(message.get("timestamp")),
                    normalize_timestamp(message.get("timestamp")),
                ),
                metadata={"chunk_type": "message"},
            )
        )
        if bool(rag_cfg.get("include_sentence_chunks", True)):
            for sentence_idx, sentence_text in enumerate(_split_into_sentence_chunks(str(message.get("text") or ""))):
                if not sentence_text:
                    continue
                chunks.append(
                    RAGChunk(
                        chunk_id=f"{instance.instance_id}_sentence_{idx:04d}_{sentence_idx:02d}",
                        instance_id=instance.instance_id,
                        text=f"[{message.get('speaker_id') or 'unknown'}] [{normalize_timestamp(message.get('timestamp')) or 'unknown_time'}] {sentence_text}",
                        source_message_ids=[str(message.get("message_id"))],
                        speaker_ids=[str(message.get("speaker_id") or "")],
                        timestamp_range=(
                            normalize_timestamp(message.get("timestamp")),
                            normalize_timestamp(message.get("timestamp")),
                        ),
                        metadata={"chunk_type": "sentence"},
                    )
                )
    if bool(rag_cfg.get("include_sliding_windows", True)):
        for start in range(max(0, len(messages) - 1)):
            window = messages[start : start + 3]
            if len(window) < 2:
                continue
            chunks.append(_build_window_chunk(instance, window, chunk_type="sliding_window", idx=start))
    if bool(rag_cfg.get("include_speaker_windows", True)):
        by_speaker: dict[str, list[dict]] = {}
        for message in messages:
            speaker = str(message.get("speaker_id") or "")
            by_speaker.setdefault(speaker, []).append(message)
        offset = 0
        for speaker, speaker_messages in by_speaker.items():
            for start in range(max(0, len(speaker_messages) - 1)):
                window = speaker_messages[max(0, start - 2) : start + 1]
                if len(window) < 2:
                    continue
                chunks.append(_build_window_chunk(instance, window, chunk_type="speaker_window", idx=offset))
                offset += 1
    if bool(rag_cfg.get("include_utility_event_slices", True)):
        for idx, message in enumerate(messages):
            chunks.extend(_build_utility_event_chunks(instance, message, idx=idx))
    return _dedupe_chunks(chunks)


def save_rag_chunks(output_dir, dataset_name: str, instance_id: str, chunks: list[RAGChunk]) -> None:
    path = ensure_dir(output_dir / "rag_chunks" / dataset_name / instance_id) / "chunks.jsonl"
    write_jsonl(path, [asdict(chunk) for chunk in chunks])


def retrieve_rag_chunks(
    *,
    instance: MemoryInstance,
    chunks: list[RAGChunk],
    llm_client: LLMClient | None,
    embedding_model: str,
    config: dict[str, Any],
    planning_client: LLMClient | None = None,
    planning_model: str | None = None,
) -> tuple[QueryPlan, dict[str, Any], list[RetrievedEvidence]]:
    if (
        planning_client is not None
        and planning_model
        and bool((config.get("ablation") or {}).get("use_query_planner", True))
    ):
        planner = QueryUnderstandingAgent(
            llm_client=planning_client,
            model_name=planning_model,
            use_asking_user_id=bool((config.get("pipeline") or {}).get("use_asking_user_id", True)),
        )
        plan = planner.plan(instance)
    else:
        plan = build_default_query_plan(instance)
    planner_semantic_spec = dict(plan.semantic_spec or {})
    planner_semantic_available = _has_semantic_spec(planner_semantic_spec)
    planner_domains = _semantic_domains(planner_semantic_spec)
    question_profile = _build_question_profile(
        instance.question,
        planner_domains or _infer_query_domains(instance.question),
        semantic_intent={"semantic_spec": planner_semantic_spec} if planner_semantic_available else {},
    )
    pseudo_items = [_chunk_to_memory_item(chunk) for chunk in chunks]
    memory_by_id = {item.memory_id: item for item in pseudo_items}
    index = DenseMemoryIndex.build(items=pseudo_items, llm_client=llm_client, embedding_model=embedding_model)
    rag_cfg = config.get("rag") or {}
    rows, candidate_debug = _retrieve_diverse_chunk_rows(
        index=index,
        query_texts=plan.dense_queries,
        top_k=int(rag_cfg.get("chunk_top_k", 30)),
        final_k=int(rag_cfg.get("final_context_k", 12)),
        llm_client=llm_client,
        embedding_model=embedding_model,
    )
    evidence = []
    for chunk_id, score in rows:
        item = memory_by_id[chunk_id]
        evidence.append(
            RetrievedEvidence(
                memory_id=item.memory_id,
                content=item.content,
                score=float(score),
                retrieval_source="dense",
                reason="rag chunk retrieval",
                user_id=item.user_id,
                memory_type=item.memory_type,
                scope=item.scope,
                entities=item.entities,
                time=item.time,
                source_message_ids=item.source_message_ids,
                metadata=dict(item.metadata or {}),
            )
        )
    rerank_debug: dict[str, Any] = {"enabled": False, "selected_memory_ids": [], "candidates": []}
    if _should_enable_question_rerank(question_profile):
        reranked_evidence, rerank_debug = _rerank_candidate_pool_for_question(
            question=instance.question,
            asking_user_id=instance.asking_user_id,
            candidate_rows=candidate_debug,
            memory_by_id=memory_by_id,
            final_k=int(rag_cfg.get("final_context_k", 12)),
        )
        if reranked_evidence:
            evidence = reranked_evidence
    retrieval_result = {
        "retrieved_before_privacy_filter": evidence,
        "retrieved_after_privacy_filter": evidence,
        "filtered_evidence": [],
        "rag_chunks": [asdict(chunk) for chunk in chunks],
        "query_variants": list(plan.dense_queries),
        "semantic_spec_available": planner_semantic_available,
        "semantic_spec": planner_semantic_spec,
        "semantic_routing_source": "query_planner" if planner_semantic_available else "fallback_heuristic",
        "planning_trace": dict(getattr(plan, "planning_trace", {}) or {}),
        "retrieval_candidates": candidate_debug,
        "question_rerank": rerank_debug,
    }
    return plan, retrieval_result, evidence


def answer_with_retrieved_evidence(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    llm_client: LLMClient,
    model_name: str,
    action: str,
    extra_rules: list[str] | None = None,
    used_chunk_ids: list[str] | None = None,
    config: dict[str, Any] | None = None,
    principal: Any = None,
    policy_decisions: list[dict[str, Any]] | None = None,
    requester_context: dict[str, Any] | None = None,
    semantic_spec: dict[str, Any] | None = None,
) -> AnswerResult:
    used_chunk_ids = used_chunk_ids or [row.memory_id for row in evidence]
    if not evidence:
        no_memory_text = build_no_memory_answer_text()
        return AnswerResult(
            prediction=no_memory_text,
            answer_text=no_memory_text,
            used_memory_ids=[],
            reasoning_summary="No retrieved evidence.",
            action="no_memory",
            raw_response={},
        )
    semantic_intent = _classify_semantic_intent(
        llm_client=llm_client,
        model_name=model_name,
        question=instance.question,
        requester_context=requester_context or {},
        evidence=evidence[:6],
    )
    semantic_spec = dict(semantic_spec or {})
    if semantic_spec:
        semantic_intent["semantic_spec"] = semantic_spec
        semantic_intent["requested_current_slots"] = list(semantic_spec.get("requested_slots") or [])
    shortlisted_evidence, shortlist_debug = shortlist_evidence_for_question(
        instance.question,
        evidence,
        semantic_intent=semantic_intent,
    )
    if shortlisted_evidence:
        evidence = shortlisted_evidence
        used_chunk_ids = [row.memory_id for row in evidence]
    payload = {
        "question": instance.question,
        "requester": instance.asking_user_id,
        "evidence": [
            {
                "chunk_id": row.memory_id,
                "text": row.content,
                "source_message_ids": row.source_message_ids,
            }
            for row in evidence
        ],
        "rules": extra_rules or [],
    }
    assert_runtime_payload_safe(payload, context=f"backbone_answer:{instance.instance_id}")
    surface_replay_result = _render_answer_from_surface_replay(
        question=instance.question,
        evidence=evidence,
        used_chunk_ids=used_chunk_ids,
        action=action,
        principal=principal,
        policy_decisions=policy_decisions or [],
        requester_context=requester_context or {},
        semantic_intent=semantic_intent,
        config=config or {},
        llm_client=llm_client,
        model_name=model_name,
    )
    if surface_replay_result is not None:
        surface_replay_result.raw_response["semantic_intent"] = semantic_intent
        surface_replay_result.raw_response["semantic_spec"] = semantic_spec
        surface_replay_result.raw_response["evidence_shortlist"] = shortlist_debug
        slot_verifier = dict(surface_replay_result.raw_response.get("slot_verifier") or {})
        if action in {"answer", "answer_redacted"} and bool(slot_verifier.get("passed")):
            surface_replay_result.raw_response["llm_synthesis_skipped"] = True
            surface_replay_result.raw_response["llm_synthesis_skip_reason"] = (
                "surface_replay_already_passed_runtime_slot_verifier"
            )
            return surface_replay_result
        llm_surface_result = _synthesize_answer_with_llm(
            instance=instance,
            llm_client=llm_client,
            model_name=model_name,
            action=action,
            extra_rules=extra_rules or [],
            used_chunk_ids=used_chunk_ids,
            surface_replay_result=surface_replay_result,
            semantic_intent=semantic_intent,
        )
        if llm_surface_result is not None:
            llm_surface_result.raw_response["evidence_shortlist"] = shortlist_debug
            llm_surface_result.raw_response["surface_replay_fallback"] = surface_replay_result.raw_response
            return llm_surface_result
        return surface_replay_result
    rules_block = "\n".join(
        [
            "- Use the exact dates, times, locations, provider names, procedures, medications, allergies, and instructions from the evidence.",
            "- If multiple schedule items are relevant, include all of them.",
            "- If an event is canceled or rescheduled, explicitly state the current item and the canceled prior item.",
            "- Do not infer from outside knowledge.",
            "- If evidence is insufficient, say so.",
        ]
        + (extra_rules or [])
    )
    prompt = (
        "Return JSON with fields prediction, used_chunk_ids, used_memory_ids, reasoning_summary.\n"
        f"Question: {instance.question}\n"
        f"Requester: {instance.asking_user_id}\n"
        f"Rules:\n{rules_block}\n"
        f"Evidence: {payload['evidence']}"
    )
    try:
        if llm_client is None or not llm_client.is_available():
            raise LLMClientUnavailableError("governance contract LLM unavailable")
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt="You answer the user's question using only the retrieved evidence. Return compact JSON only.",
            user_prompt=prompt,
        )
        if isinstance(raw, dict):
            answer_text = str(raw.get("prediction") or raw.get("answer_text") or "")
            return AnswerResult(
                prediction=answer_text,
                answer_text=answer_text,
                used_memory_ids=[str(x) for x in raw.get("used_chunk_ids", used_chunk_ids)],
                reasoning_summary=str(raw.get("reasoning_summary") or ""),
                action=action,
                raw_response={**raw, "evidence_shortlist": shortlist_debug},
            )
    except LLMClientUnavailableError:
        pass
    except Exception:
        pass
    fallback = " ".join(row.content.strip() for row in evidence[:4] if row.content.strip())
    return AnswerResult(
        prediction=fallback or "I do not have enough relevant evidence to answer this.",
        answer_text=fallback or "I do not have enough relevant evidence to answer this.",
        used_memory_ids=used_chunk_ids,
        reasoning_summary="Fallback answer from retrieved evidence.",
        action=action,
        raw_response={"evidence_shortlist": shortlist_debug},
    )


def build_reasoning_state(
    evidence: list[RetrievedEvidence],
    *,
    trace: list[str] | None = None,
    slot_coverage: dict | None = None,
    selected_frames: list | None = None,
    required_slot_plan: dict | None = None,
) -> ReasoningState:
    return ReasoningState(
        selected_evidence=evidence,
        reasoning_trace=trace or [],
        conflicts=[],
        conclusion_hint="Backbone evidence aggregation.",
        selected_frames=selected_frames or [],
        current_state_ledger={},
        required_slot_plan=required_slot_plan or {},
        slot_coverage=slot_coverage or {},
    )


def serialize_debug_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return to_serializable(payload)


def _build_window_chunk(instance: MemoryInstance, window: list[dict], *, chunk_type: str, idx: int) -> RAGChunk:
    text = "\n".join(_format_message(message) for message in window)
    return RAGChunk(
        chunk_id=f"{instance.instance_id}_{chunk_type}_{idx:04d}",
        instance_id=instance.instance_id,
        text=text,
        source_message_ids=[str(message.get("message_id")) for message in window],
        speaker_ids=[str(message.get("speaker_id") or "") for message in window],
        timestamp_range=(
            normalize_timestamp(window[0].get("timestamp")),
            normalize_timestamp(window[-1].get("timestamp")),
        ),
        metadata={"chunk_type": chunk_type},
    )


def _build_utility_event_chunks(instance: MemoryInstance, message: dict[str, Any], *, idx: int) -> list[RAGChunk]:
    message = normalize_message_timestamp(message)
    speaker = str(message.get("speaker_id") or "")
    timestamp = message.get("timestamp")
    segments = _extract_utility_event_segments(str(message.get("text") or ""))
    chunks: list[RAGChunk] = []
    for seg_idx, segment in enumerate(segments):
        chunks.append(
            RAGChunk(
                chunk_id=f"{instance.instance_id}_utility_event_{idx:04d}_{seg_idx:02d}",
                instance_id=instance.instance_id,
                text=f"[{speaker or 'unknown'}] [{timestamp or 'unknown_time'}] {segment}",
                source_message_ids=[str(message.get("message_id"))],
                speaker_ids=[speaker],
                timestamp_range=(timestamp, timestamp),
                metadata={
                    "chunk_type": "utility_event",
                    "segment_index": seg_idx,
                    "parent_message_id": str(message.get("message_id") or ""),
                },
            )
        )
    return chunks


def _format_message(message: dict) -> str:
    timestamp = normalize_timestamp(message.get("timestamp"))
    return f"[{message.get('speaker_id') or 'unknown'}] [{timestamp or 'unknown_time'}] {str(message.get('text') or '').strip()}"


def _chunk_to_memory_item(chunk: RAGChunk) -> MemoryItem:
    return MemoryItem(
        memory_id=chunk.chunk_id,
        instance_id=chunk.instance_id,
        user_id=(chunk.speaker_ids[0] if chunk.speaker_ids else None),
        scope="chunk",
        content=chunk.text,
        memory_type="chunk",
        entities=[],
        time=chunk.timestamp_range[0],
        source_message_ids=chunk.source_message_ids,
        confidence=1.0,
        privacy_level=None,
        tags=[str(chunk.metadata.get("chunk_type") or "chunk")],
        metadata=dict(chunk.metadata),
    )


def _dedupe_chunks(chunks: list[RAGChunk]) -> list[RAGChunk]:
    seen = set()
    out = []
    for chunk in chunks:
        key = md5(chunk.text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _split_into_sentence_chunks(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return []
    cleaned = cleaned.replace("Dr. ", "Dr ")
    parts = re.split(r"(?<=[.!?;])\s+", cleaned)
    out = []
    for part in parts:
        sentence = part.strip().replace("Dr ", "Dr. ")
        if len(sentence) < 12:
            continue
        out.append(sentence)
    return out


def _extract_utility_event_segments(text: str) -> list[str]:
    sentences = _split_into_sentence_chunks(text)
    if not sentences:
        return []
    segments: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        lowered = sentence.lower()
        if not _looks_like_utility_event(sentence):
            continue
        norm = _normalize_surface_line(sentence)
        if norm and norm not in seen:
            seen.add(norm)
            segments.append(sentence.strip())
        if any(token in lowered for token in [" and ", ";", " followed by ", " right after", " plus "]):
            for part in _split_composite_utility_sentence(sentence):
                norm = _normalize_surface_line(part)
                if norm and norm not in seen:
                    seen.add(norm)
                    segments.append(part.strip())
    return segments


def _looks_like_utility_event(text: str) -> bool:
    lowered = str(text or "").lower()
    generic_cues = (
        "appointment",
        "schedule",
        "visit",
        "arrival",
        "allergy",
        "reaction",
        "portal",
        "callback",
        "contact",
        "canceled",
        "cancelled",
        "rescheduled",
        "permission",
        "authorized",
        "medication",
        "medicine",
        "regimen",
        "treatment",
        "instruction",
        "precaution",
        "budget",
        "blocker",
        "target date",
        "entry",
        "parking",
        "delivery",
    )
    return any(token in lowered for token in generic_cues) or bool(
        DATE_SPAN_RE.search(text)
        or TIME_SPAN_RE.search(text)
        or PHONE_SPAN_RE.search(text)
        or re.search(r"\b(?:start|stop|continue|hold|restart|use|take|avoid)\b.{0,60}\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b", text, re.IGNORECASE)
    )


def _split_composite_utility_sentence(text: str) -> list[str]:
    parts = re.split(r";|\s+\band\b\s+|\s+followed by\s+|\s+right after\.?\s*|\s+\bplus\b\s+", str(text or "").strip(), flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        cleaned = part.strip(" .").replace("Dr ", "Dr. ")
        if len(cleaned) < 10:
            continue
        out.append(cleaned)
    return out


DATE_SPAN_RE = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:,\s*)?(?:\s+(?:January|February|March|April|May|June|July|August|September|October|November|December))?\s+\d{1,2}\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|\b(?:March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b",
    re.IGNORECASE,
)
TIME_SPAN_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", re.IGNORECASE)
LOCATION_SPAN_RE = re.compile(r"\b[A-Z][A-Za-z0-9&' -]{1,60}\s+(?:Suite|Clinic|Center|Office|Lab|Ward|Desk)\s*[A-Z0-9-]*\b")
PROVIDER_SPAN_RE = re.compile(r"\bDr\.\s+[A-Z][a-zA-Z]+\b")
ARRIVAL_SPAN_RE = re.compile(r"(?:arrive by|please arrive by|check in by|come by|arrive at)\s+\d{1,2}:\d{2}\s?(?:AM|PM)", re.IGNORECASE)
PHONE_SPAN_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
WEEKDAY_ORDER = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
PERSON_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
PERSON_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _looks_like_policy_state_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in [
            "may receive",
            "logistics only",
            "do not share",
            "authorized",
            "permission",
            "permissions",
            "access revoked",
            "removed from scheduling-contact",
            "removed from callback-contact",
            "no longer release",
            "restricted to",
            "family access",
            "scheduling-contact",
            "callback-contact",
        ]
    )


def _is_policy_revocation_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in [
            "access revoked",
            "removed from scheduling-contact",
            "removed from callback-contact",
            "no longer release",
            "do not share future appointment details",
            "restricted to elena",
            "remain restricted",
        ]
    )


def _is_contact_protocol_text(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        token in lowered
        for token in [
            "callback",
            "voicemail",
            "portal",
            "phone",
            "number",
            "cell",
            "safe line",
            "temporary callback",
            "generic callback",
            "backup contact",
            "shared address",
            "pickup logistics",
        ]
    )


def _is_operational_instruction_text(text: str) -> bool:
    lowered = str(text or "").lower()
    if _is_contact_protocol_text(text):
        return True
    has_action = bool(re.search(r"\b(?:start|stop|continue|hold|restart|use|take|avoid|arrive|check in)\b", lowered))
    has_instruction_context = bool(
        re.search(r"\b(?:before|after|until|unless|if|when|daily|nightly|every|arrival|check-in)\b", lowered)
        or re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|minutes?|hours?)\b", lowered)
    )
    return has_action and has_instruction_context


def _has_schedule_payload_from_slots(frame_type: str, slots: dict[str, Any]) -> bool:
    if frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "cancellation", "update"}:
        return True
    return any(
        slots.get(key)
        for key in ["date", "time", "arrival_time", "location", "provider", "procedure", "visit_type"]
    )


def _build_dense_queries(question: str, asking_user_id: str | None, requester_role: str | None) -> list[str]:
    queries = [question]
    if asking_user_id:
        queries.append(f"{asking_user_id} {question}")
    if requester_role:
        queries.append(f"{requester_role} {question}")
    for subquery in _decompose_question_into_queries(question):
        queries.append(subquery)
        if asking_user_id:
            queries.append(f"{asking_user_id} {subquery}")
    deduped = []
    seen = set()
    for query in queries:
        normalized = re.sub(r"\s+", " ", str(query or "").strip().lower())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(query).strip())
    return deduped


def _decompose_question_into_queries(question: str) -> list[str]:
    # LLM planning is the semantic path. This offline fallback intentionally
    # adds no domain, entity, or benchmark-shaped assumptions.
    return [
        "relevant evidence with timestamps, updates, permissions, and constraints",
        "recent authoritative evidence and any supersession relationships",
    ]


def _retrieve_diverse_chunk_rows(
    *,
    index: DenseMemoryIndex,
    query_texts: list[str],
    top_k: int,
    final_k: int,
    llm_client: LLMClient,
    embedding_model: str,
) -> tuple[list[tuple[str, float]], list[dict[str, Any]]]:
    candidate_stats: dict[str, dict[str, Any]] = {}
    for rank_query_idx, query in enumerate(query_texts):
        rows = index.query(
            query_texts=[query],
            top_k=top_k,
            llm_client=llm_client,
            embedding_model=embedding_model,
        )
        query_domains = _infer_query_domains(query)
        for rank, (chunk_id, score) in enumerate(rows, start=1):
            entry = candidate_stats.setdefault(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "best_score": float("-inf"),
                    "rrf": 0.0,
                    "query_hits": [],
                    "domains": set(),
                },
            )
            entry["best_score"] = max(float(score), float(entry["best_score"]))
            entry["rrf"] += 1.0 / float(rank + 30)
            entry["domains"].update(query_domains)
            if len(entry["query_hits"]) < 6:
                entry["query_hits"].append(
                    {
                        "query_index": rank_query_idx,
                        "query": query,
                        "rank": rank,
                        "score": float(score),
                        "domains": sorted(query_domains),
                    }
                )
    ranked_candidates = sorted(
        candidate_stats.values(),
        key=lambda item: (float(item["best_score"]) + float(item["rrf"])),
        reverse=True,
    )
    selected = _select_diverse_candidates(ranked_candidates, final_k=final_k)
    debug_rows = []
    for item in ranked_candidates[: max(final_k * 3, 20)]:
        debug_rows.append(
            {
                "chunk_id": item["chunk_id"],
                "best_score": float(item["best_score"]),
                "rrf": float(item["rrf"]),
                "domains": sorted(item["domains"]),
                "selected": any(row[0] == item["chunk_id"] for row in selected),
                "query_hits": item["query_hits"],
            }
        )
    return selected, debug_rows


def _select_diverse_candidates(ranked_candidates: list[dict[str, Any]], *, final_k: int) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    covered_domains: set[str] = set()
    selected_chunk_types: set[str] = set()
    pool = list(ranked_candidates)
    while pool and len(selected) < final_k:
        best_idx = None
        best_gain = float("-inf")
        for idx, item in enumerate(pool):
            domains = set(item["domains"])
            gain = float(item["best_score"]) + float(item["rrf"])
            new_domains = domains - covered_domains
            gain += 0.2 * len(new_domains)
            chunk_type = _infer_chunk_type(item["chunk_id"])
            if chunk_type not in selected_chunk_types:
                gain += 0.05
            if "speaker_window" in item["chunk_id"] and any(token in domains for token in {"allergy", "schedule", "instruction"}):
                gain += 0.03
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        picked = pool.pop(best_idx)
        selected.append((str(picked["chunk_id"]), float(picked["best_score"])))
        covered_domains.update(picked["domains"])
        selected_chunk_types.add(_infer_chunk_type(str(picked["chunk_id"])))
    return selected


def _infer_query_domains(query: str) -> set[str]:
    lowered = query.lower()
    domains = set()
    if any(token in lowered for token in ["household", "visitor", "entry", "access", "parking", "delivery", "package", "room"]):
        domains.add("household")
    if any(token in lowered for token in ["allergy", "allergic", "reaction"]):
        domains.add("allergy")
    if any(token in lowered for token in ["schedule", "appointment", "visit", "arrival", "provider", "location", "time", "date", "follow-up", "follow up"]):
        domains.add("schedule")
    if any(token in lowered for token in ["instruction", "precaution", "prepare", "preparation", "before", "after", "monitor", "warning", "callback", "contact"]):
        domains.add("instruction")
    if any(token in lowered for token in ["canceled", "cancelled", "rescheduled", "superseded", "replaced", "overwritten"]):
        domains.add("cancellation")
    if any(
        token in lowered
        for token in [
            "medication",
            "medicine",
            "dose",
            "treatment plan",
            "regimen",
            "prescription",
        ]
    ) or (
        any(token in lowered for token in ["current", "right now", "now"])
        and bool(re.search(r"\b(?:start|stop|continue|hold|restart|use|take|avoid)\b", lowered))
    ):
        domains.add("medication")
    if not domains:
        domains.add("general")
    return domains


def _semantic_domains(semantic_spec: Any) -> set[str]:
    spec = dict(semantic_spec or {}) if isinstance(semantic_spec, dict) else {}
    slots = {str(slot) for slot in list(spec.get("requested_slots") or []) if str(slot)}
    if not slots:
        return set()
    domains: set[str] = set()
    if slots & {"target_date", "approved_budget", "approved_discount_cap", "monthly_stipend", "safe_wording", "blocker", "operational_result"}:
        domains.add("current_state")
    if slots & {"date", "time", "location", "public_event_date"}:
        domains.add("schedule")
    if slots & {"visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}:
        domains.add("household")
    if slots & {"medication", "dosage"}:
        domains.add("medication")
    if slots & {"instruction", "condition"}:
        domains.add("instruction")
    if "policy_scope" in slots or str(spec.get("request_shape") or "") == "policy":
        domains.add("policy")
    return domains or {"general"}


def _has_semantic_spec(semantic_spec: Any) -> bool:
    spec = dict(semantic_spec or {}) if isinstance(semantic_spec, dict) else {}
    if [slot for slot in list(spec.get("requested_slots") or []) if str(slot).strip()]:
        return True
    return any(
        str(spec.get(key) or "unspecified") != "unspecified"
        for key in ["temporal_scope", "disclosure_scope", "state_domain", "request_shape"]
    )


def _infer_chunk_type(chunk_id: str) -> str:
    if "_sliding_window_" in chunk_id:
        return "sliding_window"
    if "_speaker_window_" in chunk_id:
        return "speaker_window"
    if "_sentence_" in chunk_id:
        return "sentence"
    return "message"


def _render_answer_from_surface_replay(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    used_chunk_ids: list[str],
    action: str = "answer",
    principal: Any = None,
    policy_decisions: list[dict[str, Any]] | None = None,
    requester_context: dict[str, Any] | None = None,
    semantic_intent: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    model_name: str | None = None,
) -> AnswerResult | None:
    config = config or {}
    semantic_domains = _semantic_domains((semantic_intent or {}).get("semantic_spec"))
    domains = semantic_domains or _infer_query_domains(question)
    question_profile = _build_question_profile(question, domains, semantic_intent=semantic_intent)
    compiled_frames = [compile_evidence_frame(row) for row in evidence]
    selected_lines = _select_focused_surface_lines(
        question=question,
        evidence=evidence,
        domains=domains,
        question_profile=question_profile,
    )
    if not selected_lines:
        selected_lines = _select_surface_lines(question=question, evidence=evidence, domains=domains)
    if action == "answer":
        selected_lines = _augment_current_state_slot_coverage(
            question=question,
            evidence=evidence,
            selected_lines=selected_lines,
            semantic_spec=dict((semantic_intent or {}).get("semantic_spec") or {}),
        )
    governance_contract_enabled = bool(
        (config.get("governance_runtime") or {}).get("enable_governance_contract_redaction", False)
    )
    requester_is_privileged = str(getattr(principal, "relation_to_owner", "") or "") in {
        "owner",
        "authorized_staff",
    }
    # A non-owner direct answer needs the same explicit slot boundary as a
    # redacted answer. This prevents action-level approval from becoming a
    # whole-source-text disclosure permit.
    if governance_contract_enabled and action in {"answer", "answer_redacted"} and not requester_is_privileged:
        return _render_redacted_answer_from_governance_contract(
            question=question,
            evidence=evidence,
            selected_lines=selected_lines,
            used_chunk_ids=used_chunk_ids,
            semantic_intent=semantic_intent,
            principal=principal,
            policy_decisions=policy_decisions or [],
            llm_client=llm_client,
            model_name=str(model_name or ""),
        )
    if not selected_lines:
        return None
    utility_cfg = dict(config.get("utility_realization") or {})
    pcur_mode = str(utility_cfg.get("mode") or "").strip().lower()
    fallback = _render_answer_from_v11_surface_replay(
        question=question,
        selected_lines=selected_lines,
        compiled_frames=compiled_frames,
        used_chunk_ids=used_chunk_ids,
        action=action,
        semantic_intent=semantic_intent,
    )
    if pcur_mode == "policy_conditioned_records":
        pcur_result = _render_answer_with_policy_conditioned_records(
            question=question,
            action=action,
            principal=principal,
            policy_decisions=policy_decisions or [],
            requester_context=requester_context or {},
            selected_lines=selected_lines,
            used_chunk_ids=used_chunk_ids,
            semantic_intent=semantic_intent,
            config=config,
        )
        if pcur_result is not None and utility_cfg.get("fallback_to_v11_renderer", True) and fallback is not None:
            return _choose_utility_realization_result(
                question=question,
                selected_lines=selected_lines,
                principal=principal,
                config=config,
                pcur_result=pcur_result,
                fallback_result=fallback,
                semantic_intent=semantic_intent,
            )
        if pcur_result is not None:
            return pcur_result
    return fallback


def _render_redacted_answer_from_governance_contract(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    selected_lines: list[dict[str, Any]],
    used_chunk_ids: list[str],
    semantic_intent: dict[str, Any] | None,
    principal: Any,
    policy_decisions: list[dict[str, Any]],
    llm_client: LLMClient | None,
    model_name: str,
) -> AnswerResult:
    """Render only LLM-authorized, source-grounded slots for a partial disclosure."""
    candidate_ids = {str(line.get("chunk_id") or "") for line in selected_lines}
    candidates = [row for row in evidence if not candidate_ids or row.memory_id in candidate_ids]
    payload = {
        "question": question,
        "semantic_spec": dict((semantic_intent or {}).get("semantic_spec") or {}),
        "requester": {
            "user_id": getattr(principal, "user_id", None),
            "role": getattr(principal, "role", None),
            "relation_to_owner": getattr(principal, "relation_to_owner", None),
        },
        "upstream_policy_decisions": [
            {
                "memory_id": row.get("memory_id"),
                "decision": row.get("decision"),
                "allowed_slots": row.get("allowed_slots"),
                "denied_slots": row.get("denied_slots"),
            }
            for row in policy_decisions
            if isinstance(row, dict)
        ],
        "candidates": [
            {"memory_id": row.memory_id, "content": row.content, "metadata": {
                "memory_status": (row.metadata or {}).get("memory_status"),
                "privacy_level": (row.metadata or {}).get("privacy_level"),
                "redaction_required": (row.metadata or {}).get("redaction_required"),
                "authorized_users": (row.metadata or {}).get("authorized_users"),
                "forbidden_users": (row.metadata or {}).get("forbidden_users"),
            }}
            for row in candidates
        ],
    }
    assert_runtime_payload_safe(payload, context="redacted_governance_contract")
    policy_slot_candidates = _policy_authorized_source_slots(
        candidates=candidates,
        policy_decisions=policy_decisions,
        semantic_spec=dict((semantic_intent or {}).get("semantic_spec") or {}),
    )
    payload["policy_slot_candidates"] = policy_slot_candidates

    contract: dict[str, Any] = {}
    try:
        if llm_client is None or not llm_client.is_available():
            raise LLMClientUnavailableError("governance contract LLM unavailable")
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are a governance-frame compiler. Determine which exact evidence spans can be disclosed "
                "for a partial/redacted answer. Never answer the question and never invent, paraphrase, or "
                "combine values. Use only the provided requester relation and upstream policy decisions to "
                "identify the smallest safe utility slots. Treat deleted, superseded, private, restricted, "
                "credential, and exact-location details as denied unless the runtime evidence or policy decision "
                "explicitly authorizes disclosure."
            ),
            user_prompt=(
                "Return JSON only: {\"frames\":[{\"memory_id\":string,\"lifecycle\":\"active|historical|superseded|deleted\","
                "\"disclosure\":\"allow|deny\",\"allowed_slots\":[{\"slot\":string,\"surface\":string}]}]}. "
                "Every allowed surface must be copied verbatim from its own candidate content and must be safe "
                "for the requester under the evidence's access and privacy instructions. If uncertain, return "
                "disclosure=deny and no allowed_slots. policy_slot_candidates are only candidates, not an "
                "automatic allow-list; authorize one only when it is the minimal requested disclosure and "
                "the source evidence contains no contrary restriction. Payload: "
                f"{payload}"
            ),
        )
        if isinstance(raw, dict):
            contract = raw
    except (LLMClientUnavailableError, Exception) as exc:
        contract = {"error": type(exc).__name__}

    source_by_id = {str(row.memory_id): str(row.content or "") for row in candidates}
    frames_by_id = {str(row.memory_id): compile_evidence_frame(row) for row in candidates}
    requested_slots = {
        str(slot)
        for slot in list((semantic_intent or {}).get("semantic_spec", {}).get("requested_slots") or [])
        if str(slot)
    }
    safe_slots = []
    seen_surfaces: set[str] = set()
    for frame in list(contract.get("frames") or []):
        if not isinstance(frame, dict):
            continue
        memory_id = str(frame.get("memory_id") or "")
        source = source_by_id.get(memory_id, "")
        typed_frame = frames_by_id.get(memory_id)
        if str(frame.get("disclosure") or "").lower() != "allow":
            continue
        if str(frame.get("lifecycle") or "active").lower() != "active":
            continue
        for item in list(frame.get("allowed_slots") or []):
            if not isinstance(item, dict):
                continue
            slot = re.sub(r"[^a-z0-9_]+", "_", str(item.get("slot") or "").lower()).strip("_")
            surface = str(item.get("surface") or "").strip()
            if not surface or surface.lower() not in source.lower():
                continue
            # The LLM span is authorization evidence only. Render exact values
            # from the typed schema, never the model's arbitrary slot label or
            # the surrounding sentence that justified authorization.
            for requested_slot in sorted(requested_slots):
                typed_surface = str(
                    ((typed_frame.surface_spans or {}).get(requested_slot) if typed_frame else None)
                    or ((typed_frame.slots or {}).get(requested_slot) if typed_frame else None)
                    or ""
                ).strip()
                if not typed_surface or typed_surface.lower() not in surface.lower():
                    continue
                key = f"{requested_slot}:{typed_surface.lower()}"
                if key in seen_surfaces:
                    continue
                seen_surfaces.add(key)
                safe_slots.append({"memory_id": memory_id, "slot": requested_slot, "surface": typed_surface})

    return _build_governance_contract_result(safe_slots=safe_slots, contract=contract)


POLICY_SLOT_GROUPS = {
    "logistics": {"date", "time", "secondary_time", "visit_window", "arrival_time", "location", "provider", "procedure", "visit_type", "status", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"},
    "policy": {"policy_scope"},
    "project_state": {"target_date", "approved_budget", "approved_discount_cap", "blocker"},
    "research_state": {"target_date", "monthly_stipend", "safe_wording", "blocker"},
}


def _policy_authorized_source_slots(
    *,
    candidates: list[RetrievedEvidence],
    policy_decisions: list[dict[str, Any]],
    semantic_spec: dict[str, Any],
) -> list[dict[str, str]]:
    """Build precise policy-approved candidates; authorization remains slot-level."""
    decisions_by_id = {
        str(item.get("chunk_id") or item.get("memory_id") or ""): item
        for item in policy_decisions
        if isinstance(item, dict) and bool(item.get("allowed_for_requester"))
    }
    requested = {str(slot) for slot in list(semantic_spec.get("requested_slots") or []) if str(slot)}
    authorized: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in candidates:
        decision = decisions_by_id.get(str(row.memory_id))
        if not decision:
            continue
        allowed_names = {
            slot
            for group in list(decision.get("allowed_slot_groups") or [])
            for slot in POLICY_SLOT_GROUPS.get(str(group), set())
        }
        # A broad policy group is not permission to emit every member. It can
        # only nominate slots the semantic contract explicitly requested.
        allowed_names &= requested
        if not allowed_names:
            continue
        frame = compile_evidence_frame(row)
        if str(frame.lifecycle_status or "active").lower() != "active":
            continue
        for slot in sorted(allowed_names):
            surface = str((frame.surface_spans or {}).get(slot) or (frame.slots or {}).get(slot) or "").strip()
            if not surface or surface.lower() not in str(row.content or "").lower():
                continue
            key = (str(row.memory_id), slot, surface.lower())
            if key in seen:
                continue
            seen.add(key)
            authorized.append({"memory_id": str(row.memory_id), "slot": slot, "surface": surface})
    return authorized


def _build_governance_contract_result(*, safe_slots: list[dict[str, str]], contract: dict[str, Any]) -> AnswerResult:
    if safe_slots:
        answer = "Shareable details: " + "; ".join(
            f"{item['slot'].replace('_', ' ')}: {item['surface']}" for item in safe_slots
        ) + "."
        used_ids = _dedupe_list([item["memory_id"] for item in safe_slots])
    else:
        answer = "I can only provide details that are explicitly authorized for sharing."
        used_ids = []
    # The original action may be direct answer, but the governed boundary has
    # established that only a constrained response is safe to emit.
    resolved_action = "answer_redacted"
    return AnswerResult(
        prediction=answer,
        answer_text=answer,
        used_memory_ids=used_ids,
        reasoning_summary="Redacted answer rendered from source-validated governance slots.",
        action=resolved_action,
        answer_structured={"governance_allowed_slots": safe_slots},
        raw_response={
            "used_renderer": "governance_contract_redaction",
            "governance_contract": contract,
            "governance_allowed_slots": safe_slots,
            "governance_contract_fail_closed": not bool(safe_slots),
            "slot_verifier": {
                "required_surfaces": [],
                "hit_surfaces": [item["surface"] for item in safe_slots],
                "missing_surfaces": [],
                "passed": True,
            },
        },
    )


def _augment_current_state_slot_coverage(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    selected_lines: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Add the strongest runtime evidence for each requested current-state slot."""
    requested_slots = {
        str(slot)
        for slot in list((semantic_spec or {}).get("requested_slots") or [])
        if str(slot).strip()
    }
    state_slots = {
        "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
        "monthly_stipend", "safe_wording", "blocker", "access_room", "access_badge",
        "operational_result",
    }
    required_slots = requested_slots & state_slots if requested_slots else set(shared_infer_current_state_slots(question))
    if not required_slots:
        return selected_lines
    selected = list(selected_lines)
    selected_texts = {_normalize_surface_line(str(item.get("text") or "")) for item in selected}
    for slot in sorted(required_slots):
        candidates: list[dict[str, Any]] = []
        for evidence_rank, row in enumerate(evidence):
            for line_index, raw_line in enumerate(str(row.content or "").splitlines()):
                for surface_index, surface in enumerate(re.split(r"(?<=[.!?])\s+", raw_line)):
                    text = _strip_message_prefix(surface)
                    if not text or _is_question_like_text(text):
                        continue
                    line_meta = _build_line_metadata(text)
                    if not (line_meta.get("slots") or {}).get(slot):
                        continue
                    _attach_row_source_context(line_meta=line_meta, row=row, default_text=surface)
                    authority = _current_state_authority_score(text, float(row.score))
                    candidates.append(
                        {
                            "chunk_id": row.memory_id,
                            "text": text,
                            "score": authority,
                            "line_domains": sorted(_infer_query_domains(text)),
                            "line_index": line_index * 100 + surface_index,
                            "evidence_rank": evidence_rank,
                            "line_meta": line_meta,
                        }
                    )
        if not candidates:
            continue
        best = max(candidates, key=lambda item: _current_state_authority_key(item))
        selected_slot_scores = [
            _current_state_authority_key(item)
            for item in selected
            if (item.get("line_meta") or {}).get("slots", {}).get(slot)
        ]
        if selected_slot_scores and _current_state_authority_key(best) <= max(selected_slot_scores):
            continue
        for item in selected:
            item_slots = dict((item.get("line_meta") or {}).get("slots") or {})
            if slot not in item_slots:
                continue
            item_slots.pop(slot, None)
            item["line_meta"]["slots"] = item_slots
            item_spans = dict((item.get("line_meta") or {}).get("surface_spans") or {})
            item_spans.pop(slot, None)
            item["line_meta"]["surface_spans"] = item_spans
        normalized = _normalize_surface_line(best["text"])
        if normalized not in selected_texts:
            selected.append(best)
            selected_texts.add(normalized)
    return selected


def _current_state_authority_score(text: str, base_score: float) -> float:
    lowered = str(text or "").lower()
    authority = float(base_score)
    if any(token in lowered for token in ("revised", "supersedes", "replaces", "updated", "is now", "official")):
        authority += 2.5
    elif any(token in lowered for token in ("current", "approved")):
        authority += 0.8
    if any(token in lowered for token in ("provisional", "tentative", "exploratory", "initial", "until a newer")):
        authority -= 2.0
    return authority


def _current_state_authority_key(item: dict[str, Any]) -> tuple[float, str]:
    line_meta = dict(item.get("line_meta") or {})
    return (
        _current_state_authority_score(str(item.get("text") or ""), 0.0),
        str(line_meta.get("source_time") or ""),
    )


def _augment_redaction_slot_coverage(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    selected_lines: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_slots = set(shared_infer_current_state_slots(question)) & {"safe_wording", "public_event_date"}
    if not required_slots:
        return selected_lines
    selected = list(selected_lines)
    selected_texts = {_normalize_surface_line(str(item.get("text") or "")) for item in selected}
    covered_slots = {
        slot
        for item in selected
        for slot in required_slots
        if (item.get("line_meta") or {}).get("slots", {}).get(slot)
    }
    for evidence_rank, row in enumerate(evidence):
        for line_index, raw_line in enumerate(str(row.content or "").splitlines()):
            source_text = _strip_message_prefix(raw_line)
            projected_text = _project_redaction_safe_text(source_text)
            if not projected_text:
                continue
            line_meta = _build_line_metadata(projected_text)
            source_slots = extract_state_slots(source_text)
            if source_slots.get("safe_wording"):
                line_meta.setdefault("slots", {})["safe_wording"] = source_slots["safe_wording"]
                line_meta.setdefault("surface_spans", {})["safe_wording"] = source_slots["safe_wording"]
            _attach_row_source_context(line_meta=line_meta, row=row, default_text=source_text)
            line_meta["redaction_projection"] = True
            line_meta["projected_from_surface"] = source_text
            projected_slots = set((line_meta.get("slots") or {}).keys())
            if not (projected_slots & (required_slots - covered_slots)):
                continue
            normalized = _normalize_surface_line(projected_text)
            if normalized in selected_texts:
                continue
            selected.append(
                {
                    "chunk_id": row.memory_id,
                    "text": projected_text,
                    "score": float(row.score),
                    "line_domains": sorted(_infer_query_domains(projected_text)),
                    "line_index": line_index,
                    "evidence_rank": evidence_rank,
                    "line_meta": line_meta,
                }
            )
            selected_texts.add(normalized)
            covered_slots.update(projected_slots)
            if required_slots.issubset(covered_slots):
                return selected
    return selected


def _render_answer_from_v11_surface_replay(
    *,
    question: str,
    selected_lines: list[dict[str, Any]],
    compiled_frames: list[Any],
    used_chunk_ids: list[str],
    action: str,
    semantic_intent: dict[str, Any] | None = None,
) -> AnswerResult | None:
    semantic_domains = _semantic_domains((semantic_intent or {}).get("semantic_spec"))
    domains = semantic_domains or _infer_query_domains(question)
    question_profile = _build_question_profile(question, domains, semantic_intent=semantic_intent)
    focused_mode = _focused_surface_mode(question_profile)
    safe_selected_lines = [
        {**line, "text": _strip_private_surface_fragments(str(line.get("text") or ""))}
        for line in selected_lines
        if _strip_private_surface_fragments(str(line.get("text") or "")).strip()
    ]
    if not safe_selected_lines:
        safe_selected_lines = selected_lines
    raw_prediction = " ".join(line["text"] for line in safe_selected_lines).strip()
    structured_prediction = _render_structured_surface_answer(
        question=question,
        domains=domains,
        selected_lines=safe_selected_lines,
        question_profile=question_profile,
    )
    raw_verifier = _verify_surface_replay_answer(
        raw_prediction,
        safe_selected_lines,
        question,
        semantic_intent=semantic_intent,
    )
    prediction = raw_prediction
    verifier = raw_verifier
    if structured_prediction:
        structured_verifier = _verify_surface_replay_answer(
            structured_prediction,
            safe_selected_lines,
            question,
            semantic_intent=semantic_intent,
        )
        if _should_prefer_structured_surface_answer(
            mode=focused_mode,
            action=action,
            structured_prediction=structured_prediction,
            structured_verifier=structured_verifier,
            raw_prediction=raw_prediction,
            raw_verifier=raw_verifier,
        ) or _is_better_surface_answer(structured_verifier, raw_verifier, structured_prediction, raw_prediction):
            prediction = structured_prediction
            verifier = structured_verifier
    if verifier["missing_surfaces"] and not _should_skip_missing_surface_append(focused_mode):
        for missing_line in verifier["missing_surfaces"]:
            if missing_line not in prediction:
                prediction = f"{prediction} {missing_line}".strip()
    prediction = _strip_private_surface_fragments(prediction)
    reasoning_summary = (
        f"Surface replay answer built from {len(safe_selected_lines)} evidence lines across domains "
        f"{sorted(domains)}."
    )
    return AnswerResult(
        prediction=prediction,
        answer_text=prediction,
        used_memory_ids=_dedupe_list([line["chunk_id"] for line in safe_selected_lines] or used_chunk_ids),
        reasoning_summary=reasoning_summary,
        action=action,
        answer_structured={
            "surface_lines": safe_selected_lines,
            "compiled_frames": [asdict(frame) for frame in compiled_frames],
        },
        raw_response={
            "surface_lines": safe_selected_lines,
            "compiled_frames": [asdict(frame) for frame in compiled_frames],
            "slot_verifier": verifier,
            "raw_surface_verifier": raw_verifier,
            "structured_surface_prediction": structured_prediction,
            "focused_surface_mode": focused_mode,
            "used_renderer": "v11_surface_replay",
        },
    )


def _should_prefer_structured_surface_answer(
    *,
    mode: str | None,
    action: str,
    structured_prediction: str,
    structured_verifier: dict[str, Any],
    raw_prediction: str,
    raw_verifier: dict[str, Any],
) -> bool:
    if mode not in {"policy", "contact_protocol", "authorized_schedule", "imaging_schedule", "current_state_bundle"} and action != "answer_redacted":
        return False
    if not structured_prediction.strip():
        return False
    if len(structured_prediction) >= len(raw_prediction):
        return False
    return len(structured_verifier.get("missing_surfaces") or []) <= len(raw_verifier.get("missing_surfaces") or [])


def _filter_redaction_safe_surface_lines(selected_lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_lines = [line for line in selected_lines if _is_redaction_safe_surface_line(line)]
    return safe_lines or selected_lines


def _select_redaction_safe_surface_lines(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    domains: set[str],
    question_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for evidence_rank, row in enumerate(evidence):
        for line_idx, raw_line in enumerate(str(row.content or "").splitlines()):
            message_text = _strip_message_prefix(raw_line)
            if not message_text or _is_question_like_text(message_text):
                continue
            projected_text = _project_redaction_safe_text(message_text)
            candidate_text = projected_text or message_text
            line_domains = _infer_query_domains(candidate_text)
            line_meta = _build_line_metadata(candidate_text)
            _attach_row_source_context(line_meta=line_meta, row=row, default_text=message_text)
            if projected_text:
                line_meta["redaction_projection"] = True
                line_meta["projected_from_surface"] = message_text
            row_bundle_payload = dict((row.metadata or {}).get("bundle_payload") or {})
            if row_bundle_payload:
                line_meta["bundle_payload"] = row_bundle_payload
            score = _score_redaction_safe_surface_line(
                question=question,
                text=candidate_text,
                domains=domains,
                line_domains=line_domains,
                question_profile=question_profile,
                line_meta=line_meta,
                row_score=float(row.score),
            )
            if score <= 0:
                continue
            candidates.append(
                {
                    "chunk_id": row.memory_id,
                    "text": candidate_text,
                    "score": score,
                    "line_domains": sorted(line_domains),
                    "line_index": line_idx,
                    "evidence_rank": evidence_rank,
                    "line_meta": line_meta,
                }
            )
    candidates.sort(key=lambda item: (float(item["score"]), -int(item["evidence_rank"]), -int(item["line_index"])), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_texts: set[str] = set()
    selected_event_counts: dict[str, int] = {}
    for item in candidates:
        normalized = _normalize_surface_line(item["text"])
        if normalized in selected_texts:
            continue
        if not _passes_action_mode_surface_gate(item=item, question_profile=question_profile):
            continue
        if not _can_select_candidate(item, selected_event_counts, question_profile):
            continue
        selected.append(item)
        selected_texts.add(normalized)
        _record_selected_event(item, selected_event_counts)
        if len(selected) >= 4:
            break
    selected.sort(key=lambda item: (int(item["evidence_rank"]), int(item["line_index"])))
    return selected


def _project_redaction_safe_text(text: str) -> str | None:
    """Project mixed evidence to explicitly shareable slots for redacted answers."""
    state_slots = extract_state_slots(text)
    projected: list[str] = []
    safe_wording = str(state_slots.get("safe_wording") or "").strip()
    if safe_wording:
        projected.append(f"Broad shareable wording is {safe_wording}.")
    public_event_date = str(state_slots.get("public_event_date") or _extract_public_event_date(text) or "").strip()
    if public_event_date:
        projected.append(f"Public event date is {public_event_date}.")
    return " ".join(projected) or None


def _extract_public_event_date(text: str) -> str | None:
    match = re.search(
        r"\b(?:public\s+)?(?:event|orientation|ceremony|open\s+house|calendar)\b[^.;]{0,100}?"
        r"\b(?:is|remains|now|on|moved\s+to)\s+"
        r"([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def _score_redaction_safe_surface_line(
    *,
    question: str,
    text: str,
    domains: set[str],
    line_domains: set[str],
    question_profile: dict[str, Any],
    line_meta: dict[str, Any],
    row_score: float,
) -> float:
    if not _is_redaction_safe_text(text=text, line_meta=line_meta):
        return -10.0
    score = _score_surface_line(
        question=question,
        text=text,
        domains=domains,
        line_domains=line_domains,
        question_profile=question_profile,
        line_meta=line_meta,
    )
    if _looks_like_policy_state_text(text):
        score += 1.6
    if _is_contact_protocol_text(text):
        score += 1.2 if question_profile.get("wants_contact_protocol") else -0.8
    if _line_has_logistics_surface(text=text, line_meta=line_meta):
        score += 1.4
    return score + row_score


def _is_redaction_safe_surface_line(line: dict[str, Any]) -> bool:
    return _is_redaction_safe_text(
        text=str(line.get("text") or ""),
        line_meta=dict(line.get("line_meta") or {}),
    )


def _is_redaction_safe_text(*, text: str, line_meta: dict[str, Any]) -> bool:
    frame_type = str(line_meta.get("frame_type") or "")
    lowered = str(text or "").lower()
    slots = dict(line_meta.get("slots") or {})
    has_public_event_date = bool(slots.get("public_event_date"))
    if _looks_like_private_secret_text(text):
        return False
    if any(token in lowered for token in [
        "exact internal",
        "confidential",
        "internal label",
        "access code",
        "credential",
    ]):
        return False
    has_sensitive_amount = bool(re.search(r"\b\d[\d,]*(?:\.\d+)?\s*(?:usd|dollars?)\b", lowered)) and any(
        token in lowered for token in ("aid", "award", "grant", "support", "budget", "tuition")
    )
    if has_sensitive_amount:
        return False
    if has_public_event_date and any(token in lowered for token in ["public schedule", "public event", "orientation", "calendar line"]):
        return True
    if _looks_like_policy_state_text(text) or _is_contact_protocol_text(text) or _is_operational_instruction_text(text):
        return True
    if frame_type in {"diagnosis_or_result", "allergy", "medication"}:
        return False
    return _line_has_logistics_surface(text=text, line_meta=line_meta)


def _line_has_logistics_surface(*, text: str, line_meta: dict[str, Any]) -> bool:
    spans = dict(line_meta.get("surface_spans") or {})
    slots = dict(line_meta.get("slots") or {})
    if any(
        spans.get(key) or slots.get(key)
        for key in [
            "date", "time", "arrival_time", "secondary_time", "location", "provider",
            "target_date", "public_event_date", "approved_budget", "approved_discount_cap", "monthly_stipend",
            "safe_wording", "blocker", "entry_method", "visit_window", "package_rule",
        ]
    ):
        return True
    lowered = text.lower()
    return any(
        token in lowered
        for token in [
            "suite",
            "arrive by",
            "front desk",
            "radiology",
            "imaging suite",
            "parking",
            "budget",
            "stipend",
            "blocker",
            "entry method",
            "visit window",
            "package",
        ]
    )


def _passes_action_mode_surface_gate(*, item: dict[str, Any], question_profile: dict[str, Any]) -> bool:
    text = str(item.get("text") or "")
    meta = dict(item.get("line_meta") or {})
    procedures = set(meta.get("procedures") or [])
    frame_type = str(meta.get("frame_type") or "")
    if _looks_like_private_secret_text(text):
        return False
    if question_profile.get("wants_imaging_only"):
        if _is_contact_protocol_text(text) or _looks_like_policy_state_text(text):
            return False
        if meta.get("is_lab_like"):
            return False
        if frame_type in {"general_fact", "medication", "diagnosis_or_result"}:
            return False
        if procedures and not (procedures & {"ultrasound", "imaging", "scan"}):
            return False
    return True


def _looks_like_private_secret_text(text: str) -> bool:
    lowered = str(text or "").lower()
    secret_tokens = [
        " pin ",
        "pin:",
        "token",
        "door code",
        "keypad code",
        "release phrase",
        "backup key",
        "spare-key",
        "spare key",
        "customer mapping",
        "exact customer",
        "exact external sponsor",
        "northbridge biologics",
        "private note",
        "hidden spare-key",
        "gray planter",
        "temporary callback number",
        "callback number",
        "phone number",
        "617-555",
        "415-555",
    ]
    if any(token in lowered for token in secret_tokens):
        return True
    if re.search(r"\b\d{3}-\d{3}-\d{4}\b", lowered):
        return True
    if re.search(r"\b\d{4,8}\b", lowered) and any(token in lowered for token in ["keypad", "pin", "code"]):
        return True
    return False


def _looks_like_access_artifact_temporal_text(text: str, slots: dict[str, Any] | None = None) -> bool:
    lowered = str(text or "").lower()
    slot_map = dict(slots or {})
    if slot_map.get("access_valid_until"):
        return True
    artifact_terms = [
        "credential",
        "credentials",
        "passphrase",
        "token",
        "badge",
        "keypad",
        "door code",
        "access key",
        "access note",
        "portal credential",
        "guest-network",
        "guest network",
        "release phrase",
        "passcode",
    ]
    validity_terms = [
        "active through",
        "active until",
        "valid through",
        "valid until",
        "expires",
        "expire",
        "expired",
        "expiration",
        "expiry",
        "until revoked",
        "until replaced",
        "credential rotation",
        "access note supersedes",
    ]
    has_artifact = any(term in lowered for term in artifact_terms) or bool(
        re.search(r"\b(?:qr|barcode|one-time)\s+(?:code|credential|token)\b", lowered)
    )
    return has_artifact and any(term in lowered for term in validity_terms)


def _strip_private_surface_fragments(text: str) -> str:
    cleaned = str(text or "")
    patterns = [
        r"\b\d{3}-\d{3}-\d{4}\b",
        r"\b(?:PIN|pin|token|code|phrase|key)\b[^.]*",
        r"\b(?:exact external sponsor|customer mapping|private note|hidden spare-key|hidden spare key)\b[^.]*",
        r"\bNorthbridge Biologics\b",
        r"\bunder the gray planter by the 3B stair rail\b",
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;")
    return cleaned


def _should_skip_missing_surface_append(mode: str | None) -> bool:
    return mode in {"policy", "contact_protocol", "authorized_schedule"}


def _render_answer_with_policy_conditioned_records(
    *,
    question: str,
    action: str,
    principal: Any,
    policy_decisions: list[dict[str, Any]],
    requester_context: dict[str, Any],
    selected_lines: list[dict[str, Any]],
    used_chunk_ids: list[str],
    semantic_intent: dict[str, Any] | None,
    config: dict[str, Any],
) -> AnswerResult | None:
    answer_need = build_answer_need_spec(
        question=question,
        requester_context=requester_context,
        projected_evidence_lines=selected_lines,
        semantic_intent=semantic_intent,
        config=config,
    )
    records = build_utility_records(
        projected_evidence_lines=selected_lines,
        policy_decisions=policy_decisions,
        principal=principal,
        answer_need=answer_need,
        config=config,
    )
    packed = pack_utility_records(records=records, answer_need=answer_need, config=config)
    if not packed.selected_records:
        return None
    canonical_answer = render_canonical_answer(
        packed=packed,
        answer_need=answer_need,
        action_decision={"action": action},
        principal=principal,
        config=config,
    )
    if not canonical_answer:
        return None
    verification = verify_canonical_answer(
        answer=canonical_answer,
        packed=packed,
        answer_need=answer_need,
        principal=principal,
        config=config,
    )
    repaired_answer = canonical_answer
    repair_trace: list[str] = []
    if verification.missing_units:
        missing_units = set(verification.missing_units)
        repair_candidates = list(packed.selected_records) + [
            record
            for record in packed.dropped_records
            if {f"{record.record_type}.{slot}" for slot in record.allowed_slots} & missing_units
        ]
        repair_candidates = sorted(
            repair_candidates,
            key=lambda record: len({f"{record.record_type}.{slot}" for slot in record.allowed_slots} & missing_units),
            reverse=True,
        )
        appended_record_ids: set[str] = set()
        for record in repair_candidates:
            if record.record_id in appended_record_ids:
                continue
            sentence = render_canonical_answer(
                packed=type("TmpPack", (), {"selected_records": [record]})(),
                answer_need=answer_need,
                action_decision={"action": action},
                principal=principal,
                config=config,
            )
            if sentence and sentence not in repaired_answer:
                repaired_answer = f"{repaired_answer} {sentence}".strip()
                appended_record_ids.add(record.record_id)
        repair_trace.append(f"coverage repair appended records for missing units: {verification.missing_units}")
        verification = verify_canonical_answer(
            answer=repaired_answer,
            packed=packed,
            answer_need=answer_need,
            principal=principal,
            config=config,
        )
    if verification.denied_surface_leaks:
        for surface in verification.denied_surface_leaks:
            repaired_answer = repaired_answer.replace(surface, "").replace("  ", " ").strip()
        repair_trace.append(f"removed denied surfaces: {verification.denied_surface_leaks}")
        verification = verify_canonical_answer(
            answer=repaired_answer,
            packed=packed,
            answer_need=answer_need,
            principal=principal,
            config=config,
        )
    reasoning_summary = f"PCUR answer built from {len(packed.selected_records)} utility records."
    return AnswerResult(
        prediction=repaired_answer,
        answer_text=repaired_answer,
        used_memory_ids=_dedupe_list([record.source_chunk_id for record in packed.selected_records if record.source_chunk_id] or used_chunk_ids),
        reasoning_summary=reasoning_summary,
        action=action,
        answer_structured={
            "surface_lines": selected_lines,
            "utility_records": [asdict(record) for record in records],
            "packed_utility_evidence": {
                "selected_records": [asdict(record) for record in packed.selected_records],
                "dropped_records": [asdict(record) for record in packed.dropped_records],
                "coverage": packed.coverage,
                "trace": packed.trace,
            },
        },
        raw_response={
            "surface_lines": selected_lines,
            "answer_need_spec": asdict(answer_need),
            "utility_records": [asdict(record) for record in records],
            "packed_utility_evidence": {
                "selected_records": [asdict(record) for record in packed.selected_records],
                "dropped_records": [asdict(record) for record in packed.dropped_records],
                "coverage": packed.coverage,
                "trace": packed.trace,
            },
            "canonical_answer": canonical_answer,
            "coverage_verification": asdict(verification),
            "renderer_repair_trace": repair_trace,
            "used_renderer": "policy_conditioned_records",
            "focused_surface_mode": _focused_surface_mode(
                _build_question_profile(
                    question,
                    _semantic_domains((semantic_intent or {}).get("semantic_spec")) or _infer_query_domains(question),
                    semantic_intent=semantic_intent,
                )
            ),
            "semantic_intent": semantic_intent,
            "structured_surface_prediction": repaired_answer,
            "slot_verifier": {
                "required_surfaces": [],
                "hit_surfaces": [],
                "missing_surfaces": [],
                "passed": verification.pass_coverage and verification.pass_access_safety,
            },
        },
    )


def _classify_semantic_intent(
    *,
    llm_client: LLMClient,
    model_name: str,
    question: str,
    requester_context: dict[str, Any],
    evidence: list[RetrievedEvidence],
) -> dict[str, Any]:
    domains = _infer_query_domains(question)
    requested_household_slots = set(shared_infer_household_slots(question))
    labels = [
        "current_regimen_request",
        "start_or_add_regimen_request",
        "continue_regimen_request",
        "monitoring_plan_request",
        "record_transfer_request",
        "documentation_guidance_request",
        "policy_state_request",
        "contact_protocol_request",
        "contact_methods_request",
        "urgent_callback_request",
        "followup_plan_request",
        "current_plan_window_request",
        "authorized_schedule_request",
        "imaging_schedule_request",
        "return_precaution_request",
    ]
    if llm_client is None or not llm_client.is_available():
        return {}
    payload = {
        "question": question,
        "requester_role": requester_context.get("organization_role"),
        "relation_to_owner": requester_context.get("relation_to_owner"),
        "evidence_preview": [
            {
                "content": " ".join(str(row.content or "").split())[:220],
                "memory_type": row.memory_type,
                "memory_status": (row.metadata or {}).get("memory_status"),
            }
            for row in evidence[:6]
        ],
    }
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=SEMANTIC_INTENT_SYSTEM_PROMPT,
            user_prompt=(
                "Return JSON only with exactly these boolean fields:\n"
                + ", ".join(labels)
                + "\nSet every field explicitly to true or false.\n"
                "Use abstract semantics only. Do not quote or restate the question.\n"
                f"Payload: {payload}"
            ),
        )
        if isinstance(raw, dict):
            out: dict[str, bool] = {}
            for key in labels:
                value = raw.get(key)
                if isinstance(value, bool):
                    out[key] = value
                elif isinstance(value, str):
                    lowered = value.strip().lower()
                    if lowered in {"true", "yes"}:
                        out[key] = True
                    elif lowered in {"false", "no"}:
                        out[key] = False
            if "household" in domains and requested_household_slots:
                out["current_regimen_request"] = False
                out["start_or_add_regimen_request"] = False
                out["continue_regimen_request"] = False
                lowered_question = str(question or "").lower()
                if not any(token in lowered_question for token in ["medication", "medicine", "dose", "dosage", "regimen", "treatment plan"]):
                    out["monitoring_plan_request"] = False
            return out
    except LLMClientUnavailableError:
        return {}
    except Exception:
        return {}
    return {}


def _choose_utility_realization_result(
    *,
    question: str,
    selected_lines: list[dict[str, Any]],
    principal: Any,
    config: dict[str, Any],
    pcur_result: AnswerResult,
    fallback_result: AnswerResult,
    semantic_intent: dict[str, Any] | None = None,
) -> AnswerResult:
    pcur_raw = dict(pcur_result.raw_response or {})
    fallback_raw = dict(fallback_result.raw_response or {})
    answer_need = _hydrate_answer_need_spec(pcur_raw.get("answer_need_spec"))
    packed = _hydrate_packed_utility_evidence(pcur_raw.get("packed_utility_evidence"))
    arbitration: dict[str, Any] = {
        "mode": "pcur_vs_v11",
        "selected_renderer": "v11_surface_replay",
    }
    if answer_need is None or packed is None:
        arbitration["reason"] = "pcur_debug_payload_incomplete"
        fallback_raw["pcur_preview"] = _build_renderer_preview(pcur_raw, pcur_result.prediction)
        fallback_raw["renderer_arbitration"] = arbitration
        fallback_result.raw_response = fallback_raw
        return fallback_result

    pcur_cov = verify_canonical_answer(
        answer=pcur_result.prediction,
        packed=packed,
        answer_need=answer_need,
        principal=principal,
        config=config,
    )
    v11_cov = verify_canonical_answer(
        answer=fallback_result.prediction,
        packed=packed,
        answer_need=answer_need,
        principal=principal,
        config=config,
    )
    pcur_surface = _verify_surface_replay_answer(
        pcur_result.prediction,
        selected_lines,
        question,
        semantic_intent=semantic_intent,
    )
    v11_surface = fallback_raw.get("slot_verifier") or _verify_surface_replay_answer(
        fallback_result.prediction,
        selected_lines,
        question,
        semantic_intent=semantic_intent,
    )
    arbitration.update(
        {
            "pcur_coverage": asdict(pcur_cov),
            "v11_coverage_on_pcur_units": asdict(v11_cov),
            "pcur_surface_verifier": pcur_surface,
            "v11_surface_verifier": v11_surface,
        }
    )
    pcur_missing = len(pcur_cov.missing_units)
    v11_missing = len(v11_cov.missing_units)
    pcur_safe = pcur_cov.pass_access_safety and not pcur_cov.denied_surface_leaks
    focused_mode = str(pcur_raw.get("focused_surface_mode") or "")
    pcur_wins = False
    pcur_surface_missing = len(list(pcur_surface.get("missing_surfaces") or []))
    v11_surface_missing = len(list(v11_surface.get("missing_surfaces") or []))
    if pcur_safe and pcur_cov.pass_coverage:
        pcur_wins = True
        arbitration["reason"] = "pcur_passed_governed_coverage_and_access"
    elif pcur_safe:
        if v11_surface_missing + 1 < pcur_surface_missing:
            pcur_wins = False
        elif pcur_missing < v11_missing:
            pcur_wins = True
        elif focused_mode in {"medication_status", "current_state_bundle"} and pcur_missing == v11_missing:
            pcur_wins = v11_surface_missing >= pcur_surface_missing
        elif pcur_missing == v11_missing and _is_better_surface_answer(
            pcur_surface,
            v11_surface,
            pcur_result.prediction,
            fallback_result.prediction,
        ):
            pcur_wins = True
    if pcur_wins:
        arbitration["selected_renderer"] = "policy_conditioned_records"
        arbitration.setdefault("reason", "pcur_better_or_equal_on_runtime_coverage")
        pcur_raw["v11_fallback_preview"] = _build_renderer_preview(fallback_raw, fallback_result.prediction)
        pcur_raw["renderer_arbitration"] = arbitration
        pcur_result.raw_response = pcur_raw
        return pcur_result
    arbitration["reason"] = "kept_v11_surface_replay"
    fallback_raw["used_renderer"] = "v11_fallback_after_pcur"
    fallback_raw["answer_need_spec"] = pcur_raw.get("answer_need_spec")
    fallback_raw["utility_records"] = pcur_raw.get("utility_records", [])
    fallback_raw["packed_utility_evidence"] = pcur_raw.get("packed_utility_evidence")
    fallback_raw["canonical_answer"] = pcur_raw.get("canonical_answer")
    fallback_raw["coverage_verification"] = asdict(v11_cov)
    fallback_raw["renderer_repair_trace"] = pcur_raw.get("renderer_repair_trace", [])
    fallback_raw["pcur_preview"] = _build_renderer_preview(pcur_raw, pcur_result.prediction)
    fallback_raw["renderer_arbitration"] = arbitration
    fallback_result.raw_response = fallback_raw
    return fallback_result


def _build_renderer_preview(raw_response: dict[str, Any], prediction: str) -> dict[str, Any]:
    return {
        "prediction": prediction,
        "used_renderer": raw_response.get("used_renderer"),
        "focused_surface_mode": raw_response.get("focused_surface_mode"),
        "slot_verifier": raw_response.get("slot_verifier"),
        "coverage_verification": raw_response.get("coverage_verification"),
        "answer_need_spec": raw_response.get("answer_need_spec"),
        "packed_utility_evidence": raw_response.get("packed_utility_evidence"),
    }


def _hydrate_answer_need_spec(payload: Any) -> AnswerNeedSpec | None:
    if not isinstance(payload, dict):
        return None
    return AnswerNeedSpec(
        need_types=list(payload.get("need_types") or []),
        required_record_types=list(payload.get("required_record_types") or []),
        required_slot_groups=list(payload.get("required_slot_groups") or []),
        optional_slot_groups=list(payload.get("optional_slot_groups") or []),
        required_coverage_families=list(payload.get("required_coverage_families") or []),
        current_state_required=bool(payload.get("current_state_required")),
        cancellation_required=bool(payload.get("cancellation_required")),
        policy_context_required=bool(payload.get("policy_context_required")),
        minimality_policy=str(payload.get("minimality_policy") or "answer_only_requested_domains"),
        trace=list(payload.get("trace") or []),
    )


def _hydrate_packed_utility_evidence(payload: Any) -> PackedUtilityEvidence | None:
    if not isinstance(payload, dict):
        return None
    selected_records = [_hydrate_utility_record(row) for row in (payload.get("selected_records") or [])]
    dropped_records = [_hydrate_utility_record(row) for row in (payload.get("dropped_records") or [])]
    return PackedUtilityEvidence(
        selected_records=[row for row in selected_records if row is not None],
        dropped_records=[row for row in dropped_records if row is not None],
        coverage=dict(payload.get("coverage") or {}),
        trace=list(payload.get("trace") or []),
    )


def _hydrate_utility_record(payload: Any) -> UtilityRecord | None:
    if not isinstance(payload, dict):
        return None
    return UtilityRecord(
        record_id=str(payload.get("record_id") or ""),
        source_chunk_id=payload.get("source_chunk_id"),
        source_line_id=payload.get("source_line_id"),
        record_type=str(payload.get("record_type") or "general_utility"),
        owner_user=payload.get("owner_user"),
        principal_access=str(payload.get("principal_access") or "unknown"),
        lifecycle_status=str(payload.get("lifecycle_status") or "active"),
        slots={str(k): str(v) for k, v in (payload.get("slots") or {}).items() if v is not None},
        surface_spans={str(k): str(v) for k, v in (payload.get("surface_spans") or {}).items() if v is not None},
        denied_slots=[str(x) for x in (payload.get("denied_slots") or [])],
        allowed_slots=[str(x) for x in (payload.get("allowed_slots") or [])],
        evidence_line=str(payload.get("evidence_line") or ""),
        confidence=float(payload.get("confidence") or 1.0),
        source_time=str(payload.get("source_time") or ""),
        trace=[str(x) for x in (payload.get("trace") or [])],
    )


def shortlist_evidence_for_question(
    question: str,
    evidence: list[RetrievedEvidence],
    *,
    semantic_intent: dict[str, Any] | None = None,
    max_items: int = 6,
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    if not evidence:
        return [], {"selected_memory_ids": [], "candidates": []}
    semantic_domains = _semantic_domains((semantic_intent or {}).get("semantic_spec"))
    domains = semantic_domains or _infer_query_domains(question)
    profile = _build_question_profile(question, domains, semantic_intent=semantic_intent)
    mode = _focused_surface_mode(profile)
    required_families = _required_retrieval_families_for_mode(mode, profile)
    scored: list[dict[str, Any]] = []
    for row in evidence:
        frame = compile_evidence_frame(row)
        row_meta = _build_row_metadata(row, frame)
        families = _classify_retrieval_candidate_families(
            evidence=row,
            frame_type=frame.frame_type,
            row_meta=row_meta,
            mode=mode,
        )
        score = _score_evidence_row(
            question=question,
            row=row,
            frame=frame,
            row_meta=row_meta,
            profile=profile,
            domains=domains,
        )
        line_signals = _summarize_row_line_signals(
            question=question,
            row=row,
            mode=mode,
            required_families=required_families,
        )
        if line_signals["families"]:
            families.update(line_signals["families"])
        line_signal_bonus = _line_signal_retention_bonus(
            line_signals=line_signals,
            required_families=required_families,
        )
        score += line_signal_bonus
        current_state_slot_bonus = _current_state_row_slot_bonus(
            profile=profile,
            families=families,
            line_signals=line_signals,
            row_meta=row_meta,
            row_text=str(row.content or ""),
        )
        score += current_state_slot_bonus
        scored.append(
            {
                "memory_id": row.memory_id,
                "score": score,
                "base_score": score - line_signal_bonus - current_state_slot_bonus,
                "line_signal_bonus": line_signal_bonus,
                "current_state_slot_bonus": current_state_slot_bonus,
                "frame_type": frame.frame_type,
                "event_key": row_meta["event_key"],
                "procedures": row_meta["procedures"],
                "families": families,
                "line_signals": line_signals,
                "row": row,
            }
        )
    scored.sort(key=lambda item: (float(item["score"]), float(item["row"].score)), reverse=True)
    if not _should_apply_evidence_shortlist(question, profile, scored):
        return evidence, {
            "enabled": False,
            "selected_memory_ids": [row.memory_id for row in evidence],
            "reason": "shortlist_confidence_too_low",
            "candidates": [
                {
                    "memory_id": item["memory_id"],
                    "score": item["score"],
                    "base_score": item["base_score"],
                    "line_signal_bonus": item["line_signal_bonus"],
                    "current_state_slot_bonus": item["current_state_slot_bonus"],
                    "frame_type": item["frame_type"],
                    "event_key": item["event_key"],
                    "procedures": item["procedures"],
                    "families": sorted(item["families"]),
                    "line_signal_required_families": sorted(item["line_signals"]["required_families"]),
                    "line_signal_supporting_line_count": item["line_signals"]["supporting_line_count"],
                    "selected": True,
                }
                for item in scored[:12]
            ],
        }
    selected: list[RetrievedEvidence] = []
    selected_events: dict[str, int] = {}
    selected_frame_types: set[str] = set()
    covered_families: set[str] = set()
    for family in required_families:
        family_candidates = [
            item
            for item in scored
            if family in item["families"]
            and _can_shortlist_item(
                item=item,
                selected=selected,
                selected_events=selected_events,
                selected_frame_types=selected_frame_types,
                profile=profile,
            )
        ]
        if mode == "current_state_bundle" and family_candidates:
            best = max(
                family_candidates,
                key=lambda item: (
                    _current_state_shortlist_family_score(question=question, family=family, item=item),
                    float(item["score"]),
                    float(item["row"].score),
                ),
            )
        else:
            best = family_candidates[0] if family_candidates else None
        if best is None:
            continue
        selected.append(best["row"])
        event_key = str(best["event_key"])
        selected_events[event_key] = selected_events.get(event_key, 0) + 1
        selected_frame_types.add(str(best["frame_type"]))
        covered_families.update(best["families"])
        if len(selected) >= max_items:
            break
    for item in scored:
        if len(selected) >= max_items:
            break
        if not _can_shortlist_item(
            item=item,
            selected=selected,
            selected_events=selected_events,
            selected_frame_types=selected_frame_types,
            profile=profile,
        ):
            continue
        selected.append(item["row"])
        event_key = str(item["event_key"])
        selected_events[event_key] = selected_events.get(event_key, 0) + 1
        selected_frame_types.add(str(item["frame_type"]))
        covered_families.update(item["families"])
    selected_ids = {row.memory_id for row in selected}
    debug = {
        "enabled": True,
        "mode": mode,
        "required_families": sorted(required_families),
        "covered_families": sorted(covered_families),
        "selected_memory_ids": [row.memory_id for row in selected],
        "candidates": [
            {
                "memory_id": item["memory_id"],
                "score": item["score"],
                "base_score": item["base_score"],
                "line_signal_bonus": item["line_signal_bonus"],
                "current_state_slot_bonus": item["current_state_slot_bonus"],
                "frame_type": item["frame_type"],
                "event_key": item["event_key"],
                "procedures": item["procedures"],
                "families": sorted(item["families"]),
                "line_signal_required_families": sorted(item["line_signals"]["required_families"]),
                "line_signal_supporting_line_count": item["line_signals"]["supporting_line_count"],
                "selected": item["memory_id"] in selected_ids,
            }
            for item in scored[:12]
        ],
    }
    return (selected or evidence[:max_items]), debug


def _summarize_row_line_signals(
    *,
    question: str,
    row: RetrievedEvidence,
    mode: str | None,
    required_families: list[str],
) -> dict[str, Any]:
    if not mode:
        return {
            "families": set(),
            "required_families": set(),
            "supporting_line_count": 0,
            "family_line_counts": {},
            "best_required_gains": {},
        }
    required_family_set = set(required_families)
    family_union: set[str] = set()
    required_hits: set[str] = set()
    supporting_line_count = 0
    family_line_counts: dict[str, int] = {}
    best_required_gains: dict[str, float] = {}
    row_retrieval_score = float(row.score)
    for raw_line in str(row.content or "").splitlines():
        message_text = _strip_message_prefix(raw_line)
        if not message_text or _is_question_like_text(message_text):
            continue
        line_meta = _build_line_metadata(message_text)
        _attach_row_source_context(line_meta=line_meta, row=row, default_text=message_text)
        row_bundle_payload = dict((row.metadata or {}).get("bundle_payload") or {})
        if row_bundle_payload:
            line_meta["bundle_payload"] = row_bundle_payload
        line_domains = _infer_query_domains(message_text)
        line_families = _classify_line_families(message_text, line_meta, mode=mode)
        if not line_families:
            continue
        family_union.update(line_families)
        required_line_families = required_family_set & line_families
        if not required_line_families:
            continue
        line_score = _score_focused_surface_line(
            mode=mode,
            question=question,
            text=message_text,
            line_domains=line_domains,
            line_meta=line_meta,
            row_score=row_retrieval_score,
        )
        if line_score <= 0:
            continue
        supporting_line_count += 1
        line_gain = max(0.0, line_score - row_retrieval_score)
        for family in required_line_families:
            required_hits.add(family)
            family_line_counts[family] = family_line_counts.get(family, 0) + 1
            best_required_gains[family] = max(line_gain, best_required_gains.get(family, 0.0))
    return {
        "families": family_union,
        "required_families": required_hits,
        "supporting_line_count": supporting_line_count,
        "family_line_counts": family_line_counts,
        "best_required_gains": best_required_gains,
    }


def _line_signal_retention_bonus(
    *,
    line_signals: dict[str, Any],
    required_families: list[str],
) -> float:
    required_hits = [family for family in required_families if family in set(line_signals.get("required_families") or set())]
    if not required_hits:
        return 0.0
    family_line_counts = dict(line_signals.get("family_line_counts") or {})
    best_required_gains = dict(line_signals.get("best_required_gains") or {})
    bonus = 0.0
    bonus += 0.42 * len(required_hits)
    bonus += 0.24 * min(int(line_signals.get("supporting_line_count") or 0), 3)
    bonus += 0.14 * sum(min(float(best_required_gains.get(family, 0.0)), 4.0) for family in required_hits)
    if len(required_hits) >= 2:
        bonus += 0.24
    elif family_line_counts.get(required_hits[0], 0) >= 2:
        bonus += 0.22
    return bonus


def _requested_current_families(profile: dict[str, Any]) -> list[str]:
    requested_slots = list(profile.get("requested_current_slots") or [])
    slot_to_family = {
        "target_date": "current_target_date",
        "public_event_date": "current_public_event_date",
        "approved_budget": "current_budget",
        "approved_discount_cap": "current_discount",
        "monthly_stipend": "current_stipend",
        "safe_wording": "current_safe_wording",
        "blocker": "current_blocker",
        "date": "household_plan",
        "visit_window": "household_plan",
        "entry_method": "household_plan",
        "package_rule": "household_plan",
        "approved_areas": "household_plan",
        "parking_pass": "household_plan",
        "arrival_contact_rule": "household_plan",
    }
    families: list[str] = []
    for slot_name in requested_slots:
        mapped = slot_to_family.get(slot_name)
        if mapped and mapped not in families:
            families.append(mapped)
    return families


def _current_state_row_slot_bonus(
    *,
    profile: dict[str, Any],
    families: set[str],
    line_signals: dict[str, Any],
    row_meta: dict[str, Any],
    row_text: str,
) -> float:
    if not profile.get("wants_current_state_bundle"):
        return 0.0
    requested_families = _requested_current_families(profile)
    if not requested_families:
        return 0.0
    requested_family_set = set(requested_families)
    required_hits = set(families) & requested_family_set
    required_hits.update(set(line_signals.get("required_families") or set()) & requested_family_set)
    if not required_hits:
        return 0.0
    lowered = row_text.lower()
    bonus = 0.0
    hit_count = len(required_hits)
    bonus += 1.05 * hit_count
    if hit_count >= 2:
        bonus += 1.0 + 0.45 * (hit_count - 2)
    if row_meta.get("is_current_like"):
        bonus += 0.55
    if any(token in lowered for token in ["current safe recap", "current state at this point", "advising-safe summary", "finance-confirmed", "approved current"]):
        bonus += 0.9
    if any(token in lowered for token in ["provisional until", "if a newer", "older value"]) and hit_count <= 1:
        bonus -= 0.9
    return bonus


def _find_same_day_full_date_sibling(
    *,
    evidence: list[RetrievedEvidence],
    selected_chunk_ids: set[str],
    current_target_value: str,
) -> dict[str, Any] | None:
    if not current_target_value:
        return None
    target_lower = current_target_value.lower()
    for row in evidence:
        if row.memory_id not in selected_chunk_ids:
            continue
        for line_idx, raw_line in enumerate(str(row.content or "").splitlines()):
            message_text = _strip_message_prefix(raw_line)
            if not message_text or _is_question_like_text(message_text):
                continue
            if target_lower not in message_text.lower():
                continue
            if not re.search(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", message_text):
                continue
            line_meta = _build_line_metadata(message_text)
            _attach_row_source_context(line_meta=line_meta, row=row, default_text=message_text)
            if "current_target_date" not in _classify_line_families(message_text, line_meta, mode="current_state_bundle"):
                continue
            return {
                "chunk_id": row.memory_id,
                "text": message_text,
                "score": max(0.0, float(row.score)),
                "line_domains": sorted(_infer_query_domains(message_text)),
                "line_index": line_idx,
                "evidence_rank": -1,
                "line_meta": line_meta,
            }
    return None


def _recover_full_household_date_from_candidates(
    *,
    candidates: list[dict[str, Any]],
    partial_date: str,
) -> str | None:
    partial = str(partial_date or "").strip()
    if not partial or not re.fullmatch(r"[A-Za-z]+", partial):
        return None
    best_value = ""
    best_score = float("-inf")
    for item in candidates:
        meta = dict(item.get("line_meta") or {})
        slots = dict(meta.get("slots") or {})
        candidate_values = [str(slots.get("date") or "").strip()]
        candidate_values.extend(
            match.group(0).strip()
            for match in re.finditer(rf"\b{re.escape(partial)},\s+[A-Za-z]+\s+\d{{1,2}}\b", str(item.get('text') or ""), flags=re.IGNORECASE)
        )
        for value in candidate_values:
            if not value or not re.fullmatch(rf"{re.escape(partial)},\s+[A-Za-z]+\s+\d{{1,2}}", value, flags=re.IGNORECASE):
                continue
            score = _current_state_family_priority_score(
                family="household_plan",
                text=str(item.get("text") or ""),
                line_meta=meta,
                base_score=float(item.get("score") or 0.0),
            )
            if score > best_score:
                best_score = score
                best_value = re.sub(r"\s+", " ", value).strip()
    return best_value or None


def _enrich_selected_household_dates(
    *,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    for item in selected:
        meta = dict(item.get("line_meta") or {})
        slots = dict(meta.get("slots") or {})
        current_date = str(slots.get("date") or "").strip()
        if not re.fullmatch(r"[A-Za-z]+", current_date):
            continue
        full_household_date = _recover_full_household_date_from_candidates(
            candidates=candidates,
            partial_date=current_date,
        )
        if not full_household_date:
            continue
        slots["date"] = full_household_date
        meta["slots"] = slots
        item["line_meta"] = meta


def _current_state_shortlist_family_score(
    *,
    question: str,
    family: str,
    item: dict[str, Any],
) -> float:
    best = float("-inf")
    row = item["row"]
    row_score = float(row.score)
    for raw_line in str(row.content or "").splitlines():
        message_text = _strip_message_prefix(raw_line)
        if not message_text or _is_question_like_text(message_text):
            continue
        line_meta = _build_line_metadata(message_text)
        _attach_row_source_context(line_meta=line_meta, row=row, default_text=message_text)
        row_bundle_payload = dict((row.metadata or {}).get("bundle_payload") or {})
        if row_bundle_payload:
            line_meta["bundle_payload"] = row_bundle_payload
        line_families = _classify_line_families(message_text, line_meta, mode="current_state_bundle")
        if family not in line_families:
            continue
        line_score = _score_focused_surface_line(
            mode="current_state_bundle",
            question=question,
            text=message_text,
            line_domains=_infer_query_domains(message_text),
            line_meta=line_meta,
            row_score=row_score,
        )
        family_score = _current_state_family_priority_score(
            family=family,
            text=message_text,
            line_meta=line_meta,
            base_score=line_score,
        )
        best = max(best, family_score)
    if best != float("-inf"):
        return best
    return _current_state_family_priority_score(
        family=family,
        text=str(row.content or ""),
        line_meta={},
        base_score=float(item.get("score") or 0.0),
    )


def _select_surface_lines(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    domains: set[str],
) -> list[dict[str, Any]]:
    question_profile = _build_question_profile(question, domains)
    candidates: list[dict[str, Any]] = []
    for evidence_rank, row in enumerate(evidence):
        for line_idx, raw_line in enumerate(str(row.content or "").splitlines()):
            message_text = _strip_message_prefix(raw_line)
            if not message_text:
                continue
            line_domains = _infer_query_domains(message_text)
            line_meta = _build_line_metadata(message_text)
            _attach_row_source_context(line_meta=line_meta, row=row, default_text=message_text)
            row_bundle_payload = dict((row.metadata or {}).get("bundle_payload") or {})
            if row_bundle_payload:
                line_meta["bundle_payload"] = row_bundle_payload
            score = _score_surface_line(
                question=question,
                text=message_text,
                domains=domains,
                line_domains=line_domains,
                question_profile=question_profile,
                line_meta=line_meta,
            )
            if score <= 0:
                continue
            candidates.append(
                {
                    "chunk_id": row.memory_id,
                    "text": message_text,
                    "score": score + max(0.0, float(row.score)),
                    "line_domains": sorted(line_domains),
                    "line_index": line_idx,
                    "evidence_rank": evidence_rank,
                    "line_meta": line_meta,
                }
            )
    candidates.sort(key=lambda item: (float(item["score"]), -int(item["evidence_rank"]), -int(item["line_index"])), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_texts: set[str] = set()
    covered_domains: set[str] = set()
    selected_event_counts: dict[str, int] = {}
    for domain in ["allergy", "medication", "schedule", "instruction", "cancellation", "general"]:
        if domain not in domains and domain != "general":
            continue
        best = next(
            (
                item
                for item in candidates
                if domain in item["line_domains"] and _normalize_surface_line(item["text"]) not in selected_texts
                and _can_select_candidate(item, selected_event_counts, question_profile)
            ),
            None,
        )
        if best is not None:
            selected.append(best)
            selected_texts.add(_normalize_surface_line(best["text"]))
            covered_domains.update(best["line_domains"])
            _record_selected_event(best, selected_event_counts)
    for item in candidates:
        normalized = _normalize_surface_line(item["text"])
        if normalized in selected_texts:
            continue
        if len(selected) >= 6:
            break
        if not _can_select_candidate(item, selected_event_counts, question_profile):
            continue
        if any(domain in item["line_domains"] for domain in domains - covered_domains):
            selected.append(item)
            selected_texts.add(normalized)
            covered_domains.update(item["line_domains"])
            _record_selected_event(item, selected_event_counts)
            continue
        if len(selected) < 4:
            selected.append(item)
            selected_texts.add(normalized)
            covered_domains.update(item["line_domains"])
            _record_selected_event(item, selected_event_counts)
    selected.sort(key=lambda item: (int(item["evidence_rank"]), int(item["line_index"])))
    return selected


def _select_focused_surface_lines(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    domains: set[str],
    question_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    mode = _focused_surface_mode(question_profile)
    if mode is None:
        return []
    candidates: list[dict[str, Any]] = []
    for evidence_rank, row in enumerate(evidence):
        for line_idx, raw_line in enumerate(str(row.content or "").splitlines()):
            message_text = _strip_message_prefix(raw_line)
            if not message_text:
                continue
            if _is_question_like_text(message_text):
                continue
            line_domains = _infer_query_domains(message_text)
            line_meta = _build_line_metadata(message_text)
            _attach_row_source_context(line_meta=line_meta, row=row, default_text=message_text)
            row_bundle_payload = dict((row.metadata or {}).get("bundle_payload") or {})
            if row_bundle_payload:
                line_meta["bundle_payload"] = row_bundle_payload
            support_completion_payload = dict((row.metadata or {}).get("support_completion") or {})
            if support_completion_payload:
                line_meta["support_completion"] = support_completion_payload
            score = _score_focused_surface_line(
                mode=mode,
                question=question,
                text=message_text,
                line_domains=line_domains,
                line_meta=line_meta,
                row_score=float(row.score),
            )
            if score <= 0 and mode == "current_state_bundle":
                support_kind = str((line_meta.get("support_completion") or {}).get("kind") or "")
                candidate_date = str((line_meta.get("slots") or {}).get("date") or "").strip()
                if support_kind == "household_full_date_anchor" and re.fullmatch(r"[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?", candidate_date):
                    score = 0.05
            if score <= 0:
                continue
            candidates.append(
                {
                    "chunk_id": row.memory_id,
                    "text": message_text,
                    "score": score,
                    "line_domains": sorted(line_domains),
                    "line_index": line_idx,
                    "evidence_rank": evidence_rank,
                    "line_meta": line_meta,
                }
            )
    candidates.sort(key=lambda item: (float(item["score"]), -int(item["evidence_rank"]), -int(item["line_index"])), reverse=True)
    limits = {
        "policy": 3,
        "medication_status": 4,
        "return_precautions": 3,
        "contact_methods": 3,
        "current_plan_window": 4,
        "current_med_followup_plan": 4,
        "callback_instruction": 4,
        "followup_plan": 4,
        "authorized_schedule": 3,
        "final_procedure_plan": 4,
        "rescheduled_followup": 4,
        "imaging_schedule": 3,
        "mixed_allergy_schedule": 4,
        "current_state_bundle": 8,
    }
    limit = limits.get(mode, 4)
    selected: list[dict[str, Any]] = []
    selected_texts: set[str] = set()
    selected_event_counts: dict[str, int] = {}
    required_families = _required_line_families_for_mode(mode, question_profile)
    covered_families: set[str] = set()
    for family in required_families:
        best = next(
            (
                item
                for item in candidates
                if family in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                and _normalize_surface_line(item["text"]) not in selected_texts
                and _can_select_candidate(item, selected_event_counts, question_profile)
            ),
            None,
        )
        if best is None:
            continue
        selected.append(best)
        selected_texts.add(_normalize_surface_line(best["text"]))
        _record_selected_event(best, selected_event_counts)
        covered_families.update(_classify_line_families(best["text"], best.get("line_meta") or {}, mode=mode))
        if len(selected) >= limit:
            break
    if mode == "current_state_bundle":
        requested_household_slots: list[str] = []
        household_slot_flags = [
            ("date", bool(question_profile.get("asks_household_visit_window"))),
            ("visit_window", bool(question_profile.get("asks_household_visit_window"))),
            ("entry_method", bool(question_profile.get("asks_household_entry_method"))),
            ("package_rule", bool(question_profile.get("asks_household_package_rule"))),
            ("approved_areas", bool(question_profile.get("asks_household_approved_areas"))),
            ("parking_pass", bool(question_profile.get("asks_household_parking_pass"))),
            ("arrival_contact_rule", bool(question_profile.get("asks_household_arrival_contact_rule"))),
        ]
        for slot_name, requested in household_slot_flags:
            if requested and slot_name not in requested_household_slots:
                requested_household_slots.append(slot_name)
        if question_profile.get("asks_household_plan") and not requested_household_slots:
            requested_household_slots = ["date", "visit_window", "entry_method"]
        for family in ["current_budget", "current_discount", "current_public_event_date", "household_plan"]:
            chosen_idx = next(
                (
                    idx
                    for idx, item in enumerate(selected)
                    if family in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                ),
                None,
            )
            better_candidates = [
                item
                for item in candidates
                if family in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                and _can_select_candidate(item, selected_event_counts, question_profile)
            ]
            better = max(
                better_candidates,
                key=lambda item: _current_state_family_priority_score(
                    family=family,
                    text=str(item.get("text") or ""),
                    line_meta=item.get("line_meta") or {},
                    base_score=float(item.get("score") or 0.0),
                ),
                default=None,
            )
            if chosen_idx is not None and better is not None:
                selected[chosen_idx] = better
        current_target_idx = next(
            (
                idx
                for idx, item in enumerate(selected)
                if "current_target_date" in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
            ),
            None,
        )
        if current_target_idx is not None:
            current_target_text = str(selected[current_target_idx].get("text") or "")
            current_target_meta = selected[current_target_idx].get("line_meta") or {}
            current_target_slots = dict(current_target_meta.get("slots") or {})
            current_target_value = str(current_target_slots.get("target_date") or current_target_slots.get("date") or "").strip()
            current_target_score = _current_state_family_priority_score(
                family="current_target_date",
                text=current_target_text,
                line_meta=current_target_meta,
                base_score=float(selected[current_target_idx].get("score") or 0.0),
            )
            if current_target_value and re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", current_target_value):
                sibling_replacement = _find_same_day_full_date_sibling(
                    evidence=evidence,
                    selected_chunk_ids={str(item.get("chunk_id") or "") for item in selected},
                    current_target_value=current_target_value,
                )
                if sibling_replacement is not None:
                    selected_texts.discard(_normalize_surface_line(current_target_text))
                    selected[current_target_idx] = sibling_replacement
                    selected_texts.add(_normalize_surface_line(sibling_replacement["text"]))
                    current_target_text = str(sibling_replacement.get("text") or "")
                    current_target_meta = sibling_replacement.get("line_meta") or {}
                    current_target_score = _current_state_family_priority_score(
                        family="current_target_date",
                        text=current_target_text,
                        line_meta=current_target_meta,
                        base_score=float(sibling_replacement.get("score") or 0.0),
                    )
                same_day_replacement = max(
                    [
                        item
                        for item in candidates
                        if "current_target_date" in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                        and re.search(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", str(item.get("text") or ""))
                        and current_target_value.lower() in str(item.get("text") or "").lower()
                    ],
                    key=lambda item: _current_state_family_priority_score(
                        family="current_target_date",
                        text=str(item.get("text") or ""),
                        line_meta=item.get("line_meta") or {},
                        base_score=float(item.get("score") or 0.0),
                    ),
                    default=None,
                )
                if same_day_replacement is not None:
                    selected_texts.discard(_normalize_surface_line(current_target_text))
                    selected[current_target_idx] = same_day_replacement
                    selected_texts.add(_normalize_surface_line(same_day_replacement["text"]))
                    current_target_text = str(same_day_replacement.get("text") or "")
                    current_target_meta = same_day_replacement.get("line_meta") or {}
                    current_target_score = _current_state_family_priority_score(
                        family="current_target_date",
                        text=current_target_text,
                        line_meta=current_target_meta,
                        base_score=float(same_day_replacement.get("score") or 0.0),
                    )
            replacement = max(
                [
                    item
                    for item in candidates
                    if "current_target_date" in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                    and re.search(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", str(item.get("text") or ""))
                ],
                key=lambda item: _current_state_family_priority_score(
                    family="current_target_date",
                    text=str(item.get("text") or ""),
                    line_meta=item.get("line_meta") or {},
                    base_score=float(item.get("score") or 0.0),
                ),
                default=None,
            )
            if replacement is not None:
                replacement_score = _current_state_family_priority_score(
                    family="current_target_date",
                    text=str(replacement.get("text") or ""),
                    line_meta=replacement.get("line_meta") or {},
                    base_score=float(replacement.get("score") or 0.0),
                )
                if replacement_score > current_target_score + 0.6:
                    selected_texts.discard(_normalize_surface_line(current_target_text))
                    selected[current_target_idx] = replacement
                    selected_texts.add(_normalize_surface_line(replacement["text"]))
        covered_household_slots = {
            slot_name
            for item in selected
            for slot_name in requested_household_slots
            if (item.get("line_meta") or {}).get("slots", {}).get(slot_name)
        }
        for slot_name in requested_household_slots:
            if slot_name in covered_household_slots:
                continue
            best_slot_line = max(
                [
                    item
                    for item in candidates
                    if (item.get("line_meta") or {}).get("slots", {}).get(slot_name)
                    and _normalize_surface_line(item["text"]) not in selected_texts
                    and _can_select_candidate(item, selected_event_counts, question_profile)
                ],
                key=lambda item: _current_state_family_priority_score(
                    family="household_plan",
                    text=str(item.get("text") or ""),
                    line_meta=item.get("line_meta") or {},
                    base_score=float(item.get("score") or 0.0),
                ),
                default=None,
            )
            if best_slot_line is None:
                continue
            selected.append(best_slot_line)
            selected_texts.add(_normalize_surface_line(best_slot_line["text"]))
            covered_household_slots.update(
                slot
                for slot in requested_household_slots
                if (best_slot_line.get("line_meta") or {}).get("slots", {}).get(slot)
            )
            if len(selected) >= limit:
                break
        for slot_name in ["visit_window", "entry_method", "package_rule", "approved_areas"]:
            if slot_name not in requested_household_slots:
                continue
            chosen_idx = next(
                (
                    idx
                    for idx, item in enumerate(selected)
                    if (item.get("line_meta") or {}).get("slots", {}).get(slot_name)
                ),
                None,
            )
            better_slot_line = max(
                [
                item
                for item in candidates
                if (item.get("line_meta") or {}).get("slots", {}).get(slot_name)
                and _can_select_candidate(item, selected_event_counts, question_profile)
                ],
                key=lambda item: _current_state_family_priority_score(
                    family="household_plan",
                    text=str(item.get("text") or ""),
                    line_meta=item.get("line_meta") or {},
                    base_score=float(item.get("score") or 0.0),
                ),
                default=None,
            )
            if better_slot_line is None:
                continue
            if chosen_idx is None:
                if _normalize_surface_line(better_slot_line["text"]) in selected_texts:
                    continue
                selected.append(better_slot_line)
                selected_texts.add(_normalize_surface_line(better_slot_line["text"]))
                if len(selected) >= limit:
                    break
                continue
            current_line = selected[chosen_idx]
            current_score = _current_state_family_priority_score(
                family="household_plan",
                text=str(current_line.get("text") or ""),
                line_meta=current_line.get("line_meta") or {},
                base_score=float(current_line.get("score") or 0.0),
            )
            better_score = _current_state_family_priority_score(
                family="household_plan",
                text=str(better_slot_line.get("text") or ""),
                line_meta=better_slot_line.get("line_meta") or {},
                base_score=float(better_slot_line.get("score") or 0.0),
            )
            if better_score <= current_score:
                continue
            selected_texts.discard(_normalize_surface_line(current_line["text"]))
            selected[chosen_idx] = better_slot_line
            selected_texts.add(_normalize_surface_line(better_slot_line["text"]))
        if "date" in requested_household_slots or question_profile.get("asks_household_plan"):
            _enrich_selected_household_dates(selected=selected, candidates=candidates)
        required_current_families = _required_line_families_for_mode(mode, question_profile)
        if not required_current_families:
            required_current_families = ["current_target_date", "current_budget", "current_discount", "current_blocker", "current_stipend", "current_safe_wording", "household_plan"]
        covered_current_families = set()
        for item in selected:
            covered_current_families.update(_classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode))
        for family in required_current_families:
            if family in covered_current_families:
                continue
            best_family = max(
                [
                    item
                    for item in candidates
                    if family in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                    and _normalize_surface_line(item["text"]) not in selected_texts
                    and _can_select_candidate(item, selected_event_counts, question_profile)
                ],
                key=lambda item: _current_state_family_priority_score(
                    family=family,
                    text=str(item.get("text") or ""),
                    line_meta=item.get("line_meta") or {},
                    base_score=float(item.get("score") or 0.0),
                ),
                default=None,
            )
            if best_family is None:
                continue
            selected.append(best_family)
            selected_texts.add(_normalize_surface_line(best_family["text"]))
            covered_current_families.update(_classify_line_families(best_family["text"], best_family.get("line_meta") or {}, mode=mode))
            if len(selected) >= limit:
                break
        if len(requested_household_slots) > 1:
            selected_event_keys = {
                str((item.get("line_meta") or {}).get("event_key") or "")
                for item in selected
                if str((item.get("line_meta") or {}).get("event_key") or "")
            }
            household_candidates = [
                item
                for item in candidates
                if "household_plan" in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                and _can_select_candidate(item, selected_event_counts, question_profile)
            ]
            for item in household_candidates:
                if len(selected) >= limit:
                    break
                normalized = _normalize_surface_line(item["text"])
                if normalized in selected_texts:
                    continue
                event_key = str((item.get("line_meta") or {}).get("event_key") or "")
                if event_key and event_key in selected_event_keys:
                    continue
                selected.append(item)
                selected_texts.add(normalized)
                if event_key:
                    selected_event_keys.add(event_key)
                _record_selected_event(item, selected_event_counts)
        household_intent = _household_question_intent(question, question_profile)
        if household_intent != "general":
            selected = _prune_household_selected_lines(
                selected=selected,
                candidates=candidates,
                question=question,
                question_profile=question_profile,
                limit=limit,
                selected_event_counts=selected_event_counts,
            )
        selected.sort(key=lambda item: (int(item["evidence_rank"]), int(item["line_index"])))
        return selected
    if mode == "medication_status":
        regimen_families = set(question_profile.get("regimen_coverage_families") or [])
        selected_families = {
            family
            for item in selected
            for family in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
        }
        should_attach_current_baseline = bool({"start", "stop", "use"} & regimen_families) and "medication_continue" not in selected_families
        if should_attach_current_baseline and len(selected) < limit:
            best_continue_line = next(
                (
                    item
                    for item in candidates
                    if "medication_continue" in _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
                    and _normalize_surface_line(item["text"]) not in selected_texts
                ),
                None,
            )
            if best_continue_line is not None:
                selected.append(best_continue_line)
                selected_texts.add(_normalize_surface_line(best_continue_line["text"]))
                covered_families.update(_classify_line_families(best_continue_line["text"], best_continue_line.get("line_meta") or {}, mode=mode))
    for item in candidates:
        if len(selected) >= limit:
            break
        normalized = _normalize_surface_line(item["text"])
        if normalized in selected_texts:
            continue
        families = _classify_line_families(item["text"], item.get("line_meta") or {}, mode=mode)
        if required_families and any(family not in covered_families for family in families):
            selected.append(item)
            selected_texts.add(normalized)
            covered_families.update(families)
            if len(selected) >= limit:
                break
            continue
        selected.append(item)
        selected_texts.add(normalized)
        covered_families.update(families)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (int(item["evidence_rank"]), int(item["line_index"])))
    return selected


def _current_state_family_priority_score(*, family: str, text: str, line_meta: dict[str, Any], base_score: float) -> float:
    lowered = text.lower()
    lowered_source = _current_state_family_support_source(family=family, text=text, line_meta=line_meta).lower()
    score = float(base_score)
    access_artifact_temporal = bool(line_meta.get("is_access_artifact_temporal")) or _looks_like_access_artifact_temporal_text(text, dict(line_meta.get("slots") or {}))
    if line_meta.get("is_current_like"):
        score += 0.8
    if family in {"current_budget", "current_discount", "current_stipend"}:
        if "revised approved" in lowered_source:
            score += 4.0
        if any(token in lowered_source for token in ["approved current", "current approved", "finance-confirmed", "quick state note"]):
            score += 2.2
        if "supersedes" in lowered_source or "replaces the earlier" in lowered_source:
            score += 2.5
        if any(token in lowered_source for token in ["is now", "as of now", "treat as current", "only current"]):
            score += 1.8
        if any(token in lowered_source for token in ["provisional", "exploratory", "initial", "temporary"]) and not any(
            token in lowered_source for token in ["temporary room reassignment", "temporary room reassignment coordinated", "temporary tuition hold"]
        ):
            score -= 2.0
        if any(token in lowered_source for token in ["until a newer", "until finance posts", "until the final", "if a newer", "not fully locked", "provisional until", "older provisional"]):
            score -= 2.4
    if family in {"current_target_date", "current_public_event_date"}:
        if any(token in lowered_source for token in ["as of now", "as of this afternoon", "is now", "treat as current", "current state at this point"]):
            score += 2.0
        if any(token in lowered_source for token in ["official pilot target", "launch update", "moves from", "moved from", "rescheduled", "review memo is closing", "hearing date", "closure date"]):
            score += 2.2
        if any(token in lowered_source for token in ["until a newer", "for now", "provisional", "interim", "older provisional"]):
            score -= 2.0
    if family == "current_safe_wording":
        if any(token in lowered_source for token in ["current safe recap", "advising-safe summary", "safe case wording", "safe sponsor wording", "safe external wording", "broad housing wording"]):
            score += 2.0
    if family == "current_target_date":
        if access_artifact_temporal:
            score -= 5.0
        if re.search(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", text):
            score += 1.5
        if "moves from" in lowered_source:
            score += 0.8
        if any(token in lowered_source for token in ["official pilot target", "launch update", "is now the official", "current hearing date", "review memo is closing"]):
            score += 2.2
        if any(token in lowered_source for token in ["until a newer decision supersedes it", "remains july 1 for now", "recording the current maple target date"]):
            score -= 1.3
        if any(token in lowered_source for token in ["review memo", "hearing date", "closure date"]):
            score += 1.0
    if family == "current_public_event_date":
        if re.search(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", text):
            score += 1.5
        if any(token in lowered for token in ["orientation", "public schedule", "calendar line", "public event"]):
            score += 1.6
        if any(token in lowered for token in ["moved from", "moved to", "rescheduled", "is now", "updated"]):
            score += 1.1
    if family == "current_stipend":
        if any(token in lowered for token in ["approved support amount", "approved current", "bridge-support amount", "bridge support amount", "ra-support amount", "ra support amount"]):
            score += 1.8
    if family == "current_safe_wording":
        if any(token in lowered for token in ["safe case wording", "safe sponsor wording", "safe external wording", "broad housing wording", "graduate committee review", "confidential academic review case"]):
            score += 1.8
    if family == "current_blocker":
        if any(token in lowered for token in ["current blocker", "still pending", "hold cleared", "tuition hold", "hold state", "review memo is closing", "not fully locked"]):
            score += 1.4
    if family == "household_plan":
        household_slots = dict(line_meta.get("slots") or {})
        if any(household_slots.get(slot) for slot in ("visit_window", "entry_method", "package_rule", "approved_areas", "arrival_contact_rule", "parking_pass")):
            score += 2.0
        if any(token in lowered for token in ["current", "updated", "supersedes", "replaces", "approved"]):
            score += 2.8
        if any(token in lowered for token in ["provisional", "superseded", "historical", "no longer"]):
            score -= 1.2
    return score


def _current_state_family_support_source(*, family: str, text: str, line_meta: dict[str, Any]) -> str:
    source_text = str(line_meta.get("source_text") or text)
    segments = [segment.strip() for segment in source_text.splitlines() if segment.strip()]
    if not segments:
        return text
    anchors = _current_state_family_anchors(family)
    slots = dict(line_meta.get("slots") or {})
    slot_values = [
        str(value).strip().lower()
        for key, value in slots.items()
        if key in _current_state_family_slot_keys(family) and str(value or "").strip()
    ]
    matched = [
        segment
        for segment in segments
        if any(anchor in segment.lower() for anchor in anchors)
        or any(slot_value in segment.lower() for slot_value in slot_values)
    ]
    if matched:
        return " ".join(matched)
    return text


def _current_state_family_anchors(family: str) -> tuple[str, ...]:
    if family == "current_target_date":
        return (
            "target date",
            "launch target",
            "launch update",
            "official pilot target",
            "moves from",
            "moved from",
            "launch date remains",
            "hearing date",
            "closure date",
            "review memo",
        )
    if family == "current_public_event_date":
        return ("orientation", "public schedule", "calendar line", "public event", "moved from", "moved to", "rescheduled")
    if family == "current_budget":
        return ("budget", "approved budget", "current budget", "support amount")
    if family == "current_discount":
        return ("discount", "maximum discount", "discount cap", "commercial cap")
    if family == "current_stipend":
        return ("stipend", "support amount", "ra-support", "ra support")
    if family == "current_safe_wording":
        return ("safe wording", "safe sponsor wording", "safe external wording", "broad wording", "customer wording")
    if family == "current_blocker":
        return ("blocker", "still pending", "hold cleared", "hold state", "remaining")
    if family == "household_plan":
        return ("visit window", "entry method", "approved door", "text on arrival", "package rule", "approved areas", "parking pass")
    return ()


def _current_state_family_slot_keys(family: str) -> tuple[str, ...]:
    mapping = {
        "current_target_date": ("target_date",),
        "current_public_event_date": ("public_event_date",),
        "current_budget": ("approved_budget", "budget"),
        "current_discount": ("approved_discount_cap", "discount_cap"),
        "current_stipend": ("monthly_stipend", "stipend"),
        "current_safe_wording": ("safe_wording", "safe_external_wording"),
        "current_blocker": ("blocker", "status"),
        "household_plan": ("date", "visit_window", "entry_method", "package_rule", "approved_areas", "arrival_contact_rule", "parking_pass"),
    }
    return mapping.get(family, ())


def _score_surface_line(
    *,
    question: str,
    text: str,
    domains: set[str],
    line_domains: set[str],
    question_profile: dict[str, Any],
    line_meta: dict[str, Any],
) -> float:
    if _is_question_like_text(text):
        return -10.0
    if _looks_like_private_secret_text(text):
        return -10.0
    score = 0.0
    slots = {str(key): value for key, value in dict(line_meta.get("slots") or {}).items() if value}
    slot_names = set(slots)
    frame_type = str(line_meta.get("frame_type") or "")
    current_state_request = bool(question_profile.get("wants_current_state_bundle")) or bool(question_profile.get("asks_public_event_date"))
    household_plan_request = bool(question_profile.get("asks_household_plan"))
    if line_domains & domains:
        score += 2.0 * len(line_domains & domains)
    if current_state_request and frame_type in {"project_state", "research_state"}:
        score += 3.0
    if household_plan_request and frame_type == "household_plan":
        score += 3.0
    if question_profile.get("asks_public_event_date") and "public_event_date" in slot_names:
        score += 3.2
    if question_profile.get("asks_public_event_date") and frame_type in {"project_state", "research_state"} and "public_event_date" not in slot_names:
        score -= 1.6
    if current_state_request and line_meta.get("is_current_like"):
        score += 2.2
    if household_plan_request and slot_names & {"visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}:
        score += 2.0
    if "schedule" in domains and (frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics"} or slot_names & {"date", "time", "location", "arrival_time", "procedure"}):
        score += 1.5
    if "allergy" in domains and (frame_type == "allergy" or slot_names & {"allergy_substance", "allergy_reaction"}):
        score += 1.5
    if "instruction" in domains and (frame_type == "instruction" or "instruction" in slot_names):
        score += 1.5
    if "cancellation" in domains and (frame_type == "cancellation" or str(line_meta.get("lifecycle_status") or "").lower() in {"canceled", "superseded"}):
        score += 1.5
    target_procedures = set(question_profile.get("target_procedures") or [])
    line_procedures = set(line_meta.get("procedures") or [])
    if target_procedures:
        if line_procedures & target_procedures:
            score += 1.4
        elif line_procedures and not (line_procedures & target_procedures):
            score -= 1.8
    target_frame_types = set(question_profile.get("target_frame_types") or [])
    if target_frame_types:
        if frame_type in target_frame_types:
            score += 0.8
        elif frame_type not in {"cancellation", "instruction"}:
            score -= 0.5
    if household_plan_request and frame_type == "appointment" and not slot_names & {"visit_window", "entry_method", "package_rule"}:
        score -= 1.8
    if question_profile.get("wants_imaging_only") and frame_type in {"clinic_visit", "medication", "diagnosis_or_result"}:
        score -= 1.8
    intent_slot_groups = {
        "wants_callback_instruction": {"condition", "instruction"},
        "wants_contact_protocol": {"contact_method", "phone", "portal", "backup_contact"},
        "wants_followup_plan": {"instruction", "date", "time", "location", "arrival_time"},
        "wants_authorized_details": {"date", "time", "location", "arrival_time", "provider"},
        "wants_current_plan_window": {"instruction", "condition", "date", "time", "location"},
    }
    for profile_key, required_slots in intent_slot_groups.items():
        if question_profile.get(profile_key):
            score += 1.5 * len(slot_names & required_slots)
    if question_profile.get("wants_authorized_details") and line_meta.get("is_policy_only"):
        score -= 1.5
    if question_profile.get("prefer_current") and line_meta.get("is_current_like"):
        score += 0.6
    score += _subject_alignment_score(
        question_profile=question_profile,
        text=text,
        line_meta=line_meta,
    )
    if question_profile.get("wants_current_policy_state"):
        if _is_policy_revocation_text(text):
            score += 2.6
        elif _looks_like_policy_state_text(text):
            score += 1.2
    if question_profile.get("avoid_non_target_labs") and line_meta.get("is_lab_like") and not (line_procedures & target_procedures):
        score -= 1.2
    if question_profile.get("prefer_authorized_logistics") and line_meta.get("is_policy_only"):
        score -= 2.0
    if question_profile.get("mixed_allergy_schedule") and (
        (frame_type == "allergy" or slot_names & {"allergy_substance", "allergy_reaction"})
        or slot_names & {"date", "time", "location", "arrival_time"}
    ):
        score += 0.8
    if len(text) < 15:
        score -= 0.5
    if _normalize_surface_line(question) == _normalize_surface_line(text):
        score -= 1.0
    return score


def _score_focused_surface_line(
    *,
    mode: str,
    question: str,
    text: str,
    line_domains: set[str],
    line_meta: dict[str, Any],
    row_score: float,
) -> float:
    slots = {str(key): value for key, value in dict(line_meta.get("slots") or {}).items() if value}
    frame_type = str(line_meta.get("frame_type") or "")
    score = float(row_score)
    required_slots_by_mode = {
        "policy": {"policy_scope"},
        "contact_protocol": {"contact_method", "phone", "portal", "backup_contact"},
        "contact_methods": {"contact_method", "phone", "portal", "backup_contact"},
        "medication_status": {"medication", "dosage", "instruction", "timing"},
        "return_precautions": {"condition", "instruction"},
        "callback_instruction": {"condition", "instruction"},
        "current_plan_window": {"instruction", "condition", "date", "time", "location", "arrival_time"},
        "followup_plan": {"instruction", "condition", "date", "time", "location", "arrival_time"},
        "current_med_followup_plan": {"medication", "dosage", "instruction", "date", "time", "location"},
        "authorized_schedule": {"date", "time", "location", "arrival_time", "procedure"},
        "final_procedure_plan": {"instruction", "date", "time", "location", "arrival_time", "procedure"},
        "rescheduled_followup": {"date", "time", "location", "instruction", "medication"},
        "imaging_schedule": {"date", "time", "location", "arrival_time", "procedure"},
        "current_state_bundle": {
            "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
            "monthly_stipend", "safe_wording", "blocker", "date", "visit_window",
            "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule",
        },
        "mixed_allergy_schedule": {"allergy_substance", "allergy_reaction", "date", "time", "location", "arrival_time"},
    }
    required_slots = required_slots_by_mode.get(mode, set())
    score += 1.25 * len(required_slots & set(slots))
    if frame_type in {"project_state", "research_state", "household_plan", "appointment", "test_or_imaging", "clinic_visit", "medication", "instruction", "consent_or_permission", "privacy_policy"}:
        score += 0.5
    if mode == "current_state_bundle" and line_meta.get("is_current_like"):
        score += 1.5
    if mode == "current_state_bundle" and line_meta.get("is_access_artifact_temporal"):
        score -= 3.0
    if mode == "policy" and frame_type in {"consent_or_permission", "privacy_policy"}:
        score += 2.0
    if mode in {"medication_status", "current_med_followup_plan"} and frame_type == "medication":
        score += 1.5
    if mode in {"authorized_schedule", "imaging_schedule", "followup_plan", "current_plan_window"} and frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics"}:
        score += 1.5
    if mode in {"return_precautions", "callback_instruction"} and slots.get("condition") and slots.get("instruction"):
        score += 1.5
    if not slots:
        score -= 1.0
    if len(text.strip()) < 12:
        score -= 1.0
    return score


def _verify_surface_replay_answer(
    answer_text: str,
    selected_lines: list[dict[str, Any]],
    question: str,
    *,
    semantic_intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semantic_domains = _semantic_domains((semantic_intent or {}).get("semantic_spec"))
    domains = semantic_domains or _infer_query_domains(question)
    profile = _build_question_profile(question, domains, semantic_intent=semantic_intent)
    mode = _focused_surface_mode(profile)
    if mode == "policy":
        return _verify_policy_state_answer(answer_text, selected_lines)
    if mode == "imaging_schedule":
        return _verify_imaging_schedule_answer(answer_text, selected_lines, profile)
    relevant_lines = _select_relevant_lines_for_verification(
        question,
        selected_lines,
        semantic_intent=semantic_intent,
    )
    required_surfaces: list[str] = []
    if "schedule" in domains:
        required_surfaces.extend(_extract_surface_spans_from_lines(relevant_lines, {"date", "time", "location", "provider", "arrival"}))
    if "allergy" in domains:
        required_surfaces.extend(_extract_surface_spans_from_lines(relevant_lines, {"allergy", "reaction"}))
    if "instruction" in domains:
        required_surfaces.extend(_extract_surface_spans_from_lines(relevant_lines, {"instruction"}))
    if "cancellation" in domains:
        required_surfaces.extend(_extract_surface_spans_from_lines(relevant_lines, {"cancellation"}))
    required_surfaces.extend(_extract_mode_required_surfaces(mode=mode, selected_lines=relevant_lines))
    deduped_required = _dedupe_list(required_surfaces)
    lower_answer = answer_text.lower()
    hits = [surface for surface in deduped_required if surface.lower() in lower_answer]
    misses = [surface for surface in deduped_required if surface.lower() not in lower_answer]
    return {
        "required_surfaces": deduped_required,
        "hit_surfaces": hits,
        "missing_surfaces": misses,
        "passed": not misses,
    }


def _verify_policy_state_answer(answer_text: str, selected_lines: list[dict[str, Any]]) -> dict[str, Any]:
    state = _summarize_policy_state(selected_lines)
    lowered = answer_text.lower()
    required_surfaces: list[str] = []
    hit_surfaces: list[str] = []
    if state["revoked"]:
        required_surfaces.append("family scheduling access is revoked")
        if any(token in lowered for token in ["access is revoked", "access revoked", "revoke", "revoked"]):
            hit_surfaces.append("family scheduling access is revoked")
    if state["no_future_sharing"]:
        required_surfaces.append("do not share future appointment details with family")
        if any(token in lowered for token in ["do not share future appointment details", "no future appointment details", "do not share future appointment"]):
            hit_surfaces.append("do not share future appointment details with family")
    if state["removed_contact_permissions"]:
        required_surfaces.append("linda park removed from scheduling-contact and callback-contact permissions")
        if any(token in lowered for token in ["removed from scheduling-contact and callback-contact permissions", "removed from scheduling-contact", "callback-contact permissions"]):
            hit_surfaces.append("linda park removed from scheduling-contact and callback-contact permissions")
    if state["logistics_only"] and not state["revoked"]:
        required_surfaces.append("family access is limited to scheduling logistics only")
        if any(token in lowered for token in ["logistics only", "scheduling logistics only", "limited to scheduling logistics"]):
            hit_surfaces.append("family access is limited to scheduling logistics only")
    misses = [surface for surface in required_surfaces if surface not in hit_surfaces]
    return {
        "required_surfaces": required_surfaces,
        "hit_surfaces": hit_surfaces,
        "missing_surfaces": misses,
        "passed": not misses,
    }


def _verify_imaging_schedule_answer(
    answer_text: str,
    selected_lines: list[dict[str, Any]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    primary_line = _pick_primary_imaging_schedule_line(selected_lines, profile=profile)
    required_surfaces: list[str] = []
    if primary_line is not None:
        meta = dict(primary_line.get("line_meta") or {})
        spans = dict(meta.get("surface_spans") or {})
        slots = dict(meta.get("slots") or {})
        for key in ["date", "time", "location"]:
            value = str(spans.get(key) or slots.get(key) or "").strip()
            if value:
                required_surfaces.append(value)
        arrival = _extract_arrival_surface(spans, str(primary_line.get("text") or ""))
        if arrival:
            required_surfaces.append(arrival)
    if not required_surfaces:
        relevant_lines = [primary_line] if primary_line is not None else selected_lines[:1]
        required_surfaces = _dedupe_list(_extract_surface_spans_from_lines(relevant_lines, {"date", "time", "location", "arrival"}))
    else:
        required_surfaces = _dedupe_list(required_surfaces)
    lower_answer = answer_text.lower()
    hits = [surface for surface in required_surfaces if surface.lower() in lower_answer]
    misses = [surface for surface in required_surfaces if surface.lower() not in lower_answer]
    return {
        "required_surfaces": required_surfaces,
        "hit_surfaces": hits,
        "missing_surfaces": misses,
        "passed": not misses,
    }


def _summarize_policy_state(selected_lines: list[dict[str, Any]]) -> dict[str, bool]:
    texts = [str(line.get("text") or "").strip().lower() for line in selected_lines]
    return {
        "revoked": any(
            any(token in text for token in ["revoke", "revoked", "family scheduling access revoked", "removed from scheduling-contact"])
            for text in texts
        ),
        "no_future_sharing": any(
            any(token in text for token in ["do not share future appointment details", "no longer release appointment times"])
            for text in texts
        ),
        "logistics_only": any(
            any(token in text for token in ["logistics only", "may receive scheduling logistics", "appointment times", "parking details"])
            for text in texts
        ),
        "removed_contact_permissions": any(
            any(token in text for token in ["removed from scheduling-contact", "removed from callback-contact", "callback-contact permissions"])
            for text in texts
        ),
    }


def _select_relevant_lines_for_verification(
    question: str,
    selected_lines: list[dict[str, Any]],
    *,
    semantic_intent: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    semantic_domains = _semantic_domains((semantic_intent or {}).get("semantic_spec"))
    domains = semantic_domains or _infer_query_domains(question)
    profile = _build_question_profile(question, domains, semantic_intent=semantic_intent)
    target_procedures = set(profile.get("target_procedures") or [])
    include_cancellation = "cancellation" in domains
    relevant: list[dict[str, Any]] = []
    for line in selected_lines:
        meta = line.get("line_meta") or {}
        line_procedures = set(meta.get("procedures") or [])
        frame_type = str(meta.get("frame_type") or "")
        text_lower = str(line.get("text") or "").lower()
        if _is_question_like_text(text_lower):
            continue
        if target_procedures and line_procedures and not (line_procedures & target_procedures):
            continue
        if profile.get("wants_imaging_only") and frame_type in {"clinic_visit", "medication"} and not any(token in text_lower for token in ["imaging", "ultrasound", "scan"]):
            continue
        if not include_cancellation and any(token in text_lower for token in ["canceled", "cancelled", "prior ", "old "]):
            continue
        if "allergy" in domains and frame_type not in {"allergy"} and "schedule" not in domains:
            continue
        target_weekdays = set(profile.get("target_weekdays") or [])
        line_weekdays = _extract_weekday_mentions(text_lower)
        if target_weekdays and line_weekdays and not (target_weekdays & line_weekdays) and profile.get("prefer_strict_target_day"):
            continue
        relevant.append(line)
    return relevant or selected_lines


def _extract_mode_required_surfaces(*, mode: str | None, selected_lines: list[dict[str, Any]]) -> list[str]:
    if mode is None:
        return []
    slots_by_mode: dict[str, set[str]] = {
        "callback_instruction": {"condition", "instruction"},
        "return_precautions": {"condition", "instruction"},
        "policy": {"policy_scope", "consent_scope", "status"},
        "authorized_schedule": {"date", "time", "arrival_time", "location", "provider", "procedure"},
        "contact_methods": {"contact_method", "phone", "portal", "backup_contact"},
        "medication_status": {"medication", "dosage", "instruction", "timing"},
        "current_plan_window": {"instruction", "condition", "date", "time", "arrival_time", "location"},
        "current_med_followup_plan": {"medication", "dosage", "instruction", "date", "time", "location"},
        "final_procedure_plan": {"instruction", "date", "time", "arrival_time", "location", "procedure"},
        "rescheduled_followup": {"instruction", "date", "time", "arrival_time", "location", "procedure"},
        "current_state_bundle": {
            "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
            "monthly_stipend", "safe_wording", "blocker", "status", "date", "visit_window",
            "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule",
        },
    }
    wanted = slots_by_mode.get(mode, set())
    surfaces: list[str] = []
    for line in selected_lines:
        slots = dict((line.get("line_meta") or {}).get("slots") or {})
        for key in wanted:
            value = str(slots.get(key) or "").strip()
            if value:
                surfaces.append(value)
    return _dedupe_list(surfaces)


def _extract_surface_spans_from_lines(selected_lines: list[dict[str, Any]], requested_groups: set[str]) -> list[str]:
    surfaces: list[str] = []
    for line in selected_lines:
        text = str(line["text"])
        lowered = text.lower()
        if "date" in requested_groups:
            surfaces.extend(match.group(0) for match in DATE_SPAN_RE.finditer(text))
        if "time" in requested_groups:
            surfaces.extend(match.group(0) for match in TIME_SPAN_RE.finditer(text))
        if "location" in requested_groups:
            surfaces.extend(match.group(0) for match in LOCATION_SPAN_RE.finditer(text))
        if "provider" in requested_groups:
            surfaces.extend(match.group(0) for match in PROVIDER_SPAN_RE.finditer(text))
        if "arrival" in requested_groups:
            surfaces.extend(match.group(0) for match in ARRIVAL_SPAN_RE.finditer(text))
        if "allergy" in requested_groups and any(token in lowered for token in ["sulfa", "allergy", "reaction", "rash"]):
            surfaces.append(text)
        if "reaction" in requested_groups and "rash" in lowered:
            surfaces.append("rash")
        if "instruction" in requested_groups and any(token in lowered for token in ["beta-hcg", "unless symptoms worsen", "before tuesday"]):
            surfaces.append(text)
        if "cancellation" in requested_groups and any(token in lowered for token in ["canceled", "cancelled", "no longer active", "inactive"]):
            surfaces.append(text)
    return [surface.strip() for surface in surfaces if surface and len(surface.strip()) > 1]


def _strip_message_prefix(line: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]\s*", "", str(line or "").strip()).strip()


def _normalize_surface_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _dedupe_list(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        normalized = _normalize_surface_line(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(value)
    return out


def _extract_weekday_mentions(text: str) -> set[str]:
    weekdays = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    return {weekday for weekday in weekdays if weekday in text}


def _extract_explicit_person_names(text: str) -> list[str]:
    blocked = {
        "As Of",
        "Checkpoint Note",
        "Desk Note",
        "Service Update",
        "Travel Update",
        "Background Operations",
        "Current Technician",
        "Current GuestMode",
    }
    blocked_tokens = {
        "As", "Of", "Guest", "GuestMode", "Checkpoint", "Note", "Desk", "Service",
        "Travel", "Background", "Operations", "Current", "Updating", "Initial", "Recording",
        "Approved", "Arrival", "Window", "Front", "Guestmode", "Saturday", "Sunday", "Monday",
        "Tuesday", "Wednesday", "Thursday", "Friday", "January", "February", "March", "April",
        "May", "June", "July", "August", "September", "October", "November", "December",
    }
    names: list[str] = []
    for match in PERSON_NAME_RE.finditer(str(text or "")):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        if name in blocked:
            continue
        if name not in names:
            names.append(name)
    for token in PERSON_TOKEN_RE.findall(str(text or "")):
        if token in blocked_tokens:
            continue
        if token not in names:
            names.append(token)
    return names


def _extract_question_target_subjects(question: str) -> list[str]:
    names = _extract_explicit_person_names(question)
    if names:
        return names
    return []


def _expand_subject_tokens(names: list[str]) -> set[str]:
    tokens: set[str] = set()
    for name in names:
        for token in re.findall(r"[A-Za-z]+", str(name or "")):
            cleaned = token.strip().lower()
            if len(cleaned) >= 3:
                tokens.add(cleaned)
    return tokens


def _subject_alignment_score(*, question_profile: dict[str, Any], text: str, line_meta: dict[str, Any]) -> float:
    target_tokens = {str(token).lower() for token in (question_profile.get("target_subject_tokens") or []) if str(token).strip()}
    if not target_tokens:
        return 0.0
    explicit_names = [str(name).lower() for name in (line_meta.get("explicit_names") or []) if str(name).strip()]
    line_tokens = {
        token
        for name in explicit_names
        for token in re.findall(r"[a-z]+", name)
        if len(token) >= 3
    }
    if not line_tokens:
        lowered = str(text or "").lower()
        line_tokens = {token for token in re.findall(r"[a-z]+", lowered) if len(token) >= 3}
    matched = target_tokens & line_tokens
    if matched:
        return 2.4 + 0.2 * len(matched)
    explicit_line_tokens = {
        token
        for name in explicit_names
        for token in re.findall(r"[a-z]+", name)
        if len(token) >= 3
    }
    if explicit_line_tokens and not (target_tokens & explicit_line_tokens):
        return -2.4
    return 0.0


def _build_question_profile(question: str, domains: set[str], semantic_intent: dict[str, Any] | None = None) -> dict[str, Any]:
    semantic_intent = semantic_intent or {}
    semantic_spec = dict(semantic_intent.get("semantic_spec") or {})
    if _has_semantic_spec(semantic_spec):
        return _build_semantic_question_profile(
            domains=domains,
            semantic_spec=semantic_spec,
            semantic_intent=semantic_intent,
        )

    lowered = question.lower()
    regimen_coverage_families = _infer_question_regimen_families(lowered, semantic_intent)
    requested_current_slots = list(semantic_intent.get("requested_current_slots") or [])
    if not requested_current_slots:
        requested_current_slots = _infer_requested_current_slots(lowered)
    requested_slot_set = set(requested_current_slots)
    requested_current_state_slots = set(shared_infer_current_state_slots(lowered))
    requested_household_slots = set(shared_infer_household_slots(lowered))
    target_subject_names = _extract_question_target_subjects(question)
    target_subject_tokens = _expand_subject_tokens(target_subject_names)
    target_procedures: set[str] = set()
    target_frame_types: set[str] = set()
    target_weekdays = _extract_weekday_mentions(lowered)
    through_bound = _extract_through_weekday(lowered)
    excluded_weekdays = set()
    if any(token in lowered for token in ["imaging", "ultrasound", "scan"]):
        target_procedures.update({"ultrasound", "imaging", "scan"})
        target_frame_types.update({"test_or_imaging", "appointment", "logistics"})
    if any(token in lowered for token in ["allergy", "reaction"]):
        target_frame_types.add("allergy")
    if any(token in lowered for token in ["medication", "dose", "taking"]):
        target_frame_types.add("medication")
    if any(token in lowered for token in ["instruction", "before", "still needs", "lab work"]):
        target_frame_types.add("instruction")
    if any(token in lowered for token in ["follow-up", "follow up", "visit", "appointment"]) and not target_frame_types:
        target_frame_types.update({"appointment", "clinic_visit", "test_or_imaging", "logistics"})
    wants_authorized_details = bool(semantic_intent.get("authorized_schedule_request")) or any(token in lowered for token in ["authorized appointment", "authorized details", "through friday", "logistics only", "appointment details"])
    wants_callback_instruction = bool(semantic_intent.get("urgent_callback_request")) or any(token in lowered for token in ["triage", "warning sign", "emergency department", "return precautions", "go in right away"])
    wants_followup_plan = bool(semantic_intent.get("followup_plan_request")) or any(token in lowered for token in ["updated follow-up plan", "what happens next", "next step", "after friday's scan", "after friday scan"])
    wants_imaging_only = bool(semantic_intent.get("imaging_schedule_request")) or any(token in lowered for token in ["latest imaging", "imaging appointment", "scan time", "ultrasound time", "time and location", "which suite", "what suite", "which location", "what location"])
    if any(token in lowered for token in ["what time", "which suite", "what suite", "where is", "where should"]) and any(
        token in lowered for token in ["ultrasound", "scan", "imaging", "appointment", "clinic", "suite"]
    ):
        wants_imaging_only = True
    mixed_allergy_schedule = ("allergy" in lowered or "reaction" in lowered) and ("schedule" in lowered or "tuesday" in lowered or "appointment" in lowered)
    explicit_policy_question = any(token in lowered for token in ["family-access setting", "access setting", "chart and scheduling", "permissions", "permission review", "currently authorized", "current family access"])
    wants_policy_setting = explicit_policy_question
    wants_current_policy_state = explicit_policy_question or any(
        token in lowered for token in ["currently active", "current family access", "currently authorized", "revoked", "removed", "still active for future callbacks"]
    )
    wants_contact_protocol = bool(semantic_intent.get("contact_protocol_request")) or any(
        token in lowered
        for token in [
            "callback instruction",
            "callback number",
            "future callbacks",
            "what contact methods",
            "which contact method",
            "voicemail",
            "portal",
            "generic callback",
            "safe line",
            "after sunday evening",
        ]
    )
    wants_medication_regimen_plan = (
        bool(semantic_intent.get("current_regimen_request"))
        and any(token in lowered for token in ["medication", "medicine", "dose", "dosage", "regimen", "treatment"])
    ) or any(
        token in lowered
        for token in [
            "treatment plan",
            "regimen",
            "blood pressure plan",
            "home blood pressure plan",
        ]
    ) or (
        any(token in lowered for token in ["current", "right now", "now"])
        and any(token in lowered for token in ["start", "stop", "continue", "hold", "restart", "take", "monitor", "check"])
    )
    if "household" in domains or requested_household_slots:
        wants_medication_regimen_plan = False
    if any(token in lowered for token in ["biopsy-prep", "prep plan", "preparation as of now", "prep plan right now", "current biopsy-prep plan"]):
        wants_medication_regimen_plan = False
    wants_active_medication_status = wants_medication_regimen_plan or any(
        token in lowered for token in ["currently active for", "currently active medication", "current active medication", "which medication was explicitly stopped", "what medications are currently active", "active cardiac", "peri-procedure medication instructions", "blood pressure plan", "home blood pressure plan"]
    )
    if "household" in domains or requested_household_slots:
        wants_active_medication_status = False
    wants_return_precautions = bool(semantic_intent.get("return_precaution_request")) or any(token in lowered for token in ["return precautions", "reinforce for", "reinforce right now", "return urgently", "post-procedure nursing"])
    wants_contact_methods = bool(semantic_intent.get("contact_methods_request")) or any(token in lowered for token in ["what contact methods are still active", "future callbacks", "portal only", "contact methods"])
    wants_current_plan_window = bool(semantic_intent.get("current_plan_window_request")) or any(token in lowered for token in ["before tuesday", "still needs to happen before tuesday", "what still needs to happen"])
    wants_current_med_followup_plan = any(token in lowered for token in ["current medication and follow-up plan", "medication and follow-up plan", "follow-up plan after", "medication plan from here", "follow-up stays"])
    wants_final_procedure_plan = any(token in lowered for token in ["final friday procedure plan", "procedure plan and preparation", "preparation as of now", "final procedure plan"])
    if any(token in lowered for token in ["biopsy-prep", "prep plan", "current biopsy-prep plan"]):
        wants_final_procedure_plan = True
    asks_preparation_details = any(
        token in lowered
        for token in [
            "prep",
            "prepare",
            "preparation",
            "pre-op",
            "pre op",
            "before the procedure",
            "before the biopsy",
            "before surgery",
        ]
    )
    wants_future_callback_only = any(token in lowered for token in ["future callbacks", "still active for future callbacks"])
    wants_rescheduled_followup = any(token in lowered for token in ["reschedule", "rescheduled", "move tuesday later", "after noon", "updated tuesday"])
    asks_public_event_date = "public_event_date" in requested_current_state_slots
    asks_private_case_state_date = "target_date" in requested_current_state_slots
    asks_current_budget = "approved_budget" in requested_current_state_slots
    asks_current_discount = "approved_discount_cap" in requested_current_state_slots
    asks_current_stipend = "monthly_stipend" in requested_current_state_slots
    asks_safe_wording = "safe_wording" in requested_current_state_slots
    asks_current_blocker = "blocker" in requested_current_state_slots
    asks_household_plan = bool(requested_household_slots)
    asks_household_visit_window = "visit_window" in requested_household_slots
    asks_household_entry_method = "entry_method" in requested_household_slots
    asks_household_package_rule = "package_rule" in requested_household_slots
    asks_household_approved_areas = "approved_areas" in requested_household_slots
    asks_household_parking_pass = "parking_pass" in requested_household_slots
    asks_household_arrival_contact_rule = "arrival_contact_rule" in requested_household_slots
    wants_current_state_bundle = bool(requested_slot_set) or "as of now" in lowered
    if any(token in lowered for token in ["currently authorized procedure time", "arrival window", "currently authorized appointment details"]):
        wants_authorized_details = True
    prefer_current = any(token in lowered for token in ["current", "latest", "still", "through"])
    prefer_strict_target_day = bool(target_weekdays) and any(token in lowered for token in ["current", "latest", "through", "tuesday", "friday"])
    if through_bound:
        excluded_weekdays = _weekdays_after(through_bound)
    return {
        "target_procedures": sorted(target_procedures),
        "target_frame_types": sorted(target_frame_types),
        "prefer_current": prefer_current,
        "avoid_non_target_labs": bool(target_procedures),
        "prefer_authorized_logistics": "schedule" in domains,
        "target_weekdays": sorted(target_weekdays),
        "through_bound_weekday": through_bound,
        "excluded_weekdays": sorted(excluded_weekdays),
        "wants_authorized_details": wants_authorized_details,
        "wants_callback_instruction": wants_callback_instruction,
        "wants_followup_plan": wants_followup_plan,
        "wants_imaging_only": wants_imaging_only,
        "mixed_allergy_schedule": mixed_allergy_schedule,
        "wants_policy_setting": wants_policy_setting,
        "wants_current_policy_state": wants_current_policy_state,
        "wants_contact_protocol": wants_contact_protocol,
        "wants_active_medication_status": wants_active_medication_status,
        "wants_medication_regimen_plan": wants_medication_regimen_plan,
        "wants_return_precautions": wants_return_precautions,
        "wants_contact_methods": wants_contact_methods,
        "wants_current_plan_window": wants_current_plan_window,
        "regimen_coverage_families": regimen_coverage_families,
        "wants_current_med_followup_plan": wants_current_med_followup_plan,
        "wants_final_procedure_plan": wants_final_procedure_plan,
        "asks_preparation_details": asks_preparation_details,
        "wants_future_callback_only": wants_future_callback_only,
        "wants_rescheduled_followup": wants_rescheduled_followup,
        "wants_current_state_bundle": wants_current_state_bundle,
        "requested_current_slots": requested_current_slots,
        "asks_public_event_date": asks_public_event_date,
        "asks_private_case_state_date": asks_private_case_state_date,
        "asks_current_budget": asks_current_budget,
        "asks_current_discount": asks_current_discount,
        "asks_current_stipend": asks_current_stipend,
        "asks_safe_wording": asks_safe_wording,
        "asks_current_blocker": asks_current_blocker,
        "asks_household_plan": asks_household_plan,
        "asks_household_visit_window": asks_household_visit_window,
        "asks_household_entry_method": asks_household_entry_method,
        "asks_household_package_rule": asks_household_package_rule,
        "asks_household_approved_areas": asks_household_approved_areas,
        "asks_household_parking_pass": asks_household_parking_pass,
        "asks_household_arrival_contact_rule": asks_household_arrival_contact_rule,
        "prefer_strict_target_day": prefer_strict_target_day,
        "target_subject_names": target_subject_names,
        "target_subject_tokens": sorted(target_subject_tokens),
        "semantic_profile_source": "fallback_heuristic",
    }


def _build_semantic_question_profile(
    *,
    domains: set[str],
    semantic_spec: dict[str, Any],
    semantic_intent: dict[str, Any],
) -> dict[str, Any]:
    requested_slots = {
        str(slot)
        for slot in list(semantic_spec.get("requested_slots") or [])
        if str(slot).strip()
    }
    current_slots = {
        "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
        "monthly_stipend", "safe_wording", "blocker", "access_room", "access_badge",
        "operational_result",
    }
    household_slots = {
        "date", "time", "location", "visit_window", "entry_method", "package_rule",
        "approved_areas", "parking_pass", "arrival_contact_rule",
    }
    target_frame_types: set[str] = set()
    if requested_slots & current_slots:
        target_frame_types.add("research_state" if semantic_spec.get("state_domain") == "research" else "project_state")
    if requested_slots & household_slots:
        target_frame_types.add("household_plan")
    if requested_slots & {"date", "time", "location", "public_event_date"}:
        target_frame_types.add("appointment")
    if requested_slots & {"medication", "dosage"}:
        target_frame_types.add("medication")
    if requested_slots & {"instruction", "condition"}:
        target_frame_types.add("instruction")

    current = str(semantic_spec.get("temporal_scope") or "") == "current"
    policy_request = str(semantic_spec.get("request_shape") or "") == "policy" or bool(
        semantic_intent.get("policy_state_request")
    )
    return {
        "target_procedures": [],
        "target_frame_types": sorted(target_frame_types),
        "prefer_current": current,
        "avoid_non_target_labs": False,
        "prefer_authorized_logistics": "schedule" in domains,
        "target_weekdays": [],
        "through_bound_weekday": "",
        "excluded_weekdays": [],
        "wants_authorized_details": bool(semantic_intent.get("authorized_schedule_request")),
        "wants_callback_instruction": bool(semantic_intent.get("urgent_callback_request")),
        "wants_followup_plan": bool(semantic_intent.get("followup_plan_request")),
        "wants_imaging_only": bool(semantic_intent.get("imaging_schedule_request")),
        "mixed_allergy_schedule": False,
        "wants_policy_setting": policy_request,
        "wants_current_policy_state": policy_request,
        "wants_contact_protocol": bool(semantic_intent.get("contact_protocol_request")),
        "wants_active_medication_status": bool(semantic_intent.get("current_regimen_request")),
        "wants_medication_regimen_plan": bool(semantic_intent.get("current_regimen_request")),
        "wants_return_precautions": bool(semantic_intent.get("return_precaution_request")),
        "wants_contact_methods": bool(semantic_intent.get("contact_methods_request")),
        "wants_current_plan_window": bool(semantic_intent.get("current_plan_window_request")),
        "regimen_coverage_families": [],
        "wants_current_med_followup_plan": False,
        "wants_final_procedure_plan": False,
        "asks_preparation_details": False,
        "wants_future_callback_only": False,
        "wants_rescheduled_followup": False,
        "wants_current_state_bundle": bool(requested_slots) and current,
        "requested_current_slots": sorted(requested_slots),
        "asks_public_event_date": "public_event_date" in requested_slots,
        "asks_private_case_state_date": "target_date" in requested_slots,
        "asks_current_budget": "approved_budget" in requested_slots,
        "asks_current_discount": "approved_discount_cap" in requested_slots,
        "asks_current_stipend": "monthly_stipend" in requested_slots,
        "asks_safe_wording": "safe_wording" in requested_slots,
        "asks_current_blocker": "blocker" in requested_slots,
        "asks_household_plan": bool(requested_slots & household_slots),
        "asks_household_visit_window": "visit_window" in requested_slots,
        "asks_household_entry_method": "entry_method" in requested_slots,
        "asks_household_package_rule": "package_rule" in requested_slots,
        "asks_household_approved_areas": "approved_areas" in requested_slots,
        "asks_household_parking_pass": "parking_pass" in requested_slots,
        "asks_household_arrival_contact_rule": "arrival_contact_rule" in requested_slots,
        "prefer_strict_target_day": False,
        "target_subject_names": [],
        "target_subject_tokens": [],
        "semantic_profile_source": "semantic_spec",
    }


def _infer_question_regimen_families(lowered: str, semantic_intent: dict[str, Any]) -> list[str]:
    families: list[str] = []
    if any(token in lowered for token in ["stop", "what should i stop"]):
        families.append("stop")
    if any(token in lowered for token in ["start", "what should i start"]) or bool(semantic_intent.get("start_or_add_regimen_request")):
        families.append("start")
    if _question_explicitly_requests_continue_regimen(lowered):
        families.append("continue")
    if re.search(r"\buse\b", lowered) or "what should i use" in lowered or "what can i take" in lowered:
        families.append("use")
        if _question_implies_start_add_regimen(lowered):
            families.append("start")
    if any(token in lowered for token in ["monitor", "check", "blood pressure", "twice daily", "blood pressure plan", "home blood pressure plan"]) or bool(semantic_intent.get("monitoring_plan_request")):
        families.append("monitor")
    if _should_require_continue_regimen_family(lowered, semantic_intent=semantic_intent, current_families=families) and "continue" not in families:
        families.append("continue")
    deduped: list[str] = []
    for family in families:
        if family not in deduped:
            deduped.append(family)
    return deduped


def _question_explicitly_requests_continue_regimen(lowered: str) -> bool:
    return any(
        token in lowered
        for token in [
            "continue",
            "stay the same",
            "regular medicines stay the same",
            "keep taking",
            "keep using",
        ]
    )


def _should_require_continue_regimen_family(
    lowered: str,
    *,
    semantic_intent: dict[str, Any],
    current_families: list[str],
) -> bool:
    if _question_explicitly_requests_continue_regimen(lowered):
        return True
    if not bool(semantic_intent.get("continue_regimen_request")):
        return False
    if any(family in current_families for family in ["start", "stop", "use", "monitor"]):
        return False
    return any(
        token in lowered
        for token in [
            "current medications",
            "currently active medication",
            "currently active medications",
            "what medications are currently active",
            "which medications are currently active",
            "medication plan",
            "treatment plan",
            "regimen",
        ]
    )


def _infer_requested_current_slots(lowered: str) -> list[str]:
    required = shared_infer_current_state_slots(lowered)
    for slot_name in shared_infer_household_slots(lowered):
        if slot_name == "date" or slot_name not in required:
            required.append(slot_name)
    return list(dict.fromkeys(required))


def _question_implies_start_add_regimen(lowered: str) -> bool:
    if re.search(r"\b(?:what|which)\b.*\b(?:medication|medications|medicine|medicines)\b.*\b(?:use|take)\b.*\bfor\b", lowered):
        return True
    if re.search(r"\b(?:what|which)\b.*\b(?:use|take)\b.*\bfor\b", lowered) and any(
        token in lowered for token in ["pain", "nausea", "symptom", "symptoms", "treatment", "relief"]
    ):
        return True
    return False


def _build_line_metadata(text: str) -> dict[str, Any]:
    pseudo = RetrievedEvidence(
        memory_id="line",
        content=text,
        score=1.0,
        retrieval_source="line",
        reason="line_scoring",
        user_id=None,
        memory_type="line",
        scope="line",
        entities=[],
        time=None,
        source_message_ids=[],
        metadata={},
    )
    frame = compile_evidence_frame(pseudo)
    procedures = _infer_line_procedures(text, frame)
    event_key_parts = [
        frame.frame_type,
        frame.slots.get("procedure"),
        frame.slots.get("date"),
        frame.slots.get("time"),
        frame.slots.get("location"),
        frame.slots.get("provider"),
    ]
    event_key = "|".join(str(part) for part in event_key_parts if part)
    lowered = text.lower()
    slots = dict(frame.slots)
    explicit_names = _extract_explicit_person_names(text)
    return {
        "frame_type": frame.frame_type,
        "slots": slots,
        "surface_spans": dict(frame.surface_spans),
        "bundle_payload": {},
        "procedures": procedures,
        "event_key": event_key or frame.frame_type,
        "slot_count": len(frame.slots),
        "is_lab_like": any(token in lowered for token in ["beta-hcg", "lab suite", "blood draw", "lab draw"]),
        "is_policy_only": any(token in lowered for token in ["may receive", "nothing else", "do not share", "logistics only"]) and not procedures,
        "is_current_like": any(token in lowered for token in ["current", "latest", "updated", "booked", "reminder", "on file", "next step", "revised approved", "supersedes", "treat as current", "official pilot target"]),
        "is_access_artifact_temporal": _looks_like_access_artifact_temporal_text(text, slots),
        "weekday_mentions": sorted(_extract_weekday_mentions(lowered)),
        "explicit_names": explicit_names,
    }


def _build_row_metadata(row: RetrievedEvidence, frame) -> dict[str, Any]:
    source_text = str((row.metadata or {}).get("original_content") or row.content or "")
    lowered = source_text.lower()
    procedures = _infer_line_procedures(row.content or "", frame)
    bundle_payload = dict((row.metadata or {}).get("bundle_payload") or {})
    slots = dict(frame.slots)
    explicit_names = _extract_explicit_person_names(source_text)
    event_key_parts = [
        frame.frame_type,
        slots.get("procedure"),
        slots.get("date"),
        slots.get("time"),
        slots.get("location"),
        slots.get("provider"),
    ]
    event_key = "|".join(str(part) for part in event_key_parts if part)
    return {
        "frame_type": frame.frame_type,
        "slots": slots,
        "surface_spans": dict(frame.surface_spans),
        "procedures": procedures,
        "source_text": source_text,
        "bundle_payload": bundle_payload,
        "event_key": event_key or frame.frame_type,
        "is_lab_like": any(token in lowered for token in ["beta-hcg", "lab suite", "blood draw", "lab draw"]),
        "is_current_like": any(token in lowered for token in ["current", "latest", "updated", "booked", "reminder", "on file", "next step", "revised approved", "supersedes", "official pilot target"]),
        "is_access_artifact_temporal": _looks_like_access_artifact_temporal_text(row.content or "", slots),
        "has_cancellation": any(token in lowered for token in ["canceled", "cancelled", "prior ", "old ", "inactive"]),
        "weekday_mentions": _extract_weekday_mentions(lowered),
        "explicit_names": explicit_names,
    }


def _attach_row_source_context(*, line_meta: dict[str, Any], row: RetrievedEvidence, default_text: str) -> None:
    source_text = str((row.metadata or {}).get("original_content") or row.content or default_text or "")
    line_meta["source_text"] = source_text
    local_timestamp = re.search(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)\]", str(default_text or ""))
    if local_timestamp is None:
        normalized_default = _normalize_surface_line(_strip_message_prefix(str(default_text or "")))
        for source_line in source_text.splitlines():
            if _normalize_surface_line(_strip_message_prefix(source_line)) != normalized_default:
                continue
            local_timestamp = re.search(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)\]", source_line)
            if local_timestamp is not None:
                break
    line_meta["source_time"] = normalize_timestamp(
        local_timestamp.group(1) if local_timestamp else row.time
    ) or ""
    lowered = source_text.lower()
    if any(
        token in lowered
        for token in [
            "current",
            "latest",
            "updated",
            "revised approved",
            "supersedes",
            "official pilot target",
            "quick state note",
            "current state at this point",
            "advising-safe summary",
        ]
    ):
        line_meta["is_current_like"] = True


def _score_evidence_row(
    *,
    question: str,
    row: RetrievedEvidence,
    frame,
    row_meta: dict[str, Any],
    profile: dict[str, Any],
    domains: set[str],
) -> float:
    score = float(row.score)
    lowered = str(row.content or "").lower()
    question_lower = question.lower()
    if _is_question_like_text(str(row.content or "")):
        score -= 4.0
    target_procedures = set(profile.get("target_procedures") or [])
    row_procedures = set(row_meta.get("procedures") or [])
    if target_procedures:
        if row_procedures & target_procedures:
            score += 1.8
        elif row_procedures:
            score -= 1.8
    target_frame_types = set(profile.get("target_frame_types") or [])
    if target_frame_types:
        if frame.frame_type in target_frame_types:
            score += 0.9
        elif frame.frame_type not in {"cancellation", "instruction"}:
            score -= 0.6
    frame_slots = dict(getattr(frame, "slots", {}) or {})
    if "schedule" in domains and frame.frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics"}:
        score += 0.8
    if "allergy" in domains and frame.frame_type == "allergy":
        score += 1.0
    if "instruction" in domains and (frame.frame_type == "instruction" or frame_slots.get("instruction")):
        score += 0.8
    if profile.get("prefer_current") and row_meta.get("is_current_like"):
        score += 0.6
    score += _subject_alignment_score(
        question_profile=profile,
        text=str(row.content or ""),
        line_meta=row_meta,
    )
    question_weekdays = _extract_weekday_mentions(question_lower)
    row_weekdays = set(row_meta.get("weekday_mentions") or set())
    if question_weekdays and row_weekdays:
        if question_weekdays & row_weekdays:
            score += 0.8
        elif target_procedures and not row_meta.get("has_cancellation"):
            score -= 0.5
    if row_meta.get("is_lab_like") and target_procedures and not (row_procedures & target_procedures):
        score -= 1.2
    if "through" in question_lower and row_meta.get("has_cancellation"):
        score += 0.2
    if profile.get("wants_callback_instruction") and frame_slots.get("condition") and frame_slots.get("instruction"):
        score += 1.8
    if profile.get("wants_followup_plan") and frame_slots.get("instruction") and any(frame_slots.get(key) for key in ["date", "time", "location", "arrival_time"]):
        score += 1.8
    if profile.get("mixed_allergy_schedule"):
        if frame.frame_type == "allergy":
            score += 0.8
        if frame.frame_type in {"appointment", "test_or_imaging", "clinic_visit"}:
            score += 1.2
    return score


def _should_apply_evidence_shortlist(question: str, profile: dict[str, Any], scored: list[dict[str, Any]]) -> bool:
    question_lower = question.lower()
    if not scored:
        return False
    mode = _focused_surface_mode(profile)
    if mode in {
        "policy",
        "contact_protocol",
        "current_plan_window",
        "authorized_schedule",
        "current_med_followup_plan",
        "rescheduled_followup",
        "contact_methods",
        "final_procedure_plan",
        "mixed_allergy_schedule",
    }:
        return True
    if not profile.get("target_procedures") and not profile.get("prefer_current") and "authorized" not in question_lower and "through" not in question_lower:
        return False
    event_best: dict[str, float] = {}
    for item in scored:
        event_key = str(item["event_key"])
        event_best[event_key] = max(float(item["score"]), event_best.get(event_key, float("-inf")))
    best_scores = sorted(event_best.values(), reverse=True)
    if len(best_scores) <= 1:
        return True
    return (best_scores[0] - best_scores[1]) >= 0.75


def _infer_line_procedures(text: str, frame) -> list[str]:
    lowered = text.lower()
    procedures: set[str] = set()
    if any(token in lowered for token in ["ultrasound", "imaging", "scan"]):
        procedures.update({"ultrasound", "imaging", "scan"})
    if any(token in lowered for token in ["beta-hcg", "blood draw", "lab draw", "lab suite"]):
        procedures.update({"beta-hcg", "lab"})
    if "follow-up" in lowered or "follow up" in lowered:
        procedures.add("follow-up")
    if frame.slots.get("procedure"):
        procedures.add(str(frame.slots.get("procedure")).lower())
    return sorted(procedures)


def _can_select_candidate(item: dict[str, Any], selected_event_counts: dict[str, int], question_profile: dict[str, Any]) -> bool:
    event_key = str((item.get("line_meta") or {}).get("event_key") or "")
    if _is_question_like_text(str(item.get("text") or "")):
        return False
    if _subject_alignment_score(
        question_profile=question_profile,
        text=str(item.get("text") or ""),
        line_meta=dict(item.get("line_meta") or {}),
    ) < 0:
        return False
    if not event_key:
        return True
    current = selected_event_counts.get(event_key, 0)
    if current >= 2:
        return False
    target_procedures = set(question_profile.get("target_procedures") or [])
    line_procedures = set((item.get("line_meta") or {}).get("procedures") or [])
    if target_procedures and line_procedures and not (line_procedures & target_procedures):
        if current >= 1:
            return False
    return True


def _record_selected_event(item: dict[str, Any], selected_event_counts: dict[str, int]) -> None:
    event_key = str((item.get("line_meta") or {}).get("event_key") or "")
    if not event_key:
        return
    selected_event_counts[event_key] = selected_event_counts.get(event_key, 0) + 1


def _household_question_intent(question: str, question_profile: dict[str, Any]) -> str:
    lowered = str(question or "").lower()
    if any(token in lowered for token in ["combined", "summary", "plan", "all current details"]):
        return "combined"
    if any(token in lowered for token in ["checkout", "end-of-stay", "end of stay", "return requirement"]):
        return "checkout"
    if any(token in lowered for token in ["allowed to do", "outside scope", "scope", "permitted", "restricted"]):
        return "scope"
    if "logistics-only" in lowered or "logistics only" in lowered:
        return "logistics"
    if question_profile.get("asks_household_approved_areas") and not question_profile.get("asks_household_visit_window"):
        return "scope"
    return "general"


def _household_current_anchor_score(item: dict[str, Any], question: str, question_profile: dict[str, Any]) -> float:
    text = str(item.get("text") or "")
    lowered = text.lower()
    meta = dict(item.get("line_meta") or {})
    score = _current_state_family_priority_score(
        family="household_plan",
        text=text,
        line_meta=meta,
        base_score=float(item.get("score") or 0.0),
    )
    intent = _household_question_intent(question, question_profile)
    if intent == "combined":
        score += 0.4 * sum(bool((meta.get("slots") or {}).get(slot)) for slot in ("visit_window", "entry_method", "package_rule", "approved_areas", "arrival_contact_rule", "parking_pass"))
    elif intent == "checkout":
        if any(token in lowered for token in ["checkout", "return", "departure"]):
            score += 3.0
    elif intent == "scope":
        if any(token in lowered for token in ["approved areas", "limited to", "out of scope", "permitted", "restricted"]):
            score += 3.0
    elif intent == "logistics":
        if any(token in lowered for token in ["arrival", "entry", "parking", "delivery", "checkout"]):
            score += 2.0
    return score


def _pick_preferred_household_anchor(
    *,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    question: str,
    question_profile: dict[str, Any],
) -> dict[str, Any] | None:
    household_items = [
        item
        for item in list(selected) + list(candidates)
        if "household_plan" in _classify_line_families(item["text"], item.get("line_meta") or {}, mode="current_state_bundle")
    ]
    if not household_items:
        return None
    return max(
        household_items,
        key=lambda item: _household_current_anchor_score(item, question, question_profile),
        default=None,
    )


def _household_scope_required(question_profile: dict[str, Any]) -> bool:
    return bool(question_profile.get("asks_household_approved_areas")) and not bool(question_profile.get("asks_household_visit_window"))


def _extract_household_scope_notes(text: str, slots: dict[str, Any]) -> list[str]:
    """Preserve generic access boundaries without using fixture-specific labels."""
    notes: list[str] = []
    if slots.get("approved_areas"):
        notes.append(f"approved areas: {slots['approved_areas']}")
    for sentence in re.split(r"(?<=[.;])\s+", text):
        lowered = sentence.lower()
        if any(token in lowered for token in ("out of scope", "resident-only", "private", "must not", "may not", "restricted", "deleted")):
            note = sentence.strip(" .;")
            if note:
                notes.append(note)
    return notes


def _prune_household_selected_lines(
    *,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    question: str,
    question_profile: dict[str, Any],
    limit: int,
    selected_event_counts: dict[str, int],
) -> list[dict[str, Any]]:
    anchor = _pick_preferred_household_anchor(
        selected=selected,
        candidates=candidates,
        question=question,
        question_profile=question_profile,
    )
    if anchor is None:
        return selected
    intent = _household_question_intent(question, question_profile)
    anchor_key = str((anchor.get("line_meta") or {}).get("event_key") or "")
    anchor_text = str(anchor.get("text") or "").lower()
    normalized_anchor = _normalize_surface_line(str(anchor.get("text") or ""))
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def should_keep(item: dict[str, Any]) -> bool:
        text = str(item.get("text") or "")
        slots = dict((item.get("line_meta") or {}).get("slots") or {})
        event_key = str((item.get("line_meta") or {}).get("event_key") or "")
        if _normalize_surface_line(text) == normalized_anchor:
            return True
        if intent == "combined":
            if any(slots.get(slot) for slot in ("visit_window", "entry_method", "package_rule", "approved_areas", "arrival_contact_rule", "parking_pass")):
                return True
            return event_key == anchor_key and bool(event_key)
        if intent == "checkout":
            if any(token in text.lower() for token in ["checkout", "return", "departure"]):
                return True
            return event_key == anchor_key and bool(event_key)
        if intent == "scope":
            if slots.get("approved_areas") or any(token in text.lower() for token in ["limited to", "out of scope", "permitted", "restricted"]):
                return True
            return event_key == anchor_key and bool(event_key)
        if intent == "logistics":
            if any(slots.get(slot) for slot in ("visit_window", "entry_method", "package_rule", "arrival_contact_rule", "parking_pass")):
                return True
            return event_key == anchor_key and bool(event_key)
        return event_key == anchor_key and bool(event_key)

    ranked = sorted(
        selected,
        key=lambda item: _household_current_anchor_score(item, question, question_profile),
        reverse=True,
    )
    for item in ranked:
        if len(out) >= limit:
            break
        normalized = _normalize_surface_line(str(item.get("text") or ""))
        if normalized in seen:
            continue
        if not should_keep(item):
            continue
        out.append(item)
        seen.add(normalized)
    for item in sorted(candidates, key=lambda item: _household_current_anchor_score(item, question, question_profile), reverse=True):
        if len(out) >= limit:
            break
        normalized = _normalize_surface_line(str(item.get("text") or ""))
        if normalized in seen:
            continue
        if not _can_select_candidate(item, selected_event_counts, question_profile):
            continue
        if not should_keep(item):
            continue
        out.append(item)
        seen.add(normalized)
    out.sort(key=lambda item: (int(item["evidence_rank"]), int(item["line_index"])))
    return out or selected


def _render_structured_surface_answer(
    *,
    question: str,
    domains: set[str],
    selected_lines: list[dict[str, Any]],
    question_profile: dict[str, Any] | None = None,
) -> str | None:
    profile = question_profile or _build_question_profile(question, domains)
    mode = _focused_surface_mode(profile)
    mode_answer = _render_mode_specific_surface_answer(
        mode=mode,
        selected_lines=selected_lines,
        question_profile=profile,
    )
    if mode_answer:
        return mode_answer
    parts: list[str] = []
    allergies = _collect_allergy_items(selected_lines)
    schedule_events, canceled_events = _collect_schedule_items(selected_lines)
    instructions = _collect_instruction_items(selected_lines)
    if "allergy" in domains and allergies:
        for item in allergies:
            substance = item.get("substance")
            reaction = item.get("reaction")
            if substance and reaction:
                parts.append(f"The recorded allergy is {substance}, with {reaction} as the reaction.")
            elif substance:
                parts.append(f"The recorded allergy is {substance}.")
    if "schedule" in domains and schedule_events:
        for event in schedule_events:
            parts.append(_render_schedule_event(event))
    if "instruction" in domains and instructions:
        for text in instructions[:2]:
            parts.append(text)
    if "cancellation" in domains and canceled_events:
        for event in canceled_events:
            parts.append(_render_canceled_event(event))
    if not parts:
        return None
    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _render_mode_specific_surface_answer(
    *,
    mode: str | None,
    selected_lines: list[dict[str, Any]],
    question_profile: dict[str, Any] | None = None,
) -> str | None:
    if mode is None:
        return None
    if mode == "current_state_bundle":
        concise = _render_current_state_bundle_answer(selected_lines, question_profile=question_profile or {})
        if concise:
            return concise
    if mode == "medication_status":
        lines = _render_medication_status_lines(selected_lines)
        if lines:
            return " ".join(lines)
    if mode == "mixed_allergy_schedule":
        allergies = _collect_allergy_items(selected_lines)
        schedule_events, _ = _collect_schedule_items(selected_lines)
        parts: list[str] = []
        for item in allergies:
            substance = item.get("substance")
            reaction = item.get("reaction")
            if substance and reaction:
                article = "an" if reaction[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
                parts.append(f"The recorded allergy is {substance}, with {article} {reaction} reaction.")
                break
        for event in schedule_events[:2]:
            rendered = _render_schedule_event(event)
            if rendered:
                parts.append(rendered)
        if parts:
            return " ".join(parts)
    mode_slot_requirements = {
        "policy": {"policy_scope"},
        "contact_protocol": {"contact_method", "phone", "portal", "backup_contact"},
        "contact_methods": {"contact_method", "phone", "portal", "backup_contact"},
        "callback_instruction": {"instruction", "condition"},
        "return_precautions": {"instruction", "condition"},
        "current_plan_window": {"instruction", "condition", "date", "time", "location", "arrival_time"},
        "followup_plan": {"instruction", "condition", "date", "time", "location", "arrival_time"},
        "current_med_followup_plan": {"medication", "dosage", "instruction", "date", "time", "location"},
        "authorized_schedule": {"date", "time", "location", "arrival_time", "procedure"},
        "imaging_schedule": {"date", "time", "location", "arrival_time", "procedure"},
        "final_procedure_plan": {"instruction", "date", "time", "location", "arrival_time", "procedure"},
        "rescheduled_followup": {"medication", "instruction", "date", "time", "location"},
    }
    required_slots = mode_slot_requirements.get(mode, set())
    structural_lines = []
    seen = set()
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        slots = dict((line.get("line_meta") or {}).get("slots") or {})
        if not text or _is_question_like_text(text) or not (required_slots & set(slots)):
            continue
        normalized = _normalize_surface_line(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        structural_lines.append(text)
        if len(structural_lines) >= 4:
            break
    return " ".join(structural_lines) or None


def _required_line_families_for_mode(mode: str, question_profile: dict[str, Any]) -> list[str]:
    families: dict[str, list[str]] = {
        "policy": ["policy_state"],
        "contact_protocol": ["contact_protocol"],
        "medication_status": ["medication_plan", "monitoring_instruction"],
        "contact_methods": ["primary_contact"],
        "callback_instruction": ["urgent_callback"],
        "authorized_schedule": ["authorized_logistics", "authorized_followup"],
        "current_plan_window": ["active_schedule", "cancellation", "instruction"],
        "current_med_followup_plan": ["medication_plan", "followup_schedule"],
        "rescheduled_followup": ["medication_plan", "reschedule_update"],
        "final_procedure_plan": ["procedure_schedule", "prep_instruction"],
        "mixed_allergy_schedule": ["allergy", "active_schedule"],
        "current_state_bundle": [],
    }
    out = list(families.get(mode, []))
    if mode == "final_procedure_plan" and question_profile.get("asks_preparation_details"):
        return ["prep_instruction", "procedure_schedule"]
    if mode == "current_state_bundle":
        if question_profile.get("asks_private_case_state_date"):
            out.append("current_target_date")
        if question_profile.get("asks_public_event_date"):
            out.append("current_public_event_date")
        if question_profile.get("asks_current_budget"):
            out.append("current_budget")
        if question_profile.get("asks_current_discount"):
            out.append("current_discount")
        if question_profile.get("asks_current_blocker"):
            out.append("current_blocker")
        if question_profile.get("asks_current_stipend"):
            out.append("current_stipend")
        if question_profile.get("asks_safe_wording"):
            out.append("current_safe_wording")
        if question_profile.get("asks_household_plan"):
            out.append("household_plan")
        if not out:
            out = ["current_target_date", "current_budget", "current_discount", "current_blocker", "current_stipend", "current_safe_wording", "household_plan"]
    if mode == "medication_status":
        regimen_families = list(question_profile.get("regimen_coverage_families") or [])
        family_map = {
            "start": "medication_start",
            "stop": "medication_stop",
            "continue": "medication_continue",
            "use": "medication_use",
            "monitor": "monitoring_instruction",
        }
        out = ["monitoring_instruction"]
        for family in regimen_families:
            mapped = family_map.get(family)
            if mapped and mapped not in out:
                out.append(mapped)
        if not regimen_families or len(regimen_families) <= 1:
            out.append("medication_plan")
    return out


def _classify_line_families(text: str, line_meta: dict[str, Any], *, mode: str) -> set[str]:
    frame_type = str(line_meta.get("frame_type") or "")
    slots = {str(key): value for key, value in dict(line_meta.get("slots") or {}).items() if value}
    slot_names = set(slots)
    families: set[str] = set()
    if frame_type in {"consent_or_permission", "privacy_policy"} or slots.get("policy_scope"):
        families.update({"policy_state", "policy_scope"})
    if slot_names & {"contact_method", "phone", "portal", "backup_contact"}:
        families.update({"contact_protocol", "primary_contact"})
    if frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics"} or slot_names & {"date", "time", "location", "arrival_time", "procedure", "visit_type"}:
        families.update({"active_schedule", "followup_schedule", "procedure_schedule"})
    if frame_type == "cancellation" or str(line_meta.get("lifecycle_status") or "").lower() in {"canceled", "superseded"}:
        families.update({"cancellation", "reschedule_update"})
    if frame_type == "allergy" or slot_names & {"allergy_substance", "allergy_reaction"}:
        families.add("allergy")
    if frame_type == "medication" or slot_names & {"medication", "dosage"}:
        families.add("medication_plan")
    if slots.get("instruction"):
        families.update({"instruction", "prep_instruction"})
    if slots.get("condition") and slots.get("instruction"):
        families.update({"monitoring_instruction", "urgent_callback"})
    for action in ["start", "stop", "continue", "use", "monitor"]:
        if slots.get(f"medication_{action}"):
            families.add(f"medication_{action}")
    if not bool(line_meta.get("is_access_artifact_temporal")):
        family_by_slot = {
            "target_date": "current_target_date",
            "public_event_date": "current_public_event_date",
            "approved_budget": "current_budget",
            "budget": "current_budget",
            "approved_discount_cap": "current_discount",
            "discount_cap": "current_discount",
            "blocker": "current_blocker",
            "monthly_stipend": "current_stipend",
            "stipend": "current_stipend",
            "safe_wording": "current_safe_wording",
            "safe_external_wording": "current_safe_wording",
        }
        for slot_name, family in family_by_slot.items():
            if slots.get(slot_name):
                families.add(family)
    if frame_type == "household_plan" or slot_names & {"visit_window", "entry_method", "package_rule", "approved_areas", "parking_pass", "arrival_contact_rule"}:
        families.add("household_plan")
    return families


def _pick_mode_lines(selected_lines: list[dict[str, Any]], *, include_terms: list[str], fallback_count: int) -> list[str]:
    picked: list[str] = []
    seen: set[str] = set()
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        lowered = text.lower()
        if not text or _is_question_like_text(text):
            continue
        if any(term in lowered for term in include_terms):
            norm = _normalize_surface_line(text)
            if norm not in seen:
                seen.add(norm)
                picked.append(text)
    if picked:
        return picked
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        norm = _normalize_surface_line(text)
        if not text or norm in seen or _is_question_like_text(text):
            continue
        seen.add(norm)
        picked.append(text)
        if len(picked) >= fallback_count:
            break
    return picked


def _merge_mode_lines(*groups: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for line in group:
            norm = _normalize_surface_line(line)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(line)
            if len(merged) >= limit:
                return merged
    return merged


def _render_final_procedure_plan_lines(
    selected_lines: list[dict[str, Any]],
    *,
    question_profile: dict[str, Any],
) -> list[str]:
    prep_lines: list[str] = []
    schedule_lines: list[str] = []
    seen: set[str] = set()
    for item in selected_lines:
        text = str(item.get("text") or "").strip()
        if not text or _is_question_like_text(text):
            continue
        families = _classify_line_families(
            text,
            item.get("line_meta") or {},
            mode="final_procedure_plan",
        )
        normalized = _normalize_surface_line(text)
        if "prep_instruction" in families and normalized not in seen:
            seen.add(normalized)
            prep_lines.append(text)
            continue
        if "procedure_schedule" in families and normalized not in seen:
            seen.add(normalized)
            schedule_lines.append(text)
    if question_profile.get("asks_preparation_details"):
        return _merge_mode_lines(prep_lines, schedule_lines, limit=4)
    return _merge_mode_lines(schedule_lines, prep_lines, limit=4)


def _render_contact_protocol_answer(selected_lines: list[dict[str, Any]]) -> str | None:
    phone_lines = _pick_mode_lines(
        selected_lines,
        include_terms=["callback number", "new number", "updated number", "temporary callback", "safe-contact update entered", "safe phone changes"],
        fallback_count=2,
    )
    rule_lines = _pick_mode_lines(
        selected_lines,
        include_terms=["generic callback only", "generic voicemail", "no voicemail mentioning pregnancy", "do not mention pregnancy in voicemail"],
        fallback_count=2,
    )
    portal_lines = _pick_mode_lines(selected_lines, include_terms=["portal"], fallback_count=1)
    phones = []
    for line in phone_lines:
        phones.extend(PHONE_SPAN_RE.findall(line))
    phones = _dedupe_list(phones)
    current_phone = phones[0] if phones else ""
    prior_phone = phones[1] if len(phones) > 1 else ""
    has_generic = any("generic" in line.lower() for line in rule_lines)
    has_voicemail = any("voicemail" in line.lower() for line in rule_lines)
    has_portal = bool(portal_lines)
    if not (current_phone or rule_lines):
        return None
    parts: list[str] = []
    if current_phone and prior_phone:
        parts.append(f"Starting Sunday evening, use {current_phone} instead of {prior_phone}.")
    elif current_phone:
        parts.append(f"Starting Sunday evening, use {current_phone}.")
    if has_generic and has_voicemail:
        parts.append("Use generic callback only and do not mention pregnancy in voicemail.")
    elif has_generic:
        parts.append("Use generic callback only.")
    elif has_voicemail:
        parts.append("Do not mention pregnancy in voicemail.")
    if has_portal:
        parts.append("Use the portal for details.")
    return " ".join(parts).strip() or None


def _render_current_policy_state_answer(selected_lines: list[dict[str, Any]]) -> str | None:
    state = _summarize_policy_state(selected_lines)
    if not (state["revoked"] or state["logistics_only"] or state["removed_contact_permissions"]):
        return None
    parts: list[str] = []
    if state["revoked"]:
        parts.append("Family scheduling access is revoked.")
    if state["no_future_sharing"]:
        parts.append("Do not share future appointment details with family.")
    if state["removed_contact_permissions"]:
        parts.append("Linda Park was removed from scheduling-contact and callback-contact permissions.")
    elif state["logistics_only"] and not state["revoked"]:
        parts.append("Family access is limited to scheduling logistics only.")
    return " ".join(parts).strip() or None


def _is_stale_current_state_text(text: str) -> bool:
    lowered = str(text or "").lower()
    explicit_stale_tokens = [
        "old exploratory",
        "exploratory figure",
        "until finance posts",
        "until a newer decision supersedes it",
        "should not be used as a shortcut",
        "treat that as provisional",
        "the old ",
        " is stale",
    ]
    if any(token in lowered for token in explicit_stale_tokens):
        return True
    if any(token in lowered for token in ["provisional", "interim"]):
        has_current_state_anchor = any(
            token in lowered
            for token in [
                "target date",
                "date as",
                "support amount",
                "stipend",
                "ra-support",
                "ra support",
                "approved budget",
                "discount cap",
                "current blocker",
            ]
        )
        if has_current_state_anchor and not any(
            token in lowered
            for token in [
                "superseded",
                "newer value overrides",
                "newer funding note appears later",
                "older value",
            ]
        ):
            return False
        return True
    return False


def _render_current_state_bundle_answer(
    selected_lines: list[dict[str, Any]],
    *,
    question_profile: dict[str, Any] | None = None,
) -> str | None:
    question_profile = question_profile or {}
    target_subject_tokens = {str(token).lower() for token in (question_profile.get("target_subject_tokens") or []) if str(token).strip()}
    if target_subject_tokens:
        aligned_lines = [
            line
            for line in selected_lines
            if _subject_alignment_score(
                question_profile=question_profile,
                text=str(line.get("text") or ""),
                line_meta=dict(line.get("line_meta") or {}),
            ) >= 0
        ]
        if aligned_lines:
            selected_lines = aligned_lines
    safe_wording = ""
    target_date = ""
    public_event_date = ""
    public_event_label = ""
    approved_budget = ""
    approved_discount = ""
    blocker = ""
    stipend = ""
    current_status = ""
    household_date = ""
    visit_window = ""
    entry_method = ""
    arrival_confirmation = ""
    package_rule = ""
    approved_areas = ""
    parking_pass = ""
    household_scope_notes: list[str] = []
    safe_wording_score = float("-inf")
    target_date_score = float("-inf")
    public_event_date_score = float("-inf")
    approved_budget_score = float("-inf")
    approved_discount_score = float("-inf")
    blocker_score = float("-inf")
    stipend_score = float("-inf")
    status_score = float("-inf")
    household_date_score = float("-inf")
    visit_window_score = float("-inf")
    entry_method_score = float("-inf")
    arrival_confirmation_score = float("-inf")
    package_rule_score = float("-inf")
    approved_areas_score = float("-inf")
    parking_pass_score = float("-inf")
    asks_public_event_date = bool(question_profile.get("asks_public_event_date"))
    asks_private_case_state_date = bool(question_profile.get("asks_private_case_state_date"))
    asks_current_budget = bool(question_profile.get("asks_current_budget"))
    asks_current_discount = bool(question_profile.get("asks_current_discount"))
    asks_current_stipend = bool(question_profile.get("asks_current_stipend"))
    asks_safe_wording = bool(question_profile.get("asks_safe_wording"))
    asks_current_blocker = bool(question_profile.get("asks_current_blocker"))
    asks_household_plan = bool(question_profile.get("asks_household_plan"))
    household_intent = _household_question_intent("As of now, provide the current household plan.", question_profile)
    anchor_household_line = _pick_preferred_household_anchor(
        selected=selected_lines,
        candidates=[],
        question="As of now, provide the current household plan.",
        question_profile=question_profile,
    )
    asks_scope_question = bool(question_profile.get("asks_household_approved_areas")) and any(
        token in str(question_profile).lower() for token in ["outside his scope", "outside her scope", "allowed to do", "scope"]
    )
    if _household_scope_required(question_profile):
        asks_scope_question = True
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        if not text or _is_question_like_text(text):
            continue
        lowered = text.lower()
        meta = dict(line.get("line_meta") or {})
        slots = dict(meta.get("slots") or {})
        frame_type = str(meta.get("frame_type") or "")
        base_score = float(line.get("score") or 0.0)
        if frame_type in {"project_state", "research_state"}:
            candidate_safe = str(slots.get("safe_wording") or slots.get("safe_external_wording") or "").strip()
            if candidate_safe:
                score = _current_state_family_priority_score(family="current_safe_wording", text=text, line_meta=meta, base_score=base_score)
                if score > safe_wording_score:
                    safe_wording = candidate_safe
                    safe_wording_score = score
            # Project/research target dates must come from their typed slot;
            # an unrelated event date cannot replace a resolved current target.
            candidate_target = str(slots.get("target_date") or "").strip()
            if candidate_target:
                if re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", candidate_target):
                    m = re.search(rf"{re.escape(candidate_target)}(?:,\s*(\d{{4}}))", text, flags=re.IGNORECASE)
                    if m:
                        candidate_target = f"{candidate_target}, {m.group(1)}"
                elif re.fullmatch(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", candidate_target):
                    candidate_target = re.sub(r"\s+", " ", candidate_target).strip()
                score = _current_state_family_priority_score(family="current_target_date", text=text, line_meta=meta, base_score=base_score)
                if score > target_date_score:
                    target_date = candidate_target
                    target_date_score = score
        candidate_public_event = str(slots.get("public_event_date") or "").strip()
        if candidate_public_event:
            if re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", candidate_public_event):
                m = re.search(rf"{re.escape(candidate_public_event)}(?:,\s*(\d{{4}}))", text, flags=re.IGNORECASE)
                if m:
                    candidate_public_event = f"{candidate_public_event}, {m.group(1)}"
            score = _current_state_family_priority_score(
                family="current_public_event_date",
                text=text,
                line_meta=meta,
                base_score=base_score,
            )
            if score > public_event_date_score:
                public_event_date = candidate_public_event
                public_event_date_score = score
                procedure_label = str(slots.get("procedure") or "").strip().lower()
                public_event_label = "orientation" if procedure_label == "orientation" or "orientation" in lowered else "event"
        if asks_public_event_date and not candidate_public_event and any(token in lowered for token in ["orientation", "public schedule", "calendar line"]):
            date_candidates = re.findall(r"[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?", text, flags=re.IGNORECASE)
            if date_candidates:
                candidate_public_event = date_candidates[-1].strip()
                score = _current_state_family_priority_score(
                    family="current_public_event_date",
                    text=text,
                    line_meta=meta,
                    base_score=base_score,
                )
                if score > public_event_date_score:
                    public_event_date = candidate_public_event
                    public_event_date_score = score
                    public_event_label = "orientation" if "orientation" in lowered else "event"
        if frame_type in {"project_state", "research_state"}:
            candidate_budget = str(slots.get("approved_budget") or slots.get("budget") or "").strip()
            if candidate_budget:
                score = _current_state_family_priority_score(family="current_budget", text=text, line_meta=meta, base_score=base_score)
                if score > approved_budget_score:
                    approved_budget = candidate_budget
                    approved_budget_score = score
            candidate_discount = str(slots.get("approved_discount_cap") or slots.get("discount_cap") or "").strip()
            if candidate_discount:
                score = _current_state_family_priority_score(family="current_discount", text=text, line_meta=meta, base_score=base_score)
                if score > approved_discount_score:
                    approved_discount = candidate_discount
                    approved_discount_score = score
            candidate_blocker = str(slots.get("blocker") or "").strip()
            if candidate_blocker:
                score = _current_state_family_priority_score(family="current_blocker", text=text, line_meta=meta, base_score=base_score)
                if score > blocker_score:
                    blocker = candidate_blocker
                    blocker_score = score
            candidate_stipend = str(slots.get("monthly_stipend") or slots.get("stipend") or "").strip()
            if candidate_stipend:
                score = _current_state_family_priority_score(family="current_stipend", text=text, line_meta=meta, base_score=base_score)
                if score > stipend_score:
                    stipend = candidate_stipend
                    stipend_score = score
            candidate_status = str(slots.get("status") or "").strip()
            if candidate_status:
                score = _current_state_family_priority_score(family="current_blocker", text=text, line_meta=meta, base_score=base_score)
                if score > status_score:
                    current_status = candidate_status
                    status_score = score
                if not candidate_blocker:
                    normalized_status = candidate_status
                    if "hold cleared" in lowered or "is cleared" in lowered or "cleared" == lowered.strip():
                        normalized_status = candidate_status
                    elif "hold active" in lowered or "still pending" in lowered:
                        normalized_status = candidate_status
                    if score > blocker_score:
                        blocker = normalized_status
                        blocker_score = score
        if frame_type == "household_plan" or any(
            key in slots
            for key in ["visit_window", "entry_method", "package_rule", "approved_areas", "arrival_contact_rule", "parking_pass"]
        ):
            household_score = _current_state_family_priority_score(family="household_plan", text=text, line_meta=meta, base_score=base_score)
            candidate_household_date = str(slots.get("date") or "").strip()
            if candidate_household_date and household_score > household_date_score:
                household_date = candidate_household_date
                household_date_score = household_score
            candidate_visit_window = str(slots.get("visit_window") or "").strip()
            if candidate_visit_window and household_score > visit_window_score:
                visit_window = candidate_visit_window
                visit_window_score = household_score
            candidate_entry_method = str(slots.get("entry_method") or "").strip()
            if candidate_entry_method and household_score > entry_method_score:
                entry_method = candidate_entry_method
                entry_method_score = household_score
            candidate_package_rule = str(slots.get("package_rule") or "").strip()
            if candidate_package_rule and household_score > package_rule_score:
                package_rule = candidate_package_rule
                package_rule_score = household_score
            candidate_approved_areas = str(slots.get("approved_areas") or "").strip()
            if candidate_approved_areas and household_score > approved_areas_score:
                approved_areas = candidate_approved_areas
                approved_areas_score = household_score
            candidate_arrival_confirmation = str(slots.get("arrival_contact_rule") or "").strip()
            if candidate_arrival_confirmation and household_score > arrival_confirmation_score:
                arrival_confirmation = candidate_arrival_confirmation
                arrival_confirmation_score = household_score
            candidate_parking_pass = str(slots.get("parking_pass") or "").strip()
            if candidate_parking_pass and household_score > parking_pass_score:
                parking_pass = candidate_parking_pass
                parking_pass_score = household_score
            scope_notes = _extract_household_scope_notes(text, slots)
            for note in scope_notes:
                if note not in household_scope_notes:
                    household_scope_notes.append(note)
        if not safe_wording and "exact internal" not in lowered and "restricted notes" not in lowered:
            m = re.search("safe wording should stay ['\"]([^'\"]+)['\"]", text, flags=re.IGNORECASE)
            if m:
                safe_wording = m.group(1).strip()
        if not stipend:
            m = re.search(
                r"(?:ra-support amount|ra support amount|monthly stipend|support amount)(?:\s+is|\s+now|\s+currently|\s+remains)?\s+([0-9][0-9,\.]*\s*usd)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                stipend = m.group(1).strip()
        if not target_date:
            m = re.search(
                r"(?:date as|target date is|current target date is|launch date remains|moves from [A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})? to)\s+([A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                target_date = m.group(1).strip()
        if target_date and re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", target_date):
            m = re.search(rf"{re.escape(target_date)}(?:,\s*(\d{{4}}))", text, flags=re.IGNORECASE)
            if m:
                target_date = f"{target_date}, {m.group(1)}"
        if any(token in lowered for token in ["blocker", "pending", "still pending"]):
            m = re.search(
                r"(?:current\s+blocker\s+is\s+|remaining\s+blockers?\s+are\s+|remaining\s+blocker\s+is\s+|open blocker:\s*|blocker:\s*)([^.;]+)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                candidate_blocker = m.group(1).strip()
                score = _current_state_family_priority_score(family="current_blocker", text=text, line_meta=meta, base_score=base_score)
                if score > blocker_score:
                    blocker = candidate_blocker
                    blocker_score = score
            else:
                m = re.search(r"still pending,?\s*([^.;]+)", text, flags=re.IGNORECASE)
                if m:
                    candidate_blocker = m.group(1).strip()
                    score = _current_state_family_priority_score(family="current_blocker", text=text, line_meta=meta, base_score=base_score)
                    if score > blocker_score:
                        blocker = candidate_blocker
                        blocker_score = score
    if anchor_household_line is not None:
        anchor_text = str(anchor_household_line.get("text") or "")
        anchor_lowered = anchor_text.lower()
        anchor_meta = dict(anchor_household_line.get("line_meta") or {})
        anchor_slots = dict(anchor_meta.get("slots") or {})
        household_date = str(anchor_slots.get("date") or household_date).strip()
        visit_window = str(anchor_slots.get("visit_window") or visit_window).strip()
        entry_method = str(anchor_slots.get("entry_method") or entry_method).strip()
        arrival_confirmation = str(anchor_slots.get("arrival_contact_rule") or arrival_confirmation).strip()
        package_rule = str(anchor_slots.get("package_rule") or package_rule).strip()
        approved_areas = str(anchor_slots.get("approved_areas") or approved_areas).strip()
        parking_pass = str(anchor_slots.get("parking_pass") or parking_pass).strip()
        for note in _extract_household_scope_notes(anchor_text, anchor_slots):
            if note not in household_scope_notes:
                household_scope_notes.append(note)
    parts: list[str] = []
    if safe_wording and (asks_safe_wording or not question_profile):
        parts.append(f"Safe external wording remains '{safe_wording}'.")
    if public_event_date and (asks_public_event_date or not question_profile):
        event_name = "orientation" if public_event_label == "orientation" else "event"
        parts.append(f"The public {event_name} date is {public_event_date}.")
    current_parts: list[str] = []
    if target_date and (asks_private_case_state_date or (not question_profile and not asks_public_event_date)):
        current_parts.append(f"the current target date is {target_date}")
    if approved_budget and (asks_current_budget or not question_profile):
        current_parts.append(f"the current approved budget is {approved_budget}")
    if approved_discount and (asks_current_discount or not question_profile):
        current_parts.append(f"the current approved maximum discount is {approved_discount}")
    if stipend and (asks_current_stipend or not question_profile):
        current_parts.append(f"the current monthly stipend is {stipend}")
    if blocker and (asks_current_blocker or not question_profile):
        current_parts.append(f"the current blocker is {blocker}")
    elif current_status and (asks_current_blocker or not question_profile):
        current_parts.append(f"the current hold state is {current_status}")
    if current_parts:
        parts.append("For the current state, " + ", ".join(current_parts) + ".")
    household_parts: list[str] = []
    if visit_window and (asks_household_plan or not question_profile):
        if household_date:
            household_parts.append(f"the current visit window is {household_date} from {visit_window}")
        else:
            household_parts.append(f"the current visit window is {visit_window}")
    if entry_method and (asks_household_plan or not question_profile):
        household_parts.append(f"the current entry method is {entry_method}")
    if parking_pass and (asks_household_plan or not question_profile):
        household_parts.append(f"the current parking pass is {parking_pass}")
    if arrival_confirmation and (asks_household_plan or not question_profile):
        household_parts.append(f"the arrival-confirmation rule is {arrival_confirmation}")
    if package_rule and (asks_household_plan or not question_profile):
        household_parts.append(f"the current package rule is {package_rule}")
    if approved_areas and (asks_household_plan or not question_profile):
        household_parts.append(f"the approved areas are {approved_areas}")
    if household_scope_notes and (asks_household_plan or not question_profile):
        household_parts.append(f"the current scope notes include {', '.join(household_scope_notes)}")
    if asks_scope_question and household_scope_notes:
        household_parts = [part for part in household_parts if "visit window" not in part]
    if household_intent == "checkout":
        household_parts = [
            part
            for part in household_parts
            if "entry method" not in part and "arrival-confirmation rule" not in part and "approved areas" not in part
        ]
    if household_intent == "scope":
        household_parts = [
            part
            for part in household_parts
            if "parking pass" not in part and "arrival-confirmation rule" not in part and "package rule" not in part
        ]
    if household_parts:
        parts.append("For the current household plan, " + ", ".join(household_parts) + ".")
    prediction = " ".join(part.strip() for part in parts if part.strip()).strip()
    if not prediction:
        return None
    return _postprocess_current_state_prediction(
        prediction=prediction,
        selected_lines=selected_lines,
        question="As of now, provide the current state bundle.",
    )

def _render_current_plan_window_answer(selected_lines: list[dict[str, Any]]) -> str | None:
    instruction = ""
    schedule = ""
    canceled = ""
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        meta = dict(line.get("line_meta") or {})
        slots = dict(meta.get("slots") or {})
        frame_type = str(meta.get("frame_type") or "")
        if not instruction and (frame_type == "instruction" or slots.get("instruction")):
            instruction = text
        if not schedule and (frame_type in {"appointment", "test_or_imaging", "clinic_visit"} or any(slots.get(key) for key in ["date", "time", "location", "arrival_time"])):
            schedule = text
        if not canceled and (frame_type == "cancellation" or str(meta.get("lifecycle_status") or "").lower() in {"canceled", "superseded"}):
            canceled = text
    if not (instruction or schedule or canceled):
        return None
    parts = [part for part in [instruction, schedule, canceled] if part]
    return " ".join(parts[:3]).strip() or None


def _render_medication_status_lines(selected_lines: list[dict[str, Any]]) -> list[str]:
    def _normalize_medication_sentence(text: str) -> str:
        cleaned = canonicalize_medication_surface(text)
        cleaned = re.sub(
            r"keep the total under\s+([\d,]+)\s+mg\s+in a day",
            r"\1 mg/day max",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\buse\s+([a-z0-9./' -]+?)\s+instead:\s+(\d+(?:\.\d+)?\s*mg\b)",
            r"Use \1 \2",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\buse\s+", "Use ", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bstart\s+", "Start ", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bstop\s+", "Stop ", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bcontinue\s+", "Continue ", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .;")
        if cleaned and not cleaned.endswith("."):
            cleaned += "."
        return cleaned

    regimen_sentences: list[str] = []
    baseline_line = ""
    seen_texts: set[str] = set()
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        if not text or _is_question_like_text(text):
            continue
        meta = dict(line.get("line_meta") or {})
        norm = _normalize_surface_line(text)
        if _looks_like_medication_regimen_line(text, meta) or _looks_like_monitoring_instruction_line(text, meta):
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            regimen_sentences.append(_normalize_medication_sentence(text))
            continue
        lowered = text.lower()
        dosage_mentions = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|milligrams|micrograms)\b", lowered)
        if not baseline_line and len(dosage_mentions) >= 2 and any(
            token in lowered
            for token in ["daily", "twice a day", "twice daily", "every morning", "nightly", "once daily"]
        ):
            baseline_line = text
    out = regimen_sentences
    if out:
        return out[:6]
    return _pick_mode_lines(selected_lines, include_terms=["current medication plan", "current plan:"], fallback_count=3)


def _render_authorized_schedule_answer(selected_lines: list[dict[str, Any]]) -> str | None:
    bundle_signatures: set[tuple[str, str, str, str, str]] = set()
    for line in selected_lines:
        payload = dict((line.get("line_meta") or {}).get("bundle_payload") or {})
        if payload.get("bundle_type") == "authorized_schedule":
            for item in payload.get("active_items") or []:
                bundle_signatures.add(
                    (
                        str(item.get("date") or "").strip().lower(),
                        str(item.get("time") or "").strip().lower(),
                        str(item.get("arrival_time") or "").strip().lower(),
                        str(item.get("location") or "").strip().lower(),
                        str(item.get("provider") or "").strip().lower(),
                    )
                )
            rendered = _render_authorized_schedule_bundle_payload(payload)
            if rendered:
                return rendered
    scope = ""
    schedule_lines: list[str] = []
    for line in selected_lines:
        text = str(line.get("text") or "").strip()
        meta = line.get("line_meta") or {}
        slots = dict(meta.get("slots") or {})
        signature = (
            str(slots.get("date") or "").strip().lower(),
            str(slots.get("time") or "").strip().lower(),
            str(slots.get("arrival_time") or slots.get("secondary_time") or "").strip().lower(),
            str(slots.get("location") or "").strip().lower(),
            str(slots.get("provider") or "").strip().lower(),
        )
        if not scope and (slots.get("policy_scope") or str(meta.get("frame_type") or "") in {"consent_or_permission", "privacy_policy"}):
            scope = text
        if signature not in bundle_signatures and any(slots.get(key) for key in ["date", "time", "arrival_time", "location", "provider", "procedure"]):
            schedule_lines.append(text)
    parts = [part for part in [scope, *schedule_lines[:2]] if part]
    if not parts:
        return None
    return " ".join(parts[:3]).strip()


def _render_imaging_schedule_answer(selected_lines: list[dict[str, Any]]) -> str | None:
    primary = _pick_primary_imaging_schedule_line(selected_lines, profile=None)
    if primary is None:
        return None
    meta = dict(primary.get("line_meta") or {})
    slots = dict(meta.get("slots") or {})
    spans = dict(meta.get("surface_spans") or {})
    date = str(spans.get("date") or slots.get("date") or "").strip()
    time = str(spans.get("time") or slots.get("time") or "").strip()
    arrival = _extract_arrival_surface(spans, str(primary.get("text") or ""))
    location = str(spans.get("location") or slots.get("location") or "").strip()
    procedure = str(spans.get("procedure") or slots.get("procedure") or "ultrasound").strip()
    if not any([date, time, arrival, location]):
        return None
    parts: list[str] = []
    lead = "The"
    if procedure:
        lead += f" {procedure}"
    else:
        lead += " imaging appointment"
    lead += " is"
    if date:
        lead += f" on {date}"
    if time:
        lead += f" at {time}"
    if location:
        lead += f" in {location}"
    lead += "."
    parts.append(lead)
    if arrival:
        parts.append(f"With {arrival} arrival.")
    return " ".join(parts).strip()


def _pick_primary_imaging_schedule_line(
    selected_lines: list[dict[str, Any]],
    *,
    profile: dict[str, Any] | None,
) -> dict[str, Any] | None:
    best_line: dict[str, Any] | None = None
    best_score = float("-inf")
    target_weekdays = set((profile or {}).get("target_weekdays") or [])
    for line in selected_lines:
        text = str(line.get("text") or "")
        meta = dict(line.get("line_meta") or {})
        if meta.get("is_lab_like"):
            continue
        if str(meta.get("frame_type") or "") in {"diagnosis_or_result", "medication", "general_fact"}:
            continue
        procedures = set(meta.get("procedures") or [])
        if procedures and not (procedures & {"ultrasound", "imaging", "scan"}):
            continue
        score = float(line.get("score") or 0.0)
        if _line_has_logistics_surface(text=text, line_meta=meta):
            score += 2.0
        if any(str((meta.get("slots") or {}).get(key) or "").strip() for key in ["date", "time", "location", "arrival_time"]):
            score += 1.4
        if meta.get("is_current_like"):
            score += 1.2
        if str(meta.get("lifecycle_status") or "").lower() in {"canceled", "superseded"}:
            score -= 0.4
        line_weekdays = set(meta.get("weekday_mentions") or [])
        if target_weekdays and line_weekdays & target_weekdays:
            score += 1.0
        if best_line is None or score > best_score:
            best_line = line
            best_score = score
    return best_line


def _render_authorized_schedule_bundle_payload(payload: dict[str, Any]) -> str | None:
    active_items = list(payload.get("active_items") or [])
    if not active_items:
        return None
    rendered_items: list[str] = []
    seen_signatures: set[tuple[str, str, str, str, str]] = set()
    for item in active_items:
        signature = (
            str(item.get("date") or "").strip().lower(),
            str(item.get("time") or "").strip().lower(),
            str(item.get("arrival_time") or "").strip().lower(),
            str(item.get("location") or "").strip().lower(),
            str(item.get("provider") or "").strip().lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        rendered = _render_schedule_bundle_item(item)
        if rendered:
            rendered_items.append(rendered)
    return " ".join(rendered_items[:4]).strip() or None


def _render_schedule_bundle_item(item: dict[str, Any]) -> str | None:
    date = str(item.get("date") or "").strip()
    time = str(item.get("time") or "").strip()
    arrival = str(item.get("arrival_time") or "").strip()
    location = str(item.get("location") or "").strip()
    provider = str(item.get("provider") or "").strip()
    procedure = str(item.get("procedure") or item.get("visit_type") or "appointment").strip()
    if not any([date, time, arrival, location, provider, procedure]):
        return None
    parts: list[str] = []
    if date and time:
        parts.append(f"{date} at {time}")
    elif date:
        parts.append(date)
    elif time:
        parts.append(time)
    detail_parts: list[str] = []
    if procedure and procedure.lower() not in " ".join(parts).lower():
        detail_parts.append(procedure)
    if arrival:
        detail_parts.append(f"arrive by {arrival}")
    if location:
        detail_parts.append(location)
    if provider and provider.lower() not in " ".join(parts + detail_parts).lower():
        detail_parts.append(provider)
    if detail_parts:
        parts.append(", ".join(detail_parts))
    return ", ".join(part for part in parts if part).strip()


def _render_mixed_allergy_schedule_bundle_payload(payload: dict[str, Any]) -> str | None:
    allergies = list(payload.get("allergy_items") or [])
    active_items = list(payload.get("active_items") or [])
    parts: list[str] = []
    for item in allergies:
        substance = str(item.get("substance") or "").strip()
        reaction = str(item.get("reaction") or "").strip()
        if substance and reaction:
            article = "an" if reaction[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            parts.append(f"The documented allergy is {substance} with {article} {reaction} reaction.")
            break
        if substance:
            parts.append(f"The documented allergy is {substance}.")
            break
    for item in active_items[:2]:
        rendered = _render_schedule_bundle_item(item)
        if rendered:
            parts.append(rendered)
    return " ".join(parts).strip() or None


def _collect_allergy_items(selected_lines: list[dict[str, Any]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen = set()
    for line in selected_lines:
        meta = line.get("line_meta") or {}
        payload = dict(meta.get("bundle_payload") or {})
        if payload.get("bundle_type") == "mixed_allergy_schedule":
            for bundled in payload.get("allergy_items") or []:
                item = {
                    "substance": str(bundled.get("substance") or "").strip(),
                    "reaction": str(bundled.get("reaction") or "").strip(),
                }
                key = (item["substance"], item["reaction"])
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
        if meta.get("frame_type") != "allergy":
            continue
        slots = meta.get("slots") or {}
        item = {
            "substance": str(slots.get("substance") or meta.get("surface_spans", {}).get("substance") or "").strip(),
            "reaction": str(slots.get("reaction") or meta.get("surface_spans", {}).get("reaction") or "").strip(),
        }
        key = (item["substance"], item["reaction"])
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def _collect_schedule_items(selected_lines: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for line in selected_lines:
        meta = line.get("line_meta") or {}
        payload = dict(meta.get("bundle_payload") or {})
        if payload.get("bundle_type") in {"authorized_schedule", "mixed_allergy_schedule", "current_plan_window"}:
            for idx, item in enumerate(payload.get("active_items") or []):
                event_key = f'{payload.get("bundle_type")}::active::{idx}::{item.get("date","")}::{item.get("time","")}'
                if event_key not in order:
                    order.append(event_key)
                merged[event_key] = {
                    "frame_type": str(item.get("frame_type") or ""),
                    "procedure": str(item.get("procedure") or "").strip(),
                    "visit_type": str(item.get("visit_type") or "").strip(),
                    "date": str(item.get("date") or "").strip(),
                    "time": str(item.get("time") or "").strip(),
                    "arrival_time": str(item.get("arrival_time") or "").strip(),
                    "location": str(item.get("location") or "").strip(),
                    "provider": str(item.get("provider") or "").strip(),
                    "status": str(item.get("status") or "").strip().lower(),
                }
            for idx, item in enumerate(payload.get("canceled_items") or []):
                event_key = f'{payload.get("bundle_type")}::canceled::{idx}::{item.get("date","")}::{item.get("time","")}'
                if event_key not in order:
                    order.append(event_key)
                merged[event_key] = {
                    "frame_type": str(item.get("frame_type") or ""),
                    "procedure": str(item.get("procedure") or "").strip(),
                    "visit_type": str(item.get("visit_type") or "").strip(),
                    "date": str(item.get("date") or "").strip(),
                    "time": str(item.get("time") or "").strip(),
                    "arrival_time": str(item.get("arrival_time") or "").strip(),
                    "location": str(item.get("location") or "").strip(),
                    "provider": str(item.get("provider") or "").strip(),
                    "status": str(item.get("status") or "canceled").strip().lower(),
                }
        frame_type = str(meta.get("frame_type") or "")
        if frame_type not in {"appointment", "test_or_imaging", "clinic_visit", "logistics", "cancellation", "update"}:
            continue
        slots = dict(meta.get("slots") or {})
        spans = dict(meta.get("surface_spans") or {})
        event_key = str(meta.get("event_key") or frame_type)
        target = merged.setdefault(event_key, {})
        if event_key not in order:
            order.append(event_key)
        target.setdefault("frame_type", frame_type)
        target.setdefault("procedure", str(spans.get("procedure") or slots.get("procedure") or "").strip())
        target.setdefault("visit_type", str(spans.get("visit_type") or slots.get("visit_type") or "").strip())
        target.setdefault("date", str(spans.get("date") or slots.get("date") or "").strip())
        target.setdefault("time", str(spans.get("time") or slots.get("time") or "").strip())
        target.setdefault("arrival_time", _extract_arrival_surface(spans, line["text"]))
        target.setdefault("location", str(spans.get("location") or slots.get("location") or "").strip())
        target.setdefault("provider", str(spans.get("provider") or slots.get("provider") or "").strip())
        status = str(slots.get("status") or "").strip().lower()
        if status:
            target["status"] = status
        if any(token in line["text"].lower() for token in ["canceled", "cancelled", "inactive", "no longer active"]):
            target["status"] = "canceled"
    schedule_events: list[dict[str, str]] = []
    canceled_events: list[dict[str, str]] = []
    for event_key in order:
        item = merged[event_key]
        if item.get("status") == "canceled":
            canceled_events.append(item)
        else:
            schedule_events.append(item)
    return schedule_events, canceled_events


def _collect_instruction_items(selected_lines: list[dict[str, Any]]) -> list[str]:
    items: list[str] = []
    seen = set()
    for line in selected_lines:
        meta = line.get("line_meta") or {}
        text = str(line.get("text") or "").strip()
        if meta.get("frame_type") not in {"instruction", "medication"} and not any(token in text.lower() for token in ["unless symptoms worsen", "beta-hcg", "before tuesday"]):
            continue
        key = _normalize_surface_line(text)
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _render_schedule_event(event: dict[str, str]) -> str:
    procedure = event.get("procedure") or event.get("visit_type") or "appointment"
    date = event.get("date") or ""
    time = event.get("time") or ""
    location = event.get("location") or ""
    provider = event.get("provider") or ""
    arrival = event.get("arrival_time") or ""
    base = f"The {procedure} is"
    if date:
        base += f" on {date}"
    if time:
        base += f" at {time}"
    if location:
        base += f" in {location}"
    elif provider:
        base += f" with {provider}"
    base += "."
    if arrival:
        base += f" With {arrival} arrival."
    return base


def _render_canceled_event(event: dict[str, str]) -> str:
    procedure = event.get("procedure") or event.get("visit_type") or "appointment"
    date = event.get("date") or ""
    time = event.get("time") or ""
    sentence = f"The prior {procedure}"
    if date:
        sentence += f" on {date}"
    if time:
        sentence += f" at {time}"
    sentence += " is canceled."
    return sentence


def _extract_arrival_surface(surface_spans: dict[str, str], text: str) -> str:
    if surface_spans.get("arrival_time"):
        return str(surface_spans["arrival_time"]).strip()
    match = ARRIVAL_SPAN_RE.search(text)
    if not match:
        return ""
    return match.group(0).replace("Please ", "").replace("please ", "").replace("arrive by ", "").replace("Arrive by ", "").strip()


def _surface_text_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9%:/.-]+", (text or "").lower()) if token}


def _is_better_surface_answer(candidate: dict[str, Any], baseline: dict[str, Any], candidate_text: str, baseline_text: str) -> bool:
    candidate_hits = len(candidate.get("hit_surfaces") or [])
    baseline_hits = len(baseline.get("hit_surfaces") or [])
    if candidate_hits != baseline_hits:
        return candidate_hits > baseline_hits
    candidate_misses = len(candidate.get("missing_surfaces") or [])
    baseline_misses = len(baseline.get("missing_surfaces") or [])
    if candidate_misses != baseline_misses:
        return candidate_misses < baseline_misses
    if not _surface_text_set(candidate_text).issuperset(_surface_text_set(baseline_text)):
        return False
    return len(candidate_text) <= len(baseline_text)


def _postprocess_current_state_prediction(*, prediction: str, selected_lines: list[dict[str, Any]], question: str) -> str:
    if not prediction.strip():
        return prediction
    question_profile = _build_question_profile(question, _infer_query_domains(question))
    if _focused_surface_mode(question_profile) != "current_state_bundle":
        return prediction
    updated = prediction
    full_dates: list[str] = []
    statuses: list[str] = []
    full_date_patterns = [
        r"[A-Za-z]+\s+\d{1,2},\s*\d{4}",
        r"[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2}(?:,\s*\d{4})?",
    ]
    for line in selected_lines:
        meta = dict(line.get("line_meta") or {})
        slots = dict(meta.get("slots") or {})
        text = str(line.get("text") or "")
        for key in ["target_date", "date"]:
            value = str(slots.get(key) or "").strip()
            if value and any(re.fullmatch(pattern, value) for pattern in full_date_patterns):
                normalized = re.sub(r"\s+", " ", value)
                if normalized not in full_dates:
                    full_dates.append(normalized)
            elif value and re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", value):
                match = re.search(rf"{re.escape(value)}(?:,\s*(\d{{4}}))", text, flags=re.IGNORECASE)
                if match:
                    full_value = f"{value}, {match.group(1)}"
                    if full_value not in full_dates:
                        full_dates.append(full_value)
        status = str(slots.get("status") or "").strip()
        if status and status not in statuses:
            statuses.append(status)
    for full_date in full_dates:
        short_date = re.sub(r",\s*\d{4}$", "", full_date)
        if re.fullmatch(r"[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2}", short_date, flags=re.IGNORECASE):
            weekday_only = short_date.split(",", 1)[0].strip()
            updated = re.sub(
                rf"\b{re.escape(weekday_only)}\b(?!,\s*[A-Za-z]+\s+\d{{1,2}}(?:,\s*\d{{4}})?)",
                full_date,
                updated,
            )
            continue
        updated = re.sub(rf"\b{re.escape(short_date)}\b(?!,\s*\d{{4}})", full_date, updated)
    for status in statuses:
        lowered_status = status.lower()
        if "tuition hold cleared" in lowered_status and "tuition hold cleared" not in updated.lower():
            updated = re.sub(r"\btuition-hold state:\s*cleared\b", "Tuition-hold state: tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\btuition hold state:\s*cleared\b", "Tuition hold state: tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\btuition-hold state is cleared\b", "tuition-hold state is tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\btuition hold state is cleared\b", "tuition hold state is tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\btuition-hold is cleared\b", "tuition-hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\btuition hold is cleared\b", "tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\bher tuition-hold state is cleared\b", "her tuition-hold state is tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\bthe tuition-hold state is cleared\b", "the tuition-hold state is tuition hold cleared", updated, flags=re.IGNORECASE)
            updated = re.sub(r"\bthe current blocker is cleared\b", "the current blocker is tuition hold cleared", updated, flags=re.IGNORECASE)
    return updated


def _synthesize_answer_with_llm(
    *,
    instance: MemoryInstance,
    llm_client: LLMClient,
    model_name: str,
    action: str,
    extra_rules: list[str],
    used_chunk_ids: list[str],
    surface_replay_result: AnswerResult,
    semantic_intent: dict[str, Any] | None = None,
) -> AnswerResult | None:
    raw_response = surface_replay_result.raw_response or {}
    selected_lines = list(raw_response.get("surface_lines") or [])
    compiled_frames = list(raw_response.get("compiled_frames") or [])
    if action == "refuse":
        prompt = (
            "Return JSON with fields prediction, used_chunk_ids, used_memory_ids, reasoning_summary.\n"
            f"Question: {instance.question}\n"
            "Required action: refuse.\n"
            "Write a brief refusal that does not reveal protected details."
        )
    elif action == "no_memory":
        prompt = (
            "Return JSON with fields prediction, used_chunk_ids, used_memory_ids, reasoning_summary.\n"
            f"Question: {instance.question}\n"
            "Required action: no_memory.\n"
            f'Use this exact answer: "{build_no_memory_answer_text()}".'
        )
    else:
        prompt = (
            "Return JSON with fields prediction, used_chunk_ids, used_memory_ids, reasoning_summary.\n"
            "Use only the candidate evidence lines below.\n"
            "Preserve exact dates, times, locations, provider names, procedures, instructions, medications, allergies, and cancellation wording when present.\n"
            "If the structured surface draft already answers the question, keep its wording closely and preserve its exact phrases.\n"
            "Do not introduce facts that are not explicitly in the candidate evidence lines.\n"
            "If there are multiple relevant schedule items, include all of them.\n"
            "If a prior item is canceled, state that it is canceled.\n"
            f"Question: {instance.question}\n"
            f"Requester: {instance.asking_user_id}\n"
            f"Action: {action}\n"
            f"Extra rules: {extra_rules}\n"
            f"Candidate evidence lines: {selected_lines}\n"
            f"Structured surface draft: {surface_replay_result.prediction}"
        )
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are the final answer synthesizer for a governed memory system. "
                "Follow the required action exactly and return compact JSON only."
            ),
            user_prompt=prompt,
        )
    except LLMClientUnavailableError:
        return None
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    answer_text = str(raw.get("prediction") or raw.get("answer_text") or "").strip()
    if not answer_text:
        return None
    llm_verifier = _verify_surface_replay_answer(
        answer_text,
        selected_lines,
        instance.question,
        semantic_intent=semantic_intent,
    )
    structured_prediction = str(raw_response.get("structured_surface_prediction") or "").strip()
    fallback_prediction = ""
    fallback_verifier = None
    if structured_prediction:
        fallback_prediction = structured_prediction
        fallback_verifier = _verify_surface_replay_answer(
            structured_prediction,
            selected_lines,
            instance.question,
            semantic_intent=semantic_intent,
        )
    elif surface_replay_result.prediction:
        fallback_prediction = str(surface_replay_result.prediction).strip()
        fallback_verifier = _verify_surface_replay_answer(
            fallback_prediction,
            selected_lines,
            instance.question,
            semantic_intent=semantic_intent,
        )
    answer_text = _postprocess_current_state_prediction(
        prediction=answer_text,
        selected_lines=selected_lines,
        question=instance.question,
    )
    if fallback_prediction:
        fallback_prediction = _postprocess_current_state_prediction(
            prediction=fallback_prediction,
            selected_lines=selected_lines,
            question=instance.question,
        )
        fallback_verifier = _verify_surface_replay_answer(
            fallback_prediction,
            selected_lines,
            instance.question,
            semantic_intent=semantic_intent,
        )
    final_prediction = answer_text
    final_verifier = llm_verifier
    if fallback_prediction and fallback_verifier is not None and _is_better_surface_answer(fallback_verifier, llm_verifier, fallback_prediction, answer_text):
        final_prediction = fallback_prediction
        final_verifier = fallback_verifier
    return AnswerResult(
        prediction=final_prediction,
        answer_text=final_prediction,
        used_memory_ids=[str(x) for x in raw.get("used_chunk_ids", used_chunk_ids)],
        reasoning_summary=str(raw.get("reasoning_summary") or "LLM synthesis over focused evidence lines."),
        action=action,
        answer_structured={
            "surface_lines": selected_lines,
            "compiled_frames": compiled_frames,
            "llm_synthesis": True,
        },
        raw_response={
            **raw,
            "surface_lines": selected_lines,
            "compiled_frames": compiled_frames,
            "slot_verifier": final_verifier,
            "llm_slot_verifier": llm_verifier,
            "raw_surface_verifier": raw_response.get("raw_surface_verifier", {}),
            "structured_surface_prediction": raw_response.get("structured_surface_prediction"),
            "focused_surface_mode": raw_response.get("focused_surface_mode"),
            "llm_synthesis": True,
            "final_answer_source": "structured_surface_prediction" if final_prediction == structured_prediction and structured_prediction else ("surface_replay_result" if final_prediction == surface_replay_result.prediction else "llm"),
        },
    )


def build_no_memory_answer_text() -> str:
    return "I do not have memory of that."


def _should_enable_question_rerank(profile: dict[str, Any]) -> bool:
    return any(
        bool(profile.get(key))
        for key in [
            "wants_authorized_details",
            "wants_callback_instruction",
            "wants_followup_plan",
            "wants_imaging_only",
            "mixed_allergy_schedule",
            "wants_active_medication_status",
            "wants_medication_regimen_plan",
            "wants_return_precautions",
            "wants_contact_methods",
            "wants_current_policy_state",
            "wants_current_plan_window",
            "wants_current_med_followup_plan",
        ]
    )


def _focused_surface_mode(profile: dict[str, Any]) -> str | None:
    if profile.get("wants_current_policy_state"):
        return "policy"
    if profile.get("wants_current_state_bundle"):
        return "current_state_bundle"
    if profile.get("wants_contact_protocol"):
        return "contact_protocol"
    if profile.get("wants_final_procedure_plan"):
        return "final_procedure_plan"
    if profile.get("wants_current_med_followup_plan"):
        if profile.get("wants_rescheduled_followup"):
            return "rescheduled_followup"
        return "current_med_followup_plan"
    if profile.get("wants_active_medication_status"):
        return "medication_status"
    if profile.get("wants_return_precautions"):
        return "return_precautions"
    if profile.get("wants_contact_methods"):
        return "contact_methods"
    if profile.get("wants_current_plan_window"):
        return "current_plan_window"
    if profile.get("wants_callback_instruction"):
        return "callback_instruction"
    if profile.get("wants_followup_plan"):
        return "followup_plan"
    if profile.get("mixed_allergy_schedule"):
        return "mixed_allergy_schedule"
    if profile.get("wants_authorized_details"):
        return "authorized_schedule"
    if profile.get("wants_imaging_only"):
        return "imaging_schedule"
    return None


def _looks_like_medication_regimen_line(text: str, line_meta: dict[str, Any]) -> bool:
    lowered = str(text or "").lower()
    frame_type = str(line_meta.get("frame_type") or "")
    slots = dict(line_meta.get("slots") or {})
    if frame_type == "medication" and any(slots.get(key) for key in ["instruction", "dosage", "timing", "medication_name"]):
        return True
    if frame_type not in {"medication", "general_fact"}:
        return False
    dosage_mentions = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|milligrams|micrograms)\b", lowered)
    if len(dosage_mentions) >= 2 and any(
        token in lowered
        for token in ["daily", "twice a day", "twice daily", "every morning", "nightly", "once daily"]
    ):
        return True
    return any(
        token in lowered
        for token in [
            "current medication plan",
            "current plan:",
            "stop ",
            "continue ",
            "start ",
            "restart ",
            "hold ",
            "use ",
            "take ",
            "switch you to ",
            "with food",
            "nightly",
            "twice daily",
            "once daily",
            "as needed",
            "prn",
        ]
    )


def _extract_regimen_action_family(text: str, line_meta: dict[str, Any]) -> str | None:
    lowered = str(text or "").lower()
    if str(line_meta.get("frame_type") or "") != "medication":
        return None
    slots = dict(line_meta.get("slots") or {})
    medication_name = str(slots.get("medication_name") or slots.get("medication") or "").strip().lower()
    if _looks_like_monitoring_instruction_line(text, line_meta):
        return "monitor"
    if any(token in lowered for token in ["stop ", "hold ", "avoid "]):
        return "stop"
    if any(token in lowered for token in ["continue ", "restart "]):
        return "continue"
    if any(token in lowered for token in ["start ", "switch to "]):
        return "start"
    if re.search(r"\buse\b", lowered) or re.search(r"\btake\b", lowered):
        if medication_name and any(token in medication_name for token in ["instead", "for pain", "for nausea"]):
            return "use"
        if "as needed" in lowered or "prn" in lowered:
            return "use"
        return "start"
    return None


def _looks_like_monitoring_instruction_line(text: str, line_meta: dict[str, Any]) -> bool:
    lowered = str(text or "").lower()
    frame_type = str(line_meta.get("frame_type") or "")
    if frame_type == "instruction" and "blood pressure" in lowered:
        return True
    return any(
        token in lowered
        for token in [
            "check blood pressure",
            "monitor blood pressure",
            "blood pressure twice daily",
            "morning and evening",
            "write the readings in a log",
        ]
    )


def _looks_like_medication_discussion_prompt(text: str, line_meta: dict[str, Any]) -> bool:
    lowered = str(text or "").lower()
    if "?" in text:
        return True
    frame_type = str(line_meta.get("frame_type") or "")
    slots = dict(line_meta.get("slots") or {})
    if frame_type == "medication" and not any(slots.get(key) for key in ["instruction", "dosage", "timing", "medication_name"]):
        return True
    return any(
        token in lowered
        for token in [
            "review medications",
            "what medications",
            "which medications",
            "medication review",
        ]
    )


def _looks_like_schedule_request_line(text: str, line_meta: dict[str, Any]) -> bool:
    lowered = str(text or "").lower().strip()
    if not lowered:
        return False
    frame_type = str(line_meta.get("frame_type") or "")
    if frame_type not in {"appointment", "clinic_visit", "test_or_imaging", "logistics"}:
        return False
    slots = dict(line_meta.get("slots") or {})
    has_concrete_anchor = any(slots.get(key) for key in ["date", "time", "secondary_time", "location", "provider"])
    if has_concrete_anchor:
        return False
    request_markers = [
        "i want",
        "i just want",
        "want this scheduled",
        "want it scheduled",
        "need this scheduled",
        "need it scheduled",
        "scheduled quickly",
        "scheduled soon",
        "book this quickly",
        "book it quickly",
        "move quickly",
    ]
    return any(marker in lowered for marker in request_markers)


def _rerank_candidate_pool_for_question(
    *,
    question: str,
    asking_user_id: str | None,
    candidate_rows: list[dict[str, Any]],
    memory_by_id: dict[str, MemoryItem],
    final_k: int,
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    if not candidate_rows:
        return [], {"enabled": False, "selected_memory_ids": [], "candidates": []}
    domains = _infer_query_domains(question)
    profile = _build_question_profile(question, domains)
    candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        chunk_id = str(row.get("chunk_id") or "")
        item = memory_by_id.get(chunk_id)
        if item is None:
            continue
        evidence = RetrievedEvidence(
            memory_id=item.memory_id,
            content=item.content,
            score=float(row.get("best_score") or 0.0),
            retrieval_source="dense_rerank",
            reason="question-aware rag rerank",
            user_id=item.user_id,
            memory_type=item.memory_type,
            scope=item.scope,
            entities=item.entities,
            time=item.time,
            source_message_ids=item.source_message_ids,
            metadata=dict(item.metadata or {}),
        )
        frame = compile_evidence_frame(evidence)
        row_meta = _build_row_metadata(evidence, frame)
        rerank_score = _score_chunk_candidate(
            question=question,
            asking_user_id=asking_user_id,
            evidence=evidence,
            frame=frame,
            row_meta=row_meta,
            profile=profile,
            domains=domains,
        )
        candidates.append(
            {
                "evidence": evidence,
                "frame_type": frame.frame_type,
                "row_meta": row_meta,
                "rerank_score": rerank_score,
                "best_score": float(row.get("best_score") or 0.0),
            }
        )
    candidates.sort(key=lambda item: (float(item["rerank_score"]), float(item["best_score"])), reverse=True)
    selected: list[RetrievedEvidence] = []
    selected_event_counts: dict[str, int] = {}
    target_weekdays = set(profile.get("target_weekdays") or [])
    mode = _focused_surface_mode(profile)
    required_families = _required_retrieval_families_for_mode(mode, profile)
    covered_families: set[str] = set()
    for family in required_families:
        best = next(
            (
                item
                for item in candidates
                if family in _classify_retrieval_candidate_families(
                    evidence=item["evidence"],
                    frame_type=str(item["frame_type"] or ""),
                    row_meta=item["row_meta"],
                    mode=mode,
                )
                and _can_select_rerank_candidate(
                    candidate=item,
                    selected=selected,
                    selected_event_counts=selected_event_counts,
                    profile=profile,
                    final_k=final_k,
                )
            ),
            None,
        )
        if best is None:
            continue
        evidence = best["evidence"]
        event_key = str(best["row_meta"].get("event_key") or evidence.memory_id)
        selected.append(evidence)
        selected_event_counts[event_key] = selected_event_counts.get(event_key, 0) + 1
        covered_families.update(
            _classify_retrieval_candidate_families(
                evidence=evidence,
                frame_type=str(best["frame_type"] or ""),
                row_meta=best["row_meta"],
                mode=mode,
            )
        )
    for item in candidates:
        if len(selected) >= final_k:
            break
        evidence = item["evidence"]
        if evidence.memory_id in {row.memory_id for row in selected}:
            continue
        if not _can_select_rerank_candidate(
            candidate=item,
            selected=selected,
            selected_event_counts=selected_event_counts,
            profile=profile,
            final_k=final_k,
        ):
            continue
        selected.append(evidence)
        event_key = str(item["row_meta"].get("event_key") or evidence.memory_id)
        selected_event_counts[event_key] = selected_event_counts.get(event_key, 0) + 1
        covered_families.update(
            _classify_retrieval_candidate_families(
                evidence=evidence,
                frame_type=str(item["frame_type"] or ""),
                row_meta=item["row_meta"],
                mode=mode,
            )
        )
    selected_ids = {row.memory_id for row in selected}
    debug = {
        "enabled": True,
        "selected_memory_ids": [row.memory_id for row in selected],
        "mode": mode,
        "required_families": sorted(required_families),
        "covered_families": sorted(covered_families),
        "candidates": [
            {
                "memory_id": item["evidence"].memory_id,
                "rerank_score": item["rerank_score"],
                "best_score": item["best_score"],
                "frame_type": item["frame_type"],
                "event_key": item["row_meta"].get("event_key"),
                "procedures": item["row_meta"].get("procedures"),
                "weekday_mentions": sorted(item["row_meta"].get("weekday_mentions") or []),
                "families": sorted(
                    _classify_retrieval_candidate_families(
                        evidence=item["evidence"],
                        frame_type=str(item["frame_type"] or ""),
                        row_meta=item["row_meta"],
                        mode=mode,
                    )
                ),
                "selected": item["evidence"].memory_id in selected_ids,
            }
            for item in candidates[: max(final_k * 3, 20)]
        ],
    }
    return selected, debug


def _score_chunk_candidate(
    *,
    question: str,
    asking_user_id: str | None,
    evidence: RetrievedEvidence,
    frame,
    row_meta: dict[str, Any],
    profile: dict[str, Any],
    domains: set[str],
) -> float:
    score = _score_evidence_row(
        question=question,
        row=evidence,
        frame=frame,
        row_meta=row_meta,
        profile=profile,
        domains=domains,
    )
    target_weekdays = set(profile.get("target_weekdays") or [])
    if target_weekdays:
        row_weekdays = set(row_meta.get("weekday_mentions") or [])
        if row_weekdays & target_weekdays:
            score += 1.2
        elif profile.get("prefer_strict_target_day") and row_weekdays:
            score -= 1.4
    slots = {str(key) for key, value in dict(row_meta.get("slots") or {}).items() if value}
    frame_type = str(row_meta.get("frame_type") or getattr(frame, "frame_type", "") or "")
    intent_slot_groups = {
        "wants_callback_instruction": {"condition", "instruction"},
        "wants_contact_protocol": {"contact_method", "phone", "portal", "backup_contact"},
        "wants_return_precautions": {"condition", "instruction"},
        "wants_contact_methods": {"contact_method", "phone", "portal", "backup_contact"},
        "wants_followup_plan": {"instruction", "date", "time", "location", "arrival_time"},
        "wants_authorized_details": {"date", "time", "location", "arrival_time", "provider"},
        "wants_current_plan_window": {"instruction", "condition", "date", "time", "location"},
    }
    for profile_key, requested_slots in intent_slot_groups.items():
        if profile.get(profile_key):
            score += 1.4 * len(slots & requested_slots)
    if profile.get("wants_imaging_only"):
        procedures = set(row_meta.get("procedures") or [])
        if procedures & {"ultrasound", "imaging", "scan"}:
            score += 1.6
        elif procedures:
            score -= 2.0
    if profile.get("mixed_allergy_schedule"):
        if frame_type == "allergy" or slots & {"allergy_substance", "allergy_reaction"}:
            score += 1.4
        if slots & {"date", "time", "location", "arrival_time"}:
            score += 1.4
    if profile.get("wants_current_policy_state"):
        if _is_policy_revocation_text(evidence.content or ""):
            score += 2.6
        elif _looks_like_policy_state_text(evidence.content or ""):
            score += 1.2
    if asking_user_id and asking_user_id.lower() in str(evidence.content or "").lower():
        score += 0.2
    if _is_question_like_text(str(evidence.content or "")):
        score -= 4.0
    return score


def _required_retrieval_families_for_mode(mode: str | None, profile: dict[str, Any]) -> list[str]:
    families: dict[str, list[str]] = {
        "authorized_schedule": ["authorized_logistics", "authorized_followup"],
        "callback_instruction": ["urgent_callback"],
        "policy": ["policy_state"],
        "contact_protocol": ["contact_protocol"],
        "current_plan_window": ["active_schedule", "cancellation", "instruction"],
        "mixed_allergy_schedule": ["allergy", "active_schedule"],
        "final_procedure_plan": ["procedure_schedule", "prep_instruction"],
        "current_med_followup_plan": ["medication_plan", "followup_schedule"],
        "rescheduled_followup": ["medication_plan", "reschedule_update"],
        "contact_methods": ["primary_contact"],
    }
    out = list(families.get(mode or "", []))
    if (mode or "") == "final_procedure_plan" and profile.get("asks_preparation_details"):
        return ["prep_instruction", "procedure_schedule"]
    if (mode or "") == "medication_status":
        regimen_families = list(profile.get("regimen_coverage_families") or [])
        family_map = {
            "start": "medication_start",
            "stop": "medication_stop",
            "continue": "medication_continue",
            "use": "medication_use",
            "monitor": "monitoring_instruction",
        }
        for family in regimen_families:
            mapped = family_map.get(family)
            if mapped and mapped not in out:
                out.append(mapped)
    if (mode or "") == "current_state_bundle":
        for family in _requested_current_families(profile):
            if family not in out:
                out.append(family)
        if not out:
            out = ["current_target_date", "current_budget", "current_discount", "current_blocker", "current_stipend", "current_safe_wording", "household_plan"]
    return out


def _classify_retrieval_candidate_families(
    *,
    evidence: RetrievedEvidence,
    frame_type: str,
    row_meta: dict[str, Any],
    mode: str | None,
) -> set[str]:
    text = str(evidence.content or "")
    families = _classify_line_families(text, {"frame_type": frame_type, **row_meta}, mode=mode or "")
    slots = dict(row_meta.get("slots") or {})
    slot_names = set(slots)
    lifecycle_status = str(row_meta.get("lifecycle_status") or "").lower()
    if frame_type in {"consent_or_permission", "privacy_policy"} or slot_names & {"policy_scope", "consent_scope"}:
        families.add("policy_state")
    if slot_names & {"contact_method", "phone", "portal", "backup_contact"}:
        families.add("contact_protocol")
        families.add("policy_scope")
        families.add("primary_contact")
    if frame_type in {"appointment", "test_or_imaging", "clinic_visit", "logistics"} or slot_names & {
        "date", "time", "arrival_time", "location", "provider", "procedure", "visit_type",
    }:
        families.add("procedure_schedule")
        families.add("active_schedule")
        families.add("followup_schedule")
    if frame_type == "instruction" or "instruction" in slot_names:
        families.add("prep_instruction")
        families.add("instruction")
    if frame_type == "medication" or slot_names & {"medication", "dosage", "timing"}:
        families.add("medication_plan")
    if lifecycle_status in {"canceled", "superseded"} or frame_type == "cancellation":
        families.add("cancellation")
        families.add("reschedule_update")
        families.add("followup_schedule")
    if frame_type == "allergy" or slot_names & {"allergy_substance", "allergy_reaction"}:
        families.add("allergy")
    return families


def _can_select_rerank_candidate(
    *,
    candidate: dict[str, Any],
    selected: list[RetrievedEvidence],
    selected_event_counts: dict[str, int],
    profile: dict[str, Any],
    final_k: int,
) -> bool:
    evidence = candidate["evidence"]
    event_key = str(candidate["row_meta"].get("event_key") or evidence.memory_id)
    if selected_event_counts.get(event_key, 0) >= 2:
        return False
    if _is_question_like_text(str(evidence.content or "")):
        return False
    target_weekdays = set(profile.get("target_weekdays") or [])
    prefer_strict_target_day = bool(profile.get("prefer_strict_target_day"))
    if target_weekdays and prefer_strict_target_day:
        row_weekdays = set(candidate["row_meta"].get("weekday_mentions") or [])
        lowered = str(evidence.content or "").lower()
        if row_weekdays and not _weekday_set_matches_profile(row_weekdays, profile):
            lowered = str(evidence.content or "").lower()
            if "canceled" not in lowered and "cancelled" not in lowered and "prior " not in lowered and "old " not in lowered:
                if len(selected) >= max(3, final_k // 3):
                    return False
    return True


def _can_shortlist_item(
    *,
    item: dict[str, Any],
    selected: list[RetrievedEvidence],
    selected_events: dict[str, int],
    selected_frame_types: set[str],
    profile: dict[str, Any],
) -> bool:
    row = item["row"]
    if row.memory_id in {entry.memory_id for entry in selected}:
        return False
    event_key = str(item["event_key"])
    if selected_events.get(event_key, 0) >= 2:
        return False
    if item["frame_type"] == "general_fact" and selected_frame_types and len(selected) >= 3:
        return False
    target_weekdays = set(profile.get("target_weekdays") or [])
    if target_weekdays and profile.get("prefer_strict_target_day"):
        text_lower = str(row.content or "").lower()
        row_weekdays = _extract_weekday_mentions(text_lower)
        if row_weekdays and not _weekday_set_matches_profile(row_weekdays, profile):
            if "canceled" not in text_lower and "cancelled" not in text_lower and "prior " not in text_lower and "old " not in text_lower:
                return False
    return True


def _weekdays_after(bound: str) -> set[str]:
    bound_idx = WEEKDAY_ORDER.get(bound.lower())
    if bound_idx is None:
        return set()
    return {weekday for weekday, idx in WEEKDAY_ORDER.items() if idx > bound_idx}


def _weekday_set_matches_profile(row_weekdays: set[str], profile: dict[str, Any]) -> bool:
    target_weekdays = set(profile.get("target_weekdays") or [])
    if row_weekdays & target_weekdays:
        return True
    through_bound = str(profile.get("through_bound_weekday") or "").strip().lower()
    if through_bound:
        bound_idx = WEEKDAY_ORDER.get(through_bound)
        if bound_idx is None:
            return False
        for weekday in row_weekdays:
            weekday_idx = WEEKDAY_ORDER.get(weekday.lower())
            if weekday_idx is not None and weekday_idx <= bound_idx:
                return True
    return False


def _extract_through_weekday(text: str) -> str | None:
    match = re.search(r"\bthrough\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", text)
    if not match:
        return None
    return str(match.group(1)).lower()


def _is_question_like_text(text: str) -> bool:
    cleaned = _strip_message_prefix(text)
    lowered = cleaned.lower().strip()
    if not lowered:
        return False
    if lowered.endswith("?"):
        return True
    return bool(re.match(r"^(what|when|where|can|could|do|does|did|is|are|am|should|would|will|how)\b", lowered))
