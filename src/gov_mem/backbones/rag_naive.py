from __future__ import annotations

import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# Reuse the released GateMem prompt/domain helpers without requiring callers to
# set PYTHONPATH manually.
OFFICIAL_BENCH_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "GateMem-official"
if str(OFFICIAL_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_BENCH_ROOT))

from gov_mem.backbones.common import (
    BackboneRunResult,
    RAGChunk,
    build_reasoning_state,
    _chunk_to_memory_item,
    save_rag_chunks,
)
from gov_mem.data.schema import (
    AnswerResult,
    GovernedActionDecision,
    MemoryInstance,
    QueryPlan,
    RetrievedEvidence,
)
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.reasoning.operators import build_required_slot_plan
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.backbones.stage2_typed_rerank import (
    _DATE_RE,
    _candidate_matches_request_slot,
    _EXPLICIT_SENSITIVE_FIELD_PATTERNS,
    _is_competing_sensitive_evidence,
    _slot_has_concrete_value,
    _WEEKDAY_RE,
    Stage2Decision,
    deletion_gate_reason,
    explicit_sensitive_boundary_reason,
    mixed_answer_boundary_reason,
    project_mixed_current_state_evidence,
    reason_mixed_evidence_with_llm,
    resolve_long_context_field_ledger,
    rerank_typed_scalar_evidence,
    build_summary_only_evidence,
    summary_only_boundary_reason,
)
from bench.domains import format_relationship_fact, get_domain_label, get_query_policy_block


OFFICIAL_QUERY_PROMPT = OFFICIAL_BENCH_ROOT / "bench" / "prompts" / "query_prompt.txt"


def _build_turn_chunks(instance: MemoryInstance) -> list[RAGChunk]:
    """Match GateMem's RAG-Naive V0: one retrievable chunk per visible turn."""
    chunks: list[RAGChunk] = []
    for index, message in enumerate(instance.messages):
        speaker_id = str(message.get("speaker_id") or "unknown")
        role = str(message.get("speaker_role") or "")
        text = str(message.get("text") or "").strip()
        prefix = f"[{role}:{speaker_id}]" if role else f"[{speaker_id}]"
        chunks.append(
            RAGChunk(
                chunk_id=f"{instance.instance_id}_msg_{index:04d}",
                instance_id=instance.instance_id,
                text=f"{prefix} {text}".strip(),
                source_message_ids=[str(message.get("message_id") or "")],
                speaker_ids=[speaker_id],
                timestamp_range=(message.get("timestamp"), message.get("timestamp")),
                metadata={"chunk_type": "turn"},
            )
        )
    return chunks


def _relationship_block(instance: MemoryInstance) -> str:
    raw_episode = dict((instance.metadata.get("raw_sample") or {}).get("episode") or {})
    entities = dict(raw_episode.get("entities") or {})
    relationships = list(entities.get("relationships") or [])
    requester = str(instance.asking_user_id or "")
    relevant = []
    for relationship in relationships:
        if not isinstance(relationship, dict):
            continue
        if any(
            str(value) == requester
            for key, value in relationship.items()
            if str(key).lower().endswith("_id")
        ):
            relevant.append(format_relationship_fact(relationship))
    return "\n".join(relevant) if relevant else "(none)"


def _format_retrieved_memory(evidence: list[RetrievedEvidence]) -> str:
    if not evidence:
        return "(none)"
    lines = []
    for index, row in enumerate(evidence, 1):
        speaker = str(row.metadata.get("speaker_id") or row.user_id or "unknown")
        lines.append(f"Memory {index} (speaker={speaker}): {row.content}")
    return "\n".join(lines)


_STAGE2_FIELD_LABELS = {
    "date": "date/day",
    "visit_window": "visit or arrival window",
    "setup_window": "setup window",
    "helper_window": "helper window",
    "desk_buzz_rule": "desk buzz rule",
    "delivery_window": "delivery/staging window",
    "entry_method": "entry/entrance method",
    "approved_areas": "approved spaces/areas",
    "signoff_window": "signoff window",
    "overflow_point": "overflow point",
    "label_color": "label color",
    "release_rule": "release rule",
    "fallback_rule": "fallback/contingency rule",
    "handling_constraints": "handling constraints",
}


def _stage2_answer_instruction(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Return a narrow completeness cue for safe mixed projections.

    Stage 2 has already selected a complete evidence carrier set.  This cue
    only helps the answer model preserve field-level qualifiers; it does not
    add facts, authorize disclosure, or run another model call.
    """
    if decision.route != "mixed":
        summary_reason = summary_only_boundary_reason(instance=instance)
        if not summary_reason:
            return ""
    question = str(instance.question or "")
    if deletion_gate_reason(question):
        return ""
    if any(pattern.search(question.casefold()) for _, pattern in _EXPLICIT_SENSITIVE_FIELD_PATTERNS) and not summary_only_boundary_reason(instance=instance):
        return ""
    requested_slots: list[str] = list(decision.long_context_fields)
    for row in evidence:
        requested_slots.extend(str(value) for value in (row.metadata or {}).get("projection_requested_slots") or [])
    requested_slots = list(dict.fromkeys(requested_slots))
    labels = [
        _STAGE2_FIELD_LABELS.get(slot.rsplit(".", 1)[-1], slot)
        for slot in requested_slots
    ]
    if len(labels) < 2 and not decision.long_context_applied:
        return ""
    instructions = []
    if labels:
        instructions.append(
            "Stage 2 field-completeness check: answer every explicitly requested "
            f"field ({', '.join(labels)}). Preserve exact qualifiers and conditions "
            "from the evidence, including words such as only, after, before, "
            "still, and instead of. Do not replace a specific method or condition "
            "with a broader paraphrase, and do not invent missing values."
        )
    if decision.long_context_applied:
        instructions.append(
            "Some fields are backed by a verified Stage 2 long-context ledger. "
            "Use its source-bound quotes for the named fields, and do not add any "
            "fact that is not present in those quotes or the other retrieved evidence."
        )
    if any(slot.endswith(".date") or slot == "date" for slot in requested_slots):
        instructions.append(
            "The date/day field is an explicitly requested answer field. Include the "
            "weekday or calendar date shown in its verified source quote (for example, "
            "Saturday); do not silently omit it or infer a different day."
        )
    if " and " in question.casefold() and any(
        marker in question.casefold() for marker in ("plan", "state", "summary", "entity")
    ):
        instructions.append(
            "The question names multiple plans or entities. Keep their fields separate "
            "and include every requested current field for each named plan/entity; a "
            "value from one plan must not stand in for another."
        )
    return "\n".join(instructions)


def _append_missing_verified_date(
    *,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Preserve a verified weekday/date when the answer model omits it."""

    if not decision.long_context_applied or not any(
        slot == "date" or slot.endswith(".date")
        for slot in decision.long_context_fields
    ):
        return answer
    answer_text = str(answer or "").strip()
    lowered_answer = answer_text.casefold()
    for row in evidence:
        metadata = dict(row.metadata or {})
        slot = str(metadata.get("stage2_long_context_slot") or "")
        if slot != "date" and not slot.endswith(".date"):
            continue
        quote = str(metadata.get("stage2_long_context_quote") or "")
        match = _WEEKDAY_RE.search(quote) or _DATE_RE.search(quote)
        if match and match.group(0).casefold() not in lowered_answer:
            return f"{answer_text}\nDate/day: {match.group(0)}."
    return answer_text


def _append_missing_verified_area_details(
    *,
    instance: MemoryInstance,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Restore omitted area nouns from verified, non-sensitive source rows."""

    if not decision.long_context_applied or not any(
        slot == "approved_areas" or slot.endswith(".approved_areas")
        for slot in decision.long_context_fields
    ):
        return answer
    answer_text = str(answer or "").strip()
    message_order = {
        str(message.get("message_id") or ""): index
        for index, message in enumerate(instance.messages)
        if isinstance(message, dict)
    }
    area_terms = re.compile(
        r"\b(?:trough|rack|boxes|balcony|counter|cart|cooling rack|shelf|tray|bin)\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int, str, str]] = []
    for row in evidence:
        quote = str(row.content or "").strip()
        if (
            not quote
            or not _candidate_matches_request_slot(quote.casefold(), "approved_areas")
            or not _slot_has_concrete_value(quote.casefold(), "approved_areas")
            or _is_competing_sensitive_evidence(
                text=quote,
                requested_slots=["approved_areas"],
            )
        ):
            continue
        terms = {match.casefold() for match in area_terms.findall(quote)}
        if not terms:
            continue
        source_order = max(
            (message_order.get(source_id, -1) for source_id in row.source_message_ids),
            default=-1,
        )
        candidates.append((len(terms), source_order, quote, " ".join(sorted(terms))))
    if not candidates:
        return answer_text

    # Keep the most specific source for each named entity.  For a single
    # entity, this also prefers the sentence that contains the actual area
    # nouns over a later "same planters" shorthand.
    named_entities = [
        " ".join(match)
        for match in re.findall(r"\b([A-Z][a-z0-9]+\s+[A-Z][a-z0-9]+)\b", instance.question)
    ]
    selected: list[str] = []
    groups = named_entities or [""]
    for entity in groups:
        entity_candidates = [
            item for item in candidates
            if not entity or entity.casefold() in item[2].casefold()
        ]
        if not entity_candidates:
            continue
        selected.append(max(entity_candidates, key=lambda item: (item[0], item[1]))[2])
    selected = list(dict.fromkeys(selected))
    lowered_answer = answer_text.casefold()
    missing_quotes = [
        quote for quote in selected
        if any(term not in lowered_answer for term in {
            match.casefold() for match in area_terms.findall(quote)
        })
    ]
    if missing_quotes:
        return answer_text + "\nVerified area detail: " + " | ".join(missing_quotes)
    return answer_text


def _direct_answer(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    stage2_decision: Stage2Decision | None = None,
    llm_client: LLMClient,
    model_name: str,
) -> AnswerResult:
    stage2_decision = stage2_decision or Stage2Decision(
        route="baseline",
        applied=False,
        original_memory_ids=[row.memory_id for row in evidence],
        selected_memory_ids=[row.memory_id for row in evidence],
        fallback_reason="no Stage 2 decision supplied",
    )
    template = OFFICIAL_QUERY_PROMPT.read_text(encoding="utf-8")
    before, after = template.split("[REQUEST CONTEXT]", 1)
    system_prompt = before.replace("[SYSTEM]", "").strip()
    user_prompt = ("[REQUEST CONTEXT]" + after).format(
        asker_principal_id=instance.asking_user_id or "",
        asker_role=(instance.metadata.get("requester") or {}).get("role") or "",
        relationship_facts_block=_relationship_block(instance),
        retrieved_memory_block=_format_retrieved_memory(evidence),
        query_text=instance.question,
        domain_label=get_domain_label(instance.domain),
        global_access_policy_block=get_query_policy_block(instance.domain or ""),
    )
    answer_instruction = _stage2_answer_instruction(
        instance=instance,
        evidence=evidence,
        decision=stage2_decision,
    )
    if answer_instruction:
        user_prompt = f"{user_prompt}\n\n{answer_instruction}"
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return AnswerResult(
            prediction="I do not have memory of that.",
            answer_text="I do not have memory of that.",
            used_memory_ids=[],
            reasoning_summary=f"RAG-Naive direct answer failed: {type(exc).__name__}",
            action="no_memory",
            raw_response={"error": type(exc).__name__},
        )

    raw = raw if isinstance(raw, dict) else {}
    action = str(raw.get("action") or ("answer" if evidence else "no_memory")).strip()
    if action not in {"answer", "answer_redacted", "refuse", "no_memory"}:
        action = "answer" if evidence else "no_memory"
    answer = str(raw.get("answer") or raw.get("answer_text") or "").strip()
    if not answer:
        answer = "I do not have memory of that." if action == "no_memory" else "I cannot provide that information."
    used_ids = [str(value) for value in list(raw.get("used_record_ids") or []) if str(value).strip()]
    if not used_ids and action in {"answer", "answer_redacted"}:
        used_ids = [row.memory_id for row in evidence]
    boundary_reason = mixed_answer_boundary_reason(
        question=instance.question,
        decision=stage2_decision,
        action=action,
        answer=answer,
    )
    if boundary_reason:
        action = "answer"
        answer = answer.replace(
            "However, I can only provide a high-level summary.", ""
        ).strip()
    if action == "answer":
        answer = _append_missing_verified_date(
            answer=answer,
            evidence=evidence,
            decision=stage2_decision,
        )
        answer = _append_missing_verified_area_details(
            instance=instance,
            answer=answer,
            evidence=evidence,
            decision=stage2_decision,
        )
    return AnswerResult(
        prediction=answer,
        answer_text=answer,
        used_memory_ids=used_ids,
        reasoning_summary=(
            f"GateMem official-compatible RAG-Naive direct answer. {boundary_reason}"
            if boundary_reason
            else "GateMem official-compatible RAG-Naive direct answer."
        ),
        action=action,
        answer_structured={},
        raw_response={"rag_naive_raw": raw},
    )


class RAGNaiveBackbone:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_client: LLMClient,
        config: dict[str, Any],
        output_dir: Path,
        dataset_name: str,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = dataset_name

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        chunks = _build_turn_chunks(instance)
        save_rag_chunks(self.output_dir, self.dataset_name, instance.instance_id, chunks)
        items = [_chunk_to_memory_item(chunk) for chunk in chunks]
        index = DenseMemoryIndex.build(
            items=items,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
        )
        top_k = int((self.config.get("rag") or {}).get("naive_top_k", 20))
        rows = index.query(
            query_texts=[instance.question],
            top_k=top_k,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
        )
        memory_by_id = {item.memory_id: item for item in items}
        evidence = [
            RetrievedEvidence(
                memory_id=memory_by_id[memory_id].memory_id,
                content=memory_by_id[memory_id].content,
                score=float(score),
                retrieval_source="dense",
                reason="official-compatible naive top-k retrieval",
                user_id=memory_by_id[memory_id].user_id,
                memory_type="chunk",
                source_message_ids=memory_by_id[memory_id].source_message_ids,
                metadata={
                    "chunk_type": "turn",
                    "speaker_id": memory_by_id[memory_id].user_id,
                },
            )
            for memory_id, score in rows
            if memory_id in memory_by_id
        ]
        stage2_before = list(evidence)
        stage2_decision = Stage2Decision(
            route="baseline",
            applied=False,
            original_memory_ids=[row.memory_id for row in evidence],
            selected_memory_ids=[row.memory_id for row in evidence],
            fallback_reason="Stage 2 disabled for the official rag_naive baseline",
        )
        if self._stage2_typed_rerank_enabled():
            evidence, stage2_decision = rerank_typed_scalar_evidence(
                instance=instance,
                evidence=evidence,
            )
        deletion_reason = (
            deletion_gate_reason(instance.question)
            if self._stage2_typed_rerank_enabled()
            else None
        )
        if deletion_reason:
            stage2_decision = replace(
                stage2_decision,
                route="access_policy",
                applied=False,
                original_memory_ids=[row.memory_id for row in stage2_before],
                selected_memory_ids=[row.memory_id for row in evidence],
                policy_gate_applied=True,
                policy_gate_reason=deletion_reason,
                long_context_reason="historical/deleted query is excluded",
            )
        plan = QueryPlan(
            query_type="utility",
            target_users=[instance.asking_user_id] if instance.asking_user_id else [],
            target_entities=[],
            required_memory_types=["chunk"],
            symbolic_filters={},
            dense_queries=[instance.question],
            reasoning_ops=[],
        )
        if deletion_reason:
            answer_result = AnswerResult(
                prediction="The requested information has been deleted and is not available.",
                answer_text="The requested information has been deleted and is not available.",
                used_memory_ids=[],
                reasoning_summary="Stage 2 closed-set deletion safety gate.",
                action="no_memory",
                raw_response={"stage2_policy_gate": stage2_decision.to_dict()},
            )
        else:
            sensitive_boundary_reason = (
                explicit_sensitive_boundary_reason(instance=instance, evidence=evidence)
                if self._stage2_typed_rerank_enabled()
                else None
            )
            summary_only_reason = (
                summary_only_boundary_reason(instance=instance)
                if self._stage2_typed_rerank_enabled()
                else None
            )
            if summary_only_reason:
                evidence = build_summary_only_evidence(evidence=evidence)
                stage2_decision = replace(
                    stage2_decision,
                    selected_memory_ids=[row.memory_id for row in evidence],
                    summary_only_applied=True,
                    summary_only_reason=summary_only_reason,
                    long_context_reason="exact evidence replaced by safe summary carriers",
                )
            if sensitive_boundary_reason and not summary_only_reason:
                stage2_decision = replace(
                    stage2_decision,
                    policy_gate_applied=True,
                    policy_gate_reason=sensitive_boundary_reason,
                    long_context_reason="explicit sensitive field query is excluded",
                )
                answer_result = AnswerResult(
                    prediction="I cannot provide that information under the current access policy.",
                    answer_text="I cannot provide that information under the current access policy.",
                    used_memory_ids=[],
                    reasoning_summary="Stage 2 explicit sensitive delivery gate.",
                    action="refuse",
                    raw_response={"stage2_policy_gate": stage2_decision.to_dict()},
                )
            else:
                evidence, llm_reasoning_info = reason_mixed_evidence_with_llm(
                    instance=instance,
                    evidence=evidence,
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "reasoning"),
                    config=self.config,
                )
                stage2_decision = replace(
                    stage2_decision,
                    selected_memory_ids=[row.memory_id for row in evidence],
                    llm_reasoning_applied=bool(llm_reasoning_info.get("applied")),
                    llm_reasoning_model=resolve_llm_model(self.config, "reasoning"),
                    llm_reasoning_reason=str(llm_reasoning_info.get("reason") or "") or None,
                    llm_reasoning_selected_memory_ids=[
                        str(value) for value in llm_reasoning_info.get("selected_memory_ids") or []
                    ],
                    llm_reasoning_ranked_memory_ids=[
                        str(value) for value in llm_reasoning_info.get("ranked_memory_ids") or []
                    ],
                    llm_reasoning_confidence=llm_reasoning_info.get("confidence"),
                )
                if self._stage2_typed_rerank_enabled() and not llm_reasoning_info.get("validated"):
                    evidence, projected_decision = project_mixed_current_state_evidence(
                        instance=instance,
                        evidence=evidence,
                    )
                    stage2_decision = replace(
                        projected_decision,
                        summary_only_applied=bool(summary_only_reason),
                        summary_only_reason=summary_only_reason,
                        llm_reasoning_model=resolve_llm_model(self.config, "reasoning"),
                        llm_reasoning_reason=str(llm_reasoning_info.get("reason") or "") or None,
                        llm_reasoning_selected_memory_ids=[
                            str(value) for value in llm_reasoning_info.get("selected_memory_ids") or []
                        ],
                        llm_reasoning_ranked_memory_ids=[
                            str(value) for value in llm_reasoning_info.get("ranked_memory_ids") or []
                        ],
                        llm_reasoning_confidence=llm_reasoning_info.get("confidence"),
                    )
                evidence, long_context_info = resolve_long_context_field_ledger(
                    instance=instance,
                    evidence=evidence,
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "answering"),
                    config=self.config,
                )
                stage2_decision = replace(
                    stage2_decision,
                    selected_memory_ids=[row.memory_id for row in evidence],
                    long_context_applied=bool(long_context_info.get("applied")),
                    long_context_fields=[
                        str(value) for value in long_context_info.get("fields") or []
                    ],
                    long_context_source_message_ids=[
                        str(value)
                        for value in long_context_info.get("source_message_ids") or []
                    ],
                    long_context_reason=str(long_context_info.get("reason") or "") or None,
                )
                answer_result = _direct_answer(
                    instance=instance,
                    evidence=evidence,
                    stage2_decision=stage2_decision,
                    llm_client=self.llm_client,
                    model_name=resolve_llm_model(self.config, "answering"),
                )
                if summary_only_reason and answer_result.action == "answer":
                    answer_result = replace(
                        answer_result,
                        action="answer_redacted",
                        reasoning_summary=(
                            f"{answer_result.reasoning_summary} "
                            "Stage 2 summary-only delivery boundary."
                        ),
                    )
                elif summary_only_reason and answer_result.action == "refuse":
                    safe_text = " ".join(str(row.content or "") for row in evidence).strip()
                    answer_result = replace(
                        answer_result,
                        prediction=safe_text or "Only a broad, non-sensitive summary is available.",
                        answer_text=safe_text or "Only a broad, non-sensitive summary is available.",
                        action="answer_redacted",
                        used_memory_ids=[row.memory_id for row in evidence],
                        reasoning_summary=(
                            f"{answer_result.reasoning_summary} "
                            "Stage 2 summary-only delivery boundary."
                        ),
                    )
        reasoning_state = build_reasoning_state(
            evidence,
            trace=[
                f"official-compatible rag_naive selected {len(evidence)} turn chunks with one query.",
            ],
            selected_frames=[],
            required_slot_plan=build_required_slot_plan(instance.question, plan),
        )
        action_decision = GovernedActionDecision(
            action=answer_result.action,
            answer_mode="direct" if answer_result.action == "answer" else "abstain",
            privacy_decision="unknown",
            forgetting_decision=None,
            evidence_memory_ids=answer_result.used_memory_ids,
            rationale_summary=answer_result.reasoning_summary,
        )
        debug_payload = {
            "experiment_mode": self._experiment_mode(),
            "rag_chunks": [asdict(chunk) for chunk in chunks],
            "retrieved_chunks": [
                {
                    "chunk_id": row.memory_id,
                    "text": row.content,
                    "score": row.score,
                    "source_message_ids": row.source_message_ids,
                }
                for row in evidence
            ],
            "stage2_decision": stage2_decision.to_dict(),
            "atomic_memories": [],
            "retrieved_atomic_memories": [],
            "policy_decisions": [],
            "selected_evidence": [
                {
                    "evidence_id": row.memory_id,
                    "source_type": "chunk",
                    "text": row.content,
                    "score": row.score,
                }
                for row in evidence
            ],
            "current_state": {},
            "slot_coverage": {},
            "action_correction_trace": [],
            "surface_lines": [],
            "compiled_frames": [],
            "retrieval_queries": [instance.question],
            "retrieval_candidates": [
                {"chunk_id": memory_id, "score": float(score)}
                for memory_id, score in rows
            ],
        }
        retrieval_result = {
            "retrieved_before_stage2": stage2_before,
            "retrieved_before_privacy_filter": evidence,
            "retrieved_after_privacy_filter": evidence,
            "filtered_evidence": [],
            "query_variants": [instance.question],
            "retrieval_candidates": debug_payload["retrieval_candidates"],
            "rag_chunks": [asdict(chunk) for chunk in chunks],
        }
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_result,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=debug_payload,
        )

    def _experiment_mode(self) -> str:
        return str((self.config.get("experiment") or {}).get("mode") or "rag_naive")

    def _stage2_typed_rerank_enabled(self) -> bool:
        return self._experiment_mode() == "rag_naive_v3_typed_rerank"
