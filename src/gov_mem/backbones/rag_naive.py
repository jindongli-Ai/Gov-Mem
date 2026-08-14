from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

# Reuse the released GateMem prompt/domain helpers without requiring callers to
# set PYTHONPATH manually. The suite runner stages this tree separately.
_configured_bench_root = os.environ.get("GOVMEM_OFFICIAL_BENCHMARK_ROOT", "").strip()
OFFICIAL_BENCH_ROOT = (
    Path(_configured_bench_root)
    if _configured_bench_root
    else Path(__file__).resolve().parents[3] / "third_party" / "GateMem-official"
)
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
    _TIME_RE,
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
    _STALE_MARKERS,
    _CURRENT_TERMS,
    _contains_marker,
)
from gov_mem.backbones.symbolic_evidence import build_symbolic_evidence
from bench.domains import format_relationship_fact, get_domain_label, get_query_policy_block


OFFICIAL_QUERY_PROMPT = OFFICIAL_BENCH_ROOT / "bench" / "prompts" / "query_prompt.txt"


def _structured_message_record(
    *,
    instance: MemoryInstance,
    message: dict[str, Any],
    turn_index: int,
) -> dict[str, Any]:
    """Preserve the observable GateMem message as typed retrieval metadata."""
    speaker_id = str(message.get("speaker_id") or "") or None
    speaker_role = str(message.get("speaker_role") or "") or None
    message_id = str(message.get("message_id") or "")
    turn_id = str(message.get("turn_id") or message_id)
    return {
        "record_type": "message",
        # Keep both Gov-Mem's normalized name and GateMem's original name.
        "message_id": message_id,
        "turn_id": turn_id,
        "turn_index": int(turn_index),
        "timestamp": message.get("timestamp"),
        "speaker": {
            "principal_id": speaker_id,
            "role": speaker_role,
        },
        "turn_kind": message.get("turn_kind"),
        "text": str(message.get("text") or ""),
        "checkpoint": {
            "as_of_turn_id": str(
                (instance.metadata.get("observable") or {}).get("as_of_turn_id") or ""
            ),
        },
        # The adapter keeps the visible source turn verbatim. This prevents
        # future GateMem fields from being lost at the RAG boundary.
        "source_turn": dict(message.get("source_turn") or {}),
    }


def _build_turn_chunks(instance: MemoryInstance) -> list[RAGChunk]:
    """Match GateMem's RAG-Naive V0: one retrievable chunk per visible turn."""
    chunks: list[RAGChunk] = []
    for index, message in enumerate(instance.messages):
        speaker_id = str(message.get("speaker_id") or "unknown")
        role = str(message.get("speaker_role") or "")
        message_id = str(message.get("message_id") or "")
        text = str(message.get("text") or "").strip()
        prefix = f"[{role}:{speaker_id}]" if role else f"[{speaker_id}]"
        chunks.append(
            RAGChunk(
                # Match the released GateMem Chunker. The identifier is local
                # to this visible turn sequence and must not expose the
                # benchmark checkpoint ID to a later Stage 2 model.
                chunk_id=f"chunk_{index + 1:04d}_{message_id}_{message_id}",
                instance_id=instance.instance_id,
                text=f"{prefix} {text}".strip(),
                source_message_ids=[str(message.get("message_id") or "")],
                speaker_ids=[speaker_id],
                timestamp_range=(message.get("timestamp"), message.get("timestamp")),
                metadata={
                    "chunk_type": "turn",
                    "structured_record": _structured_message_record(
                        instance=instance,
                        message=message,
                        turn_index=index,
                    ),
                },
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
    state_ledger = next(
        (
            (row.metadata or {}).get("symbolic_state_ledger")
            for row in evidence
            if isinstance((row.metadata or {}).get("symbolic_state_ledger"), dict)
        ),
        None,
    )
    if state_ledger:
        lines.append(
            "[SYMBOLIC_STATE_LEDGER] "
            + json.dumps(state_ledger, ensure_ascii=False, sort_keys=True)
        )
    for index, row in enumerate(evidence, 1):
        metadata = dict(row.metadata or {})
        record = metadata.get("structured_record")
        if isinstance(record, dict):
            # Keep the typed provenance visible to Stage 2 without asking it
            # to recover identity, role, or time from natural language.
            typed = {
                "record_type": record.get("record_type"),
                "message_id": record.get("message_id"),
                "turn_id": record.get("turn_id"),
                "turn_index": record.get("turn_index"),
                "timestamp": record.get("timestamp"),
                "speaker": record.get("speaker"),
                "turn_kind": record.get("turn_kind"),
                "checkpoint": record.get("checkpoint"),
                "text": record.get("text"),
                "source_turn": record.get("source_turn"),
                "symbolic_provenance": metadata.get("symbolic_provenance"),
                "symbolic_consistency": metadata.get("symbolic_consistency"),
                "symbolic_permission_claim": metadata.get("symbolic_permission_claim"),
                "symbolic_lifecycle_claim": metadata.get("symbolic_lifecycle_claim"),
                "symbolic_state_claims": metadata.get("symbolic_state_claims"),
            }
            lines.append(
                f"Memory {index} [STRUCTURED_RECORD] "
                f"{json.dumps(typed, ensure_ascii=False, sort_keys=True)}"
            )
        else:
            speaker = str(metadata.get("speaker_id") or row.user_id or "unknown")
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

_REDACTED_CONFIRMATION_QUERY_RE = re.compile(
    r"^\s*(?:is|are|was|were|does|do|did|has|have)\b",
    re.IGNORECASE,
)
_REDACTED_CONFIRMATION_PRIVACY_MARKERS = (
    "private", "restricted", "confidential", "hidden", "secret", "deleted",
    "exact", "on file", "yes or no", "associated", "linked", "exist",
)


def _answer_redacted_confirmation_boundary_reason(
    *,
    question: str,
    answer: str,
    action: str,
) -> str | None:
    """Reject confirmation text that contradicts an answer-redacted action."""

    if action != "answer_redacted":
        return None
    answer_text = str(answer or "").strip()
    if not re.match(r"^(?:yes|no)\b", answer_text, re.IGNORECASE):
        return None
    lowered_question = str(question or "").casefold()
    if not _REDACTED_CONFIRMATION_QUERY_RE.search(lowered_question):
        return None
    privacy_probe = any(
        marker in lowered_question
        for marker in _REDACTED_CONFIRMATION_PRIVACY_MARKERS
    ) or any(
        pattern.search(lowered_question)
        for _, pattern in _EXPLICIT_SENSITIVE_FIELD_PATTERNS
    )
    if not privacy_probe:
        return None
    return "answer_redacted contained a sensitive confirmation"


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
    # Do not append a second calendar value when the answer already contains
    # a date or weekday.  A ledger quote can be stale or refer to another
    # carrier; adding it beside an existing value creates a contradictory
    # answer instead of repairing an omission.
    if _WEEKDAY_RE.search(answer_text) or _DATE_RE.search(answer_text):
        return answer_text
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


def _verified_field_value(*, slot: str, quote: str) -> str | None:
    """Extract one closed-set field value from an already verified quote."""

    text = str(quote or "").strip()
    lowered_slot = str(slot or "").casefold()
    if lowered_slot in {"target_date", "public_event_date", "date"}:
        match = _DATE_RE.search(text)
        return match.group(0) if match else None
    if lowered_slot in {"monthly_stipend", "approved_budget", "approved_discount_cap"}:
        match = re.search(r"(?:\$\s*\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:USD|dollars)\b)", text, re.IGNORECASE)
        return match.group(0) if match else None
    if lowered_slot == "access_badge":
        match = re.search(
            r"\b(?:badge|access\s+code|token)\b[^.!?]{0,50}?\b(?=[a-z0-9_-]*\d)[a-z][a-z0-9]*(?:[_-][a-z0-9]+){1,}\b",
            text,
            re.IGNORECASE,
        )
        return match.group(0).split()[-1] if match else None
    if lowered_slot == "blocker":
        match = re.search(
            r"\b(?:active\s+)?blocker(?:\s+(?:is|are|now|still|remains?))?\s+"
            r"(?P<value>[^.!?;,]+?)(?=\s+(?:inside|out\s+of|private|even\s+if|and\s+only|for\s+the|may\s+be)\b|[.!?;,]|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group("value").strip()
    return None


def _repair_requester_bound_scalar_values(
    *,
    instance: MemoryInstance,
    answer: str,
    evidence: list[RetrievedEvidence],
) -> str:
    """Preserve concrete current requester-owned code and expiry values."""

    question = str(instance.question or "").casefold()
    if not any(term in question for term in ("expire", "expiry", "expires")):
        return str(answer or "").strip()
    requester = str(instance.asking_user_id or "").strip()
    if not requester:
        return str(answer or "").strip()
    message_order = {
        str(message.get("message_id") or ""): index
        for index, message in enumerate(instance.messages)
        if isinstance(message, dict)
    }
    candidates: list[tuple[int, str, str, str | None]] = []
    code_pattern = re.compile(r"\b(?=[a-z0-9_-]*\d)[a-z][a-z0-9]*(?:[_-][a-z0-9]+){2,}\b", re.IGNORECASE)
    for row in evidence:
        metadata = dict(row.metadata or {})
        owner = str(metadata.get("speaker_id") or row.user_id or "").strip()
        if owner != requester:
            continue
        text = str(row.content or "").strip()
        lowered = text.casefold()
        if not any(_contains_marker(lowered, marker) for marker in _CURRENT_TERMS):
            continue
        if not re.search(r"\b(?:access\s+code|credential|token|badge)\b", lowered):
            continue
        code_match = re.search(
            r"\b(?:access\s+code|credential|token|badge)\b"
            r"(?:\s+(?:is|was|remains|now|still|for)\b)?\s*"
            r"(?P<value>[a-z][a-z0-9]*(?:[_-][a-z0-9]+){2,})",
            text,
            re.IGNORECASE,
        )
        code = code_match.group("value") if code_match else None
        expiry = None
        if re.search(r"\b(?:expir(?:y|es|ed)|active\s+through)\b", lowered):
            date = _DATE_RE.search(text)
            time = _TIME_RE.search(text) or re.search(r"\b\d{1,2}:\d{2}\b", text)
            if date:
                expiry = date.group(0) + (f" at {time.group(0)}" if time else "")
        current_value = bool(
            re.search(r"\b(?:current|active|new|rotat(?:ed|ion)|replac(?:es|ing))\b", lowered)
        )
        if code and expiry and current_value:
            order = max((message_order.get(source_id, -1) for source_id in row.source_message_ids), default=-1)
            candidates.append((order, code, expiry, text))
    if not candidates:
        return str(answer or "").strip()
    _, code, expiry, _ = max(candidates, key=lambda item: item[0])
    repaired = str(answer or "").strip()
    available_match = re.search(
        r"\baccess\s+code\b\s+is\s+(?:currently\s+)?available\b",
        repaired,
        re.IGNORECASE,
    )
    if available_match:
        repaired = (
            repaired[:available_match.start()]
            + f"access code is {code}"
            + repaired[available_match.end():]
        )
    answer_codes = list(re.finditer(
        r"\baccess\s+code\b[^.!?]{0,30}?\b(?=[a-z0-9_-]*\d)[a-z][a-z0-9]*(?:[_-][a-z0-9]+){2,}\b",
        repaired,
        re.IGNORECASE,
    ))
    if answer_codes:
        match = answer_codes[0]
        value_match = re.search(
            r"\b(?=[a-z0-9_-]*\d)[a-z][a-z0-9]*(?:[_-][a-z0-9]+){2,}\b",
            match.group(0),
            re.IGNORECASE,
        )
        if value_match:
            value_start = match.start() + value_match.start()
            value_end = match.start() + value_match.end()
            repaired = repaired[:value_start] + code + repaired[value_end:]
    elif code.casefold() not in repaired.casefold():
        repaired = f"{repaired} The current active access code is {code}."
    if expiry.casefold() not in repaired.casefold():
        date_match = _DATE_RE.search(repaired)
        if date_match:
            repaired = repaired[:date_match.start()] + expiry + repaired[date_match.end():]
        else:
            repaired = f"{repaired} It expires {expiry}."
    else:
        repaired = re.sub(
            r"(\b\d{1,2}:\d{2}\b)(?:\s*,?\s*at\s+\1)+",
            r"\1",
            repaired,
            flags=re.IGNORECASE,
        )
    return repaired.strip()


def _repair_answer_with_verified_fields(
    *,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Keep the final wording faithful to verified Stage 2 field carriers.

    This is intentionally limited to the opt-in long-context ledger.  It does
    not infer values from raw Stage 1 evidence or make an authorization choice.
    """

    if not decision.long_context_applied:
        return str(answer or "").strip()
    answer_text = str(answer or "").strip()
    if not answer_text:
        return answer_text
    verified: list[tuple[str, str]] = []
    for row in evidence:
        metadata = dict(row.metadata or {})
        slot = str(metadata.get("stage2_long_context_slot") or "").strip()
        quote = str(metadata.get("stage2_long_context_quote") or "").strip()
        if not slot or not quote:
            continue
        value = _verified_field_value(slot=slot, quote=quote)
        if value:
            verified.append((slot, value))

    repaired = answer_text
    requested_date_slots = {
        slot for slot, _ in verified
        if slot in {"target_date", "public_event_date", "date"}
    }
    if len(requested_date_slots) == 1:
        value = next(value for slot, value in verified if slot in requested_date_slots)
        if value.casefold() not in repaired.casefold():
            answer_dates = list(_DATE_RE.finditer(repaired))
            if len(answer_dates) == 1:
                match = answer_dates[0]
                repaired = repaired[:match.start()] + value + repaired[match.end():]

    for slot, value in verified:
        if slot == "access_badge":
            missing_badge = re.search(
                r"\b(?:badge|access\s+code)\b[^.!?]{0,60}\b(?:not\s+specified|unavailable|unknown)\b",
                repaired,
                re.IGNORECASE,
            )
            if missing_badge:
                repaired = repaired[:missing_badge.start()] + f"active badge is {value}" + repaired[missing_badge.end():]
                continue
        if value.casefold() in repaired.casefold():
            continue
        if slot == "blocker":
            blocker_match = re.search(
                r"\bblockers?\b(?:\s+(?:is|are|for|of))?\s+[^.!?;,]+",
                repaired,
                re.IGNORECASE,
            )
            if blocker_match:
                repaired = (
                    repaired[:blocker_match.start()]
                    + f"blocker: {value}"
                    + repaired[blocker_match.end():]
                )
            else:
                repaired = f"{repaired} Verified current blocker: {value}."
        else:
            repaired = f"{repaired} Verified current {slot.replace('_', ' ')}: {value}."
    return repaired.strip()


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


def _append_missing_verified_safe_wording(
    *,
    answer: str,
    evidence: list[RetrievedEvidence],
    decision: Stage2Decision,
) -> str:
    """Preserve omitted concrete clauses from a verified safe-wording carrier."""

    if not decision.long_context_applied or "safe_wording" not in decision.long_context_fields:
        return str(answer or "").strip()
    answer_text = str(answer or "").strip()
    if not answer_text:
        return answer_text
    missing_quotes: list[str] = []
    for row in evidence:
        metadata = dict(row.metadata or {})
        if str(metadata.get("stage2_long_context_slot") or "").strip() != "safe_wording":
            continue
        quote = str(metadata.get("stage2_long_context_quote") or "").strip()
        if not quote or re.search(
            r"\b(?:pin|password|passcode|token|access\s+code|keypad|credential)\b",
            quote,
            re.IGNORECASE,
        ) or _is_competing_sensitive_evidence(
            text=quote,
            requested_slots=["safe_wording"],
        ):
            continue
        # A source-bound safe wording field is a composite carrier.  Repair only
        # when the answer omitted a concrete time or an explicit association
        # (person/place) from that carrier; this keeps ordinary paraphrases
        # concise while preserving requested operations.
        quote_times = {match.group(0).casefold() for match in _TIME_RE.finditer(quote)}
        answer_times = {match.group(0).casefold() for match in _TIME_RE.finditer(answer_text)}
        missing_details = []
        for pattern in (
            r"\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            r"\bnear\s+((?:the\s+)?[a-z]+(?:\s+[a-z]+){0,3})(?=[.;]|$)",
        ):
            for match in re.finditer(pattern, quote):
                detail = re.sub(r"\s+", " ", match.group(1)).strip().casefold()
                if detail and detail not in answer_text.casefold():
                    missing_details.append(detail)
        if (quote_times and not quote_times.issubset(answer_times)) or missing_details:
            missing_quotes.append(quote)
    if not missing_quotes:
        return answer_text
    return answer_text + "\nVerified current safe wording: " + " | ".join(dict.fromkeys(missing_quotes))


def _direct_answer(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
    stage2_decision: Stage2Decision | None = None,
    stage2_prompt_audit: Any = None,
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
            raw_response={
                "error": type(exc).__name__,
                "prompt_audit": {
                    "schema_version": 1,
                    "answer_prompt": {
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                        "context_text": _format_retrieved_memory(evidence),
                    },
                    "stage2_rerank_prompt": stage2_prompt_audit,
                },
            },
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
    boundary_reason = _answer_redacted_confirmation_boundary_reason(
        question=instance.question,
        answer=answer,
        action=action,
    )
    if boundary_reason:
        action = "refuse"
        answer = "I cannot provide that information under the current access policy."
        used_ids = []
    else:
        boundary_reason = mixed_answer_boundary_reason(
            question=instance.question,
            decision=stage2_decision,
            action=action,
            answer=answer,
            instance=instance,
            evidence=evidence,
        )
        if boundary_reason:
            action = "answer"
            answer = answer.replace(
                "However, I can only provide a high-level summary.", ""
            ).strip()
    if action == "answer":
        answer = _repair_requester_bound_scalar_values(
            instance=instance,
            answer=answer,
            evidence=evidence,
        )
        answer = _repair_answer_with_verified_fields(
            answer=answer,
            evidence=evidence,
            decision=stage2_decision,
        )
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
        answer = _append_missing_verified_safe_wording(
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
        raw_response={
            "rag_naive_raw": raw,
            "prompt_audit": {
                "schema_version": 1,
                "answer_prompt": {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "context_text": _format_retrieved_memory(evidence),
                },
                "stage2_rerank_prompt": stage2_prompt_audit,
            },
        },
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
            # GateMem RAG-Naive embeds the visible turn text itself. Do not
            # add Gov-Mem memory metadata to the frozen Stage 1 representation.
            embedding_texts=[chunk.text for chunk in chunks],
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
        )
        top_k = int((self.config.get("rag") or {}).get("naive_top_k", 20))
        rows = index.query(
            query_texts=[instance.question],
            top_k=top_k,
            llm_client=self.embedding_client,
            embedding_model=str(self.config["embedding"]["model"]),
            allow_fallback=bool(self.config["embedding"].get("allow_fallback", True)),
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
                time=memory_by_id[memory_id].time,
                metadata={
                    **dict(memory_by_id[memory_id].metadata or {}),
                    "chunk_type": "turn",
                    "speaker_id": memory_by_id[memory_id].user_id,
                    "speaker_role": (
                        ((memory_by_id[memory_id].metadata.get("structured_record") or {})
                         .get("speaker") or {})
                        .get("role")
                    ),
                    "source_timestamp": memory_by_id[memory_id].time,
                },
            )
            for memory_id, score in rows
            if memory_id in memory_by_id
        ]
        stage2_before = list(evidence)
        symbolic_trace: dict[str, Any] = {}
        if self._symbolic_v4_enabled():
            evidence, symbolic_trace = build_symbolic_evidence(
                instance=instance,
                evidence=evidence,
            )
        stage2_prompt_audit = None
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
                explicit_sensitive_boundary_reason(
                    instance=instance,
                    evidence=evidence,
                )
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
                stage2_prompt_audit = llm_reasoning_info.get("prompt_audit")
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
                long_context_audit = long_context_info.get("prompt_audit")
                if long_context_audit is not None:
                    existing_audits = (
                        list(stage2_prompt_audit)
                        if isinstance(stage2_prompt_audit, list)
                        else [stage2_prompt_audit]
                        if isinstance(stage2_prompt_audit, dict)
                        else []
                    )
                    stage2_prompt_audit = [*existing_audits, long_context_audit]
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
                    stage2_prompt_audit=stage2_prompt_audit,
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
        raw_response = dict(answer_result.raw_response or {})
        if not isinstance(raw_response.get("prompt_audit"), dict):
            raw_response["prompt_audit"] = {
                "schema_version": 1,
                "audit_status": "no_runtime_answer_prompt",
                "answer_prompt": None,
                "stage2_rerank_prompt": stage2_prompt_audit,
            }
            answer_result = replace(answer_result, raw_response=raw_response)
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
            "symbolic_trace": symbolic_trace,
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
            "retrieval_backend": index.last_query_backend,
            "index_backend": index.backend,
            "embedding_model": str(self.config["embedding"]["model"]),
            "embedding_fallback_reason": index.fallback_reason,
            "symbolic_trace": symbolic_trace,
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
        return self._experiment_mode() in {"rag_naive_v3_typed_rerank", "govmem_v4_symbolic"}

    def _symbolic_v4_enabled(self) -> bool:
        return self._experiment_mode() == "govmem_v4_symbolic"
