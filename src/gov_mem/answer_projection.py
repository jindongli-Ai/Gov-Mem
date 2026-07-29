"""Field-level answer planning and evidence projection.

This module is downstream of policy filtering.  It never expands the set of
records visible to a model and it never decides whether a record is allowed.
The base model performs the semantic work; the Python boundary only validates
the returned structure and provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Iterable

from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model


ANSWER_REQUEST_SYSTEM_PROMPT = """
Compile a user question into a complete answer contract after policy filtering.
You are not deciding permission and you are not retrieving memory. Use only
the question, seed fields, and allowed evidence supplied by the caller.
Return JSON only with {"fields": [...], "not_applicable": [...],
"unknown_allowed": [...]}.

Create one field object for every independently answerable fact. Preserve all
seed fields and add missing concrete fields. A field object has:
field_id, label, question_span, entity, temporal_role, cardinality, required,
and disclosure_scope. cardinality is one of single, list, range, boolean, or
unknown. required is true for a fact the question explicitly requests or a
fact needed to answer an explicitly requested aggregate view. A list means
all separately named members must be retained. temporal_role is one of
current, latest, historical, event, deadline, validity, or unspecified.

Expand plan, summary, snapshot, status, schedule, and overview requests into
their concrete fields. Keep time, person/role, location, object/status, and
method/constraint as separate fields when relevant. Keep separate lanes,
subjects, audiences, and temporal qualifiers separate. Do not invent fields
from nearby but unrelated evidence. If a requested field has no value in the
allowed evidence, keep it with required=true and use disclosure_scope or
unknown_allowed to indicate that the value may be unavailable. Do not use
evaluator metadata or hidden labels.
""".strip()


FIELD_PROJECTION_SYSTEM_PROMPT = """
Build a field-level evidence projection for an answer. Authorization has
already been decided. You must use only the supplied allowed evidence and
must not request, infer, or mention any unavailable record.
Return JSON only with {"fields": [...]}.

For every requested field, return:
field_id, status, candidate_values, selected_values, source_memory_ids,
conflict_trace, and provenance. Each candidate or selected value must be a
concrete value directly supported by its cited source memory. candidate_values
and selected_values are arrays of strings. Use status supported when a value
is directly supported, unknown when the allowed evidence does not establish
it, restricted when the requested disclosure scope is not present, and
conflict when allowed sources disagree and the conflict cannot be resolved.

Resolve current/latest values by explicit state-changing language and source
turn order. A later value that says it replaces, supersedes, remains, or is
the current state should be selected over an older value. Preserve all
members of a requested list. Keep source values bound to their own field:
never bind a location to a credential, a date to a status, or one subject's
value to another subject. Treat "same", "unchanged", and "as above" as
lineage clues, not as final values; follow the cited source chain to the
concrete value. If a field has multiple compatible values, retain all values
needed by its cardinality. provenance must contain only source memory ids,
source message ids, source turn indexes, and short source spans copied from
the supplied evidence.
An instruction to provide safe or broad wording does not by itself suppress a
supported operational date, time, location, label, amount, or status. Suppress
only a field that is explicitly restricted or absent from the policy-approved
evidence.
""".strip()


@dataclass(frozen=True)
class AnswerField:
    field_id: str
    label: str
    question_span: str = ""
    entity: str | None = None
    temporal_role: str = "unspecified"
    cardinality: str = "single"
    required: bool = True
    disclosure_scope: str = "standard"


@dataclass(frozen=True)
class AnswerRequest:
    fields: tuple[AnswerField, ...] = ()
    not_applicable: tuple[str, ...] = ()
    unknown_allowed: tuple[str, ...] = ()
    source: str = "base_llm_field_contract"


@dataclass(frozen=True)
class FieldEvidenceProjection:
    field_id: str
    label: str
    status: str
    candidate_values: tuple[str, ...] = ()
    selected_values: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    provenance: tuple[dict[str, Any], ...] = ()
    conflict_trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerProjection:
    request: AnswerRequest
    fields: tuple[FieldEvidenceProjection, ...] = ()
    errors: tuple[str, ...] = ()


_CARDINALITIES = {"single", "list", "range", "boolean", "unknown"}
_TEMPORAL_ROLES = {"current", "latest", "historical", "event", "deadline", "validity", "unspecified"}
_STATUSES = {"supported", "unknown", "restricted", "conflict"}
_GENERIC_LABELS = {"all requested fields", "everything", "details", "the plan"}


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).casefold()).strip()


def _as_list(value: object) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _field_id(value: object, index: int) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9_]+", "_", _text(value)).strip("_").lower()
    return candidate[:48] or f"field_{index:03d}"


def _source_rows(evidence: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("memory_id")): row for row in evidence if str(row.get("memory_id") or "")}


def _value_supported(value: str, rows: list[dict[str, Any]]) -> bool:
    """Accept only source-shaped values, with normalization for punctuation."""
    candidate = _norm(value)
    if not candidate:
        return False
    for row in rows:
        source = _norm(row.get("text"))
        if candidate in source:
            return True
        candidate_tokens = set(candidate.split())
        source_tokens = set(source.split())
        if len(candidate_tokens) >= 2 and len(candidate_tokens & source_tokens) >= max(2, int(len(candidate_tokens) * 0.8)):
            return True
    return False


_QUALIFIED_SCHEDULE_SPAN = re.compile(
    r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"(?:,?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,\s*20\d{2})?)?"
    r"\s*,?\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s*"
    r"(?:to|[-–—])\s*\d{1,2}:\d{2}\s*(?:AM|PM)\b",
    re.IGNORECASE,
)


def _qualify_schedule_value(value: str, rows: list[dict[str, Any]]) -> str:
    """Keep a weekday/date attached to a source-backed time range.

    A model may correctly bind a range but shorten ``Sunday, ... 10:45 AM to
    11:00 AM`` to just the clock values.  For current schedule fields, that
    loses a required qualifier.  The span is copied only from a cited source,
    with the latest cited source preferred; no calendar inference occurs.
    """
    if not re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", value, re.IGNORECASE):
        return value
    time_tokens = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", value, re.IGNORECASE)
    if len(time_tokens) < 2:
        return value
    candidates: list[tuple[int, str]] = []
    for row in rows:
        source = str(row.get("text") or "")
        turn = row.get("source_turn_index")
        turn_index = int(turn) if isinstance(turn, int) else -1
        for match in _QUALIFIED_SCHEDULE_SPAN.finditer(source):
            span = match.group(0).strip(" ,;:")
            if all(token.casefold() in span.casefold() for token in time_tokens):
                candidates.append((turn_index, span))
    if not candidates:
        return value
    return max(candidates, key=lambda item: item[0])[1]


def _merge_seed_fields(seed_fields: Iterable[object], model_fields: Iterable[object]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate([*_as_list(seed_fields), *_as_list(model_fields)], start=1):
        if isinstance(raw, str):
            raw = {"label": raw}
        if not isinstance(raw, dict):
            continue
        label = _text(raw.get("label") or raw.get("name"))
        if not label or label.casefold() in _GENERIC_LABELS or len(label.split()) > 18:
            continue
        # Query-shaped seed labels are grouping instructions, not fields. The
        # concrete clauses normally appear beside them and must remain the
        # authoritative slots.
        if "?" in label or re.search(
            r"\b(?:what|which|including|safely\s+summarize|can\s+you|i\s+can)\b",
            label,
            re.IGNORECASE,
        ):
            continue
        key = _norm(label)
        existing = by_key.get(key)
        if existing is None:
            existing = {
                "field_id": _field_id(raw.get("field_id") or label, index),
                "label": label,
                "question_span": _text(raw.get("question_span")),
                "entity": _text(raw.get("entity")) or None,
                "temporal_role": _text(raw.get("temporal_role")) or "unspecified",
                "cardinality": _text(raw.get("cardinality")) or "single",
                "required": bool(raw.get("required", True)),
                "disclosure_scope": _text(raw.get("disclosure_scope")) or "standard",
            }
            merged.append(existing)
            by_key[key] = existing
        else:
            for name in ("question_span", "entity", "temporal_role", "cardinality", "disclosure_scope"):
                if not existing.get(name) or existing.get(name) == "unspecified":
                    incoming = _text(raw.get(name))
                    if incoming:
                        existing[name] = incoming
            existing["required"] = bool(existing.get("required", True) or raw.get("required", False))
    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, row in enumerate(merged, start=1):
        field_id = _field_id(row.get("field_id") or row.get("label"), index)
        if field_id in used_ids:
            field_id = f"{field_id}_{index:03d}"
        used_ids.add(field_id)
        cardinality = _text(row.get("cardinality")).casefold()
        temporal_role = _text(row.get("temporal_role")).casefold()
        normalized.append({
            "field_id": field_id,
            "label": _text(row.get("label")),
            "question_span": _text(row.get("question_span")),
            "entity": _text(row.get("entity")) or None,
            "temporal_role": temporal_role if temporal_role in _TEMPORAL_ROLES else "unspecified",
            "cardinality": cardinality if cardinality in _CARDINALITIES else "single",
            "required": bool(row.get("required", True)),
            "disclosure_scope": _text(row.get("disclosure_scope")) or "standard",
        })
    return normalized[:32]


def _normalize_request(raw: object, *, seed_fields: Iterable[object]) -> AnswerRequest:
    payload = raw if isinstance(raw, dict) else {}
    fields = _merge_seed_fields(seed_fields, payload.get("fields"))
    return AnswerRequest(
        fields=tuple(AnswerField(**field) for field in fields),
        not_applicable=tuple(_text(value).casefold() for value in _as_list(payload.get("not_applicable")) if _text(value)),
        unknown_allowed=tuple(_text(value) for value in _as_list(payload.get("unknown_allowed")) if _text(value)),
    )


def _fallback_request(seed_fields: Iterable[object], *, question: str) -> AnswerRequest:
    fields = _merge_seed_fields(seed_fields, [])
    if not fields and _text(question):
        fields = [{
            "field_id": "answer",
            "label": "answer to the question",
            "question_span": _text(question),
            "entity": None,
            "temporal_role": "unspecified",
            "cardinality": "single",
            "required": True,
            "disclosure_scope": "standard",
        }]
    return AnswerRequest(fields=tuple(AnswerField(**field) for field in fields), source="seed_field_fallback")


def compile_answer_request(
    *,
    question: str,
    target_subject: str | None,
    request_scope: str | None,
    seed_fields: Iterable[object],
    answer_need_spec: dict[str, Any] | None,
    evidence_payload: list[dict[str, Any]],
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> AnswerRequest:
    """Compile an explicit field contract using only policy-approved evidence."""
    seed = list(_as_list(seed_fields))
    if answer_need_spec:
        seed.extend(_as_list(answer_need_spec.get("requested_fields")))
        seed.extend(
            item
            for dimension in ("when", "who", "where", "what", "how")
            for item in _as_list((answer_need_spec.get("five_w_one_h") or {}).get(dimension))
        )
    fallback = _fallback_request(seed, question=question)
    if llm_client is None or not llm_client.is_available() or not evidence_payload:
        return fallback
    payload = {
        "question": question,
        "target_subject": target_subject,
        "request_scope": request_scope,
        "seed_fields": seed,
        "allowed_evidence": [
            {"memory_id": row.get("memory_id"), "text": row.get("text"), "disclosure_role": row.get("disclosure_role")}
            for row in evidence_payload
        ],
    }
    assert_runtime_payload_safe(payload, context="answer_request_compiler_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=ANSWER_REQUEST_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
        return fallback
    result = _normalize_request(raw, seed_fields=seed)
    return result if result.fields else fallback


def _normalize_projection(
    raw: object,
    *,
    request: AnswerRequest,
    evidence_payload: list[dict[str, Any]],
) -> tuple[FieldEvidenceProjection, ...]:
    payload = raw if isinstance(raw, dict) else {}
    by_id = {field.field_id: field for field in request.fields}
    by_label = {_norm(field.label): field for field in request.fields}
    rows_by_id = _source_rows(evidence_payload)
    result: list[FieldEvidenceProjection] = []
    seen: set[str] = set()
    for raw_field in _as_list(payload.get("fields")):
        if not isinstance(raw_field, dict):
            continue
        field = by_id.get(_text(raw_field.get("field_id"))) or by_label.get(_norm(raw_field.get("label")))
        if field is None or field.field_id in seen:
            continue
        seen.add(field.field_id)
        source_ids = tuple(dict.fromkeys(
            str(value) for value in _as_list(raw_field.get("source_memory_ids"))
            if str(value) in rows_by_id
        ))
        cited_rows = [rows_by_id[memory_id] for memory_id in source_ids]
        candidates = tuple(dict.fromkeys(
            _text(value) for value in _as_list(raw_field.get("candidate_values"))
            if _text(value) and cited_rows and _value_supported(_text(value), cited_rows)
        ))
        selected = tuple(dict.fromkeys(
            _text(value) for value in _as_list(raw_field.get("selected_values"))
            if _text(value) and cited_rows and _value_supported(_text(value), cited_rows)
        ))
        if re.search(r"\b(?:date|day|time|window|schedule|arrival|departure|deadline)\b", field.label, re.IGNORECASE):
            selected = tuple(dict.fromkeys(
                _qualify_schedule_value(value, cited_rows)
                for value in selected
            ))
        # A selected value without a valid source is never allowed to reach the
        # answer model. Keep the field unresolved so the final answer can say
        # that the authorized evidence is insufficient.
        status = _text(raw_field.get("status")).casefold()
        if selected:
            status = "supported"
        elif status not in _STATUSES:
            status = "unknown"
        provenance: list[dict[str, Any]] = []
        for memory_id in source_ids:
            row = rows_by_id[memory_id]
            provenance.append({
                "memory_id": memory_id,
                "source_message_ids": list(row.get("source_message_ids") or []),
                "source_turn_index": row.get("source_turn_index"),
                "source_span": _text(raw_field.get("source_span")),
            })
        result.append(FieldEvidenceProjection(
            field_id=field.field_id,
            label=field.label,
            status=status,
            candidate_values=candidates,
            selected_values=selected,
            source_memory_ids=source_ids,
            provenance=tuple(provenance),
            conflict_trace=tuple(_text(value) for value in _as_list(raw_field.get("conflict_trace")) if _text(value)),
        ))
    for field in request.fields:
        if field.field_id not in seen:
            result.append(FieldEvidenceProjection(
                field_id=field.field_id,
                label=field.label,
                status="unknown",
            ))
    return tuple(result)


def project_field_evidence(
    *,
    request: AnswerRequest,
    question: str,
    evidence_payload: list[dict[str, Any]],
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> AnswerProjection:
    """Bind each contract field to source-local values after retrieval."""
    if not request.fields or not evidence_payload:
        return AnswerProjection(request=request, errors=("empty_request_or_evidence",))
    if llm_client is None or not llm_client.is_available():
        return AnswerProjection(request=request, errors=("llm_unavailable",))
    projected: list[FieldEvidenceProjection] = []
    errors: list[str] = []
    # Separate calls prevent a long multi-field answer from making the model
    # spend its output budget on the first easy slot. The cost is deliberate:
    # Utility completeness is the current priority, while every call remains
    # bounded by the same already-authorized evidence set.
    for field in request.fields:
        payload = {
            "question": question,
            "field": asdict(field),
            "allowed_evidence": evidence_payload,
        }
        assert_runtime_payload_safe(payload, context="field_evidence_projection_prompt")
        try:
            raw = llm_client.chat_json(
                model=resolve_llm_model(config, "answering"),
                system_prompt=FIELD_PROJECTION_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
        except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
            errors.append(f"projection_call_failed:{field.field_id}")
            continue
        if isinstance(raw, dict) and "fields" not in raw:
            raw = {"fields": [raw]}
        one = _normalize_projection(
            raw,
            request=AnswerRequest(fields=(field,)),
            evidence_payload=evidence_payload,
        )
        if one:
            projected.append(one[0])
        else:
            errors.append(f"projection_schema_empty:{field.field_id}")
    if not projected:
        return AnswerProjection(request=request, errors=tuple(errors or ["projection_schema_empty"]))
    by_id = {field.field_id: field for field in projected}
    projected = [
        by_id.get(field.field_id)
        or FieldEvidenceProjection(field_id=field.field_id, label=field.label, status="unknown")
        for field in request.fields
    ]
    return AnswerProjection(request=request, fields=tuple(projected), errors=tuple(errors))


def projection_to_dict(projection: AnswerProjection) -> dict[str, Any]:
    return asdict(projection)


def projection_required_labels(projection: AnswerProjection) -> list[str]:
    return [field.label for field in projection.request.fields if field.required]


def projection_coverage_gaps(
    projection: AnswerProjection,
    *,
    answer_text: str,
    contract: dict[str, Any],
) -> list[str]:
    """Audit required field values after realization without generating text."""
    text = _norm(answer_text)
    gaps: list[str] = []
    contract_fields = list(contract.get("requested_fields") or [])
    for projected in projection.fields:
        field = next((item for item in contract_fields if _norm(item.get("label")) == _norm(projected.label)), None)
        request_field = next((item for item in projection.request.fields if item.field_id == projected.field_id), None)
        if request_field is None or not request_field.required:
            continue
        if projected.status == "supported" and projected.selected_values:
            for value in projected.selected_values:
                if _norm(value) not in text:
                    gaps.append(f"missing_projected_value:{projected.label}:{value}")
        elif projected.status in {"unknown", "restricted", "conflict"}:
            if not field or str(field.get("status") or "").casefold() not in {"unknown", "omitted"}:
                gaps.append(f"unresolved_projected_field:{projected.label}")
    return list(dict.fromkeys(gaps))
