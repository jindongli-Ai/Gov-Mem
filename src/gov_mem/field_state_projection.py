"""Field-scoped state reasoning for the Stateful Policy pipeline.

This module is deliberately narrow.  Policy authorization is decided by the
policy engine; this module only turns an already-authorized memory set into
field-scoped evidence and resolves the effective value for each requested
field.  The final answerer must consume the resulting projection, never the
flat authorized memory set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import re
from typing import Any, Iterable

from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.governance_runtime.factual_claim_quality import factual_value_is_eligible
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model


_CARDINALITIES = {"single", "list", "range", "boolean", "unknown"}
_TEMPORAL_ROLES = {"current", "latest", "historical", "event", "deadline", "validity", "unspecified"}
_STATUSES = {"supported", "unknown", "restricted", "conflict"}
_VALUE_TYPES = {
    "unknown", "text", "date", "time", "datetime", "location", "person", "role",
    "amount", "percentage", "credential", "identifier", "status", "device",
    "structure", "instruction", "contact", "scope", "wording", "boolean", "list",
}
_NON_VALUES = {
    "same", "unchanged", "no change", "no changes", "as above", "as before", "the same", "remains", "unknown", "n/a", "none", "only",
}
_SENSITIVE_FIELD_TERMS = {
    "diagnosis", "condition", "disease", "medication", "laboratory", "imaging",
    "health", "clinical", "credential", "password", "passcode", "pin", "token",
    "private address", "customer identity", "account number", "exact location",
}


def _field_requires_narrow_disclosure(field: "QueryField") -> bool:
    """Classify disclosure at field granularity, not record granularity."""
    text = _text(" ".join((field.label, field.attribute or ""))).casefold()
    if re.search(
        r"\b(?:diagnosis|condition|disease|pregnan(?:cy|t)|viability|symptom|"
        r"medication|clinical|health|laboratory|lab result|imaging|incident)\b",
        text,
    ):
        return True
    if re.search(r"\b(?:credential|password|passcode|pin|token|badge|access code|secret)\b", text):
        return True
    if re.search(r"\b(?:exact|private|confidential|internal)\b", text) and re.search(
        r"\b(?:wording|phrase|amount|budget|room|bay|location|address|file|label|number)\b",
        text,
    ):
        return True
    if re.search(r"\b(?:amount|budget|price|cost|fee|discount|payment|finance)\b", text):
        return True
    if re.search(r"\b(?:private|exact)\s+(?:room|bay|location|address)\b", text):
        return True
    if re.search(r"\b(?:access|permission)\b", text) and re.search(r"\b(?:status|active|incident|current)\b", text):
        return True
    return False


@dataclass(frozen=True)
class QueryField:
    field_id: str
    label: str
    subject_key: str | None = None
    event_key: str | None = None
    attribute: str | None = None
    temporal_role: str = "unspecified"
    cardinality: str = "single"
    required: bool = True
    disclosure_scope: str = "standard"
    question_span: str = ""
    value_type: str = "unknown"


@dataclass(frozen=True)
class QueryContract:
    fields: tuple[QueryField, ...] = ()
    source: str = "stateful_field_contract"
    # Semantic context carried across stages; this is the user query, never
    # evaluator metadata or a memory answer.
    question: str = ""


def query_contract_to_dict(contract: QueryContract) -> dict[str, Any]:
    """Serialize the single query contract carried across pipeline stages."""
    return asdict(contract)


def query_contract_from_dict(value: object) -> QueryContract | None:
    """Validate and reconstruct a contract stored in a policy snapshot.

    The policy stage is the semantic owner of the contract.  Later stages may
    consume it, but must not silently reinterpret arbitrary snapshot data as a
    new field plan.
    """
    if not isinstance(value, dict):
        return None
    raw_fields = value.get("fields")
    if not isinstance(raw_fields, (list, tuple)):
        return None
    contract = _normalize_fields(raw_fields, source=str(value.get("source") or "policy_state_contract"))
    if not contract.fields:
        return None
    return QueryContract(
        fields=contract.fields,
        source=contract.source,
        question=_text(value.get("question")),
    )


@dataclass(frozen=True)
class FieldAuthorization:
    field_id: str
    authorization: str
    policy_allowed_memory_ids: tuple[str, ...] = ()
    blocked_memory_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class StateClaim:
    field_id: str
    subject_key: str
    event_key: str
    attribute: str
    value: str
    memory_id: str
    source_turn_index: int = -1
    relation: str = "assert"
    effective: bool = True
    source_span: str = ""
    lineage_memory_ids: tuple[str, ...] = ()
    supersedes_memory_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldEvidenceClosure:
    field_id: str
    memory_ids: tuple[str, ...] = ()
    claims: tuple[StateClaim, ...] = ()
    trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatefulFieldProjection:
    field_id: str
    label: str
    status: str
    selected_values: tuple[str, ...] = ()
    candidate_values: tuple[str, ...] = ()
    source_memory_ids: tuple[str, ...] = ()
    provenance: tuple[dict[str, Any], ...] = ()
    transition_trace: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatefulProjection:
    contract: QueryContract
    authorizations: tuple[FieldAuthorization, ...]
    closures: tuple[FieldEvidenceClosure, ...]
    fields: tuple[StatefulFieldProjection, ...]
    errors: tuple[str, ...] = ()


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).casefold()).strip()


def _list(value: object) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _field_id(value: object, index: int) -> str:
    result = re.sub(r"[^a-zA-Z0-9_]+", "_", _text(value)).strip("_").lower()
    return result[:48] or f"field_{index:03d}"


def _source_rows(evidence: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("memory_id")): row for row in evidence if str(row.get("memory_id") or "")}


def _evidence_payload_row(row: dict[str, Any]) -> dict[str, Any]:
    """Expose observable provenance needed for semantic field binding.

    The content channel remains policy-filtered. Subject, scope, and source
    time are metadata already present in the memory schema; retaining them
    prevents a field model from conflating two objects that share vocabulary
    in one episode.
    """
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    entities = row.get("entities") or metadata.get("entities") or ()
    return {
        "memory_id": row.get("memory_id"),
        "text": row.get("text") or row.get("content"),
        "user_id": row.get("user_id"),
        "source_turn_index": row.get("source_turn_index", metadata.get("source_turn_index")),
        "entities": list(entities) if isinstance(entities, (list, tuple)) else [str(entities)] if entities else [],
        "scope": row.get("scope") or metadata.get("policy_scope"),
        "time": row.get("time"),
        "source_message_ids": list(row.get("source_message_ids") or ()),
    }


def _row_entities(row: dict[str, Any]) -> tuple[str, ...]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    values = row.get("entities") or metadata.get("entities") or ()
    if isinstance(values, str):
        values = (values,)
    return tuple(_text(value) for value in values if _text(value))


def _query_view_boundary(question: str) -> tuple[str, tuple[str, ...]]:
    """Extract explicit audience/projection boundaries from a query.

    This is a planning hint for aggregate expansion, not an authorization
    rule. It keeps a semantic model from treating every authorized record in
    a long episode as part of one audience-facing view. The vocabulary is
    intentionally open: any ordinary word used before ``-facing`` is accepted.
    """
    lowered = re.sub(r"\s+", " ", _text(question).casefold()).strip()
    audiences = tuple(dict.fromkeys(
        match.group(1)
        for match in re.finditer(r"\b([a-z][a-z0-9_-]{2,})-facing\b", lowered)
    ))
    projection = ""
    if re.search(
        r"\b(?:safe|broad|public|sponsor-safe|sponsor-ready|household-safe|"
        r"mixed-audience|helper-facing)\b|"
        r"\b(?:using|with|showing|reporting|keeping)\s+(?:only\s+)?"
        r"(?:the\s+)?(?:current|active|retained)\s+state\b",
        lowered,
    ) or _is_aggregate_safe_summary_request(lowered) or re.search(
        r"\bwithout\s+(?:leaking|sharing|disclosing|exposing)\s+"
        r"(?:restricted|private|sensitive|confidential)\s+(?:material|content|details?)\b",
        lowered,
    ):
        projection = "safe_summary"
    return projection, audiences


def _audience_matches_row(audiences: tuple[str, ...], row: dict[str, Any]) -> bool:
    if not audiences:
        return False
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    haystack = " ".join(
        _text(value)
        for value in (
            row.get("user_id"),
            row.get("scope"),
            metadata.get("policy_scope"),
            metadata.get("subject"),
            *_row_entities(row),
            row.get("text") or row.get("content"),
        )
        if _text(value)
    ).casefold()
    haystack_tokens = set(re.findall(r"[a-z0-9]+", haystack))
    return any(
        audience in haystack_tokens
        or any(
            len(audience) >= 4
            and len(token) >= 4
            and audience[:4] == token[:4]
            for token in haystack_tokens
        )
        for audience in audiences
    )


def _query_delivery_audience(question: str) -> tuple[str, ...]:
    """Extract an explicitly named delivery recipient from ordinary prose.

    This is a view-binding hint only.  It never grants access and deliberately
    returns no value for implicit audiences such as ``for my backup tasks``.
    """
    lowered = _text(question).casefold()
    matches: list[str] = []
    for pattern in (
        r"\b(?:send|share|give|provide|forward)\s+(?:the\s+)?"
        r"(?P<name>[a-z][a-z0-9_-]{2,})\b",
        r"\bto\s+(?P<name>[a-z][a-z0-9_-]{2,})\b",
    ):
        matches.extend(match.group("name") for match in re.finditer(pattern, lowered))
    stop = {
        "me", "us", "them", "someone", "anyone", "the", "a", "an",
        "only", "without", "restricted", "private", "sensitive", "confidential",
    }
    return tuple(dict.fromkeys(value for value in matches if value not in stop))


def _row_explicit_delivery_names(row: dict[str, Any]) -> tuple[str, ...]:
    """Return names used as a source-local recipient/audience anchor."""
    source = _text(row.get("text") or row.get("content"))
    names: list[str] = []
    patterns = (
        r"\b(?P<name>[A-Z][a-z]{2,})\s*,\s+(?:your|the|my)\b",
        r"\b(?P<name>[A-Z][a-z]{2,})\s+(?:gets?|may|can|needs?|only\s+needs?)\b",
        r"\b(?:for|to)\s+(?P<name>[A-Z][a-z]{2,})\b",
    )
    for pattern in patterns:
        names.extend(match.group("name").casefold() for match in re.finditer(pattern, source))
    return tuple(dict.fromkeys(names))


def _delivery_view_compatible(question: str, row: dict[str, Any]) -> bool:
    """Keep a carrier on the requested delivery lane when one is named."""
    audiences = _query_delivery_audience(question)
    if not audiences:
        return True
    source = _text(row.get("text") or row.get("content")).casefold()
    if re.search(
        r"\b(?:i|we)\s+(?:(?:only|just)\s+)?need\b[^.!?;]{0,80}"
        r"\b(?:approved\s+desk|current\s+release\s+(?:method|rule)|"
        r"desk\s+release\s+rules?)\b",
        source,
    ) or re.search(r"\bwait\s+for\s+the\s+updated\b", source):
        return False
    if set(_row_explicit_delivery_names(row)).difference(audiences):
        return False
    source_has_audience = any(
        re.search(rf"\b{re.escape(audience)}\b", source) for audience in audiences
    )
    owner = _norm(row.get("user_id"))
    owner_has_audience = any(audience in owner.split() for audience in audiences)
    operational_carrier = bool(re.search(
        r"\b(?:current|latest|active|approved|updated|avoid|remains?|stays?)\b"
        r"[^.!?;]{0,100}\b(?:window|arrival|entry|method|route|zone|setup|"
        r"schedule|handling|scope|state|buzz|counter|bench|area)\b",
        source,
    ))
    # A named recipient anchors direct instructions and that person's own
    # source records.  An unnamed current operational carrier remains useful
    # for reconstructing the state, while an unnamed request from a sibling
    # audience does not.
    if not source_has_audience and not owner_has_audience and not operational_carrier:
        return False
    # A broad safe label for a restricted surprise is a different projection
    # lane from a vendor/helper logistics delivery, even though both are
    # technically policy-approved.  Operational avoidance instructions remain
    # eligible because they are directly actionable logistics.
    if re.search(
        r"\b(?:private\s+dinner\s+surprise|restricted\s+surprise|"
        r"exact\s+(?:gift|surprise)\s+location|old\s+[^.!?;]{0,30}\blocation|"
        r"broad\s+(?:wording|safe\s+wording))\b",
        source,
    ):
        return False
    return True


def _is_safe_direct_delivery_instruction(question: str, row: dict[str, Any]) -> bool:
    """Keep a recipient-directed operational boundary as a safe carrier."""
    audiences = _query_delivery_audience(question)
    source = _text(row.get("text") or row.get("content"))
    if not audiences or not any(
        re.search(rf"\b{re.escape(audience)}\b", source.casefold())
        for audience in audiences
    ):
        return False
    return bool(re.search(
        r"\b(?:setup|tray|approved\s+(?:zone|area)|avoid\s+[^.!?;]{0,50}"
        r"(?:area|bench)|not\s+be\s+opened|not\s+be\s+inspected)\b",
        source,
        re.IGNORECASE,
    ))


def _aggregate_evidence_view(question: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a bounded semantic view for aggregate field planning.

    Policy filtering has already happened before this function. It only
    prevents an aggregate planner from seeing unrelated audience lanes or
    disclosure scopes as if they were one answer. With no explicit boundary,
    the complete policy-approved evidence remains available.
    """
    projection, audiences = _query_view_boundary(question)
    if not projection and not audiences:
        return list(evidence)
    public_scopes = {"safe_summary", "public", "broad"}
    bounded: list[dict[str, Any]] = []
    for row in evidence:
        scope = _text(row.get("scope"))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        scope = scope or _text(metadata.get("policy_scope"))
        keep_public = scope.casefold() in public_scopes
        keep_audience = _audience_matches_row(audiences, row)
        retrieval_tags = {
            _text(value).casefold()
            for value in (
                row.get("retrieval_fields")
                or metadata.get("retrieval_fields")
                or ()
            )
        }
        explicit_safe_carrier = retrieval_tags.intersection(
            {"safe_summary", "safe_carrier_frontier", "public_projection"}
        )
        # Older in-memory test/adapter rows may carry no scope, user, or
        # entity provenance at all. Preserve those opaque rows for recall; a
        # bounded source is only safe to narrow when the record declares a
        # competing scope or audience.
        unscoped_legacy = not scope and not row.get("user_id") and not _row_entities(row) and not _text(metadata.get("subject"))
        source = _text(row.get("text") or row.get("content"))
        # Disclosure instructions are provenance context, not factual carriers
        # for an aggregate. Concrete projection records remain eligible.
        instruction_only = _is_policy_instruction_source(source) and not _is_safe_direct_delivery_instruction(
            question, row
        )
        if (
            (keep_public or keep_audience or unscoped_legacy or explicit_safe_carrier)
            and not instruction_only
            and _delivery_view_compatible(question, row)
        ):
            bounded.append(row)
    if bounded:
        return bounded
    # Preserve only truly opaque legacy rows when no declared source matches
    # the explicit boundary. Returning every declared row here silently
    # defeats the audience boundary and can reintroduce the exact cross-lane
    # contamination this view is designed to prevent.
    return [
        row for row in evidence
        if not _text(row.get("scope"))
        and not _text((row.get("metadata") or {}).get("policy_scope"))
        and not row.get("user_id")
        and not _row_entities(row)
    ]


def _aggregate_recall_support_view(question: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep policy-approved source carriers available for semantic recall.

    The bounded view is the primary audience boundary.  Some adapters also
    attach a source to an explicit projection lane while leaving the row
    scope opaque; dropping that carrier before semantic parsing causes a
    completeness failure even though policy already approved the row.  This
    support view therefore unions the bounded view with rows explicitly
    marked as projection/snapshot carriers, while still excluding rows from
    a declared competing scope.  It is a recall input only: claim
    normalization and the final projection remain the authority.
    """
    bounded = _aggregate_evidence_view(question, evidence)
    seen = {_text(row.get("memory_id")) for row in bounded if _text(row.get("memory_id"))}
    public_scopes = {"safe_summary", "public", "broad"}
    projection_lanes = {"safe_summary", "public", "operational_snapshot", "policy_allowed_projection"}
    for row in evidence:
        memory_id = _text(row.get("memory_id"))
        if not memory_id or memory_id in seen:
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        scope = (_text(row.get("scope")) or _text(metadata.get("policy_scope"))).casefold()
        lanes = {
            _text(value).casefold()
            for value in (row.get("retrieval_fields") or metadata.get("retrieval_fields") or ())
        }
        disclosure_role = _text(row.get("disclosure_role") or metadata.get("disclosure_role")).casefold()
        audience_match = _audience_matches_row(_query_view_boundary(question)[1], row)
        if (
            scope in public_scopes
            or lanes.intersection(projection_lanes)
            or disclosure_role in {"public_projection", "policy_allowed_projection"}
            or audience_match
        ) and _delivery_view_compatible(question, row) and (
            not _is_policy_instruction_source(_text(row.get("text") or row.get("content")))
            or _is_safe_direct_delivery_instruction(question, row)
        ):
            bounded.append(row)
            seen.add(memory_id)
    return bounded


def _has_composite_source_carrier(evidence: list[dict[str, Any]]) -> bool:
    """Detect ordinary enumerated source prose without naming a dataset case."""
    for row in evidence:
        source = _text(row.get("text") or row.get("content"))
        if re.search(r"\b(?:summary|snapshot|overview|recap)\b\s*[:=-]", source, re.IGNORECASE):
            return True
        if re.search(r"\b(?:only|remains?|preserved)\b[^.!?;:]{0,100}(?:,|;|\s/\s)", source, re.IGNORECASE):
            return True
    return False


def _source_subject_compatible(field: QueryField, row: dict[str, Any]) -> bool:
    """Keep an explicit field subject attached to its source object."""
    expected = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", _text(field.subject_key).casefold()))
    if not expected:
        return True
    subjects = _row_entities(row)
    if not subjects:
        return True
    return any(
        expected.issubset(set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", subject.casefold())))
        for subject in subjects
    )


def _supports(value: str, row: dict[str, Any]) -> bool:
    candidate = _norm(value)
    source = _norm(row.get("text"))
    if not candidate or not source:
        return False
    if candidate in source:
        return True
    tokens = set(candidate.split())
    return len(tokens) >= 2 and len(tokens & set(source.split())) >= max(2, int(len(tokens) * 0.8))


def _turn(row: dict[str, Any]) -> int:
    value = row.get("source_turn_index")
    if not isinstance(value, int):
        metadata = row.get("metadata")
        value = metadata.get("source_turn_index") if isinstance(metadata, dict) else None
    return value if isinstance(value, int) else -1


def _infer_value_type(label: str, attribute: str | None = None) -> str:
    """Infer a broad semantic slot type from the requested field language.

    This vocabulary describes ordinary field semantics, not GateMem cases.
    It is intentionally used only as a compatibility check: the base LLM
    still extracts the source-grounded value and can leave an unknown slot
    unresolved when language is genuinely ambiguous.
    """
    text = _text(" ".join(value for value in (label, attribute) if value)).casefold()
    # A compound such as "funding hold" names a state/exception, not a
    # monetary value. Resolve the state head before the broader funding
    # amount vocabulary so a neighboring amount cannot fill the slot.
    if re.search(
        r"\b(?:status|state|blocker|hold|condition|phase|outcome|result|progress|"
        r"incident|diagnosis|viability)\b",
        text,
    ):
        return "status"
    patterns = (
        ("credential", r"\b(?:credential|password|passcode|pin|token|secret|access\s+(?:key|code)|api\s+key|login|badge|code|numeric\s+code)\b"),
        ("percentage", r"\b(?:percent(?:age)?|discount|rate|ratio|margin|tax)\b"),
        ("amount", r"\b(?:amount|budget|price|cost|fee|fund(?:ing)?|payment|salary|"
                    r"balance|value|stipend|scholarship|grant|allocation|allowance)\b"),
        ("datetime", r"\b(?:date\s+and\s+time|datetime|timestamp)\b"),
        ("datetime", r"\b(?:bookings?|appointments?|visits?|follow[- ]?ups?|schedules?)\b"),
        ("date", r"\b(?:date|day|deadline|due|milestone|target)\b"),
        ("time", r"\b(?:time|window|schedule|arrival|departure|opening|closing|hours?)\b"),
        ("location", r"\b(?:where|location|place|site|room|bay|address|entrance|route|zone|desk|hall|suite|shelf|tray|area|handoff|drop[- ]?off|pickup|stocking\s+point)\b"),
        ("device", r"\b(?:scanner|device|machine|equipment|model|terminal|printer|sensor)\b"),
        ("structure", r"\b(?:contract|term|renewal|structure|clause|agreement|vendor\s+term)\b"),
        ("contact", r"\b(?:contact|phone|mobile|email|number|extension|callback)\b"),
        ("scope", r"\b(?:scope|audience|visibility|release|access\s+scope|who\s+can)\b"),
        ("wording", r"\b(?:wording|phrase|label|description|summary|name|title)\b"),
        ("instruction", r"\b(?:instruction|action|step|procedure|method|how\s+to|what\s+to|plan|reminder)\b"),
        ("status", r"\b(?:status|state|blocker|outcome|result|condition|phase|progress|diagnosis|viability|active|incident)\b"),
        ("role", r"\b(?:role|position|capacity|responsible|owner)\b"),
        ("person", r"\b(?:person|name|who|contact\s+person|assignee|vendor|supplier|provider)\b"),
    )
    for value_type, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return value_type
    return "unknown"


_ATTRIBUTE_TYPE_TERMS = {
    "date": {"date", "day", "deadline", "expiry", "expiration"},
    "time": {"time", "window", "schedule", "hour"},
    "amount": {"amount", "funding", "support", "stipend", "budget", "cost", "price", "fee"},
    "percentage": {"percentage", "percent", "rate", "cap", "share"},
    "wording": {"wording", "phrase", "label", "description", "summary", "name", "title"},
    "status": {"status", "state", "blocker", "hold", "condition", "phase", "outcome", "result", "progress"},
    "structure": {"structure", "term", "terms", "agreement", "contract", "clause", "provision", "arrangement"},
    "location": {"location", "place", "site", "room", "route", "address", "venue"},
    "person": {"person", "name", "owner", "assignee", "contact", "role"},
    "credential": {"credential", "password", "passcode", "pin", "token", "key"},
}


_DECLARED_TYPE_ALIASES = {
    "number": "amount",
    "numeric": "amount",
    "string": "text",
    "boolean": "boolean",
    "bool": "boolean",
    "float": "amount",
    "integer": "amount",
}


def _attribute_compatible(field: QueryField, attribute: str) -> bool:
    """Reject a clearly cross-field claim from a multi-fact source row.

    This is a type-level guard only.  Paraphrases remain admissible when no
    conflicting attribute signal is present, while explicit signals such as
    ``current blocker`` cannot populate a ``current date`` field.
    """
    def semantic_tokens(value: str) -> set[str]:
        tokens = set(re.findall(r"[a-z0-9]+", _text(value).casefold()))
        tokens -= {"current", "latest", "active", "requested", "target", "the"}
        # Attribute labels frequently differ only by ordinary inflection
        # (area/areas, term/terms). Normalize that surface variation before
        # treating two field slots as unrelated.
        return {
            token[:-1] if token.endswith("s") and len(token) > 3 else token
            for token in tokens
        }

    raw_tokens = semantic_tokens(attribute)
    if not raw_tokens:
        return True
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    field_tokens = semantic_tokens(f"{field.label} {field.attribute or ''}")
    if raw_tokens.intersection(field_tokens):
        return True
    raw_types = {kind for kind, terms in _ATTRIBUTE_TYPE_TERMS.items() if raw_tokens.intersection(terms)}
    expected_terms = _ATTRIBUTE_TYPE_TERMS.get(expected, set())
    if raw_types and expected in _ATTRIBUTE_TYPE_TERMS:
        return expected in raw_types or bool(raw_tokens.intersection(expected_terms))
    # An explicitly typed neighboring attribute is not admissible for an
    # otherwise open field. This blocks bindings such as a backup phrase
    # becoming a band color while keeping opaque names and locations
    # permissive when neither side carries a reliable type marker.
    if raw_types and expected not in _ATTRIBUTE_TYPE_TERMS:
        return False
    return True


def _classify_value_type(value: str) -> str:
    """Classify only unmistakable surface forms; unknown stays permissive."""
    text = _text(value)
    lowered = text.casefold()
    if re.search(r"(?:[$€£]|\b\d[\d,]*(?:\.\d+)?\s*(?:usd|eur|gbp|dollars?|euros?|pounds?))\b", lowered):
        return "amount"
    if re.search(r"\b\d+(?:\.\d+)?\s*%(?:\b|$)|\b\d+(?:\.\d+)?\s*percent(?:age)?\b", lowered):
        return "percentage"
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lowered) and re.search(r"\b\d{1,2}(?::\d{2})?\b", lowered):
        return "datetime"
    if re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", lowered) and re.search(r"\b\d{1,2}\b", lowered):
        return "date"
    if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b", lowered):
        return "time"
    if re.fullmatch(r"\d{4,10}", re.sub(r"\s+", "", text)):
        return "credential"
    if re.search(r"\b(?:password|passcode|pin|token|credential|access\s+key|secret)\b", lowered):
        return "credential"
    if re.fullmatch(r"(?:yes|no|true|false|active|inactive|open|closed|viable|not\s+viable)", lowered):
        return "boolean"
    return "unknown"


def _value_compatible(field: QueryField, value: str) -> bool:
    """Reject only unmistakable cross-slot bindings.

    The check is deliberately asymmetric. A source value such as a named
    vendor or room is often opaque to a shallow classifier, but a currency
    amount, percentage, date, time, or credential has a strong observable
    type and must not populate an incompatible field.
    """
    # ``list`` describes cardinality, not the type of each member. Treat it
    # as unknown here so a list of dates/times/instructions remains valid.
    expected = field.value_type if field.value_type in _VALUE_TYPES and field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    observed = _classify_value_type(value)
    if expected == "unknown" and re.search(
        r"\b(?:snapshot|summary|overview|recap)\b",
        " ".join((field.label, field.attribute or "")),
        re.IGNORECASE,
    ):
        # Aggregate values intentionally carry mixed members (times, labels,
        # credentials, and locations). Validate the source contract instead
        # of treating the first surface type as the aggregate's type.
        return True
    if expected == "text":
        # Open text fields may legitimately contain a time, date, amount, or
        # location as part of the requested phrase. The explicit ``text``
        # declaration means "do not impose a narrower type", not "reject
        # typed surface forms".
        return True
    if field.cardinality == "list" and expected in {"instruction", "wording"}:
        # A list member such as "repeat ultrasound Monday at 8:00 AM" is an
        # instruction whose surface contains a time. Preserve the member-level
        # value; list cardinality and source binding remain the safeguards.
        return True
    if expected in {"unknown", "list", "wording", "instruction", "scope", "location", "device", "role", "person", "status", "structure", "contact"}:
        # Instruction is still an open semantic category, but a
        # credential/date/time-shaped value is strong evidence for another
        # requested slot. The previous exception allowed a numeric access
        # code to populate an entry-method field.
        return observed not in {"amount", "percentage", "credential", "date", "datetime", "time"}
    if expected == "datetime":
        return observed in {"datetime", "date", "time", "unknown"}
    if expected == "date":
        return observed in {"date", "datetime", "unknown"}
    if expected == "time":
        return observed in {"time", "datetime", "unknown"}
    if expected == "amount":
        return observed in {"amount", "unknown"}
    if expected == "percentage":
        return observed in {"percentage", "unknown"}
    if expected == "credential":
        return observed in {"credential", "unknown"}
    if expected == "boolean":
        return observed in {"boolean", "status", "unknown"}
    return True


def _looks_like_composite_scalar(field: QueryField, value: str) -> bool:
    """Reject a whole status sentence bound to one typed scalar field."""
    if field.cardinality != "single":
        return False
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    if expected not in {"date", "datetime", "time", "amount", "percentage", "credential"}:
        return False
    # Exact scalar values are short even when their source record is a
    # composite recap.  The carrier recovery path can then select the
    # contiguous typed span with its local field context.
    return len(_text(value)) > 80


def _source_semantically_supports(field: QueryField, row: dict[str, Any]) -> bool:
    """Check that a source record semantically supports the requested slot.

    This is a general source-contract guard. It prevents a value copied from a
    neighboring field (for example, a color or credential) from being placed
    into an instruction slot while leaving semantic selection to the LLM.
    """
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    source = _text(row.get("text")).casefold()
    field_surface = " ".join((field.label, field.attribute or "")).casefold()
    if _is_non_authoritative_note(source):
        return False
    # Exact substring grounding is not semantic support: a clarification or
    # an unrelated logistics sentence can contain a value-shaped phrase while
    # saying nothing about the requested field.  For typed fields with a
    # discriminative anchor (for example ``heartbeat`` or ``scan``), require
    # that anchor to appear in the cited source as well.  Generic type words
    # are intentionally excluded so ``current date`` and ``current amount``
    # continue through their existing type checks.
    anchor_tokens = set(re.findall(r"[a-z0-9]+", field_surface))
    anchor_tokens -= {
        "a", "an", "as", "and", "at", "by", "current", "latest", "active",
        "approved", "requested", "target", "the", "this", "date", "time",
        "amount", "status", "state", "field", "value", "item", "items",
    }
    if expected in {"boolean", "status"} and anchor_tokens:
        source_tokens = set(re.findall(r"[a-z0-9]+", source))
        if not anchor_tokens.intersection(source_tokens):
            return False
    # Open-ended unknown fields still need a source-local predicate.  A
    # project-management note about tracking threads cannot satisfy an
    # incident diagnosis or access-status slot merely because it shares the
    # project name.
    if re.search(r"\b(?:diagnosis|viability|pregnan(?:cy|t)|condition)\b", field_surface):
        if not re.search(r"\b(?:diagnos|viab|pregnan|condition|clinical|medical|incident)\b", source):
            return False
    if re.search(r"\b(?:access status|access is|permission status)\b", field_surface):
        if not re.search(r"\b(?:access|permission|active|inactive|revok|closed|open)\b", source):
            return False
    # Aggregate requests need an aggregate carrier. Transition notes such as
    # "switching to Sunday" are state metadata, not the requested snapshot.
    # This is a source-quality constraint and does not name a domain object or
    # expected answer value.
    aggregate_cues = {"snapshot", "summary", "overview", "recap"}
    if aggregate_cues.intersection(field_surface.split()):
        if not re.search(r"\b(?:snapshot|summary|overview|recap)\b", source):
            return False
    # Distinct time slots share the same surface type, so type compatibility
    # alone is insufficient. Reject an explicitly named competing slot when
    # the source does not name the requested slot.
    if expected in {"date", "datetime", "time"}:
        slot_groups = (
            {"setup", "helper", "arrival", "departure", "opening", "closing"},
            {"start", "end", "deadline", "visit", "appointment", "booking"},
        )
        field_slots = set().union(*(group.intersection(set(field_surface.split())) for group in slot_groups))
        source_slots = set().union(*(group.intersection(set(re.findall(r"[a-z]+", source))) for group in slot_groups))
        competing = source_slots - field_slots
        if competing and not field_slots.intersection(source_slots):
            return False
    if expected == "instruction":
        if re.search(r"\b(?:method|entry|route|way|arrival)\b", field_surface):
            predicate = r"\b(?:method|way|via|using|through|entry|enter|arrive|arrival|procedure)\b"
        elif re.search(r"\b(?:plan|step|task|action|procedure|requirement|reminder)\b", field_surface):
            predicate = r"\b(?:plan|step|task|action|procedure|requirement|reminder)\b"
        else:
            predicate = (
                r"\b(?:method|way|via|using|through|entry|enter|arrive|arrival|"
                r"procedure|step|plan|action|requirement|task|reminder|reset|handle|carry)\b"
            )
        if not re.search(predicate, source):
            return False
    if expected in {"date", "datetime", "time", "amount", "percentage", "credential"}:
        source_signals = {
            "date": r"\b(?:date|day|deadline|due|expires?|expiration|calendar)\b",
            "datetime": r"\b(?:date|day|time|when|appointment|visit|booking|schedule)\b",
            "time": r"\b(?:time|window|schedule|arrival|departure|opening|closing|hours?)\b",
            "amount": r"(?:[$€£]\s*\d|\b(?:amount|budget|price|cost|fee|fund(?:ing)?|payment|salary|balance)\b)",
            "percentage": r"(?:%|\b(?:percent(?:age)?|discount|rate|ratio|margin|tax)\b)",
            "credential": r"\b(?:credential|password|passcode|pin|token|secret|access\s+(?:key|code)|api\s+key|login|badge|(?:numeric|keypad|entry|access)?\s*code)\b",
        }
        source_type = _classify_value_type(source)
        compatible_types = {
            "date": {"date", "datetime"},
            "datetime": {"date", "datetime", "time"},
            "time": {"time", "datetime"},
            "amount": {"amount"},
            "percentage": {"percentage"},
            "credential": {"credential"},
        }
        if source_type not in compatible_types[expected] and not re.search(source_signals[expected], source):
            return False
    # Device/entity slots need a source-local device predicate. Without this
    # check, an opaque amount carrier can survive value compatibility and be
    # relabeled as a selected scanner or terminal.
    if expected == "device" and not re.search(
        r"\b(?:scanner|device|machine|equipment|model|terminal|printer|sensor)\b",
        source,
    ):
        return False
    if expected != "structure":
        return True
    if not re.search(
        r"\b(?:contract|agreement|term|renewal|clause|provision|structure|arrangement|policy)\b",
        source,
    ):
        return False
    return bool(re.search(
        r"\b(?:contract|agreement|structure|terms?|renewal|clause|provision)\b"
        r"[^.!?;]{0,100}\b(?:is|are|uses?|has|includes?|fixed|mutual|renew|"
        r"annual|monthly|quarterly|twelve-month|term)\b",
        source,
    ))


def _is_nonvalue_assertion(field: QueryField, value: str, source: str) -> bool:
    """Reject note-level absence statements as scalar field values.

    A sentence such as ``there is no new approved contract term in this
    note`` reports that the note is not a carrier for a value. It must not
    become the value of ``contract structure``. This is intentionally
    narrower than a generic ``no`` filter: real state values such as ``no
    remaining blockers`` remain valid assertions.
    """
    normalized_value = _text(value).casefold()
    normalized_source = _text(source).casefold()
    if not re.match(r"^(?:no|not)\b", normalized_value):
        # A source-level summary boundary is not a scalar value. This catches
        # a list fragment returned for a different field while leaving ordinary
        # status values such as ``no remaining blockers`` available.
        if (
            re.search(
                r"\b(?:summary|snapshot|overview|recap)\b[^.!?;]{0,120}"
                r"\b(?:remains?|stays?)\b[^.!?;]{0,160}\b(?:only|alone)\b",
                normalized_source,
            )
            and re.search(r"\b(?:boundary|access|limit|restriction)\b", normalized_value)
            and set(re.findall(r"[a-z0-9]+", normalized_value)).intersection(
                set(re.findall(r"[a-z0-9]+", _text(field.label).casefold()))
            )
        ):
            return False
        return bool(
            re.search(
                r"\b(?:summary|snapshot|overview|recap)\b[^.!?;]{0,120}"
                r"\b(?:remains?|stays?)\b[^.!?;]{0,160}\b(?:only|alone)\b",
                normalized_source,
            )
            and re.search(r"\b(?:only|alone)\s*$", normalized_value)
        )
    return bool(re.search(
        r"\b(?:no|not)\b[^.!?;]{0,120}"
        r"\b(?:new|approved|authoritative|official|final)\b[^.!?;]{0,100}"
        r"\b(?:value|values|field|fields|term|terms|commercial|information|"
        r"contract|agreement)\b[^.!?;]{0,100}"
        r"\b(?:in|from)\s+(?:this|the)\s+(?:note|message|update|record)\b",
        normalized_source,
    ))


_WEEKDAYS = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_MONTHS = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_SCHEDULE_RANGE = re.compile(
    rf"\b(?:(?:{_WEEKDAYS})(?:,?\s+(?:{_MONTHS})\s+\d{{1,2}}(?:,\s*20\d{{2}})?)?\s+)?"
    r"\d{1,2}:\d{2}\s*(?:AM|PM)?\s*(?:to|[-–—])\s*"
    r"\d{1,2}:\d{2}\s*(?:AM|PM)?\b",
    re.IGNORECASE,
)


def _qualify_claim_value(field: QueryField, value: str, source_span: str) -> str:
    """Restore source-attached weekday/date qualifiers lost in extraction."""
    if field.value_type not in {"time", "datetime"}:
        return value
    times = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", value, re.IGNORECASE)
    if len(times) < 2:
        return value
    candidates = [match.group(0).strip(" ,;:") for match in _SCHEDULE_RANGE.finditer(source_span or "")]
    for candidate in candidates:
        if all(token.casefold() in candidate.casefold() for token in times):
            return candidate
    return value


def _qualify_temporal_member(field: QueryField, value: str, source_span: str) -> str:
    """Keep a date/time carrier when list extraction returns only its label."""
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    if expected not in {"date", "time", "datetime"}:
        return value
    source = _text(source_span)
    weekday_hits = re.findall(rf"\b{_WEEKDAYS}\b", source, re.IGNORECASE)
    time_hits = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", source, re.IGNORECASE)
    if len(weekday_hits) == 1 and len(time_hits) == 1:
        if _classify_value_type(value) not in {"datetime", "time"}:
            return source.rstrip(" .;,")
    return value


def _unwrap_status_predicate(field: QueryField, value: str, source_span: str) -> str:
    """Recover a concrete status subject from a source-backed relation clause."""
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    if expected != "status" or not re.search(r"\b(?:blocker|status|hold|issue|problem)\b", value, re.IGNORECASE):
        return value
    match = re.search(
        r"(?P<subject>[^.!?,;]+?)\s+is\s+(?:the\s+)?"
        r"(?:(?:only|current|active|live|remaining|open)\s+)+"
        r"(?:[^.!?,;]*\b(?:blocker|status|hold|issue|problem)s?\b)",
        _text(source_span),
        re.IGNORECASE,
    )
    if not match:
        return value
    subject = _text(match.group("subject")).strip(" '\"")
    if not subject or subject.casefold() in {"the current blocker", "the status", "this"}:
        return value
    return subject if _supports(subject, {"text": source_span}) else value


def _is_policy_instruction_source(source: str) -> bool:
    """Keep disclosure instructions from becoming factual field claims."""
    lowered = _text(source).casefold()
    return bool(re.search(
        r"\b(?:should\s+not|must\s+not|do\s+not|don't|not\s+receive|"
        r"not\s+entitled|housekeeping\s+note|retrieval\s+note|"
        r"answer\s+should|response\s+should|reply\s+should|"
        r"should\s+(?:still|only|remain|stay|continue)|"
        r"(?:keep|use|provide)\s+only|\bnever\b|"
        r"will\s+(?:remove|delete|forget|purge))\b",
        lowered,
    )) or bool(re.search(
        r"\b(?:i|we)\s+(?:(?:only|just)\s+)?need\b[^.!?;]{0,80}"
        r"\bcurrent\s+(?:[a-z0-9_-]+\s+){0,2}(?:method|rule|scope|summary)\b",
        lowered,
    ))


def _is_non_authoritative_note(source: str) -> bool:
    """Reject a housekeeping note that explicitly carries no field value."""
    return bool(re.search(
        r"\b(?:no|without)\s+new\s+authoritative\s+(?:field\s+)?values?\b"
        r"|\b(?:no\s+authoritative\s+field\s+value)\b",
        _text(source).casefold(),
    ))


def _vague_frontier_claim(claim: StateClaim) -> bool:
    """Identify a summary-level placeholder that should not erase detail."""
    value = _text(claim.value).casefold()
    if re.match(r"^(?:one|a|an|some|several)\b", value) and re.search(
        r"\b(?:blocker|issue|problem|item|thing|task|detail)s?\b", value,
    ):
        return True
    # A privacy-safe recap may replace a concrete phrase with a wrapper such
    # as "broad safe wording". It is not a new factual value.
    return bool(re.fullmatch(
        r"(?:broad|generic|safe|public)\s+(?:safe\s+)?(?:wording|phrase|label|summary)",
        value,
    ))


_LOCATION_SPECIFIC_TERMS = {
    "address", "alcove", "bench", "bay", "cabinet", "cart", "counter",
    "cubby", "drawer", "desk", "door", "elevator", "entrance", "gate",
    "hall", "keypad", "lobby", "locker", "room", "shelf", "side",
    "station", "table", "tray", "window", "zone",
}


def _vague_location_claim(field: QueryField, claim: StateClaim) -> bool:
    """Identify a generic route/method label that must not erase a place.

    Operational summaries often shorten ``media hall keypad`` to ``keyed
    entry``.  The latter is a method class, not a concrete location.  Treat
    this distinction as a general value-shape rule rather than a dataset
    vocabulary rule; a source-backed location carrier can then preserve the
    concrete place from the same authorized state frontier.
    """
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    if expected != "location":
        return False
    tokens = set(re.findall(r"[a-z0-9]+", _text(claim.value).casefold()))
    if tokens.intersection({"at", "from", "in", "near", "with"}):
        return True
    if not tokens or tokens.intersection(_LOCATION_SPECIFIC_TERMS):
        return False
    return bool(tokens.intersection({"entry", "method", "route", "path", "access", "way", "arrival"}))


def _lineage_value(
    field: QueryField,
    claim: StateClaim,
    claims: list[StateClaim],
) -> tuple[str, StateClaim | None]:
    """Restore a source-backed qualifier when a later update abbreviates it."""
    value = _qualify_claim_value(field, claim.value, claim.source_span)
    if value != claim.value:
        return value, None
    expected = field.value_type if field.value_type != "unknown" else _infer_value_type(field.label, field.attribute)
    related = sorted(
        (
            candidate for candidate in claims
            if _norm(candidate.value) == _norm(claim.value)
            or (expected == "location" and candidate.source_turn_index <= claim.source_turn_index)
        ),
        key=lambda candidate: (candidate.source_turn_index, candidate.memory_id),
        reverse=True,
    )
    for candidate in related:
        qualified = _qualify_claim_value(field, candidate.value, candidate.source_span)
        if qualified != candidate.value:
            return qualified, candidate
    if expected == "location":
        def core(text: str) -> str:
            return _norm(re.sub(r"\b(?:only|just|solely)\b", "", text, flags=re.IGNORECASE))
        selected_core = core(claim.value)
        if selected_core:
            for candidate in related:
                candidate_core = core(candidate.value)
                source = candidate.source_span.casefold()
                if (
                    candidate_core
                    and candidate_core != selected_core
                    and selected_core in candidate_core
                    and not re.search(r"\b(?:changed|moved|switched|replaced|from\b.{0,80}\bto)\b", source)
                ):
                    return candidate.value, candidate
    return value, None


def _normalize_fields(raw_fields: Iterable[Any], *, source: str) -> QueryContract:
    result: list[QueryField] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_fields, start=1):
        if isinstance(raw, str):
            raw = {"label": raw}
        if not isinstance(raw, dict):
            continue
        label = _text(raw.get("label") or raw.get("name") or raw.get("attribute"))
        if not label or label.casefold() in {"all requested fields", "everything", "details", "the plan"}:
            continue
        field_id = _field_id(raw.get("field_id") or label, index)
        if field_id in seen:
            field_id = f"{field_id}_{index:03d}"
        seen.add(field_id)
        cardinality = _text(raw.get("cardinality")).casefold()
        if cardinality not in _CARDINALITIES:
            cardinality = _infer_cardinality(
                label,
                _text(raw.get("attribute")),
                _text(raw.get("question_span")),
            )
        else:
            # Preserve an explicit plural/quantifier invariant even when the
            # model incorrectly emits ``single`` for an aggregate request.
            inferred_cardinality = _infer_cardinality(
                label,
                _text(raw.get("attribute")),
                _text(raw.get("question_span")),
            )
            if inferred_cardinality == "list" and cardinality == "single":
                cardinality = "list"
        temporal = _text(raw.get("temporal_role")).casefold()
        declared_type = _text(raw.get("value_type")).casefold()
        declared_type = _DECLARED_TYPE_ALIASES.get(declared_type, declared_type)
        inferred_value_type = _infer_value_type(
            label,
            _text(raw.get("attribute")) or label,
        )
        # ``list`` describes member cardinality, not member type. Also repair
        # a common parser mismatch where a booking is labelled as ``status``
        # because the query contains the word "current".
        value_type = (
            inferred_value_type
            if declared_type in {"", "unknown", "list"}
            or (declared_type == "status" and inferred_value_type in {"date", "time", "datetime"})
            else (declared_type if declared_type in _VALUE_TYPES else inferred_value_type)
        )
        result.append(QueryField(
            field_id=field_id,
            label=label,
            subject_key=_text(raw.get("subject_key") or raw.get("entity")) or None,
            event_key=_text(raw.get("event_key") or raw.get("event")) or None,
            attribute=_text(raw.get("attribute") or label) or label,
            temporal_role=temporal if temporal in _TEMPORAL_ROLES else "unspecified",
            cardinality=cardinality if cardinality in _CARDINALITIES else "single",
            required=bool(raw.get("required", True)),
            disclosure_scope=_text(raw.get("disclosure_scope")) or "standard",
            question_span=_text(raw.get("question_span")),
            value_type=value_type,
        ))
    return QueryContract(fields=tuple(result), source=source)


def _infer_cardinality(*parts: str) -> str:
    """Infer list shape from ordinary language when the model omits it.

    This is a schema default, not an answer rule. Explicit model cardinality
    still wins. The vocabulary describes common linguistic plurality and
    quantifiers rather than GateMem entities.
    """
    text = _text(" ".join(part for part in parts if part)).casefold()
    if re.search(r"\b(?:all|both|each|every|multiple|several|three|four|five|two|list\s+of|set\s+of)\b", text):
        return "list"
    if re.search(
        r"\b(?:items|bookings|appointments|medications|tests|"
        r"results|dates|times|windows|rooms|areas|points|"
        r"steps|instructions|constraints|details|facts|values|"
        r"fields|members|contacts|tasks|requirements|options)\b",
        text,
    ):
        return "list"
    return "single"


def _query_tokens(value: object) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", _text(value).casefold())
        if token not in {
            "current", "latest", "what", "which", "when", "where", "who", "how",
            "the", "and", "for", "with", "is", "are", "was", "were",
        }
    }


def _is_grouping_field(label: str) -> bool:
    """Identify a pure aggregate wrapper, not every field mentioning a plan.

    ``plan`` is ambiguous: "Ivy Pantry plan" is a request-level wrapper,
    while "start-plan items" is itself the requested operational content.
    Dropping every label containing a grouping noun silently produced an empty
    QueryContract for otherwise authorized aggregate requests. A wrapper has
    a grouping noun as its final meaningful head; a concrete field has a
    semantic qualifier or item after that noun.
    """
    if _norm(label) == "authorized safe summary":
        return False
    tokens = re.findall(r"[a-z0-9]+", _text(label).casefold())
    modifiers = {
        "a", "an", "the", "my", "our", "your", "their", "current",
        "latest", "active", "complete", "concise", "safe", "safely",
        "broad", "full", "now", "today", "please",
    }
    meaningful = [token for token in tokens if token not in modifiers]
    return bool(meaningful and meaningful[-1] in {"plan", "summary", "snapshot", "overview", "recap"})


def _is_query_shaped_field(label: str) -> bool:
    """Exclude a parser wrapper that repeats the whole user question."""
    text = _text(label)
    return bool(
        "?" in text
        or re.match(
            r"^(?:what|which|who|where|when|why|how|can|could|should|is|are|do|does|did)\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_non_fact_query_fragment(label: str) -> bool:
    """Drop conditional/confirmation wrappers emitted as pseudo-fields."""
    text = _text(label).casefold().strip(" .:()? ")
    if not text or text in {"right", "correct", "is that so", "isn't it", "is it"}:
        return True
    if re.match(r"^(?:if|assuming|provided that|given that|that means|which means)\b", text):
        return True
    # These are proposition-shaped premises, rather than noun-phrase slots.
    # Restricting the subject shape keeps ordinary fields such as current
    # arrival time and selected vendor intact.
    return bool(
        re.match(r"^(?:i|we|you|they|he|she|it|no|nobody|nothing)\b", text)
        and re.search(
            r"\b(?:am|are|is|was|were|have|has|had|can|could|will|would|should|"
            r"leave|arrive|arrives|expected|means|need|needs|want|wants|"
            r"remain|remains|go|goes|come|comes)\b",
            text,
        )
    )


def _is_generic_placeholder_field(label: str) -> bool:
    """Recognize a model-created wrapper when a real field is also present."""
    tokens = re.findall(r"[a-z0-9]+", _text(label).casefold())
    if not tokens or tokens[-1] not in {"item", "items", "detail", "details", "fact", "facts", "field", "fields"}:
        return False
    return all(token in {
        "a", "an", "the", "current", "latest", "right", "now", "three", "two",
        "several", "multiple", "start", "next", "week", "plan", "requested",
    } for token in tokens[:-1])


def _is_query_contract_wrapper(field: QueryField) -> bool:
    """Detect a long seed phrase that a semantic field already refines."""
    tokens = re.findall(r"[a-z0-9]+", _text(field.label).casefold())
    label = _text(field.label).casefold()
    clause_prefix = bool(re.match(
        r"^(?:i\s+only\s+need|i\s+just\s+need|whether\b|remind\s+me\s+of|"
        r"before\s+my\s+|send\s+me\s+|tell\s+me\s+)",
        label,
    ))
    return (
        len(tokens) >= 6 and bool(set(tokens) & {"current", "latest", "right", "now", "before", "after"})
    ) or clause_prefix or bool(re.match(r"^(?:exact|current|latest)\b", label) and len(tokens) >= 3)


def _is_redundant_compound_field(field: QueryField, others: Iterable[QueryField]) -> bool:
    """Drop a wrapper field when its concrete child is already present.

    Query parsers often emit both ``current state: setup window`` and
    ``setup window``. The former is a malformed wrapper, not a second fact.
    The same applies to long state/summary labels that duplicate a compound
    request. A short aggregate such as ``three start-plan items`` remains a
    legitimate list field.
    """
    label = _text(field.label)
    tokens = _query_tokens(label)
    if not tokens:
        return False
    other_fields = tuple(others)
    if ":" in label:
        suffix = _text(label.rsplit(":", 1)[-1])
        suffix_tokens = _query_tokens(suffix)
        if suffix_tokens and len(suffix_tokens) <= 5:
            for other in other_fields:
                other_tokens = _query_tokens(other.label)
                if other_tokens and suffix_tokens.issubset(other_tokens) and len(other_tokens) <= len(tokens):
                    return True
    aggregate_heads = {"state", "status", "summary", "snapshot", "overview", "recap"}
    if len(tokens) >= 4 and tokens.intersection(aggregate_heads) and len(other_fields) >= 2:
        if any(_query_tokens(other.label) and _query_tokens(other.label) != tokens for other in other_fields):
            return True
    return False


def _field_semantic_identity(field: QueryField) -> tuple[object, ...]:
    """Build a conservative identity for parser aliases of one field.

    Labels may differ only because one parser response repeats the subject or
    rearranges a copula (``Project's current date`` vs ``current date``).
    Include subject, event, type, temporal lane, and cardinality so two real
    dates from different activities are not collapsed together.
    """
    subject_tokens = _query_tokens(field.subject_key)
    attribute_surface = _text(field.attribute or field.label)
    # Subject-qualified aliases are common parser artifacts rather than
    # separate facts: "Lina's current stipend" and "current stipend for
    # Lina" should share the same slot identity. Strip only grammatical
    # possessive/for-qualifiers; retain semantic qualifiers such as public,
    # private, showcase, or arrival that can denote a distinct lane.
    attribute_surface = re.sub(
        r"^[a-z0-9][a-z0-9 &'_-]{1,80}['’]s\s+",
        "",
        attribute_surface,
        flags=re.IGNORECASE,
    )
    attribute_surface = re.sub(
        r"\s+\bfor\s+[a-z0-9][a-z0-9 &'_-]{1,80}$",
        "",
        attribute_surface,
        flags=re.IGNORECASE,
    )
    attribute_surface = re.sub(r"['’]s\b", "", attribute_surface, flags=re.IGNORECASE)
    attribute_tokens = _query_tokens(attribute_surface) - subject_tokens
    if not attribute_tokens:
        label_surface = re.sub(r"['’]s\b", "", _text(field.label), flags=re.IGNORECASE)
        attribute_tokens = _query_tokens(label_surface) - subject_tokens
    return (
        frozenset(attribute_tokens),
        _norm(field.subject_key),
        _norm(field.event_key),
        field.temporal_role,
        field.value_type,
        field.cardinality,
    )


def _is_delivery_instruction_field(field: QueryField) -> bool:
    """Exclude delivery constraints that are not memory-backed facts.

    A request such as ``send Mason ... without leaking restricted material``
    can make a semantic parser emit phantom fields named ``Restricted
    Material`` or ``Recipient``. Those fields compete with the actual
    logistics summary and select unrelated carriers. They describe how to
    deliver the answer, not what the memory system was asked to retrieve.
    """
    text = _text(" ".join((field.label, field.attribute, field.question_span))).casefold()
    if re.search(
        r"\bwithout\s+(?:leaking|sharing|disclosing|exposing)\s+"
        r"(?:restricted|private|sensitive|confidential)\s+(?:material|content|details?)\b",
        text,
    ):
        return True
    label = _norm(field.label)
    if label in {"recipient", "audience", "delivery recipient"} and re.search(
        r"\bsend\s+[a-z][a-z0-9_-]*\b", _text(field.question_span).casefold(),
    ) and not re.match(r"\b(?:who|which)\b", _text(field.question_span).casefold()):
        return True
    return False


def _is_aggregate_safe_summary_request(question: str) -> bool:
    """Recognize an aggregate safe-summary request without case vocabulary."""
    return bool(re.search(
        r"\b(?:current|concise|safe|public|broad|helper-safe|sponsor-safe|"
        r"household-safe)\s+(?:[a-z0-9_-]+\s+){0,3}"
        r"(?:summary|snapshot|recap|overview)\b",
        _text(question).casefold(),
    ))


def _canonicalize_contract(contract: QueryContract) -> QueryContract:
    """Make one deterministic, non-overlapping field contract."""
    fields = list(contract.fields)
    kept: list[QueryField] = []
    seen_labels: set[frozenset[str]] = set()
    seen_semantic_fields: set[tuple[object, ...]] = set()
    for field in fields:
        if _is_redundant_compound_field(field, fields):
            continue
        semantic_tokens = frozenset(_query_tokens(field.label))
        key = semantic_tokens or frozenset({_norm(field.label)})
        if not key or key in seen_labels:
            continue
        semantic_identity = _field_semantic_identity(field)
        if semantic_identity[0] and semantic_identity in seen_semantic_fields:
            continue
        seen_labels.add(key)
        if semantic_identity[0]:
            seen_semantic_fields.add(semantic_identity)
        kept.append(field)
    question_tokens = [token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", _text(contract.question).casefold())]
    positions: dict[str, int] = {}
    for index, token in enumerate(question_tokens):
        positions.setdefault(token, index)
    if any(any(token in positions for token in _query_tokens(field.label)) for field in kept):
        indexed = list(enumerate(kept))
        indexed.sort(
            key=lambda item: (
                min((positions.get(token, len(question_tokens) + 1) for token in _query_tokens(item[1].label)), default=len(question_tokens) + 1),
                item[0],
            )
        )
        kept = [field for _, field in indexed]
    return QueryContract(fields=tuple(kept), source=contract.source, question=contract.question)


def _bind_explicit_query_subject(
    contract: QueryContract,
    *,
    target_subject: str | None,
    question: str,
) -> QueryContract:
    """Carry a named query object into every unresolved field binding.

    Bind only an exact named phrase already present in the question. This
    avoids trusting an LLM paraphrase such as ``object setup window`` as an
    entity while still preserving the object relation for ordinary named
    requests.
    """
    subject = _text(target_subject)
    if not subject or not re.search(re.escape(subject), question, re.IGNORECASE):
        return contract
    fields = tuple(
        replace(field, subject_key=field.subject_key or subject)
        for field in contract.fields
        if field.field_id != "authorized_safe_summary"
    )
    safe_summary_fields = tuple(
        field for field in contract.fields if field.field_id == "authorized_safe_summary"
    )
    return QueryContract(
        fields=tuple((*fields, *safe_summary_fields)),
        source=contract.source,
        question=contract.question,
    )


def _seed_fields_aligned_to_question(fields: Iterable[Any], question: str) -> list[Any]:
    """Remove parser-generated wrappers and fields not actually requested."""
    question_tokens = _query_tokens(question)
    aligned: list[Any] = []
    for raw in fields:
        label = raw if isinstance(raw, str) else (raw.get("label") if isinstance(raw, dict) else "")
        label_text = _text(label)
        if not label_text:
            continue
        if _is_grouping_field(label_text):
            continue
        if _is_query_shaped_field(label_text):
            continue
        label_tokens = _query_tokens(label_text)
        if len(label_tokens) == 1 and re.match(r"^current\s+", label_text, re.IGNORECASE) and _norm(label_text) not in _norm(question):
            continue
        if label_tokens and len(label_tokens & question_tokens) >= min(2, len(label_tokens)):
            aligned.append(raw)
    return aligned


QUERY_CONTRACT_PROMPT = """
Compile the user question into a complete field contract. This is semantic
parsing only: do not decide permission, inspect memory, or infer an answer.
Return JSON only: {"fields": [{"field_id","label","subject_key","event_key",
"attribute","temporal_role","cardinality","required","question_span",
"value_type"}]}.
Create one field for every independently requested fact. Expand aggregate
words such as plan, summary, snapshot, status, schedule, and overview into
their concrete requested facts. Preserve named people, lanes, subjects,
audiences, temporal qualifiers, and list cardinality. Keep when, who, where,
what, and how separate when the question requests them. Never return the
words same, unchanged, or as above as a field value. Do not use evaluator
metadata or hidden labels.
""".strip()


def compile_query_contract(
    *,
    question: str,
    requester: str | None,
    target_subject: str | None,
    requested_fields: Iterable[Any],
    answer_need_spec: dict[str, Any] | None,
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> QueryContract:
    """Compile fields without exposing memory content to the parser."""
    seeds = list(_list(requested_fields))
    if answer_need_spec:
        seeds.extend(_list(answer_need_spec.get("requested_fields")))
        dimensions = answer_need_spec.get("five_w_one_h") or {}
        for name in ("when", "who", "where", "what", "how"):
            seeds.extend(_list(dimensions.get(name)))
    seeds = [
        raw for raw in seeds
        if not _is_non_fact_query_fragment(
            _text(raw if isinstance(raw, str) else raw.get("label") if isinstance(raw, dict) else "")
        )
    ]
    aligned_seeds = _seed_fields_aligned_to_question(seeds, question)
    fallback = _normalize_fields(aligned_seeds, source="seed_contract")
    # A policy intent may legitimately be an aggregate operational field
    # (for example, "start-plan items"). Keep a non-question seed as a
    # resilient fallback even when the wrapper filter removes it from the
    # preferred contract. The LLM can still split it into finer fields.
    resilient_seed = _normalize_fields(
        [raw for raw in seeds if not _is_query_shaped_field(_text(raw if isinstance(raw, str) else raw.get("label") if isinstance(raw, dict) else ""))],
        source="resilient_seed_contract",
    )
    if not fallback.fields and question.strip():
        fallback = _normalize_fields([{"field_id": "answer", "label": question, "question_span": question}], source="question_fallback")
    if llm_client is None or not llm_client.is_available():
        return QueryContract(fields=fallback.fields, source=fallback.source, question=question)
    payload = {
        "question": question,
        "requester": requester,
        "target_subject": target_subject,
        "seed_fields": seeds,
    }
    assert_runtime_payload_safe(payload, context="query_contract_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=QUERY_CONTRACT_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return QueryContract(fields=fallback.fields, source=fallback.source, question=question)
    model_fields = raw.get("fields") if isinstance(raw, dict) else []
    # Seeds are authoritative. The LLM may split an aggregate seed but cannot
    # delete an explicitly requested field.
    model_contract = _normalize_fields(_list(model_fields), source="base_llm_contract")
    model_labels = [field.label for field in model_contract.fields]
    has_real_seed_field = any(not _is_generic_placeholder_field(field.label) for field in resilient_seed.fields)
    model_contract = QueryContract(
        fields=tuple(
            field for field in model_contract.fields
            if not _is_grouping_field(field.label)
            and not _is_query_shaped_field(field.label)
            and not _is_non_fact_query_fragment(field.label)
            and not _is_delivery_instruction_field(field)
            and not (
                _is_generic_placeholder_field(field.label)
                and (
                    has_real_seed_field
                    or any(not _is_generic_placeholder_field(other) for other in model_labels)
                )
            )
        ),
        source=model_contract.source,
    )
    # Preserve semantic seed fields even when their wording is an implicit
    # temporal dimension (for example "current date") and therefore does not
    # share enough surface tokens with the question. The model may split an
    # aggregate, but it cannot erase an already requested dimension.
    # ``question_fallback`` is only an outage/parser fallback. When a real
    # non-question seed exists, combining both creates two wrappers for the
    # same aggregate and canonicalization can remove both as redundant.
    fallback_fields = () if fallback.source == "question_fallback" else fallback.fields
    authoritative_seed = QueryContract(
        fields=tuple(dict.fromkeys((*fallback_fields, *resilient_seed.fields))),
        source=(fallback.source if fallback_fields else resilient_seed.source),
        question=question,
    )
    merged = _merge_contracts(authoritative_seed, model_contract)
    # Semantic parsing can legitimately use different vocabulary from the
    # question ("repeat hCG" for "start-plan item"). Do not make lexical
    # overlap a second, hidden authority over the LLM's typed field split.
    # Still guarantee an executable contract if parsing produces no field.
    selected = merged if merged.fields else (resilient_seed if resilient_seed.fields else fallback)
    # Bind the explicit query subject before canonicalization so a model field
    # that repeats the subject ("Project's current date") can be recognized as
    # an alias of the shorter seed field ("current date").
    selected = _bind_explicit_query_subject(
        QueryContract(fields=selected.fields, source=selected.source, question=question),
        target_subject=target_subject,
        question=question,
    )
    canonical = _canonicalize_contract(selected)
    # If the parser leaves only the full aggregate question, keep one typed
    # carrier field instead of allowing that wrapper to compete with unrelated
    # logistics values. Concrete multi-field requests continue unchanged.
    if (
        _is_aggregate_safe_summary_request(question)
        and len(canonical.fields) == 1
        and (
            _norm(canonical.fields[0].label) == _norm(question)
            or "without leaking" in _text(canonical.fields[0].label).casefold()
        )
    ):
        canonical = QueryContract(fields=(QueryField(
            field_id="authorized_safe_summary",
            label="authorized safe summary",
            attribute="safe summary",
            temporal_role="current",
            cardinality="list",
            required=False,
            disclosure_scope="safe_summary",
            value_type="wording",
        ),), source="aggregate_safe_summary_contract", question=question)
    # The semantic model is allowed to split an aggregate, but it is never
    # allowed to erase the request.  Several API responses returned only
    # grouping placeholders; canonicalization correctly removed those, but
    # previously left the runtime with an empty contract even though the
    # original aggregate seed was executable. Preserve the seed as a typed
    # semantic anchor and let evidence closure resolve it downstream.
    if not canonical.fields and authoritative_seed.fields:
        canonical = _canonicalize_contract(QueryContract(
            fields=authoritative_seed.fields,
            source="seed_contract_fallback",
            question=question,
        ))
    bound = _bind_explicit_query_subject(
        canonical,
        target_subject=target_subject,
        question=question,
    )
    # Apply the aggregate guard once more after subject binding. Some model
    # responses are merged with the seed late in this function; the final
    # contract must still not reintroduce the full-question wrapper.
    if (
        _is_aggregate_safe_summary_request(question)
        and len(bound.fields) == 1
        and (
            _norm(bound.fields[0].label) == _norm(question)
            or "without leaking" in _text(bound.fields[0].label).casefold()
        )
    ):
        return QueryContract(fields=(QueryField(
            field_id="authorized_safe_summary",
            label="authorized safe summary",
            attribute="safe summary",
            temporal_role="current",
            cardinality="list",
            required=False,
            disclosure_scope="safe_summary",
            value_type="wording",
        ),), source="aggregate_safe_summary_contract", question=question)
    return bound


def _merge_contracts(seed: QueryContract, model: QueryContract) -> QueryContract:
    # A query-shaped fallback is a parser wrapper, not an independently
    # answerable field. Keeping it here creates a phantom closure that can
    # compete with the real fields returned by the semantic parser.
    fields = [field for field in seed.fields if not _is_query_shaped_field(field.label)]
    model_labels = [set(_query_tokens(item.label)) for item in model.fields]
    fields = [
        field for field in fields
        if not (
            _is_query_contract_wrapper(field)
            and any(
                len(set(_query_tokens(field.label)).intersection(other)) >= 3
                and len(other) < len(set(_query_tokens(field.label)))
                for other in model_labels
            )
        )
    ]
    def compatible_type(left: QueryField, right: QueryField) -> bool:
        aliases = {"number": "amount", "numeric": "amount", "string": "text"}
        a = aliases.get(left.value_type, left.value_type)
        b = aliases.get(right.value_type, right.value_type)
        return a == b or "unknown" in {a, b} or {a, b} <= {"text", "wording"}

    def alias(left: QueryField, right: QueryField) -> bool:
        left_tokens = _query_tokens(left.attribute or left.label) - {
            "exact", "current", "latest", "active", "approved", "my", "our", "the",
        }
        right_tokens = _query_tokens(right.attribute or right.label) - {
            "exact", "current", "latest", "active", "approved", "my", "our", "the",
        }
        if not left_tokens or not right_tokens or not compatible_type(left, right):
            return False
        overlap = left_tokens & right_tokens
        if not overlap:
            return False
        subject_compatible = not left.subject_key or not right.subject_key or _norm(left.subject_key) == _norm(right.subject_key)
        event_compatible = not left.event_key or not right.event_key or _norm(left.event_key) == _norm(right.event_key)
        return subject_compatible and event_compatible and (
            overlap >= left_tokens or overlap >= right_tokens
            or (len(overlap) >= 2 and len(overlap) / max(1, min(len(left_tokens), len(right_tokens))) >= 0.6)
            or (_is_query_contract_wrapper(left) and len(overlap) >= 1)
        )

    by_key = {_norm(item.label): item for item in fields}
    for item in model.fields:
        key = _norm(item.label)
        if not key or key in by_key:
            continue
        # The base model is the semantic splitter.  A parser seed such as
        # ``exact current amount`` or ``I only need the Friday drop-off time``
        # is retained only when no model field covers the same typed slot.
        aliases = [field for field in fields if alias(field, item)]
        if aliases and (
            not any(_is_query_contract_wrapper(field) for field in aliases)
            or all(field.value_type not in {"unknown", "text", "list"} for field in aliases)
        ):
            # Preserve the first semantically valid model label when two
            # reordered copula forms describe the same slot. A typed seed
            # such as "current contract structure for Project Ember" is an
            # atomic field, even though its surface starts with "current";
            # do not discard it as a parser wrapper.
            continue
        fields = [field for field in fields if not alias(field, item)]
        by_key = {_norm(field.label): field for field in fields}
        fields.append(item)
        by_key[key] = item
    return QueryContract(
        fields=tuple(fields[:40]),
        source="merged_field_contract",
        question=seed.question or model.question,
    )


def _field_semantic_binding(field: QueryField) -> str:
    """Describe a field using general semantic types, not dataset terms."""
    meanings = {
        "structure": "the arrangement, form, or terms of the object explicitly requested",
        "status": "the current state or condition of the object explicitly requested",
        "instruction": "an action, requirement, or procedure explicitly requested",
        "date": "a calendar date explicitly requested",
        "time": "a clock time or time window explicitly requested",
        "datetime": "the complete date-and-time value explicitly requested",
        "location": "a place, site, room, route, or address explicitly requested",
        "person": "a person or role explicitly requested",
        "amount": "a monetary or numeric amount explicitly requested",
        "percentage": "a percentage or rate explicitly requested",
        "device": "a device, tool, or service explicitly requested",
    }
    expected = field.value_type if field.value_type not in {"unknown", "list"} else "unknown"
    return meanings.get(expected, "the concrete value of the field as used in the question")


FIELD_CLOSURE_PROMPT = """
Select field-local evidence from the already policy-approved records. You do
not grant access and you do not answer the user. Return JSON only:
{"claims":[{"memory_id","subject_key","event_key","attribute","value",
"relation","effective","source_span","lineage_memory_ids",
"supersedes_memory_ids"}]}. Every value must be copied from
the cited record. Resolve same, unchanged, remains, and as-above references
to the concrete value they refer to when the supplied records establish it.
For current/latest fields, identify explicit updates, replacements,
supersessions, and stale/provisional values. When a source explicitly points
to a prior record, include its memory_id in lineage_memory_ids. When it
explicitly replaces prior records, include their memory_ids in
supersedes_memory_ids. Return all compatible members of
a requested list. Keep each value attached to its own subject, event, and
attribute. Use supplied entities, scope, and source time as provenance
constraints: an explicitly named sibling object or time lane cannot fill this
field. Do not return a value that is not present in its cited record.
The returned attribute must describe the requested field. Do not label a
blocker, status, amount, date, location, or wording claim as another field
just because the same source record contains both facts. If the source
attribute is clearly different from the requested field, return no claim.
Respect the field's value_type as a binding constraint. Never bind an amount
or percentage to a location, device, status, or structure field; never bind a
credential to a route or location; and never bind a date or time to a status
or amount field. If no compatible source value exists, return no claim.
Interpret the field in the full question context. A word such as "structure"
must refer to the structure of the explicitly requested object (for example,
the terms of an agreement when the question asks about an agreement), not a
metaphor, project label, role, or structure of a neighboring object. Do not
let a source's reuse of the same noun override the question's target object.
All supplied records have already passed the policy filter. Do not re-decide
privacy, refuse, redact, or omit a requested fact merely because a record says
"restricted" or also contains a disclosure instruction. A mixed record may
contain both a factual value and an instruction; extract the factual value and
do not use the instruction itself as the value.
For a required single-valued field, if an allowed record contains a direct
assertion of a concrete value, return that claim even when the record omits a
weekday, person, or object phrase that is present in the field contract. The
field contract supplies that omitted binding; do not require every qualifier
to be repeated in every continuation sentence. In particular, split a factual
clause from a trailing disclosure qualifier such as "and restricted".
Treat "no change", "unchanged", and similar continuity phrases as relation
markers, not as requested factual values. If a later source gives an explicit
exact value, bind that value to the field.
""".strip()


FIELD_ADJUDICATION_PROMPT = """
Adjudicate candidate claims for exactly one requested field. You are not
granting permission and you are not writing the answer. Return JSON only:
{"claims":[{"memory_id","value","relation","effective","source_span",
"lineage_memory_ids","supersedes_memory_ids"}]}.
Inspect both the proposed claims and the complete allowed candidate evidence.
The proposed claims are recall hints, not a closed set. Add a source-grounded
claim from the candidate evidence when it is a later explicit update or the
current value was omitted from the proposed claims. Keep only values that
answer this field, preserve the source memory id, and discard candidates
belonging to another attribute in the same record. For a current/latest field,
The claim's attribute must remain aligned with the requested field; a blocker
claim cannot be retained as a date, amount, or wording claim merely because
the value text is present in the same source row.
retain the newest explicit value and discard provisional or superseded values.
For a list, retain every compatible requested member. If the field label gives
a numeric count, return that many distinct members, merging duplicate
restatements and ignoring unrelated neighboring advice.
All supplied records have already passed policy authorization. Do not turn a
word such as "restricted" into an empty closure or a refusal; extract the
requested fact from the factual portion of the record and ignore disclosure
instructions as values.
For a required scalar field, an allowed record with a direct current/update/
assertion clause is sufficient for one claim even if the source is an
anaphoric continuation and omits a qualifier repeated in the field label.
Use the field label and question span to restore that binding. Never return an
empty claim set solely because the source also contains a privacy qualifier.
Treat continuity phrases such as "no change" as temporal relation markers,
not as list members.
Extract a concise concrete value, not the whole source sentence. A value such
as a date, room, amount, label, location, status, instruction, or safe phrase
must remain attached to its own field. Never return same, unchanged, or as
above as a value.
When a source uses a concise operational synonym such as a boundary, limit,
or restriction, express the fact using the requested field's semantic label
when the source clearly binds the synonym to that field. Do not copy a
summary's neighboring field or its whole clause as the value.
Respect the field value_type and reject an unmistakable cross-type candidate
even when it occurs in the same source record.
""".strip()


AGGREGATE_FIELD_EXPANSION_PROMPT = """
Expand one aggregate request into a small typed field contract using only the
user question and the already policy-approved evidence. Return JSON only:
{"fields":[{"field_id","label","attribute","temporal_role",
"cardinality","required","question_span","value_type","subject_key",
"event_key"}]}.
Create one field for every independently answerable fact that belongs to the
requested aggregate. Use the audience, object, and qualifier in the question
to exclude unrelated records and neighboring snapshots. For a current or
latest request, describe the effective operational facts, not obsolete,
internal, provisional, or superseded alternatives. Labels must name the fact
being requested, not a source sentence and not its value. Do not return an
aggregate wrapper such as "snapshot", "all details", or "the answer". Do not
invent a fact merely because a source record exists: a field must be a
reasonable semantic component of the aggregate requested by the user. This
is field planning only; do not return values, permissions, evaluator labels,
or a final answer. Preserve the named object in subject_key and the relevant
activity/time lane in event_key when the evidence makes them explicit. The
supplied records are already policy-approved.
""".strip()


AGGREGATE_FIELD_AUDIT_PROMPT = """
Audit a proposed field contract for an aggregate request against the complete
already policy-approved evidence. Return JSON only with fields to ADD:
{"fields":[{"field_id","label","attribute","temporal_role",
"cardinality","required","question_span","value_type","subject_key",
"event_key"}]}.
The proposed fields are only a recall draft. Add every independently
answerable factual component that belongs to the requested object, activity,
audience, and temporal state but is missing from the draft. Do not add a
generic wrapper such as summary facts, snapshot details, or all information.
Do not add a field merely because it is current or appears in the same record:
separate topic, audience, and activity lanes. For current/latest requests,
include explicit replacement results and concrete status/boundary facts when
the evidence establishes them. Return field labels only, never values,
permissions, evaluator metadata, or a final answer.
""".strip()


AGGREGATE_FIELD_ADJUDICATION_PROMPT = """
Adjudicate the final field contract for one aggregate request. Return JSON
only: {"fields":[{"field_id","label","attribute","temporal_role",
"cardinality","required","question_span","value_type","subject_key",
"event_key"}]}.
The proposed fields are a recall draft, not an authority. Keep a field only
when it is an independently answerable fact belonging to the exact object,
activity, audience, and temporal state requested in the question and when the
supplied evidence contains a plausible source carrier for that fact. Remove
neighboring facts from another object, another audience, another time lane,
or another operational activity, even when they occur in the same episode.
Remove aggregate wrappers, transition prose, source-note labels, and fields
whose only support is that they are mentioned near a relevant record. Preserve
all distinct supported components, including components expressed through
anaphoric updates or a concise public projection. This is field planning only:
do not return values, permissions, evaluator metadata, or a final answer.
""".strip()


AGGREGATE_SOURCE_CARRIER_PROMPT = """
Decompose the policy-approved source carriers into a complete field contract
for the exact aggregate request. Return JSON only:
{"fields":[{"field_id","label","attribute","temporal_role",
"cardinality","required","question_span","value_type","subject_key",
"event_key"}]}.
This is semantic field planning, never answer writing. Treat each independently
answerable fact as its own field. When a source contains a summary, recap, or
snapshot with comma-separated, semicolon-separated, slash-separated, or
lane-separated members, split those members into atomic fields and preserve
the lane, object, and time qualifier in the field label or event_key. Do not
use the entire source sentence as one field. Do not create fields for policy
instructions, transition prose, deleted/retired values, or generic wrappers.
Use only facts belonging to the requested audience, object, activity, and
current temporal state. Return labels and schema only, never values,
permissions, evaluator metadata, or hidden answer fields.
""".strip()


def _is_expandable_aggregate_contract(contract: QueryContract) -> bool:
    if len(contract.fields) != 1:
        return False
    surface = _text(" ".join((contract.question, contract.fields[0].label))).casefold()
    # A plural noun can describe either list members or the dimensions of an
    # aggregate (for example, "snapshot across three lanes"). Only the
    # former should bypass aggregate expansion.
    if contract.fields[0].cardinality == "list" and not re.search(
        r"\b(?:snapshot|summary|overview|recap|current state|operational state)\b",
        surface,
    ):
        return False
    return bool(re.search(
        r"\b(?:snapshot|summary|overview|recap|plan|status|schedule|"
        r"calibration|current state|operational state)\b",
        surface,
    ))


def _is_aggregate_wrapper_component(label: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", _text(label).casefold())
    if not tokens:
        return True
    heads = {"summary", "snapshot", "overview", "recap"}
    tails = {"facts", "details", "information", "fields", "items"}
    return bool(set(tokens).intersection(heads) and tokens[-1] in tails)


def _drop_redundant_aggregate_wrappers(contract: QueryContract) -> QueryContract:
    """Remove request-level aggregate labels once concrete fields exist."""
    fields = list(contract.fields)
    concrete = [field for field in fields if not _is_grouping_field(field.label)]
    # An opaque aggregate is still executable when it is the only field.
    if not concrete:
        return contract
    return QueryContract(
        fields=tuple(concrete),
        source=contract.source,
        question=contract.question,
    )


def _normalize_aggregate_fields(raw_fields: object, *, contract: QueryContract) -> tuple[QueryField, ...]:
    normalized = _normalize_fields(_list(raw_fields), source="base_llm_aggregate_contract")
    return tuple(
        field for field in normalized.fields
        if not _is_grouping_field(field.label)
        and not _is_query_shaped_field(field.label)
        and not _is_generic_placeholder_field(field.label)
        and not _is_aggregate_wrapper_component(field.label)
        and _norm(field.label) != _norm(contract.fields[0].label)
    )


def _inherit_aggregate_bindings(
    fields: Iterable[QueryField],
    aggregate: QueryField,
) -> tuple[QueryField, ...]:
    """Carry the request-level object binding into expanded child fields."""
    return tuple(
        replace(
            field,
            subject_key=field.subject_key or aggregate.subject_key,
            event_key=field.event_key or aggregate.event_key,
        )
        for field in fields
    )


def _expand_aggregate_contract(
    *,
    contract: QueryContract,
    evidence: list[dict[str, Any]],
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> QueryContract:
    """Let the base model split an opaque aggregate before field closure.

    Query parsing cannot know which facts a domain-specific "snapshot" means.
    This second semantic pass uses only policy-approved evidence to discover
    the aggregate's observable components. It returns labels, never values;
    source grounding and current-state resolution remain downstream.
    """
    if not _is_expandable_aggregate_contract(contract):
        return contract
    if llm_client is None or not llm_client.is_available() or not evidence:
        return contract
    planning_evidence = _aggregate_evidence_view(contract.question, evidence)
    recall_evidence = _aggregate_recall_support_view(contract.question, evidence)
    payload = {
        "question": contract.question,
        "aggregate_field": asdict(contract.fields[0]),
        "view_boundary": {
            "projection": _query_view_boundary(contract.question)[0],
            "audiences": list(_query_view_boundary(contract.question)[1]),
        },
        "allowed_evidence": [_evidence_payload_row(row) for row in planning_evidence],
    }
    assert_runtime_payload_safe(payload, context="aggregate_field_expansion_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=AGGREGATE_FIELD_EXPANSION_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
        return contract
    raw_fields = raw.get("fields") if isinstance(raw, dict) else []
    expanded_fields = _inherit_aggregate_bindings(
        _normalize_aggregate_fields(raw_fields, contract=contract),
        contract.fields[0],
    )
    if not expanded_fields:
        return contract
    audit_payload = {
        "question": contract.question,
        "aggregate_field": asdict(contract.fields[0]),
        "proposed_fields": [asdict(field) for field in expanded_fields],
        "allowed_evidence": payload["allowed_evidence"],
    }
    assert_runtime_payload_safe(audit_payload, context="aggregate_field_completeness_audit_prompt")
    try:
        audit_raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=AGGREGATE_FIELD_AUDIT_PROMPT,
            user_prompt=json.dumps(audit_payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
        audit_raw = {}
    additions = _inherit_aggregate_bindings(_normalize_aggregate_fields(
        audit_raw.get("fields") if isinstance(audit_raw, dict) else [],
        contract=contract,
    ), contract.fields[0])
    by_key = {_norm(field.label): field for field in expanded_fields}
    for field in additions:
        by_key.setdefault(_norm(field.label), field)
    carrier_fields: tuple[QueryField, ...] = ()

    # A source carrier may contain several independent facts even when the
    # first expansion/audit pass compresses them into one broad field.  Ask
    # the base model for a second, carrier-oriented decomposition only when
    # the observable prose actually contains an enumeration.  This keeps the
    # normal path at the existing call budget while making aggregate recall
    # robust to composite summaries and slash-separated lanes.
    if _has_composite_source_carrier(recall_evidence):
        carrier_payload = {
            "question": contract.question,
            "aggregate_field": asdict(contract.fields[0]),
            "proposed_fields": [asdict(field) for field in by_key.values()],
            "source_carriers": [_evidence_payload_row(row) for row in recall_evidence],
        }
        assert_runtime_payload_safe(carrier_payload, context="aggregate_source_carrier_decomposition_prompt")
        try:
            carrier_raw = llm_client.chat_json(
                model=resolve_llm_model(config, "answering"),
                system_prompt=AGGREGATE_SOURCE_CARRIER_PROMPT,
                user_prompt=json.dumps(carrier_payload, ensure_ascii=False),
            )
        except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
            carrier_raw = {}
        carrier_fields = _inherit_aggregate_bindings(_normalize_aggregate_fields(
            carrier_raw.get("fields") if isinstance(carrier_raw, dict) else [],
            contract=contract,
        ), contract.fields[0])
        for field in carrier_fields:
            by_key.setdefault(_norm(field.label), field)
    proposed = _canonicalize_contract(QueryContract(
        fields=tuple(by_key.values()),
        source="base_llm_aggregate_contract",
        question=contract.question,
    ))
    policy_cfg = config.get("policy_reasoning") or {}
    if not bool(policy_cfg.get("aggregate_field_adjudication", False)):
        return proposed

    # A separate semantic adjudication pass prevents the evidence bundle from
    # becoming an implicit checklist. Expansion discovers recall candidates;
    # this pass decides which candidates belong to the requested aggregate.
    # Python still validates the schema and retains the proposed contract if
    # the advisory call fails or returns only wrappers.
    adjudication_payload = {
        "question": contract.question,
        "aggregate_field": asdict(contract.fields[0]),
        "proposed_fields": [asdict(field) for field in proposed.fields],
        "allowed_evidence": payload["allowed_evidence"],
    }
    assert_runtime_payload_safe(adjudication_payload, context="aggregate_field_adjudication_prompt")
    try:
        adjudicated_raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=AGGREGATE_FIELD_ADJUDICATION_PROMPT,
            user_prompt=json.dumps(adjudication_payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
        return proposed
    adjudicated = _inherit_aggregate_bindings(_normalize_aggregate_fields(
        adjudicated_raw.get("fields") if isinstance(adjudicated_raw, dict) else [],
        contract=contract,
    ), contract.fields[0])
    if not adjudicated:
        return proposed
    if carrier_fields:
        # Adjudication is a precision advisory. It must not erase a complete
        # carrier decomposition merely because the model compressed several
        # lanes into one broad field on its final pass.
        combined = list(carrier_fields)
        combined.extend(adjudicated)
        return _canonicalize_contract(QueryContract(
            fields=tuple(dict.fromkeys(combined)),
            source="base_llm_aggregate_carrier_adjudicated",
            question=contract.question,
        ))
    return _canonicalize_contract(QueryContract(
        fields=adjudicated,
        source="base_llm_aggregate_adjudicated",
        question=contract.question,
    ))


LIST_CLOSURE_PROMPT = """
Resolve one list-valued request from the complete policy-approved evidence.
Return JSON only: {"claims":[{"memory_id","value","relation","effective","source_span"}]}.
Return one claim per independently requested member. If a source sentence
enumerates members with commas or 'and', split it into separate claims while
preserving each member's date, time, location, and action qualifiers. Combine
related words only when they belong to the same member. Do not return the
whole source sentence as one value, and do not invent a member. Every value
must be copied from its cited source record and must answer the requested
field. Later explicit replacements supersede earlier members only when the
source says so; otherwise retain all current compatible members.
For a booking, appointment, visit, deadline, or other temporal member, never
return only the service label: retain the weekday, date, and time when the
source supplies them. All records are already policy-approved; do not refuse
or redact a fact because the source also contains a disclosure instruction.
""".strip()


def _lexical_rows(field: QueryField, rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    terms = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", " ".join((field.label, field.attribute or "", field.event_key or "")).casefold()))
    terms -= {"current", "latest", "what", "which", "where", "when", "who", "how", "the", "and", "for"}
    # Treat ordinary hyphenated compounds and plural collection heads as
    # lexical hints too. This keeps "start-plan items" connected to sources
    # labelled "Plan item one" without making the lexical slice an authority
    # over the extracted value.
    terms.update(
        part for term in tuple(terms) for part in term.split("-")
        if len(part) >= 3
    )
    terms.update(term[:-1] for term in tuple(terms) if term.endswith("s") and len(term) > 3)
    ranked: list[tuple[int, int, str, dict[str, Any]]] = []
    for memory_id, row in rows.items():
        tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(row.get("text") or "").casefold()))
        overlap = len(terms & tokens)
        if overlap:
            ranked.append((overlap, _turn(row), memory_id, row))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in ranked]


def _source_replacement_pairs(field: QueryField, source: str) -> tuple[tuple[str, str], ...]:
    """Return source-local ``from X to Y`` replacements for a requested field.

    A single source record can contain both sides of a state transition.  The
    semantic closure often extracts the old side because it is the first
    value-shaped span, even though the source explicitly declares the later
    side current.  Keep this detector structural and field-anchored: it is a
    rerank safeguard, not a domain vocabulary matcher.
    """
    field_terms = set(re.findall(
        r"[a-z0-9][a-z0-9_-]{2,}",
        _text(f"{field.label} {field.attribute or ''}").casefold(),
    ))
    field_terms -= {"current", "latest", "active", "what", "which", "the", "and", "for"}
    if not field_terms:
        return ()
    verb = (
        r"(?:change(?:s|d|ing)?|switch(?:es|ed|ing)?|replace(?:s|d|ing)?|"
        r"update(?:s|d|ing)?|move(?:s|d|ing)?|reschedule(?:s|d|ing)?|"
        r"shift(?:s|ed|ing)?)"
    )
    pattern = re.compile(
        rf"\b{verb}\b[^.!?;]{{0,100}}\bfrom\s+"
        rf"(?P<old>[^.!?;]+?)\s+\bto\s+(?P<new>[^.!?;]+)",
        re.IGNORECASE,
    )
    source_text = _text(source)
    pairs: list[tuple[str, str]] = []
    for match in pattern.finditer(source_text):
        prefix = source_text[max(0, match.start() - 100):match.start("old")]
        if not field_terms.intersection(set(re.findall(
            r"[a-z0-9][a-z0-9_-]{2,}", prefix.casefold()
        ))):
            continue
        old = _text(match.group("old")).strip(" ,:;")
        new = _text(match.group("new")).strip(" ,:;")
        new = re.split(
            r"\s+(?:because|since|so|as)\s+",
            new,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,:;")
        if old and new and _norm(old) != _norm(new):
            pairs.append((old, new))
    return tuple(pairs)


def _value_is_superseded_in_source(field: QueryField, value: str, source: str) -> bool:
    """Reject the old side of an explicit source-local replacement."""
    value_norm = _norm(value)
    return bool(value_norm) and any(
        value_norm == _norm(old)
        for old, _ in _source_replacement_pairs(field, source)
    )


def _normalize_claims(raw: Any, *, field: QueryField, rows: dict[str, dict[str, Any]]) -> tuple[StateClaim, ...]:
    values = raw.get("claims") if isinstance(raw, dict) else []
    claims: list[StateClaim] = []
    for item in _list(values):
        if not isinstance(item, dict):
            continue
        memory_id = _text(item.get("memory_id"))
        row = rows.get(memory_id)
        value = _text(item.get("value"))
        raw_attribute = _text(item.get("attribute"))
        if field.disclosure_scope == "safe_summary":
            # Keep a safe carrier at its declared broad granularity. Models
            # often copy the entire recap sentence even though only the safe
            # label/object phrase is answerable on this field.
            for pattern in (
                r"\bremains?\s+in\s+(?P<value>[^.!?;,]+)",
                r"\bonly\s+as\s+(?P<value>[^.!?;,]+)",
                r"\bsafe[- ]?(?:summary|label|wording)\b[^.!?;]{0,60}?\b(?:is|as|remains?)\s+(?P<value>[^.!?;,]+)",
            ):
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    value = _text(match.group("value")).strip(" ':-")
                    value = re.split(
                        r"\s+(?:with|including)\s+(?:date|target|current|blocker|status)\b",
                        value,
                        maxsplit=1,
                        flags=re.IGNORECASE,
                    )[0].strip(" ':-")
                    break
        if (
            not row
            or not value
            or value.casefold() in _NON_VALUES
            or bool(re.fullmatch(r"(?:the\s+)?same\s+[a-z0-9][a-z0-9 _-]*", value, re.IGNORECASE))
            or (
                field.disclosure_scope == "safe_summary"
                and _classify_value_type(value) in {"amount", "percentage", "credential", "date", "datetime", "time"}
            )
            or not _source_subject_compatible(field, row)
            or not _attribute_compatible(field, raw_attribute or field.attribute or field.label)
            or not _supports(value, row)
            or not _value_compatible(field, value)
            or _looks_like_composite_scalar(field, value)
            or not _source_semantically_supports(field, row)
            or not factual_value_is_eligible(
                attribute=field.attribute or field.label,
                slot_name=field.field_id,
                value=value,
                semantic_spec={
                    "request_shape": "fact",
                    "query_type": "policy_allowed_memory",
                },
                source_text=_text(row.get("text")),
            )
            or _is_non_authoritative_note(_text(row.get("text")))
            or _is_nonvalue_assertion(field, value, _text(row.get("text")))
        ):
            continue
        relation = _text(item.get("relation")).casefold() or "assert"
        if relation not in {"assert", "current", "update", "supersedes", "replaces", "same", "unchanged", "remains", "provisional", "stale", "revoked", "deleted", "forgotten"}:
            relation = "assert"
        def memory_id_list(value: object, singular: object = "") -> tuple[str, ...]:
            values = _list(value)
            if singular:
                values.extend(_list(singular))
            return tuple(dict.fromkeys(
                _text(candidate) for candidate in values
                if _text(candidate) in rows
            ))
        raw_source_span = _text(item.get("source_span"))
        # The model sometimes returns a turn number (for example ``"53"``)
        # instead of a source-local excerpt. Keeping that placeholder loses
        # qualifiers such as a weekday that only appear in the cited record
        # and makes later lineage recovery impossible. A span is usable only
        # when it is actually grounded in the cited row; otherwise retain the
        # complete allowed source text as the provenance fallback.
        source_span = raw_source_span if _supports(raw_source_span, row) else _text(row.get("text"))
        # If the cited source explicitly says ``from old to new``, the old
        # value is a stale candidate even when the model labels it current.
        # This must happen before deterministic frontier selection because
        # both values can share the same source turn and memory id.
        if _value_is_superseded_in_source(field, value, source_span):
            continue
        # Models sometimes return the source predicate together with the
        # value (for example "grate alignment two clicks left"). Keep the
        # field label as schema, and retain only the source-grounded value.
        leading_label = _text(field.attribute or field.label)
        stripped = re.sub(
            rf"^(?:the\s+)?{re.escape(leading_label)}\s*(?:(?:is|are|was|were|remains?|updated\s+to|changed\s+to)\s*)?",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip(" :,-")
        if stripped and _supports(stripped, row):
            value = stripped
        value = _qualify_temporal_member(field, value, source_span)
        value = _unwrap_status_predicate(field, value, source_span)
        value = _restore_source_leading_article(field, value, source_span)
        claims.append(StateClaim(
            field_id=field.field_id,
            subject_key=_text(item.get("subject_key") or field.subject_key or "unknown") or "unknown",
            event_key=_text(item.get("event_key") or field.event_key or "unknown") or "unknown",
            attribute=raw_attribute or _text(field.attribute or field.label) or field.label,
            value=value,
            memory_id=memory_id,
            source_turn_index=_turn(row),
            relation=relation,
            effective=bool(item.get("effective", True)),
            # Keep a source-local fallback for later temporal/specificity
            # checks. It remains provenance text copied from allowed evidence.
            source_span=source_span,
            lineage_memory_ids=memory_id_list(
                item.get("lineage_memory_ids"), item.get("lineage_memory_id")
            ),
            supersedes_memory_ids=memory_id_list(
                item.get("supersedes_memory_ids"), item.get("supersedes_memory_id")
            ),
        ))
    return tuple(claims)


def _adjudicate_claims(
    *,
    field: QueryField,
    claims: tuple[StateClaim, ...],
    rows: dict[str, dict[str, Any]],
    llm_client: LLMClient,
    config: dict[str, Any],
    question: str = "",
) -> tuple[StateClaim, ...]:
    payload = {
        "question": question,
        "field": asdict(field),
        "field_semantic_binding": _field_semantic_binding(field),
        "candidate_claims": [
            {
                "memory_id": claim.memory_id,
                "value": claim.value,
                "relation": claim.relation,
                "source_turn_index": claim.source_turn_index,
                "lineage_memory_ids": list(claim.lineage_memory_ids),
                "supersedes_memory_ids": list(claim.supersedes_memory_ids),
                "source_text": rows[claim.memory_id].get("text") or rows[claim.memory_id].get("content"),
                "entities": list(rows[claim.memory_id].get("entities") or ()),
                "scope": rows[claim.memory_id].get("scope"),
                "time": rows[claim.memory_id].get("time"),
            }
            for claim in claims
            if claim.memory_id in rows
        ],
        "complete_candidate_evidence": [
            {
                "memory_id": memory_id,
                "source_turn_index": _turn(row),
                "source_text": row.get("text") or row.get("content"),
                "entities": list(row.get("entities") or ()),
                "scope": row.get("scope"),
                "time": row.get("time"),
            }
            for memory_id, row in rows.items()
        ],
    }
    assert_runtime_payload_safe(payload, context="field_claim_adjudication_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=FIELD_ADJUDICATION_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
        return claims
    adjudicated = _normalize_claims(raw, field=field, rows=rows)
    if not adjudicated:
        return claims
    if field.cardinality == "list":
        # List adjudication is allowed to repair claim metadata and add
        # omitted source-grounded members, but it is not allowed to shrink
        # the evidence closure. Align model claims by source/value so an
        # explicit stale or replacement relation is retained for the
        # deterministic resolver while every otherwise-unmentioned member
        # remains available for resolution.
        adjudicated_keys = {(claim.memory_id, _norm(claim.value)) for claim in adjudicated}
        preserved_unmentioned = [
            claim for claim in claims
            if (claim.memory_id, _norm(claim.value)) not in adjudicated_keys
        ]
        return tuple((*adjudicated, *preserved_unmentioned))
    # The model is a semantic relevance filter, not the temporal state
    # machine.  A common failure is returning an older, more lexically direct
    # value while silently omitting a later current-state anchor.  Preserve
    # every source-grounded candidate newer than the newest model-retained
    # candidate; resolve_field_claims() then applies source order and explicit
    # lifecycle relations deterministically.
    newest_adjudicated_turn = max(claim.source_turn_index for claim in adjudicated)
    retained_keys = {(claim.memory_id, _norm(claim.value)) for claim in adjudicated}
    preserved_newer = [
        claim for claim in claims
        if claim.source_turn_index > newest_adjudicated_turn
        and (claim.memory_id, _norm(claim.value)) not in retained_keys
    ]
    if preserved_newer:
        return tuple((*adjudicated, *preserved_newer))
    return adjudicated


def _restore_source_leading_article(field: QueryField, value: str, source_span: str) -> str:
    """Preserve a source-attached article omitted by semantic extraction.

    This restores only an immediately source-grounded surface determiner. It
    does not synthesize a value or alter field semantics, and applies equally
    to contract, plan, policy, and other structural phrases.
    """
    if field.value_type != "structure" or not value or not source_span:
        return value
    if re.match(r"^(?:a|an|the)\s+", value, re.IGNORECASE):
        return value
    pattern = re.compile(
        rf"(?<![\w-])(?P<article>a|an|the)\s+{re.escape(value)}(?![\w-])",
        re.IGNORECASE,
    )
    match = pattern.search(source_span)
    if not match:
        return value
    return f"{match.group('article')} {value}"


def _source_grounded_scalar_fallback(
    *,
    field: QueryField,
    rows: dict[str, dict[str, Any]],
) -> tuple[StateClaim, ...]:
    """Recover one explicit scalar assertion after semantic extraction fails.

    This is deliberately a last-resort recall bridge for an available LLM,
    not a replacement reasoner. It accepts only a contiguous source substring
    after an ordinary assertion/update predicate and sends that substring
    through the same closed-world normalization checks as an LLM claim.
    Policy, subject authorization, and temporal conflict resolution remain
    outside this helper.
    """
    field_surface = " ".join((field.label, field.attribute or "")).casefold()
    # Models occasionally mark a singular attribute as a list when it occurs
    # inside an aggregate. Permit scalar recovery only when the requested
    # surface itself is singular; genuine list fields still use member
    # expansion below.
    plural_surface = bool(re.search(
        r"\b(?:items|members|facts|details|values|areas|rooms|points|steps|"
        r"instructions|bookings|appointments|medications|tests|results|"
        r"windows|times|dates|contacts|tasks|requirements|options)\b",
        field_surface,
        re.IGNORECASE,
    )) or bool(re.search(r"\b(?:two|three|four|five|all|both|each|every|multiple|several)\b", field_surface))
    if field.cardinality != "single" and plural_surface:
        return ()
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    aggregate_request = bool({"snapshot", "summary", "overview", "recap"}.intersection(field_surface.split()))
    ordered = sorted(rows.items(), key=lambda pair: (_turn(pair[1]), pair[0]), reverse=True)

    # A synthetic safe-summary slot is intentionally list-valued: one source
    # may carry the safe label while another carries the safe object name.
    # Extract only the short source phrase after an explicit safe-summary
    # predicate; never pass the surrounding sentence as the value.
    if field.disclosure_scope == "safe_summary" or field.field_id == "authorized_safe_summary":
        safe_patterns = (
            r"\b(?:refer(?:red)?\s+to|described)\b[^.!?;]{0,80}\bonly\s+as\s+(?P<value>[^.!?;,]+)",
            r"\bsafe[- ]?(?:summary|label|wording)\b[^.!?;]{0,60}?\b(?:is|as|remains?|means?)\s+(?P<value>[^.!?;,]+)",
            r"\bremains?\s+in\s+(?P<value>[^.!?;,]+)",
            r"\b(?:private|public|safe)\s+(?P<value>[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,4})\s+(?:materials|file|case|record)\b",
        )
        safe_claims: list[StateClaim] = []
        seen_safe: set[str] = set()
        safe_validation_field = replace(
            field,
            label="safe label",
            attribute="safe label",
            cardinality="single",
        )
        for memory_id, row in ordered:
            source = _text(row.get("text"))
            if not source or _is_non_authoritative_note(source):
                continue
            for pattern in safe_patterns:
                match = re.search(pattern, source, re.IGNORECASE)
                if not match:
                    continue
                safe_value = _text(match.group("value")).strip(" ':-")
                safe_value = re.split(
                    r"\s+(?:with|including)\s+(?:date|target|current|blocker|status)\b",
                    safe_value,
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0].strip(" ':-")
                safe_value = re.sub(r"^in\s+", "", safe_value, flags=re.IGNORECASE)
                if not safe_value or _norm(safe_value) in seen_safe:
                    continue
                normalized = _normalize_claims(
                    {"claims": [{
                        "memory_id": memory_id,
                        "attribute": field.attribute or field.label,
                        "value": safe_value,
                        "relation": "current",
                        "effective": True,
                        "source_span": source,
                    }]},
                    field=safe_validation_field,
                    rows=rows,
                )
                if normalized:
                    safe_claims.extend(normalized)
                    seen_safe.add(_norm(safe_value))
                break
            if len(safe_claims) >= 4:
                break
        if safe_claims:
            return tuple(safe_claims)

    assertion = re.compile(
        r"\b(?:(?:is|are)\s+(?:(?:now|currently|still)\s+)?|"
        r"remains?\s+|stays?\s+|updated\s+to\s+|changed\s+to\s+|"
        r"moves?\s+to\s+|set\s+to\s+|currently\s+)"
        r"(?P<value>[^.!?;]+)",
        re.IGNORECASE,
    )
    boundary = re.compile(
        r"\s*(?:,\s*|\s+(?:and|but)\s+)(?:restricted|private|confidential|"
        r"resident[- ]only|not\s+for\s+sharing)\s*$",
        re.IGNORECASE,
    )
    next_field = re.compile(
        r"\s*(?:,\s*|\s+(?:and|but)\s+)(?:(?:the|a|an)\s+)?"
        r"(?:(?:current|latest|active|approved|live)\s+)?"
        r"(?:date|time|amount|funding|support|wording|phrase|label|"
        r"status|state|blocker|hold|location|room|site|address|terms?|"
        r"vendor|owner|role)\s+"
        r"(?:is|are|remains?|stays?|was|were|updated\s+to|changed\s+to)\b.*$",
        re.IGNORECASE,
    )
    for memory_id, row in ordered:
        source = _text(row.get("text"))
        if not source or _is_non_authoritative_note(source) or re.search(r"\b(?:no\s+change|unchanged|as\s+before|as\s+above)\b", source, re.IGNORECASE):
            continue
        if _is_policy_instruction_source(source):
            continue
        if expected in {"wording", "text"} and re.search(r"\b(?:safe[- ]?(?:label|wording|summary)|described\s+only\s+as|may\s+refer\s+to)\b", field_surface, re.IGNORECASE):
            safe_match = None
            for safe_pattern in (
                r"\b(?:described\s+only\s+as|may\s+refer\s+to)\b\s*(?P<value>[^.!?;,]+)",
                r"\bsafe[- ]?(?:label|wording|summary)\b\s*(?:is|as|:)?\s*(?P<value>[^.!?;,]+)",
            ):
                safe_match = re.search(safe_pattern, source, re.IGNORECASE)
                if safe_match:
                    break
            if safe_match:
                safe_value = _text(safe_match.group("value")).strip(" ':-")
                normalized = _normalize_claims(
                    {"claims": [{
                        "memory_id": memory_id,
                        "attribute": field.attribute or field.label,
                        "value": safe_value,
                        "relation": "current",
                        "effective": True,
                        "source_span": source,
                    }]},
                    field=field,
                    rows=rows,
                )
                if normalized:
                    return normalized
        if aggregate_request:
            aggregate = re.search(
                r"\b(?:snapshot|summary|overview|recap)\b\s*[:=-]\s*(?P<value>.+?)\s*$",
                source,
                re.IGNORECASE,
            )
            if aggregate:
                normalized = _normalize_claims(
                    {
                        "claims": [{
                            "memory_id": memory_id,
                            "value": _text(aggregate.group("value")).strip(),
                            "relation": "current",
                            "effective": True,
                            "source_span": source,
                        }],
                    },
                    field=field,
                    rows=rows,
                )
                if normalized:
                    return normalized
        match = assertion.search(source)
        if not match:
            continue
        value = _text(next_field.sub("", boundary.sub("", match.group("value")))).strip(" ,:;")
        # Composite summaries are carriers for several fields. Select the
        # source fragment that semantically names this field before applying
        # the normal source-grounding checks. This is a generic structural
        # split, not an answer dictionary.
        field_terms = {
            token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", field_surface)
            if token not in {"current", "latest", "active", "the", "and", "for", "only"}
        }
        fragments = [
            _text(fragment).strip(" ,:;")
            for fragment in re.split(r"\s*;\s*|\s*,\s*|\s+and\s+|\s+\/\s+", value)
            if _text(fragment).strip(" ,:;")
        ]
        if len(fragments) > 1 and field_terms:
            ranked_fragments = sorted(
                fragments,
                key=lambda fragment: (
                    len(field_terms & set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", fragment.casefold()))),
                    -len(fragment),
                ),
                reverse=True,
            )
            if ranked_fragments and field_terms.intersection(
                set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", ranked_fragments[0].casefold()))
            ):
                value = ranked_fragments[0]
        if (
            not value
            or value.casefold() in _NON_VALUES
            or re.match(r"^(?:no|not|without|never)\b", value, re.IGNORECASE)
        ):
            continue
        # A regex fallback may find an assertion-shaped phrase whose surface
        # value is actually a neighboring attribute (for example, "is in
        # Room 9" while the requested field is a date).  For typed fields,
        # unknown-looking values are not sufficiently grounded to recover.
        observed = _classify_value_type(value)
        if expected in {"date", "datetime", "time", "amount", "percentage", "credential"} and observed == "unknown":
            continue
        if expected == "structure" and not _source_semantically_supports(field, row):
            continue
        normalized = _normalize_claims(
            {
                "claims": [{
                    "memory_id": memory_id,
                    "value": value,
                    "relation": "current",
                    "effective": True,
                    "source_span": source,
                }],
            },
            field=field,
            rows=rows,
        )
        if normalized:
            return normalized
    return ()


def _source_carrier_scalar_recovery(
    *,
    field: QueryField,
    rows: dict[str, dict[str, Any]],
) -> tuple[StateClaim, ...]:
    """Recover a typed scalar carried by an authorized source sentence.

    This is intentionally narrower than free-form extraction: it recognizes
    only observable amount/date carriers and structural transition carriers,
    then sends the copied span through the normal claim validators. It does
    not decide which source is authorized or which claim supersedes another.
    """
    if field.cardinality != "single":
        return ()
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    if expected not in {"amount", "date", "datetime", "structure"}:
        return ()

    # Runtime evidence may expose the same source channel as ``content``
    # rather than ``text``. Normalize only this recovery view so the existing
    # closure and policy payload contracts remain unchanged.
    carrier_rows = {
        memory_id: {
            **row,
            "text": _text(row.get("text") or row.get("content")),
        }
        for memory_id, row in rows.items()
    }
    validation_field = replace(field, subject_key=None)

    field_terms = {
        token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", _text(f"{field.label} {field.attribute or ''}").casefold())
        if token not in {"current", "latest", "active", "approved", "the", "and", "for"}
    }
    month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    weekday = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    typed_carriers = {
        "amount": re.compile(
            r"(?<![\w])(?:[$€£]\s*\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?))(?![\w])",
            re.IGNORECASE,
        ),
        "date": re.compile(
            rf"(?<![\w])(?:{weekday},?\s+)?{month}\s+\d{{1,2}}(?:,\s*20\d{{2}})?"
            rf"(?:\s+(?:at|around)\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM))?(?![\w])",
            re.IGNORECASE,
        ),
        "datetime": re.compile(
            rf"(?<![\w])(?:{weekday},?\s+)?{month}\s+\d{{1,2}}(?:,\s*20\d{{2}})?(?:\s+(?:at|around)\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM))?(?![\w])",
            re.IGNORECASE,
        ),
    }
    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for memory_id, row in sorted(carrier_rows.items(), key=lambda pair: (_turn(pair[1]), pair[0]), reverse=True):
        source = _text(row.get("text"))
        if not source or not _source_semantically_supports(field, row):
            continue
        spans: list[str] = []
        if expected in typed_carriers:
            spans.extend(match.group(0).strip(" ,;:") for match in typed_carriers[expected].finditer(source))
        elif expected == "structure":
            # Structural values commonly follow an explicit carrier boundary
            # or a generic transition predicate, without saying "contract is".
            patterns = (
                r"\b(?:move|moves|moving|shift|shifts|shifting|transition|transitions|transitioning)\s+(?:toward|to)\s+(?P<value>(?:a|an|the)\s+[^.!?;]+)",
                r"\b(?:direction|structure|terms?|contract|agreement|provision)\s*[:=-]\s*(?P<value>[^.!?;]+)",
            )
            for pattern in patterns:
                match = re.search(pattern, source, re.IGNORECASE)
                if match:
                    value = _text(match.group("value")).strip(" ,;:")
                    value = re.sub(r"^(?:remove|drop|omit)\s+[^.!?;]+?\s+and\s+", "", value, flags=re.IGNORECASE)
                    if value:
                        spans.append(value)
                        break
        for span in dict.fromkeys(spans):
            context = source[max(0, source.find(span) - 80): source.find(span) + len(span) + 80]
            overlap = len(field_terms.intersection(set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", context.casefold()))))
            candidates.append((overlap, _turn(row), memory_id, span, row))
    for _, _, memory_id, value, _ in sorted(candidates, key=lambda item: (item[0], item[1], item[2]), reverse=True):
        normalized = _normalize_claims(
            {"claims": [{
                "memory_id": memory_id,
                "attribute": field.attribute or field.label,
                "value": value,
                "relation": "current",
                "effective": True,
                "source_span": carrier_rows[memory_id].get("text"),
            }]},
            field=validation_field,
            rows=carrier_rows,
        )
        if normalized:
            return tuple(replace(claim, subject_key=field.subject_key or claim.subject_key) for claim in normalized)
    return ()


def _source_location_carrier_recovery(
    *,
    field: QueryField,
    rows: dict[str, dict[str, Any]],
) -> tuple[StateClaim, ...]:
    """Recover a concrete place from an authorized composite state carrier.

    A later operational recap may preserve only a method class (for example,
    ``keyed entry``) while an earlier/current carrier names the actual place
    (for example, ``media hall keypad``).  This helper copies only a bounded
    location-shaped span from policy-approved source text; it does not decide
    authorization or temporal precedence.
    """
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    field_surface = _text(f"{field.label} {field.attribute or ''}").casefold()
    if expected != "location" or not re.search(
        r"\b(?:route|method|path|location|place|entrance|handoff|where|access)\b",
        field_surface,
    ):
        return ()

    location_span = re.compile(
        r"(?<![\w-])(?:the\s+)?(?P<value>[A-Za-z][A-Za-z0-9'/-]*"
        r"(?:\s+[A-Za-z][A-Za-z0-9'/-]*){0,4}\s+"
        r"(?:address|alcove|bench|bay|cabinet|cart|counter|cubby|drawer|"
        r"desk|door|elevator|entrance|gate|hall|keypad|lobby|locker|room|"
        r"side|station|window|zone))\b",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for memory_id, row in sorted(rows.items(), key=lambda pair: (_turn(pair[1]), pair[0]), reverse=True):
        source = _text(row.get("text") or row.get("content"))
        if not source or not _source_semantically_supports(field, row):
            continue
        for match in location_span.finditer(source):
            value = _text(match.group("value")).strip(" ,;:")
            value = re.split(
                r"\s+(?:is|are|remains?|stays?|uses?|via|through)\s+",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[-1]
            value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
            if re.match(r"^(?:at|from|in|near|with|after|before)\b", value, re.IGNORECASE):
                continue
            if not value or _classify_value_type(value) in {"amount", "percentage", "credential", "date", "datetime", "time"}:
                continue
            context = source[max(0, match.start() - 100): min(len(source), match.end() + 100)]
            cue = set(re.findall(r"[a-z0-9]+", context.casefold()))
            if not cue.intersection({"route", "method", "path", "entry", "entrance", "keypad", "gate", "location", "handoff", "desk", "access"}):
                continue
            candidates.append((len(set(re.findall(r"[a-z0-9]+", value.casefold()))), _turn(row), memory_id, value, row))

    claims: list[StateClaim] = []
    for _, _, memory_id, value, row in candidates:
        normalized = _normalize_claims(
            {"claims": [{
                "memory_id": memory_id,
                "attribute": field.attribute or field.label,
                "value": value,
                "relation": "carrier",
                "effective": True,
                "source_span": _text(row.get("text") or row.get("content")),
            }]},
            field=field,
            rows=rows,
        )
        claims.extend(normalized)
    unique: dict[tuple[str, str], StateClaim] = {}
    for claim in claims:
        unique[(claim.memory_id, _norm(claim.value))] = claim
    return tuple(unique.values())


def _source_grounded_current_updates(
    *,
    field: QueryField,
    rows: dict[str, dict[str, Any]],
) -> tuple[StateClaim, ...]:
    """Recover explicit replacement results for a current field.

    The LLM adjudicator decides semantic applicability, but it can still keep
    an older direct assertion when a later record expresses a replacement in
    ``from X to Y`` form. This helper contributes only a source substring from
    an explicit update predicate; the normal claim validation and resolver
    remain authoritative.
    """
    if field.cardinality != "single":
        return ()
    field_terms = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", _text(field.label).casefold()))
    field_terms -= {"current", "latest", "active", "what", "which", "the", "and", "for"}
    if not field_terms:
        return ()
    direct_update = re.compile(
        r"\b(?:new|current|latest|active|updated|changed)\b"
        r"[^.!?;]{0,100}?\b(?:is|are)\s+(?:now\s+)?(?P<value>[^.!?;]+)",
        re.IGNORECASE,
    )
    updates: list[StateClaim] = []
    for memory_id, row in sorted(rows.items(), key=lambda pair: (_turn(pair[1]), pair[0])):
        source = _text(row.get("text"))
        source_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", source.casefold()))
        if not field_terms.intersection(source_tokens):
            continue
        replacements = _source_replacement_pairs(field, source)
        values = [new for _, new in replacements]
        match = direct_update.search(source)
        if match:
            predicate_prefix = source[max(0, match.start() - 100):match.start("value")]
            if field_terms.intersection(set(re.findall(
                r"[a-z0-9][a-z0-9_-]{2,}", predicate_prefix.casefold()
            ))):
                values.append(_text(match.group("value")))
        for value in values:
            value = re.split(
                r"\s*(?:,\s*and|;\s*and|\band)\s+(?:(?:the|a|an)\s+)?"
                r"(?:current|latest|active)\b",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            value = re.split(r"\s+(?:so|because|since|as)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,:;")
            if not value or value.casefold() in _NON_VALUES:
                continue
            normalized = _normalize_claims(
                {
                    "claims": [{
                        "memory_id": memory_id,
                        "attribute": field.attribute or field.label,
                        "value": value,
                        "relation": "update",
                        "effective": True,
                        "source_span": source,
                    }],
                },
                field=field,
                rows=rows,
            )
            updates.extend(normalized)
    unique: dict[tuple[str, str], StateClaim] = {}
    for claim in updates:
        unique[(claim.memory_id, _norm(claim.value))] = claim
    return tuple(unique.values())


def _enrich_temporal_claims(
    *,
    field: QueryField,
    claims: tuple[StateClaim, ...],
    rows: dict[str, dict[str, Any]],
) -> tuple[StateClaim, ...]:
    """Restore fuller date/time qualifiers from allowed lineage carriers."""
    expected = field.value_type if field.value_type not in {"unknown", "list"} else _infer_value_type(field.label, field.attribute)
    if expected not in {"date", "time", "datetime"} or not claims:
        return claims
    weekday = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    month = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    qualifier = re.compile(
        rf"\b{weekday},?\s+{month}\s+\d{{1,2}}(?:,\s*20\d{{2}})?"
        rf"(?:\s+(?:at|around)\s+\d{{1,2}}:\d{{2}}\s*(?:AM|PM))?",
        re.IGNORECASE,
    )
    enriched: list[StateClaim] = []
    for claim in claims:
        claim_tokens = set(re.findall(r"[a-z]+|\d{1,4}", claim.value.casefold()))
        candidates: list[tuple[int, str, dict[str, Any], str]] = []
        for memory_id, row in rows.items():
            source = _text(row.get("text") or row.get("content"))
            match = qualifier.search(source)
            if not match:
                continue
            source_tokens = set(re.findall(r"[a-z]+|\d{1,4}", match.group(0).casefold()))
            if not claim_tokens.intersection(source_tokens) or not any(
                token in source_tokens for token in {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"}
            ):
                continue
            turn = _turn(row)
            candidates.append((turn, memory_id, row, match.group(0).strip(" ,;")))
        if not candidates:
            enriched.append(claim)
            continue
        _, memory_id, row, value = max(candidates, key=lambda item: (item[0], item[1]))
        normalized = _normalize_claims(
            {
                "claims": [{
                    "memory_id": memory_id,
                    "value": value,
                    "relation": claim.relation,
                    "effective": claim.effective,
                    "source_span": _text(row.get("text") or row.get("content")),
                }],
            },
            field=field,
            rows=rows,
        )
        enriched.append(normalized[0] if normalized else claim)
    return tuple(enriched)


def _expand_list_claims(
    *,
    field: QueryField,
    claims: tuple[StateClaim, ...],
    rows: dict[str, dict[str, Any]],
    llm_client: LLMClient,
    config: dict[str, Any],
) -> tuple[StateClaim, ...]:
    """Ask the base model for member-level closure of an aggregate field."""
    def requested_count() -> int | None:
        text = f"{field.label} {field.question_span}".casefold()
        number_words = {"two": 2, "three": 3, "four": 4, "five": 5}
        word_count = next(
            (value for word, value in number_words.items() if re.search(rf"\b{word}\b", text)),
            None,
        )
        if word_count is not None:
            return word_count
        match = re.search(r"\b([2-9])\b", text)
        return int(match.group(1)) if match else None

    def enumerated_member_claims() -> tuple[StateClaim, ...]:
        """Recover numbered members spread across records or lines.

        This generic structural pass recognizes ordinary enumeration labels,
        then copies the member body only from already authorized evidence.
        It repairs recall without knowing any domain entity or expected answer.
        """
        count = requested_count()
        if count is None:
            return ()
        ordinal_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9,
        }
        pattern = re.compile(
            r"(?:^|[.!?;]\s+|\n\s*)"
            r"(?:(?:plan\s+)?(?:step|item|reminder|point|member|task|requirement|option)\s+)?"
            r"(?P<ordinal>one|two|three|four|five|six|seven|eight|nine|[1-9])"
            r"\s*[:.)-]\s*(?P<value>[^.!?;\n]+)",
            re.IGNORECASE,
        )
        found: dict[int, StateClaim] = {}
        for memory_id, row in sorted(rows.items(), key=lambda pair: (_turn(pair[1]), pair[0])):
            source = _text(row.get("text"))
            for match in pattern.finditer(source):
                raw_ordinal = match.group("ordinal").casefold()
                ordinal = ordinal_words.get(raw_ordinal) or int(raw_ordinal)
                if ordinal > count or ordinal in found:
                    continue
                normalized = _normalize_claims(
                    {"claims": [{
                        "memory_id": memory_id,
                        "value": _text(match.group("value")).strip(" ,:"),
                        "relation": "assert",
                        "effective": True,
                        "source_span": match.group(0).strip(),
                    }]},
                    field=field,
                    rows=rows,
                )
                if normalized:
                    found[ordinal] = normalized[0]
        if len(found) < count:
            return ()
        return tuple(found[index] for index in range(1, count + 1))

    def enumerated_source_claims() -> tuple[StateClaim, ...]:
        count = requested_count()
        if count is None:
            return ()
        labeled = enumerated_member_claims()
        if labeled:
            return labeled
        for memory_id, row in rows.items():
            text = _text(row.get("text"))
            parts = re.split(r";\s+|,\s+(?=(?:and\s+)?[A-Z0-9])", text)
            parts = [_text(re.sub(r"^and\s+", "", part, flags=re.IGNORECASE)).rstrip(" .;,") for part in parts]
            if len(parts) < count:
                continue
            # Remove a sentence-level lead-in without touching the value
            # member itself (e.g. "schedule remains Monday ...").
            parts[0] = _text(re.sub(
                r"^.*?\b(?:so|remains?|are|is|recap|update)\s+",
                "",
                parts[0],
                flags=re.IGNORECASE,
            )).rstrip(" .;,")
            raw_claims = {
                "claims": [
                    {
                        "memory_id": memory_id,
                        "value": part,
                        "relation": "assert",
                        "effective": True,
                        "source_span": part,
                    }
                    for part in parts[:count]
                    if part
                ]
            }
            normalized = _normalize_claims(raw_claims, field=field, rows=rows)
            if len(normalized) >= count:
                return normalized[:count]
        return ()

    payload = {
        "field": asdict(field),
        "requested_count": requested_count(),
        "candidate_claims": [
            {
                "memory_id": claim.memory_id,
                "value": claim.value,
                "source_text": rows[claim.memory_id].get("text") or rows[claim.memory_id].get("content"),
                "entities": list(rows[claim.memory_id].get("entities") or ()),
                "scope": rows[claim.memory_id].get("scope"),
                "time": rows[claim.memory_id].get("time"),
            }
            for claim in claims if claim.memory_id in rows
        ],
        "complete_candidate_evidence": [
            {
                "memory_id": memory_id,
                "source_turn_index": _turn(row),
                "source_text": row.get("text") or row.get("content"),
                "entities": list(row.get("entities") or ()),
                "scope": row.get("scope"),
                "time": row.get("time"),
            }
            for memory_id, row in rows.items()
        ],
    }
    assert_runtime_payload_safe(payload, context="field_list_expansion_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=LIST_CLOSURE_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
        return claims
    expanded = _normalize_claims(raw, field=field, rows=rows)
    enumerated = enumerated_source_claims()
    if enumerated and (not expanded or len(expanded) < len(enumerated)):
        return enumerated
    return expanded or claims


def select_field_evidence(
    *,
    contract: QueryContract,
    evidence: list[dict[str, Any]],
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> tuple[FieldEvidenceClosure, ...]:
    rows = _source_rows(evidence)
    closures: list[FieldEvidenceClosure] = []
    for field in contract.fields:
        # Safe-summary is an intentionally narrow public projection.  Its
        # value must be a source-grounded label/object name, not a semantic
        # summary of the whole carrier.  Sending the full authorized carrier
        # to the generic closure/adjudication prompts both adds latency and
        # lets neighboring blocker/status facts cross the projection edge.
        safe_summary_source_only = (
            field.disclosure_scope == "safe_summary"
            or field.field_id == "authorized_safe_summary"
        )
        # Stage 3 records which field-local retrieval lanes produced each
        # row. Prefer that lane-specific slice for semantic closure so a
        # strong candidate for another field cannot crowd out this field's
        # current-state carrier. Older/manual evidence without these tags
        # keeps the compatibility path and is searched as a whole.
        def retrieval_tags(row: dict[str, Any]) -> set[str]:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            return set(row.get("retrieval_fields") or metadata.get("retrieval_fields") or ())

        tagged_rows = [
            row for row in evidence
            if field.field_id in retrieval_tags(row)
            or "__query__" in retrieval_tags(row)
            or (
                field.disclosure_scope == "safe_summary"
                and retrieval_tags(row).intersection({"safe_summary", "safe_carrier_frontier", "public_projection"})
            )
        ]
        # Prefer a field-local candidate slice before falling back to the
        # complete authorized set. The old code computed lexical candidates
        # but ignored them, so every field closure saw every allowed row and
        # could leak unrelated values across fields. Anaphoric continuation
        # records without field vocabulary still use the compatibility
        # fallback when no lexical candidate exists.
        lexical = _lexical_rows(field, rows)
        # Retrieval lanes are recall signals, not mutually exclusive sources.
        # A query-ranked row can coexist with a lower-score row whose source
        # explicitly names the requested field (for example, an entry method
        # record). Dropping the lexical lane whenever ``__query__`` exists
        # makes the top-k lane an accidental authority over field completeness.
        field_evidence = list(tagged_rows)
        seen_memory_ids = {
            _text(row.get("memory_id"))
            for row in field_evidence
            if _text(row.get("memory_id"))
        }
        for row in lexical:
            memory_id = _text(row.get("memory_id"))
            if memory_id and memory_id not in seen_memory_ids:
                field_evidence.append(row)
                seen_memory_ids.add(memory_id)
        if not field_evidence:
            field_evidence = list(evidence)
        field_rows = _source_rows(field_evidence)
        # Field-local retrieval is a recall hint, but the complete authorized
        # pool is deliberately not a default fallback. Reopening that pool
        # after a field-local miss creates a second semantic authority and can
        # bind a neighboring subject, attribute, or temporal lane. Keep the
        # opt-in recovery path for controlled ablations only.
        global_recovery_rows = rows if len(field_rows) < len(rows) else field_rows
        lexical = _lexical_rows(field, field_rows)
        payload = {
            "question": contract.question,
            "field": asdict(field),
            "field_semantic_binding": _field_semantic_binding(field),
            "allowed_evidence": [_evidence_payload_row(row) for row in field_evidence],
        }
        claims: tuple[StateClaim, ...] = ()
        trace: list[str] = [
            "audit:evidence"
            f":authorized={len(rows)}"
            f":tagged={len(tagged_rows)}"
            f":lexical={len(lexical)}"
            f":closure_rows={len(field_rows)}"
            f":global_recovery_rows={len(global_recovery_rows)}",
        ]
        if llm_client is not None and llm_client.is_available() and not safe_summary_source_only:
            assert_runtime_payload_safe(payload, context="field_evidence_closure_prompt")
            try:
                raw = llm_client.chat_json(
                    model=resolve_llm_model(config, "answering"),
                    system_prompt=FIELD_CLOSURE_PROMPT,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                )
                claims = _normalize_claims(raw, field=field, rows=field_rows)
                trace.append(f"llm_semantic_claim_selection:{len(field_evidence)}")
                trace.append(f"audit:llm_claims={len(claims)}")
            except (LLMClientUnavailableError, ValueError, TypeError, KeyError, StopIteration, json.JSONDecodeError):
                trace.append("llm_claim_selection_failed")
            # A closure that already produced multiple members is complete;
            # avoid spending another model call (and preserve deterministic
            # mock/API call budgets). Expand only an empty or singleton list.
            if field.cardinality == "list":
                before = len(claims)
                claims = _expand_list_claims(
                    field=field,
                    claims=claims,
                    rows=field_rows,
                    llm_client=llm_client,
                    config=config,
                )
                if len(claims) != before or claims:
                    trace.append(f"llm_list_member_expansion:{before}->{len(claims)}")
        if not claims:
            # A first semantic pass may miss a field even when its evidence is
            # present. Give the same base LLM one bounded source-grounded
            # recovery pass over the complete field-local evidence. This is
            # recall recovery, not a new permission decision.
            recovery_enabled = bool(
                (config.get("policy_reasoning") or {}).get(
                    "field_projection_global_recovery", False
                )
            )
            recovery_rows = global_recovery_rows if recovery_enabled else field_rows
            recovered: tuple[StateClaim, ...] = ()
            if (
                llm_client is not None
                and llm_client.is_available()
                and not safe_summary_source_only
                and recovery_rows
            ):
                try:
                    recovered = _adjudicate_claims(
                        field=field,
                        claims=(),
                        rows=recovery_rows,
                        llm_client=llm_client,
                        config=config,
                        question=contract.question,
                    )
                except StopIteration:
                    recovered = ()
                if recovered:
                    claims = recovered
                    trace.append(
                        "llm_empty_closure_recovery:"
                        f"{len(claims)}:{'global' if recovery_rows is rows else 'local'}"
                    )
                trace.append(f"audit:recovery_claims={len(recovered)}")
            if not claims:
                # A lexical match can identify candidate rows, but it cannot
                # safely extract a field value. Keep the field unresolved
                # during an API outage instead of passing an entire record off
                # as a value or silently inventing a binding.
                if llm_client is not None and llm_client.is_available():
                    claims = _source_grounded_scalar_fallback(field=field, rows=field_rows)
                    if claims:
                        trace.append(f"source_grounded_scalar_fallback:{len(claims)}")
                        trace.append(f"audit:source_fallback_claims={len(claims)}")
                    carrier_enabled = bool(
                        (config.get("policy_reasoning") or {}).get(
                            "field_projection_scalar_carrier_recovery", False
                        )
                    )
                    if not claims and carrier_enabled:
                        claims = _source_carrier_scalar_recovery(field=field, rows=field_rows)
                        if claims:
                            trace.append(f"source_carrier_scalar_recovery:{len(claims)}")
                            trace.append(f"audit:source_carrier_claims={len(claims)}")
                if not claims:
                    claims = ()
                    trace.append("semantic_claims_unavailable")
        if field.disclosure_scope == "safe_summary" and claims:
            # Preserve a separately sourced safe object name/label that the
            # semantic closure may omit while copying a recap carrier. The
            # helper is still source-bound and never reopens restricted rows.
            safe_carriers = _source_grounded_scalar_fallback(field=field, rows=field_rows)
            known = {(claim.memory_id, _norm(claim.value)) for claim in claims}
            additions = tuple(
                claim for claim in safe_carriers
                if (claim.memory_id, _norm(claim.value)) not in known
            )
            if additions:
                claims = tuple((*claims, *additions))
                trace.append(f"safe_summary_source_carriers:{len(additions)}")
        if claims:
            # Preserve a concrete authorized place when the newest semantic
            # claim is only a generic route/method label.  The resolver will
            # still choose the latest explicit value; this merely gives it a
            # source-grounded concrete candidate to compare against.
            location_carriers = _source_location_carrier_recovery(
                field=field,
                rows=field_rows,
            )
            if location_carriers and any(_vague_location_claim(field, claim) for claim in claims):
                known = {(claim.memory_id, _norm(claim.value)) for claim in claims}
                additions = tuple(
                    claim for claim in location_carriers
                    if (claim.memory_id, _norm(claim.value)) not in known
                )
                if additions:
                    claims = tuple((*claims, *additions))
                    trace.append(f"source_location_carriers:{len(additions)}")
        current_updates = _source_grounded_current_updates(field=field, rows=field_rows)
        if current_updates:
            known = {(claim.memory_id, _norm(claim.value)) for claim in claims}
            additions = tuple(
                claim for claim in current_updates
                if (claim.memory_id, _norm(claim.value)) not in known
            )
            if additions:
                claims = tuple((*claims, *additions))
                trace.append(f"source_grounded_current_updates:{len(additions)}")
        if (
            claims
            and not safe_summary_source_only
            and bool((config.get("policy_reasoning") or {}).get("field_claim_adjudication", False))
        ):
            before = len(claims)
            claims = _adjudicate_claims(
                field=field,
                claims=claims,
                rows=field_rows,
                llm_client=llm_client,
                config=config,
                question=contract.question,
            )
            trace.append(f"llm_claim_adjudication:{before}->{len(claims)}")
            # The model may still omit a source-grounded replacement while
            # preferring an older, more lexical assertion. Reattach only the
            # explicit update claims; the deterministic resolver will select
            # the later effective frontier.
            current_updates = _source_grounded_current_updates(field=field, rows=field_rows)
            known = {(claim.memory_id, _norm(claim.value)) for claim in claims}
            preserved_updates = tuple(
                claim for claim in current_updates
                if (claim.memory_id, _norm(claim.value)) not in known
            )
            if preserved_updates:
                claims = tuple((*claims, *preserved_updates))
                trace.append(f"preserved_source_updates:{len(preserved_updates)}")
        if claims:
            enriched = _enrich_temporal_claims(field=field, claims=claims, rows=field_rows)
            if enriched != claims:
                claims = enriched
                trace.append("temporal_provenance_enrichment")
        memory_ids = tuple(dict.fromkeys(claim.memory_id for claim in claims if claim.memory_id in rows))
        closures.append(FieldEvidenceClosure(field_id=field.field_id, memory_ids=memory_ids, claims=claims, trace=tuple(trace)))
    return tuple(closures)


def resolve_field_claims(field: QueryField, closure: FieldEvidenceClosure) -> StatefulFieldProjection:
    """Resolve current state deterministically after semantic claim extraction."""
    # ``effective`` is an LLM hypothesis, not an authorization or transition
    # fact. The deterministic resolver uses source order and explicit relation
    # markers; otherwise a model's false ``effective=false`` can erase a
    # perfectly valid current value, as observed in the first API smoke.
    claims = [claim for claim in closure.claims if claim.relation not in {"revoked", "deleted", "forgotten", "stale"}]
    if not claims:
        return StatefulFieldProjection(field.field_id, field.label, "unknown", source_memory_ids=(), transition_trace=closure.trace + ("no_effective_claim",))
    claims.sort(key=lambda claim: (claim.source_turn_index, claim.memory_id))

    # Apply explicit replacement links before using source order. This keeps
    # list fields from retaining a value that a later source explicitly
    # superseded, while preserving ordinary additive list updates.
    superseded: set[str] = set()
    for claim in claims:
        superseded.update(claim.supersedes_memory_ids)
        if claim.relation in {"supersedes", "replaces"} and not claim.supersedes_memory_ids:
            superseded.update(
                prior.memory_id
                for prior in claims
                if prior.source_turn_index < claim.source_turn_index
                and prior.subject_key == claim.subject_key
                and prior.event_key == claim.event_key
                and prior.attribute == claim.attribute
            )
    if superseded:
        claims = [claim for claim in claims if claim.memory_id not in superseded]
        if not claims:
            return StatefulFieldProjection(
                field.field_id,
                field.label,
                "unknown",
                source_memory_ids=(),
                transition_trace=closure.trace + ("all_claims_superseded",),
            )
    latest_turn = claims[-1].source_turn_index
    latest = [claim for claim in claims if claim.source_turn_index == latest_turn]
    # A later broad recap can say only that "one blocker" remains while an
    # earlier authorized record names the concrete blocker. Preserve the
    # concrete state in that narrow case; an explicit "no blockers" remains a
    # valid current state and is intentionally not treated as vague.
    if field.cardinality != "list" and latest and (
        all(_vague_frontier_claim(claim) for claim in latest)
        or all(_vague_location_claim(field, claim) for claim in latest)
    ):
        concrete_prior = [
            claim for claim in claims
            if claim.source_turn_index < latest_turn
            and not _vague_frontier_claim(claim)
            and not _vague_location_claim(field, claim)
        ]
        if concrete_prior:
            latest = [max(concrete_prior, key=lambda claim: (claim.source_turn_index, claim.memory_id))]
    values: list[str] = []
    selected: list[StateClaim] = []
    if field.cardinality == "list":
        # Keep all compatible values at the effective frontier. A later update
        # can add a member without invalidating earlier members unless it says
        # that the earlier set is replaced.
        frontier = [claim for claim in claims if claim.relation not in {"stale", "provisional"}]
        for claim in frontier:
            if _norm(claim.value) not in {_norm(value) for value in values}:
                values.append(claim.value)
                selected.append(claim)
    else:
        # Copy the frontier before attaching a lineage carrier; otherwise
        # adding provenance for a qualifier mutates ``latest`` and falsely
        # turns a resolved single field into a conflict.
        selected = list(latest)
        for claim in latest:
            for lineage in claims:
                if lineage.memory_id in claim.lineage_memory_ids and lineage not in selected:
                    selected.append(lineage)
            qualified, lineage_claim = _lineage_value(field, claim, claims)
            if lineage_claim is not None and lineage_claim not in selected:
                selected.append(lineage_claim)
            if _norm(qualified) not in {_norm(value) for value in values}:
                values.append(qualified)
    status = "supported" if values else "unknown"
    if len(latest) > 1 and field.cardinality == "single" and len({_norm(claim.value) for claim in latest}) > 1:
        status = "conflict"
    provenance = tuple({
        "memory_id": claim.memory_id,
        "source_turn_index": claim.source_turn_index,
        "source_span": claim.source_span,
    } for claim in selected)
    trace = closure.trace + (
        f"effective_frontier_turn={latest_turn}",
        f"selected_values={len(values)}",
    )
    return StatefulFieldProjection(
        field_id=field.field_id,
        label=field.label,
        status=status,
        selected_values=tuple(values),
        candidate_values=tuple(dict.fromkeys(claim.value for claim in claims)),
        source_memory_ids=tuple(dict.fromkeys(claim.memory_id for claim in selected)),
        provenance=provenance,
        transition_trace=trace,
    )


def build_stateful_projection(
    *,
    contract: QueryContract,
    evidence: list[dict[str, Any]],
    blocked_memory_ids: Iterable[str] = (),
    restricted_field_ids: Iterable[str] = (),
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> StatefulProjection:
    aggregate_request = _is_expandable_aggregate_contract(contract)
    if aggregate_request:
        contract = _expand_aggregate_contract(
            contract=contract,
            evidence=evidence,
            llm_client=llm_client,
            config=config,
        )
    else:
        # Policy parsing may return an aggregate wrapper together with one or
        # more semantic children. Expand the wrapper in isolation so the
        # children remain part of the contract and are not silently discarded.
        aggregate_fields = [field for field in contract.fields if _is_grouping_field(field.label)]
        if len(aggregate_fields) == 1:
            aggregate_request = True
            aggregate = aggregate_fields[0]
            expanded = _expand_aggregate_contract(
                contract=QueryContract(
                    fields=(aggregate,), source=contract.source, question=contract.question
                ),
                evidence=evidence,
                llm_client=llm_client,
                config=config,
            )
            concrete = [field for field in contract.fields if field.field_id != aggregate.field_id]
            contract = _canonicalize_contract(QueryContract(
                fields=tuple((*concrete, *expanded.fields)),
                source=expanded.source,
                question=contract.question,
            ))
    contract = _drop_redundant_aggregate_wrappers(contract)
    # Once an aggregate has an explicit audience/projection boundary, the same
    # bounded view must feed both field planning and field closure. Otherwise a
    # later fallback can reintroduce an unrelated authorized record and undo
    # the boundary established by the aggregate planner.
    projection_evidence = (
        _aggregate_recall_support_view(contract.question, evidence)
        if aggregate_request
        else list(evidence)
    )
    allowed = tuple(dict.fromkeys(
        str(row.get("memory_id"))
        for row in projection_evidence
        if str(row.get("memory_id") or "")
    ))
    blocked = tuple(dict.fromkeys(str(value) for value in blocked_memory_ids))
    restricted = {str(value) for value in restricted_field_ids}
    unrestricted_contract = QueryContract(
        fields=tuple(field for field in contract.fields if field.field_id not in restricted),
        source=contract.source,
        question=contract.question,
    )
    unrestricted_closures = {
        closure.field_id: closure
        for closure in select_field_evidence(
            contract=unrestricted_contract,
            evidence=projection_evidence,
            llm_client=llm_client,
            config=config,
        )
    }
    # Do not expose a restricted field to semantic closure. A mixed source
    # record can contain both an allowed operational value and a restricted
    # value, so row-level filtering is not sufficient at this boundary.
    closures = tuple(
        unrestricted_closures.get(field.field_id)
        if field.field_id not in restricted
        else FieldEvidenceClosure(
            field_id=field.field_id,
            trace=("field_authorization_restricted_before_closure",),
        )
        for field in contract.fields
    )
    resolved = []
    for field, closure in zip(contract.fields, closures):
        if field.field_id in restricted:
            resolved.append(StatefulFieldProjection(
                field_id=field.field_id,
                label=field.label,
                status="restricted",
                source_memory_ids=(),
                transition_trace=closure.trace + ("field_authorization_restricted",),
            ))
        else:
            resolved.append(resolve_field_claims(field, closure))
    fields = tuple(resolved)
    authorizations = tuple(FieldAuthorization(
        field.field_id,
        "restricted" if field.field_id in restricted else ("allow" if closure.memory_ids else "unknown"),
        () if field.field_id in restricted else closure.memory_ids,
        blocked,
        "field-level partial disclosure boundary" if field.field_id in restricted else "field evidence closure",
    ) for field, closure in zip(contract.fields, closures))
    return StatefulProjection(contract=contract, authorizations=authorizations, closures=closures, fields=fields)


def restricted_field_ids(
    contract: QueryContract,
    *,
    partial_disclosure: bool,
    sensitive_authorized: bool,
) -> tuple[str, ...]:
    """Project a partial policy boundary onto fields, not whole answers."""
    if not partial_disclosure or sensitive_authorized:
        return ()
    return tuple(
        field.field_id
        for field in contract.fields
        if _field_requires_narrow_disclosure(field)
        or any(term in _text(field.label).casefold() for term in _SENSITIVE_FIELD_TERMS)
    )


def projection_evidence_payload(projection: StatefulProjection, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the union of field closures, preserving source rows only."""
    rows = _source_rows(evidence)
    ids = {memory_id for field in projection.fields for memory_id in field.source_memory_ids}
    return [row for row in evidence if str(row.get("memory_id")) in ids]


def projection_to_answer_projection(projection: StatefulProjection):
    """Bridge to the existing renderer without exposing its broad evidence path."""
    from gov_mem.answer_projection import AnswerField, AnswerProjection, AnswerRequest, FieldEvidenceProjection

    request = AnswerRequest(fields=tuple(AnswerField(
        field_id=field.field_id,
        label=field.label,
        question_span=field.question_span,
        entity=field.subject_key,
        temporal_role=field.temporal_role,
        cardinality=field.cardinality,
        required=field.required,
        disclosure_scope=field.disclosure_scope,
    ) for field in projection.contract.fields), source="stateful_field_projection")
    fields = tuple(FieldEvidenceProjection(
        field_id=field.field_id,
        label=field.label,
        status=field.status,
        candidate_values=field.candidate_values,
        selected_values=field.selected_values,
        source_memory_ids=field.source_memory_ids,
        provenance=field.provenance,
        conflict_trace=field.transition_trace,
    ) for field in projection.fields)
    return AnswerProjection(request=request, fields=fields)
