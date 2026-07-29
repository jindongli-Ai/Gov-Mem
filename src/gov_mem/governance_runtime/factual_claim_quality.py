"""Request-aware quality checks for source-grounded factual values.

Exact source grounding proves provenance, but it does not prove that a span is
the value of the requested field.  This module rejects generic instruction,
policy, and scope prose at the typed-claim boundary while preserving ordinary
negative states such as ``not available`` or ``no remaining blocker``.
"""

from __future__ import annotations

import re
from typing import Any


_POLICY_SHAPES = {"policy", "access_policy", "permission", "authorization"}

# These patterns describe speech acts, rather than domain facts.  Keep them
# deliberately narrow: broad negation rules would reject valid current states.
_META_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:should|must|may|cannot|can't|can not|do not|don't|never)\s+"
        r"(?:be\s+)?(?:interpreted|treated|return(?:ed)?|shared|disclosed|included|"
        r"used|considered|counted|reported|copied|exposed|revealed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:does not|doesn't|do not|don't|never)\s+"
        r"(?:change|include|belong|apply|authorize|disclose|share|form\s+part)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:outside|within|under|for)\s+(?:the\s+)?"
        r"(?:scope|policy|context|rule|contract)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:this|that|the)\s+(?:message|sentence|statement|record)\b"
        r"[^.]{0,100}\b(?:policy|scope|rule|instruction|disclosure)\b",
        re.IGNORECASE,
    ),
    # These are update/control predicates. They can be facts when the query
    # explicitly asks for an operation or policy, but they are not the value
    # of an ordinary object/location/amount/wording field.
    re.compile(
        r"\b(?:should|must|may)\s+(?:override|replace|supersede|remain|stay)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bfor\s+[^.;]{0,100}\b(?:summary|summaries|wording|label)\b\s+"
        r"(?:remains?|is|stays?)\b",
        re.IGNORECASE,
    ),
)

_DELETION_VALUE_PATTERN = re.compile(
    r"\b(?:delet(?:e|ed|ing)|remov(?:e|ed|ing)|retir(?:e|ed|ing)|"
    r"supersed(?:e|ed|ing))\b|"
    r"\b(?:old|former|previous|prior)\s+[a-z0-9][^.;,!?]*",
    re.IGNORECASE,
)

_STATUS_VALUE_PATTERN = re.compile(
    r"^(?:still\s+)?(?:open|closed|pending|enough|sufficient|adequate|"
    r"available|unavailable|complete|incomplete|remaining|unchanged)$",
    re.IGNORECASE,
)

_STATUS_ATTRIBUTE_TOKENS = {
    "status", "state", "sufficiency", "adequacy", "availability", "progress",
    "condition", "outcome", "decision", "permission", "authorization", "stable",
}

_NON_ASSERTIVE_SOURCE_PATTERNS = (
    # Requests and clarification questions describe what someone wants to
    # know; they do not establish the requested fact.
    re.compile(
        r"^\s*(?:i(?:'m| am)\s+(?:just\s+)?asking|"
        r"(?:i|we|they|helpers?)\s+(?:only\s+)?need|"
        r"(?:please|can\s+you|could\s+you)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:asking\s+(?:if|whether)|does\s+this\s+mean|"
        r"not\s+(?:whether|confirm|establish|say|show))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:what|which|who|where|when|why|how|is|are|does|do|"
        r"can|could|would)\b[^.!?]*\?\s*$",
        re.IGNORECASE,
    ),
)

_META_PLACEHOLDER_VALUES = (
    re.compile(r"\b(?:stable|ready|sufficient|enough)\s+enough\s+to\s+log\b", re.IGNORECASE),
    re.compile(r"\b(?:current|latest)\s+(?:release|handoff|arrival|entry|backup)\s+(?:method|rule)\b", re.IGNORECASE),
)


def _attribute_requests_status(attribute: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    return bool(tokens & _STATUS_ATTRIBUTE_TOKENS)


def _attribute_requests_deletion_or_history(attribute: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    return bool(tokens & {"deletion", "deleted", "history", "historical", "previous", "prior", "old"})


def factual_value_is_eligible(
    *,
    attribute: str,
    slot_name: str,
    value: str,
    semantic_spec: dict[str, Any] | None = None,
    source_text: str = "",
) -> bool:
    """Return whether a grounded span can represent a requested fact.

    Policy-shaped contracts intentionally bypass this factual projection
    check.  The source, slot, lifecycle, and authorization checks remain the
    responsibility of the caller.
    """
    text = str(value or "").strip()
    if not text:
        return False
    spec = dict(semantic_spec or {})
    request_shape = str(spec.get("request_shape") or "fact").strip().lower()
    if request_shape in _POLICY_SHAPES:
        return True
    # A policy answer can also be represented by a contract with an explicit
    # policy query type.  Do not infer this from arbitrary source wording.
    query_type = str(spec.get("query_type") or "").strip().lower()
    if query_type in _POLICY_SHAPES:
        return True
    source = str(source_text or "").strip()
    if source and any(pattern.search(source) for pattern in _NON_ASSERTIVE_SOURCE_PATTERNS):
        return False
    if (
        any(pattern.search(text) for pattern in _META_PLACEHOLDER_VALUES)
        and not _attribute_requests_status(attribute)
    ):
        return False
    if any(pattern.search(text) for pattern in _META_VALUE_PATTERNS):
        return False
    if (
        _STATUS_VALUE_PATTERN.fullmatch(text)
        and not _attribute_requests_status(attribute)
        and not _attribute_requests_deletion_or_history(attribute)
    ):
        return False
    if (
        _DELETION_VALUE_PATTERN.search(text)
        and not _attribute_requests_deletion_or_history(attribute)
        and any(token in str(attribute or "").lower() for token in ("current", "latest", "remaining", "active"))
    ):
        return False
    return True


def factual_value_rejection_reason(
    *,
    attribute: str,
    slot_name: str,
    value: str,
    semantic_spec: dict[str, Any] | None = None,
    source_text: str = "",
) -> str | None:
    """Provide a stable diagnostic reason for rejected typed candidates."""
    if factual_value_is_eligible(
        attribute=attribute,
        slot_name=slot_name,
        value=value,
        semantic_spec=semantic_spec,
        source_text=source_text,
    ):
        return None
    return "instruction_or_meta_value_for_fact_request"
