"""Compatibility boundary for query semantics.

The paper-facing Gov-Mem-v4 path obtains semantic fields from the
query-conditioned LLM semantic compiler. This module intentionally contains
no natural-language alias, ontology, trigger, or value-extraction tables.
Legacy callers remain import-compatible and fail closed without a structured
contract.
"""

from __future__ import annotations

import re


CURRENT_STATE_SLOT_ALIASES: dict[str, list[str]] = {}
CURRENT_STATE_DOMAIN_ALIASES: dict[str, list[str]] = {}
HOUSEHOLD_SLOT_ALIASES: dict[str, list[str]] = {}
HOUSEHOLD_DELIVERY_SLOT_ALIASES: dict[str, list[str]] = {}
HOUSEHOLD_STATE_TEXT_CUES: list[str] = []
HOUSEHOLD_COMPOSITE_SLOT_GROUPS: dict[str, dict[str, list[str]]] = {}
PUBLIC_EVENT_ALIASES: list[str] = []
SAFE_WORDING_EXPLICIT_ALIASES: list[str] = []
STATE_SLOT_FRAME_PREFIXES: dict[str, str] = {}


def normalize_query_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold())
    return " ".join(normalized.split())


def contains_query_alias(text: str, aliases: list[str]) -> bool:
    normalized = f" {normalize_query_text(text)} "
    return any(alias and f" {normalize_query_text(alias)} " in normalized for alias in aliases)


def infer_current_state_domain(question: str) -> str:
    del question
    return "project"


def infer_current_state_slots(question: str) -> list[str]:
    del question
    return []


def infer_household_slots(question: str) -> list[str]:
    del question
    return []


def infer_household_delivery_slots(question: str) -> list[str]:
    del question
    return []


def has_household_state_signal(*, text: str, slots: dict[str, object] | None = None) -> bool:
    del text
    return bool(dict(slots or {}))


def infer_action_families(question: str) -> set[str]:
    del question
    return set()


def requests_derived_presence_inference(question: str) -> bool:
    del question
    return False


def classify_state_slot_families(*, text: str, slots: dict[str, object] | None = None) -> set[str]:
    del text
    return {str(key) for key, value in dict(slots or {}).items() if value}


def infer_state_record_type(*, text: str, slots: dict[str, object] | None = None, frame_type: str | None = None) -> str | None:
    del text
    if frame_type in {"project_state", "research_state", "household_plan"}:
        return frame_type
    explicit = slots.get("record_type") if isinstance(slots, dict) else None
    return str(explicit) if explicit in {"project_state", "research_state", "household_plan"} else None


def infer_prefixed_state_slots(question: str) -> list[str]:
    del question
    return []


def infer_household_composite_required_slots(question: str) -> list[str]:
    del question
    return []


def _has_household_scope_context(normalized_question: str, *, helper_scope_request: bool) -> bool:
    del normalized_question
    return bool(helper_scope_request)


def extract_state_slots(text: str) -> dict[str, str]:
    """Disabled lexical state extraction; v4 consumes grounded semantic atoms."""
    del text
    return {}
