"""Helpers for preserving second-resolution source timestamps."""

from __future__ import annotations

import re
from typing import Any


_ISO_TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?P<separator>T|\s)"
    r"(?P<hour>\d{2}):(?P<minute>\d{2})"
    r"(?P<seconds>:\d{2}(?:\.\d+)?)?"
    r"(?P<zone>Z|[+-]\d{2}:?\d{2})?$"
)


def normalize_timestamp(value: Any) -> str | None:
    """Return an ISO-like timestamp with explicit hour, minute, and second.

    GateMem records commonly use minute precision. Appending ``:00`` makes
    that precision explicit without inventing a non-zero second. Existing
    seconds, fractional seconds, timezone suffixes, and missing values are
    preserved.
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _ISO_TIMESTAMP_RE.fullmatch(text)
    if not match or match.group("seconds"):
        return text
    return (
        f"{match.group('date')}{match.group('separator')}"
        f"{match.group('hour')}:{match.group('minute')}:00"
        f"{match.group('zone') or ''}"
    )


def normalize_message_timestamp(message: dict[str, Any]) -> dict[str, Any]:
    """Copy a message and normalize its timestamp and retained source turn."""

    normalized = dict(message)
    timestamp = normalize_timestamp(message.get("timestamp"))
    if timestamp is not None:
        normalized["timestamp"] = timestamp
    source_turn = message.get("source_turn")
    if isinstance(source_turn, dict):
        normalized_source_turn = dict(source_turn)
        source_timestamp = normalize_timestamp(source_turn.get("timestamp"))
        if source_timestamp is not None:
            normalized_source_turn["timestamp"] = source_timestamp
        normalized["source_turn"] = normalized_source_turn
    return normalized

