"""Source-local text views for typed evidence validation.

Atomic memories may retain a concise content field while their extractor
stores longer, explicitly source-aligned spans.  These spans are evidence
surfaces, not inferred metadata, and can be used to validate a value that is
absent from the concise projection.
"""

from __future__ import annotations

from typing import Any


def grounded_source_text(*, content: str = "", metadata: dict[str, Any] | None = None) -> str:
    """Return the closed source-text view available for one evidence record."""
    metadata = dict(metadata or {})
    values: list[str] = [str(content or "").strip()]

    surface_spans = metadata.get("surface_spans")
    if isinstance(surface_spans, dict):
        values.extend(
            str(value or "").strip()
            for value in surface_spans.values()
            if str(value or "").strip()
        )

    semantic_tags = metadata.get("semantic_tags")
    if isinstance(semantic_tags, dict):
        evidence_span = str(semantic_tags.get("evidence_span") or "").strip()
        if evidence_span:
            values.append(evidence_span)
        for claim in list(semantic_tags.get("claims") or []):
            if not isinstance(claim, dict):
                continue
            claim_span = str(claim.get("claim_span") or "").strip()
            if claim_span:
                values.append(claim_span)

    for candidate in list(metadata.get("typed_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        candidate_source = str(candidate.get("source_text") or "").strip()
        if candidate_source:
            values.append(candidate_source)

    return "\n".join(dict.fromkeys(value for value in values if value))


def row_grounded_source_text(row: Any) -> str:
    """Build a source view from a RetrievedEvidence-like row."""
    return grounded_source_text(
        content=str(getattr(row, "content", "") or ""),
        metadata=dict(getattr(row, "metadata", {}) or {}),
    )


def record_grounded_source_text(record: dict[str, Any] | None) -> str:
    """Build a source view from a closed-set adjudicator record."""
    record = dict(record or {})
    return grounded_source_text(
        content=str(record.get("source_text") or ""),
        metadata=dict(record.get("source_metadata") or {}),
    )
