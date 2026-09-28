"""Exact excerpts actually shown to Stage 2, without hidden history expansion."""
from __future__ import annotations


def collect_prompt_sources(candidates: list[dict], graph: dict, ingestion: dict | None) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}

    def add(turn_id, text):
        if turn_id and isinstance(text, str) and text:
            items = sources.setdefault(str(turn_id), [])
            if text not in items:
                items.append(text)

    for candidate in candidates:
        turns = candidate.get("source_message_ids") or []
        # A mixed chunk has no character-level turn attribution. Its entire text
        # must not be attached to each source ID.
        if len(turns) == 1:
            add(turns[0], candidate.get("text"))

    def visit(value):
        if isinstance(value, dict):
            add(value.get("turn_id"), value.get("span"))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(graph)
    for row in (ingestion or {}).get("new_turns", []) + (ingestion or {}).get("context_turns", []):
        add(row.get("turn_id"), row.get("text"))
    return sources
