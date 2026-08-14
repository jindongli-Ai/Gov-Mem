"""Lightweight typed relation graph for Gov-Mem v4.

This module keeps the graph auxiliary. It verifies speaker provenance and
materializes GateMem's episode-local principal/entity relationships as typed
edges. It does not reorder or filter evidence, make access decisions, or add
LLM calls.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from gov_mem.data.schema import MemoryInstance, RetrievedEvidence
from gov_mem.query_semantics import (
    extract_state_slots,
    infer_current_state_slots,
    infer_household_composite_required_slots,
    infer_household_delivery_slots,
    infer_household_slots,
)


def _record(row: RetrievedEvidence) -> dict[str, Any] | None:
    value = (row.metadata or {}).get("structured_record")
    return value if isinstance(value, dict) else None


def _roster(instance: MemoryInstance) -> dict[str, str]:
    episode = dict((instance.metadata.get("raw_sample") or {}).get("episode") or {})
    principals = list((episode.get("entities") or {}).get("principals") or [])
    return {
        str(item.get("principal_id")): str(item.get("role"))
        for item in principals
        if isinstance(item, dict) and item.get("principal_id") and item.get("role")
    }


def _episode_entities(instance: MemoryInstance) -> dict[str, Any]:
    episode = dict((instance.metadata.get("raw_sample") or {}).get("episode") or {})
    entities = episode.get("entities") or {}
    return entities if isinstance(entities, dict) else {}


_STATE_CURRENT_MARKERS = (
    "active", "approved", "as of now", "confirmed", "current", "latest",
    "now", "remains", "settled", "updated", "unchanged",
)
_STATE_HISTORICAL_MARKERS = (
    "archived", "earlier", "former", "historical", "old", "previous",
    "retired", "stale", "superseded", "deleted", "removed", "no longer",
)
_STATUS_RE = re.compile(
    r"\b(?:status|state)\s*(?:is|was|remains|now|:)?\s*"
    r"(?P<value>[^,.;!?]{2,80})",
    re.IGNORECASE,
)
_TREATED_STATUS_RE = re.compile(
    r"\b(?:now\s+)?(?:treated|regarded|considered)\s+as\s+"
    r"(?P<value>closed|approved|active|open|pending|complete|completed|"
    r"cancelled|canceled|in[- ]progress|retired|revoked)\b",
    re.IGNORECASE,
)
_CURRENT_RECORD_STATUS_RE = re.compile(
    r"\bcurrent\s+(?P<value>closed|approved|open|pending|complete|"
    r"completed|cancelled|canceled|in[- ]progress)\b"
    r"[^.;!?]{0,60}\b(?:recap|record|summary|case|placement|project)\b",
    re.IGNORECASE,
)
_CLOSED_RECORD_STATUS_RE = re.compile(
    r"\b(?P<value>closed|approved|active|open|pending)\s*[- ]"
    r"(?:record|case|placement|summary)\b",
    re.IGNORECASE,
)


def _requested_state_slots(question: str) -> list[str]:
    """Infer transferable state fields without using an episode/domain name."""

    slots = [*infer_current_state_slots(question), *infer_household_slots(question)]
    # Household delivery slots are opt-in by query shape. This prevents a
    # generic "current summary" from acquiring an unrelated window field.
    if re.search(
        r"\b(?:window|arrival|delivery|setup|support|entry|entrance|route|path|"
        r"areas?|zones?|spaces?|signoff|overflow|fallback|contingency)\b",
        str(question or ""),
        re.IGNORECASE,
    ):
        slots.extend(infer_household_delivery_slots(question))
        slots.extend(infer_household_composite_required_slots(question))
    if re.search(r"\b(?:current\s+)?(?:status|state)\b", str(question or ""), re.IGNORECASE):
        slots.append("status")
    # A generic current-date query is a date slot even when it does not use a
    # benchmark-specific alias such as "review date".
    if re.search(
        r"\b(?:current|latest|settled|now|summary)\b[^.!?]{0,40}\bdate\b",
        str(question or ""),
        re.IGNORECASE,
    ):
        slots.append("target_date")
    return [slot for slot in dict.fromkeys(slots) if slot != "date"]


def _extract_status(text: str) -> str | None:
    """Extract only explicit status assertions, never a bare status adjective."""

    value = _TREATED_STATUS_RE.search(str(text or ""))
    if value:
        return " ".join(value.group("value").split()).strip(" ,:")
    value = _STATUS_RE.search(str(text or ""))
    if value:
        candidate = " ".join(value.group("value").split()).strip(" ,:")
        # A status sentence may continue with another field. Keep the first
        # clause, which remains source-bound and avoids copying a full recap.
        return candidate
    value = _CURRENT_RECORD_STATUS_RE.search(str(text or ""))
    if value:
        return value.group("value").casefold()
    value = _CLOSED_RECORD_STATUS_RE.search(str(text or ""))
    if value:
        return value.group("value").casefold()
    return None


def _state_values(text: str) -> dict[str, str]:
    values = dict(extract_state_slots(str(text or "")))
    if str(values.get("access_room") or "").casefold() in {
        "exact", "confirmation", "refinement posted",
    }:
        values.pop("access_room", None)
    if "target_date" not in values:
        date_match = re.search(
            r"\b(?:current|latest|settled|final)\s+date\s*"
            r"(?:is|was|remains|:)?\s*(?P<value>"
            r"(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?)",
            str(text or ""),
            re.IGNORECASE,
        )
        if date_match:
            values["target_date"] = date_match.group("value")
    if "target_date" not in values and (
        re.search(r"\b(?:current|latest|settled|final)\s+date\b", str(text or ""), re.IGNORECASE)
        or re.search(r"\b(?:current|approved)\b[^.;!?]{0,80}\bdate\b", str(text or ""), re.IGNORECASE)
    ):
        direct_date = re.search(
            r"\b(?:current|latest|settled|final)\s+"
            r"(?P<value>(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?)\s+date\b",
            str(text or ""),
            re.IGNORECASE,
        )
        if direct_date:
            values["target_date"] = direct_date.group("value")
        dates = re.findall(
            r"\b(?:January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b",
            str(text or ""),
            re.IGNORECASE,
        )
        if dates and "target_date" not in values:
            values["target_date"] = dates[-1]
    if "access_room" not in values:
        site_match = re.search(
            r"\b(?:current|active|private)?\s*(?:site|location)\s*"
            r"(?:is|was|remains|now|:)?\s*(?P<value>"
            r"[A-Za-z0-9][^.;!?]{1,70}?)(?=,\s*(?:active|current|no|and|safe|support|credential|status)|[.;!?]|$)",
            str(text or ""),
            re.IGNORECASE,
        )
        if site_match:
            candidate = site_match.group("value").strip(" ,:")
            if candidate.casefold() not in {"exact", "confirmation", "refinement posted"}:
                values["access_room"] = candidate
    if "access_room" not in values:
        reverse_site_match = re.search(
            r",\s*(?P<value>[A-Za-z0-9][^,.;!?]{1,60})\s+(?:site|location)\b",
            str(text or ""),
            re.IGNORECASE,
        )
        if reverse_site_match:
            values["access_room"] = reverse_site_match.group("value").strip(" ,:")
    if "access_badge" not in values:
        credential_match = re.search(
            r"\b(?:current|active|temporary|practicum)?\s*"
            r"(?:credential|badge|access\s+code)\s*"
            r"(?:is|was|remains|now|issued|posted|:)?[^.;!?]{0,35}?"
            r"(?P<value>[A-Za-z]{2,}[A-Za-z0-9]*[_-][A-Za-z0-9_-]{3,})\b",
            str(text or ""),
            re.IGNORECASE,
        )
        if credential_match:
            values["access_badge"] = credential_match.group("value").strip(" ,:")
    if "access_expiry" not in values:
        expiry_match = re.search(
            r"\b(?:expir(?:es|ing|y|ation)|valid\s+through|active\s+through)\s+"
            r"(?P<value>(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?"
            r"(?:\s+at\s+\d{1,2}:\d{2})?)",
            str(text or ""),
            re.IGNORECASE,
        )
        if expiry_match:
            values["access_expiry"] = expiry_match.group("value")
    status = _extract_status(text)
    if status:
        values["status"] = status
    return {str(key): str(value).strip() for key, value in values.items() if str(value).strip()}


def _state_claim_is_historical(text: str, *, current_query: bool) -> bool:
    if not current_query:
        return False
    lowered = " ".join(str(text or "").casefold().split())
    return any(marker in lowered for marker in _STATE_HISTORICAL_MARKERS)


def _build_state_ledger(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build a source-bound current-state ledger from retrieved turns only."""

    requested = _requested_state_slots(question)
    current_query = bool(re.search(
        r"\b(?:as of now|current|currently|latest|now|settled|active)\b",
        str(question or ""),
        re.IGNORECASE,
    ))
    candidates: dict[str, list[dict[str, Any]]] = {slot: [] for slot in requested}
    claims_by_memory: dict[str, dict[str, Any]] = {}
    for row in evidence:
        record = _record(row) or {}
        text = str(record.get("text") or row.content or "")
        values = _state_values(text)
        turn_index = record.get("turn_index")
        if not isinstance(turn_index, int):
            turn_index = -1
        lifecycle = dict((row.metadata or {}).get("symbolic_lifecycle_claim") or {})
        current_signal = any(marker in text.casefold() for marker in _STATE_CURRENT_MARKERS)
        if current_query and lifecycle.get("status") in {"deleted", "revoked", "superseded"} and not current_signal:
            continue
        if _state_claim_is_historical(text, current_query=current_query) and not current_signal:
            continue
        row_claims: dict[str, Any] = {}
        for slot in requested:
            if slot == "access_room" and re.search(
                r"\b(?:not tied|not a proxy|does not (?:alter|update)|should not)\b"
                r"|\bexact\s+(?:site|location)\b[^.;!?]{0,80}\b(?:restricted|private)\b",
                text,
                re.IGNORECASE,
            ):
                continue
            if slot == "target_date" and re.search(
                r"\bpublic\b[^.;!?]{0,80}\bdate\b",
                text,
                re.IGNORECASE,
            ) and not re.search(
                r"\b(?:current|approved|settled|review|target)\b[^.;!?]{0,80}\bdate\b",
                text,
                re.IGNORECASE,
            ):
                # Keep public schedules in public_event_date rather than
                # allowing them to overwrite the subject's current date.
                continue
            value = values.get(slot)
            if not value:
                continue
            candidate = {
                "slot": slot,
                "value": value,
                "memory_id": row.memory_id,
                "turn_id": str(record.get("turn_id") or record.get("message_id") or ""),
                "turn_index": turn_index,
                "quote": text,
                "current_signal": any(marker in text.casefold() for marker in _STATE_CURRENT_MARKERS),
            }
            candidates[slot].append(candidate)
            row_claims[slot] = {key: candidate[key] for key in ("value", "turn_id", "turn_index")}
        if row_claims:
            claims_by_memory[row.memory_id] = row_claims

    fields: dict[str, Any] = {}
    for slot, slot_candidates in candidates.items():
        if not slot_candidates:
            fields[slot] = {"status": "missing", "candidates": []}
            continue
        ranked = sorted(
            slot_candidates,
            key=lambda item: (int(item["turn_index"]), bool(item["current_signal"]), str(item["memory_id"])),
            reverse=True,
        )
        selected = ranked[0]
        distinct_values = list(dict.fromkeys(str(item["value"]) for item in ranked))
        fields[slot] = {
            "status": "resolved",
            "value": selected["value"],
            "source_memory_id": selected["memory_id"],
            "source_turn_id": selected["turn_id"],
            "source_turn_index": selected["turn_index"],
            "quote": selected["quote"],
            "conflict_count": max(0, len(distinct_values) - 1),
            "candidate_count": len(ranked),
            "candidate_values": distinct_values,
        }
    ledger = {
        "version": "state-ledger-v1",
        "mode": "retrieved_evidence_only",
        "requested_slots": requested,
        "fields": fields,
        "resolved_count": sum(item.get("status") == "resolved" for item in fields.values()),
        "missing_count": sum(item.get("status") == "missing" for item in fields.values()),
        "conflict_count": sum(int(item.get("conflict_count") or 0) for item in fields.values()),
        "enforcement_applied": False,
        "new_llm_calls": 0,
    }
    return ledger, claims_by_memory


def _relation_endpoints(
    relationship: dict[str, Any],
    roster: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Extract only typed *_id endpoints; prose policy fields stay attributes."""
    endpoints: list[tuple[str, str, str]] = []
    for field, value in relationship.items():
        if not field.endswith("_id") or field in {"episode_id", "turn_id", "message_id"}:
            continue
        if not isinstance(value, (str, int)) or not str(value).strip():
            continue
        value_text = str(value)
        node_type = "principal" if value_text in roster or "principal" in field else "entity"
        node_id = f"{node_type}::{value_text}"
        endpoints.append((field, value_text, node_id))
    return endpoints


def _aliases(value: str) -> set[str]:
    tokens = {token.lower() for token in re.findall(r"[a-z0-9]+", value)}
    return {token for token in tokens if len(token) >= 3}


def _lifecycle_claim(text: str) -> dict[str, Any] | None:
    """Recognize only explicit lifecycle language; ordinary freshness is unknown."""
    normalized = " ".join(str(text or "").casefold().split())
    # GateMem query turns often mention a deleted value while asking for it.
    # A question or request is not itself a state transition assertion.
    if "?" in normalized or re.match(
        r"^(?:what|when|where|who|which|was|were|is|are|did|does|can|could|please|tell me)\b",
        normalized,
    ):
        return None

    def is_negated(match: re.Match[str]) -> bool:
        prefix = normalized[max(0, match.start() - 36) : match.start()]
        return bool(re.search(r"\b(?:do not|does not|did not|never|unless|without|not)\b", prefix))

    patterns = (
        (
            "deleted",
            (
                r"\b(?:delete|deleted|remove|removed|purge|purged|forget|forgotten)\b.*\b(?:memory|record|value|entry|note|number|token|mapping|phrase|line|badge|code|scope)\b",
                r"\b(?:should|must)\s+no longer be available\b",
                r"\b(?:deleted|removed)\b.*\bunavailable\b",
            ),
        ),
        (
            "revoked",
            (
                r"\brevok(?:e|ed|ing)\b.*\b(?:access|permission|authorization|scope|credential)\b",
                r"\b(?:retired|deactivated|expired)\b.*\b(?:access|permission|credential|code|badge|token|key|phrase)\b",
            ),
        ),
        (
            "superseded",
            (
                r"\b(?:supersed(?:e|ed|es|ing)|replac(?:e|ed|es|ing))\b\s+(?:the\s+)?(?:earlier|old|previous|stale|draft|value|version|method|rule|code|token)\b",
                r"\b(?:earlier|old|previous|stale|draft|value|version|method|rule|code|token)\b[^.]{0,80}\b(?:is|was)\s+superseded\b",
                r"\bno longer\s+(?:the\s+)?(?:current|latest|active)\b",
            ),
        ),
        (
            "updated",
            (
                r"\b(?:revised|changed|updated)\s+(?:from\b.*\bto\b|to\b)",
                r"\bnow\s+(?:revised|changed|updated)\b",
            ),
        ),
    )
    for status, candidates in patterns:
        for cue in candidates:
            match = re.search(cue, normalized)
            if match and not is_negated(match):
                return {
                    "status": status,
                    "explicit": True,
                    "cue": match.group(0),
                    "inference": "explicit_text_only",
                }
    return None


def _validity_certificate(lifecycle_claim: dict[str, Any] | None) -> dict[str, Any]:
    """Project explicit lifecycle language into an auditable shadow state."""
    if lifecycle_claim is None:
        return {
            "mode": "shadow",
            "state": "unknown",
            "current_answer_eligibility": "unknown",
            "explicit": False,
            "reason": "no_explicit_lifecycle_assertion",
        }
    status = str(lifecycle_claim.get("status") or "unknown")
    if status in {"deleted", "revoked", "superseded"}:
        return {
            "mode": "shadow",
            "state": "explicit_inactive",
            "current_answer_eligibility": "blocked_in_enforced_mode",
            "explicit": True,
            "lifecycle_status": status,
            "reason": "explicit_lifecycle_assertion",
        }
    if status == "updated":
        return {
            "mode": "shadow",
            "state": "explicit_update",
            "current_answer_eligibility": "candidate_in_enforced_mode",
            "explicit": True,
            "lifecycle_status": status,
            "reason": "explicit_lifecycle_assertion",
        }
    return {
        "mode": "shadow",
        "state": "unknown",
        "current_answer_eligibility": "unknown",
        "explicit": bool(lifecycle_claim.get("explicit")),
        "lifecycle_status": status,
        "reason": "unrecognized_lifecycle_status",
    }


_LIFECYCLE_TARGET_NOISE = {
    "a", "an", "and", "available", "be", "before", "by", "current",
    "delete", "deleted", "does", "earlier", "entry", "from", "has",
    "in", "is", "it", "latest", "memory", "must", "no", "not", "now",
    "of", "old", "on", "previous", "record", "removed", "replace",
    "replaced", "should", "superseded", "the", "then", "to", "updated",
    "value", "was", "will", "with",
}


def _target_tokens(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", str(text or ""))
    }
    return {token for token in tokens if token not in _LIFECYCLE_TARGET_NOISE and len(token) >= 2}


def _lifecycle_target_text(text: str, claim: dict[str, Any]) -> str:
    """Keep explicit target wording while removing lifecycle boilerplate."""
    normalized = " ".join(str(text or "").casefold().split())
    status = str(claim.get("status") or "")
    if status == "updated":
        match = re.search(r"\bfrom\s+(.+?)\s+to\s+", normalized)
        if match:
            return match.group(1)
    return normalized


def _bind_lifecycle_target(
    *,
    lifecycle_row: RetrievedEvidence,
    lifecycle_claim: dict[str, Any],
    evidence: list[RetrievedEvidence],
) -> dict[str, Any]:
    """Bind an explicit lifecycle assertion to one prior retrieved fact.

    This is deliberately evidence-local. It does not search hidden episode
    turns and does not infer a target when lexical anchors are absent or tied.
    """
    lifecycle_record = _record(lifecycle_row)
    if lifecycle_record is None:
        return {"status": "unbound", "reason": "missing_lifecycle_record"}
    source_index = lifecycle_record.get("turn_index")
    if not isinstance(source_index, int):
        return {"status": "unbound", "reason": "missing_turn_order"}

    anchors = _target_tokens(_lifecycle_target_text(
        str(lifecycle_record.get("text") or ""), lifecycle_claim
    ))
    if not anchors:
        return {"status": "unbound", "reason": "no_target_anchors"}

    candidates: list[dict[str, Any]] = []
    for row in evidence:
        if row.memory_id == lifecycle_row.memory_id:
            continue
        record = _record(row)
        if record is None or not isinstance(record.get("turn_index"), int):
            continue
        if int(record["turn_index"]) >= source_index:
            continue
        candidate_tokens = _target_tokens(str(record.get("text") or ""))
        overlap = anchors.intersection(candidate_tokens)
        strong_overlap = {
            token for token in overlap
            if any(char.isdigit() for char in token) or "-" in token or "_" in token
        }
        if len(overlap) < 2 and not strong_overlap:
            continue
        score = len(overlap) + (2 * len(strong_overlap))
        candidates.append({
            "memory_id": row.memory_id,
            "turn_id": str(record.get("turn_id") or record.get("message_id") or ""),
            "score": score,
            "overlap": sorted(overlap),
        })

    if not candidates:
        return {
            "status": "unbound",
            "reason": "no_prior_unique_anchor_match",
            "target_anchors": sorted(anchors),
        }
    candidates.sort(key=lambda item: (-int(item["score"]), str(item["memory_id"])))
    best_score = int(candidates[0]["score"])
    best = [item for item in candidates if int(item["score"]) == best_score]
    if len(best) != 1:
        return {
            "status": "ambiguous",
            "reason": "multiple_prior_matches_at_same_score",
            "target_anchors": sorted(anchors),
            "candidates": candidates,
        }

    edge_type = "supersedes" if str(lifecycle_claim.get("status")) == "updated" else "invalidates"
    return {
        "status": "bound",
        "edge_type": edge_type,
        "target_memory_id": best[0]["memory_id"],
        "target_turn_id": best[0]["turn_id"],
        "target_anchors": sorted(anchors),
        "overlap": best[0]["overlap"],
        "match_score": best_score,
        "inference": "explicit_lifecycle_prior_overlap",
    }


def _relation_graph(
    *,
    instance: MemoryInstance,
    roster: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, set[str]]]:
    entities = _episode_entities(instance)
    relationships = [item for item in entities.get("relationships") or [] if isinstance(item, dict)]
    nodes_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    principal_edges: dict[str, list[dict[str, Any]]] = {}
    endpoint_aliases: dict[str, set[str]] = {}

    for principal_id, role in roster.items():
        node_id = f"principal::{principal_id}"
        nodes_by_id[node_id] = {
            "node_id": node_id,
            "node_type": "principal",
            "principal_id": principal_id,
            "roster_role": role,
        }
        endpoint_aliases[node_id] = _aliases(principal_id)

    for relationship in relationships:
        relation_type = str(relationship.get("type") or "unspecified")
        endpoints = _relation_endpoints(relationship, roster)
        for _field, value, node_id in endpoints:
            if node_id not in nodes_by_id:
                nodes_by_id[node_id] = {
                    "node_id": node_id,
                    "node_type": node_id.split("::", 1)[0],
                    "value": value,
                }
                endpoint_aliases[node_id] = _aliases(value)
        principal_endpoints = [item for item in endpoints if item[2].startswith("principal::")]
        if not principal_endpoints:
            continue
        # GateMem relationship objects are directional. Prefer the explicit
        # principal_id source; otherwise use the first principal endpoint and
        # keep every remaining endpoint as a target. Never manufacture reverse
        # edges merely because both endpoints happen to be principals.
        source = next(
            (item for item in principal_endpoints if item[0] == "principal_id"),
            principal_endpoints[0],
        )
        source_field, source_value, source_node_id = source
        for target_field, target_value, target_node_id in endpoints:
            if target_node_id == source_node_id:
                continue
            edge = {
                "edge_type": relation_type,
                "source": source_node_id,
                "target": target_node_id,
                "source_field": source_field,
                "target_field": target_field,
                "attributes": {
                    key: value
                    for key, value in relationship.items()
                    if key not in {"type", source_field, target_field}
                },
            }
            edge_key = (edge["edge_type"], edge["source"], edge["target"], edge["target_field"])
            if any(
                (item["edge_type"], item["source"], item["target"], item["target_field"]) == edge_key
                for item in edges
            ):
                continue
            edges.append(edge)
            principal_edges.setdefault(source_value, []).append(edge)

    return list(nodes_by_id.values()), edges, principal_edges, endpoint_aliases


def build_symbolic_evidence(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    """Annotate role and typed relation consistency without changing ranking."""
    roster = _roster(instance)
    violations: list[dict[str, Any]] = []
    annotated: list[RetrievedEvidence] = []
    graph_nodes, relation_edges, principal_edges, endpoint_aliases = _relation_graph(
        instance=instance,
        roster=roster,
    )
    graph_edges: list[dict[str, Any]] = list(relation_edges)
    principal_evidence_counts: dict[str, int] = {}
    lifecycle_status_counts: dict[str, int] = {}
    validity_state_counts: dict[str, int] = {}
    lifecycle_binding_status_counts: dict[str, int] = {}
    lifecycle_bindings: list[dict[str, Any]] = []

    for row in evidence:
        record = _record(row)
        if not record:
            continue
        principal_id = str((record.get("speaker") or {}).get("principal_id") or "")
        if principal_id:
            principal_evidence_counts[principal_id] = principal_evidence_counts.get(principal_id, 0) + 1
    for row in evidence:
        record = _record(row)
        metadata = dict(row.metadata or {})
        if record is None:
            violations.append({"memory_id": row.memory_id, "kind": "missing_structured_record"})
            metadata["symbolic_provenance"] = {
                "record_complete": False,
                "principal_role_consistent": False,
                "role_check": "missing_record",
            }
        else:
            speaker = dict(record.get("speaker") or {})
            principal_id = str(speaker.get("principal_id") or "")
            role = str(speaker.get("role") or "")
            expected_role = roster.get(principal_id)
            if not principal_id or not role:
                role_check = "missing_speaker_field"
                violations.append({
                    "memory_id": row.memory_id,
                    "kind": "missing_speaker_principal_or_role",
                    "principal_id": principal_id,
                    "role": role,
                })
            elif expected_role is None:
                role_check = "principal_not_in_roster"
            elif expected_role != role:
                role_check = "conflict"
                violations.append({
                    "memory_id": row.memory_id,
                    "kind": "principal_role_conflict",
                    "principal_id": principal_id,
                    "observed_role": role,
                    "roster_role": expected_role,
                })
            else:
                role_check = "consistent"

            evidence_node_id = f"evidence::{row.memory_id}"
            graph_nodes.append({
                "node_id": evidence_node_id,
                "node_type": "evidence",
                "turn_id": str(record.get("turn_id") or record.get("message_id") or ""),
                "timestamp": record.get("timestamp"),
            })
            principal_node_id = f"principal::{principal_id}" if principal_id else None
            if principal_node_id:
                graph_edges.append({
                    "edge_type": "spoken_by",
                    "source": evidence_node_id,
                    "target": principal_node_id,
                    "observed_role": role,
                })

            text_aliases = _aliases(str(record.get("text") or "").lower())
            about_node_ids = sorted(
                node_id
                for node_id, aliases in endpoint_aliases.items()
                if node_id != principal_node_id and aliases and aliases.intersection(text_aliases)
            )
            for node_id in about_node_ids:
                graph_edges.append({
                    "edge_type": "about",
                    "source": evidence_node_id,
                    "target": node_id,
                    "inference": "conservative_token_match",
                })

            lifecycle_claim = _lifecycle_claim(str(record.get("text") or ""))
            validity_certificate = _validity_certificate(lifecycle_claim)
            validity_state = str(validity_certificate["state"])
            validity_state_counts[validity_state] = validity_state_counts.get(validity_state, 0) + 1
            lifecycle_binding = {
                "status": "not_applicable",
                "reason": "no_explicit_lifecycle_assertion",
            }
            if lifecycle_claim:
                lifecycle_status = str(lifecycle_claim["status"])
                lifecycle_status_counts[lifecycle_status] = lifecycle_status_counts.get(lifecycle_status, 0) + 1
                lifecycle_binding = _bind_lifecycle_target(
                    lifecycle_row=row,
                    lifecycle_claim=lifecycle_claim,
                    evidence=evidence,
                )
                binding_status = str(lifecycle_binding.get("status") or "unknown")
                lifecycle_binding_status_counts[binding_status] = (
                    lifecycle_binding_status_counts.get(binding_status, 0) + 1
                )
                lifecycle_node_id = f"lifecycle::{row.memory_id}"
                graph_nodes.append({
                    "node_id": lifecycle_node_id,
                    "node_type": "lifecycle_event",
                    "status": lifecycle_status,
                    "memory_id": row.memory_id,
                })
                graph_edges.append({
                    "edge_type": "asserts_lifecycle",
                    "source": evidence_node_id,
                    "target": lifecycle_node_id,
                    "status": lifecycle_status,
                    "inference": "explicit_text_only",
                })
                if binding_status == "bound":
                    target_memory_id = str(lifecycle_binding["target_memory_id"])
                    graph_edges.append({
                        "edge_type": str(lifecycle_binding["edge_type"]),
                        "source": lifecycle_node_id,
                        "target": f"evidence::{target_memory_id}",
                        "target_memory_id": target_memory_id,
                        "target_turn_id": str(lifecycle_binding["target_turn_id"]),
                        "inference": str(lifecycle_binding["inference"]),
                        "overlap": list(lifecycle_binding["overlap"]),
                    })
                    lifecycle_bindings.append({
                        "source_memory_id": row.memory_id,
                        **lifecycle_binding,
                    })

            metadata["symbolic_provenance"] = {
                "record_complete": all(
                    key in record
                    for key in ("turn_id", "timestamp", "speaker", "turn_kind", "text", "checkpoint")
                ),
                "principal_role_consistent": role_check == "consistent",
                "role_check": role_check,
                "checked_fields": ["speaker.principal_id", "speaker.role"],
            }
            metadata["graph_context"] = {
                "graph_type": "evidence_principal_typed_relation_lifecycle",
                "evidence_node_id": evidence_node_id,
                "principal_node_id": principal_node_id,
                "source_relation": "spoken_by",
                "speaker_principal_id": principal_id,
                "speaker_role": role,
                "roster_role": expected_role,
                "role_consistent": role_check == "consistent",
                "same_speaker_evidence_count": principal_evidence_counts.get(principal_id, 0),
                "relation_count": len(principal_edges.get(principal_id, [])),
                "relations": [
                    {
                        "edge_type": edge["edge_type"],
                        "target": edge["target"],
                        "target_field": edge["target_field"],
                        "attributes": edge["attributes"],
                    }
                    for edge in principal_edges.get(principal_id, [])
                ],
                "about_entity_node_ids": about_node_ids,
                "lifecycle_claim": lifecycle_claim,
                "validity_certificate": validity_certificate,
            }
            metadata["symbolic_lifecycle_claim"] = lifecycle_claim
            metadata["symbolic_validity_certificate"] = validity_certificate
            metadata["symbolic_lifecycle_target_binding"] = lifecycle_binding

        annotated.append(replace(row, metadata=metadata))

    consistency = {
        "passed": not violations,
        "violation_count": len(violations),
        "violation_kinds": sorted({str(item.get("kind") or "") for item in violations}),
    }
    annotated = [
        replace(row, metadata={**dict(row.metadata or {}), "symbolic_consistency": consistency})
        for row in annotated
    ]
    state_ledger, claims_by_memory = _build_state_ledger(
        question=instance.question,
        evidence=annotated,
    )
    ledger_attached = False
    ledger_annotated: list[RetrievedEvidence] = []
    for row in annotated:
        metadata = dict(row.metadata or {})
        if row.memory_id in claims_by_memory:
            metadata["symbolic_state_claims"] = claims_by_memory[row.memory_id]
        if not ledger_attached:
            metadata["symbolic_state_ledger"] = state_ledger
            ledger_attached = True
        ledger_annotated.append(replace(row, metadata=metadata))
    annotated = ledger_annotated
    trace = {
        "version": "Gov-Mem-v4-Symbolic-dev2",
        "symbolic_step": "typed_principal_entity_relation_graph_v1",
        "candidate_count": len(evidence),
        "structured_record_count": sum(_record(row) is not None for row in evidence),
        "graph_type": "evidence_principal_typed_relation_lifecycle",
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "graph_node_count": len(graph_nodes),
        "graph_edge_count": len(graph_edges),
        "lifecycle_status_counts": lifecycle_status_counts,
        "lifecycle_claim_count": sum(lifecycle_status_counts.values()),
        "lifecycle_target_binding": {
            "status_counts": lifecycle_binding_status_counts,
            "bound_count": lifecycle_binding_status_counts.get("bound", 0),
            "ambiguous_count": lifecycle_binding_status_counts.get("ambiguous", 0),
            "unbound_count": lifecycle_binding_status_counts.get("unbound", 0),
            "bindings": lifecycle_bindings,
        },
        "validity_projection": {
            "mode": "shadow",
            "state_counts": validity_state_counts,
            "explicit_inactive_count": validity_state_counts.get("explicit_inactive", 0),
            "candidate_count": len(evidence),
            "enforcement_applied": False,
        },
        "state_ledger": state_ledger,
        "consistency": {**consistency, "violations": violations},
        "ordering_changed": False,
        "filtering_applied": False,
        "new_llm_calls": 0,
    }
    return annotated, trace
