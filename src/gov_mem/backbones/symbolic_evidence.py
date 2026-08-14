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
        "consistency": {**consistency, "violations": violations},
        "ordering_changed": False,
        "filtering_applied": False,
        "new_llm_calls": 0,
    }
    return annotated, trace
