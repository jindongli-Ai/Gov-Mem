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
from gov_mem.backbones.semantic_compiler import _atom, grounded_text_contains, verify_grounded_atom


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


def _normalized_grounding_text(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def _build_semantic_state_ledger(
    *,
    evidence: list[RetrievedEvidence],
    query_analysis: dict[str, Any],
    semantic_atoms: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compile verified open-vocabulary atoms into the symbolic state ledger."""

    specs: list[dict[str, Any]] = []
    for index, item in enumerate(query_analysis.get("fields") or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        specs.append({
            "slot_id": str(item.get("slot_id") or f"qslot_{index}"),
            "name": name,
            "target_entity": item.get("target_entity"),
            "required": bool(item.get("required", True)),
            "temporal_requirement": str(
                item.get("temporal_requirement") or "unknown"
            ).casefold(),
        })

    evidence_by_id = {row.memory_id: row for row in evidence}
    candidates: dict[str, list[dict[str, Any]]] = {
        spec["slot_id"]: [] for spec in specs
    }
    claims_by_memory: dict[str, dict[str, Any]] = {}
    valid_lifecycle = {
        "assert", "update", "supersede", "cancel", "delete", "none", "uncertain",
    }
    valid_temporal = {"current", "historical", "unknown"}

    for atom in semantic_atoms:
        if not isinstance(atom, dict):
            continue
        slot_id = str(atom.get("slot_id") or "")
        if slot_id not in candidates:
            continue
        spec = next(item for item in specs if item["slot_id"] == slot_id)
        expected_target = _normalized_grounding_text(spec.get("target_entity"))
        atom_target = _normalized_grounding_text(atom.get("target_entity"))
        if expected_target and atom_target and not (
            expected_target == atom_target
            or expected_target in atom_target
            or atom_target in expected_target
        ):
            continue
        source = dict(atom.get("source") or {})
        memory_id = str(source.get("chunk_id") or source.get("memory_id") or "")
        row = evidence_by_id.get(memory_id)
        if row is None:
            continue
        record = _record(row) or {}
        text = str(record.get("text") or row.content or "")
        span = str(source.get("span") or "").strip()
        normalized_text = _normalized_grounding_text(text)
        normalized_span = _normalized_grounding_text(span)
        value = str(atom.get("value") or "").strip()
        normalized_value = _normalized_grounding_text(value)
        if not normalized_span or normalized_span not in normalized_text:
            continue
        if normalized_value and normalized_value not in normalized_span:
            continue
        supplied_turn_id = str(source.get("turn_id") or "")
        turn_id = str(record.get("turn_id") or record.get("message_id") or "")
        if supplied_turn_id and supplied_turn_id != turn_id:
            continue
        turn_index = record.get("turn_index")
        if not isinstance(turn_index, int):
            turn_index = -1
        lifecycle_type = str(
            dict(atom.get("lifecycle_semantics") or {}).get("type") or "none"
        ).casefold()
        if lifecycle_type not in valid_lifecycle:
            lifecycle_type = "uncertain"
        temporal_state = str(
            dict(atom.get("temporal") or {}).get("state") or "unknown"
        ).casefold()
        if temporal_state not in valid_temporal:
            temporal_state = "unknown"
        candidate = {
            "atom_id": str(atom.get("atom_id") or ""),
            "slot_id": slot_id,
            "slot": str(atom.get("slot_name") or ""),
            "value": value or None,
            "memory_id": memory_id,
            "turn_id": turn_id,
            "turn_index": turn_index,
            "quote": span,
            "temporal": dict(atom.get("temporal") or {}),
            "temporal_anchor": {
                "span": str(
                    dict(atom.get("temporal") or {}).get("anchor_span") or ""
                ).strip(),
                "source": dict(
                    dict(atom.get("temporal") or {}).get("anchor_source") or {}
                ),
            },
            "temporal_state": temporal_state,
            "lifecycle_type": lifecycle_type,
            "confidence": float(atom.get("confidence") or 0.0),
            "target_entity": atom.get("target_entity"),
            "sensitivity_type": str(
                dict(atom.get("sensitivity_semantics") or {}).get("type") or "unknown"
            ).casefold(),
            "authorization_candidate_type": str(
                dict(atom.get("authorization_semantics") or {}).get("type") or "none"
            ).casefold(),
            "lifecycle_target_atom_id": dict(atom.get("lifecycle_semantics") or {}).get("target_atom_id"),
            "lifecycle_target_value": dict(atom.get("lifecycle_semantics") or {}).get("target_value"),
        }
        candidates[slot_id].append(candidate)
        if value:
            claims_by_memory.setdefault(memory_id, {})[slot_id] = {
                "slot_name": candidate["slot"],
                "value": value,
                "turn_id": turn_id,
                "turn_index": turn_index,
                "atom_id": candidate["atom_id"],
            }

    fields: dict[str, Any] = {}
    for spec in specs:
        slot_id = spec["slot_id"]
        slot_name = spec["name"]
        all_candidates = candidates[slot_id]
        terminal = [
            item for item in all_candidates
            if item["lifecycle_type"] in {"cancel", "delete"}
            and item["turn_index"] >= 0
            and any(
                previous["turn_index"] >= 0
                and previous["turn_index"] < item["turn_index"]
                and previous["target_entity"] == item["target_entity"]
                and (
                    previous["atom_id"] == item["lifecycle_target_atom_id"]
                    or (
                        item["lifecycle_target_value"]
                        and _normalized_grounding_text(previous["value"])
                        == _normalized_grounding_text(item["lifecycle_target_value"])
                        and grounded_text_contains(item["quote"], item["lifecycle_target_value"])
                    )
                )
                for previous in all_candidates if previous["value"]
            )
        ]
        values = [
            item for item in all_candidates
            if item["value"] and item["lifecycle_type"] not in {"cancel", "delete"}
        ]
        requirement = spec["temporal_requirement"]
        if requirement == "current":
            explicit = [item for item in values if item["temporal_state"] == "current"]
            unknown = [item for item in values if item["temporal_state"] == "unknown"]
            eligible = explicit or unknown
        elif requirement == "historical":
            explicit = [item for item in values if item["temporal_state"] == "historical"]
            eligible = explicit or values
        else:
            eligible = values

        latest_terminal = max(
            terminal,
            key=lambda item: (item["turn_index"], item["confidence"], item["atom_id"]),
            default=None,
        )
        latest_value_turn = max(
            (item["turn_index"] for item in eligible), default=-1,
        )
        if latest_terminal and latest_terminal["turn_index"] > latest_value_turn:
            fields[slot_name] = {
                "slot_id": slot_id,
                "status": "unavailable",
                "lifecycle_state": latest_terminal["lifecycle_type"],
                "source_memory_id": latest_terminal["memory_id"],
                "source_turn_id": latest_terminal["turn_id"],
                "source_turn_index": latest_terminal["turn_index"],
                "quote": latest_terminal["quote"],
                "candidate_count": len(all_candidates),
            }
            continue
        if not eligible:
            fields[slot_name] = {
                "slot_id": slot_id,
                "status": "missing",
                "candidate_count": len(all_candidates),
            }
            continue

        ranked = sorted(
            eligible,
            key=lambda item: (
                item["turn_index"],
                item["temporal_state"] == requirement,
                item["confidence"],
                item["memory_id"],
            ),
            reverse=True,
        )
        selected = ranked[0]
        # A later update may change only the value while the calendar anchor
        # appears in an earlier candidate for the same query slot. Preserve
        # that source-grounded anchor for downstream realization instead of
        # deriving a date from a record timestamp.
        selected_anchor = dict(selected.get("temporal_anchor") or {})
        inherited_anchor = False
        if not str(selected_anchor.get("span") or "").strip():
            prior_anchors = [
                item for item in all_candidates
                if item.get("value")
                and int(item.get("turn_index", -1))
                <= int(selected.get("turn_index", -1))
                and str(
                    dict(item.get("temporal_anchor") or {}).get("span") or ""
                ).strip()
            ]
            prior_anchors.sort(
                key=lambda item: (
                    int(item.get("turn_index", -1)),
                    float(item.get("confidence") or 0.0),
                ),
                reverse=True,
            )
            anchor_spans = list(dict.fromkeys(
                str(dict(item.get("temporal_anchor") or {}).get("span") or "").strip()
                for item in prior_anchors
            ))
            if len(anchor_spans) == 1 and prior_anchors:
                selected_anchor = dict(prior_anchors[0].get("temporal_anchor") or {})
                inherited_anchor = True
        top_rank = (
            selected["turn_index"], selected["temporal_state"] == requirement,
        )
        top_values = list(dict.fromkeys(
            str(item["value"])
            for item in ranked
            if (item["turn_index"], item["temporal_state"] == requirement) == top_rank
        ))
        distinct_values = list(dict.fromkeys(str(item["value"]) for item in ranked))
        status = "conflict" if len(top_values) > 1 else "resolved"
        fields[slot_name] = {
            "slot_id": slot_id,
            "status": status,
            "value": selected["value"] if status == "resolved" else None,
            "source_memory_id": selected["memory_id"],
            "source_turn_id": selected["turn_id"],
            "source_turn_index": selected["turn_index"],
            "source_atom_id": selected["atom_id"],
            "quote": selected["quote"],
            "temporal_state": selected["temporal_state"],
            "temporal_requirement": requirement,
            "temporal_raw": str(
                dict(selected.get("temporal") or {}).get("raw") or ""
            ),
            "temporal_anchor": selected_anchor,
            "temporal_anchor_inherited": inherited_anchor,
            "lifecycle_state": selected["lifecycle_type"],
            "sensitivity_type": selected["sensitivity_type"],
            "authorization_candidate_type": selected["authorization_candidate_type"],
            "conflict_count": max(0, len(top_values) - 1),
            "candidate_count": len(ranked),
            "candidate_values": distinct_values,
        }

    ledger = {
        "version": "state-ledger-v2-semantic-atoms",
        "mode": "verified_semantic_atoms_closed_evidence",
        "requested_slots": [spec["name"] for spec in specs],
        "fields": fields,
        "resolved_count": sum(item.get("status") == "resolved" for item in fields.values()),
        "missing_count": sum(item.get("status") == "missing" for item in fields.values()),
        "unavailable_count": sum(item.get("status") == "unavailable" for item in fields.values()),
        "conflict_count": sum(item.get("status") == "conflict" for item in fields.values()),
        "enforcement_applied": False,
        "new_llm_calls": 0,
    }
    return ledger, claims_by_memory


def _build_state_ledger(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    required_slot_plan: dict[str, Any] | None = None,
    query_analysis: dict[str, Any] | None = None,
    semantic_atoms: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build a source-bound current-state ledger from retrieved turns only."""

    semantic_compiler_path = bool(
        (query_analysis or {}).get("semantic_compiler_contract")
    )
    if semantic_compiler_path:
        return _build_semantic_state_ledger(
            evidence=evidence,
            query_analysis=dict(query_analysis or {}),
            semantic_atoms=[item for item in semantic_atoms or [] if isinstance(item, dict)],
        )

    # The released v4 path has no raw-text state extraction fallback.  If a
    # caller omits the compiler contract, retain provenance but expose no
    # state claim rather than reactivating a hand-written parser.
    del question, evidence, required_slot_plan
    return {
        "version": "state-ledger-v2-semantic-atoms",
        "mode": "disabled_without_semantic_contract",
        "requested_slots": [],
        "fields": {},
        "resolved_count": 0,
        "missing_count": 0,
        "unavailable_count": 0,
        "conflict_count": 0,
        "enforcement_applied": False,
        "new_llm_calls": 0,
    }, {}


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


def _semantic_lifecycle_claim(atoms: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Project a verified compiler lifecycle candidate into symbolic enums."""

    status_by_type = {
        "assert": "asserted",
        "update": "updated",
        "supersede": "superseded",
        "cancel": "cancelled",
        "delete": "deleted",
    }
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for atom in atoms:
        lifecycle_type = str(
            dict(atom.get("lifecycle_semantics") or {}).get("type") or "none"
        ).casefold()
        status = status_by_type.get(lifecycle_type)
        if not status:
            continue
        candidates.append((float(atom.get("confidence") or 0.0), status, atom))
    if not candidates:
        return None
    _, status, atom = max(candidates, key=lambda item: (item[0], item[1]))
    source = dict(atom.get("source") or {})
    return {
        "status": status,
        "explicit": True,
        "cue": str(source.get("span") or ""),
        "inference": "verified_semantic_compiler",
        "atom_id": str(atom.get("atom_id") or ""),
        "slot_id": str(atom.get("slot_id") or ""),
        "slot_name": str(atom.get("slot_name") or ""),
        "target_entity": atom.get("target_entity"),
    }


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
    if status in {"cancelled", "deleted", "revoked", "superseded"}:
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


def _requester_role(instance: MemoryInstance) -> str:
    return str((instance.metadata.get("requester") or {}).get("role") or "").strip()


def _policy_target_matches(target_role: str, requester_role: str) -> bool:
    if target_role == "any" or not requester_role:
        return True
    if target_role == requester_role:
        return True
    role_groups = {
        "family_member": {"family_member"},
        "clinical_staff": {"clinician", "nurse", "pharmacist", "lab_tech"},
        "operations_staff": {"scheduler", "reception", "billing"},
    }
    return requester_role in role_groups.get(target_role, set())


def _build_policy_consistency(
    *,
    instance: MemoryInstance,
    evidence: list[RetrievedEvidence],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Compile retrieved permission language into a conservative certificate."""

    # Raw-text scope classification was removed from the paper-facing code.
    # Authorization policy candidates must arrive as grounded semantic atoms;
    # this compatibility function therefore returns an unknown certificate.
    return {
        "version": "policy-consistency-v1",
        "mode": "disabled_without_semantic_atoms",
        "decision": "unknown",
        "requester": {
            "principal_id": instance.asking_user_id,
            "role": _requester_role(instance),
        },
        "requested_scopes": [],
        "scope_decisions": {},
        "fact_count": 0,
        "supporting_evidence_ids": [],
        "conflict_scopes": [],
        "enforcement_applied": False,
        "new_llm_calls": 0,
    }, {}


def extract_authorization_assertions(text: str) -> list[dict[str, Any]]:
    """Disabled compatibility hook for the removed raw-text policy parser."""
    del text
    return []


def _clean_auth_phrase(value: Any) -> str:
    return " ".join(str(value or "").split()).strip(" ,:()-")


def _auth_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", str(value or ""))
        if len(token) > 1
    }


def _resolve_roster_principal(value: str, roster: list[dict[str, Any]]) -> str | None:
    """Resolve only an explicit roster id/name/role; never invent an actor."""
    candidate = _clean_auth_phrase(value)
    if not candidate:
        return None
    lowered = candidate.casefold()
    for item in roster:
        principal_id = str(item.get("principal_id") or "").strip()
        display_name = str(item.get("display_name") or "").strip()
        role = str(item.get("role") or "").strip()
        aliases = [principal_id, display_name, role]
        if any(alias and lowered == alias.casefold() for alias in aliases):
            return principal_id or None
    candidate_tokens = _auth_tokens(candidate)
    matches: list[str] = []
    for item in roster:
        principal_id = str(item.get("principal_id") or "").strip()
        aliases = [
            str(item.get("display_name") or ""),
            str(item.get("principal_id") or ""),
        ]
        alias_tokens = set().union(*(_auth_tokens(alias) for alias in aliases))
        if candidate_tokens and candidate_tokens.issubset(alias_tokens):
            matches.append(principal_id)
    return matches[0] if len(matches) == 1 else None


def _authorization_roster(instance: MemoryInstance) -> list[dict[str, Any]]:
    episode = dict((instance.metadata.get("raw_sample") or {}).get("episode") or {})
    principals = (episode.get("entities") or {}).get("principals") or []
    return [item for item in principals if isinstance(item, dict) and item.get("principal_id")]


def _grounded_policy_reference(
    reference: Any,
    *,
    evidence_by_id: dict[str, RetrievedEvidence],
) -> tuple[RetrievedEvidence, dict[str, Any], str] | None:
    """Validate an optional cross-span policy reference.

    Policy relations are often expressed across turns (for example, a prior
    turn names the subject and a later turn names the resource). The semantic
    compiler may provide an explicit reference for either side, but symbolic
    reasoning accepts it only when the chunk, turn/message IDs, and verbatim
    span all belong to the closed retrieved evidence set.
    """
    if not isinstance(reference, dict):
        return None
    memory_id = str(reference.get("chunk_id") or reference.get("memory_id") or "")
    row = evidence_by_id.get(memory_id)
    if row is None:
        return None
    record = _record(row) or {}
    span = str(reference.get("span") or "").strip()
    text = str(record.get("text") or row.content or "")
    if not span or not grounded_text_contains(text, span):
        return None
    expected_turn = str(record.get("turn_id") or record.get("message_id") or "")
    expected_message = str(record.get("message_id") or "")
    supplied_turn = str(reference.get("turn_id") or "")
    supplied_message = str(reference.get("message_id") or "")
    if supplied_turn and supplied_turn != expected_turn:
        return None
    if supplied_message and supplied_message != expected_message:
        return None
    return row, record, span


def _authorization_temporal_key(event: dict[str, Any]) -> tuple[str, Any] | None:
    turn_index = event.get("turn_index")
    if isinstance(turn_index, int) and turn_index >= 0:
        return ("turn_index", turn_index)
    timestamp = str(event.get("timestamp") or "").strip()
    if timestamp:
        return ("timestamp", timestamp)
    return None


def _semantic_atom_authorization_events(
    *,
    evidence: list[RetrievedEvidence],
    semantic_atoms: list[dict[str, Any]] | None,
    roster: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize only source-grounded LLM policy candidates into graph events.

    This is intentionally structural: the neural compiler proposes an explicit
    policy relation, while exact source/span checks and roster resolution decide
    whether the proposal is admissible to symbolic authorization reasoning.
    """
    by_id = {row.memory_id: row for row in evidence}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for atom in semantic_atoms or []:
        if not isinstance(atom, dict):
            continue
        semantics = dict(atom.get("authorization_semantics") or {})
        effect = str(semantics.get("type") or semantics.get("effect") or "").casefold()
        if effect not in {"allow", "deny", "revoke"}:
            continue
        source = dict(atom.get("source") or {})
        memory_id = str(source.get("chunk_id") or source.get("memory_id") or "")
        row = by_id.get(memory_id)
        span = str(source.get("span") or "").strip()
        record = _record(row) if row is not None else None
        text = str((record or {}).get("text") or (row.content if row else "") or "")
        normalized_span = " ".join(span.split()).casefold()
        normalized_text = " ".join(text.split()).casefold()
        subject = str(
            semantics.get("subject") or semantics.get("principal")
            or semantics.get("subject_principal_id") or ""
        ).strip()
        principal = _resolve_roster_principal(subject, roster)
        resource = _clean_auth_phrase(
            semantics.get("resource") or semantics.get("target")
        )
        if row is None or not normalized_span or normalized_span not in normalized_text:
            rejected.append({"atom_id": atom.get("atom_id"), "reason": "source_not_grounded"})
            continue
        if not principal:
            rejected.append({"atom_id": atom.get("atom_id"), "reason": "subject_not_in_roster"})
            continue
        if not resource or not _auth_tokens(resource):
            rejected.append({"atom_id": atom.get("atom_id"), "reason": "missing_resource"})
            continue
        principal_record = next(
            (item for item in roster if str(item.get("principal_id") or "") == principal),
            {},
        )
        subject_grounded = any(
            grounded_text_contains(span, alias)
            for alias in (
                subject,
                principal_record.get("principal_id"),
                principal_record.get("display_name"),
            )
            if str(alias or "").strip()
        )
        subject_ref = _grounded_policy_reference(
            semantics.get("subject_source"), evidence_by_id=by_id,
        )
        resource_ref = _grounded_policy_reference(
            semantics.get("resource_source"), evidence_by_id=by_id,
        )
        if not subject_grounded and subject_ref is not None:
            subject_text = subject_ref[2]
            subject_grounded = any(
                grounded_text_contains(subject_text, alias)
                for alias in (
                    subject,
                    principal_record.get("principal_id"),
                    principal_record.get("display_name"),
                )
                if str(alias or "").strip()
            )
        if not subject_grounded:
            rejected.append({"atom_id": atom.get("atom_id"), "reason": "subject_not_in_source_span"})
            continue
        resource_grounded = grounded_text_contains(span, resource)
        if not resource_grounded and resource_ref is not None:
            resource_grounded = grounded_text_contains(resource_ref[2], resource)
        if not resource_grounded:
            rejected.append({"atom_id": atom.get("atom_id"), "reason": "resource_not_in_source_span"})
            continue
        supporting_ids = {row.memory_id}
        if subject_ref is not None:
            supporting_ids.add(subject_ref[0].memory_id)
        if resource_ref is not None:
            supporting_ids.add(resource_ref[0].memory_id)
        accepted.append({
            "effect": effect,
            "principal": principal,
            "role": None,
            "subject": subject,
            "resource": resource,
            "event_kind": "semantic_compiler_policy_candidate",
            "source_memory_id": row.memory_id,
            "source_turn_id": str((record or {}).get("turn_id") or (record or {}).get("message_id") or ""),
            "turn_index": (record or {}).get("turn_index"),
            "timestamp": (record or {}).get("timestamp"),
            "source": "grounded_semantic_compiler",
            "atom_id": str(atom.get("atom_id") or ""),
            "supporting_evidence_ids": sorted(supporting_ids),
        })
    return accepted, rejected


def _entity_list_authorization_events(
    *,
    evidence: list[RetrievedEvidence],
    entity_lists: dict[str, list[dict[str, Any]]] | None,
    roster: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize retrieved entity-list permission history into policy events."""
    by_id = {row.memory_id: row for row in evidence}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for entity_id, records in (entity_lists or {}).items():
        for record in records or []:
            if not isinstance(record, dict):
                continue
            effect = str(record.get("effect") or "").casefold()
            if str(record.get("relation_type") or "") != "permission" or effect not in {"allow", "deny", "revoke"}:
                continue
            memory_id = str(record.get("source_chunk_id") or "")
            row = by_id.get(memory_id)
            source_span = str(record.get("source_span") or "").strip()
            text = str(((_record(row) or {}).get("text") if row else None) or (row.content if row else ""))
            if row is None or not source_span or not grounded_text_contains(text, source_span):
                rejected.append({"entity_id": entity_id, "reason": "source_not_grounded", "memory_id": memory_id})
                continue
            subject = str(record.get("subject") or "").strip()
            principal = _resolve_roster_principal(subject, roster)
            if not principal:
                rejected.append({"entity_id": entity_id, "reason": "subject_not_in_roster", "subject": subject})
                continue
            source_record = _record(row) or {}
            resource = str(record.get("target") or entity_id).strip()
            accepted.append({
                "effect": effect,
                "principal": principal,
                "role": None,
                "subject": subject,
                "resource": resource,
                "event_kind": "governed_slot_graph_entity_list",
                "source_memory_id": memory_id,
                "source_turn_id": str(source_record.get("turn_id") or source_record.get("message_id") or record.get("source_message_id") or ""),
                "turn_index": source_record.get("turn_index"),
                "timestamp": source_record.get("timestamp"),
                "source": "retrieved_entity_relation_list",
                "entity_id": str(entity_id),
                "source_span": source_span,
                "supporting_evidence_ids": [memory_id],
            })
    return accepted, rejected


def _build_temporal_authorization_graph(
    *, instance: MemoryInstance, evidence: list[RetrievedEvidence],
    semantic_atoms: list[dict[str, Any]] | None = None,
    query_analysis: dict[str, Any] | None = None,
    governed_entity_lists: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build a provenance-grounded authorization state graph in shadow mode."""
    roster = _authorization_roster(instance)
    roster_by_id = {str(item["principal_id"]): item for item in roster}
    events: list[dict[str, Any]] = []
    events_by_memory: dict[str, dict[str, Any]] = {}
    unknown_events: list[dict[str, Any]] = []
    semantic_events, rejected_semantic_events = _semantic_atom_authorization_events(
        evidence=evidence,
        semantic_atoms=semantic_atoms,
        roster=roster,
    )
    entity_list_events, rejected_entity_list_events = _entity_list_authorization_events(
        evidence=evidence,
        entity_lists=governed_entity_lists,
        roster=roster,
    )
    seen_event_keys: set[tuple[str, str, str, str]] = set()
    for event in [*semantic_events, *entity_list_events]:
        event_key = (
            str(event.get("effect") or ""),
            str(event.get("principal") or ""),
            " ".join(str(event.get("resource") or "").casefold().split()),
            str(event.get("source_memory_id") or ""),
        )
        if event_key in seen_event_keys:
            continue
        seen_event_keys.add(event_key)
        event["temporal_key"] = _authorization_temporal_key(event)
        if event["temporal_key"] is None:
            unknown_events.append({**event, "reason": "missing_temporal_source"})
            continue
        events.append(event)
        payload = events_by_memory.setdefault(event["source_memory_id"], {"event_count": 0, "events": []})
        payload["events"].append(event)
        payload["event_count"] = len(payload["events"])

    as_of_turn_id = str((instance.metadata.get("observable") or {}).get("as_of_turn_id") or "")
    as_of_index = None
    for message in instance.messages:
        if str(message.get("turn_id") or message.get("message_id") or "") == as_of_turn_id:
            as_of_index = message.get("turn_index")
            if not isinstance(as_of_index, int):
                as_of_index = instance.messages.index(message)
            break
    applied: list[dict[str, Any]] = []
    future_count = 0
    for event in events:
        if as_of_index is not None and event["temporal_key"][0] == "turn_index" and int(event["temporal_key"][1]) > int(as_of_index):
            future_count += 1
            continue
        applied.append(event)

    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    for principal_id, item in roster_by_id.items():
        principal_node = f"principal::{principal_id}"
        role = str(item.get("role") or "").strip()
        graph_nodes.append({"node_id": principal_node, "node_type": "Principal", "principal_id": principal_id})
        if role:
            role_node = f"role::{role}"
            graph_nodes.append({"node_id": role_node, "node_type": "Role", "role": role})
            graph_edges.append({"edge_type": "has_role", "source": principal_node, "target": role_node})

    event_nodes: dict[int, str] = {}
    for event_number, event in enumerate(applied):
        event_node = f"policy_event::{event['source_memory_id']}::{event_number}"
        event_nodes[event_number] = event_node
        principal_node = str(event["principal"])
        if not principal_node.startswith(("principal::", "role::")):
            principal_node = f"principal::{principal_node}"
        resource_key = " ".join(str(event["resource"]).casefold().split())
        resource_node = f"resource::{resource_key}"
        graph_nodes.extend([
            {
                "node_id": event_node,
                "node_type": "PolicyEvent",
                "effect": event["effect"],
                "source": event.get("source"),
                "source_memory_id": event["source_memory_id"],
                "turn_index": event.get("turn_index"),
            },
            {"node_id": principal_node, "node_type": "Principal" if principal_node.startswith("principal::") else "Role", "value": principal_node.split("::", 1)[1]},
            {"node_id": resource_node, "node_type": "Resource", "value": event["resource"]},
        ])
        graph_edges.append({"edge_type": "allows" if event["effect"] == "allow" else "denies" if event["effect"] == "deny" else "revokes", "source": event_node, "target": principal_node})
        graph_edges.append({"edge_type": "applies_to", "source": event_node, "target": resource_node})
        if str(event.get("event_kind") or "").casefold() in {"supersedes", "replaces", "replacement"}:
            prior_indexes = [
                index
                for index, candidate in enumerate(applied[:event_number])
                if str(candidate.get("principal")) == str(event.get("principal"))
                and " ".join(str(candidate.get("resource")).casefold().split()) == resource_key
            ]
            if prior_indexes:
                graph_edges.append({
                    "edge_type": "supersedes",
                    "source": event_node,
                    "target": event_nodes[prior_indexes[-1]],
                })

    state_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for event in applied:
        state_groups.setdefault((str(event["principal"]), " ".join(str(event["resource"]).casefold().split())), []).append(event)
    current_states: list[dict[str, Any]] = []
    conflict_count = 0
    for (principal, resource), group in state_groups.items():
        ordered = sorted(group, key=lambda item: (item["temporal_key"][0], str(item["temporal_key"][1]), str(item["source_memory_id"])))
        state = None
        current = None
        for event in ordered:
            if current is not None and event["temporal_key"] == current["temporal_key"] and event["effect"] != current["effect"]:
                state = "unknown"
                conflict_count += 1
                current = {**event, "conflict": True}
                continue
            state = "deny" if event["effect"] in {"deny", "revoke"} else event["effect"]
            current = event
        if current is None:
            continue
        current_states.append({
            "principal": principal,
            "resource": resource,
            "decision": state or "unknown",
            "source_memory_id": current["source_memory_id"],
            "source_turn_id": current["source_turn_id"],
            "source_turn_index": current.get("turn_index"),
            "supporting_evidence_ids": sorted({
                str(source_id)
                for item in group
                for source_id in (
                    list(item.get("supporting_evidence_ids") or [])
                    or [item["source_memory_id"]]
                )
            }),
            "conflict": bool(current.get("conflict") or state == "unknown"),
        })

    query_tokens = _auth_tokens(instance.question)
    for field in list((query_analysis or {}).get("fields") or []):
        if isinstance(field, dict):
            query_tokens.update(_auth_tokens(str(field.get("name") or "")))
    query_states = [
        item for item in current_states
        if (_resolve_roster_principal(instance.question, roster) or str(instance.asking_user_id or "")) == item["principal"]
        and query_tokens.intersection(_auth_tokens(item["resource"]))
    ]
    if not query_states:
        query_states = [item for item in current_states if query_tokens.intersection(_auth_tokens(item["resource"]))]
    decisions = {str(item["decision"]) for item in query_states}
    decision = next(iter(decisions)) if len(decisions) == 1 else "unknown"
    certificate = {
        "version": "temporal-authorization-v1",
        "mode": "retrieved_evidence_only",
        "decision": decision,
        "query": {"principal_id": instance.asking_user_id, "as_of_turn_id": as_of_turn_id},
        "query_resource_tokens": sorted(query_tokens),
        "current_authorization": current_states,
        "event_count": len(applied),
        "unknown_event_count": len(unknown_events),
        "semantic_candidate_event_count": len(semantic_events),
        "entity_list_event_count": len(entity_list_events),
        "rejected_semantic_candidate_events": rejected_semantic_events,
        "rejected_entity_list_events": rejected_entity_list_events,
        "ignored_future_event_count": future_count,
        "conflict_count": conflict_count,
        "supporting_evidence_ids": sorted({str(item["source_memory_id"]) for item in query_states}),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "enforcement_applied": False,
        "new_llm_calls": 0,
    }
    return certificate, events_by_memory


def _authorization_resource_match(*, resource: str, text: str) -> bool:
    """Match a governed resource to visible evidence without broad keywords."""
    resource_tokens = _auth_tokens(resource)
    text_tokens = _auth_tokens(text)
    if not resource_tokens or not text_tokens:
        return False
    overlap = resource_tokens.intersection(text_tokens)
    # Require the complete short resource phrase, or a strong match for a
    # longer phrase. A single shared role/name token must never authorize a
    # field-level evidence release or denial.
    if resource_tokens.issubset(text_tokens):
        return True
    return len(overlap) >= 2 and len(overlap) / len(resource_tokens) >= 0.75


def apply_authorization_evidence_boundary(
    *,
    evidence: list[RetrievedEvidence],
    certificate: dict[str, Any],
    semantic_atoms: list[dict[str, Any]] | None = None,
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    """Apply a narrow, provenance-grounded answer-evidence boundary.

    Absence of an authorization event is not treated as denial. Enforcement is
    limited to an explicit deny/revoke or a conflicting current authorization
    state whose resource is grounded in the retrieved evidence. This keeps the
    first enforcement step conservative and avoids dataset-specific policy
    assumptions.
    """
    certificate_decision = str(certificate.get("decision") or "unknown").casefold()
    if certificate_decision not in {"deny", "revoke", "conflict", "unknown"}:
        return list(evidence), {
            "enabled": True,
            "enforcement_applied": False,
            "decision": certificate_decision,
            "governed_state_count": 0,
            "filtered_memory_ids": [],
            "reasons": [],
        }
    query_tokens = {
        str(value).casefold()
        for value in list(certificate.get("query_resource_tokens") or [])
        if str(value).strip()
    }
    states = [
        item for item in list(certificate.get("current_authorization") or [])
        if isinstance(item, dict)
        and (
            not query_tokens
            or _auth_tokens(str(item.get("resource") or "")).intersection(query_tokens)
        )
    ]
    governed_states = [
        item for item in states
        if str(item.get("decision") or "").casefold() in {"deny", "revoke", "conflict", "unknown"}
        and (
            str(item.get("decision") or "").casefold() in {"deny", "revoke", "conflict"}
            or bool(item.get("conflict"))
        )
    ]
    # A deny applies to a governed claim, not automatically to every other
    # claim that happened to be retrieved from the same turn.  The semantic
    # compiler gives us source-grounded, query-conditioned claim spans.  We
    # can therefore redact only those spans without introducing a field-name
    # lexicon.  If no admissible span exists, retain the previous fail-closed
    # whole-memory exclusion.
    atoms_by_memory: dict[str, list[dict[str, Any]]] = {}
    for atom in semantic_atoms or []:
        if not isinstance(atom, dict):
            continue
        source = dict(atom.get("source") or {})
        memory_id = str(source.get("chunk_id") or source.get("memory_id") or "")
        span = str(source.get("span") or "").strip()
        slot_name = str(atom.get("slot_name") or "").strip()
        if memory_id and span and slot_name:
            atoms_by_memory.setdefault(memory_id, []).append(atom)

    if not governed_states:
        return list(evidence), {
            "enabled": True, "enforcement_applied": False,
            "decision": str(certificate.get("decision") or "unknown"),
            "governed_state_count": 0, "filtered_memory_ids": [], "reasons": [],
        }

    def protected_atoms_for(
        row: RetrievedEvidence, states_for_row: list[dict[str, Any]], text: str,
    ) -> list[dict[str, Any]]:
        normalized_text = " ".join(text.split()).casefold()
        protected: list[dict[str, Any]] = []
        for atom in atoms_by_memory.get(row.memory_id, []):
            source = dict(atom.get("source") or {})
            span = str(source.get("span") or "").strip()
            if not span or " ".join(span.split()).casefold() not in normalized_text:
                continue
            slot_name = str(atom.get("slot_name") or "").strip()
            if any(
                str(state.get("semantic_slot_id") or "") == str(atom.get("slot_id") or "")
                or _authorization_resource_match(
                    resource=str(state.get("resource") or ""), text=slot_name,
                )
                for state in states_for_row
            ):
                protected.append(atom)
        return protected

    def redact_spans(text: str, atoms: list[dict[str, Any]]) -> tuple[str, list[str]]:
        projected = text
        redacted_atom_ids: list[str] = []
        # Longest first avoids leaving part of one grounded span visible when
        # two candidate atoms overlap.
        for atom in sorted(atoms, key=lambda item: len(str(dict(item.get("source") or {}).get("span") or "")), reverse=True):
            span = str(dict(atom.get("source") or {}).get("span") or "").strip()
            if not span:
                continue
            # This is source-span replacement, not semantic text matching.
            pattern = re.escape(span).replace(r"\ ", r"\s+")
            projected, replacements = re.subn(
                pattern, "[REDACTED GOVERNED CLAIM]", projected,
                flags=re.IGNORECASE,
            )
            if replacements:
                redacted_atom_ids.append(str(atom.get("atom_id") or ""))
        return projected, redacted_atom_ids

    released: list[RetrievedEvidence] = []
    filtered_memory_ids: list[str] = []
    projected_memory_ids: list[str] = []
    reasons: list[dict[str, Any]] = []
    for row in evidence:
        record = _record(row) or {}
        visible_text = str(record.get("text") or row.content or "")
        matched_states: list[dict[str, Any]] = []
        for state in governed_states:
            resource = str(state.get("resource") or "").strip()
            supporting_ids = {
                str(value) for value in list(state.get("supporting_evidence_ids") or [])
            }
            if row.memory_id in supporting_ids or _authorization_resource_match(
                resource=resource,
                text=visible_text,
            ):
                matched_states.append(state)
        if not matched_states:
            released.append(row)
            continue
        protected_atoms = protected_atoms_for(row, matched_states, visible_text)
        supporting_ids = {
            str(value)
            for state in matched_states
            for value in list(state.get("supporting_evidence_ids") or [])
        }
        if not protected_atoms and row.memory_id in supporting_ids:
            # Cross-span policy candidates may cite a separate subject or
            # context chunk. That supporting chunk is not itself the denied
            # value carrier, so keep it available for symbolic context.
            released.append(row)
            continue
        if protected_atoms:
            projected_text, redacted_atom_ids = redact_spans(visible_text, protected_atoms)
            # A projection consisting solely of placeholders is not useful
            # evidence. Excluding it prevents the answer model from treating a
            # redaction marker as support for a protected value.
            residual = projected_text.replace("[REDACTED GOVERNED CLAIM]", "").strip()
            if residual:
                metadata = dict(row.metadata or {})
                projected_record = dict(record)
                if projected_record:
                    projected_record["text"] = projected_text
                    if isinstance(projected_record.get("source_turn"), dict):
                        projected_record["source_turn"] = {
                            **projected_record["source_turn"], "text": projected_text,
                        }
                    projected_record["authorization_assertions"] = []
                    metadata["structured_record"] = projected_record
                # These are derived text carriers, not independent sources.
                # Rebuild them from the released text instead of leaking the
                # removed span through a duplicate annotation.
                for key in (
                    "symbolic_state_claims", "symbolic_state_ledger",
                    "symbolic_lifecycle_claim", "symbolic_permission_claim",
                    "graph_context", "symbolic_policy_facts",
                ):
                    metadata.pop(key, None)
                metadata["semantic_compiler_atoms"] = [
                    atom for atom in metadata.get("semantic_compiler_atoms", [])
                    if grounded_text_contains(projected_text, dict(atom.get("source") or {}).get("span"))
                ]
                metadata["authorization_claim_projection"] = {
                    "mode": "source_grounded_span_redaction",
                    "redacted_atom_ids": redacted_atom_ids,
                    "resources": sorted({str(item.get("resource") or "") for item in matched_states}),
                }
                released.append(replace(row, content=projected_text, metadata=metadata))
                projected_memory_ids.append(row.memory_id)
                reasons.append({
                    "memory_id": row.memory_id,
                    "mode": "source_grounded_span_redaction",
                    "redacted_atom_ids": redacted_atom_ids,
                    "resources": sorted({str(item.get("resource") or "") for item in matched_states}),
                })
                continue
        filtered_memory_ids.append(row.memory_id)
        reasons.append({
            "memory_id": row.memory_id,
            "mode": "whole_memory_exclusion_no_safe_projection",
            "decisions": sorted({str(item.get("decision") or "unknown") for item in matched_states}),
            "resources": sorted({str(item.get("resource") or "") for item in matched_states}),
            "supporting_evidence": sorted({
                str(value)
                for item in matched_states
                for value in list(item.get("supporting_evidence_ids") or [])
            }),
        })

    return released, {
        "enabled": True,
        "enforcement_applied": bool(filtered_memory_ids or projected_memory_ids),
        "decision": str(certificate.get("decision") or "unknown"),
        "governed_state_count": len(governed_states),
        "filtered_memory_ids": filtered_memory_ids,
        "projected_memory_ids": projected_memory_ids,
        "reasons": reasons,
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
    required_slot_plan: dict[str, Any] | None = None,
    policy_consistency_enabled: bool = False,
    temporal_authorization_enabled: bool = False,
    temporal_authorization_enforcement: bool = False,
    query_analysis: dict[str, Any] | None = None,
    semantic_atoms: list[dict[str, Any]] | None = None,
    governed_entity_lists: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    """Annotate role and typed relation consistency without changing ranking."""
    semantic_compiler_path = bool(
        (query_analysis or {}).get("semantic_compiler_contract")
    )
    # The paper-facing v4 path obtains policy/lifecycle semantics from
    # source-grounded compiler atoms. Never activate the legacy raw-text
    # policy scope/authorization lexicons alongside that path.
    if semantic_compiler_path:
        policy_consistency_enabled = False
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
    if semantic_compiler_path:
        specs = {
            str(item.get("slot_id") or f"qslot_{index}"): item
            for index, item in enumerate((query_analysis or {}).get("fields") or [])
            if isinstance(item, dict)
        }
        names = {slot_id: str(spec.get("name") or "") for slot_id, spec in specs.items()}
        checked_atoms = []
        for index, item in enumerate(semantic_atoms or []):
            atom = _atom(item, index, set(specs), names)
            if atom is None:
                continue
            target = specs[atom.slot_id].get("target_entity")
            check = verify_grounded_atom(
                atom, evidence, set(specs),
                [{"canonical_reference": target}] if target else [],
            )
            if check.accepted_for_symbolic_reasoning:
                checked_atoms.append(atom.to_dict())
        semantic_atoms = checked_atoms
    atoms_by_memory: dict[str, list[dict[str, Any]]] = {}
    for atom in semantic_atoms or []:
        if not isinstance(atom, dict):
            continue
        source = dict(atom.get("source") or {})
        source_id = str(source.get("chunk_id") or source.get("memory_id") or "")
        if source_id:
            atoms_by_memory.setdefault(source_id, []).append(atom)

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
        if row.memory_id in atoms_by_memory:
            metadata["semantic_compiler_atoms"] = atoms_by_memory[row.memory_id]
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

            lifecycle_claim = _semantic_lifecycle_claim(
                atoms_by_memory.get(row.memory_id, [])
            )
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
                lifecycle_binding = {
                    "status": "candidate_bound",
                    "reason": "grounded_candidate_references_query_slot_not_whole_memory",
                    "atom_id": lifecycle_claim.get("atom_id"),
                    "slot_id": lifecycle_claim.get("slot_id"),
                    "slot_name": lifecycle_claim.get("slot_name"),
                    "target_entity": lifecycle_claim.get("target_entity"),
                }
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
                if binding_status in {"bound", "candidate_bound"}:
                    slot_node_id = f"query_slot::{lifecycle_binding['slot_id']}"
                    if not any(
                        node.get("node_id") == slot_node_id for node in graph_nodes
                    ):
                        graph_nodes.append({
                            "node_id": slot_node_id,
                            "node_type": "query_slot",
                            "slot_id": lifecycle_binding["slot_id"],
                            "slot_name": lifecycle_binding["slot_name"],
                            "target_entity": lifecycle_binding.get("target_entity"),
                        })
                    graph_edges.append({
                        "edge_type": "applies_lifecycle_to_slot",
                        "source": lifecycle_node_id,
                        "target": slot_node_id,
                        "atom_id": lifecycle_binding["atom_id"],
                        "inference": "verified_semantic_compiler",
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
    policy_certificate: dict[str, Any] | None = None
    policy_facts_by_memory: dict[str, list[dict[str, Any]]] = {}
    if policy_consistency_enabled:
        policy_certificate, policy_facts_by_memory = _build_policy_consistency(
            instance=instance,
            evidence=annotated,
        )
        policy_annotated: list[RetrievedEvidence] = []
        certificate_attached = False
        for row in annotated:
            metadata = dict(row.metadata or {})
            row_facts = policy_facts_by_memory.get(row.memory_id, [])
            if row_facts:
                metadata["symbolic_policy_facts"] = row_facts
            if not certificate_attached:
                metadata["symbolic_policy_certificate"] = policy_certificate
                certificate_attached = True
            policy_annotated.append(replace(row, metadata=metadata))
        annotated = policy_annotated
    temporal_authorization_certificate: dict[str, Any] | None = None
    temporal_events_by_memory: dict[str, dict[str, Any]] = {}
    if temporal_authorization_enabled:
        temporal_authorization_certificate, temporal_events_by_memory = _build_temporal_authorization_graph(
            instance=instance,
            evidence=annotated,
            semantic_atoms=semantic_atoms,
            query_analysis=query_analysis,
            governed_entity_lists=governed_entity_lists,
        )
        temporal_annotated: list[RetrievedEvidence] = []
        certificate_attached = False
        for row in annotated:
            metadata = dict(row.metadata or {})
            event_payload = temporal_events_by_memory.get(row.memory_id)
            if event_payload:
                metadata["symbolic_temporal_authorization_events"] = event_payload
            if not certificate_attached:
                metadata["symbolic_temporal_authorization_certificate"] = temporal_authorization_certificate
                certificate_attached = True
            temporal_annotated.append(replace(row, metadata=metadata))
        annotated = temporal_annotated
    authorization_boundary: dict[str, Any] = {
        "enabled": bool(temporal_authorization_enforcement),
        "enforcement_applied": False,
        "decision": str((temporal_authorization_certificate or {}).get("decision") or "unknown"),
        "governed_state_count": 0,
        "filtered_memory_ids": [],
        "reasons": [],
    }
    if temporal_authorization_enforcement and temporal_authorization_certificate:
        annotated, authorization_boundary = apply_authorization_evidence_boundary(
            evidence=annotated,
            certificate=temporal_authorization_certificate,
            semantic_atoms=semantic_atoms,
        )
        temporal_authorization_certificate = {
            **temporal_authorization_certificate,
            "enforcement_applied": bool(authorization_boundary.get("enforcement_applied")),
        }
        if annotated:
            first = annotated[0]
            first_metadata = dict(first.metadata or {})
            first_metadata["symbolic_temporal_authorization_certificate"] = temporal_authorization_certificate
            annotated[0] = replace(first, metadata=first_metadata)
    # Resolve values only after the authorization boundary. A denied atom or
    # redacted claim span must not survive indirectly inside the ledger that is
    # shown to the answer model.
    state_ledger, claims_by_memory = _build_state_ledger(
        question=instance.question,
        evidence=annotated,
        required_slot_plan=required_slot_plan,
        query_analysis=query_analysis,
        semantic_atoms=semantic_atoms,
    )
    ledger_annotated: list[RetrievedEvidence] = []
    for index, row in enumerate(annotated):
        metadata = dict(row.metadata or {})
        metadata.pop("symbolic_state_claims", None)
        metadata.pop("symbolic_state_ledger", None)
        if row.memory_id in claims_by_memory:
            metadata["symbolic_state_claims"] = claims_by_memory[row.memory_id]
        if index == 0:
            metadata["symbolic_state_ledger"] = state_ledger
        ledger_annotated.append(replace(row, metadata=metadata))
    annotated = ledger_annotated
    trace = {
        "version": (
            "Gov-Mem-v4-Symbolic-dev7"
            if temporal_authorization_enforcement
            else "Gov-Mem-v4-Symbolic-dev5"
            if temporal_authorization_enabled
            else "Gov-Mem-v4-Symbolic-dev2"
        ),
        "symbolic_step": "typed_principal_entity_relation_graph_v1",
        "candidate_count": len(evidence),
        "semantic_compiler_atom_count": sum(len(items) for items in atoms_by_memory.values()),
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
        "policy_consistency": policy_certificate or {
            "enabled": False,
            "enforcement_applied": False,
            "new_llm_calls": 0,
        },
        "temporal_authorization": temporal_authorization_certificate or {
            "enabled": False,
            "enforcement_applied": False,
            "new_llm_calls": 0,
        },
        "governed_entity_lists": {
            "entity_count": len(governed_entity_lists or {}),
            "relation_count": sum(len(items) for items in (governed_entity_lists or {}).values()),
            "entities": sorted(str(entity_id) for entity_id in (governed_entity_lists or {})),
        },
        "authorization_evidence_boundary": authorization_boundary,
        "consistency": {**consistency, "violations": violations},
        "ordering_changed": False,
        "filtering_applied": bool(authorization_boundary.get("filtered_memory_ids")),
        "new_llm_calls": 0,
    }
    return annotated, trace
