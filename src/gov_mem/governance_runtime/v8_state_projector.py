"""Deterministic, prefix-safe projection of the V8 permission/event graph."""
from __future__ import annotations
from datetime import datetime
from typing import Any
from gov_mem.governance_runtime.v8_event_store import V8EventStore


def _event_turn_ids(row: dict[str, Any]) -> list[str]:
    return [str(s.get("turn_id") or "") for s in row.get("source_spans") or [] if isinstance(s, dict)]


def _date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _activation(event: dict, order: dict[str, int], checkpoint_time: str | None) -> str:
    start = event.get("valid_from_turn_id")
    if start and start not in order:
        return "pending_or_unresolved"
    until = event.get("valid_until")
    now, end = _date(checkpoint_time), _date(until)
    if until:
        if until in order:
            return "expired"
        if now is not None and end is not None and (now.tzinfo is None) == (end.tzinfo is None):
            if now >= end:
                return "expired"
        else:
            return "conditional"
    return "conditional" if event.get("condition") else "active"


def project_v8_state(store: V8EventStore, *, visible_turn_ids: list[str],
                     requester_principal_id: str | None, question: str,
                     checkpoint_time: str | None = None) -> dict[str, Any]:
    order = {turn: i for i, turn in enumerate(visible_turn_ids)}

    def visible(row):
        ids = _event_turn_ids(row)
        return bool(ids) and all(t in order for t in ids)

    def sequence(row):
        return max((order[t] for t in _event_turn_ids(row)), default=-1)

    def rows(collection):
        return sorted((r for r in collection if visible(r)), key=sequence)

    scenes = {}
    for row in rows(store.scenes):
        key = row["scene_id"]
        old = scenes.get(key, {})
        merged = {**old, **row}
        for field in ("participant_principal_ids", "source_spans"):
            merged[field] = list(old.get(field) or [])
            for value in row.get(field) or []:
                if value not in merged[field]:
                    merged[field].append(value)
        scenes[key] = merged
    entities = {}
    for row in rows(store.entities):
        old = entities.get(row["entity_id"], {})
        merged = {**old, **row}
        for field in ("aliases", "source_spans"):
            merged[field] = list(old.get(field) or [])
            for value in row.get(field) or []:
                if value not in merged[field]:
                    merged[field].append(value)
        entities[row["entity_id"]] = merged
    relations = rows(store.relations)
    for relation in relations:
        if relation.get("relation_type") == "participates_in":
            scene = scenes.get(relation.get("target_id"))
            if scene is not None and relation.get("source_id") not in scene["participant_principal_ids"]:
                scene["participant_principal_ids"].append(relation["source_id"])
                scene["source_spans"].extend(s for s in relation["source_spans"] if s not in scene["source_spans"])

    current, history, tombstones = {}, {}, []
    for fact in rows(store.facts):
        key = (str(fact.get("scene_id") or ""), str(fact.get("subject_id") or ""),
               str(fact.get("field_name") or "").casefold())
        history.setdefault("::".join(key), []).append(fact)
        if fact.get("lifecycle") in {"delete", "cancel"}:
            current.pop(key, None)
            tombstones.append(dict(fact))
        else:
            current[key] = fact

    events = rows(store.governance_events)
    states = {}
    for event in events:
        if event.get("event_type") != "permission":
            continue
        until = event.get("valid_until")
        active = _activation(event, order, checkpoint_time)
        if active == "pending_or_unresolved":
            continue  # Retained in event history, not effective current state.
        base = (event.get("issuer_principal_id"), event.get("grantee_principal_id"),
                event.get("resource_id"), event.get("action", "access"), event.get("scene_id"))
        included = tuple(sorted(event.get("included_scopes") or []))
        excluded = tuple(sorted(event.get("excluded_scopes") or []))
        key = (*base, included, excluded, event.get("condition"), until)
        # An unconditional whole-resource change replaces earlier narrower
        # states of that issuer/resource/action. Conditional events remain for
        # the language reasoner; they cannot erase an unconditional state.
        if active == "active" and not included and not excluded:
            states = {k: v for k, v in states.items() if k[:5] != base}
        states[key] = {
            **event, "decision": "deny" if event.get("effect") == "revoke" else event.get("effect", "unknown"),
            "activation": active, "history_count": sum(
                1 for e in events if e.get("resource_id") == event.get("resource_id")
                and e.get("grantee_principal_id") == event.get("grantee_principal_id")
                and sequence(e) <= sequence(event)),
        }
    permissions = list(states.values())
    requester_states = [s for s in permissions if s.get("grantee_principal_id") == requester_principal_id]
    return {
        "schema_version": "govmem-v8-state-projection-2",
        "as_of_turn_id": visible_turn_ids[-1] if visible_turn_ids else None,
        "checkpoint_time": checkpoint_time,
        "visible_turn_count": len(order), "current_scenes": list(scenes.values()),
        "current_entities": list(entities.values()), "relations": relations,
        "duty_edges": [r for r in relations if r.get("relation_type") == "operational_duty"],
        "current_facts": list(current.values()), "fact_history": history,
        "lifecycle_tombstones": tombstones,
        # Lifecycle EVENT records are distinct from FACT tombstones. Preserve
        # their complete visible history for semantic scope/time resolution;
        # never silently drop deletion/update/cancel events without a grantee.
        "lifecycle_events": [e for e in events if e.get("event_type") == "lifecycle"],
        "lifecycle_state": [{**e, "activation": _activation(e, order, checkpoint_time)}
                            for e in events if e.get("event_type") == "lifecycle"],
        "current_permissions": permissions,
        "requester_permissions": requester_states,
        "requester_permission_events": [e for e in events if e.get("grantee_principal_id") == requester_principal_id],
        "query_relevant_permissions": requester_states,
        "visible_event_count": len(events),
        "future_event_count_excluded": len(store.governance_events) - len(events),
    }
