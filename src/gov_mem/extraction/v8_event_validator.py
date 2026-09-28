"""Source and roster validation for Gov-Mem v8 extraction output."""

from __future__ import annotations

import re
from hashlib import sha1
from typing import Any

from gov_mem.extraction.v8_event_schema import (
    V8Entity,
    V8ExtractionBatch,
    V8Fact,
    V8GovernanceEvent,
    V8Relation,
    V8Scene,
    V8SourceSpan,
)
from gov_mem.extraction.v8_principal_registry import V8PrincipalRegistry


_EFFECTS = {"allow", "deny", "revoke", "require_permission"}
_EVENT_TYPES = {"permission", "lifecycle"}
_LIFECYCLES = {"assert", "update", "supersede", "cancel", "delete", "unknown"}
_TEMPORAL_STATES = {"current", "historical", "unknown"}


def _norm(value: Any) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = sha1("||".join(str(part or "") for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _validated_spans(raw: Any, turn_text: dict[str, str | list[str]]) -> list[V8SourceSpan]:
    output: list[V8SourceSpan] = []
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        turn_id = str(item.get("turn_id") or "").strip()
        span = str(item.get("span") or "").strip()
        source = turn_text.get(turn_id)
        excerpts = [source] if isinstance(source, str) else (source or [])
        if not span or not any(span in excerpt for excerpt in excerpts):
            continue
        key = (turn_id, _norm(span))
        if key not in seen:
            seen.add(key)
            output.append(V8SourceSpan(turn_id=turn_id, span=span))
    return output


def validate_v8_extraction(
    raw: Any,
    *,
    turn_text: dict[str, str | list[str]],
    registry: V8PrincipalRegistry,
    allowed_output_turn_ids: set[str] | None = None,
    existing_entity_ids: set[str] | None = None,
) -> V8ExtractionBatch:
    """Normalize one LLM batch and reject only structurally unsafe records."""

    payload = raw if isinstance(raw, dict) else {}
    batch = V8ExtractionBatch()
    source_repairs = []
    known_ids = set(registry.principals) | set(existing_entity_ids or set())

    for item in payload.get("scenes") or []:
        if not isinstance(item, dict):
            continue
        spans = _validated_spans(item.get("source_spans"), turn_text)
        name = str(item.get("canonical_name") or item.get("surface") or "").strip()
        if not name or not spans or not any(name in span.span for span in spans):
            batch.rejected.append({"kind": "scene", "reason": "missing_grounded_name"})
            continue
        if allowed_output_turn_ids and not any(span.turn_id in allowed_output_turn_ids for span in spans):
            batch.rejected.append({"kind": "scene", "reason": "context_only_scene"})
            continue
        participants = [
            str(value).strip()
            for value in item.get("participant_principal_ids") or []
            if str(value).strip()
        ]
        if any(not registry.contains(value) for value in participants):
            batch.rejected.append({"kind": "scene", "reason": "unknown_scene_participant"})
            continue
        scene_id = str(item.get("scene_id") or "").strip() or _stable_id(
            "scene", item.get("scene_type"), name,
        )
        batch.scenes.append(V8Scene(
            scene_id=scene_id,
            scene_type=str(item.get("scene_type") or "activity").casefold(),
            canonical_name=name,
            participant_principal_ids=list(dict.fromkeys(participants)),
            status=str(item.get("status") or "active").casefold(),
            source_spans=spans,
            confidence=_confidence(item.get("confidence")),
        ))
        known_ids.add(scene_id)

    for item in payload.get("entities") or []:
        if not isinstance(item, dict):
            continue
        spans = _validated_spans(item.get("source_spans"), turn_text)
        name = str(item.get("canonical_name") or item.get("surface") or "").strip()
        if not name or not spans or not any(name in span.span for span in spans):
            batch.rejected.append({"kind": "entity", "reason": "missing_grounded_name"})
            continue
        entity_id = str(item.get("entity_id") or "").strip() or _stable_id(
            "entity", item.get("entity_type"), name, spans[0].turn_id,
        )
        batch.entities.append(V8Entity(
            entity_id=entity_id,
            entity_type=str(item.get("entity_type") or "resource").casefold(),
            canonical_name=name,
            aliases=[str(value).strip() for value in item.get("aliases") or [] if str(value).strip()],
            scene_id=str(item.get("scene_id") or "").strip() or None,
            source_spans=spans,
            confidence=_confidence(item.get("confidence")),
        ))
        known_ids.add(entity_id)

    for item in payload.get("facts") or []:
        if not isinstance(item, dict):
            continue
        spans = _validated_spans(item.get("source_spans"), turn_text)
        field_name = str(item.get("field_name") or item.get("slot") or "").strip()
        value = str(item.get("value") or "").strip()
        lifecycle = str(item.get("lifecycle") or "assert").casefold()
        if (spans and allowed_output_turn_ids is not None
                and not any(span.turn_id in allowed_output_turn_ids for span in spans)):
            cites_new = any(isinstance(c, dict) and c.get("turn_id") in allowed_output_turn_ids
                            for c in item.get("source_spans") or [])
            batch.rejected.append({"kind": "fact",
                                   "reason": "new_source_not_grounded" if cites_new else "context_only_record",
                                   "lifecycle": lifecycle})
            continue
        # A lifecycle record describes a change. Its literal supporting sentence
        # is a useful memory even when the model redundantly encodes a boolean,
        # omits a redundant value, or repeats a deleted value absent from that quote.
        # Preserve the entire uniquely cited sentence, never infer a replacement
        # value or splice multiple sources. This is not a claim release: answer
        # values still require their own exact candidate grounding.
        exact_sources = {span.span for span in spans}
        source_only_value = (not value or value.casefold() in {"true", "false"}
                             or lifecycle in {"delete", "cancel"})
        if (lifecycle in {"update", "supersede", "cancel", "delete"} and source_only_value
                and len(exact_sources) == 1
                and (not value or not any(value in span.span for span in spans))):
            exact = next(iter(exact_sources))
            source_repairs.append({"kind": "fact", "reason": "lifecycle_stored_as_exact_source",
                                   "original_value": value, "source_value": exact})
            value = exact
        combined_spans = " ".join(span.span for span in spans)
        if value and spans and not any(value in span.span for span in spans):
            # Recover a compressed value only from a unique exact cited span.
            # All value tokens must occur in order; retain the entire source,
            # including omitted qualifiers/negations, rather than the paraphrase.
            tokens = re.findall(r"\w+", value.casefold())
            matches = set()
            for span in spans:
                remaining = iter(re.findall(r"\w+", span.span.casefold()))
                if tokens and all(any(word == token for word in remaining) for token in tokens):
                    matches.add(span.span)
            if len(matches) == 1:
                exact = matches.pop()
                source_repairs.append({"kind": "fact", "reason": "compressed_value_replaced_with_exact_source",
                                       "original_value": value, "source_value": exact})
                value = exact
        if not spans or not field_name or not value or not any(value in span.span for span in spans):
            batch.rejected.append({"kind": "fact", "reason": "value_not_source_grounded", "lifecycle": str(item.get("lifecycle") or "assert")})
            continue
        subject_id = str(item.get("subject_id") or "").strip() or None
        if subject_id and subject_id not in known_ids:
            batch.rejected.append({"kind": "fact", "reason": "unknown_subject_id", "subject_id": subject_id, "lifecycle": str(item.get("lifecycle") or "assert")})
            continue
        lifecycle = str(item.get("lifecycle") or "assert").casefold()
        temporal = str(item.get("temporal_state") or "unknown").casefold()
        batch.facts.append(V8Fact(
            fact_id=str(item.get("fact_id") or "").strip() or _stable_id("fact", field_name, value, spans[0].turn_id),
            scene_id=str(item.get("scene_id") or "").strip() or None,
            subject_id=subject_id,
            field_name=field_name,
            value=value,
            temporal_state=temporal if temporal in _TEMPORAL_STATES else "unknown",
            lifecycle=lifecycle if lifecycle in _LIFECYCLES else "unknown",
            source_spans=spans,
            confidence=_confidence(item.get("confidence")),
        ))

    for item in payload.get("relations") or []:
        if not isinstance(item, dict):
            continue
        spans = _validated_spans(item.get("source_spans"), turn_text)
        source_id = str(item.get("source_id") or "").strip()
        target_id = str(item.get("target_id") or "").strip()
        if not spans or source_id not in known_ids or target_id not in known_ids:
            batch.rejected.append({"kind": "relation", "reason": "ungrounded_or_unknown_endpoint"})
            continue
        relation_type = str(item.get("relation_type") or "related_to").casefold()
        attributes = dict(item.get("attributes") or {})
        if relation_type == "operational_duty":
            if (not registry.contains(source_id)
                    or not all(isinstance(attributes.get(k), str) and attributes[k].strip()
                               for k in ("role", "action", "resource_category"))
                    or (attributes.get("subject_id") and attributes["subject_id"] not in known_ids)):
                batch.rejected.append({"kind": "relation", "reason": "invalid_typed_duty"})
                continue
        batch.relations.append(V8Relation(
            relation_id=str(item.get("relation_id") or "").strip() or _stable_id(
                "relation", relation_type, source_id, target_id, spans[0].turn_id,
            ),
            relation_type=relation_type,
            source_id=source_id,
            target_id=target_id,
            scene_id=str(item.get("scene_id") or "").strip() or None,
            attributes=attributes,
            source_spans=spans,
            confidence=_confidence(item.get("confidence")),
        ))

    for item in payload.get("governance_events") or []:
        if not isinstance(item, dict):
            continue
        spans = _validated_spans(item.get("source_spans"), turn_text)
        event_type = str(item.get("event_type") or "permission").casefold()
        effect = str(item.get("effect") or "").casefold()
        issuer = str(item.get("issuer_principal_id") or "").strip() or None
        grantee = str(item.get("grantee_principal_id") or "").strip() or None
        # Small models occasionally omit event_type on lifecycle updates. A
        # lifecycle effect has no grantee by definition; recover that
        # unambiguous shape before applying the permission contract. Never
        # infer a lifecycle event for allow/deny/revoke/require_permission.
        if effect in _LIFECYCLES:
            event_type = "lifecycle"
        resource_surface = str(item.get("resource_surface") or "").strip()
        resource_id = str(item.get("resource_id") or "").strip()
        if not spans:
            batch.rejected.append({"kind": "governance_event", "reason": "missing_source_span"})
            continue
        if allowed_output_turn_ids and not any(span.turn_id in allowed_output_turn_ids for span in spans):
            cites_new = any(isinstance(c, dict) and c.get("turn_id") in allowed_output_turn_ids
                            for c in item.get("source_spans") or [])
            batch.rejected.append({"kind": "governance_event",
                                   "reason": "new_source_not_grounded" if cites_new else "context_only_event"})
            continue
        if event_type not in _EVENT_TYPES or effect not in _EFFECTS | _LIFECYCLES:
            batch.rejected.append({"kind": "governance_event", "reason": "invalid_event_enum"})
            continue
        if issuer and not registry.contains(issuer):
            batch.rejected.append({"kind": "governance_event", "reason": "unknown_issuer", "value": issuer})
            continue
        if grantee and not registry.contains(grantee):
            batch.rejected.append({"kind": "governance_event", "reason": "unknown_grantee", "value": grantee})
            continue
        if event_type == "permission" and not grantee:
            batch.rejected.append({"kind": "governance_event", "reason": "missing_grantee"})
            continue
        if not resource_surface or not resource_id:
            batch.rejected.append({"kind": "governance_event", "reason": "missing_resource"})
            continue
        # The resource may be grounded in any cited span, while the subject can
        # be resolved through prior context and the closed roster.
        grounding_surfaces = [
            resource_surface,
            *[str(value) for value in item.get("included_scopes") or []],
            *[str(value) for value in item.get("excluded_scopes") or []],
        ]
        combined_spans = " ".join(span.span for span in spans)
        if not any(_norm(value) in _norm(combined_spans) for value in grounding_surfaces if _norm(value)):
            batch.rejected.append({
                "kind": "governance_event", "reason": "resource_not_grounded",
                "resource_surface": resource_surface,
            })
            continue
        batch.governance_events.append(V8GovernanceEvent(
            event_id=str(item.get("event_id") or "").strip() or _stable_id(
                "governance", event_type, effect, grantee, resource_id, spans[-1].turn_id,
            ),
            event_type=event_type,
            effect=effect,
            issuer_principal_id=issuer,
            grantee_principal_id=grantee,
            action=str(item.get("action") or "access").casefold(),
            resource_id=resource_id,
            resource_surface=resource_surface,
            scene_id=str(item.get("scene_id") or "").strip() or None,
            included_scopes=[str(value).strip() for value in item.get("included_scopes") or [] if str(value).strip()],
            excluded_scopes=[str(value).strip() for value in item.get("excluded_scopes") or [] if str(value).strip()],
            condition=str(item.get("condition") or "").strip() or None,
            valid_from_turn_id=str(item.get("valid_from_turn_id") or spans[-1].turn_id),
            valid_until=str(item.get("valid_until") or "").strip() or None,
            source_spans=spans,
            confidence=_confidence(item.get("confidence")),
        ))

    # Context may resolve a reference, but never creates another observation
    # unless at least one exact source is newly visible.
    for field in ("entities", "facts", "relations"):
        accepted = []
        for record in getattr(batch, field):
            if allowed_output_turn_ids is not None and not any(
                span.turn_id in allowed_output_turn_ids for span in record.source_spans
            ):
                batch.rejected.append({"kind": field, "reason": "context_only_record"})
            else:
                accepted.append(record)
        setattr(batch, field, accepted)

    batch.audit = {
        "source_repairs": source_repairs,
        "raw_counts": {
            key: len(payload.get(key) or [])
            for key in ("scenes", "entities", "facts", "relations", "governance_events")
        },
        "accepted_counts": {
            "scenes": len(batch.scenes),
            "entities": len(batch.entities),
            "facts": len(batch.facts),
            "relations": len(batch.relations),
            "governance_events": len(batch.governance_events),
        },
        "rejected_count": len(batch.rejected),
        "rejected_reasons": dict(
            (reason, sum(1 for item in batch.rejected if item.get("reason") == reason))
            for reason in sorted({str(item.get("reason") or "unknown") for item in batch.rejected})
        ),
        "rejected_samples": list(batch.rejected[:12]),
    }
    return batch
