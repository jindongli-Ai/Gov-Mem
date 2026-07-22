"""Closed-set semantic realization for typed memory answers.

The base model chooses among observed record-local fields.  Deterministic
validation then enforces source grounding and lifecycle constraints; no
domain vocabulary is used by this module.
"""

from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import AnswerResult, RetrievedEvidence
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


_LIFECYCLE_BLOCKLIST = {"deleted", "superseded", "canceled", "historical", "retired"}


def realize_typed_request(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    llm_client: LLMClient | None,
    model_name: str,
    action: str = "answer",
    redacted_attributes: set[str] | None = None,
) -> AnswerResult | None:
    """Realize a request from a closed source-record and slot set.

    The model is not allowed to create values, widen the source closure, or
    decide access.  A failed audit returns ``None`` so the caller can retain
    its existing terminal-policy behavior.
    """
    requested = _requested_attributes(semantic_spec)
    records = _record_payload(evidence)
    if not requested or not records or llm_client is None or not llm_client.is_available():
        return None
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are a typed memory realization auditor. Return JSON only. "
                "Select exact values only from the closed candidate records. "
                "Do not infer, paraphrase, authorize, or expose a neighboring field."
            ),
            user_prompt=(
                "Return {\"realizations\":[{\"attribute\":string,\"memory_id\":string,"
                "\"slot_name\":string,\"value\":string,\"evidence_span\":string,"
                "\"status\":\"renderable|unresolved|retired\","
                "\"reason\":string}]}. For each requested attribute, select the most direct"
                " source-local typed slot whose value answers that attribute. A source record"
                " may contain several fields: emit only fields that realize a requested"
                " attribute, never every field in the record. Values must be copied exactly"
                " from a listed slot and evidence_span must be copied exactly from the same"
                " record's source_text. If the request asks for a retired/deleted predecessor"
                " and the candidates only state a current replacement, mark it retired or"
                " unresolved and do not emit the replacement as the answer. A current status"
                " predicate is not the value of the property it describes when another listed"
                " slot contains that value. For collections, emit each independently supported"
                " record-local value, preserving attribute/value association. Do not return"
                " an item for an attribute that has no exact source-grounded realization."
                " Respect typed role compatibility: do not use a status/state/condition slot as"
                " the value of another property, do not map time/window fields to locations or"
                " destinations (or vice versa), and do not map access artifacts such as PINs,"
                " tokens, passwords, or credentials to physical zones or locations. When several"
                " typed components in one source claim jointly realize a composite attribute,"
                " emit the smallest exact source substring containing the complete value."
                f"\nQuestion: {question}\nSemantic contract: {semantic_spec}"
                f"\nRequested attributes: {requested}"
                f"\nClosed candidate records: {records}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return None

    accepted: list[dict[str, Any]] = []
    records_by_id = {str(row["memory_id"]): row for row in records}
    requested_set = set(requested)
    for item in _realization_rows(raw):
        attribute = str(item.get("attribute") or "").strip()
        memory_id = str(item.get("memory_id") or "").strip()
        slot_name = str(item.get("slot_name") or "").strip()
        value = str(item.get("value") or "").strip()
        evidence_span = str(item.get("evidence_span") or "").strip()
        status = str(item.get("status") or "").strip().lower()
        record = records_by_id.get(memory_id)
        if (
            attribute not in requested_set
            or status != "renderable"
            or record is None
            or str(record.get("lifecycle_status") or "active").lower() in _LIFECYCLE_BLOCKLIST
            or not slot_name
            or not value
            or not evidence_span
            or not _slot_compatible(attribute, slot_name)
            or _is_outer_collection_slot(attribute, slot_name, semantic_spec)
            or _contains_non_access_artifact(attribute, slot_name, value)
            or not _valid_slot(record, slot_name, value)
            or value not in str(record.get("source_text") or "")
            or evidence_span not in str(record.get("source_text") or "")
            or value not in evidence_span
        ):
            continue
        if redacted_attributes is not None and attribute not in redacted_attributes:
            continue
        accepted.append({
            "attribute": attribute,
            "memory_id": memory_id,
            "slot_name": slot_name,
            "value": value,
            "evidence_span": evidence_span,
        })

    # A realization call is a semantic selector, not a completeness oracle.
    # Preserve the attribute binding emitted by Stage 2 so a transiently
    # incomplete response cannot drop an otherwise admitted field from a
    # multi-attribute utility answer.  The completion remains closed-set:
    # source text, lifecycle, slot value, and the original requested
    # attribute must all match the already admitted candidate.
    accepted_attributes = {str(item.get("attribute") or "") for item in accepted}
    collection_attributes = {
        str(binding.get("attribute") or "").strip()
        for key in ("attribute_bindings", "certifiable_needs")
        for binding in list(semantic_spec.get(key) or [])
        if isinstance(binding, dict)
        and str(binding.get("attribute") or binding.get("need_id") or "").strip()
        and str(binding.get("need_kind") or binding.get("binding_kind") or "").strip()
        in {"record_collection", "collection", "list"}
    }
    collection_attributes.update(
        attribute
        for attribute in requested
        if _implicit_collection_attribute(attribute, semantic_spec)
    )
    for attribute in requested:
        if attribute in collection_attributes:
            existing = {
                (str(item.get("memory_id") or ""), str(item.get("slot_name") or ""), str(item.get("value") or ""))
                for item in accepted
                if str(item.get("attribute") or "") == attribute
            }
            for fallback in _attribute_bound_fallbacks(attribute, records, semantic_spec):
                key = (fallback["memory_id"], fallback["slot_name"], fallback["value"])
                if key not in existing:
                    accepted.append(fallback)
                    existing.add(key)
            if existing:
                accepted_attributes.add(attribute)
            continue
        if attribute in accepted_attributes:
            continue
        fallback = _attribute_bound_fallback(attribute, records, semantic_spec)
        if fallback is not None:
            accepted.append(fallback)
            accepted_attributes.add(attribute)

    # Keep the first source-grounded value for a scalar attribute, while
    # preserving distinct values for a typed collection.
    chosen: list[dict[str, Any]] = []
    seen_scalar: set[str] = set()
    seen_values: set[tuple[str, str, str]] = set()
    best_collection_item: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in accepted:
        key = (item["attribute"], item["memory_id"], item["value"])
        if key in seen_values:
            continue
        if item["attribute"] in collection_attributes:
            source_slot = (item["attribute"], item["memory_id"], item["slot_name"])
            prior = best_collection_item.get(source_slot)
            if prior is not None:
                if len(str(prior.get("value") or "")) >= len(str(item.get("value") or "")):
                    continue
                chosen.remove(prior)
            best_collection_item[source_slot] = item
        if item["attribute"] not in collection_attributes and item["attribute"] in seen_scalar:
            continue
        seen_values.add(key)
        seen_scalar.add(item["attribute"])
        chosen.append(item)
    if not chosen:
        return AnswerResult(
            prediction="",
            answer_text="",
            used_memory_ids=[],
            reasoning_summary="Closed-set typed realization found no renderable value.",
            action=action,
            answer_structured={
                "audit_status": "retired" if any(
                    str(item.get("status") or "").lower() == "retired"
                    for item in _realization_rows(raw)
                ) else "unresolved",
                "realizations": _realization_rows(raw),
                "realization_mode": "closed_set_typed_realization_audit",
            },
        )

    typed_slots: dict[str, str | list[str]] = {}
    clauses: list[str] = []
    used_memory_ids: list[str] = []
    for item in chosen:
        attribute = item["attribute"]
        value = item["value"]
        existing = typed_slots.get(attribute)
        is_new_value = (
            existing is None
            or (isinstance(existing, list) and value not in existing)
            or (not isinstance(existing, list) and existing != value)
        )
        if existing is None:
            typed_slots[attribute] = value
        elif isinstance(existing, list):
            if value not in existing:
                existing.append(value)
        elif existing != value:
            typed_slots[attribute] = [existing, value]
        if item["memory_id"] not in used_memory_ids:
            used_memory_ids.append(item["memory_id"])
        if is_new_value:
            clauses.append(f"{attribute.replace('_', ' ')}: {value}")
    if not clauses:
        return None
    return AnswerResult(
        prediction="; ".join(clauses) + ".",
        answer_text="; ".join(clauses) + ".",
        used_memory_ids=used_memory_ids,
        reasoning_summary="Closed-set typed realization with deterministic source and lifecycle validation.",
        action=action,
        answer_structured={
            "typed_slots": typed_slots,
            "audit_status": "rendered",
            "realization_mode": "closed_set_typed_realization_audit",
            "realizations": chosen,
        },
    )


def restrict_semantic_spec(
    semantic_spec: dict[str, Any],
    attributes: set[str],
) -> dict[str, Any]:
    """Create a contract for the subset already admitted by upstream stages."""
    result = dict(semantic_spec or {})
    requested = _requested_attributes(result)
    kept = [attribute for attribute in requested if attribute in attributes]
    result["requested_attributes"] = kept
    result["requested_slots"] = kept
    for key in ("attribute_bindings", "certifiable_needs"):
        values = result.get(key)
        if not isinstance(values, list):
            continue
        result[key] = [
            item for item in values
            if isinstance(item, dict)
            and str(item.get("attribute") or item.get("need_id") or "").strip() in attributes
        ]
    return result


def _requested_attributes(semantic_spec: dict[str, Any]) -> list[str]:
    needs = semantic_spec.get("certifiable_needs")
    if isinstance(needs, list):
        values = [
            str(item.get("attribute") or item.get("need_id") or "").strip()
            for item in needs
            if isinstance(item, dict)
        ]
    else:
        values = list(semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or [])
    return list(dict.fromkeys(value for value in values if value))


def _implicit_collection_attribute(attribute: str, semantic_spec: dict[str, Any]) -> bool:
    """Recognize open-schema bundle labels without domain vocabulary."""
    target = str(attribute or "").strip()
    explicit_binding_seen = False
    for key in ("attribute_bindings", "certifiable_needs"):
        for item in list(semantic_spec.get(key) or []):
            if not isinstance(item, dict):
                continue
            item_attribute = str(item.get("attribute") or item.get("need_id") or "").strip()
            if item_attribute != target:
                continue
            explicit_binding_seen = True
            binding_kind = str(item.get("need_kind") or item.get("binding_kind") or "").strip().lower()
            return binding_kind in {"record_collection", "collection", "list"}
    # Once the planner has named a scalar field explicitly, the outer request
    # shape must not reinterpret that field as a collection.  This matters
    # for mixed requests such as {start, stop, changes, plan}.
    if explicit_binding_seen:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    if tokens & {"snapshot", "summary", "overview", "recap", "plan"}:
        return True
    return str(semantic_spec.get("request_shape") or "").strip().lower() in {"list", "plan"}


def _is_collection_attribute(attribute: str, semantic_spec: dict[str, Any]) -> bool:
    """Read explicit collection binding before applying open-schema cues."""
    target = str(attribute or "").strip()
    for key in ("attribute_bindings", "certifiable_needs"):
        for item in list(semantic_spec.get(key) or []):
            if not isinstance(item, dict):
                continue
            item_attribute = str(item.get("attribute") or item.get("need_id") or "").strip()
            if item_attribute != target:
                continue
            return str(item.get("need_kind") or item.get("binding_kind") or "").strip().lower() in {
                "record_collection", "collection", "list"
            }
    return _implicit_collection_attribute(target, semantic_spec)


def _record_payload(evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in evidence:
        memory_id = str(row.memory_id or "").strip()
        if not memory_id or memory_id in seen:
            continue
        metadata = dict(row.metadata or {})
        semantic_tags = dict(metadata.get("semantic_tags") or {})
        claim_slots = [
            {
                "slot_name": str(claim.get("property_label") or "").strip(),
                "value": str(
                    dict(semantic_tags.get("attributes") or {}).get(
                        str(claim.get("property_label") or "").strip(),
                        claim.get("value_span") or "",
                    )
                ).strip(),
            }
            for claim in list(semantic_tags.get("claims") or [])
            if isinstance(claim, dict)
            and str(claim.get("property_label") or "").strip()
            and str(claim.get("value_span") or "").strip()
        ]
        adjudicated_fields = [
            field
            for field in list(metadata.get("claim_adjudication") or [])
            if isinstance(field, dict)
            and str(field.get("decision") or "").strip().lower() in {"answer", "redact"}
            and str(field.get("attribute") or "").strip()
            and str(field.get("slot_name") or "").strip()
            and str(field.get("value") or "").strip()
        ]
        adjudicated_attributes = {
            str(field.get("attribute") or "").strip()
            for field in adjudicated_fields
        }
        stage2_fields = [
            field
            for field in list((metadata.get("stage2_semantic_rerank") or {}).get("typed_fields") or [])
            if isinstance(field, dict)
            and str(field.get("attribute") or "").strip()
            and str(field.get("value") or "").strip()
            and str(field.get("attribute") or "").strip() not in adjudicated_attributes
        ]
        slots: dict[str, Any] = {}
        # Semantic claim spans are the most precise closed-set values. Seed
        # them before heuristic aliases so a sentence-shaped extractor field
        # cannot replace a typed property with its neighboring context.
        for claim in claim_slots:
            slots.setdefault(claim["slot_name"], claim["value"])
        for container in (
            metadata.get("slots"),
            metadata.get("surface_spans"),
            semantic_tags.get("attributes"),
            semantic_tags.get("surface_values"),
        ):
            if isinstance(container, dict):
                for key, value in container.items():
                    key = str(key or "").strip()
                    if isinstance(value, list):
                        values = [str(item or "").strip() for item in value if str(item or "").strip()]
                        if key and values and key not in slots:
                            slots[key] = values
                    else:
                        value = str(value or "").strip()
                        if key and value and key not in slots:
                            slots[key] = value
        # Stage 2's source-local binding is the strongest available typed
        # representation.  Generic extractor slots can retain a neighboring
        # value under a broad name (for example a public budget beside a
        # private amount), so let the admitted typed field win that collision.
        stage2_attributes = {str(field.get("attribute") or "").strip() for field in stage2_fields}
        for field in adjudicated_fields:
            slot_name = str(field.get("slot_name") or "").strip()
            value = str(field.get("value") or "").strip()
            if slot_name and value:
                # Claim adjudication has already resolved the source-local
                # field and may preserve a complete claim span where Stage 2
                # kept only a short extractor alias.
                slots[slot_name] = value
        for field in stage2_fields:
            slot_name = str(field.get("slot_name") or field.get("attribute") or "").strip()
            value = _prefer_precise_claim_value(
                slot_name=slot_name,
                proposed_value=str(field.get("value") or "").strip(),
                claim_slots=claim_slots,
                source_text=str(row.content or ""),
            )
            if slot_name and value:
                slots.setdefault(slot_name, value)
        for candidate in list(metadata.get("typed_candidates") or []):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("attribute") or "").strip() in stage2_attributes:
                continue
            slot_name = str(candidate.get("slot_name") or "").strip()
            value = str(candidate.get("value") or "").strip()
            if slot_name and value:
                slots.setdefault(slot_name, value)
        source_text = str(row.content or "")
        candidate_sources = [
            str(candidate.get("source_text") or "").strip()
            for candidate in list(metadata.get("typed_candidates") or [])
            if isinstance(candidate, dict) and str(candidate.get("source_text") or "").strip()
        ]
        if candidate_sources and not stage2_fields:
            source_text = "\n".join(dict.fromkeys(candidate_sources))
        rows.append({
            "memory_id": memory_id,
            "source_text": source_text,
            "slots": slots,
            "lifecycle_status": str(metadata.get("lifecycle_status") or metadata.get("memory_status") or "active"),
            "source_message_ids": list(row.source_message_ids or []),
            "typed_attributes": [
                {
                    "attribute": str(field.get("attribute") or "").strip(),
                    "slot_name": str(field.get("slot_name") or "").strip(),
                    "value": str(field.get("value") or "").strip(),
                    "claim_span": str(field.get("claim_span") or "").strip(),
                    "evidence_span": str(field.get("evidence_span") or source_text).strip(),
                    "source_memory_id": memory_id,
                }
                for field in adjudicated_fields
                if str(field.get("slot_name") or "").strip()
                and str(field.get("value") or "").strip()
            ] + [
                {
                    "attribute": str(field.get("attribute") or "").strip(),
                    "slot_name": str(field.get("slot_name") or field.get("attribute") or "").strip(),
                    "value": _prefer_precise_claim_value(
                        slot_name=str(field.get("slot_name") or field.get("attribute") or "").strip(),
                        proposed_value=str(field.get("value") or "").strip(),
                        claim_slots=claim_slots,
                        source_text=source_text,
                    ),
                    "evidence_span": str(
                        field.get("evidence_span")
                        or (metadata.get("stage2_semantic_rerank") or {}).get("support_span")
                        or source_text
                    ).strip(),
                    "source_memory_id": memory_id,
                }
                for field in stage2_fields
                if str(field.get("slot_name") or field.get("attribute") or "").strip()
                and str(field.get("value") or "").strip()
            ] + [
                {
                    "attribute": str(candidate.get("attribute") or "").strip(),
                    "slot_name": str(candidate.get("slot_name") or "").strip(),
                    "value": str(candidate.get("value") or "").strip(),
                    "claim_span": str(candidate.get("claim_span") or "").strip(),
                    "evidence_span": str(candidate.get("evidence_span") or source_text).strip(),
                    "source_memory_id": str(candidate.get("source_memory_id") or memory_id),
                }
                for candidate in list(metadata.get("typed_candidates") or [])
                if isinstance(candidate, dict)
                and str(candidate.get("attribute") or "").strip() not in stage2_attributes
                and str(candidate.get("attribute") or "").strip()
                and str(candidate.get("slot_name") or "").strip()
                and str(candidate.get("value") or "").strip()
            ],
        })
        seen.add(memory_id)
    return rows


def _prefer_precise_claim_value(
    *,
    slot_name: str,
    proposed_value: str,
    claim_slots: list[dict[str, str]],
    source_text: str,
) -> str:
    """Prefer an exact claim value over a sentence-shaped legacy alias."""
    proposed = str(proposed_value or "").strip()
    if not proposed:
        return proposed
    values = [
        str(claim.get("value") or "").strip()
        for claim in claim_slots
        if str(claim.get("slot_name") or "").strip() == str(slot_name or "").strip()
        and str(claim.get("value") or "").strip()
        and str(claim.get("value") or "").strip() in source_text
    ]
    if not values or proposed in values:
        return proposed
    narrower = [value for value in values if value in proposed]
    return max(narrower, key=len) if narrower else proposed


def _valid_slot(record: dict[str, Any], slot_name: str, value: str) -> bool:
    slots = dict(record.get("slots") or {})
    observed = slots.get(slot_name)
    if isinstance(observed, list):
        observed_values = {str(item or "").strip() for item in observed}
    else:
        observed_values = {str(observed or "").strip()}
    if value in observed_values:
        return True
    return bool(value and value in str(record.get("source_text") or "") and any(
        item and item in value for item in observed_values
    ))


def _slot_compatible(attribute: str, slot_name: str) -> bool:
    """Reject generic cross-role bindings before rendering source text."""
    attr_tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    slot_tokens = set(re.findall(r"[a-z0-9]+", str(slot_name or "").lower()))
    ignored = {"a", "an", "the", "and", "current", "latest", "active", "for", "of", "to"}
    attr_tokens -= ignored
    slot_tokens -= ignored
    status_tokens = {"state", "status", "condition", "predicate"}
    temporal_tokens = {"time", "window", "date", "schedule", "interval", "weekday", "day", "hour", "deadline"}
    spatial_tokens = {"location", "locations", "destination", "destinations", "zone", "zones", "area", "areas", "room", "rooms", "desk", "desks", "path", "paths", "route", "routes", "place", "places", "bay", "bays", "point", "points"}
    access_tokens = {"pin", "password", "passcode", "token", "credential", "secret", "key", "phrase"}
    if slot_tokens & status_tokens and not attr_tokens & status_tokens:
        # Presence/absence questions are intentionally realized from a
        # source-grounded status predicate.  The attribute is often phrased
        # as a question (whether/any/remain) while the record uses a typed
        # state slot (status/condition).  Keep the binding closed-set and
        # exact; only this semantic role bridge is allowed.
        presence_tokens = {
            "whether", "remain", "remains", "remaining", "any", "some",
            "none", "present", "presence", "exist", "exists", "left",
            "available", "missing", "absence",
        }
        if not attr_tokens & presence_tokens:
            return False
    if (attr_tokens & temporal_tokens and slot_tokens & spatial_tokens) or (
        attr_tokens & spatial_tokens and slot_tokens & temporal_tokens
    ):
        return False
    if attr_tokens & spatial_tokens and slot_tokens & access_tokens and not attr_tokens & access_tokens:
        return False
    return True


def _attribute_bound_fallback(
    attribute: str,
    records: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Recover one omitted field from Stage-2's typed attribute binding."""
    fallbacks = _attribute_bound_fallbacks(attribute, records, semantic_spec)
    return fallbacks[0] if fallbacks else None


def _attribute_bound_fallbacks(
    attribute: str,
    records: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Recover every closed-set typed member bound to one collection need."""
    ranked: list[tuple[int, int, str, dict[str, Any], str, str, str]] = []
    for record in records:
        source_text = str(record.get("source_text") or "")
        if not source_text or str(record.get("lifecycle_status") or "active").lower() in _LIFECYCLE_BLOCKLIST:
            continue
        for candidate in list(record.get("typed_attributes") or []):
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("attribute") or "").strip() != attribute:
                continue
            slot_name = str(candidate.get("slot_name") or "").strip()
            value = str(candidate.get("value") or "").strip()
            evidence_span = str(candidate.get("evidence_span") or source_text).strip()
            claim_span = str(candidate.get("claim_span") or "").strip()
            if (
                _is_collection_attribute(attribute, semantic_spec or {})
                and claim_span
                and value in claim_span
                and claim_span in source_text
            ):
                # An extractor slot can retain only the first member of a
                # list-shaped claim. The adjudicated claim span is the
                # source-local complete value for that collection.
                value = claim_span
            if (
                not slot_name
                or not value
                or not _slot_compatible(attribute, slot_name)
                or _is_outer_collection_slot(attribute, slot_name, semantic_spec)
                or _contains_non_access_artifact(attribute, slot_name, value)
                or not _valid_slot(record, slot_name, value)
                or value not in source_text
                or not evidence_span
                or evidence_span not in source_text
                or value not in evidence_span
            ):
                continue
            ranked.append((
                int(bool(candidate.get("source_memory_id"))),
                len(value),
                str(record.get("memory_id") or ""),
                record,
                slot_name,
                value,
                evidence_span,
            ))
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for _, _, memory_id, _, slot_name, value, evidence_span in sorted(
        ranked, key=lambda item: item[:3], reverse=True
    ):
        key = (memory_id, slot_name, value)
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "attribute": attribute,
            "memory_id": memory_id,
            "slot_name": slot_name,
            "value": value,
            "evidence_span": evidence_span,
        })
    return selected


def _is_outer_collection_slot(
    attribute: str,
    slot_name: str,
    semantic_spec: dict[str, Any] | None,
) -> bool:
    """Identify a collection contract key reused as a raw source slot."""
    attribute = str(attribute or "").strip()
    slot_name = str(slot_name or "").strip()
    if not attribute or not slot_name:
        return False
    is_collection = any(
        isinstance(item, dict)
        and str(item.get("attribute") or item.get("need_id") or "").strip() == attribute
        and str(item.get("need_kind") or item.get("binding_kind") or "").strip().lower()
        in {"record_collection", "collection", "list"}
        for key in ("attribute_bindings", "certifiable_needs")
        for item in list((semantic_spec or {}).get(key) or [])
    )
    if not is_collection:
        return False
    structural = {"summary", "overview", "recap"}
    attr_tokens = set(re.findall(r"[a-z0-9]+", attribute.lower())) - structural
    slot_tokens = set(re.findall(r"[a-z0-9]+", slot_name.lower())) - structural
    return bool(attr_tokens and attr_tokens == slot_tokens)


def _contains_non_access_artifact(attribute: str, slot_name: str, value: str) -> bool:
    """Reject compound credential-bearing values for non-access roles."""
    access_tokens = {"pin", "password", "passcode", "token", "credential", "secret", "key", "phrase"}
    attr_tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    slot_tokens = set(re.findall(r"[a-z0-9]+", str(slot_name or "").lower()))
    value_tokens = set(re.findall(r"[a-z0-9]+", str(value or "").lower()))
    return bool(
        not attr_tokens & access_tokens
        and not slot_tokens & access_tokens
        and value_tokens & access_tokens
        and re.search(r"\d", str(value or ""))
    )


def _realization_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for container in (raw.get("realizations"), raw.get("items"), raw.get("results")):
        if isinstance(container, list):
            return [dict(item) for item in container if isinstance(item, dict)]
    return []
