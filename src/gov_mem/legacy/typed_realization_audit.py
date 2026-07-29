"""LEGACY: closed-set semantic realization for typed memory answers.

The base model chooses among observed record-local fields.  Deterministic
validation then enforces source grounding and lifecycle constraints; no
domain vocabulary is used by this module.
"""

from __future__ import annotations

import re
import calendar
from typing import Any

from gov_mem.data.schema import AnswerResult, RetrievedEvidence
from gov_mem.legacy.claim_adjudicator import (
    _meaningful_field_tokens,
    _normalize_observed_value,
    _slot_matches_attribute,
)
from gov_mem.governance_runtime.factual_claim_quality import factual_value_is_eligible
from gov_mem.governance_runtime.source_grounding import row_grounded_source_text
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
    records = _record_payload(evidence, semantic_spec)
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
            or not _slot_compatible(attribute, slot_name, semantic_spec)
            or not _slot_matches_attribute(
                attribute,
                slot_name,
                semantic_spec,
                record,
            )
            or _is_outer_collection_slot(attribute, slot_name, semantic_spec)
            or _contains_non_access_artifact(attribute, slot_name, value)
            or not factual_value_is_eligible(
                attribute=attribute,
                slot_name=slot_name,
                value=value,
                semantic_spec=semantic_spec,
                source_text=str(record.get("source_text") or ""),
            )
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
    scalar_requested = set(requested) - collection_attributes
    if scalar_requested:
        # Once a request exposes named scalar members, an outer bundle is
        # only a retrieval container. Keeping it as another answer field
        # reintroduces unrelated neighbors and stale state.
        accepted = [
            item for item in accepted
            if str(item.get("attribute") or "") not in collection_attributes
        ]
        # A mixed request can still ask for the outer record itself. Preserve
        # only a source-grounded temporal anchor from that record when it is
        # not represented by an explicit scalar member. This keeps the
        # record's identity (for example, its calendar date) without reviving
        # superseded operational fields from the same source record.
        accepted.extend(
            _collection_context_items(
                records=records,
                collection_attributes=collection_attributes,
                requested_attributes=set(requested),
                redacted_attributes=redacted_attributes,
            )
        )
    for attribute in requested:
        if attribute in collection_attributes:
            if scalar_requested:
                # An explicitly expanded mixed request uses the outer bundle
                # only as a retrieval container. Do not re-add it through the
                # collection fallback loop after the initial scalar filter.
                continue
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

    # Scalar requests are chronology-resolved after closed-set validation.
    # Stage 2 can return several source-grounded candidates for one field;
    # keeping the first response would revive an older value merely because
    # it appeared earlier in the model output.
    for attribute in requested:
        if attribute in collection_attributes:
            continue
        candidates_for_attribute = [
            item for item in accepted
            if str(item.get("attribute") or "") == attribute
        ]
        if len(candidates_for_attribute) <= 1:
            continue
        winner = max(
            candidates_for_attribute,
            key=lambda item: _scalar_current_rank(
                item,
                records_by_id.get(str(item.get("memory_id") or ""), {}),
                semantic_spec,
            ),
        )
        accepted = [
            item for item in accepted
            if str(item.get("attribute") or "") != attribute or item is winner
        ]

    # A collection may be represented by one complete source-local record or
    # by several one-member records. Prefer the former when it is available;
    # this keeps a coherent recap without hard-coding list vocabulary.
    for attribute in collection_attributes:
        collection_items = [
            item for item in accepted
            if str(item.get("attribute") or "") == attribute
        ]
        preferred = _prefer_collection_coherent_items(
            collection_items,
            records_by_id,
            semantic_spec,
        )
        if preferred and len(preferred) < len(collection_items):
            accepted = [
                item for item in accepted
                if str(item.get("attribute") or "") != attribute or item in preferred
            ]

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
    date_context = " ".join(
        str(record.get("source_text") or "")
        for record in records
        if isinstance(record, dict)
    )
    for item in chosen:
        attribute = item["attribute"]
        value = item["value"]
        display_value = _expand_partial_date(value, date_context) if "date" in attribute.lower() else value
        existing = typed_slots.get(attribute)
        is_new_value = (
            existing is None
            or (isinstance(existing, list) and display_value not in existing)
            or (not isinstance(existing, list) and existing != display_value)
        )
        if existing is None:
            typed_slots[attribute] = display_value
        elif isinstance(existing, list):
            if display_value not in existing:
                existing.append(display_value)
        elif existing != display_value:
            typed_slots[attribute] = [existing, display_value]
        if item["memory_id"] not in used_memory_ids:
            used_memory_ids.append(item["memory_id"])
        if is_new_value:
            clauses.append(f"{attribute.replace('_', ' ')}: {display_value}")
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


def _expand_partial_date(value: str, source_context: str) -> str:
    """Normalize a certified month/day using a unique year in the source set."""
    raw = str(value or "").strip()
    if not raw or re.search(r",\s*\d{4}\b", raw):
        return raw
    if not re.fullmatch(
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
        raw,
        re.IGNORECASE,
    ) or raw.lower() not in source_context.lower():
        return raw
    years = list(dict.fromkeys(re.findall(r"\b(20\d{2})\b", source_context)))
    return f"{raw}, {years[0]}" if len(years) == 1 else raw


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


def _record_payload(
    evidence: list[RetrievedEvidence], semantic_spec: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    collection_attributes = {
        str(item.get("attribute") or item.get("need_id") or "").strip()
        for key in ("attribute_bindings", "certifiable_needs")
        for item in list((semantic_spec or {}).get(key) or [])
        if isinstance(item, dict)
        and str(item.get("attribute") or item.get("need_id") or "").strip()
        and str(item.get("need_kind") or item.get("binding_kind") or "").strip().lower()
        in {"record_collection", "collection", "list"}
    }
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
                source_text=row_grounded_source_text(row),
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
        source_text = row_grounded_source_text(row)
        candidate_sources = [
            str(candidate.get("source_text") or "").strip()
            for candidate in list(metadata.get("typed_candidates") or [])
            if isinstance(candidate, dict) and str(candidate.get("source_text") or "").strip()
        ]
        if candidate_sources and not stage2_fields:
            source_text = "\n".join(dict.fromkeys(candidate_sources))
        # Semantic claims are the record's source-local typed closure. They
        # matter when Stage 2 admitted only a neighboring extractor field
        # from the same collection record; adding them here lets the final
        # audit complete that record without widening retrieval. Leave
        # claim_span empty so a claim member is rendered as its exact value,
        # not as the entire recap sentence.
        typed_attributes = [
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
        ]
        if collection_attributes:
            typed_collection_attributes = {
                str(item.get("attribute") or "").strip()
                for item in typed_attributes
                if isinstance(item, dict)
                and str(item.get("attribute") or "").strip() in collection_attributes
            }
            claim_targets = typed_collection_attributes
            if not claim_targets and len(collection_attributes) == 1:
                claim_targets = set(collection_attributes)
            for claim in claim_slots:
                slot_name = str(claim.get("slot_name") or "").strip()
                value = str(claim.get("value") or "").strip()
                if not slot_name or not value or not claim_targets:
                    continue
                for attribute in claim_targets:
                    typed_attributes.append({
                        "attribute": attribute,
                        "slot_name": slot_name,
                        "value": value,
                        "claim_span": "",
                        "evidence_span": value,
                        "source_memory_id": memory_id,
                    })
        rows.append({
            "memory_id": memory_id,
            "source_text": source_text,
            "slots": slots,
            "semantic_tags": dict(metadata.get("semantic_tags") or {}),
            "lifecycle_status": str(metadata.get("lifecycle_status") or metadata.get("memory_status") or "active"),
            "source_message_ids": list(row.source_message_ids or []),
            "typed_attributes": typed_attributes,
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


def _slot_compatible(
    attribute: str,
    slot_name: str,
    semantic_spec: dict[str, Any] | None = None,
) -> bool:
    """Reject generic cross-role bindings before rendering source text."""
    attr_tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    slot_tokens = set(re.findall(r"[a-z0-9]+", str(slot_name or "").lower()))
    ignored = {"a", "an", "the", "and", "current", "latest", "active", "for", "of", "to"}
    attr_tokens -= ignored
    slot_tokens -= ignored
    status_tokens = {"state", "status", "condition", "predicate"}
    temporal_tokens = {"time", "window", "date", "schedule", "interval", "weekday", "day", "hour", "deadline"}
    spatial_tokens = {"location", "locations", "destination", "destinations", "zone", "zones", "area", "areas", "room", "rooms", "desk", "desks", "path", "paths", "route", "routes", "place", "places", "bay", "bays", "point", "points"}
    access_tokens = {
        "pin", "password", "passcode", "token", "credential", "secret", "key",
        "phrase", "badge", "badge_id", "credential_id",
    }
    is_collection = _is_collection_attribute(attribute, semantic_spec or {})
    if slot_tokens & status_tokens and not attr_tokens & status_tokens and not is_collection:
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
    if (
        len(attr_tokens & slot_tokens) == 1
        and attr_tokens - (attr_tokens & slot_tokens)
        and slot_tokens - (attr_tokens & slot_tokens)
        and (
            (attr_tokens & slot_tokens) <= {
                "current", "active", "private", "public", "approved", "main",
                "primary", "secondary", "latest", "scheduled", "target",
            }
            or (attr_tokens & slot_tokens) & {"secret"}
        )
    ):
        return False
    return True


def _scalar_current_rank(
    item: dict[str, Any], record: dict[str, Any], semantic_spec: dict[str, Any]
) -> tuple[int, str, int, int]:
    current = str(semantic_spec.get("temporal_scope") or "").strip().lower() == "current"
    slot = str(item.get("slot_name") or "").strip().lower()
    historical = bool(
        set(re.findall(r"[a-z0-9]+", slot))
        & {"previous", "prior", "initial", "first", "old", "former", "original", "opening", "historical"}
    )
    turns = [
        int(match.group(1))
        for source_id in list(record.get("source_message_ids") or [])
        for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
        if match
    ]
    return (
        int(current and not historical),
        str(record.get("timestamp") or ""),
        max(turns, default=-1),
        len(str(item.get("value") or "")),
    )


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
                or not _slot_compatible(attribute, slot_name, semantic_spec)
                or _is_outer_collection_slot(attribute, slot_name, semantic_spec)
                or _contains_non_access_artifact(attribute, slot_name, value)
                or not factual_value_is_eligible(
                    attribute=attribute,
                    slot_name=slot_name,
                    value=value,
                    semantic_spec=semantic_spec,
                    source_text=source_text,
                )
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
    if _current_collection_has_coherent_record(attribute, semantic_spec or {}):
        coherent_memory_ids = _coherent_collection_memory_ids(ranked, records, semantic_spec or {})
        if coherent_memory_ids:
            ranked = [item for item in ranked if item[2] in coherent_memory_ids]
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


def _collection_context_items(
    *,
    records: list[dict[str, Any]],
    collection_attributes: set[str],
    requested_attributes: set[str],
    redacted_attributes: set[str] | None,
) -> list[dict[str, Any]]:
    """Recover non-overlapping temporal anchors from an outer record bundle."""
    anchor_slots = {"date", "calendar_date", "event_date", "scheduled_date", "target_date"}
    scalar_source_texts = [
        str(record.get("source_text") or "")
        for record in records
        if any(
            isinstance(candidate, dict)
            and str(candidate.get("attribute") or "").strip() not in collection_attributes
            for candidate in list(record.get("typed_attributes") or [])
        )
        and str(record.get("source_text") or "").strip()
    ]
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(record.get("lifecycle_status") or "active").lower() in _LIFECYCLE_BLOCKLIST:
            continue
        source_text = str(record.get("source_text") or "")
        candidates = [
            candidate
            for candidate in list(record.get("typed_attributes") or [])
            if isinstance(candidate, dict)
        ]
        # A source-local record can expose its calendar anchor through the
        # generic slot map even when Stage 2 only typed the requested scalar
        # fields. Keep that anchor available for a mixed collection answer;
        # the collection itself still has to be present on the same record.
        has_collection_candidate = any(
            str(candidate.get("attribute") or "").strip() in collection_attributes
            for candidate in candidates
        )
        if has_collection_candidate:
            observed_slots = dict(record.get("slots") or {})
            for slot_name, observed_value in observed_slots.items():
                slot_name = str(slot_name or "").strip()
                if not slot_name or not _is_temporal_anchor_slot(slot_name):
                    continue
                values = observed_value if isinstance(observed_value, list) else [observed_value]
                for value in values:
                    value = str(value or "").strip()
                    if value:
                        candidates.append({
                            "attribute": next(iter(collection_attributes)),
                            "slot_name": slot_name,
                            "value": value,
                            "evidence_span": source_text,
                            "source_memory_id": record.get("memory_id"),
                        })
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_attribute = str(candidate.get("attribute") or "").strip()
            if candidate_attribute not in collection_attributes:
                continue
            slot_name = str(candidate.get("slot_name") or "").strip()
            if not _is_temporal_anchor_slot(slot_name):
                continue
            value = str(candidate.get("value") or "").strip()
            if not _temporal_anchor_matches_scalar_context(value, scalar_source_texts):
                continue
            if slot_name in requested_attributes:
                continue
            if redacted_attributes is not None and slot_name not in redacted_attributes:
                continue
            evidence_span = str(candidate.get("evidence_span") or source_text).strip()
            if (
                not value
                or not evidence_span
                or value not in source_text
                or value not in evidence_span
                or not _valid_slot(record, slot_name, value)
            ):
                continue
            item = {
                "attribute": slot_name,
                "memory_id": str(candidate.get("source_memory_id") or record.get("memory_id") or "").strip(),
                "slot_name": slot_name,
                "value": value,
                "evidence_span": evidence_span,
            }
            if not item["memory_id"]:
                continue
            prior = selected.get(slot_name)
            rank = (
                len(value),
                str(record.get("timestamp") or ""),
                max(
                    [
                        int(match.group(1))
                        for source_id in list(record.get("source_message_ids") or [])
                        for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
                        if match
                    ],
                    default=-1,
                ),
            )
            prior_rank = prior.get("_context_rank", ()) if prior else ()
            if prior is None or rank > prior_rank:
                item["_context_rank"] = rank
                selected[slot_name] = item
    for item in selected.values():
        item.pop("_context_rank", None)
    return list(selected.values())


def _is_temporal_anchor_slot(slot_name: str) -> bool:
    """Recognize schema-level calendar anchors without domain vocabulary."""
    normalized = str(slot_name or "").strip().lower().replace("-", "_")
    return normalized in {
        "date", "calendar_date", "event_date", "scheduled_date", "target_date",
    } or "date" in set(normalized.split("_"))


def _temporal_anchor_matches_scalar_context(value: str, source_texts: list[str]) -> bool:
    """Reject a bundle date whose explicit weekday conflicts with scalar facts."""
    if not source_texts:
        return True
    lowered_value = str(value or "").lower()
    weekdays = {
        day.lower() for day in calendar.day_name
        if day and re.search(rf"\b{day.lower()}\b", lowered_value)
    }
    months = {
        month.lower() for month in calendar.month_name
        if month and re.search(rf"\b{month.lower()}\b", lowered_value)
    }
    if not weekdays and not months:
        return True
    scalar_text = " ".join(str(text or "").lower() for text in source_texts)
    anchors = weekdays or months
    return any(re.search(rf"\b{anchor}\b", scalar_text) for anchor in anchors)


def _current_collection_has_coherent_record(
    attribute: str, semantic_spec: dict[str, Any]
) -> bool:
    return bool(
        str(semantic_spec.get("temporal_scope") or "").strip().lower() == "current"
        and _is_collection_attribute(attribute, semantic_spec)
    )


def _coherent_collection_memory_ids(
    ranked: list[tuple[int, int, str, dict[str, Any], str, str, str]],
    records: list[dict[str, Any]],
    semantic_spec: dict[str, Any],
) -> set[str]:
    groups: dict[str, list[tuple[int, int, str, dict[str, Any], str, str, str]]] = {}
    for item in ranked:
        groups.setdefault(item[2], []).append(item)
    if not groups:
        return set()
    def source_turn(record: dict[str, Any]) -> int:
        return max(
            [
                int(match.group(1))
                for source_id in list(record.get("source_message_ids") or [])
                for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
                if match
            ],
            default=-1,
        )
    scored = []
    for memory_id, items in groups.items():
        record = items[0][3]
        claim_count = sum(
            1 for item in items
            if str(item[6] or "").strip()
            and str(item[5] or "").strip() in str(item[3].get("source_text") or "")
        )
        scored.append((claim_count, source_turn(record), str(record.get("timestamp") or ""), memory_id))
    best = max(scored)
    # Do not collapse ordinary lists whose records each contain only one
    # member. A coherent source-local record is identifiable by two or more
    # typed claim values.
    return {best[3]} if best[0] >= 2 else set()


def _prefer_collection_coherent_items(
    items: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    semantic_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    if not items or str(semantic_spec.get("temporal_scope") or "").strip().lower() != "current":
        return items
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get("memory_id") or ""), []).append(item)
    if not groups:
        return items
    requested_tokens = _meaningful_field_tokens(
        str((semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or [""])[0])
    )

    def identity_overlap(group: list[dict[str, Any]]) -> int:
        record = records_by_id.get(str(group[0].get("memory_id") or ""), {})
        tags = dict(record.get("semantic_tags") or {})
        identity = dict(tags.get("event_identity") or {})
        identity_tokens = _meaningful_field_tokens(
            str(identity.get("entity_key") or identity.get("entity_surface_span") or "")
        )
        return len(requested_tokens & identity_tokens)

    # A collection can be distributed across several source turns. When at
    # least one candidate carries an event identity aligned with the request,
    # discard neighboring records from a different identity before scoring
    # chronology or completeness. This keeps a plan collection together while
    # excluding an adjacent symptom/logistics note without domain vocabulary.
    aligned_groups = [group for group in groups.values() if identity_overlap(group)]
    if aligned_groups:
        items = [item for group in aligned_groups for item in group]
        groups = {
            str(group[0].get("memory_id") or ""): group
            for group in aligned_groups
            if group
        }

    def score(group: list[dict[str, Any]]) -> tuple[int, int, int, str]:
        record = records_by_id.get(str(group[0].get("memory_id") or ""), {})
        turns = [
            int(match.group(1))
            for source_id in list(record.get("source_message_ids") or [])
            for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
            if match
        ]
        tags = dict(record.get("semantic_tags") or {})
        identity = dict(tags.get("event_identity") or {})
        identity_tokens = _meaningful_field_tokens(
            str(identity.get("entity_key") or identity.get("entity_surface_span") or "")
        )
        identity_overlap = len(requested_tokens & identity_tokens)
        direct_claims = sum(
            1
            for candidate in list(record.get("typed_attributes") or [])
            if isinstance(candidate, dict)
            and str(candidate.get("claim_span") or "").strip()
        )
        return identity_overlap, direct_claims, len(group), str(record.get("timestamp") or "")
    best_group = max(groups.values(), key=score)
    return best_group if len(best_group) >= 2 else items


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
