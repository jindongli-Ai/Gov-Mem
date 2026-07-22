"""Minimal realization for graph-certified, source-grounded typed slots."""

from __future__ import annotations

import re
import calendar
from typing import Any

from gov_mem.data.schema import AnswerResult, RetrievedEvidence
from gov_mem.governance_runtime.claim_adjudicator import _slot_matches_attribute


def build_graph_authorized_projection(
    *, certificate: dict[str, Any], semantic_spec: dict[str, Any]
) -> RetrievedEvidence | None:
    """Make a synthetic evidence row containing only certified source spans."""
    if not certificate.get("authorized"):
        return None
    if not graph_certificate_typed_compatibility(
        certificate=certificate,
        semantic_spec=semantic_spec,
    ):
        return None
    requested = _requested_slots(semantic_spec)
    certified = dict(certificate.get("slots") or {})
    # A mixed request can have an explicitly authorized subset and separately
    # denied attributes.  Keep the certified subset for Stage 3 rather than
    # discarding it because a different requested attribute is redacted.
    requested = [slot for slot in requested if slot in certified]
    if not requested:
        return None
    slots: dict[str, str | list[str]] = {}
    typed_candidates: list[dict[str, str]] = []
    source_ids: list[str] = []
    for slot in requested:
        item = dict(certified[slot] or {})
        realizations = [
            dict(realization or {})
            for realization in list(certificate.get("realizations") or [])
            if str((realization or {}).get("attribute") or "") == slot
        ] or [item]
        values = []
        for realization in realizations:
            value = str(realization.get("value") or "").strip()
            source_text = str(realization.get("source_text") or "")
            source_id = str(realization.get("source_memory_id") or realization.get("source_atom_id") or "")
            if not value or value.lower() not in source_text.lower():
                return None
            if value not in values:
                values.append(value)
                typed_candidates.append({
                    "attribute": slot,
                    "slot_name": str(realization.get("slot_name") or slot),
                    "value": value,
                    "source_text": source_text,
                    "source_memory_id": source_id,
                })
            if source_id:
                source_ids.append(source_id)
        if not values:
            return None
        slots[slot] = values[0] if len(values) == 1 else values
    content = "; ".join(
        f"{slot}: " + ("; ".join(value) if isinstance(value, list) else value)
        for slot, value in slots.items()
    )
    return RetrievedEvidence(
        memory_id="graph_authorized_typed_slot_projection",
        content=content,
        score=1.0,
        retrieval_source="graph_authorized_typed_slot_projection",
        reason="explicit graph allow with verbatim source validation",
        source_message_ids=list(dict.fromkeys(source_ids)),
        metadata={
            "slots": slots,
            "surface_spans": slots,
            "typed_candidates": typed_candidates,
            "lifecycle_status": "active",
            "temporal_scope": str(certificate.get("temporal_scope") or "unspecified"),
            "graph_certificate": certificate,
            "redacted_requested_attributes": [
                slot for slot in _requested_slots(semantic_spec) if slot not in certified
            ],
        },
    )


def graph_certificate_typed_compatibility(
    *, certificate: dict[str, Any], semantic_spec: dict[str, Any] | None
) -> bool:
    """Validate every certified attribute/slot pair against the typed contract."""
    if not certificate.get("authorized"):
        return False
    spec = dict(semantic_spec or {})
    alignment = dict((certificate.get("semantic_alignment") or {}).get("bindings") or {})
    bindings = [dict(item) for item in list(spec.get("attribute_bindings") or []) if isinstance(item, dict)]
    by_attribute = {
        str(item.get("attribute") or item.get("need_id") or "").strip(): item
        for item in bindings
        if str(item.get("attribute") or item.get("need_id") or "").strip()
    }
    for attribute, binding in alignment.items():
        attribute = str(attribute).strip()
        if not attribute:
            continue
        aligned = dict(binding or {})
        slot_names = [
            str(value).strip()
            for value in list(aligned.get("slot_names") or [])
            if str(value).strip()
        ]
        if str(aligned.get("slot_name") or "").strip():
            slot_names.append(str(aligned.get("slot_name") or "").strip())
        contract_binding = dict(by_attribute.get(attribute) or {"attribute": attribute})
        # The graph alignment is the source-local typed evidence link. Make
        # its concrete slot visible to the common validator without changing
        # the user's semantic contract or adding a domain vocabulary.
        if slot_names:
            contract_binding["evidence_slot_hint"] = slot_names[0]
        by_attribute[attribute] = contract_binding
    if by_attribute:
        spec["attribute_bindings"] = list(by_attribute.values())

    certified = dict(certificate.get("slots") or {})
    realizations = list(certificate.get("realizations") or [])
    collection_attributes = {
        str(item.get("attribute") or item.get("need_id") or "").strip()
        for key in ("attribute_bindings", "certifiable_needs")
        for item in list(spec.get(key) or [])
        if isinstance(item, dict)
        and str(item.get("attribute") or item.get("need_id") or "").strip()
        and str(item.get("need_kind") or item.get("binding_kind") or "").strip().lower()
        in {"record_collection", "collection", "list"}
    }
    for attribute, payload in certified.items():
        attribute = str(attribute).strip()
        if not attribute:
            return False
        rows = [
            dict(item or {})
            for item in realizations
            if str((item or {}).get("attribute") or "").strip() == attribute
        ] or [dict(payload or {})]
        if not rows:
            return False
        for row in rows:
            slot_name = str(row.get("slot_name") or attribute).strip()
            value = str(row.get("typed_slot_value") or row.get("value") or "").strip()
            source_text = str(row.get("source_text") or "")
            if not slot_name or not value or value.lower() not in source_text.lower():
                return False
            record = {
                "source_text": source_text,
                "slots": {slot_name: value},
                "stage2_served_attributes": [attribute],
                "stage2_typed_fields": [{
                    "attribute": attribute,
                    "slot_name": slot_name,
                    "value": value,
                }],
            }
            # A record collection intentionally contains heterogeneous typed
            # fields. Its concrete members are already checked as individual
            # certified scalar slots; applying the outer label to every
            # member would reject valid time/access/context roles.
            if attribute not in collection_attributes and not _slot_matches_attribute(
                attribute, slot_name, spec, record
            ):
                return False
    return True


def render_graph_authorized_slots(
    *, certificate: dict[str, Any], action: str, semantic_spec: dict[str, Any] | None = None
) -> AnswerResult:
    """Render values only; labels are derived from schema keys, never evidence prose."""
    certified = dict(certificate.get("slots") or {})
    realizations = list(certificate.get("realizations") or [])
    if not realizations:
        realizations = [{"attribute": slot, **dict(payload or {})} for slot, payload in certified.items()]
    collection_attributes = {
        str(item.get("attribute") or item.get("need_id") or "").strip()
        for key in ("attribute_bindings", "certifiable_needs")
        for item in list((semantic_spec or {}).get(key) or [])
        if isinstance(item, dict)
        and str(item.get("attribute") or item.get("need_id") or "").strip()
        and str(item.get("need_kind") or item.get("binding_kind") or "").strip()
        in {"record_collection", "collection", "list"}
    }
    # Graph certification can recover a composite record even when the
    # planner's surface binding omitted ``need_kind``.  Reuse that typed
    # contract instead of silently collapsing a multi-field plan to one
    # scalar winner.
    alignment = dict((certificate.get("semantic_alignment") or {}).get("bindings") or {})
    collection_attributes.update(
        str(attribute).strip()
        for attribute, binding in alignment.items()
        if isinstance(binding, dict)
        and str(binding.get("binding_kind") or "").strip().lower()
        in {"record_collection", "collection", "list"}
    )
    # Preserve the generic compatibility contract used by legacy synthetic
    # certificates when no semantic spec is attached.
    collection_attributes.update(
        attribute
        for attribute in {"record_collection"}
        if attribute in {str(item.get("attribute") or "").strip() for item in realizations}
    )
    requested_attributes = set(_requested_slots(semantic_spec or {}))
    scalar_members = requested_attributes - collection_attributes
    suppressed_collection_attributes = (
        collection_attributes & requested_attributes if scalar_members else set()
    )
    if not semantic_spec:
        # Legacy synthetic certificates encode record bundles directly.  A
        # multi-field source record is a collection contract in that form;
        # real requests use the explicit semantic alignment above.
        slots_by_attribute_source: dict[tuple[str, str], set[str]] = {}
        for item in realizations:
            key = (
                str(item.get("attribute") or "").strip(),
                str(item.get("source_atom_id") or "").strip(),
            )
            slot_name = str(item.get("slot_name") or "").strip()
            if key[0] and slot_name:
                slots_by_attribute_source.setdefault(key, set()).add(slot_name)
        collection_attributes.update(
            attribute for attribute, _source in slots_by_attribute_source
            if len(slots_by_attribute_source[(attribute, _source)]) > 1
        )
    clauses = []
    source_ids = []
    typed_slots: dict[str, str | list[str]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for payload in realizations:
        item = dict(payload or {})
        grouped.setdefault((str(item.get("attribute") or ""), str(item.get("source_atom_id") or "")), []).append(item)
    scalar_winners: dict[str, dict[str, Any]] = {}
    for group in grouped.values():
        if not group:
            continue
        attribute = str(group[0].get("attribute") or "").strip()
        if not attribute or attribute in collection_attributes:
            continue
        for item in group:
            slot_name = str(item.get("slot_name") or "").strip().lower()
            attr_tokens = set(attribute.lower().replace("-", "_").split("_"))
            slot_tokens = set(slot_name.replace("-", "_").split("_"))
            overlap = len((attr_tokens - {"current", "active", "latest"}) & slot_tokens)
            score = (
                int(slot_name == attribute.lower()),
                overlap,
                int(str(item.get("slot_role") or "").strip().lower() != "claim_subject_value"),
            )
            prior = scalar_winners.get(attribute)
            if prior is None or score > prior["_renderer_score"]:
                scalar_winners[attribute] = {**item, "_renderer_score": score}
    for _, group in grouped.items():
        attribute = str(group[0].get("attribute") or "").strip() if group else ""
        if attribute in suppressed_collection_attributes:
            continue
        if attribute not in collection_attributes and attribute in scalar_winners:
            winner = scalar_winners[attribute]
            winner_key = (
                str(winner.get("slot_name") or ""),
                str(winner.get("value") or ""),
                str(winner.get("source_atom_id") or ""),
            )
            if not any(
                (
                    str(item.get("slot_name") or ""),
                    str(item.get("value") or ""),
                    str(item.get("source_atom_id") or ""),
                ) == winner_key
                for item in group
            ):
                continue
            group = [scalar_winners[attribute]]
        predicate_group = [
            item for item in group
            if str(item.get("slot_role") or "").strip().lower() != "claim_subject_value"
        ]
        # Subject spans describe what a claim is about. They are useful as a
        # requested value only when no predicate/value slot was certified for
        # that source record; otherwise they are record metadata and would
        # duplicate the answer with a misleading field label.
        if predicate_group:
            group = predicate_group
        # A record-level collection can expose both complete claim values and
        # extractor helper slots (date, time, procedure, etc.).  Once multiple
        # claim values are certified, they are the answer members; retaining
        # the helper slots would split one source claim into redundant fields.
        collection_claim_values = [
            item for item in group
            if str(item.get("slot_role") or "").strip().lower() == "claim_value"
        ]
        if attribute in collection_attributes and len(collection_claim_values) >= 2:
            group = collection_claim_values
        attribute = str(group[0].get("attribute") or "").strip() if group else attribute
        source_text = str(group[0].get("source_text") or "").strip()
        group_values = [str(item.get("value") or "").strip() for item in group]
        rendered_values: list[str] = []
        rendered_fields: list[tuple[str, str]] = []
        slot = ""
        for item in group:
            slot = str(item.get("attribute") or "").strip()
            if not slot:
                continue
            field_name = str(item.get("slot_name") or "").strip()
            value = str(item.get("value") or "").strip()
            typed_slot_value = str(item.get("typed_slot_value") or "").strip()
            if not value:
                continue
            claim_span = str(item.get("claim_span") or "").strip()
            source_text = str(item.get("source_text") or "").strip()
            slot_role = str(item.get("slot_role") or "").strip().lower()
            # Stage 2 may preserve a complete claim on the realization while
            # the graph slot still carries the exact typed component. Keep
            # the component for field-level rendering; the claim remains
            # available as provenance and context.
            if (
                attribute in collection_attributes
                and slot_role == "claim_value"
                and claim_span
                and claim_span in source_text
                and value in claim_span
            ):
                value = claim_span
            elif typed_slot_value and typed_slot_value.lower() in source_text.lower():
                value = typed_slot_value
            if (
                not typed_slot_value
                and
                not bool(item.get("record_complete", True))
                and claim_span
                and claim_span in source_text
                and value in claim_span
                and len(value.split()) <= 2
                and len(claim_span.split()) <= 8
                and not any(
                    marker in claim_span.lower()
                    for marker in ("previous", "initial", "no longer", "former", "old", "from ")
                )
            ):
                # Extractor nodes can retain only a value suffix. A
                # source-local incomplete realization should keep the claim's
                # smallest complete span so qualifiers such as a shelf, route,
                # or object label are not silently dropped.
                value = claim_span
            if (
                not typed_slot_value
                and
                slot_role in {"claim_value", "claim_subject_value"}
                and claim_span
                and value in claim_span
                and claim_span in source_text
                and (
                    slot_role == "claim_value"
                    or attribute in collection_attributes
                )
            ):
                value = claim_span
            existing = typed_slots.get(slot)
            if existing is None:
                typed_slots[slot] = value
            elif isinstance(existing, list):
                if value not in existing:
                    existing.append(value)
            elif existing != value:
                typed_slots[slot] = [existing, value]
            else:
                continue
            if value not in rendered_values:
                rendered_values.append(value)
                if field_name:
                    rendered_fields.append((field_name, value))
            source_id = str(item.get("source_memory_id") or item.get("source_atom_id") or "")
            if source_id:
                source_ids.append(source_id)
        # A source record may contain superseded values or neighboring private
        # facts.  The certificate is field-level, so realization must remain a
        # field-level projection for both direct and redacted answers.
        if slot and rendered_values:
            # A collection attribute can be authorized through several
            # intrinsic fields of one record. Preserve that dynamic schema
            # association instead of flattening the values into an ambiguous
            # sequence. Both labels and values come from the certified graph.
            has_typed_roles = any(str(item.get("slot_role") or "").strip() for item in group)
            if len(rendered_fields) > 1 and (has_typed_roles or attribute == "record_collection"):
                projection = "; ".join(
                    f"{field_name.replace('_', ' ')}: {value}"
                    for field_name, value in rendered_fields
                )
            else:
                projection = "; ".join(rendered_values)
            if (
                not has_typed_roles
                and source_text
                and len(rendered_values) > 1
                and all(value in source_text for value in rendered_values)
            ):
                # Legacy synthetic certificates do not carry slot roles. In
                # that compatibility path, preserve the already supplied
                # source-local composite instead of splitting it into fields.
                projection = source_text.rstrip(".")
            if has_typed_roles or attribute == "record_collection":
                clauses.append(f"{str(slot).replace('_', ' ')}: {projection}")
            else:
                clauses.append(projection)
    if suppressed_collection_attributes:
        # Mixed requests still need a stable record anchor, but the outer
        # collection must not reintroduce its superseded operational fields.
        # Date-like slots are source-certified context, not domain-specific
        # answer vocabulary, and are emitted only when no scalar field already
        # covers them.
        context_fields = _outer_collection_context_fields(
            realizations=realizations,
            collection_attributes=suppressed_collection_attributes,
            requested_attributes=requested_attributes,
        )
        for field_name, value, source_id in context_fields:
            if field_name not in typed_slots:
                typed_slots[field_name] = value
                clauses.append(f"{field_name.replace('_', ' ')}: {value}")
            if source_id:
                source_ids.append(source_id)
    text = "; ".join(clauses) + ("." if clauses else "")
    return AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=list(dict.fromkeys(source_ids)),
        reasoning_summary="Minimal realization from explicit graph authorization and verbatim typed source spans.",
        action=action,
        answer_structured={"typed_slots": typed_slots, "realization_mode": "graph_authorized_typed_slots"},
    )


def _requested_slots(semantic_spec: dict[str, Any]) -> list[str]:
    certifiable_needs = semantic_spec.get("certifiable_needs")
    if isinstance(certifiable_needs, list):
        return list(dict.fromkeys(
            str(item.get("attribute") or item.get("need_id") or "").strip()
            for item in certifiable_needs
            if isinstance(item, dict)
            and str(item.get("attribute") or item.get("need_id") or "").strip()
        ))
    return list(dict.fromkeys(
        str(slot).strip()
        for slot in (
            semantic_spec.get("requested_attributes")
            or semantic_spec.get("requested_slots")
            or []
        )
        if str(slot).strip()
    ))


def _outer_collection_context_fields(
    *,
    realizations: list[dict[str, Any]],
    collection_attributes: set[str],
    requested_attributes: set[str],
) -> list[tuple[str, str, str]]:
    """Select one source-grounded temporal anchor per outer collection."""
    anchor_slots = {"date", "calendar_date", "event_date", "scheduled_date", "target_date"}
    scalar_source_texts = [
        str(item.get("source_text") or "")
        for item in realizations
        if str(item.get("attribute") or "").strip() not in collection_attributes
        and str(item.get("source_text") or "").strip()
    ]
    selected: dict[str, tuple[tuple[int, str, int], str, str]] = {}
    for item in realizations:
        attribute = str(item.get("attribute") or "").strip()
        slot_name = str(item.get("slot_name") or "").strip()
        value = str(item.get("typed_slot_value") or item.get("value") or "").strip()
        source_text = str(item.get("source_text") or "")
        normalized_slot = slot_name.lower().replace("-", "_")
        slot_tokens = set(normalized_slot.split("_"))
        if (
            attribute not in collection_attributes
            or slot_name in requested_attributes
            or not (normalized_slot in anchor_slots or "date" in slot_tokens)
            or not value
            or value.lower() not in source_text.lower()
            or not _temporal_anchor_matches_scalar_context(value, scalar_source_texts)
            or str(item.get("slot_role") or "").strip().lower() == "claim_subject_value"
        ):
            continue
        source_ids = [str(source_id).strip() for source_id in list(item.get("source_message_ids") or [])]
        turn = max(
            [
                int(match.group(1))
                for source_id in source_ids
                for match in [re.fullmatch(r"t(\d+)", source_id)]
                if match
            ],
            default=-1,
        )
        rank = (len(value), str(item.get("timestamp") or ""), turn)
        prior = selected.get(slot_name)
        if prior is None or rank > prior[0]:
            selected[slot_name] = (rank, value, str(item.get("source_memory_id") or item.get("source_atom_id") or ""))
    return [
        (slot_name, value, source_id)
        for slot_name, (_rank, value, source_id) in selected.items()
    ]


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
