"""Minimal realization for graph-certified, source-grounded typed slots."""

from __future__ import annotations

from typing import Any

from gov_mem.data.schema import AnswerResult, RetrievedEvidence


def build_graph_authorized_projection(
    *, certificate: dict[str, Any], semantic_spec: dict[str, Any]
) -> RetrievedEvidence | None:
    """Make a synthetic evidence row containing only certified source spans."""
    if not certificate.get("authorized"):
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
    clauses = []
    source_ids = []
    typed_slots: dict[str, str | list[str]] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for payload in realizations:
        item = dict(payload or {})
        grouped.setdefault((str(item.get("attribute") or ""), str(item.get("source_atom_id") or "")), []).append(item)
    for _, group in grouped.items():
        attribute = str(group[0].get("attribute") or "").strip() if group else ""
        predicate_group = [
            item for item in group
            if str(item.get("slot_role") or "").strip().lower() != "claim_subject_value"
        ]
        # Subject spans describe what a claim is about. They are useful as a
        # requested value only when no predicate/value slot was certified for
        # that source record; otherwise they are record metadata and would
        # duplicate the answer with a misleading field label.
        if predicate_group and attribute not in collection_attributes:
            group = predicate_group
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
            if typed_slot_value and typed_slot_value.lower() in source_text.lower():
                value = typed_slot_value
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
                    f"{field_name.replace('_', ' ')} {value}"
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
