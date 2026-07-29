"""LEGACY: batch claim/state adjudication over a closed Stage-2 evidence set.

The model resolves claim identity, current-state scope, and field selection.
It cannot expand the evidence set or grant access.  The deterministic layer
only accepts exact candidate IDs, typed values, active lifecycle records, and
verbatim source spans.
"""

from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.factual_claim_quality import factual_value_is_eligible
from gov_mem.governance_runtime.source_grounding import row_grounded_source_text
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.query_semantics import CURRENT_STATE_SLOT_ALIASES


_ACTIVE_BLOCKLIST = {"deleted", "superseded", "canceled", "historical", "retired"}
_DECISION_CLASSES = {"answer", "redact", "omit", "uncertain"}


def adjudicate_claims(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    llm_client: LLMClient | None,
    model_name: str,
    target_entities: list[str] | None = None,
    requester_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select requested fields from Stage-2 records in one closed-set call."""
    requested = _requested_attributes(semantic_spec)
    current_scope = str((semantic_spec or {}).get("temporal_scope") or "").lower() == "current"
    # Planner target_entities can contain the query's open request object (for
    # example, a noun that is also one of requested attributes). It is not an
    # entity-identity boundary. Keep genuine named entities for scope checks,
    # but do not reject a source record merely because its event identity uses
    # a more specific typed subject than that open request object.
    scope_target_entities = [
        str(entity).strip()
        for entity in list(target_entities or [])
        if str(entity).strip() and str(entity).strip() not in set(requested)
    ]
    candidates = _candidate_payload(evidence)
    strict_stage2 = any(
        str(record.get("stage2_binding_mode") or "").startswith("closed_set_candidate_fields_v")
        for record in candidates
    )
    if strict_stage2:
        # Stage 2 v2 is already the semantic decision boundary.  Calling a
        # second adjudicator here re-ranks the same records with a different
        # prompt and can resurrect an older status/date.  Carry only the
        # source-local Stage-2 bindings across; deduplicate structurally by
        # latest source turn, without another semantic choice.
        return _stage2_passthrough_decisions(
            candidates=candidates,
            requested=requested,
            semantic_spec=semantic_spec,
        )
    if not requested or not candidates or llm_client is None or not llm_client.is_available():
        return [], {
            "available": False,
            "reason": "claim_adjudication_unavailable",
            "requested_attributes": requested,
        }

    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are a claim and disclosure adjudicator for a governed memory system. "
                "Use only the closed candidate records. Resolve current-state claims, partial updates, "
                "supersession, and complementary fields. Do not answer the user, invent values, expand "
                "the candidate set, or infer a standing permission. Return JSON only."
            ),
            user_prompt=(
                "Return {\"decisions\":[{\"attribute\":string,\"memory_id\":string,"
                "\"slot_name\":string,\"value\":string,\"evidence_span\":string,"
                "\"decision\":\"answer|redact|omit|uncertain\",\"confidence\":number,"
                "\"reason\":string}]}. For every requested attribute, select the most direct current, "
                "source-grounded typed field when one exists. A later record may update only one field: "
                "preserve complementary fields from other current claims when they describe the same "
                "operational object. A claim that explicitly clears or replaces a field supersedes that "
                "field, but does not erase unrelated fields. Prefer an explicit authoritative update over "
                "a recap, and prefer a complete current summary when it directly covers the requested field. For a "
                "summary, snapshot, safe-summary, recap, or wording request, preserve explicit identity context from the query: "
                "when the question asks for a final/current state across multiple named objects, first locate the "
                "explicit final/current state record for each named object or lane, then retain their complementary "
                "typed fields together; do not let one verbose recap displace a shorter final state record for another "
                "named object. "
                "when a later abbreviated recap uses generic aliases or pronouns while another current source names "
                "the requested thread or object, prefer the explicitly identified source or preserve the smallest "
                "set of non-conflicting named context needed to make the summary unambiguous. Do not treat a generic "
                "phrase such as 'pet and parcel plans' as equivalent to named entities in the query when that identity "
                "is part of the requested utility. "
                "Use decision=redact only for a field that is present but should not be disclosed; use omit "
                "for unrelated or unsupported fields. Values and evidence_span must be copied exactly from "
                "the listed record. If a listed typed slot is a shortened extraction of a complete value "
                "in the same source claim, keep its slot_name as the anchor and copy the complete exact "
                "source substring as value; the complete value must contain the observed slot value. "
                "For a scalar requested property, select one current source claim. For a collection-bound requested "
                "property, multiple complementary fields may come from different current memory_id/claims when their "
                "typed slots are distinct and each is source-grounded; preserve all such fields and resolve any "
                "same-slot conflict in favor of the current canonical claim. Do not return a decision for an "
                "attribute without an exact listed slot or an exact source-grounded typed field from a "
                "Stage-2-served record; when the extractor omitted a conventional slot, assign a stable semantic "
                "slot_name and copy the exact value from source_text. "
                "For a list, snapshot, or summary request, preserve complementary current logistics fields from separate "
                "source claims when they describe different parts of the same operation, such as a physical "
                "handoff location, a fallback method, and an operational access condition. Do not collapse a "
                "supported field to uncertain merely because its canonical slot name is open or narrative. "
                "For a collection, make a coverage pass over every requested attribute before returning: if an "
                "admitted source record contains an exact claim/value for a complementary field, emit that field "
                "even when Stage-2 typed_fields names a different field from the same record. For a requested "
                "status, whether, presence, or absence property, preserve an explicit source-grounded state claim "
                "that answers it, including a clear removal/absence state; do not omit an answerable operational "
                "state merely because the same source also mentions a deleted or restricted artifact. "
                "For a record-collection attribute, the outer collection label is a contract name, not a value: "
                "do not return a slot whose name is merely the requested collection attribute when typed component "
                "slots are available. Resolve each component independently in current-state scope; do not carry "
                "an older opening or recap component forward when a later source claim updates that same typed slot. "
                "Never copy an access artifact or a compound field containing an access artifact into a non-access "
                "collection field; retain only the independently typed non-access component. "
                "For composite schedule records, a later update may replace only the interval while leaving a weekday/date "
                "anchor or physical handoff location useful. Preserve those non-conflicting typed components, but do not "
                "return the older interval as the current interval. "
                "Use the semantic attribute binding and evidence_slot_hint to map an open requested attribute to "
                "the closest source-grounded typed slot, even when the canonical slot name differs. Do not map a safe label or wording attribute to a date/time slot merely "
                "because the same source sentence contains a date. Do not map an amount attribute to a descriptive "
                "amount-type label when an exact numeric amount slot is listed. "
                "Each claim exposes subject_span and value_span as different typed roles. Use the subject span when the "
                "requested property is the object or entity being assigned or identified, and use the value span when the "
                "requested property is the object's status, condition, or relation. Do not substitute a status phrase for "
                "the object, or the object for a status phrase. A negated, prohibitive, scope-setting, or meta-level span "
                "such as a statement that something is not shareable, is not part of an operation, or does not change a "
                "field is not the factual value of that field; omit it for that attribute and select a positive typed claim "
                "when one exists. Treat the complete claim span as context, not as permission to copy policy wording into "
                "an unrelated requested value. "
                "Respect typed role compatibility: a status/state/condition marker is not the value of another "
                "property; temporal fields must not stand in for locations or destinations, and locations must not "
                "stand in for temporal fields. Access artifacts such as PINs, tokens, passwords, or credentials "
                "must not be mapped to a physical zone, area, route, or location unless the requested attribute "
                "itself is that access artifact. For a composite attribute, if several typed components occur in "
                "one source claim, emit the smallest exact source substring containing the complete supported "
                "property rather than returning only its first component. For a snapshot, summary, recap, or "
                "overview attribute, if one current source claim exposes several distinct typed components, emit "
                "each independently grounded component instead of retaining only the first one. "
                "For an outer snapshot, summary, or state attribute, perform the coverage pass over the entire "
                "closed, current utility source set before choosing a representative record. When several active "
                "records share the query's operational object or lane, retain complementary typed fields from each "
                "of those records, including a field whose source sentence does not repeat the query label. Do not "
                "stop at the most direct calibration, status, or anchor record when another closed record supplies "
                "a non-conflicting current component. This remains source-local: exclude records from unrelated "
                "objects, lanes, lifecycle states, or disclosure scopes, and never revive deleted or superseded values. "
                "The closed source context may establish the lane through neighboring turns, chronology, and current "
                "state updates, so do not require every qualifying record to repeat the object name or requester role. "
                "When two typed slots share a head word but one has an extra qualifier, preserve the slot whose semantic role matches the request; do not replace a bare requested window, rule, or time with a qualified sibling field unless that qualifier is also part of the request. "
                "This is a field-level recommendation, not a permission grant; later code still checks "
                "lifecycle, provenance, deletion, and graph constraints. When state_delta.changed_fields "
                "or a claim value_span gives the current value, treat it as canonical even if an older "
                "schema alias in slots contains a superseded value. Return the exact observed slot value, "
                "not the surrounding sentence.\n"
                "Records marked utility_source_closure=true were selected as factual source turns by the "
                "utility locator before Stage 2. They remain closed candidate evidence even if the record-centric "
                "reranker omitted them; adjudicate their complementary fields instead of discarding the whole "
                "record. In a logistics-only utility request, an explicitly operational access artifact in such "
                "a selected source is answerable when it directly serves the requested contingency, while unrelated "
                "credentials remain out of scope. "
                "The requester identity is provided separately for resolving first-person fields such as "
                "'my signoff window'; it is not a permission grant and must not be used to invent evidence. "
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Target entities: {list(target_entities or [])}\n"
                f"Requester identity: {requester_id or '(not provided)'}\n"
                f"Requested attributes: {requested}\nClosed candidate records: {candidates}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return [], {
            "available": False,
            "reason": f"claim_adjudication_error:{type(exc).__name__}",
            "requested_attributes": requested,
        }

    accepted: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    candidate_by_id = {str(row["memory_id"]): row for row in candidates}
    for item in _decision_rows(raw):
        attribute = str(item.get("attribute") or "").strip()
        memory_id = str(item.get("memory_id") or "").strip()
        slot_name = str(item.get("slot_name") or "").strip()
        proposed_value = str(item.get("value") or "").strip()
        evidence_span = str(item.get("evidence_span") or "").strip()
        decision = str(item.get("decision") or "").strip().lower()
        record = candidate_by_id.get(memory_id)
        # The model may emit a stale or hallucinated id.  Closed-set
        # adjudication must reject it before any record-field validation;
        # otherwise the lifecycle check below dereferences None and aborts
        # the whole checkpoint instead of treating this as one bad proposal.
        if record is None:
            reject("unknown_memory")
            continue
        # A closed source record can still be the wrong named lane. Apply the
        # same event-identity scope boundary used by deterministic fallback
        # before accepting the model's first-choice binding.
        if record is not None and _record_has_explicit_scope_conflict(
            record,
            question=question,
            target_entities=scope_target_entities,
            attribute=attribute,
        ):
            reject("explicit_scope_conflict")
            continue
        value = _coerce_observed_slot_value(
            record,
            slot_name,
            proposed_value,
        )
        value = _normalize_current_transition_value(
            attribute=attribute,
            value=value,
            source_text=str(record.get("source_text") or "") if record else "",
        )
        source_text = str(record.get("source_text") or "") if record else ""
        checks = (
            (attribute not in requested, "unknown_attribute"),
            (memory_id not in candidate_by_id, "unknown_memory"),
            (decision not in _DECISION_CLASSES, "invalid_decision"),
            (not slot_name, "missing_slot_name"),
            (not value, "unobserved_value"),
            (not evidence_span, "missing_evidence_span"),
            (not _slot_matches(record, slot_name, proposed_value, attribute), "slot_value_mismatch"),
            (not _slot_matches_attribute(attribute, slot_name, semantic_spec, record), "slot_attribute_mismatch"),
            (value not in source_text, "value_not_source_grounded"),
            (evidence_span not in source_text, "evidence_span_not_source_grounded"),
            (value not in evidence_span, "value_not_in_evidence_span"),
            (not factual_value_is_eligible(
                attribute=attribute,
                slot_name=slot_name,
                value=value,
                semantic_spec=semantic_spec,
                source_text=source_text,
            ), "ineligible_factual_value"),
            (str(record.get("lifecycle_status") or "active").lower() in _ACTIVE_BLOCKLIST, "blocked_lifecycle"),
        )
        failed = next((reason for condition, reason in checks if condition), None)
        if failed:
            reject(failed)
            continue
        try:
            confidence = min(max(float(item.get("confidence") or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.0
        accepted.append({
            "attribute": attribute,
            "memory_id": memory_id,
            "source_message_ids": list(record.get("source_message_ids") or []) if record else [],
            "slot_name": slot_name,
            "value": value,
            "evidence_span": evidence_span,
            "claim_span": _complete_claim_span_for_value(record, proposed_value) if record else "",
            "decision": decision,
            "confidence": confidence,
            "reason": str(item.get("reason") or "").strip(),
        })

    strict_stage2 = any(
        str(record.get("stage2_binding_mode") or "").startswith("closed_set_candidate_fields_v")
        for record in candidates
    )
    if strict_stage2:
        # Stage-2 v1 is already the semantic closed-set binding boundary.
        # Do not reopen unresolved attributes through the legacy deterministic
        # fallback/chronology cascade; that cascade can replace a valid date,
        # amount, or status with a neighboring source field. Keep only the
        # model-adjudicated, source-grounded decisions above and apply only
        # structural deduplication here.
        chosen: list[dict[str, Any]] = []
        grouped_strict: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in accepted:
            if item.get("decision") not in {"answer", "redact"}:
                continue
            key = (
                str(item.get("attribute") or ""),
                str(item.get("memory_id") or ""),
                str(item.get("slot_name") or ""),
            )
            prior = grouped_strict.get(key)
            if prior is None or float(item.get("confidence") or 0.0) > float(prior.get("confidence") or 0.0):
                grouped_strict[key] = item
        scalar_best: dict[str, dict[str, Any]] = {}
        for item in grouped_strict.values():
            attribute = str(item.get("attribute") or "")
            if _is_collection_attribute(attribute, semantic_spec):
                chosen.append(item)
                continue
            prior = scalar_best.get(attribute)
            if prior is None or float(item.get("confidence") or 0.0) > float(prior.get("confidence") or 0.0):
                scalar_best[attribute] = item
        chosen.extend(scalar_best.values())
        return chosen, {
            "available": True,
            "reason": "closed_set_stage2_passthrough",
            "requested_attributes": requested,
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "selected_count": len(chosen),
            "accepted_decisions": accepted,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "legacy_fallbacks_disabled": True,
        }

    # Stage 2 has already made a closed-set, source-span-validated relevance
    # decision. If the batch adjudicator under-covers a requested attribute,
    # recover that attribute from the same admitted record rather than
    # silently dropping a field from a multi-part utility answer. This keeps
    # the recovery evidence-local and typed; it does not create new evidence
    # or broaden authorization.
    # `uncertain` is a model-level abstention, not evidence coverage. Let the
    # closed-set fallback recover a source-grounded typed field when Stage 2
    # already admitted that attribute.
    accepted_attributes = {
        str(item.get("attribute") or "")
        for item in accepted
        if item.get("decision") in {"answer", "redact"}
    }
    collection_attributes = {
        attribute for attribute in requested
        if _is_collection_attribute(attribute, semantic_spec)
    }
    for attribute in requested:
        if current_scope and attribute not in collection_attributes:
            # The model may return only one of several valid source claims for
            # a scalar field. Add the latest closed-set typed claim so the
            # chronology resolver can compare it with the model choice.
            latest = _stage2_attribute_fallback(
                attribute,
                candidates,
                semantic_spec,
                question=question,
                target_entities=scope_target_entities,
                requester_id=requester_id,
            )
            if latest is not None and (
                attribute not in accepted_attributes
                or _fallback_is_authoritative_update(
                    latest,
                    candidate_by_id,
                    attribute=attribute,
                    semantic_spec=semantic_spec,
                )
                or _record_has_current_attribute_update(
                    candidate_by_id.get(str(latest.get("memory_id") or ""), {}),
                    attribute=attribute,
                )
            ):
                latest_key = (
                    str(latest.get("memory_id") or ""),
                    str(latest.get("slot_name") or ""),
                    str(latest.get("value") or ""),
                )
                if not any(
                    (
                        str(item.get("memory_id") or ""),
                        str(item.get("slot_name") or ""),
                        str(item.get("value") or ""),
                    ) == latest_key
                    for item in accepted
                    if str(item.get("attribute") or "") == attribute
                ):
                    accepted.append(latest)
                    accepted_attributes.add(attribute)
        if attribute in accepted_attributes and not _is_collection_attribute(attribute, semantic_spec):
            continue
        if _is_collection_attribute(attribute, semantic_spec):
            existing_keys = {
                (str(item.get("memory_id") or ""), str(item.get("slot_name") or ""), str(item.get("value") or ""))
                for item in accepted
                if str(item.get("attribute") or "") == attribute
            }
            for fallback in _stage2_collection_fallbacks(
                attribute,
                candidates,
                semantic_spec,
                    question=question,
                    target_entities=scope_target_entities,
                excluded_memory_ids={
                    str(item.get("memory_id") or "")
                    for item in accepted
                    if str(item.get("attribute") or "") == attribute
                    and str(item.get("decision") or "") in {"answer", "redact"}
                },
            ):
                key = (
                    str(fallback.get("memory_id") or ""),
                    str(fallback.get("slot_name") or ""),
                    str(fallback.get("value") or ""),
                )
                if key not in existing_keys:
                    accepted.append(fallback)
                    existing_keys.add(key)
            if any(str(item.get("attribute") or "") == attribute for item in accepted):
                accepted_attributes.add(attribute)
            # A collection contract is complete only from collection members;
            # never reinterpret its outer label as a scalar field when no
            # typed member was found.
            continue
        fallback = _stage2_attribute_fallback(
            attribute,
            candidates,
            semantic_spec,
            question=question,
            target_entities=scope_target_entities,
        )
        if fallback is not None:
            accepted.append(fallback)
            accepted_attributes.add(attribute)

    # Remove cross-attribute scalar collisions before chronology selects a
    # winner. A generic slot such as ``visit_window`` can be proposed for two
    # requested attributes; keep it for the attribute that has no more
    # specific typed alternative, while preserving the alternative claim.
    scalar_slots_by_attribute = {
        other_attribute: _expected_slot_names(other_attribute, semantic_spec)
        for other_attribute in _requested_attributes(semantic_spec or {})
        if not _is_collection_attribute(other_attribute, semantic_spec or {})
    }
    normalized_accepted: list[dict[str, Any]] = []
    for item in accepted:
        attribute = str(item.get("attribute") or "")
        if attribute in scalar_slots_by_attribute:
            slot_name = str(item.get("slot_name") or "")
            reserved_by_other = any(
                other_attribute != attribute and slot_name in other_slots
                for other_attribute, other_slots in scalar_slots_by_attribute.items()
            )
            has_alternative = any(
                str(candidate.get("attribute") or "") == attribute
                and str(candidate.get("slot_name") or "") != slot_name
                and candidate.get("decision") in {"answer", "redact"}
                for candidate in accepted
            )
            if reserved_by_other and has_alternative:
                continue
        normalized_accepted.append(item)
    accepted = normalized_accepted

    # A collection/plan binding must not re-admit a temporal or scalar field
    # already covered by a dedicated requested attribute. This handles open
    # planner labels such as ``setup_schedule`` without naming any dataset
    # entity or source slot.
    scalar_attributes_present = {
        str(item.get("attribute") or "")
        for item in accepted
        if str(item.get("attribute") or "") in requested
        and not _is_collection_attribute(str(item.get("attribute") or ""), semantic_spec)
        and item.get("decision") in {"answer", "redact"}
    }
    filtered_collection: list[dict[str, Any]] = []
    for item in accepted:
        attribute = str(item.get("attribute") or "")
        if attribute not in collection_attributes:
            filtered_collection.append(item)
            continue
        slot_tokens = _meaningful_field_tokens(str(item.get("slot_name") or ""))
        collides_with_scalar = any(
            slot_tokens & _meaningful_field_tokens(other)
            for other in scalar_attributes_present
            if other != attribute
        )
        if not collides_with_scalar:
            filtered_collection.append(item)
    accepted = filtered_collection

    # Scalar fields never combine competing source claims. Collection fields
    # retain distinct source-grounded slots across the closed record set, while
    # still collapsing duplicate slots to one adjudicated value.
    chosen: list[dict[str, Any]] = []
    best_groups: dict[str, list[dict[str, Any]]] = {}
    best_scores: dict[str, float] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in accepted:
        if item["decision"] in {"answer", "redact"}:
            grouped.setdefault((item["attribute"], item["memory_id"]), []).append(item)
    # A record-collection attribute can use the same open-schema slot label
    # across several source records.  Keep those source-local claims separate
    # so a later summary cannot erase complementary fields from an earlier
    # current record merely because both are mapped to the outer attribute.
    collection_by_slot: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (attribute, _memory_id), items in grouped.items():
        by_slot: dict[str, dict[str, Any]] = {}
        for item in items:
            slot = str(item.get("slot_name") or "")
            prior = by_slot.get(slot)
            if prior is None or (
                len(str(item.get("value") or "")) > len(str(prior.get("value") or ""))
                or (
                    len(str(item.get("value") or "")) == len(str(prior.get("value") or ""))
                    and float(item.get("confidence") or 0.0) > float(prior.get("confidence") or 0.0)
                )
            ):
                by_slot[slot] = item
        items = list(by_slot.values())
        if attribute in collection_attributes:
            for item in items:
                key = (
                    attribute,
                    str(item.get("memory_id") or ""),
                    str(item.get("slot_name") or ""),
                )
                prior = collection_by_slot.get(key)
                if prior is None or (
                    float(item.get("confidence") or 0.0) > float(prior.get("confidence") or 0.0)
                    or (
                        float(item.get("confidence") or 0.0) == float(prior.get("confidence") or 0.0)
                        and len(str(item.get("value") or "")) > len(str(prior.get("value") or ""))
                    )
                ):
                    collection_by_slot[key] = item
            continue
        score = sum(float(item["confidence"]) for item in items) + 0.01 * len(items)
        if score > best_scores.get(attribute, -1.0):
            best_scores[attribute] = score
            best_groups[attribute] = items
    if current_scope:
        # For current-state scalars, source chronology is the primary
        # conflict resolver. The fallback rows have already passed exact
        # source grounding, typed-role validation, and query-scope filtering;
        # their zero model confidence means only that the LLM did not repeat
        # the field, not that the source claim is less current.
        for attribute in requested:
            if attribute in collection_attributes:
                continue
            groups = [
                items
                for (group_attribute, _memory_id), items in grouped.items()
                if group_attribute == attribute
            ]
            # The first LLM response is a semantic proposal, not the
            # chronology oracle. Always add the newest exact Stage-2 field to
            # the closed set before resolving current-state conflicts. This
            # covers cases where the model repeats an older claim while a
            # later update was retrieved but omitted from its JSON response.
            latest = _stage2_attribute_fallback(
                attribute,
                candidates,
                semantic_spec,
                question=question,
                target_entities=scope_target_entities,
                requester_id=requester_id,
            )
            if latest is not None and (
                not any(
                    str(item.get("attribute") or "") == attribute
                    and item.get("decision") in {"answer", "redact"}
                    for item in accepted
                )
                or _fallback_is_authoritative_update(
                    latest,
                    candidate_by_id,
                    attribute=attribute,
                    semantic_spec=semantic_spec,
                )
                or _record_has_current_attribute_update(
                    candidate_by_id.get(str(latest.get("memory_id") or ""), {}),
                    attribute=attribute,
                )
            ):
                latest_key = (
                    str(latest.get("memory_id") or ""),
                    str(latest.get("slot_name") or ""),
                    str(latest.get("value") or ""),
                )
                if not any(
                    (
                        str(item.get("memory_id") or ""),
                        str(item.get("slot_name") or ""),
                        str(item.get("value") or ""),
                    ) == latest_key
                    for item in accepted
                    if str(item.get("attribute") or "") == attribute
                ):
                    accepted.append(latest)
                    grouped.setdefault((attribute, latest_key[0]), []).append(latest)
                    groups = [
                        items
                        for (group_attribute, _memory_id), items in grouped.items()
                        if group_attribute == attribute
                    ]
            if not groups:
                continue
            # A current recap may intentionally abbreviate entity names while
            # an earlier still-active safe wording carries the explicit
            # identity needed by the query. For summary-like attributes,
            # preserve the most complete grounded realization first, then use
            # chronology among equally complete claims. Ordinary scalar facts
            # remain latest-first.
            summary_like = bool(
                _meaningful_field_tokens(attribute)
                & {"summary", "overview", "recap", "wording"}
            )
            latest_group = max(
                groups,
                key=lambda items: _current_group_rank(
                    items,
                    candidate_by_id,
                    prefer_complete=summary_like,
                ),
            )
            best_groups[attribute] = latest_group
    for items in best_groups.values():
        chosen.extend(items)
    collection_items = list(collection_by_slot.values())
    if current_scope:
        collection_items = _resolve_current_collection_components(
            collection_items,
            candidate_by_id=candidate_by_id,
        )
    chosen.extend(collection_items)
    return chosen, {
        "available": True,
        "reason": "closed_set_claim_adjudication_complete",
        "requested_attributes": requested,
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "selected_count": len(chosen),
        "accepted_decisions": accepted,
        "rejection_counts": dict(sorted(rejection_counts.items())),
    }


def build_canonical_field_map(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse duplicate scalar proposals into one canonical field proposal.

    Stage-2 may retain low-confidence coverage fallbacks for diagnostics and
    collection completion. They are not allowed to overwrite a stronger
    source-grounded proposal when the field map crosses into final alignment.
    The map is a handoff contract; it is not a new evidence selector.
    """
    best: dict[str, tuple[tuple[int, float, int, int], dict[str, Any]]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            continue
        attribute = str(item.get("attribute") or "").strip()
        value = str(item.get("value") or "").strip()
        if not attribute or not value:
            continue
        reason = str(item.get("reason") or "").strip()
        try:
            confidence = min(max(float(item.get("confidence") or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.0
        # A coverage fallback remains visible in the full adjudication trace,
        # but must lose to an explicit claim for the same attribute.
        explicit = int(reason != "stage2_closed_set_attribute_coverage_fallback")
        rank = (
            explicit,
            confidence,
            int(bool(str(item.get("claim_span") or "").strip())),
            len(value),
        )
        prior = best.get(attribute)
        if prior is None or rank > prior[0]:
            best[attribute] = (rank, dict(item))
    return {attribute: item for attribute, (_rank, item) in best.items()}


def build_adjudicated_projection(
    *, evidence: list[RetrievedEvidence], decisions: list[dict[str, Any]]
) -> list[RetrievedEvidence]:
    """Build field-minimal evidence rows for the final typed realization."""
    by_id = {str(row.memory_id): row for row in evidence}
    projected: list[RetrievedEvidence] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in decisions:
        if str(item.get("decision") or "") in {"answer", "redact"}:
            grouped.setdefault(str(item.get("memory_id") or ""), []).append(item)
    for memory_id, items in grouped.items():
        source = by_id.get(memory_id)
        if source is None:
            continue
        valid_items = [
            item for item in items
            if str(item.get("attribute") or "").strip()
            and str(item.get("value") or "").strip()
            and str(item.get("slot_name") or "").strip()
            and str(item.get("evidence_span") or "").strip()
        ]
        if not valid_items:
            continue
        metadata = dict(source.metadata or {})
        # The source row can carry a richer Stage-2 annotation than the
        # adjudicator selected.  Once this projection becomes final evidence,
        # retaining those unselected fields lets the realization audit choose
        # a stale or neighboring value again.  Keep the record-level decision
        # metadata, but make its typed surface exactly the adjudicated field
        # set; the original source text remains available for provenance.
        stage2_decision = dict(metadata.get("stage2_semantic_rerank") or {})
        selected_attributes = {
            str(item.get("attribute") or "").strip()
            for item in valid_items
            if str(item.get("attribute") or "").strip()
        }
        selected_typed_fields = [
            {
                "attribute": str(item.get("attribute") or "").strip(),
                "slot_name": str(item.get("slot_name") or "").strip(),
                "value": str(item.get("value") or "").strip(),
            }
            for item in valid_items
        ]
        stage2_decision["typed_fields"] = selected_typed_fields
        stage2_decision["stage2_typed_fields"] = selected_typed_fields
        stage2_decision["served_attributes"] = sorted(selected_attributes)
        metadata["stage2_semantic_rerank"] = stage2_decision
        slots = {
            str(item.get("slot_name") or "").strip(): str(item.get("value") or "").strip()
            for item in valid_items
        }
        typed_candidates = [
            {
                "attribute": str(item.get("attribute") or "").strip(),
                "slot_name": str(item.get("slot_name") or "").strip(),
                "value": str(item.get("value") or "").strip(),
                "claim_span": str(item.get("claim_span") or "").strip(),
                "source_text": row_grounded_source_text(source),
                "source_memory_id": str(source.memory_id),
            }
            for item in valid_items
        ]
        # Downstream realization consumes semantic claims before ordinary
        # slots. Replace that exposed claim surface with the adjudicated
        # fields as well; otherwise an older composite claim can re-enter the
        # answer after this projection has already narrowed the evidence.
        source_tags = dict(metadata.get("semantic_tags") or {})
        semantic_tags = {
            **source_tags,
            "claims": [
                dict(claim)
                for claim in list(source_tags.get("claims") or [])
                if isinstance(claim, dict)
            ],
            "attributes": dict(source_tags.get("attributes") or {}),
            "surface_values": dict(source_tags.get("surface_values") or {}),
        }
        selected_claims = []
        selected_attributes_map: dict[str, str] = {}
        for item in valid_items:
            value = str(item.get("value") or "").strip()
            evidence_span = str(item.get("evidence_span") or "").strip()
            claim_span = str(item.get("claim_span") or "").strip()
            if not claim_span or claim_span not in row_grounded_source_text(source) or value not in claim_span:
                claim_span = evidence_span
            selected_claims.append({
                "property_label": str(item.get("slot_name") or "").strip(),
                "value_span": value,
                "claim_span": claim_span,
                "subject_span": str(
                    item.get("subject_span")
                    or item.get("attribute")
                    or ""
                ).strip(),
            })
            selected_attributes_map[str(item.get("slot_name") or "").strip()] = value
        semantic_tags["claims"] = selected_claims
        semantic_tags["attributes"] = selected_attributes_map
        semantic_tags["surface_values"] = dict(selected_attributes_map)
        metadata["semantic_tags"] = semantic_tags
        metadata.update({
            "slots": slots,
            "surface_spans": slots,
            "typed_candidates": typed_candidates,
            "claim_adjudication": [dict(item) for item in valid_items],
            "lifecycle_status": "active",
        })
        projected.append(RetrievedEvidence(
            memory_id=str(source.memory_id),
            content=str(source.content or ""),
            score=float(source.score),
            retrieval_source="closed_set_claim_adjudication",
            reason="field-minimal claim adjudication projection",
            user_id=source.user_id,
            memory_type=source.memory_type,
            scope=source.scope,
            entities=list(source.entities or []),
            time=source.time,
            source_message_ids=list(source.source_message_ids or []),
            metadata=metadata,
        ))
    return projected


def _stage2_passthrough_decisions(
    *,
    candidates: list[dict[str, Any]],
    requested: list[str],
    semantic_spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project Stage-2 bindings without reopening semantic reranking."""
    chosen_by_attribute: dict[str, dict[str, Any]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}

    def source_order(record: dict[str, Any]) -> int:
        return max(
            (
                int(match.group(1))
                for source_id in list(record.get("source_message_ids") or [])
                for match in [re.search(r"(?:^|_)t(\d+)(?:$|_)", str(source_id), re.IGNORECASE)]
                if match
            ),
            default=-1,
        )

    for record in candidates:
        if str(record.get("lifecycle_status") or "active").lower() in _ACTIVE_BLOCKLIST:
            continue
        memory_id = str(record.get("memory_id") or "")
        source_text = str(record.get("source_text") or "")
        stage2_span = str(record.get("stage2_support_span") or "").strip()
        if not memory_id or not source_text:
            continue
        for field in list(record.get("stage2_typed_fields") or []):
            if not isinstance(field, dict):
                continue
            attribute = str(field.get("attribute") or "").strip()
            slot_name = str(field.get("slot_name") or "").strip()
            value = str(field.get("value") or "").strip()
            if attribute not in requested or not slot_name or not value or value not in source_text:
                continue
            evidence_span = stage2_span if stage2_span and stage2_span in source_text else value
            proposal = {
                "attribute": attribute,
                "memory_id": memory_id,
                "source_message_ids": list(record.get("source_message_ids") or []),
                "slot_name": slot_name,
                "value": value,
                "evidence_span": evidence_span,
                "decision": "answer",
                "confidence": 1.0,
                "reason": "stage2_attributewise_binding_passthrough",
            }
            prior = chosen_by_attribute.get(attribute)
            if prior is None or source_order(record) > source_order(rows_by_id.get(str(prior.get("memory_id") or ""), {})):
                chosen_by_attribute[attribute] = proposal
                rows_by_id[memory_id] = record

    chosen = list(chosen_by_attribute.values())
    selected_ids = {str(item.get("memory_id") or "") for item in chosen}
    return chosen, {
        "available": True,
        "reason": "closed_set_stage2_passthrough",
        "requested_attributes": requested,
        "candidate_count": len(candidates),
        "accepted_count": len(chosen),
        "selected_count": len(chosen),
        "accepted_decisions": chosen,
        "rejection_counts": {},
        "legacy_fallbacks_disabled": True,
        "selected_memory_ids": sorted(selected_ids),
    }


def _requested_attributes(semantic_spec: dict[str, Any]) -> list[str]:
    needs = semantic_spec.get("certifiable_needs")
    if isinstance(needs, list):
        values = [
            str(item.get("attribute") or item.get("need_id") or "").strip()
            for item in needs if isinstance(item, dict)
        ]
    else:
        values = list(semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or [])
    return list(dict.fromkeys(value for value in values if value))


def _is_collection_attribute(attribute: str, semantic_spec: dict[str, Any]) -> bool:
    """Read collection semantics from the typed query contract."""
    found = False
    for key in ("attribute_bindings", "certifiable_needs"):
        for item in list(semantic_spec.get(key) or []):
            if not isinstance(item, dict):
                continue
            item_attribute = str(item.get("attribute") or item.get("need_id") or "").strip()
            if item_attribute != attribute:
                continue
            found = True
            if str(item.get("need_kind") or item.get("binding_kind") or "").strip().lower() in {
                "record_collection", "collection", "list"
            }:
                return True
    if found:
        # A scalar member binding in a list-shaped contract is still a scalar
        # field. Treating every member as a collection makes open collection
        # compatibility reuse one source slot for unrelated attributes.
        return False
    if str(semantic_spec.get("request_shape") or "").strip().lower() in {"list", "plan"}:
        return True
    # Open collection semantics belong to the outer contract itself. Do not
    # infer collection-ness for scalar members such as path, label, window,
    # or amount merely because another requested field is a summary.
    role_tokens = _meaningful_field_tokens(attribute)
    return bool(role_tokens & {"snapshot", "summary", "overview", "recap", "collection"})


def _is_explicit_collection_contract(
    attribute: str, semantic_spec: dict[str, Any] | None
) -> bool:
    """Distinguish an explicit record bundle from a list-shaped scalar query."""
    binding = _binding_for_attribute(attribute, semantic_spec)
    kind = str(
        binding.get("need_kind")
        or binding.get("binding_kind")
        or ""
    ).strip().lower()
    if kind in {"record_collection", "collection", "list"}:
        hint = str(binding.get("evidence_slot_hint") or "").strip()
        if hint and hint != attribute and hint in _requested_attributes(semantic_spec or {}):
            # The planner sometimes copies an outer bundle name into every
            # member binding. That is not evidence that each scalar member is
            # itself a collection.
            return False
        return True
    tokens = _meaningful_field_tokens(attribute)
    return bool(tokens & {"snapshot", "summary", "overview", "recap", "plan", "state"})


def _candidate_payload(evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _richest_evidence_rows(evidence):
        memory_id = str(row.memory_id or "").strip()
        if not memory_id or memory_id in seen:
            continue
        metadata = dict(row.metadata or {})
        semantic_tags = dict(metadata.get("semantic_tags") or {})
        state_delta = dict(semantic_tags.get("state_delta") or {})
        claim_slots = [
            {
                "slot_name": str(claim.get("property_label") or "").strip(),
                "subject_span": str(claim.get("subject_span") or "").strip(),
                "value": str(
                    dict(semantic_tags.get("attributes") or {}).get(
                        str(claim.get("property_label") or "").strip(),
                        claim.get("value_span") or "",
                    )
                ).strip(),
                "claim_span": str(claim.get("claim_span") or "").strip(),
            }
            for claim in list(semantic_tags.get("claims") or [])
            if isinstance(claim, dict)
            and str(claim.get("property_label") or "").strip()
            and str(claim.get("value_span") or "").strip()
        ]
        slots: dict[str, Any] = {}
        # Keep heuristic aliases for recall, but put explicit current claim
        # values first so a stale alias cannot shadow the canonical value.
        for key, value in dict(state_delta.get("changed_fields") or {}).items():
            key = str(key or "").strip()
            value = str(value or "").strip()
            if key and value:
                slots[key] = value
        for claim in claim_slots:
            slots.setdefault(claim["slot_name"], claim["value"])
        for container in (
            metadata.get("slots"),
            metadata.get("surface_spans"),
            (metadata.get("semantic_tags") or {}).get("attributes"),
            (metadata.get("semantic_tags") or {}).get("surface_values"),
        ):
            if not isinstance(container, dict):
                continue
            for key, value in container.items():
                key = str(key or "").strip()
                if not key or value in (None, "", []):
                    continue
                slots.setdefault(key, value)
        for candidate in list(metadata.get("typed_candidates") or []):
            if not isinstance(candidate, dict):
                continue
            key = str(candidate.get("slot_name") or candidate.get("attribute") or "").strip()
            value = str(candidate.get("value") or "").strip()
            if key and value:
                slots.setdefault(key, value)
        source_text = row_grounded_source_text(row)
        for field in list((metadata.get("stage2_semantic_rerank") or {}).get("typed_fields") or []):
            if not isinstance(field, dict):
                continue
            key = str(field.get("slot_name") or "").strip()
            value = _prefer_precise_claim_value(
                slot_name=key,
                proposed_value=str(field.get("value") or "").strip(),
                claim_slots=claim_slots,
                source_text=row_grounded_source_text(row),
            )
            # A sentence-shaped Stage-2 alias is not useful when semantic
            # claims already expose the record's precise fields. Dropping
            # that alias lets the adjudicator select among the claim spans
            # instead of binding an open attribute to a neighboring slot.
            if key and value and value in source_text and not (
                claim_slots and value == source_text and value not in {
                    str(claim.get("value") or "").strip() for claim in claim_slots
                }
            ):
                slots.setdefault(key, value)
        typed_fields = [
            {
                "attribute": str(field.get("attribute") or "").strip(),
                "slot_name": str(field.get("slot_name") or "").strip(),
                "value": _prefer_precise_claim_value(
                    slot_name=str(field.get("slot_name") or "").strip(),
                    proposed_value=str(field.get("value") or "").strip(),
                    claim_slots=claim_slots,
                    source_text=source_text,
                ),
            }
            for field in list((metadata.get("stage2_semantic_rerank") or {}).get("typed_fields") or [])
            if isinstance(field, dict)
            and str(field.get("attribute") or "").strip()
            and str(field.get("slot_name") or "").strip()
            and str(field.get("value") or "").strip()
            and str(field.get("value") or "").strip() in source_text
            and not (
                claim_slots
                and str(field.get("value") or "").strip() == source_text
                and _prefer_precise_claim_value(
                    slot_name=str(field.get("slot_name") or "").strip(),
                    proposed_value=str(field.get("value") or "").strip(),
                    claim_slots=claim_slots,
                    source_text=source_text,
                ) == source_text
            )
        ]
        # Propagate an explicit current update onto overlapping legacy aliases
        # so stale extractor fields cannot win by name alone.
        changed_fields = dict(state_delta.get("changed_fields") or {})
        for changed_name, changed_value in changed_fields.items():
            changed_name = str(changed_name or "").strip()
            changed_value = str(changed_value or "").strip()
            if not changed_name or not changed_value or changed_value not in source_text:
                continue
            changed_tokens = _field_tokens(changed_name)
            for alias_name in list(slots):
                alias_tokens = _field_tokens(alias_name)
                # Shared structural words such as ``time``/``range`` are not
                # enough to identify an alias: setup_time_range and
                # helper_time_range are distinct source claims.
                if _field_alias_overlap(changed_tokens, alias_tokens):
                    slots[alias_name] = changed_value
        rows.append({
            "memory_id": memory_id,
            "source_text": source_text,
            "source_message_ids": list(row.source_message_ids or []),
            "timestamp": str(row.time or ""),
            "memory_type": str(row.memory_type or ""),
            "lifecycle_status": str(metadata.get("lifecycle_status") or metadata.get("memory_status") or "active"),
            "event_identity": dict(semantic_tags.get("event_identity") or {}),
            "state_delta": state_delta,
            "claim_slots": claim_slots,
            "stage2_served_attributes": list(
                ((metadata.get("stage2_semantic_rerank") or {}).get("served_attributes") or [])
            ),
            "stage2_support_span": str(
                ((metadata.get("stage2_semantic_rerank") or {}).get("support_span") or "")
            ),
            "stage2_binding_mode": str(
                ((metadata.get("stage2_semantic_rerank") or {}).get("stage2_binding_mode") or "")
            ),
            "typed_fields": typed_fields,
            "stage2_typed_fields": typed_fields,
            "utility_source_closure": bool(metadata.get("utility_source_closure")),
            "slots": slots,
        })
        seen.add(memory_id)
    return rows


def _richest_evidence_rows(evidence: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
    """Prefer the duplicate source row with the richest typed annotation."""
    chosen: dict[str, RetrievedEvidence] = {}
    for row in evidence:
        memory_id = str(row.memory_id or "").strip()
        if not memory_id:
            continue
        metadata = dict(row.metadata or {})
        tags = dict(metadata.get("semantic_tags") or {})
        stage2 = dict(metadata.get("stage2_semantic_rerank") or {})
        richness = (
            len(list(tags.get("claims") or [])) * 4
            + len(dict(tags.get("attributes") or {})) * 2
            + len(list(stage2.get("typed_fields") or []))
            + len(dict(metadata.get("slots") or {}))
        )
        prior = chosen.get(memory_id)
        if prior is None:
            chosen[memory_id] = row
            continue
        prior_metadata = dict(prior.metadata or {})
        prior_tags = dict(prior_metadata.get("semantic_tags") or {})
        prior_stage2 = dict(prior_metadata.get("stage2_semantic_rerank") or {})
        prior_richness = (
            len(list(prior_tags.get("claims") or [])) * 4
            + len(dict(prior_tags.get("attributes") or {})) * 2
            + len(list(prior_stage2.get("typed_fields") or []))
            + len(dict(prior_metadata.get("slots") or {}))
        )
        if richness > prior_richness:
            chosen[memory_id] = row
    return list(chosen.values())


def _stage2_attribute_fallback(
    attribute: str,
    candidates: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None = None,
    *,
    question: str = "",
    target_entities: list[str] | None = None,
    requester_id: str | None = None,
) -> dict[str, Any] | None:
    """Recover a missing adjudicator field from an admitted Stage-2 record."""
    normalized_attribute = _field_tokens(attribute)
    expected_slots = _expected_slot_names(attribute, semantic_spec)
    reserved_scalar_slots = {
        slot_name
        for other_attribute in _requested_attributes(semantic_spec or {})
        if other_attribute != attribute
        and not _is_collection_attribute(other_attribute, semantic_spec or {})
        for slot_name in _expected_slot_names(other_attribute, semantic_spec)
    }
    ranked: list[
        tuple[
            tuple[int, str, int, int, int, int, int, int, int, int],
            dict[str, Any],
            str,
            str,
        ]
    ] = []
    current_scope = str((semantic_spec or {}).get("temporal_scope") or "").lower() == "current"
    for record in candidates:
        served = {
            str(value).strip()
            for value in list(record.get("stage2_served_attributes") or [])
            if str(value).strip()
        }
        authoritative_update = _record_has_authoritative_update(
            record, attribute=attribute, semantic_spec=semantic_spec
        )
        # Utility locator records are already a closed, source-grounded
        # factual closure. They may be admitted under a composite/collection
        # attribute while carrying a typed scalar component (for example a
        # current interval). Let the chronology fallback compare that
        # component instead of requiring the model to repeat both bindings.
        if (
            attribute not in served
            and not bool(record.get("utility_source_closure"))
            and not authoritative_update
        ):
            continue
        source = str(record.get("source_text") or "")
        if not source or str(record.get("lifecycle_status") or "active").lower() in _ACTIVE_BLOCKLIST:
            continue
        if _record_has_explicit_scope_conflict(
            record,
            question=question,
            target_entities=target_entities,
            attribute=attribute,
        ):
            continue
        scope_score = _record_query_scope_score(
            record,
            question=question,
            target_entities=target_entities,
        )
        # Stage 2 has already assigned this exact requested attribute to the
        # closed source record.  A compact current recap may omit the entity
        # name, so entity-token scope alone must not discard it before the
        # chronology resolver compares it with older claims.  Conflicting
        # records remain closed-set candidates and are resolved below.
        current_update_context = _record_has_current_attribute_update(
            record,
            attribute=attribute,
        )
        if (
            scope_score < 0
            and attribute not in served
            and not authoritative_update
            and not current_update_context
        ):
            continue
        claim_slot_names = {
            str(claim.get("slot_name") or "").strip()
            for claim in list(record.get("claim_slots") or [])
            if isinstance(claim, dict)
        }
        slot_candidates: list[tuple[int, str, str, bool]] = []
        # Stage 2 has already bound the requested attribute to an exact
        # source-grounded typed field.  Keep that binding ahead of extractor
        # aliases: aliases are useful recall evidence, but can attach a
        # neighboring location/amount/time field to the requested attribute.
        for field in list(record.get("stage2_typed_fields") or []):
            if not isinstance(field, dict):
                continue
            if str(field.get("attribute") or "").strip() != attribute:
                continue
            slot_candidates.append((5, str(field.get("slot_name") or ""), str(field.get("value") or ""), False))
        for claim in list(record.get("claim_slots") or []):
            slot_candidates.append((2, str(claim.get("slot_name") or ""), str(claim.get("value") or ""), True))
        for slot_name, raw_value in dict(record.get("state_delta", {}).get("changed_fields") or {}).items():
            slot_candidates.append((3, str(slot_name or ""), str(raw_value or ""), False))
        for slot_name, raw_value in dict(record.get("slots") or {}).items():
            slot_candidates.append((0, str(slot_name or ""), str(raw_value or ""), False))
        for priority, slot_name, raw_value, is_claim in slot_candidates:
            value = str(raw_value).strip()
            if not slot_name or not value or value not in source:
                continue
            if not _fallback_slot_is_structurally_compatible(
                attribute=attribute,
                slot_name=slot_name,
                semantic_spec=semantic_spec,
                record=record,
            ):
                continue
            if slot_name in reserved_scalar_slots and slot_name not in expected_slots:
                continue
            slot_tokens = _field_tokens(slot_name)
            overlap = len(normalized_attribute & slot_tokens)
            exact = int(slot_tokens == normalized_attribute and bool(normalized_attribute))
            schema_match = int(slot_name in expected_slots)
            explicit_stage2_binding = bool(
                priority == 5
                and not is_claim
                and any(
                    isinstance(field, dict)
                    and str(field.get("attribute") or "").strip() == attribute
                    and str(field.get("slot_name") or "").strip() == slot_name
                    for field in list(record.get("stage2_typed_fields") or [])
                )
            )
            if (
                not exact
                and not overlap
                and not schema_match
                and not explicit_stage2_binding
                and not (is_claim and slot_name in claim_slot_names)
            ):
                continue
            # Scalar fallback normally stays on an explicit Stage-2 binding
            # or an exact claim. A utility source closure is already a
            # locator-selected, source-local factual set, so retain its
            # observed typed slot/state-delta candidates for later chronology
            # and role adjudication. Privacy/deletion paths never set this
            # marker, and the source-grounding/role checks below still apply.
            if (
                not is_claim
                and priority != 5
                and not bool(record.get("utility_source_closure"))
            ):
                continue
            # A fallback for one requested scalar must not borrow a slot that
            # Stage 2 assigned to a different requested scalar in the same
            # source record. This prevents a helper start time from becoming
            # the setup start time merely because both values are timestamps.
            if not is_claim:
                served_attributes = {
                    str(field.get("attribute") or "").strip()
                    for field in list(record.get("typed_attributes") or [])
                    if isinstance(field, dict)
                    and str(field.get("attribute") or "").strip()
                    and str(field.get("attribute") or "").strip() != attribute
                }
                if any(
                    _slot_matches_attribute(other, slot_name, semantic_spec, record)
                    for other in served_attributes
                ):
                    continue
            value_shape = _attribute_value_shape_score(attribute, value)
            if not _slot_matches_attribute(attribute, slot_name, semantic_spec, record):
                continue
            if not factual_value_is_eligible(
                attribute=attribute,
                slot_name=slot_name,
                value=value,
                semantic_spec=semantic_spec,
                source_text=source,
            ):
                continue
            specificity = _slot_semantic_specificity(
                attribute=attribute,
                slot_name=slot_name,
                record=record,
                semantic_spec=semantic_spec,
                requester_id=requester_id,
                question=question,
            )
            # Once a candidate is semantically aligned, prefer a complete
            # source-grounded value over a shorter extractor alias. Priority
            # remains a tie-breaker for equally complete values, preserving
            # authoritative state-delta precedence.
            timestamp = str(record.get("timestamp") or "") if current_scope else ""
            source_turn = max(
                (
                    int(match.group(1))
                    for source_id in list(record.get("source_message_ids") or [])
                    for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
                    if match
                ),
                default=-1,
            ) if current_scope else -1
            ranked.append((
                # Attribute binding quality must beat raw surface length: a
                # shorter Stage-2 typed value is preferable to a longer
                # neighboring extractor alias for the same requested field.
                (
                    int(bool(timestamp)),
                    timestamp,
                    source_turn,
                    specificity,
                    schema_match,
                    exact,
                    value_shape,
                    overlap,
                    priority,
                    len(value),
                    scope_score,
                ),
                record,
                str(slot_name),
                value,
            ))
    if not ranked:
        return None
    _, record, slot_name, value = max(ranked, key=lambda item: item[0])
    evidence_span = str(record.get("stage2_support_span") or "").strip()
    if not evidence_span or evidence_span not in str(record.get("source_text") or ""):
        evidence_span = str(record.get("source_text") or "")
    return {
        "attribute": attribute,
        "memory_id": str(record.get("memory_id") or ""),
        "slot_name": slot_name,
        "value": _normalize_observed_value(
            slot_name,
            value,
            str(record.get("source_text") or ""),
        ),
        "evidence_span": evidence_span,
        "decision": "answer",
        "confidence": 0.0,
        "reason": "stage2_closed_set_attribute_coverage_fallback",
    }


def _fallback_slot_is_structurally_compatible(
    *,
    attribute: str,
    slot_name: str,
    semantic_spec: dict[str, Any] | None,
    record: dict[str, Any],
) -> bool:
    """Keep scalar recovery from crossing typed roles in one source record."""
    if _paired_operational_role_conflict(attribute, slot_name):
        return False
    expected = _expected_slot_names(attribute, semantic_spec)
    if slot_name in expected:
        return True
    attribute_tokens = _meaningful_field_tokens(attribute)
    slot_tokens = _meaningful_field_tokens(slot_name)
    if attribute_tokens & slot_tokens:
        # A shared token is sufficient only when this slot is not being
        # reused as a different scalar property in the same Stage-2 record.
        # A noisy batch decision can otherwise label one path/location slot
        # as path, sort, label, and presence simultaneously.
        scalar_bindings = {
            str(field.get("attribute") or "").strip()
            for field in list(record.get("stage2_typed_fields") or [])
            if isinstance(field, dict)
            and str(field.get("slot_name") or field.get("slot") or "").strip() == str(slot_name).strip()
            and not _is_collection_attribute(
                str(field.get("attribute") or "").strip(), semantic_spec or {}
            )
        }
        if len(scalar_bindings) > 1 and str(attribute).strip() in scalar_bindings:
            direct = attribute_tokens & slot_tokens
            if not direct:
                return False
        return _slot_matches_attribute(attribute, slot_name, semantic_spec, record)
    # A source claim can use a domain-neutral slot label while its claim span
    # states the requested property explicitly (for example, ``pin_status``
    # with "no helper digits remain"). This is a valid typed bridge. It is
    # deliberately claim-local, so a neighboring source sentence cannot
    # authorize a reused slot.
    for claim in list(record.get("claim_slots") or []):
        if str(claim.get("slot_name") or "").strip() != str(slot_name).strip():
            continue
        claim_tokens = _meaningful_field_tokens(str(claim.get("claim_span") or ""))
        if claim_tokens & attribute_tokens:
            return True
    # An open label may bridge a naming gap only when the source record has a
    # unique scalar binding for it.  If Stage 2 reused the same slot for
    # several scalar attributes, accept only a binding with an independent
    # semantic signal (handled by the overlap/expected-slot paths above).
    bindings = [
        field for field in list(record.get("stage2_typed_fields") or [])
        if isinstance(field, dict)
        and str(field.get("slot_name") or field.get("slot") or "").strip() == str(slot_name).strip()
    ]
    scalar_attributes = {
        str(field.get("attribute") or "").strip()
        for field in bindings
        if str(field.get("attribute") or "").strip()
        and not _is_collection_attribute(str(field.get("attribute") or ""), semantic_spec or {})
    }
    return (
        len(scalar_attributes) <= 1
        and str(attribute).strip() in scalar_attributes
        and _slot_matches_attribute(attribute, slot_name, semantic_spec, record)
    )


def _record_has_authoritative_update(
    record: dict[str, Any], *, attribute: str, semantic_spec: dict[str, Any] | None
) -> bool:
    """Recognize an exact typed update even when Stage 2 omitted its label."""
    source = str(record.get("source_text") or "")
    for slot_name, raw_value in dict(record.get("state_delta", {}).get("changed_fields") or {}).items():
        value = str(raw_value or "").strip()
        if (
            value
            and value in source
            and _slot_matches_attribute(attribute, str(slot_name), semantic_spec, record)
        ):
            return True
    return False


def _record_has_current_attribute_update(
    record: dict[str, Any], *, attribute: str
) -> bool:
    """Recognize a source-local current update for one requested field.

    Utility source closure may include a later update whose sentence relies on
    the surrounding dialogue for object identity.  Admit it only when the
    source itself carries a current/update marker and its typed claim context
    names the requested role.  This keeps chronology recovery closed and does
    not turn an unrelated neighboring record into evidence.
    """
    if not record.get("utility_source_closure"):
        return False
    source = str(record.get("source_text") or "")
    if not re.search(
        r"\b(?:current(?:ly)?|latest|updated?|moved?|changed?|revised?|"
        r"replaced?|chang(?:e|ed|ing)|still|confirmed|remaining|held)\b",
        source,
        re.IGNORECASE,
    ):
        return False
    attribute_tokens = _meaningful_field_tokens(attribute)
    if not attribute_tokens:
        return False
    context_parts = [
        str(record.get("event_identity", {}).get(key) or "")
        for key in ("entity_key", "entity_surface_span", "subject_span")
    ]
    for claim in list(record.get("claim_slots") or []):
        context_parts.extend([
            str(claim.get("slot_name") or ""),
            str(claim.get("subject_span") or ""),
            str(claim.get("claim_span") or ""),
        ])
    return bool(attribute_tokens & _meaningful_field_tokens(" ".join(context_parts)))


def _fallback_is_authoritative_update(
    item: dict[str, Any],
    candidate_by_id: dict[str, dict[str, Any]],
    *,
    attribute: str,
    semantic_spec: dict[str, Any] | None,
) -> bool:
    """Allow current-state fallback only for an explicit typed update."""
    record = candidate_by_id.get(str(item.get("memory_id") or ""), {})
    source = str(record.get("source_text") or "")
    for slot_name, raw_value in dict(record.get("state_delta", {}).get("changed_fields") or {}).items():
        value = str(raw_value or "").strip()
        if (
            value
            and value in source
            and _slot_matches_attribute(attribute, str(slot_name), semantic_spec, record)
        ):
            return True
    return False


def _stage2_collection_fallbacks(
    attribute: str,
    candidates: list[dict[str, Any]],
    semantic_spec: dict[str, Any] | None = None,
    *,
    question: str = "",
    target_entities: list[str] | None = None,
    excluded_memory_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Recover every explicitly typed member of an admitted collection."""
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    reserved_scalar_slots = {
        slot_name
        for other_attribute in _requested_attributes(semantic_spec or {})
        if other_attribute != attribute
        and not _is_collection_attribute(other_attribute, semantic_spec or {})
        for slot_name in _expected_slot_names(other_attribute, semantic_spec)
    }
    explicit_collection_contract = _is_explicit_collection_contract(
        attribute, semantic_spec
    )
    for record in candidates:
        excluded = str(record.get("memory_id") or "") in set(excluded_memory_ids or set())
        served = attribute in {
            str(value).strip()
            for value in list(record.get("stage2_served_attributes") or [])
        }
        utility_closure = bool(record.get("utility_source_closure"))
        if not served and not (utility_closure and explicit_collection_contract):
            continue
        source = str(record.get("source_text") or "")
        if not source or str(record.get("lifecycle_status") or "active").lower() in _ACTIVE_BLOCKLIST:
            continue
        if (
            _record_query_scope_score(
                record,
                question=question,
                target_entities=target_entities,
            ) < 0
            and not served
            and not (utility_closure and explicit_collection_contract)
        ):
            # The utility locator already selected this record as part of the
            # source-message closure. For an outer collection, neighboring
            # turns may carry the lane identity implicitly; defer that
            # source-local association to the collection audit and typed role
            # checks instead of requiring repeated entity text.
            continue
        # An outer collection is completed only from fields explicitly bound
        # to that collection by Stage 2. Raw claims remain closed evidence,
        # but promoting every claim here turns policy and sibling-thread
        # annotations into answer members. Missing components are handled by
        # the batched LLM coverage pass upstream.
        field_candidates = [] if excluded or (utility_closure and not served) else [
            field for field in list(record.get("typed_fields") or [])
            if isinstance(field, dict)
            and str(field.get("attribute") or "").strip() == attribute
        ]
        if (
            not excluded
            and utility_closure
            and explicit_collection_contract
            and not served
            and _collection_memory_type_compatible(attribute, record)
        ):
            field_candidates.extend(_claim_backed_collection_fields(attribute, record))
        if not excluded and str((semantic_spec or {}).get("temporal_scope") or "").lower() == "current" and (
            served or utility_closure
        ):
            # State deltas are explicit source-local updates even when the
            # reranker served a sibling field from the same record. Expose
            # only the changed field whose role matches this collection need;
            # the current-component resolver below removes stale aliases.
            existing_slots = {
                str(field.get("slot_name") or "").strip()
                for field in field_candidates
                if isinstance(field, dict)
            }
            for changed_name, changed_value in dict(record.get("state_delta", {}).get("changed_fields") or {}).items():
                changed_name = str(changed_name or "").strip()
                changed_value = str(changed_value or "").strip()
                if (
                    changed_name
                    and changed_value
                    and changed_name not in existing_slots
                    and changed_value in source
                    and _slot_matches_attribute(attribute, changed_name, semantic_spec, record)
                ):
                    field_candidates.append({
                        "attribute": attribute,
                        "slot_name": changed_name,
                        "value": changed_value,
                    })
        for field in field_candidates:
            field_attribute = str(field.get("attribute") or "").strip()
            if field_attribute and field_attribute != attribute:
                continue
            slot_name = str(field.get("slot_name") or "").strip()
            value = str(field.get("value") or "").strip()
            if not slot_name or not value or value not in source:
                continue
            derived_collection_claim = bool(field.get("_collection_claim_derived"))
            claim_backed = derived_collection_claim or any(
                str(claim.get("slot_name") or "").strip() == slot_name
                for claim in list(record.get("claim_slots") or [])
                if isinstance(claim, dict)
            )
            # A sentence-shaped extractor alias without a claim boundary is
            # a record view, not an independently typed collection member.
            # Keeping it would replay policy/questions or collapse several
            # neighboring fields into one answer value. Claim-backed spans
            # and shorter typed components remain eligible.
            grounded_lines = {
                line.strip()
                for line in source.splitlines()
                if line.strip()
            }
            if (
                not claim_backed
                and value in grounded_lines
                and len(value.split()) >= 4
            ):
                continue
            # A source-local composite claim can contribute several typed
            # operational members, but it must still pass the role boundary.
            # The previous explicit-collection shortcut bypassed that check
            # for claim-backed fields and allowed access-like siblings to
            # re-enter an otherwise safe collection.
            if not derived_collection_claim and not _is_shared_composite_claim_component(
                attribute=attribute,
                field=field,
                record=record,
            ) or _field_requires_collection_role_check(field):
                if not _collection_claim_role_compatible(
                    attribute,
                    field,
                    semantic_spec=semantic_spec,
                    target_entities=target_entities,
                ):
                    continue
            if slot_name in reserved_scalar_slots:
                continue
            if not derived_collection_claim and not _slot_matches_attribute(attribute, slot_name, semantic_spec, record):
                continue
            if not factual_value_is_eligible(
                attribute=attribute,
                slot_name=slot_name,
                value=value,
                semantic_spec=semantic_spec,
                source_text=source,
            ):
                continue
            if (
                not _slot_matches_attribute(
                    attribute,
                    slot_name,
                    {"requested_attributes": [attribute]},
                    record,
                )
                and not any(
                    str(claim.get("slot_name") or "").strip() == slot_name
                    for claim in list(record.get("claim_slots") or [])
                )
            ):
                continue
            key = (str(record.get("memory_id") or ""), slot_name, value)
            if key in seen:
                continue
            seen.add(key)
            evidence_span = str(record.get("stage2_support_span") or "").strip()
            if not evidence_span or evidence_span not in source or value not in evidence_span:
                evidence_span = source
            result.append({
                "attribute": attribute,
                "memory_id": str(record.get("memory_id") or ""),
                "slot_name": slot_name,
                "value": value,
                "evidence_span": evidence_span,
                "claim_span": str(field.get("claim_span") or "").strip(),
                "subject_span": str(field.get("subject_span") or "").strip(),
                "decision": "answer",
                "confidence": 0.0,
                "reason": "stage2_closed_set_collection_typed_field_fallback",
            })
    return result


def _collection_memory_type_compatible(attribute: str, record: dict[str, Any]) -> bool:
    """Use a typed memory kind as a narrow outer-collection role signal."""
    memory_type_tokens = _meaningful_field_tokens(str(record.get("memory_type") or ""))
    attribute_tokens = _meaningful_field_tokens(attribute)
    if not memory_type_tokens or not attribute_tokens:
        return False
    normalized_memory = {
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in memory_type_tokens
    }
    normalized_attribute = {
        token[:-1] if token.endswith("s") and len(token) > 3 else token
        for token in attribute_tokens
    }
    return bool(normalized_memory & normalized_attribute)


def _claim_backed_collection_fields(
    attribute: str,
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project complete claim spans for a type-compatible collection record."""
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for claim in list(record.get("claim_slots") or []):
        if not isinstance(claim, dict):
            continue
        claim_span = str(claim.get("claim_span") or "").strip()
        if not claim_span or claim_span in seen:
            continue
        if not factual_value_is_eligible(
            attribute=attribute,
            slot_name="source_claim",
            value=claim_span,
            source_text=str(record.get("source_text") or ""),
        ):
            continue
        seen.add(claim_span)
        fields.append({
            "attribute": attribute,
            "slot_name": "source_claim",
            "value": claim_span,
            "claim_span": claim_span,
            "_collection_claim_derived": True,
        })
    return fields


def _resolve_current_collection_components(
    items: list[dict[str, Any]],
    *,
    candidate_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse stale aliases while preserving distinct current components."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("attribute") or ""), []).append(item)
    resolved: list[dict[str, Any]] = []
    access_tokens = {
        "access", "code", "credential", "credentials", "digit", "digits",
        "key", "passcode", "password", "pin", "secret", "token",
    }

    def recency(item: dict[str, Any]) -> tuple[str, int, float, int]:
        record = candidate_by_id.get(str(item.get("memory_id") or ""), {})
        turn = max(
            (
                int(match.group(1))
                for source_id in list(record.get("source_message_ids") or [])
                for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
                if match
            ),
            default=-1,
        )
        return (
            str(record.get("timestamp") or ""),
            turn,
            float(item.get("confidence") or 0.0),
            len(str(item.get("value") or "")),
        )

    for attribute, candidates in grouped.items():
        attr_tokens = _meaningful_field_tokens(attribute)
        if not attr_tokens:
            resolved.extend(candidates)
            continue
        # A non-access summary must not absorb a numeric credential-like
        # sibling merely because the source record was admitted as a bundle.
        if not attr_tokens & access_tokens:
            candidates = [
                item for item in candidates
                if not (
                    _meaningful_field_tokens(str(item.get("slot_name") or "")) & access_tokens
                    and re.search(r"\d", str(item.get("value") or ""))
                )
            ]
        by_value: dict[str, dict[str, Any]] = {}
        for item in candidates:
            value_key = str(item.get("value") or "").strip().lower()
            prior = by_value.get(value_key)
            if prior is None or recency(item) > recency(prior):
                by_value[value_key] = item
        candidates = list(by_value.values())
        # If one source-grounded component is a strict substring of another
        # value for the same requested property, the longer value is the
        # composite current claim and the shorter alias is redundant.
        candidates = [
            item for item in candidates
            if not any(
                str(item.get("value") or "").strip().lower()
                != str(other.get("value") or "").strip().lower()
                and str(item.get("value") or "").strip().lower()
                in str(other.get("value") or "").strip().lower()
                for other in candidates
            )
        ]
        families: list[tuple[set[str], list[dict[str, Any]]]] = []
        for item in candidates:
            slot_tokens = _meaningful_field_tokens(str(item.get("slot_name") or ""))
            family = slot_tokens & attr_tokens
            if not family:
                resolved.append(item)
                continue
            bucket = next((bucket for bucket in families if bucket[0] & family), None)
            if bucket is None:
                families.append((set(family), [item]))
            else:
                bucket[0].update(family)
                bucket[1].append(item)
        for _, family_items in families:
            # Strict substring aliases were removed above. For independent
            # values (for example initial vs updated color), chronology is
            # the current-state resolver.
            resolved.append(max(family_items, key=recency))
    return resolved


def _field_requires_collection_role_check(field: dict[str, Any]) -> bool:
    """Keep sensitive typed roles behind the normal collection role check."""
    slot_tokens = _meaningful_field_tokens(str(field.get("slot_name") or ""))
    return bool(
        slot_tokens
        & {
            "access", "code", "credential", "credentials", "key", "passcode",
            "password", "phrase", "pin", "secret", "token",
        }
    )


def _is_shared_composite_claim_component(
    *,
    attribute: str,
    field: dict[str, Any],
    record: dict[str, Any],
) -> bool:
    """Keep sibling typed fields from one explicitly served source claim."""
    if not _is_collection_attribute(attribute, {"requested_attributes": [attribute]}):
        return False
    if attribute not in {
        str(value).strip()
        for value in list(record.get("stage2_served_attributes") or [])
        if str(value).strip()
    }:
        return False
    claim_span = str(field.get("claim_span") or "").strip()
    if not claim_span:
        return False
    sibling_count = sum(
        1
        for claim in list(record.get("claim_slots") or [])
        if isinstance(claim, dict)
        and str(claim.get("claim_span") or "").strip() == claim_span
        and str(claim.get("slot_name") or "").strip()
        and str(claim.get("value") or "").strip()
    )
    return sibling_count > 1


def _record_query_scope_score(
    record: dict[str, Any],
    *,
    question: str,
    target_entities: list[str] | None,
) -> int:
    """Reject fallback claims whose event scope conflicts with the query.

    Stage 2 may admit a source-message closure for utility recall. That does
    not make every record in the closure a candidate for every requested
    scalar. Prefer explicit event identity, while allowing a record that
    mentions only the shared head of a compound entity (for example, a
    ``Pantry handoff`` record for ``Ivy Pantry``). A different salient token
    in the same event identity is treated as a scope conflict.
    """
    entities = [
        str(value).strip()
        for value in list(target_entities or [])
        if str(value).strip()
    ]
    if not entities:
        return 0
    target_phrases = [
        _scope_tokens(value)
        for value in entities
        if _scope_tokens(value)
    ]
    if not target_phrases:
        return 0
    source = str(record.get("source_text") or "")
    identity = dict(record.get("event_identity") or {})
    identity_text = " ".join(
        str(identity.get(key) or "")
        for key in ("entity_key", "entity_surface_span", "subject_span")
    )
    identity_tokens = _scope_tokens(identity_text)
    source_tokens = _scope_tokens(source)
    query_tokens = _scope_tokens(question)
    target_union = set().union(*target_phrases)

    # An explicit compound target match is the strongest generic signal.
    if any(phrase <= identity_tokens for phrase in target_phrases if len(phrase) > 1):
        return 3
    if any(phrase <= source_tokens for phrase in target_phrases if len(phrase) > 1):
        return 2

    target_heads = {
        token
        for phrase in target_phrases
        for token in phrase
        if token not in _SCOPE_MODIFIERS
    }
    identity_overlap = identity_tokens & target_heads
    if not identity_overlap:
        # A source-only head match is useful only when the query itself
        # clearly names that head; otherwise a closure record is unrelated.
        if not (source_tokens & target_heads & query_tokens):
            return -1
        return 0

    # The event identity is the preferred scope boundary. For a compound
    # target, reject a sibling entity such as ``Ivy Pots`` when the query is
    # about ``Ivy Pantry``. Structural words like handoff/state are not
    # treated as competing entities.
    identity_specific = identity_tokens - _SCOPE_GENERIC_EVENT_TOKENS
    conflicting = identity_specific - target_union
    if len(identity_overlap) < max(len(phrase) for phrase in target_phrases) and conflicting:
        return -1
    return 1


def _record_has_explicit_scope_conflict(
    record: dict[str, Any],
    *,
    question: str,
    target_entities: list[str] | None,
    attribute: str = "",
) -> bool:
    """Reject only records with an explicit conflicting event identity.

    Unnamed neighboring turns remain eligible for source-local completion.
    A named event identity that conflicts with the query, however, is a
    different lane and must not win merely because it is newer or shares a
    generic slot such as ``date``.
    """
    identity = dict(record.get("event_identity") or {})
    identity_text = " ".join(
        str(identity.get(key) or "")
        for key in ("entity_key", "entity_surface_span", "subject_span")
    ).strip()
    if not identity_text or not target_entities:
        return False
    # A current utility update can use a field identity (for example a color,
    # date, or helper window) rather than repeating the outer object name.
    # Its typed claim context is a sufficient local bridge; unrelated records
    # still fail the source-role and lifecycle checks below.
    if attribute and _record_has_current_attribute_update(record, attribute=attribute):
        return False
    return _record_query_scope_score(
        record,
        question=question,
        target_entities=target_entities,
    ) < 0


def _claim_matches_target_scope(
    claim: dict[str, Any], *, target_entities: list[str] | None
) -> bool:
    """Keep source-local collection claims bound to the requested object."""
    subject = str(claim.get("subject_span") or "").strip()
    if not subject or not target_entities:
        return True
    subject_tokens = _scope_tokens(subject)
    target_phrases = [
        _scope_tokens(value)
        for value in list(target_entities or [])
        if _scope_tokens(value)
    ]
    if any(phrase <= subject_tokens for phrase in target_phrases if len(phrase) > 1):
        return True
    target_union = set().union(*target_phrases) if target_phrases else set()
    target_heads = target_union - _SCOPE_MODIFIERS - _SCOPE_GENERIC_EVENT_TOKENS
    overlap = subject_tokens & target_heads
    if not overlap:
        return False
    subject_specific = subject_tokens - _SCOPE_MODIFIERS - _SCOPE_GENERIC_EVENT_TOKENS
    subject_descriptors = {
        "broad", "wording", "exact", "private", "safe", "household",
        "helper", "signoff", "summary", "note", "details", "detail",
    }
    return not (subject_specific - target_union - subject_descriptors)


def _collection_claim_role_compatible(
    attribute: str,
    field: dict[str, Any],
    *,
    semantic_spec: dict[str, Any] | None,
    target_entities: list[str] | None,
) -> bool:
    """Avoid treating every source-local claim as a collection component."""
    slot = str(field.get("slot_name") or "").strip()
    subject = str(field.get("subject_span") or "").strip()
    attr_tokens = _meaningful_field_tokens(attribute)
    binding = _binding_for_attribute(attribute, semantic_spec)
    attr_tokens.update(_meaningful_field_tokens(str(binding.get("support_span") or "")))
    structural = _STRUCTURAL_FIELD_TOKENS | {
        "safe", "current", "latest", "state", "status", "record", "property",
        "requested", "household", "final", "only",
    }
    attr_specific = attr_tokens - structural
    slot_specific = _meaningful_field_tokens(slot) - structural
    access_like_slots = {
        "access", "code", "credential", "credentials", "key", "passcode",
        "password", "phrase", "pin", "secret", "token",
    }
    # A source record may explicitly mention a credential while being admitted
    # for a broader safe-state collection. Keep that typed role isolated before
    # any source-local claim shortcut can treat it as a generic sibling.
    if slot_specific & access_like_slots and not attr_specific & access_like_slots:
        return False
    if attr_specific & slot_specific:
        return True
    if (
        attr_specific & {"snapshot", "summary", "overview", "recap", "plan", "state"}
        and not slot_specific & access_like_slots
    ):
        # Open-schema record summaries can expose useful operational metadata
        # under labels that have no lexical overlap with the outer request.
        # Keep the typed component, while access-like roles remain excluded
        # unless the request explicitly asks for that access artifact.
        return True
    if subject and _claim_matches_target_scope(
        field, target_entities=target_entities
    ):
        # Named, source-local summaries are useful composite components even
        # when their open slot label does not repeat the query wording.
        if slot_specific & {"private", "exact", "credential", "pin", "password", "token", "phrase"}:
            return False
        return bool(
            slot_specific
            & {"summary", "wording", "state", "status", "logistics", "usage", "contingency"}
            or attr_specific & {"summary", "snapshot", "plan", "state"}
        )
    return False


_SCOPE_MODIFIERS = {
    "current", "latest", "opening", "active", "available", "separate",
}
_SCOPE_GENERIC_EVENT_TOKENS = _SCOPE_MODIFIERS | {
    "state", "status", "plan", "handoff", "window", "note", "move",
    "moving", "delivery", "stocking", "schedule", "check", "visit",
    "rule", "policy", "scope", "thread", "details", "detail", "record",
    "event", "operation", "operation", "summary", "overview", "current",
}


def _scope_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if token not in {"a", "an", "the", "and", "or", "of", "to", "in", "for", "is"}
    }


def _attribute_value_shape_score(attribute: str, value: str) -> int:
    """Prefer values whose observed shape matches the requested typed field."""
    attribute_tokens = _field_tokens(attribute)
    numeric_field = bool(
        attribute_tokens
        & {"amount", "support", "stipend", "budget", "cost", "price", "fee", "discount", "number", "count"}
    )
    numeric_value = bool(re.search(r"\d", str(value or "")))
    return 3 if numeric_field and numeric_value else (-2 if numeric_field and not numeric_value else 0)


def _normalize_observed_value(slot_name: str, value: str, source_text: str) -> str:
    """Remove an extractor field prefix only when the shorter span is grounded."""
    normalized_slot = str(slot_name or "").lower().replace("_", " ").strip()
    normalized_value = str(value or "").strip()
    prefix = normalized_slot + " "
    if normalized_slot in {"label", "safe label", "safe wording"} and normalized_value.lower().startswith(prefix):
        candidate = normalized_value[len(prefix):].strip(" :")
        if candidate and candidate in source_text:
            return candidate
    return normalized_value


def _normalize_current_transition_value(*, attribute: str, value: str, source_text: str) -> str:
    """Project a current scalar from a source-grounded update transition.

    Typed extractors often preserve an entire old-to-current phrase in one
    slot. For non-temporal scalar fields, the current side is the useful
    value. Time windows remain composite values and are left untouched.
    """
    normalized = str(value or "").strip()
    if not normalized or not source_text:
        return normalized
    tokens = _meaningful_field_tokens(attribute)
    if tokens & {"time", "window", "interval", "schedule", "date"}:
        return normalized
    if not re.search(
        r"\b(?:from|to|into|becomes?|updates?|changes?)\b|->|=>",
        normalized,
        re.IGNORECASE,
    ):
        return normalized
    parts = re.split(
        r"\s+(?:to|into|becomes?|updates?|changes?)\s+|\s*[-=]>\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    if len(parts) < 2:
        return normalized
    current = parts[-1].strip(" .,:;")
    return current if current and current in source_text else normalized


def _expected_slot_names(attribute: str, semantic_spec: dict[str, Any] | None) -> set[str]:
    """Resolve open attribute labels to the shared typed state vocabulary."""
    spec = dict(semantic_spec or {})
    bindings = [
        item
        for item in list(spec.get("attribute_bindings") or [])
        if isinstance(item, dict) and str(item.get("attribute") or "").strip() == attribute
    ]
    phrases = {attribute.lower().replace("_", " ")}
    for item in bindings:
        for key in ("support_span", "evidence_slot_hint", "semantic_role"):
            value = str(item.get(key) or "").strip().lower().replace("_", " ")
            if value:
                phrases.add(value)
    expected: set[str] = set()
    for slot_name, aliases in CURRENT_STATE_SLOT_ALIASES.items():
        slot_phrases = [str(slot_name).replace("_", " "), *[str(alias) for alias in aliases]]
        if any(
            phrase == candidate or phrase in candidate or candidate in phrase
            for phrase in phrases
            for candidate in slot_phrases
        ):
            expected.add(str(slot_name))
    for item in bindings:
        hint = str(item.get("evidence_slot_hint") or "").strip()
        if hint:
            expected.add(hint)
    return expected


def _coerce_observed_slot_value(
    record: dict[str, Any] | None,
    slot_name: str,
    proposed_value: str,
) -> str:
    """Normalize a model-selected value to an exact observed typed value."""
    if not record:
        return proposed_value
    candidates: list[str] = []
    for claim in list(record.get("claim_slots") or []):
        if str(claim.get("slot_name") or "").strip() == slot_name:
            value = str(claim.get("value") or "").strip()
            if value:
                candidates.append(value)
    observed = record.get("slots", {}).get(slot_name)
    if isinstance(observed, list):
        candidates.extend(str(value).strip() for value in observed if str(value).strip())
    elif observed:
        candidates.append(str(observed).strip())
    candidates = list(dict.fromkeys(candidates))
    precise_claim_values = [
        str(claim.get("value") or "").strip()
        for claim in list(record.get("claim_slots") or [])
        if str(claim.get("slot_name") or "").strip() == slot_name
        and str(claim.get("value") or "").strip()
    ]
    if proposed_value in candidates:
        narrower = [
            value for value in precise_claim_values
            if value != proposed_value and value in proposed_value
        ]
        if narrower:
            return max(narrower, key=len)
        complete_span = _complete_claim_span_for_value(record, proposed_value)
        if complete_span:
            return complete_span
        return _normalize_observed_value(slot_name, proposed_value, str(record.get("source_text") or ""))
    precise_narrower = [
        value for value in precise_claim_values
        if value and value in proposed_value and value != proposed_value
    ]
    if precise_narrower:
        return max(precise_narrower, key=len)
    # A typed extractor may anchor a composite claim at one component (for
    # example, a range start). Preserve a longer value only when the model's
    # proposal is an exact source substring in that anchor's claim span and
    # contains the observed anchor. This generalizes to locations, labels,
    # amounts, and other composite values without inventing a schema rule.
    source_text = str(record.get("source_text") or "")
    if _source_grounded_composite_value(
        record=record,
        slot_name=slot_name,
        proposed_value=proposed_value,
        observed_values=candidates,
        source_text=source_text,
    ):
        return proposed_value
    containing = [value for value in candidates if value and value in proposed_value]
    selected = max(containing, key=len) if containing else proposed_value
    return _normalize_observed_value(slot_name, selected, source_text)


def _prefer_precise_claim_value(
    *,
    slot_name: str,
    proposed_value: str,
    claim_slots: list[dict[str, Any]],
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


def _complete_claim_span_for_value(
    record: dict[str, Any], proposed_value: str
) -> str:
    """Recover a complete source claim when an extractor exposed one component.

    A claim span can contain several typed components for one property, while
    the extractor may expose only the first component as ``value``.  Returning
    that existing span preserves source grounding and avoids silently dropping
    a required complement such as a renewal clause or condition.
    """
    proposal = str(proposed_value or "").strip()
    source_text = str(record.get("source_text") or "")
    if not proposal or not source_text:
        return ""
    for claim in list(record.get("claim_slots") or []):
        value = str(claim.get("value") or "").strip()
        claim_span = str(claim.get("claim_span") or "").strip()
        if (
            not value
            or not (value == proposal or value in proposal or proposal in value)
            or not claim_span
            or claim_span not in source_text
        ):
            continue
        siblings = [
            other for other in list(record.get("claim_slots") or [])
            if str(other.get("claim_span") or "").strip() == claim_span
            and str(other.get("value") or "").strip()
            and str(other.get("value") or "").strip() != proposal
        ]
        if siblings:
            return claim_span
    return ""


def _source_grounded_composite_value(
    *,
    record: dict[str, Any],
    slot_name: str,
    proposed_value: str,
    observed_values: list[str],
    source_text: str,
) -> bool:
    """Validate a longer model-selected value against one typed source claim."""
    proposal = str(proposed_value or "").strip()
    if not proposal or not source_text or proposal not in source_text:
        return False
    for claim in list(record.get("claim_slots") or []):
        if str(claim.get("slot_name") or "").strip() != slot_name:
            continue
        claim_span = str(claim.get("claim_span") or "").strip()
        if not claim_span or claim_span not in source_text or proposal not in claim_span:
            continue
        if any(
            anchor
            and anchor != proposal
            and anchor in proposal
            and anchor in claim_span
            for anchor in observed_values
        ):
            return True
    return False


def _field_tokens(value: str) -> set[str]:
    return {token for token in str(value or "").lower().replace("-", "_").split("_") if token}


_STRUCTURAL_FIELD_TOKENS = {
    "a", "an", "the", "current", "latest", "active", "available",
    "time", "times", "range", "window", "windows", "interval",
    "date", "day", "days", "value", "values", "field", "fields",
}

# Complete language-level function-word inventory used when comparing typed
# field labels. These words carry little property identity in English; they
# are deliberately separated from semantic role tokens such as ``date``,
# ``status``, or ``scope``. Keeping the inventory explicit makes the token
# normalization reproducible and avoids a handful of query-specific words
# acting as an implicit patch.
_FIELD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "yours", "yourself", "yourselves",
}


def _field_alias_overlap(left: set[str], right: set[str]) -> bool:
    """Require a discriminative token before copying one typed field to another."""
    if not left or not right:
        return False
    shared = left & right
    if not shared:
        return False
    discriminative_left = left - _STRUCTURAL_FIELD_TOKENS
    discriminative_right = right - _STRUCTURAL_FIELD_TOKENS
    return bool(
        (discriminative_left & discriminative_right)
        or (left == right)
        or (left <= right and discriminative_left)
        or (right <= left and discriminative_right)
    )


def _slot_semantic_specificity(
    *,
    attribute: str,
    slot_name: str,
    record: dict[str, Any],
    semantic_spec: dict[str, Any] | None,
    requester_id: str | None = None,
    question: str = "",
) -> int:
    """Rank source fields by typed-role specificity, not just recency."""
    binding = _binding_for_attribute(attribute, semantic_spec)
    requested_tokens = _meaningful_field_tokens(attribute)
    for key in ("support_span", "evidence_slot_hint"):
        requested_tokens.update(_meaningful_field_tokens(str(binding.get(key) or "")))
    structural = _STRUCTURAL_FIELD_TOKENS | {
        "setup", "helper", "approved", "current", "safe", "summary",
        "state", "status", "plan", "record", "property", "requested",
    }
    requested_specific = requested_tokens - structural
    slot_tokens = _meaningful_field_tokens(slot_name)
    slot_specific = slot_tokens - structural
    score = 4 * len(requested_specific & slot_specific)
    if slot_name in _expected_slot_names(attribute, semantic_spec):
        score += 2
    if slot_tokens == requested_tokens and slot_tokens:
        score += 3
    requester_tokens = _scope_tokens(str(requester_id or "").replace("_", " "))
    if _question_targets_first_person_field(question, attribute, binding):
        score += 4 * len(slot_specific & requester_tokens)
    else:
        score -= 5 * len(slot_specific & requester_tokens)
    explicit_slots = {
        str(field.get("slot_name") or "").strip()
        for field in list(record.get("stage2_typed_fields") or [])
        if isinstance(field, dict)
    }
    if slot_name in explicit_slots:
        score += 2
    if sum(
        1
        for field in list(record.get("stage2_typed_fields") or [])
        if isinstance(field, dict)
        and str(field.get("slot_name") or "").strip() == slot_name
        and str(field.get("attribute") or "").strip() != attribute
    ):
        score -= 3
    for claim in list(record.get("claim_slots") or []):
        if str(claim.get("slot_name") or "").strip() != slot_name:
            continue
        subject_tokens = _meaningful_field_tokens(str(claim.get("subject_span") or "")) - structural
        score += 3 * len(requested_specific & subject_tokens)
        score += 2 * len(subject_tokens & requester_tokens)
    return score


def _question_targets_first_person_field(
    question: str, attribute: str, binding: dict[str, Any]
) -> bool:
    """Detect a local first-person binding without treating all fields as "my"."""
    words = re.findall(r"[a-z0-9]+", str(question or "").lower())
    phrases = [
        re.findall(r"[a-z0-9]+", str(attribute or "").lower()),
        re.findall(r"[a-z0-9]+", str(binding.get("support_span") or "").lower()),
    ]
    for phrase in phrases:
        if not phrase:
            continue
        for index in range(len(words) - len(phrase) + 1):
            if words[index:index + len(phrase)] == phrase:
                nearby = words[max(0, index - 2):index + len(phrase) + 2]
                if "my" in nearby or "mine" in nearby:
                    return True
    return False


def _slot_matches(
    record: dict[str, Any] | None,
    slot_name: str,
    value: str,
    attribute: str = "",
) -> bool:
    if not record:
        return False
    slots = dict(record.get("slots") or {})
    if slot_name not in slots:
        # Stage 2 may admit a source whose extractor omitted a conventional
        # slot (for example, a logistics-only operational artifact). Permit
        # the adjudicator to assign a stable typed label only for an attribute
        # already served by that exact record and an exact source substring.
        return bool(
            attribute
            and attribute in {
                str(item).strip()
                for item in list(record.get("stage2_served_attributes") or [])
            }
            and value
            and value in str(record.get("source_text") or "")
        )
    candidate = slots.get(slot_name)
    if isinstance(candidate, list):
        observed = {str(item).strip() for item in candidate}
    else:
        observed = {str(candidate).strip()}
    if value in observed:
        return True
    return bool(value and value in str(record.get("source_text") or "") and any(
        item and item in value for item in observed
    ))


def _slot_matches_attribute(
    attribute: str,
    slot_name: str,
    semantic_spec: dict[str, Any] | None,
    record: dict[str, Any] | None = None,
) -> bool:
    """Keep a typed field in the semantic role requested by the query.

    Exact source grounding is necessary but insufficient: a broad summary
    slot can contain a label, location, and access path at once.  Treating
    that slot as any one of those scalar properties makes the final
    realization look grounded while changing the field meaning.  This check
    uses the query binding and generic field structure, and still permits a
    model-assigned semantic slot when its name is aligned with the request.
    """
    attribute = str(attribute or "").strip()
    slot_name = str(slot_name or "").strip()
    if not attribute or not slot_name:
        return False
    if _paired_operational_role_conflict(attribute, slot_name):
        return False
    binding = _binding_for_attribute(attribute, semantic_spec)
    attr_tokens = _meaningful_field_tokens(attribute)
    slot_tokens = _meaningful_field_tokens(slot_name)
    binding_tokens: set[str] = set()
    for key in ("support_span", "evidence_slot_hint", "semantic_role"):
        binding_tokens.update(_meaningful_field_tokens(str(binding.get(key) or "")))
    role_binding_tokens = set(binding_tokens)
    direct_role_tokens = attr_tokens | binding_tokens
    composite_request = bool(
        {"plan", "schedule", "workflow", "protocol", "procedure"} & direct_role_tokens
    )
    # Qualifier and governance wording can be source-grounded while still
    # describing how a fact may be disclosed rather than the requested fact
    # itself. Keep this boundary representation-level: it applies to any
    # scalar property, independent of domain vocabulary.
    qualifier_tokens = {
        "allow", "allowed", "authorization", "authorized", "condition",
        "disclose", "disclosure", "guidance", "institutional", "permission",
        "policy", "require", "required", "restriction", "scope", "sensitive",
    }
    if (
        not _is_collection_attribute(attribute, semantic_spec or {})
        and slot_tokens & qualifier_tokens
        and not direct_role_tokens & qualifier_tokens
    ):
        return False
    status_tokens = {"state", "status", "condition", "predicate"}
    presence_tokens = {"whether", "remain", "remains", "remaining", "any", "some", "none", "present", "exist", "exists", "left"}
    temporal_tokens = {"time", "window", "date", "schedule", "interval", "weekday", "day", "hour", "deadline"}
    spatial_tokens = {"location", "locations", "destination", "destinations", "zone", "zones", "area", "areas", "room", "rooms", "desk", "desks", "path", "paths", "route", "routes", "place", "places", "bay", "bays", "point", "points"}
    access_tokens = {"code", "pin", "password", "passcode", "token", "credential", "secret", "key", "phrase"}
    access_tokens.update({"badge", "badge_id", "credential_id"})
    expected_slots = _expected_slot_names(attribute, semantic_spec)
    if (
        str((semantic_spec or {}).get("temporal_scope") or "").strip().lower() == "current"
        and _is_collection_attribute(attribute, semantic_spec or {})
        and (
            slot_tokens & {"taken", "intake", "consumed", "consumption"}
            or _meaningful_field_tokens(
                str(((record or {}).get("event_identity") or {}).get("entity_key") or "")
            ) & {"intake", "consumption"}
        )
        and not attr_tokens & {"taken", "intake", "consumed", "consumption"}
    ):
        # A current collection must not reuse an event/intake observation as
        # the current state of the represented object. The distinction is
        # carried by the typed role and event identity, not by a domain name.
        return False
    if (
        str((semantic_spec or {}).get("temporal_scope") or "").strip().lower() == "current"
        and _historical_slot_tokens(slot_name)
        and not _historical_slot_tokens(attribute)
    ):
        return False
    is_summary_request = bool(
        {"summary", "overview", "recap"} & direct_role_tokens
    )
    if (
        _is_collection_attribute(attribute, semantic_spec or {})
        and not is_summary_request
        and slot_tokens == attr_tokens
        and slot_tokens
    ):
        # An outer collection key is a contract label, not a source-local
        # value. A claim-backed component is the exception: its explicit
        # claim boundary makes the source-local field independently
        # auditable, even when its extractor slot has the same normalized
        # label as the outer collection. Without that boundary, accepting the
        # slot would let an aggregate extractor field replace newer component
        # claims and carry unrelated neighboring facts.
        claim_backed_component = any(
            isinstance(claim, dict)
            and str(claim.get("slot_name") or "").strip() == slot_name
            and str(claim.get("claim_span") or "").strip()
            and str(claim.get("claim_span") or "").strip() in str(
                (record or {}).get("source_text") or ""
            )
            for claim in list((record or {}).get("claim_slots") or [])
        )
        if not claim_backed_component:
            return False
    collection_component = bool(
        _is_collection_attribute(attribute, semantic_spec or {})
        and (
            slot_name in expected_slots
            or any(
                isinstance(field, dict)
                and str(field.get("attribute") or "").strip() == attribute
                and str(field.get("slot_name") or field.get("slot") or "").strip() == slot_name
                for field in list((record or {}).get("stage2_typed_fields") or [])
            )
        )
    )
    if (
        slot_tokens & status_tokens
        and not (attr_tokens & status_tokens or attr_tokens & presence_tokens or collection_component)
    ):
        return False
    if (attr_tokens & temporal_tokens and slot_tokens & spatial_tokens) or (
        attr_tokens & spatial_tokens and slot_tokens & temporal_tokens
    ):
        return False
    # Shared words such as ``helper`` do not make a credential/phrase field
    # interchangeable with a time window. Keep the typed role boundary when
    # a source record contains several neighboring operational fields.
    if (attr_tokens & access_tokens and slot_tokens & temporal_tokens) or (
        attr_tokens & temporal_tokens and slot_tokens & access_tokens
    ):
        return False
    # A single shared qualifier is not a typed mapping.  In a source record
    # with ``active_private_badge`` and ``private_review_target``, for example,
    # the word ``private`` is common to both fields but carries no role.  Keep
    # exact semantic-contract slots and genuinely shared field heads, while
    # rejecting weak one-token bridges before Stage 2 fallback can promote
    # them into factual answers.
    if (
        not _is_collection_attribute(attribute, semantic_spec or {})
        and slot_name not in expected_slots
        and _ambiguous_single_token_overlap(attr_tokens, slot_tokens)
    ):
        return False
    if (
        _is_collection_attribute(attribute, semantic_spec or {})
        and attribute in {
            str(item).strip()
            for item in list((record or {}).get("stage2_served_attributes") or [])
        }
        and any(
            str(claim.get("slot_name") or "").strip() == slot_name
            for claim in list((record or {}).get("claim_slots") or [])
        )
    ):
        # Collection components may use open, record-local labels. Preserve
        # them after the role-conflict checks, before scalar alias heuristics
        # can reject the outer collection name.
        return True
    if record is not None:
        structural_tokens = {
            "time", "times", "window", "windows", "range", "interval", "date", "day",
            "hour", "hours", "value", "values", "field", "fields", "item", "items",
        }
        discriminative_attribute_tokens = attr_tokens - structural_tokens
        discriminative_slot_tokens = slot_tokens - structural_tokens
        if discriminative_attribute_tokens and not (
            discriminative_attribute_tokens & discriminative_slot_tokens
        ):
            for observed_slot in dict(record.get("slots") or {}):
                observed_tokens = _meaningful_field_tokens(str(observed_slot)) - structural_tokens
                if discriminative_attribute_tokens & observed_tokens:
                    return False
        if composite_request and _record_has_complete_composite_binding(
            attribute=attribute,
            record=record,
            semantic_spec=semantic_spec or {},
        ):
            # A composite request is allowed to expose several independently
            # typed members from one admitted source record. This exception is
            # source-local and requires multiple fields plus query/source
            # overlap; it cannot authorize a lone neighboring scalar.
            return True
        # A noisy Stage-2 batch can assign one component slot to several
        # attributes. When another observed slot carries more of the
        # requested role, the weaker one is a neighboring field, not an
        # equivalent answer. This catches labels/status helpers without
        # enumerating domain-specific names.
        candidate_role_tokens = direct_role_tokens - structural_tokens
        candidate_overlap = len(candidate_role_tokens & slot_tokens)
        stronger_alternative = any(
            other_name != slot_name
            and len(candidate_role_tokens & _meaningful_field_tokens(str(other_name))) > candidate_overlap
            for other_name in dict(record.get("slots") or {})
        )
        if (
            not _is_collection_attribute(attribute, semantic_spec or {})
            and candidate_overlap <= 1
            and stronger_alternative
        ):
            return False
    if attr_tokens & spatial_tokens and slot_tokens & access_tokens and not (attr_tokens & access_tokens):
        return False
    if (
        _is_collection_attribute(attribute, semantic_spec or {})
        and not slot_tokens & access_tokens
    ):
        # An open collection attribute may contain source-local operational
        # metadata with no lexical overlap with the outer request. Keep those
        # typed components while excluding access artifacts above.
        return True
    if (
        _is_collection_attribute(attribute, semantic_spec or {})
        and attribute in {
            str(item).strip()
            for item in list((record or {}).get("stage2_served_attributes") or [])
        }
        and any(
            str(claim.get("slot_name") or "").strip() == slot_name
            for claim in list((record or {}).get("claim_slots") or [])
        )
    ):
        # Collection contracts may use an open outer name while their source
        # record exposes several role-specific claim labels. Keep those
        # source-local components; the role conflict checks above still block
        # temporal/spatial/access substitutions.
        return True
    # The shared aliases provide typed vocabulary for open attributes whose
    # surface name differs from the canonical state slot.
    for canonical, aliases in CURRENT_STATE_SLOT_ALIASES.items():
        alias_tokens = _meaningful_field_tokens(" ".join([canonical, *aliases]))
        if attr_tokens & alias_tokens:
            binding_tokens.update(alias_tokens)
    is_composite_request = bool(
        {"plan", "schedule", "workflow", "protocol"} & direct_role_tokens
    )
    is_aggregate_slot = bool(
        {"summary", "overview", "recap", "details", "description", "context", "metadata"}
        & slot_tokens
    )
    if (
        _is_collection_attribute(attribute, semantic_spec or {})
        and attribute in {
            str(item).strip()
            for item in list((record or {}).get("stage2_served_attributes") or [])
        }
        and any(
            str(claim.get("slot_name") or "").strip() == slot_name
            for claim in list((record or {}).get("claim_slots") or [])
        )
    ):
        return True
    if is_aggregate_slot and not is_summary_request:
        return False
    observed_value = ""
    if record is not None:
        observed = dict(record.get("slots") or {}).get(slot_name)
        observed_value = str(observed or "")
    if (
        not (attr_tokens & access_tokens)
        and not (slot_tokens & access_tokens)
        and re.search(r"\d", observed_value)
        and _meaningful_field_tokens(observed_value) & access_tokens
    ):
        # A label/date may contain digits, but a compound field that also
        # contains a credential marker cannot satisfy a non-access need.
        return False
    if (
        not is_summary_request
        and len(re.findall(r",|\band\b", observed_value, re.IGNORECASE)) >= 2
        and {"method", "details", "information", "context"} & slot_tokens
    ):
        return False
    # Qualifier/status fields are not route/path values even when their
    # source sentence mentions a handoff. The explicit stage-2 typed field
    # can still carry an open semantic slot; this only rejects a generic
    # condition/status slot that changes the requested field's role.
    if (
        {"path", "route", "handoff"} & (attr_tokens | binding_tokens)
        and {"condition", "status", "allowed", "permission", "availability", "rule"}
        & slot_tokens
    ):
        return False
    if is_summary_request:
        # A summary may use an aggregate slot, or a source-local composite
        # slot explicitly selected for that summary. It must still be a
        # source-grounded candidate, which the caller validates separately.
        return True
    explicit_typed_field = any(
        isinstance(field, dict)
        and str(field.get("attribute") or "").strip() == attribute
        and str(field.get("slot_name") or field.get("slot") or "").strip() == slot_name
        for field in list((record or {}).get("stage2_typed_fields") or [])
    )
    if explicit_typed_field:
        # An explicit Stage-2 binding is trusted only within the requested
        # semantic role.  A composite request may contain time/location
        # components, but a bare component slot must not stand in for the
        # composite property itself.
        if is_composite_request and not (
            slot_tokens & direct_role_tokens or slot_name in _expected_slot_names(attribute, semantic_spec)
        ):
            return False
        # Stage 2 can emit a syntactically valid binding whose slot label is
        # actually a neighboring policy/qualifier field.  Keep open
        # collection labels permissive, but require scalar fields to retain
        # a semantic bridge through the requested attribute, its binding, an
        # expected slot, or the source claim span.  This prevents a broad
        # source record from turning policy prose into a factual answer field.
        if _is_collection_attribute(attribute, semantic_spec or {}):
            return True
        if (
            slot_name in _expected_slot_names(attribute, semantic_spec)
            or attr_tokens & slot_tokens
            or role_binding_tokens & slot_tokens
        ):
            return True
        for claim in list((record or {}).get("claim_slots") or []):
            if str(claim.get("slot_name") or "").strip() != slot_name:
                continue
            claim_tokens = _meaningful_field_tokens(
                str(claim.get("claim_span") or "")
            )
            if claim_tokens & direct_role_tokens:
                return True
        return False
    if record is not None and _record_has_current_attribute_update(
        record,
        attribute=attribute,
    ) and any(
        str(claim.get("slot_name") or "").strip() == slot_name
        for claim in list(record.get("claim_slots") or [])
    ):
        # The update marker and claim context establish the requested role;
        # the caller still requires an exact observed value and source span.
        return True
    if not slot_tokens:
        return False
    if slot_name in expected_slots:
        return True
    overlap = len(attr_tokens & slot_tokens)
    # Alias expansion below is for expected schema slots only. It must not
    # make a field such as ``private_room`` semantically match an unrelated
    # source-local slot merely because ``room`` has a private-room alias.
    binding_overlap = len(role_binding_tokens & slot_tokens)
    if overlap or binding_overlap:
        return True
    # Stage 2 may deliberately keep an open requested attribute while the
    # source annotation uses a domain-neutral typed label (for example, a
    # signoff window represented as an assignee time range). Once the record
    # is explicitly served for that attribute, allow the closed claim label
    # to bridge the naming gap. The role-conflict checks above still reject
    # temporal/spatial and access/non-access substitutions.
    if attribute in {
        str(item).strip()
        for item in list((record or {}).get("stage2_served_attributes") or [])
    } and any(
        str(claim.get("slot_name") or "").strip() == slot_name
        for claim in list((record or {}).get("claim_slots") or [])
    ):
        if _is_collection_attribute(attribute, semantic_spec or {}):
            return True
        claim_tokens = set()
        for claim in list((record or {}).get("claim_slots") or []):
            if str(claim.get("slot_name") or "").strip() == slot_name:
                claim_tokens.update(
                    _meaningful_field_tokens(str(claim.get("claim_span") or ""))
                )
        return bool(
            attr_tokens & slot_tokens
            or role_binding_tokens & slot_tokens
            or claim_tokens & direct_role_tokens
        )
    # A source extractor may expose the property under a different typed
    # label. Permit that only when the record's claim metadata explicitly
    # connects the slot to the requested field; this preserves recall without
    # treating neighboring fields in one sentence as interchangeable.
    for claim in list((record or {}).get("claim_slots") or []):
        if str(claim.get("slot_name") or "").strip() != slot_name:
            continue
        claim_tokens = _meaningful_field_tokens(str(claim.get("claim_span") or ""))
        if attribute in {
            str(item).strip()
            for item in list((record or {}).get("stage2_served_attributes") or [])
        } and claim_tokens and (attr_tokens & claim_tokens):
            return True
    return False


def _paired_operational_role_conflict(attribute: str, slot_name: str) -> bool:
    """Reject opposite members of a typed operational pair."""
    attribute_tokens = _meaningful_field_tokens(attribute)
    slot_tokens = _meaningful_field_tokens(slot_name)
    paired_roles = ({"pickup", "dropoff"}, {"arrival", "departure"}, {"start", "end"})
    return any(
        (attribute_tokens & pair and slot_tokens & pair)
        and (attribute_tokens & pair) != (slot_tokens & pair)
        for pair in paired_roles
    )


def _binding_for_attribute(
    attribute: str, semantic_spec: dict[str, Any] | None
) -> dict[str, Any]:
    for key in ("attribute_bindings", "certifiable_needs"):
        for item in list((semantic_spec or {}).get(key) or []):
            if not isinstance(item, dict):
                continue
            item_attribute = str(item.get("attribute") or item.get("need_id") or "").strip()
            if item_attribute == attribute:
                return item
    return {}


def _meaningful_field_tokens(value: str) -> set[str]:
    """Tokenize field labels while ignoring structural modifiers."""
    tokens = set(re.findall(r"[a-z0-9]+", str(value or "").lower()))
    meaningful = tokens - _FIELD_STOPWORDS - {
        "any", "all", "current", "latest", "opening",
        "active", "available", "for", "in", "of", "to", "where", "used", "use",
        "currently", "currently", "requested", "property",
    }
    # Typed extractors commonly use a singular field label while the query
    # contract uses its plural form (or the reverse).  Keep both forms in the
    # representation so this normalization does not depend on domain terms.
    meaningful.update(
        token[:-1]
        for token in tuple(meaningful)
        if len(token) > 3 and token.endswith("s")
    )
    return meaningful


def _record_has_complete_composite_binding(
    *, attribute: str, record: dict[str, Any], semantic_spec: dict[str, Any]
) -> bool:
    """Check whether one source record contains a complete composite answer."""
    fields = [
        field for field in list(record.get("stage2_typed_fields") or [])
        if isinstance(field, dict)
        and str(field.get("attribute") or "").strip() == str(attribute).strip()
        and str(field.get("slot_name") or "").strip()
        and str(field.get("value") or "").strip()
    ]
    if len({str(field.get("slot_name") or "").strip() for field in fields}) < 2:
        return False
    binding = _binding_for_attribute(attribute, semantic_spec)
    role_text = " ".join((str(attribute), str(binding.get("support_span") or "")))
    role_tokens = _meaningful_field_tokens(role_text) - {"plan", "schedule", "workflow", "protocol", "procedure"}
    source_tokens = _meaningful_field_tokens(str(record.get("source_text") or ""))
    return bool(
        {"plan", "schedule", "workflow", "protocol", "procedure"}
        & _meaningful_field_tokens(role_text)
        and len(role_tokens & source_tokens) >= 2
    )


def _ambiguous_single_token_overlap(left: set[str], right: set[str]) -> bool:
    """Detect a one-token overlap surrounded by different field vocabulary."""
    overlap = left & right
    return bool(
        len(overlap) == 1
        and left - overlap
        and right - overlap
        and (
            overlap <= {
                "current", "active", "private", "public", "approved", "main",
                "primary", "secondary", "latest", "scheduled", "target",
            }
            or overlap & {"secret"}
        )
    )


def _historical_slot_tokens(value: str) -> set[str]:
    return _meaningful_field_tokens(value) & {
        "previous", "prior", "initial", "first", "old", "former", "original",
        "opening", "historical", "retired", "superseded",
    }


def _decision_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    rows = raw.get("decisions")
    return [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []


def _current_group_rank(
    items: list[dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
    *,
    prefer_complete: bool = False,
) -> tuple[int, int, int, str, float, int]:
    """Rank one scalar attribute group by its latest source-grounded claim."""
    best = (0, 0, -1, "", 0.0, 0)
    for item in items:
        record = candidate_by_id.get(str(item.get("memory_id") or ""), {})
        turns = [
            int(match.group(1))
            for source_id in list(record.get("source_message_ids") or [])
            for match in [re.fullmatch(r"t(\d+)", str(source_id).strip())]
            if match
        ]
        turn = max(turns, default=-1)
        timestamp = str(record.get("timestamp") or "")
        value_length = len(str(item.get("value") or ""))
        rank = (
            value_length if prefer_complete else int(turn >= 0),
            int(turn >= 0) if prefer_complete else turn,
            turn if prefer_complete else -1,
            timestamp,
            float(item.get("confidence") or 0.0),
            value_length,
        )
        if rank > best:
            best = rank
    return best
