"""LLM-mediated alignment from query attributes to observed graph slots.

The LLM may select only an existing evidence-local SlotNode.  Authorization,
version selection, and source-span validation remain deterministic.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from gov_mem.graph.governed_graph import GovernedMemoryGraph
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


def align_requested_attributes(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    graph: GovernedMemoryGraph | None,
    owner_id: str | None,
    principal_relation: str | None = None,
    utility_atom_ids: set[str] | None,
    utility_source_message_ids: set[str] | None = None,
    stage2_record_atom_ids_by_attribute: dict[str, set[str]] | None = None,
    stage2_authorized_atom_ids: set[str] | None = None,
    stage2_authorized_atom_ids_by_attribute: dict[str, set[str]] | None = None,
    adjudicated_fields: dict[str, dict[str, str]] | None = None,
    allow_record_local_completion: bool = False,
    require_attested_evidence_span: bool = False,
    llm_client: LLMClient | None,
    model_name: str,
    semantic_contract_certifiable: bool,
) -> dict[str, Any]:
    """Bind requested attributes to observed slots without lexical fallback.

    Direct identifier equality is structural, not semantic inference.  Every
    non-identical alignment requires an available LLM and an evidence-local
    node id returned from a closed candidate set.
    """
    requested = _requested_attributes(semantic_spec)
    if not semantic_contract_certifiable:
        return {"available": False, "reason": "semantic_contract_not_certifiable", "bindings": {}}
    if graph is None:
        return {"available": False, "reason": "graph_unavailable", "bindings": {}}
    if not requested:
        return {"available": False, "reason": "no_requested_attributes", "bindings": {}}

    candidates = _candidate_slots(
        graph=graph,
        owner_id=owner_id,
        principal_relation=principal_relation,
        utility_atom_ids=utility_atom_ids,
        utility_source_message_ids=utility_source_message_ids,
        require_attested_evidence_span=require_attested_evidence_span,
        stage2_authorized_atom_ids=stage2_authorized_atom_ids,
        stage2_authorized_atom_ids_by_attribute=stage2_authorized_atom_ids_by_attribute,
    )
    by_id = {row["slot_node_id"]: row for row in candidates}
    hints = _evidence_slot_hints(semantic_spec)
    bindings: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    # Stage 2 has already made the semantic closed-set record decision. For
    # collection attributes, preserve that decision rather than asking a
    # later slot aligner to silently collapse it to one member.
    stage2_by_attribute = {
        str(attribute): {str(atom_id) for atom_id in atom_ids if str(atom_id)}
        for attribute, atom_ids in dict(stage2_record_atom_ids_by_attribute or {}).items()
    }
    # A requested record collection can be explicitly empty in the selected
    # evidence (for example, a source says no members remain). Let one
    # closed-set LLM pass recognize that source-local state before preserving
    # the usual whole-record collection closure. This avoids replaying every
    # neighboring field of a record merely to express an empty collection.
    stage2_collection_attributes = [
        attribute
        for attribute in requested
        if stage2_by_attribute.get(attribute, set())
        and _is_record_collection_attribute(attribute, semantic_spec)
    ]
    stage2_empty_state_bindings: dict[str, dict[str, Any]] = {}
    if stage2_collection_attributes and llm_client is not None and llm_client.is_available():
        stage2_collection_candidates = [
            row
            for row in candidates
            if any(row["source_atom_id"] in stage2_by_attribute[attribute] for attribute in stage2_collection_attributes)
        ]
        empty_state_raw = _request_alignment(
            llm_client=llm_client,
            model_name=model_name,
            question=question,
            semantic_spec=semantic_spec,
            requested=stage2_collection_attributes,
            candidates=stage2_collection_candidates,
            repair_instruction=(
                "This is an empty-collection check. Return a scalar binding only when a listed source explicitly "
                "states that the requested set has no members, is cleared, is complete, or is otherwise empty. "
                "Select only the exact SlotNode that expresses that source-local state. If the set has one or more "
                "members or the source is not explicit, omit the attribute entirely so record-collection processing "
                "can continue. Never select adjacent credentials, labels, or unrelated fields."
            ),
        )
        stage2_empty_state_bindings, empty_state_rejections = _consume_alignment_items(
            items=_alignment_items(empty_state_raw),
            unresolved=stage2_collection_attributes,
            by_id=by_id,
            question=question,
            semantic_spec=semantic_spec,
        )
        for attribute, binding in list(stage2_empty_state_bindings.items()):
            stage2_empty_state_bindings[attribute] = {
                **binding,
                "source": "llm_explicit_empty_collection_state",
                "binding_kind": "scalar",
            }
        for reason, count in empty_state_rejections.items():
            rejection_counts[reason] += count
    for attribute in requested:
        adjudicated = dict((adjudicated_fields or {}).get(attribute) or {})
        if adjudicated:
            selected = [
                row for row in candidates
                if str(row.get("source_memory_id") or "") == str(adjudicated.get("memory_id") or "")
                and str(row.get("slot_name") or "") == str(adjudicated.get("slot_name") or "")
                and str(row.get("slot_value") or "")
                and (
                    str(row.get("slot_value") or "") == str(adjudicated.get("value") or "")
                    or str(row.get("slot_value") or "") in str(adjudicated.get("value") or "")
                )
            ]
            if selected:
                bindings[attribute] = _binding(
                    attribute=attribute,
                    candidates=selected,
                    source="closed_set_claim_adjudication",
                )
                continue
            if allow_record_local_completion:
                # Stage 2 may name a complete record-level field that the
                # graph compiler decomposed into several typed claim slots.
                # Rebind only inside that exact source atom and source text;
                # authorization still validates every resulting slot later.
                local_record = [
                    row for row in candidates
                    if str(row.get("source_memory_id") or "") == str(adjudicated.get("memory_id") or "")
                    and str(row.get("source_atom_id") or "")
                    and str(row.get("slot_role") or "").strip().lower() == "claim_value"
                    and str(row.get("source_text") or "")
                    and (
                        str(row.get("slot_value") or "") == str(adjudicated.get("value") or "")
                        or str(row.get("slot_value") or "") in str(adjudicated.get("value") or "")
                        or str(adjudicated.get("value") or "") in str(row.get("slot_value") or "")
                    )
                ]
                source_atoms = {str(row.get("source_atom_id") or "") for row in local_record}
                if len(source_atoms) == 1 and local_record:
                    bindings[attribute] = _binding(
                        attribute=attribute,
                        candidates=local_record,
                        source="adjudicated_source_record_typed_completion",
                        binding_kind="scalar",
                    )
                    continue
        # The empty-state probe is only a hint.  For a record collection, a
        # transient scalar response from that probe can collapse a populated
        # source into its first member.  Let the normal closed-set collection
        # projection decide the members; an actually empty collection can
        # still remain unresolved rather than fabricating a member.
        empty_binding = stage2_empty_state_bindings.get(attribute)
        if empty_binding and _binding_source_expresses_empty_state(empty_binding, by_id):
            bindings[attribute] = stage2_empty_state_bindings[attribute]
            continue
        stage2_atom_ids = stage2_by_attribute.get(attribute, set())
        if stage2_atom_ids and _is_record_collection_attribute(attribute, semantic_spec):
            stage2_candidates = [row for row in candidates if row["source_atom_id"] in stage2_atom_ids]
            if stage2_candidates:
                # Stage 2 selects answer records, not every field in those
                # records.  Project each selected record back to the minimum
                # typed SlotNodes needed by this collection request.  The
                # projection is one closed-set LLM decision for all selected
                # records; deterministic code validates IDs and source spans.
                projected_raw = _request_alignment(
                    llm_client=llm_client,
                    model_name=model_name,
                    question=question,
                    semantic_spec=semantic_spec,
                    requested=[attribute],
                    candidates=stage2_candidates,
                    repair_instruction=(
                        "Stage 2 already selected these source records as the closed answer set. "
                        "Now perform a typed slot projection inside those records. Select only the "
                        "existing SlotNode IDs whose grounded values are the requested collection "
                        "members or the minimal source-grounded state needed to express that collection. "
                        "Do not select credentials, identifiers, labels, metadata, or unrelated sibling "
                        "fields merely because they share the same source record. Prefer claim_value and "
                        "claim_subject_value slots when their claim metadata directly represents the "
                        "requested collection. Return binding_kind=collection and copy exact source spans."
                    ),
                )
                projected, projection_rejections = _consume_alignment_items(
                    items=_alignment_items(projected_raw),
                    unresolved=[attribute],
                    by_id=by_id,
                    question=question,
                    semantic_spec=semantic_spec,
                )
                for reason, count in projection_rejections.items():
                    rejection_counts[reason] += count
                projected_candidates = [
                    by_id[node_id]
                    for node_id in list((projected.get(attribute) or {}).get("anchor_slot_node_ids") or [])
                    if node_id in by_id
                ]
                if projected_candidates:
                    bindings[attribute] = _binding(
                        attribute=attribute,
                        candidates=projected_candidates,
                        source="stage2_closed_record_slot_projection",
                        binding_kind="collection",
                    )
                else:
                    rejection_counts["stage2_collection_slot_projection_unavailable"] += 1
                continue
        hinted_slot = hints.get(attribute)
        exact = [row for row in candidates if row["slot_name"] == (hinted_slot or attribute)]
        # For a current-state question, a schema-name match can point to the
        # prior value inside a source that also explicitly states its
        # replacement. Let the LLM choose from the closed source slots rather
        # than promoting that accidental field-name match to a certificate.
        force_current_semantic_selection = str(semantic_spec.get("temporal_scope") or "").lower() == "current"
        # Equality is a structural shortcut only when it identifies one
        # evidence occurrence.  Multiple candidates require semantic entity/
        # record resolution rather than arbitrary iteration order.
        if force_current_semantic_selection or len(exact) != 1:
            unresolved.append(attribute)
            rejection_counts["no_unique_structural_slot_match"] += 1
            continue
        bindings[attribute] = _binding(
            attribute=attribute,
            candidates=exact,
            source="planner_evidence_slot_hint" if hinted_slot else "exact_slot_identifier",
        )

    if unresolved:
        if llm_client is None or not llm_client.is_available():
            return {
                "available": False,
                "reason": "semantic_alignment_llm_unavailable",
                "bindings": bindings,
                "unresolved_attributes": unresolved,
            }
        raw = _request_alignment(
            llm_client=llm_client,
            model_name=model_name,
            question=question,
            semantic_spec=semantic_spec,
            requested=unresolved,
            candidates=_candidates_for_unresolved_attributes(
                candidates=candidates, attributes=unresolved, hints=hints
            ),
        )
        parsed_items = _alignment_items(raw)
        accepted, rejected = _consume_alignment_items(
            items=parsed_items,
            unresolved=unresolved,
            by_id=by_id,
            question=question,
            semantic_spec=semantic_spec,
        )
        bindings.update(accepted)
        # A scalar SlotNode is one observed field occurrence.  Reusing it for
        # two separately requested properties loses the attribute/value
        # association even though both bindings have valid source spans. Keep
        # the first request-order binding and send the conflicting properties
        # through the existing closed-set repair call.
        duplicate_scalar_attributes = _duplicate_scalar_binding_attributes(
            bindings=bindings,
            requested=requested,
        )
        for attribute in duplicate_scalar_attributes:
            bindings.pop(attribute, None)
        if duplicate_scalar_attributes:
            rejection_counts["duplicate_scalar_slot_across_attributes"] += len(duplicate_scalar_attributes)
        for reason, count in rejected.items():
            rejection_counts[reason] += count
        missing_after_first = [attribute for attribute in unresolved if attribute not in bindings]
        # Audit all selected scalar claims in one closed-set call.  A typed
        # record may expose several slots for one claim, and the first pass
        # can select a slot that describes the claim's state rather than the
        # value requested by the query.  The audit uses the complete source
        # atom, but never broadens the candidate set to another record.
        selected_scalar_attributes = [
            attribute
            for attribute in requested
            if attribute in bindings
            and str((bindings.get(attribute) or {}).get("binding_kind") or "scalar") != "collection"
        ]
        claim_audit_candidates = _selected_scalar_claim_candidates(
            candidates=candidates,
            bindings=bindings,
            attributes=selected_scalar_attributes,
        )
        claim_audit_attributes = [
            attribute for attribute in selected_scalar_attributes
            if len([
                candidate for candidate in claim_audit_candidates
                if str(candidate.get("source_atom_id") or "") == _binding_source_atom_id(
                    binding=bindings.get(attribute) or {}, candidates=candidates
                )
            ]) > 1
        ]
        repair_attributes = list(dict.fromkeys([
            *missing_after_first,
            *claim_audit_attributes,
        ]))
        repair_trace: dict[str, Any] = {"attempted": False}
        if repair_attributes:
            repair_trace["attempted"] = True
            repair_candidates = _candidates_for_unresolved_attributes(
                candidates=candidates, attributes=missing_after_first, hints=hints
            )
            if claim_audit_attributes:
                repair_candidates = _merge_candidate_sets(
                    repair_candidates,
                    claim_audit_candidates,
                )
            if missing_after_first and not claim_audit_attributes:
                repair_candidates = _merge_candidate_sets(
                    repair_candidates,
                    _source_local_candidates_for_bindings(
                        candidates=candidates,
                        bindings=bindings,
                        attributes=missing_after_first,
                    ),
                )
            repair_raw = _request_alignment(
                llm_client=llm_client,
                model_name=model_name,
                question=question,
                semantic_spec=semantic_spec,
                requested=repair_attributes,
                candidates=repair_candidates,
                repair_instruction=(
                    "Re-evaluate each requested attribute against the complete typed claim(s) listed here. A prior "
                    "proposal may have been empty, failed validation, or selected a contextual/state slot instead of "
                    "the value slot that realizes the requested property. Select only listed SlotNode IDs whose "
                    "values directly and completely answer the corresponding query attribute; preserve the source "
                    "atom and do not combine unrelated claims. For every selected ID copy a fact_support_span verbatim from that "
                    "candidate's source_text that contains its exact slot_value. Do not change the query attribute "
                    "or infer a value."
                ),
            )
            repair_items = _alignment_items(repair_raw)
            repaired, repair_rejections = _consume_alignment_items(
                items=repair_items,
                unresolved=repair_attributes,
                by_id=by_id,
                question=question,
                semantic_spec=semantic_spec,
            )
            bindings.update(repaired)
            for reason, count in repair_rejections.items():
                rejection_counts[reason] += count
            repair_trace.update({
                "raw_top_level_keys": sorted(str(key) for key in repair_raw.keys()) if isinstance(repair_raw, dict) else [],
                "parsed_binding_count": len(repair_items),
                "accepted_binding_count": len(repaired),
                "rejection_counts": dict(sorted(repair_rejections.items())),
                "claim_audit_attributes": claim_audit_attributes,
                "claim_audit_candidate_count": len(claim_audit_candidates),
            })
        # Open-schema record collections cannot depend on a planner guessing
        # the producer's field labels. When slot alignment remains empty, ask
        # for closed source-record IDs; deterministic code expands only the
        # already observed slots belonging to each selected record.
        collection_fallback_attributes = [
            attribute for attribute in missing_after_first
            if _is_record_collection_attribute(attribute, semantic_spec)
            and attribute not in bindings
        ]
        for attribute in collection_fallback_attributes:
            record_raw = _request_collection_record_selection(
                llm_client=llm_client,
                model_name=model_name,
                question=question,
                semantic_spec=semantic_spec,
                attribute=attribute,
                candidates=candidates,
            )
            selected_records, record_rejections = _consume_collection_membership(
                raw=record_raw,
                attribute=attribute,
                candidates=candidates,
                question=question,
                semantic_spec=semantic_spec,
            )
            for reason, count in record_rejections.items():
                rejection_counts[reason] += count
            record_ids = selected_records.get(attribute, set())
            if not record_ids:
                continue
            record_candidates = [
                candidate for candidate in candidates
                if candidate["source_atom_id"] in record_ids
            ]
            if record_candidates:
                bindings[attribute] = _binding(
                    attribute=attribute,
                    candidates=record_candidates,
                    source="llm_collection_record_alignment",
                    binding_kind="collection",
                )
    else:
        parsed_items = []
        repair_trace = {"attempted": False}

    collection_completion: dict[str, Any] = {"attempted": False, "completed_attributes": []}
    # A collection alignment is incomplete until the base model has audited
    # the complete closed record set. The audit owns semantic conditions such
    # as ordering, state, and whether a record merely mentions another record;
    # code only verifies IDs and verbatim provenance.
    for attribute, binding in list(bindings.items()):
        if str(binding.get("source") or "") == "llm_explicit_empty_collection_state":
            continue
        is_collection = (
            str(binding.get("binding_kind") or "") == "collection"
            or _is_record_collection_attribute(attribute, semantic_spec)
        )
        if not is_collection:
            continue
        if str(binding.get("binding_kind") or "") != "collection":
            binding = {**binding, "binding_kind": "collection"}
            bindings[attribute] = binding
        # A Stage-2 collection is already a complete closed record decision:
        # every active candidate was classified there. A later slot-level
        # audit may diagnose it, but must never overwrite or narrow that
        # certified record closure.
        if str(binding.get("source") or "") in {
            "stage2_closed_record_selection",
            "stage2_closed_record_slot_projection",
        }:
            collection_completion["attempted"] = True
            collection_completion["completed_attributes"].append(attribute)
            collection_completion.setdefault("stage2_closed_record_sets", {})[attribute] = {
                "record_count": len({row["source_atom_id"] for row in candidates if row["slot_node_id"] in set(binding.get("anchor_slot_node_ids") or [])}),
                "projected_slot_count": len(list(binding.get("anchor_slot_node_ids") or [])),
            }
            continue
        if llm_client is not None and llm_client.is_available():
            audit_raw = _request_collection_record_selection(
                llm_client=llm_client,
                model_name=model_name,
                question=question,
                semantic_spec=semantic_spec,
                attribute=attribute,
                candidates=candidates,
                collection_audit=True,
            )
            audited, audit_rejections = _consume_collection_membership(
                raw=audit_raw,
                attribute=attribute,
                candidates=candidates,
                question=question,
                semantic_spec=semantic_spec,
                require_complete_audit=True,
            )
            for reason, count in audit_rejections.items():
                rejection_counts[reason] += count
            audited_record_ids = audited.get(attribute, set())
            collection_completion["attempted"] = True
            collection_completion.setdefault("full_set_audits", {})[attribute] = {
                "selected_record_count": len(audited_record_ids),
                "rejection_counts": dict(sorted(audit_rejections.items())),
            }
            initial_record_ids = {
                by_id[node_id]["source_atom_id"]
                for node_id in list(binding.get("anchor_slot_node_ids") or [])
                if node_id in by_id
            }
            incumbent_record_ids = audited_record_ids or initial_record_ids
            if incumbent_record_ids:
                review_raw = _request_collection_record_selection(
                    llm_client=llm_client,
                    model_name=model_name,
                    question=question,
                    semantic_spec=semantic_spec,
                    attribute=attribute,
                    candidates=candidates,
                    collection_audit=True,
                    incumbent_record_ids=incumbent_record_ids,
                )
                reviewed, review_rejections = _consume_collection_membership(
                    raw=review_raw,
                    attribute=attribute,
                    candidates=candidates,
                    question=question,
                    semantic_spec=semantic_spec,
                    require_complete_audit=True,
                )
                reviewed_record_ids = reviewed.get(attribute, set())
                for reason, count in review_rejections.items():
                    rejection_counts[reason] += count
                final_record_ids = reviewed_record_ids or incumbent_record_ids
                bindings[attribute] = _binding(
                    attribute=attribute,
                    candidates=[
                        row for row in candidates
                        if row["source_atom_id"] in final_record_ids
                    ],
                    source="llm_collection_full_set_audit_with_reviewer",
                    binding_kind="collection",
                )
                collection_completion["full_set_audits"][attribute].update({
                    "review_selected_record_count": len(reviewed_record_ids),
                    "review_rejection_counts": dict(sorted(review_rejections.items())),
                    "review_used_initial_incumbent": not bool(audited_record_ids),
                })
                collection_completion["completed_attributes"].append(attribute)
                continue
        selected_ids = {str(value) for value in list(binding.get("anchor_slot_node_ids") or []) if str(value)}
        # Membership is a decision about records, not independently about
        # their fields. Once any field of a record is selected, retain that
        # record as-is and ask the completion pass only about other records.
        selected_record_ids = {
            by_id[node_id]["source_atom_id"]
            for node_id in selected_ids
            if node_id in by_id
        }
        remaining = [
            candidate for candidate in candidates
            if candidate["source_atom_id"] not in selected_record_ids
        ]
        if not remaining or llm_client is None or not llm_client.is_available():
            continue
        collection_completion["attempted"] = True
        completion_raw = _request_collection_record_selection(
            llm_client=llm_client,
            model_name=model_name,
            question=question,
            semantic_spec=semantic_spec,
            attribute=attribute,
            candidates=remaining,
            incumbent_record_ids=selected_record_ids,
        )
        completed, completion_rejections = _consume_collection_membership(
            raw=completion_raw,
            attribute=attribute,
            candidates=remaining,
            question=question,
            semantic_spec=semantic_spec,
        )
        for reason, count in completion_rejections.items():
            rejection_counts[reason] += count
        extra_records = completed.get(attribute, set())
        if extra_records:
            merged_ids = list(dict.fromkeys([
                *list(binding.get("anchor_slot_node_ids") or []),
                *[
                    candidate["slot_node_id"]
                    for candidate in remaining
                    if candidate["source_atom_id"] in extra_records
                ],
            ]))
            merged_candidates = [by_id[node_id] for node_id in merged_ids if node_id in by_id]
            bindings[attribute] = _binding(
                attribute=attribute,
                candidates=merged_candidates,
                source="llm_collection_completion",
                binding_kind="collection",
            )
            collection_completion["completed_attributes"].append(attribute)
    collection_completion["completed_attributes"] = list(dict.fromkeys(collection_completion["completed_attributes"]))

    # A collection-membership audit decides which records belong to a bundle,
    # but it can fail on a large closed candidate set even when the typed
    # fields are present. Recover the field/value association in one grouped
    # projection call instead of letting the first slot of an incumbent record
    # stand in for every requested attribute.
    collection_projection_attributes = [
        attribute
        for attribute in requested
        if _is_record_collection_attribute(attribute, semantic_spec)
        and str((bindings.get(attribute) or {}).get("source") or "")
        not in {
            "llm_explicit_empty_collection_state",
            "stage2_closed_record_selection",
            "stage2_closed_record_slot_projection",
        }
    ]
    if collection_projection_attributes and llm_client is not None and llm_client.is_available():
        projection_raw = _request_alignment(
            llm_client=llm_client,
            model_name=model_name,
            question=question,
            semantic_spec=semantic_spec,
            requested=collection_projection_attributes,
            candidates=candidates,
            repair_instruction=(
                "Use one grouped typed projection over the closed candidate set. For each requested attribute, "
                "select only existing SlotNode IDs whose typed values directly realize that attribute. Include "
                "multiple IDs only when the attribute genuinely needs multiple grounded members. Prefer slots "
                "with slot_role=claim_value or an explicit typed field over claim_subject_value, date anchors, "
                "or meta/boundary slots. Do not reuse one generic slot for unrelated attributes. Preserve exact "
                "source spans for every selected slot and return binding_kind=collection."
            ),
        )
        projected, projection_rejections = _consume_alignment_items(
            items=_alignment_items(projection_raw),
            unresolved=collection_projection_attributes,
            by_id=by_id,
            question=question,
            semantic_spec=semantic_spec,
        )
        for reason, count in projection_rejections.items():
            rejection_counts[reason] += count
        for attribute, projected_binding in projected.items():
            selected_ids = list(projected_binding.get("anchor_slot_node_ids") or [])
            selected_rows = [by_id[node_id] for node_id in selected_ids if node_id in by_id]
            direct_role_rows = [
                row for row in selected_rows
                if str(row.get("slot_role") or "").strip().lower() not in {
                    "claim_subject_value", "claim_subject", "subject",
                }
            ]
            # Subject/entity slots are useful only when no typed value slot
            # was selected for that attribute. If the model supplied both,
            # retain the direct value projection and drop the subject alias.
            if direct_role_rows and len(direct_role_rows) != len(selected_rows):
                projected_binding = _binding(
                    attribute=attribute,
                    candidates=direct_role_rows,
                    source="llm_grouped_collection_typed_projection",
                    binding_kind="collection",
                )
            else:
                projected_binding = {
                    **projected_binding,
                    "source": "llm_grouped_collection_typed_projection",
                    "binding_kind": "collection",
                }
            bindings[attribute] = projected_binding

    missing = [attribute for attribute in requested if attribute not in bindings]
    return {
        "available": not missing,
        "reason": "all_requested_attributes_aligned" if not missing else "unresolved_semantic_attributes",
        "bindings": bindings,
        "unresolved_attributes": missing,
        "candidate_count": len(candidates),
        "diagnostics": {
            "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if unresolved and isinstance(raw, dict) else [],
            "parsed_binding_count": len(parsed_items),
            "accepted_binding_count": len(bindings),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "fact_span_repair": repair_trace,
            "collection_completion": collection_completion,
            "stage2_empty_state_attributes": sorted(stage2_empty_state_bindings),
        },
    }


def _source_local_candidates_for_bindings(
    *,
    candidates: list[dict[str, str]],
    bindings: dict[str, dict[str, Any]],
    attributes: list[str],
) -> list[dict[str, str]]:
    anchor_ids = {
        str((bindings.get(attribute) or {}).get("anchor_slot_node_id") or "")
        for attribute in attributes
    }
    source_ids = {
        str(candidate.get("source_atom_id") or "")
        for candidate in candidates
        if str(candidate.get("slot_node_id") or "") in anchor_ids
    }
    return [candidate for candidate in candidates if str(candidate.get("source_atom_id") or "") in source_ids]


def _binding_source_atom_id(*, binding: dict[str, Any], candidates: list[dict[str, str]]) -> str:
    anchor_id = str(binding.get("anchor_slot_node_id") or "")
    return next(
        (
            str(candidate.get("source_atom_id") or "")
            for candidate in candidates
            if str(candidate.get("slot_node_id") or "") == anchor_id
        ),
        "",
    )


def _selected_scalar_claim_candidates(
    *,
    candidates: list[dict[str, str]],
    bindings: dict[str, dict[str, Any]],
    attributes: list[str],
) -> list[dict[str, str]]:
    """Return complete typed claims for selected scalar anchors.

    The closure is structural: it follows the selected SlotNode to its
    source atom and includes only sibling slots from that exact atom.  No
    lexical property or dataset vocabulary is used here.
    """
    anchor_ids = {
        str((bindings.get(attribute) or {}).get("anchor_slot_node_id") or "")
        for attribute in attributes
    }
    source_ids = {
        str(candidate.get("source_atom_id") or "")
        for candidate in candidates
        if str(candidate.get("slot_node_id") or "") in anchor_ids
    }
    return [
        candidate for candidate in candidates
        if str(candidate.get("source_atom_id") or "") in source_ids
    ]


def _merge_candidate_sets(*candidate_sets: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate_set in candidate_sets:
        for candidate in candidate_set:
            node_id = str(candidate.get("slot_node_id") or "")
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            merged.append(candidate)
    return merged


def _duplicate_scalar_binding_attributes(
    *,
    bindings: dict[str, dict[str, Any]],
    requested: list[str],
) -> list[str]:
    """Return later scalar properties that reuse an earlier evidence slot.

    This is an integrity constraint over closed SlotNode IDs, not an attempt
    to infer whether two natural-language fields have similar names. A
    collection may intentionally include a record's multiple slots, but two
    independently requested scalar properties need independently selected
    evidence occurrences.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for attribute in requested:
        binding = bindings.get(attribute)
        if not binding or str(binding.get("binding_kind") or "scalar") == "collection":
            continue
        anchor_id = str(binding.get("anchor_slot_node_id") or "")
        if not anchor_id:
            continue
        if anchor_id in seen:
            duplicates.append(attribute)
            continue
        seen.add(anchor_id)
    return duplicates


def _normalize_value_for_containment(value: str) -> str:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return re.sub(r"^(?:the|a|an)\s+", "", normalized)


def _contains_normalized_value(container: str, value: str) -> bool:
    if not container or not value or len(container) <= len(value):
        return False
    return re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", container) is not None


def _alignment_items(raw: object) -> list[dict[str, Any]]:
    """Normalize equivalent LLM alignment envelopes before strict checking."""
    if not isinstance(raw, dict):
        return []
    containers: list[object] = [
        raw.get("bindings"),
        raw.get("attribute_bindings"),
        (raw.get("semantic_alignment") or {}).get("bindings")
        if isinstance(raw.get("semantic_alignment"), dict)
        else None,
    ]
    rows: list[dict[str, Any]] = []
    for container in containers:
        if isinstance(container, dict):
            container = list(container.values())
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict) and item not in rows:
                rows.append(item)
    return rows


def _consume_alignment_items(
    *,
    items: list[dict[str, Any]],
    unresolved: list[str],
    by_id: dict[str, dict[str, str]],
    question: str,
    semantic_spec: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    bindings: dict[str, dict[str, Any]] = {}
    rejection_counts: dict[str, int] = defaultdict(int)
    for item in items:
        attribute = str(item.get("attribute") or "").strip()
        binding_kind = _binding_kind(item)
        selected_ids = [str(value) for value in list(item.get("slot_node_ids") or []) if str(value)]
        if not selected_ids and str(item.get("slot_node_id") or ""):
            selected_ids = [str(item.get("slot_node_id") or "")]
        selected = [by_id[node_id] for node_id in selected_ids if node_id in by_id]
        if attribute not in unresolved or not selected or len(selected) != len(set(selected_ids)):
            rejection_counts["unknown_attribute_or_slot_node"] += 1
            continue
        request_shape = str(semantic_spec.get("request_shape") or "fact").lower()
        if (
            request_shape not in {"list", "plan", "mixed"}
            and binding_kind != "collection"
            and len({candidate["source_atom_id"] for candidate in selected}) != 1
        ):
            rejection_counts["scalar_cross_claim_assembly"] += 1
            continue
        query_span = str(item.get("query_support_span") or "").strip()
        if not _query_span_is_grounded(
            attribute, query_span, question, semantic_spec, allow_collection_variation=binding_kind == "collection"
        ):
            rejection_counts["query_span_not_contract_grounded"] += 1
            continue
        spans_by_id = {
            str(entry.get("slot_node_id") or ""): str(entry.get("fact_support_span") or "").strip()
            for entry in list(item.get("fact_support_spans") or [])
            if isinstance(entry, dict)
        }
        spans_by_source_atom = {
            str(entry.get("source_atom_id") or ""): str(entry.get("fact_support_span") or "").strip()
            for entry in list(item.get("fact_support_spans") or [])
            if isinstance(entry, dict)
        }
        fallback_span = str(item.get("fact_support_span") or "").strip()
        if any(
            not (span := spans_by_id.get(
                candidate["slot_node_id"], spans_by_source_atom.get(candidate["source_atom_id"], fallback_span)
            ))
            or span not in candidate["source_text"]
            or candidate["slot_value"] not in span
            for candidate in selected
        ):
            rejection_counts["fact_span_not_source_grounded"] += 1
            continue
        bindings[attribute] = _binding(
            attribute=attribute, candidates=selected, source="llm_evidence_alignment", binding_kind=binding_kind
        )
    return bindings, dict(rejection_counts)


def _candidates_for_unresolved_attributes(
    *, candidates: list[dict[str, str]], attributes: list[str], hints: dict[str, str]
) -> list[dict[str, str]]:
    """Narrow an LLM alignment call with planner-declared evidence fields."""
    hinted_names = {hints[attribute] for attribute in attributes if hints.get(attribute)}
    if not hinted_names:
        return candidates
    narrowed = [row for row in candidates if row["slot_name"] in hinted_names]
    return narrowed or candidates


def _request_alignment(
    *,
    llm_client: LLMClient,
    model_name: str,
    question: str,
    semantic_spec: dict[str, Any],
    requested: list[str],
    candidates: list[dict[str, str]],
    repair_instruction: str = "",
) -> dict[str, Any]:
    # Preserve claim context instead of presenting duplicate source text once
    # per field. The LLM still selects evidence-local SlotNode IDs, while the
    # certificate independently validates every selected field afterwards.
    claims: dict[str, dict[str, Any]] = {}
    for row in candidates:
        claim = claims.setdefault(str(row["source_atom_id"]), {
            "source_atom_id": row["source_atom_id"],
            "source_text": row["source_text"],
            "slots": [],
            "claim_subjects": [],
        })
        claim["slots"].append({
            "slot_node_id": row["slot_node_id"],
            "slot_name": row["slot_name"],
            "slot_value": row["slot_value"],
            "slot_role": row.get("slot_role", ""),
            "claim_property_label": row.get("claim_property_label", ""),
            "claim_property_labels": list(row.get("claim_property_labels") or []),
            "claim_spans": list(row.get("claim_spans") or []),
            "claim_value_spans": list(row.get("claim_value_spans") or []),
        })
        for subject in row.get("claim_subjects") or []:
            if subject not in claim["claim_subjects"]:
                claim["claim_subjects"].append(subject)
    payload = list(claims.values())
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You align a user's requested semantic attributes with already observed memory evidence. "
                "Do not answer the user, infer a value, create an attribute, or authorize disclosure. Return JSON only."
            ),
            user_prompt=(
                "Return {\"bindings\":[{\"attribute\":string,\"slot_node_ids\":[string],"
                "\"binding_kind\":\"scalar|collection\",\"query_support_span\":string,\"fact_support_spans\":[{\"slot_node_id\":string,\"source_atom_id\":string,"
                "\"fact_support_span\":string}]}]}. Work only from the closed ClaimCandidates. "
                "For each scalar attribute, first compare the meaning of the requested property with each candidate's "
                "complete source_text and claim metadata, then choose the one SlotNode whose value most directly and "
                "completely realizes it. A slot marked slot_role=claim_value is the value span of an explicitly "
                "grounded claim; prefer it when that claim states the requested property. A slot without claim_value "
                "role may still be selected when the source has no claim linkage, but a state, qualifier, or confirmation "
                "phrase must not replace a separately observed claim value in the same source claim. "
                "A claim_subject_value slot is a source-grounded value in the subject position of the listed claim; "
                "select it when that subject is the value requested by the query, and do not treat the predicate slot "
                "alone as the answer merely because its property label resembles the query attribute. "
                "A source claim that directly states the requested property is stronger than a neighboring or "
                "synonymous field whose name happens to resemble the attribute. "
                "When multiple current candidates support one property, preserve the fullest source-grounded operational "
                "value. Do not truncate a composite term, plan, scope, structure, or enumerated value to a shorter prefix "
                "when a listed candidate directly contains the complete requested value. Semantic attribute names need not "
                "match slot names; source claim meaning controls selection. "
                "For fact/comparison requests, scalar SlotNode IDs must originate from one ClaimCandidate. Use multiple "
                "SlotNode IDs only when that single claim intrinsically has a composite value needed for the one property. "
                "Each separately requested scalar property must select its own SlotNode ID; never reuse a scalar SlotNode "
                "for two distinct properties. Do not add neighboring identity, interpretation, or context fields. "
                "For list/plan/mixed requests, or for a named snapshot/summary/status whose fields jointly constitute one answer, "
                "select every necessary record candidate and set binding_kind=collection when "
                "the question asks for a set, a record collection, or a named recap/status whose fields jointly constitute "
                "that named item. Never return an arbitrary single member when another visible candidate answers the same "
                "collection request. "
                "query_support_span must be copied exactly from Question. Each fact_support_span must be copied "
                "exactly from its candidate's source_text and must contain that candidate value. Identify each span "
                "by its slot_node_id; source_atom_id is also allowed when one exact source span supports one or more "
                "selected slots from that same claim. Omit uncertain "
                f"alignments. {repair_instruction}\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Requested attributes: {requested}\nCandidates: {payload}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _request_collection_record_selection(
    *,
    llm_client: LLMClient,
    model_name: str,
    question: str,
    semantic_spec: dict[str, Any],
    attribute: str,
    candidates: list[dict[str, str]],
    collection_audit: bool = False,
    incumbent_record_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Select collection members by closed record IDs when field names vary."""
    claims: dict[str, dict[str, Any]] = {}
    for row in candidates:
        claim = claims.setdefault(str(row["source_atom_id"]), {
            "source_atom_id": row["source_atom_id"],
            "source_text": row["source_text"],
            "slots": [],
        })
        claim["slots"].append({
            "slot_node_id": row["slot_node_id"],
            "slot_name": row["slot_name"],
            "slot_value": row["slot_value"],
        })
        claim["timestamp"] = str(row.get("timestamp") or "")
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You identify members of a requested record collection using only listed evidence claims. "
                "Do not answer the user, create facts, infer fields, or authorize disclosure. Return JSON only."
            ),
            user_prompt=(
                "Return {\"audit\":[{\"source_atom_id\":string,\"classification\":\"answer_member|reference_boundary|"
                "mere_mention|duplicate_or_corroboration|unrelated\",\"reason\":string}],\"bindings\":[{\"attribute\":string,\"binding_kind\":\"collection\","
                "\"source_atom_ids\":[string],\"query_support_span\":string,"
                "\"fact_support_spans\":[{\"source_atom_id\":string,\"fact_support_span\":string}]}]}. Select all and only ClaimCandidates that are members of the "
                "requested collection. This is record selection, so do not rely on field-name similarity. Copy each "
                "source_atom_id exactly and copy that candidate's complete source_text as fact_support_span. "
                + (
                    "This is a completeness audit over the entire closed candidate set. Interpret every condition "
                    "in the question yourself, including ordering, state, inclusion, exclusion, and reference records. "
                    "For a current or final snapshot/state request, treat an explicit final/current/post-deletion "
                    "snapshot as a canonical state record when it names the requested operation or thread. If the "
                    "answer spans several named objects or lanes, include the complementary current records that "
                    "supply those objects' independently typed fields; do not keep only the single record with the "
                    "largest text span. A record can be a qualifying member even when its field labels differ from "
                    "the query, provided its source text explicitly states a current answer component. "
                    "For utility requests, assess membership from the current operational object and typed field "
                    "content, not from the speaker's role or from whether that record repeats the requester's role. "
                    "A record attributed to a different operational participant can still supply a complementary "
                    "field for the same current operation when the closed source context connects them; do not "
                    "discard such a record solely because another record is more directly addressed to the requester. "
                    "The closed source context may establish the lane through neighboring turns, chronology, and "
                    "current-state updates, so a qualifying record need not repeat the object name in its own sentence. "
                    "For a named operational snapshot, include non-conflicting current metadata records from that "
                    "same lane when they supply a typed component of the requested snapshot; the later slot-level "
                    "certificate still excludes fields whose role is incompatible or whose disclosure path is absent. "
                    "If a candidate mixes one directly requested safe component with unrelated credentials, retired "
                    "values, or private neighboring fields, classify it as a member for that safe component and let "
                    "the later typed projection omit only the unrelated fields; do not reject the whole record. "
                    "First populate audit with exactly one classification for every ClaimCandidate. Then return only "
                    "the qualifying answer_record IDs in bindings. A reference/boundary record is not an answer record "
                    "unless the question explicitly asks "
                    "for it, and a record that merely mentions another record is not thereby a member. When multiple "
                    "claims describe the same qualifying item, return only the single canonical claim that most directly "
                    "and completely states it; do not return its weaker recap or corroboration as an extra item. Check "
                    "that every qualifying record was included before responding. Do not use field-name similarity as a "
                    "proxy for meaning. "
                    if collection_audit else ""
                )
                + (
                    "This is a reviewer pass. The incumbent answer record IDs are listed below but may be incomplete "
                    "or wrong. Independently re-evaluate every ClaimCandidate, then return the complete replacement set "
                    "of canonical answer members. Prefer the candidate that most directly and completely states each "
                    "answer item under the question's current-state conditions; do not preserve an incumbent merely "
                    "because it was proposed first. "
                    if incumbent_record_ids is not None else ""
                )
                + "In every binding, query_support_span must be copied exactly from the Semantic contract's "
                "attribute support span for this Attribute. Do not use an ordering or boundary phrase in that field; "
                "use the audit classification to express boundary semantics. Return bindings=[] if no listed claim "
                "is a member.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\nAttribute: {attribute}\n"
                f"Incumbent record IDs: {sorted(incumbent_record_ids or set())}\n"
                f"ClaimCandidates: {list(claims.values())}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _consume_collection_membership(
    *,
    raw: object,
    attribute: str,
    candidates: list[dict[str, str]],
    question: str,
    semantic_spec: dict[str, Any],
    require_complete_audit: bool = False,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Validate record-level collection membership from a closed candidate set."""
    candidate_by_atom = {
        str(candidate["source_atom_id"]): candidate
        for candidate in candidates
    }
    accepted: set[str] = set()
    rejection_counts: dict[str, int] = defaultdict(int)
    candidate_ids = set(candidate_by_atom)
    audit_members: set[str] | None = None
    if require_complete_audit:
        classifications = {
            "answer_member",
            "reference_boundary",
            "mere_mention",
            "duplicate_or_corroboration",
            "unrelated",
        }
        audit_rows = raw.get("audit") if isinstance(raw, dict) else None
        if not isinstance(audit_rows, list):
            return {}, {"missing_collection_audit": 1}
        audit_by_id: dict[str, str] = {}
        for row in audit_rows:
            if not isinstance(row, dict):
                return {}, {"invalid_collection_audit_row": 1}
            record_id = str(row.get("source_atom_id") or "")
            classification = str(row.get("classification") or "")
            if record_id in audit_by_id or record_id not in candidate_ids or classification not in classifications:
                return {}, {"invalid_collection_audit_row": 1}
            audit_by_id[record_id] = classification
        if set(audit_by_id) != candidate_ids:
            return {}, {"incomplete_collection_audit": 1}
        audit_members = {
            record_id for record_id, classification in audit_by_id.items()
            if classification == "answer_member"
        }
    for item in _alignment_items(raw):
        item_attribute = str(item.get("attribute") or "").strip()
        member_ids = [str(value) for value in list(item.get("source_atom_ids") or []) if str(value)]
        if item_attribute != attribute or _binding_kind(item) != "collection" or not member_ids:
            rejection_counts["invalid_collection_membership_shape"] += 1
            continue
        if (
            len(member_ids) != len(set(member_ids))
            or any(member_id not in candidate_by_atom for member_id in member_ids)
        ):
            rejection_counts["unknown_collection_record"] += 1
            continue
        query_span = str(item.get("query_support_span") or "").strip()
        if not _query_span_is_grounded(
            attribute, query_span, question, semantic_spec,
            allow_collection_variation=True,
        ):
            rejection_counts["query_span_not_contract_grounded"] += 1
            continue
        spans_by_atom = {
            str(entry.get("source_atom_id") or ""): str(entry.get("fact_support_span") or "").strip()
            for entry in list(item.get("fact_support_spans") or [])
            if isinstance(entry, dict)
        }
        if any(
            spans_by_atom.get(member_id) != candidate_by_atom[member_id]["source_text"]
            for member_id in member_ids
        ):
            rejection_counts["collection_record_span_not_source_grounded"] += 1
            continue
        if audit_members is not None and set(member_ids) != audit_members:
            rejection_counts["collection_audit_binding_mismatch"] += 1
            continue
        accepted.update(member_ids)
    return ({attribute: accepted} if accepted else {}), dict(rejection_counts)


def _candidate_slots(
    *,
    graph: GovernedMemoryGraph,
    owner_id: str | None,
    principal_relation: str | None = None,
    utility_atom_ids: set[str] | None,
    utility_source_message_ids: set[str] | None = None,
    require_attested_evidence_span: bool = False,
    stage2_authorized_atom_ids: set[str] | None = None,
    stage2_authorized_atom_ids_by_attribute: dict[str, set[str]] | None = None,
) -> list[dict[str, str]]:
    incoming: dict[str, list[Any]] = {}
    nodes = {node.node_id: node for node in graph.nodes}
    subject_surfaces_by_semantic_node: dict[str, list[str]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target_id, []).append(edge)
        if edge.edge_type == "claim_subject":
            subject_node = nodes.get(edge.source_id)
            subject = str(subject_node.label if subject_node is not None else "").strip()
            if subject:
                subject_surfaces_by_semantic_node.setdefault(edge.target_id, []).append(subject)
    rows: list[dict[str, str]] = []
    for node in graph.nodes:
        if node.node_type != "SlotNode":
            continue
        has_slot = next((edge for edge in incoming.get(node.node_id, []) if edge.edge_type == "has_slot"), None)
        semantic_node = nodes.get(has_slot.source_id) if has_slot is not None else None
        source_atom_id = str((node.provenance or {}).get("source_atom_id") or "")
        source_message_ids = {
            str(message_id)
            for message_id in list((node.provenance or {}).get("source_message_ids") or [])
            if str(message_id)
        }
        is_direct_utility_atom = utility_atom_ids is None or source_atom_id in utility_atom_ids
        is_from_selected_utility_source = bool(
            utility_source_message_ids is not None
            and source_message_ids & utility_source_message_ids
        )
        if semantic_node is None or not (is_direct_utility_atom or is_from_selected_utility_source):
            continue
        source_atom_id = str((node.provenance or {}).get("source_atom_id") or "")
        stage2_capability_atoms = set(stage2_authorized_atom_ids or set()) | set().union(
            *(set(values) for values in dict(stage2_authorized_atom_ids_by_attribute or {}).values())
        )
        stage2_authorized = source_atom_id in stage2_capability_atoms
        owner_self_access = str(principal_relation or "").lower() in {"owner", "self"}
        if (
            owner_id
            and str((semantic_node.attributes or {}).get("owner_id") or "") != str(owner_id)
            and not (stage2_authorized or owner_self_access)
        ):
            continue
        value = str((node.attributes or {}).get("slot_value") or "")
        source_text = str(semantic_node.label or "")
        if not value or value not in source_text:
            continue
        # A Stage-2 decision is a closed-set, source-local semantic audit. A
        # selected active record can therefore retain its verbatim source
        # fields even when an older heuristic extractor omitted the redundant
        # evidence_span metadata. Unselected records still require that
        # attestation, and final span validation remains mandatory.
        if require_attested_evidence_span and not stage2_authorized:
            evidence_span = str((node.provenance or {}).get("evidence_span") or "")
            if evidence_span != source_text or value not in evidence_span:
                continue
        rows.append({
            "slot_node_id": node.node_id,
            "slot_name": str((node.attributes or {}).get("slot_name") or ""),
            "slot_value": value,
            "source_text": source_text,
            "source_atom_id": source_atom_id,
            "source_memory_id": str((node.provenance or {}).get("source_memory_id") or ""),
            "source_message_ids": sorted(source_message_ids),
            "claim_subjects": list(dict.fromkeys(
                subject_surfaces_by_semantic_node.get(semantic_node.node_id, [])
            )),
            "slot_role": str((node.attributes or {}).get("slot_role") or ""),
            "claim_property_label": str((node.attributes or {}).get("claim_property_label") or ""),
            "claim_property_labels": list((node.attributes or {}).get("claim_property_labels") or []),
            "claim_spans": list((node.attributes or {}).get("claim_spans") or []),
            "claim_value_spans": list((node.attributes or {}).get("claim_value_spans") or []),
        })
    return rows


def _binding(*, attribute: str, candidates: list[dict[str, str]], source: str, binding_kind: str = "scalar") -> dict[str, Any]:
    return {
        "attribute": attribute,
        "slot_name": candidates[0]["slot_name"],
        "slot_names": list(dict.fromkeys(candidate["slot_name"] for candidate in candidates)),
        "anchor_slot_node_id": candidates[0]["slot_node_id"],
        "anchor_slot_node_ids": [candidate["slot_node_id"] for candidate in candidates],
        "source": source,
        "binding_kind": binding_kind,
    }


def _query_span_is_grounded(
    attribute: str,
    span: str,
    question: str,
    semantic_spec: dict[str, Any],
    *,
    allow_collection_variation: bool = False,
) -> bool:
    if not span or span not in question:
        return False
    binding = next((
        item for item in list(semantic_spec.get("attribute_bindings") or [])
        if isinstance(item, dict) and str(item.get("attribute") or "").strip() == attribute
    ), None)
    expected = str((binding or {}).get("support_span") or "").strip()
    if not expected or span == expected:
        return True
    # The base model may preserve a longer, still verbatim question phrase
    # around the planner's minimal support span. This validates provenance
    # only; it neither maps an attribute nor selects evidence.
    if expected in span and span in question:
        return True
    # Collection names commonly contain a boundary qualifier while the
    # evidence aligner returns the minimal noun phrase. Both spans remain
    # verbatim query evidence; scalar properties retain exact equality.
    return bool(
        (str((binding or {}).get("need_kind") or "") == "record_collection" or allow_collection_variation)
        and len(span.split()) >= 2
        and (span in expected or expected in span)
    )


def _binding_kind(item: dict[str, Any]) -> str:
    value = str(item.get("binding_kind") or item.get("need_kind") or "scalar").strip().lower()
    return "collection" if value in {"collection", "record_collection", "list"} else "scalar"


def _is_record_collection_attribute(attribute: str, semantic_spec: dict[str, Any]) -> bool:
    # Request shape applies to the whole question, not automatically to each
    # attribute. A list can pair one record collection with a scalar (for
    # example, open items and a current date), so explicit per-attribute
    # metadata takes precedence. Use the broad shape only when the planner
    # omitted the attribute entirely.
    has_attribute_metadata = False
    for binding in list(semantic_spec.get("attribute_bindings") or []):
        if not isinstance(binding, dict) or str(binding.get("attribute") or "").strip() != attribute:
            continue
        has_attribute_metadata = True
        if str(binding.get("need_kind") or "").strip() == "record_collection":
            return True
    for need in list(semantic_spec.get("certifiable_needs") or []):
        if not isinstance(need, dict) or str(need.get("attribute") or need.get("need_id") or "").strip() != attribute:
            continue
        has_attribute_metadata = True
        if str(need.get("need_kind") or "").strip() == "record_collection":
            return True
    attribute_tokens = set(re.findall(r"[a-z0-9]+", str(attribute or "").lower()))
    if attribute_tokens & {"snapshot", "overview", "recap"}:
        return True
    return (
        not has_attribute_metadata
        and str(semantic_spec.get("request_shape") or "").strip().lower() in {"list", "plan"}
    )


def _binding_source_expresses_empty_state(
    binding: dict[str, Any], by_id: dict[str, dict[str, str]]
) -> bool:
    """Recognize a source-grounded empty collection state generically."""
    source_text = " ".join(
        str(by_id.get(str(node_id), {}).get("source_text") or "")
        for node_id in list(binding.get("anchor_slot_node_ids") or [])
    ).lower()
    return bool(re.search(
        r"\b(?:no|none|nothing|zero|empty|cleared|all\s+[^.;]+\s+(?:complete|closed))\b",
        source_text,
    ))


def _requested_attributes(semantic_spec: dict[str, Any]) -> list[str]:
    certifiable_needs = semantic_spec.get("certifiable_needs")
    if isinstance(certifiable_needs, list):
        return list(dict.fromkeys(
            str(item.get("attribute") or item.get("need_id") or "").strip()
            for item in certifiable_needs
            if isinstance(item, dict)
            and str(item.get("attribute") or item.get("need_id") or "").strip()
        ))
    return list(dict.fromkeys(
        str(value).strip()
        for value in (semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or [])
        if str(value).strip()
    ))


def _evidence_slot_hints(semantic_spec: dict[str, Any]) -> dict[str, str]:
    hints: dict[str, str] = {}
    certifiable_needs = semantic_spec.get("certifiable_needs")
    if isinstance(certifiable_needs, list):
        for need in certifiable_needs:
            if not isinstance(need, dict):
                continue
            attribute = str(need.get("attribute") or need.get("need_id") or "").strip()
            hint = str(need.get("evidence_slot_hint") or "").strip()
            if attribute and hint:
                hints[attribute] = hint
        return hints
    for binding in list(semantic_spec.get("attribute_bindings") or []):
        if not isinstance(binding, dict):
            continue
        attribute = str(binding.get("attribute") or "").strip()
        hint = str(binding.get("evidence_slot_hint") or "").strip()
        if attribute and hint:
            hints[attribute] = hint
    return hints
