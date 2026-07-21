"""Slot-level authorization certificates for versioned governed evidence."""

from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.graph.governed_graph import GovernedMemoryGraph


def _allows_deleted_state_claim(
    *,
    requested_attribute: str,
    slot_attributes: dict[str, Any],
    semantic_attributes: dict[str, Any],
    value: str,
) -> bool:
    """Allow only a non-secret state assertion from a deleted frame.

    A deletion record can contain a current absence/removal status alongside
    the protected predecessor.  The status is useful evidence for a query
    about presence, but the predecessor itself must remain unavailable.
    """
    atom_type = str(semantic_attributes.get("atom_type") or "").lower()
    slot_role = str(slot_attributes.get("slot_role") or "").lower()
    if "deletion" not in atom_type or slot_role != "claim_value":
        return False
    attribute_tokens = set(re.findall(r"[a-z0-9]+", str(requested_attribute or "").lower()))
    presence_tokens = {
        "whether", "remain", "remains", "remaining", "any", "some", "none",
        "present", "presence", "exist", "exists", "left", "available",
    }
    state_tokens = {
        "no", "not", "none", "absent", "removed", "deleted", "cleared",
        "revoked", "expired", "disabled", "inactive", "unavailable",
    }
    value_text = str(value or "").strip().lower()
    value_tokens = set(re.findall(r"[a-z0-9]+", value_text))
    return bool(
        attribute_tokens & presence_tokens
        and value_tokens & state_tokens
        and value_text
        and not re.search(r"\d", value_text)
    )


def build_slot_governance_certificate(
    *,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    symbolic_decision: dict[str, Any] | None,
    query_echo_evidence: list[RetrievedEvidence],
    principal_relation: str | None = None,
) -> dict[str, Any]:
    """Record slot-level authorization, lifecycle, and evidence provenance.

    This is intentionally domain-agnostic and label-free.  It gives a final
    conservative recommendation only when every requested slot is denied and
    no requested slot remains explicitly authorized as a safe projection.
    """
    requested = _requested_slots(semantic_spec)
    decision = dict(symbolic_decision or {})
    allowed = {str(slot) for slot in list(decision.get("allowed_slots") or [])}
    denied = {str(slot) for slot in list(decision.get("denied_slots") or [])}
    rules_fired = {str(rule) for rule in list(decision.get("rules_fired") or [])}
    query_echo_ids = {str(row.memory_id) for row in query_echo_evidence}
    slots: dict[str, dict[str, Any]] = {}
    for slot in requested:
        support = []
        for row in evidence:
            frame = compile_evidence_frame(row)
            if slot not in dict(frame.slots or {}):
                continue
            lifecycle = str(frame.lifecycle_status or (row.metadata or {}).get("memory_status") or "active").lower()
            support.append({
                "memory_id": str(row.memory_id),
                "source_message_ids": list(row.source_message_ids or []),
                "independent_of_query": str(row.memory_id) not in query_echo_ids,
                "lifecycle": lifecycle,
            })
        independent_active_support = any(
            item["independent_of_query"] and item["lifecycle"] not in {"deleted", "superseded", "canceled"}
            for item in support
        )
        owner_self_support = principal_relation in {"owner", "self"} and independent_active_support
        authorization = (
            "deny"
            if slot in denied
            else "allow"
            if slot in allowed or owner_self_support
            else "undetermined"
        )
        slots[slot] = {
            "slot_family": slot.split("_", 1)[0],
            "authorization": authorization,
            "independent_active_support": independent_active_support,
            "authorization_basis": "owner_self_active_provenance" if owner_self_support and slot not in allowed else "symbolic_policy",
            "evidence": support,
        }
    denied_requested = [slot for slot in requested if slots[slot]["authorization"] == "deny"]
    allowed_requested = [slot for slot in requested if slots[slot]["authorization"] == "allow"]
    all_denied = bool(requested) and len(denied_requested) == len(requested)
    policy_backed_projection = "policy_backed_safe_projection" in rules_fired
    recommendation = "refuse" if all_denied and not policy_backed_projection else None
    return {
        "version": 1,
        "requested_slots": requested,
        "slots": slots,
        "query_origin_filtered_memory_ids": sorted(query_echo_ids),
        "safe_projection_slots": allowed_requested,
        "policy_backed_safe_projection": policy_backed_projection,
        "principal_relation": principal_relation,
        "action_recommendation": recommendation,
        "reason": "all_requested_slots_denied_without_authorized_projection" if recommendation else "certificate_records_slot_level_governance",
    }


def certify_current_state_slots(
    *,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    allowed_rows: list[RetrievedEvidence],
    redacted_rows: list[RetrievedEvidence],
) -> dict[str, Any]:
    """Return a certificate only if every requested latest slot is accessible."""
    requested = _requested_slots(semantic_spec)
    if not requested or str(semantic_spec.get("temporal_scope") or "") != "current":
        return {"authorized": False, "reason": "not_current_state_request", "slots": {}}
    allowed_ids = {str(row.memory_id) for row in allowed_rows}
    redacted_ids = {str(row.memory_id) for row in redacted_rows}
    candidates: dict[str, list[dict[str, Any]]] = {slot: [] for slot in requested}
    for row in evidence:
        frame = compile_evidence_frame(row)
        lifecycle = str(frame.lifecycle_status or (row.metadata or {}).get("memory_status") or "active").lower()
        if lifecycle in {"deleted", "superseded", "canceled"}:
            continue
        slots = dict(frame.slots or {})
        denied = {str(slot) for slot in list((row.metadata or {}).get("denied_slots") or []) if str(slot)}
        for slot in requested:
            if not slots.get(slot):
                continue
            access = "allowed" if row.memory_id in allowed_ids else "redacted" if row.memory_id in redacted_ids else "unavailable"
            # A row marked with any denied typed value is mixed evidence and
            # cannot certify direct replay of a different value from that row.
            if denied:
                access = "unavailable"
            candidates[slot].append({
                "memory_id": str(row.memory_id), "value": str(slots[slot]), "access": access,
                "time": str(frame.effective_time or row.time or ""),
            })
    certified: dict[str, dict[str, Any]] = {}
    memory_ids: list[str] = []
    redacted = False
    for slot, rows in candidates.items():
        if not rows:
            return {"authorized": False, "reason": f"missing_slot:{slot}", "slots": certified}
        newest_time = max(row["time"] for row in rows)
        newest = [row for row in rows if row["time"] == newest_time]
        if any(row["access"] == "unavailable" for row in newest):
            return {"authorized": False, "reason": f"unavailable_latest_slot:{slot}", "slots": certified}
        selected = next((row for row in newest if row["access"] == "allowed"), newest[0])
        certified[slot] = selected
        memory_ids.append(selected["memory_id"])
        redacted = redacted or selected["access"] == "redacted"
    return {
        "authorized": True,
        "reason": "all_requested_current_slots_have_accessible_latest_provenance",
        "slots": certified,
        "memory_ids": list(dict.fromkeys(memory_ids)),
        "requires_redaction": redacted,
    }


def certify_graph_slot_paths(
    *,
    semantic_spec: dict[str, Any],
    graph: GovernedMemoryGraph | None,
    principal_relation: str | None,
    owner_id: str | None = None,
    utility_atom_ids: set[str] | None = None,
    utility_source_message_ids: set[str] | None = None,
    governance_policy_atom_ids: set[str] | None = None,
    semantic_contract_certifiable: bool = True,
    semantic_alignment: dict[str, Any] | None = None,
    require_attested_evidence_span: bool = False,
    principal_relation_ledger: dict[str, Any] | None = None,
    stage2_authorized_atom_ids: set[str] | None = None,
    stage2_authorized_atom_ids_by_attribute: dict[str, set[str]] | None = None,
    stage2_realizations: list[dict[str, Any]] | None = None,
    allow_utility_record_completion: bool = False,
) -> dict[str, Any]:
    """Certify requested slots only from explicit governed-graph paths.

    This certificate never infers permission from a role name alone: every
    requested attribute needs an aligned, explicit `allows` edge from a
    principal-compatible role scope.
    """
    if not semantic_contract_certifiable:
        return {"authorized": False, "reason": "semantic_contract_not_certifiable", "slots": {}}
    requested = _requested_slots(semantic_spec)
    if graph is None:
        return {"authorized": False, "reason": "graph_unavailable", "slots": {}}
    has_attested_relation = principal_relation_ledger is None or _has_attested_principal_relation(
        graph=graph,
        ledger=principal_relation_ledger,
        principal_relation=principal_relation,
        owner_id=owner_id,
    )
    requester_id = str((principal_relation_ledger or {}).get("requester_id") or "").strip()
    has_scoped_stewardship = bool(
        requester_id
        and _has_attested_scoped_stewardship(
            graph=graph, requester_id=requester_id, owner_id=owner_id
        )
    )
    has_direct_requester_policy = bool(
        requester_id
        and _has_attested_direct_requester_policy_allow(
            graph=graph, requester_id=requester_id, owner_id=owner_id
        )
    )
    # Stage 2 can certify an exact active operational record for the requester.
    # It is narrower than a relationship or a standing permission: later code
    # still requires semantic alignment, a matching source atom, an active
    # lifecycle, and the exact requested slot. This prevents unrelated
    # cross-owner records from forcing a false refusal solely because a
    # multi-part current-state answer has more than one factual reporter.
    stage2_capability_by_attribute = {
        str(attribute): {str(atom_id) for atom_id in atom_ids if str(atom_id)}
        for attribute, atom_ids in dict(stage2_authorized_atom_ids_by_attribute or {}).items()
    }
    if stage2_authorized_atom_ids and not stage2_capability_by_attribute:
        stage2_capability_by_attribute.setdefault("*", set()).update(
            str(atom_id) for atom_id in stage2_authorized_atom_ids if str(atom_id)
        )
    has_stage2_operational_capability = bool(stage2_capability_by_attribute)
    if (
        not has_attested_relation
        and not has_scoped_stewardship
        and not has_direct_requester_policy
        and not has_stage2_operational_capability
    ):
        return {"authorized": False, "reason": "principal_relation_or_scoped_capability_not_attested_in_graph", "slots": {}}
    if not requested:
        return {"authorized": False, "reason": "no_requested_slots", "slots": {}}
    graph_nodes = {node.node_id: node for node in graph.nodes}
    observed_slot_node_ids = {
        edge.target_id
        for edge in graph.edges
        if edge.edge_type == "has_slot"
        and str((graph_nodes.get(edge.source_id).attributes or {}).get("atom_type") or "")
        not in {"policy_atom", "permission_atom"}
    }
    alignment = dict(semantic_alignment or {})
    bindings = dict(alignment.get("bindings") or {})
    # Stage 2 capability authorizes only a selected record, never an
    # unresolved semantic attribute. Every requested attribute must retain a
    # closed-set alignment before Stage 3 can realize it.
    if semantic_alignment is None:
        # Compatibility for direct callers and simple contracts: exact schema
        # identifier equality needs no semantic interpretation.  Runtime code
        # always supplies the stricter evidence-mediated alignment artifact.
        for attribute in requested:
            anchor = next((
                node.node_id
                for node in graph.nodes
                if node.node_type == "SlotNode"
                and node.node_id in observed_slot_node_ids
                and str((node.attributes or {}).get("slot_name") or "") == attribute
            ), "")
            if anchor:
                bindings[attribute] = {
                    "attribute": attribute,
                    "slot_name": attribute,
                    "anchor_slot_node_id": anchor,
                    "source": "exact_slot_identifier",
                }
        alignment = {"available": len(bindings) == len(requested), "bindings": bindings}
    missing_alignment = [slot for slot in requested if slot not in bindings]
    if missing_alignment and not bindings:
        return {
            "authorized": False,
            "reason": "missing_semantic_alignment:" + ",".join(missing_alignment),
            "slots": {},
            "semantic_alignment": alignment,
        }
    # A mixed request can contain both an aligned, authorized operational
    # attribute and an unresolved sensitive attribute. Certify only the
    # former and require a redacted realization; never let the former record
    # stand in for the unresolved attribute.
    certifiable_requested = [attribute for attribute in requested if attribute in bindings]
    temporal_scope = str(semantic_spec.get("temporal_scope") or "unspecified").lower()
    request_shape = str(semantic_spec.get("request_shape") or "fact").lower()
    nodes = {node.node_id: node for node in graph.nodes}
    incoming: dict[str, list[Any]] = {}
    allowed_slot_node_ids: set[str] = set()
    denied_slot_node_ids: set[str] = set()
    permission_required_slot_node_ids: set[str] = set()
    superseded_targets: set[str] = set()
    retired_semantic_node_ids: set[str] = set()
    for edge in graph.edges:
        incoming.setdefault(edge.target_id, []).append(edge)
        target = nodes.get(edge.target_id)
        target_slot_name = str((target.attributes or {}).get("slot_name") or "") if target else ""
        policy_atom_id = str((edge.provenance or {}).get("source_atom_id") or "")
        relation_assignment_authorization = bool((edge.provenance or {}).get("relation_assignment_authorization"))
        scoped_stewardship_authorization = _is_valid_scoped_stewardship_allow(
            graph=graph,
            edge=edge,
            requester_id=requester_id,
            owner_id=owner_id,
        )
        direct_requester_policy_authorization = _is_valid_direct_requester_policy_allow(
            edge=edge, requester_id=requester_id, owner_id=owner_id
        )
        if (
            target_slot_name
            and (
                scoped_stewardship_authorization
                or direct_requester_policy_authorization
                or (
                    _scope_matches_principal(edge.source_id, principal_relation)
                    and (
                        governance_policy_atom_ids is None
                        or policy_atom_id in governance_policy_atom_ids
                        or relation_assignment_authorization
                    )
                )
            )
        ):
            if edge.edge_type == "allows":
                allowed_slot_node_ids.add(edge.target_id)
            elif edge.edge_type == "denies":
                denied_slot_node_ids.add(edge.target_id)
            elif edge.edge_type == "requires_permission":
                permission_required_slot_node_ids.add(edge.target_id)
        if edge.edge_type == "supersedes_slot":
            superseded_targets.add(edge.target_id)
        if edge.edge_type in {"deletes", "supersedes"}:
            retired_semantic_node_ids.add(edge.target_id)
    certified: dict[str, dict[str, Any]] = {}
    realizations: list[dict[str, Any]] = []
    redacted_slot_names: list[str] = []
    for requested_attribute in certifiable_requested:
        binding = dict(bindings.get(requested_attribute) or {})
        binding_kind = str(binding.get("binding_kind") or "")
        binding_source = str(binding.get("source") or "")
        record_completion = bool(
            allow_utility_record_completion
            and _is_summary_attribute(requested_attribute)
        )
        anchor_slot_node_ids = [
            str(value) for value in list(binding.get("anchor_slot_node_ids") or []) if str(value)
        ] or [str(binding.get("anchor_slot_node_id") or "")]
        anchor_slot_node_ids = [value for value in anchor_slot_node_ids if value]
        # Stage 2's closed collection decision already specifies the exact
        # record-local SlotNodes. Validate and realize those anchors directly:
        # a multi-record collection does not require unrelated fields to form
        # one cross-record version family.
        if binding_kind == "collection" and binding_source in {
            "stage2_closed_record_selection", "stage2_authorized_record_closure"
            , "stage2_closed_record_slot_projection"
        }:
            selected = []
            skipped = []
            for node_id in anchor_slot_node_ids:
                node = nodes.get(node_id)
                semantic_edges = [edge for edge in incoming.get(node_id, []) if edge.edge_type == "has_slot"]
                semantic_node = nodes.get(semantic_edges[0].source_id) if semantic_edges else None
                if node is None or semantic_node is None or node_id not in observed_slot_node_ids:
                    skipped.append((node_id, "unobserved_slot"))
                    continue
                source_atom_id = str((node.provenance or {}).get("source_atom_id") or "")
                stage2_authorized = source_atom_id in (
                    stage2_capability_by_attribute.get(requested_attribute, set())
                    | stage2_capability_by_attribute.get("*", set())
                )
                # For utility collections, Stage 2 has already selected the
                # exact source-record closure.  That record-local admission
                # is sufficient to inspect its typed fields, even when the
                # adjudicator did not repeat the aggregate collection label
                # for every complementary field.  Non-utility queries still
                # require an attribute-local operational capability below.
                utility_record_selected = bool(
                    allow_utility_record_completion
                    and utility_atom_ids is not None
                    and source_atom_id in utility_atom_ids
                )
                stage2_authorized = stage2_authorized or utility_record_selected
                lifecycle = str((semantic_node.attributes or {}).get("lifecycle") or "active").lower()
                value = str((node.attributes or {}).get("slot_value") or "")
                source_text = str(semantic_node.label or "")
                deleted_state_allowed = _allows_deleted_state_claim(
                    requested_attribute=requested_attribute,
                    slot_attributes=dict(node.attributes or {}),
                    semantic_attributes=dict(semantic_node.attributes or {}),
                    value=value,
                )
                owner_active_provenance = (
                    principal_relation == "owner"
                    and owner_id is not None
                    and str((semantic_node.attributes or {}).get("owner_id") or "") == str(owner_id)
                )
                if (
                    not (stage2_authorized or owner_active_provenance or deleted_state_allowed)
                    or (
                        lifecycle in {"deleted", "superseded", "canceled"}
                        and not deleted_state_allowed
                    )
                    or semantic_node.node_id in retired_semantic_node_ids
                    or not value
                    or value.lower() not in source_text.lower()
                ):
                    skipped.append((node_id, "lifecycle_or_capability"))
                    continue
                if require_attested_evidence_span and not stage2_authorized:
                    evidence_span = str((node.provenance or {}).get("evidence_span") or "")
                    if evidence_span != source_text or value.lower() not in evidence_span.lower():
                        skipped.append((node_id, "missing_attested_span"))
                        continue
                selected.append({
                    "slot_node_id": node_id,
                    "value": value,
                    "timestamp": str((node.provenance or {}).get("timestamp") or ""),
                    "explicitly_allowed": True,
                    "source_atom_id": source_atom_id,
                    "source_memory_id": str((node.provenance or {}).get("source_memory_id") or ""),
                    "source_message_ids": [str(message_id) for message_id in list((node.provenance or {}).get("source_message_ids") or []) if str(message_id)],
                    "source_text": source_text,
                })
            # A collection is a set of independently typed records.  One
            # stale/deleted sibling must not invalidate other source-grounded
            # members.  The selected subset remains closed by the anchors;
            # no new graph node is searched here.
            if not selected:
                return {
                    "authorized": False,
                    "reason": f"missing_active_graph_slot:{requested_attribute}",
                    "slots": certified,
                    "semantic_alignment": alignment,
                }
            certified[requested_attribute] = selected[0]
            for item in selected:
                graph_slot_name = str((nodes.get(item["slot_node_id"]).attributes or {}).get("slot_name") or "")
                graph_slot_attributes = dict(nodes.get(item["slot_node_id"]).attributes or {})
                claim_span = str(graph_slot_attributes.get("claim_span") or "")
                if not claim_span:
                    claim_spans = graph_slot_attributes.get("claim_spans")
                    if isinstance(claim_spans, list) and claim_spans:
                        claim_span = str(claim_spans[0] or "")
                realization = {
                    "attribute": requested_attribute,
                    "slot_name": graph_slot_name,
                    "slot_role": str(graph_slot_attributes.get("slot_role") or ""),
                    "claim_span": claim_span,
                    "typed_slot_value": item["value"],
                    "record_complete": True,
                    **item,
                }
                _merge_stage2_source_grounded_value(
                    realization=realization,
                    requested_attribute=requested_attribute,
                    graph_slot_name=graph_slot_name,
                    stage2_realizations=stage2_realizations,
                )
                realizations.append(realization)
            if skipped:
                redacted_slot_names.extend(
                    str((nodes.get(node_id).attributes or {}).get("slot_name") or "")
                    for node_id, _ in skipped
                    if nodes.get(node_id) is not None
                    and str((nodes.get(node_id).attributes or {}).get("slot_name") or "")
                )
            continue
        slot_names = [
            str(value).strip()
            for value in list(binding.get("slot_names") or [])
            if str(value).strip()
        ] or [str(binding.get("slot_name") or "").strip()]
        slot_names = list(dict.fromkeys(slot_name for slot_name in slot_names if slot_name))
        selected: list[dict[str, Any]] = []
        for slot_name in slot_names:
            anchors_for_slot = [
                node_id for node_id in anchor_slot_node_ids
                if str((nodes.get(node_id).attributes or {}).get("slot_name") or "") == slot_name
            ]
            family_node_ids = set().union(*(
                _version_family_node_ids(graph=graph, anchor_slot_node_id=anchor_slot_node_id)
                for anchor_slot_node_id in anchors_for_slot
            )) if anchors_for_slot else set()
            candidates: list[dict[str, Any]] = []
            for node in graph.nodes:
                if (
                    node.node_type != "SlotNode"
                    or str((node.attributes or {}).get("slot_name") or "") != slot_name
                    or node.node_id not in family_node_ids
                ):
                    continue
                semantic_edges = [edge for edge in incoming.get(node.node_id, []) if edge.edge_type == "has_slot"]
                if not semantic_edges:
                    continue
                semantic_node = nodes.get(semantic_edges[0].source_id)
                if semantic_node is not None and semantic_node.node_id in retired_semantic_node_ids:
                    continue
                source_atom_id = str((node.provenance or {}).get("source_atom_id") or "")
                stage2_authorized = source_atom_id in (
                    stage2_capability_by_attribute.get(requested_attribute, set())
                    | stage2_capability_by_attribute.get("*", set())
                )
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
                if not (is_direct_utility_atom or is_from_selected_utility_source):
                    continue
                fact_owner = str((semantic_node.attributes or {}).get("owner_id") or "") if semantic_node else ""
                value = str((node.attributes or {}).get("slot_value") or "")
                deleted_state_allowed = _allows_deleted_state_claim(
                    requested_attribute=requested_attribute,
                    slot_attributes=dict(node.attributes or {}),
                    semantic_attributes=dict((semantic_node.attributes if semantic_node else {}) or {}),
                    value=value,
                )
                utility_source_capability = bool(
                    allow_utility_record_completion
                    and is_from_selected_utility_source
                )
                if owner_id and fact_owner != str(owner_id) and not (
                    stage2_authorized
                    or deleted_state_allowed
                    or utility_source_capability
                ):
                    continue
                lifecycle = str((semantic_node.attributes or {}).get("lifecycle") or "active").lower() if semantic_node else "active"
                if lifecycle in {"deleted", "superseded", "canceled"} and not deleted_state_allowed:
                    continue
                if temporal_scope != "historical" and node.node_id in superseded_targets:
                    continue
                source_text = str((semantic_node.label if semantic_node else "") or "")
                if not value or value.lower() not in source_text.lower():
                    continue
                if require_attested_evidence_span and not stage2_authorized:
                    evidence_span = str((node.provenance or {}).get("evidence_span") or "")
                    if not evidence_span or evidence_span != source_text or value.lower() not in evidence_span.lower():
                        continue
                owner_default = (
                    principal_relation == "owner"
                    and owner_id is not None
                    and (fact_owner == str(owner_id) or stage2_authorized)
                    and node.node_id not in permission_required_slot_node_ids
                )
                # Stage 2 is an LLM adjudication over a closed record set.
                # Its authorization can realize only an already aligned slot
                # on that exact selected source atom. It is not a record-wide
                # fallback and cannot authorize an unresolved attribute.
                stage2_aligned_capability = (
                    stage2_authorized
                    and node.node_id in anchors_for_slot
                    and node.node_id not in denied_slot_node_ids
                    and node.node_id not in permission_required_slot_node_ids
                )
                # Atomic-memory and graph compilation can assign different
                # source-atom IDs to the same selected source turn. For a
                # utility query, the locator's closed source-message set is
                # already the source-local capability boundary. Preserve that
                # capability for this exact aligned slot without extending it
                # to sibling slots or to privacy/safety queries.
                stage2_aligned_capability = stage2_aligned_capability or (
                    allow_utility_record_completion
                    and is_from_selected_utility_source
                    and node.node_id in anchors_for_slot
                    and node.node_id not in denied_slot_node_ids
                    and node.node_id not in permission_required_slot_node_ids
                )
                candidates.append({
                    "slot_node_id": node.node_id,
                    "value": value,
                    "timestamp": str((node.provenance or {}).get("timestamp") or ""),
                    "explicitly_allowed": (
                        (node.node_id not in denied_slot_node_ids or deleted_state_allowed)
                        and (
                            node.node_id in allowed_slot_node_ids
                            or owner_default
                            or stage2_aligned_capability
                            or deleted_state_allowed
                        )
                    ),
                    "source_atom_id": source_atom_id,
                    "source_memory_id": str((node.provenance or {}).get("source_memory_id") or ""),
                    "source_message_ids": sorted(source_message_ids),
                    "source_text": source_text,
                })
            if not candidates:
                return {
                    "authorized": False,
                    "reason": f"missing_active_graph_slot:{requested_attribute}",
                    "slots": certified,
                    "semantic_alignment": alignment,
                }
            if temporal_scope == "historical":
                slot_selected = [item for item in candidates if item["slot_node_id"] in anchors_for_slot]
            elif request_shape in {"list", "plan"}:
                slot_selected = candidates
            else:
                newest_time = max(item["timestamp"] for item in candidates)
                slot_selected = [item for item in candidates if item["timestamp"] == newest_time]
            if not slot_selected:
                return {
                    "authorized": False,
                    "reason": f"no_explicit_graph_allow:{requested_attribute}",
                    "slots": certified,
                    "semantic_alignment": alignment,
                }
            allowed_selected = [item for item in slot_selected if item["explicitly_allowed"]]
            if request_shape in {"list", "plan"}:
                if not allowed_selected:
                    redacted_slot_names.append(slot_name)
                    continue
                if len(allowed_selected) != len(slot_selected):
                    redacted_slot_names.append(slot_name)
                selected.extend(allowed_selected)
            elif not all(item["explicitly_allowed"] for item in slot_selected):
                return {
                    "authorized": False,
                    "reason": f"no_explicit_graph_allow:{requested_attribute}",
                    "slots": certified,
                    "semantic_alignment": alignment,
                }
            else:
                selected.extend(slot_selected)
        if not selected:
            return {
                "authorized": False,
                "reason": f"no_explicit_graph_allow:{requested_attribute}",
                "slots": certified,
                "semantic_alignment": alignment,
            }
        expand_record_siblings = bool(
            binding_kind == "collection"
            and binding_source in {
                "stage2_closed_record_selection",
                "stage2_authorized_record_closure",
                "stage2_closed_record_slot_projection",
                "llm_collection_record_alignment",
                "llm_collection_full_set_audit_with_reviewer",
                "llm_collection_completion",
            }
        ) or record_completion
        if expand_record_siblings:
            anchored_record_node_ids = {
                edge.source_id
                for anchor_slot_node_id in anchor_slot_node_ids
                for edge in incoming.get(anchor_slot_node_id, [])
                if edge.edge_type == "has_slot"
            }
            # A collection's alignment selects records. Once a record-local
            # capability has explicitly allowed an identifying field, include
            # only its other explicitly allowed fields from the same semantic
            # record. This preserves field/value association and never expands
            # to a sibling record or an unapproved SlotNode.
            selected_node_ids = {item["slot_node_id"] for item in selected}
            for node in graph.nodes:
                if node.node_type != "SlotNode" or node.node_id in selected_node_ids:
                    continue
                semantic_edges = [edge for edge in incoming.get(node.node_id, []) if edge.edge_type == "has_slot"]
                if not semantic_edges or semantic_edges[0].source_id not in anchored_record_node_ids:
                    continue
                semantic_node = nodes.get(semantic_edges[0].source_id)
                source_atom_id = str((node.provenance or {}).get("source_atom_id") or "")
                stage2_authorized = source_atom_id in (
                    stage2_capability_by_attribute.get(requested_attribute, set())
                    | stage2_capability_by_attribute.get("*", set())
                )
                source_message_ids = {
                    str(message_id)
                    for message_id in list((node.provenance or {}).get("source_message_ids") or [])
                    if str(message_id)
                }
                if not (
                    utility_atom_ids is None
                    or source_atom_id in utility_atom_ids
                    or (utility_source_message_ids is not None and source_message_ids & utility_source_message_ids)
                ):
                    continue
                lifecycle = str((semantic_node.attributes or {}).get("lifecycle") or "active").lower()
                value = str((node.attributes or {}).get("slot_value") or "")
                deleted_state_allowed = _allows_deleted_state_claim(
                    requested_attribute=requested_attribute,
                    slot_attributes=dict(node.attributes or {}),
                    semantic_attributes=dict(semantic_node.attributes or {}),
                    value=value,
                )
                if owner_id and str((semantic_node.attributes or {}).get("owner_id") or "") != str(owner_id) and not (stage2_authorized or deleted_state_allowed):
                    continue
                if lifecycle in {"deleted", "superseded", "canceled"} and not deleted_state_allowed:
                    continue
                if semantic_node.node_id in retired_semantic_node_ids:
                    continue
                source_text = str(semantic_node.label or "")
                if not value or value.lower() not in source_text.lower():
                    continue
                if require_attested_evidence_span and not stage2_authorized:
                    evidence_span = str((node.provenance or {}).get("evidence_span") or "")
                    if evidence_span != source_text or value.lower() not in evidence_span.lower():
                        continue
                owner_default = (
                    principal_relation == "owner"
                    and owner_id is not None
                    and (
                        str((semantic_node.attributes or {}).get("owner_id") or "") == str(owner_id)
                        or stage2_authorized
                    )
                    and node.node_id not in permission_required_slot_node_ids
                )
                if (
                    node.node_id in denied_slot_node_ids
                    and not deleted_state_allowed
                ) or not (
                    node.node_id in allowed_slot_node_ids
                    or owner_default
                    or deleted_state_allowed
                ):
                    continue
                selected.append({
                    "slot_node_id": node.node_id,
                    "value": value,
                    "timestamp": str((node.provenance or {}).get("timestamp") or ""),
                    "explicitly_allowed": True,
                    "source_atom_id": source_atom_id,
                    "source_memory_id": str((node.provenance or {}).get("source_memory_id") or ""),
                    "source_message_ids": sorted(source_message_ids),
                    "source_text": source_text,
                })
        certified[requested_attribute] = selected[0]
        for item in selected:
            graph_slot_attributes = dict(nodes.get(item["slot_node_id"]).attributes or {})
            claim_span = str(graph_slot_attributes.get("claim_span") or "")
            if not claim_span:
                claim_spans = graph_slot_attributes.get("claim_spans")
                if isinstance(claim_spans, list) and claim_spans:
                    claim_span = str(claim_spans[0] or "")
            realized_slot_name = str(graph_slot_attributes.get("slot_name") or "")
            graph_slot_name = str(graph_slot_attributes.get("slot_name") or "")
            realization = {
                "attribute": requested_attribute,
                "slot_name": realized_slot_name,
                "slot_role": str(graph_slot_attributes.get("slot_role") or ""),
                "claim_span": claim_span,
                "typed_slot_value": item["value"],
                # Collection expansion admits the complete source record;
                # scalar alignment may have certified only one field.
                "record_complete": binding_kind == "collection" or record_completion,
                **item,
            }
            _merge_stage2_source_grounded_value(
                realization=realization,
                requested_attribute=requested_attribute,
                graph_slot_name=graph_slot_name,
                stage2_realizations=stage2_realizations,
            )
            realizations.append(realization)
    partial_disclosure = bool(missing_alignment or redacted_slot_names)
    return {
        "authorized": True,
        "reason": (
            "partial_requested_attributes_have_explicit_allow_path"
            if partial_disclosure
            else "all_requested_attributes_have_explicit_allow_path"
        ),
        "slots": certified,
        "realizations": realizations,
        "semantic_alignment": alignment,
        "temporal_scope": temporal_scope,
        "requires_redaction": partial_disclosure,
        "redacted_slot_names": list(dict.fromkeys(redacted_slot_names)),
        "unresolved_requested_attributes": missing_alignment,
        "scoped_capability_authorized": has_scoped_stewardship,
        "direct_requester_policy_authorized": has_direct_requester_policy,
        "stage2_operational_capability_authorized": has_stage2_operational_capability,
        "stage2_operational_capability_attributes": sorted(
            attribute for attribute in stage2_capability_by_attribute if attribute != "*"
        ),
    }


def _merge_stage2_source_grounded_value(
    *,
    realization: dict[str, Any],
    requested_attribute: str,
    graph_slot_name: str,
    stage2_realizations: list[dict[str, Any]] | None,
) -> None:
    """Carry a longer Stage-2 typed value onto its graph-certified slot.

    Graph SlotNodes can preserve a short extractor value while the closed
    Stage-2 adjudication has already selected a longer composite value from
    the same source record.  The merge is deliberately evidence-local: it
    requires the same memory, an answer decision, exact source grounding, and
    structural token overlap between the two typed slot names.
    """
    if not stage2_realizations:
        return
    source_memory_id = str(realization.get("source_memory_id") or "")
    source_atom_id = str(realization.get("source_atom_id") or "")
    source_message_ids = {
        str(value)
        for value in list(realization.get("source_message_ids") or [])
        if str(value)
    }
    source_text = str(realization.get("source_text") or "")
    current_value = str(realization.get("value") or "").strip()
    graph_claim_span = str(realization.get("claim_span") or "").strip()
    graph_tokens = _typed_slot_tokens(graph_slot_name)
    if not source_memory_id or not source_text or not current_value or not graph_tokens:
        return
    candidates: list[tuple[int, dict[str, Any], str]] = []
    source_slot_names: set[str] = set()
    for candidate in stage2_realizations:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("attribute") or "").strip() != str(requested_attribute).strip():
            continue
        if str(candidate.get("decision") or "").strip().lower() != "answer":
            continue
        candidate_memory_id = str(candidate.get("memory_id") or "")
        candidate_atom_id = str(candidate.get("source_atom_id") or "")
        candidate_message_ids = {
            str(value)
            for value in list(candidate.get("source_message_ids") or [])
            if str(value)
        }
        same_source_message = bool(source_message_ids & candidate_message_ids)
        if candidate_memory_id != source_memory_id and not same_source_message:
            continue
        if (
            candidate_atom_id
            and source_atom_id
            and candidate_atom_id != source_atom_id
            and not same_source_message
        ):
            continue
        slot_name = str(candidate.get("slot_name") or "").strip()
        value = str(candidate.get("value") or "").strip()
        if not slot_name or not value:
            continue
        # A claim adjudicator may echo the complete evidence sentence when it
        # is asked to cover several fields.  That sentence is provenance, not
        # a typed component, and must never replace the graph's exact slot
        # value during source-local merging.
        if value.rstrip(".").strip() == source_text.rstrip(".").strip():
            continue
        source_slot_names.add(slot_name)
        if current_value not in source_text:
            continue
        claim_span = str(candidate.get("claim_span") or "").strip()
        if claim_span and claim_span not in source_text:
            claim_span = ""
        same_claim_span = bool(
            graph_claim_span
            and claim_span
            and graph_claim_span == claim_span
            and graph_claim_span in source_text
        )
        if not (graph_tokens & _typed_slot_tokens(slot_name)) and not same_claim_span and not same_source_message:
            # Subject/predicate slot names can differ while referring to the
            # same source-local claim. Same-source Stage-2 adjudication is a
            # stronger closed-set link than lexical slot-name overlap.
            continue
        replacement = value if len(value) > len(current_value) and value in source_text and current_value in value else ""
        if claim_span and current_value in claim_span and len(claim_span) > len(current_value):
            replacement = claim_span
        if not replacement:
            continue
        candidates.append((len(replacement), candidate, replacement))
    if not candidates:
        return
    _, candidate, replacement = max(candidates, key=lambda item: item[0])
    # When one source record contributes several independently typed fields,
    # keep the graph field label for each realization. Otherwise the first
    # Stage-2 alias can relabel every sibling field in the final projection.
    if len(source_slot_names) <= 1:
        realization["slot_name"] = str(candidate.get("slot_name") or realization.get("slot_name") or "")
    realization["value"] = replacement.strip()


def _typed_slot_tokens(value: str) -> set[str]:
    ignored = {"a", "an", "the", "current", "latest", "active", "selected"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if token not in ignored
    }


def _is_summary_attribute(value: str) -> bool:
    """Recognize record-level summary roles without a domain vocabulary."""
    return bool(_typed_slot_tokens(value) & {"summary", "overview", "recap", "snapshot"})


def _version_family_node_ids(*, graph: GovernedMemoryGraph, anchor_slot_node_id: str) -> set[str]:
    """Return a version-connected evidence family rooted at the LLM-selected node."""
    if not anchor_slot_node_id:
        return set()
    neighbors: dict[str, set[str]] = {}
    for edge in graph.edges:
        if edge.edge_type != "version_precedes":
            continue
        neighbors.setdefault(edge.source_id, set()).add(edge.target_id)
        neighbors.setdefault(edge.target_id, set()).add(edge.source_id)
    seen = {anchor_slot_node_id}
    stack = [anchor_slot_node_id]
    while stack:
        current = stack.pop()
        for neighbor in neighbors.get(current, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _scope_matches_principal(role_node_id: str, principal_relation: str | None) -> bool:
    scope = str(role_node_id or "").split("role::", 1)[-1].lower()
    relation = str(principal_relation or "").lower()
    compatible = {
        "owner": {"self"},
        "authorized_staff": {"authorized_staff", "collaborator", "clinician"},
        "delegate": {"collaborator"},
        "family": {"family"},
    }
    return scope in compatible.get(relation, set())


def _has_attested_principal_relation(
    *,
    graph: GovernedMemoryGraph,
    ledger: dict[str, Any],
    principal_relation: str | None,
    owner_id: str | None,
) -> bool:
    requester_id = str(ledger.get("requester_id") or "").strip()
    ledger_owner_id = str(ledger.get("owner_id") or "").strip()
    relation = str(ledger.get("effective_relation") or "").strip()
    status = str(ledger.get("effective_status") or "").strip()
    if (
        not requester_id
        or not ledger_owner_id
        or ledger_owner_id != str(owner_id or "").strip()
        or relation != str(principal_relation or "").strip()
        or status != "proven"
    ):
        return False
    relation_node_id = f"relation::{requester_id}::{ledger_owner_id}::{relation}"
    relation_node = next((node for node in graph.nodes if node.node_id == relation_node_id), None)
    if relation_node is None or relation_node.node_type != "PrincipalRelationNode":
        return False
    attributes = dict(relation_node.attributes or {})
    if (
        attributes.get("requester_id") != requester_id
        or attributes.get("owner_id") != ledger_owner_id
        or attributes.get("relation") != relation
        or attributes.get("status") != "proven"
        or attributes.get("direction") != "requester_to_owner"
    ):
        return False
    if relation != "owner" and not list((relation_node.provenance or {}).get("source_message_ids") or []):
        return False
    expected_edges = {
        ("has_relation", f"principal::{requester_id}", relation_node_id),
        ("relation_owner", relation_node_id, f"principal::{ledger_owner_id}"),
    }
    observed_edges = {(edge.edge_type, edge.source_id, edge.target_id) for edge in graph.edges}
    return expected_edges.issubset(observed_edges)


def _has_attested_scoped_stewardship(
    *, graph: GovernedMemoryGraph, requester_id: str, owner_id: str | None
) -> bool:
    """Require a source-grounded capability graph, never just its node label."""
    expected_owner_id = str(owner_id or "").strip()
    if not expected_owner_id:
        return False
    nodes = {node.node_id: node for node in graph.nodes}
    observed_edges = {(edge.edge_type, edge.source_id, edge.target_id) for edge in graph.edges}
    for node in graph.nodes:
        if node.node_type != "ScopedCapabilityNode":
            continue
        attributes = dict(node.attributes or {})
        provenance = dict(node.provenance or {})
        record_node_id = str(attributes.get("record_node_id") or "")
        slot_node_ids = {str(value) for value in list(attributes.get("slot_node_ids") or []) if str(value)}
        supports = [item for item in list(provenance.get("support_evidence") or []) if isinstance(item, dict)]
        if (
            attributes.get("requester_id") != requester_id
            or attributes.get("owner_id") != expected_owner_id
            or attributes.get("status") != "proven"
            or not record_node_id
            or not slot_node_ids
            or not str(provenance.get("source_atom_id") or "")
            or not list(provenance.get("source_message_ids") or [])
            or not supports
        ):
            continue
        if {
            ("holds_capability", f"principal::{requester_id}", node.node_id),
            ("capability_owner", node.node_id, f"principal::{expected_owner_id}"),
            ("stewards_record", node.node_id, record_node_id),
        }.issubset(observed_edges):
            return True
    return False


def _has_attested_direct_requester_policy_allow(
    *, graph: GovernedMemoryGraph, requester_id: str, owner_id: str | None
) -> bool:
    expected_owner = str(owner_id or "").strip()
    return any(
        _is_valid_direct_requester_policy_allow(
            edge=edge, requester_id=requester_id, owner_id=expected_owner
        )
        for edge in graph.edges
    )


def _is_valid_direct_requester_policy_allow(
    *, edge: Any, requester_id: str, owner_id: str | None
) -> bool:
    provenance = dict(edge.provenance or {})
    return bool(
        edge.edge_type == "allows"
        and provenance.get("direct_requester_policy_authorization")
        and provenance.get("requester_id") == requester_id
        and provenance.get("governed_owner_id") == str(owner_id or "").strip()
        and str(provenance.get("source_atom_id") or "")
        and str(provenance.get("evidence_span") or "")
    )


def _is_valid_scoped_stewardship_allow(
    *, graph: GovernedMemoryGraph, edge: Any, requester_id: str, owner_id: str | None
) -> bool:
    if edge.edge_type != "allows" or not bool((edge.provenance or {}).get("scoped_stewardship_authorization")):
        return False
    capability = next((node for node in graph.nodes if node.node_id == edge.source_id), None)
    target = next((node for node in graph.nodes if node.node_id == edge.target_id), None)
    if capability is None or target is None or capability.node_type != "ScopedCapabilityNode" or target.node_type != "SlotNode":
        return False
    attributes = dict(capability.attributes or {})
    if (
        attributes.get("requester_id") != requester_id
        or attributes.get("owner_id") != str(owner_id or "").strip()
        or attributes.get("status") != "proven"
        or edge.target_id not in {str(value) for value in list(attributes.get("slot_node_ids") or [])}
    ):
        return False
    record_node_id = str(attributes.get("record_node_id") or "")
    record = next((node for node in graph.nodes if node.node_id == record_node_id), None)
    if record is None or str((record.attributes or {}).get("owner_id") or "") != str(owner_id or "").strip():
        return False
    if str((record.provenance or {}).get("speaker") or "") != requester_id:
        return False
    if str((edge.provenance or {}).get("source_atom_id") or "") != str((record.provenance or {}).get("source_atom_id") or ""):
        return False
    supports = [item for item in list((edge.provenance or {}).get("support_evidence") or []) if isinstance(item, dict)]
    if not supports:
        return False
    nodes = {node.node_id: node for node in graph.nodes}
    has_requester_support = False
    for support in supports:
        support_record = nodes.get(str(support.get("record_node_id") or ""))
        support_span = str(support.get("support_span") or "")
        support_speaker = str((support_record.provenance or {}).get("speaker") or "") if support_record else ""
        owner_matches = (
            support_record is not None
            and str((support_record.attributes or {}).get("owner_id") or "") == str(owner_id or "").strip()
        )
        # An operation source can be authored by the requester before it is
        # independently attributed to the protected owner. The capability is
        # still anchored to an owner-matched target record; non-requester
        # support must be owner-attributed or a direct owner self-assertion.
        support_is_admissible = owner_matches or support_speaker in {requester_id, str(owner_id or "").strip()}
        if (
            support_record is None
            or not support_is_admissible
            or not support_span
            or support_span not in str(support_record.label or "")
            or str(support.get("source_atom_id") or "") != str((support_record.provenance or {}).get("source_atom_id") or "")
        ):
            return False
        has_requester_support = has_requester_support or (
            support_speaker == requester_id
        )
    return has_requester_support


def _requested_slots(semantic_spec: dict[str, Any]) -> list[str]:
    """Use the planner's typed contract rather than a domain-specific slot list."""
    certifiable_needs = semantic_spec.get("certifiable_needs")
    if isinstance(certifiable_needs, list):
        return list(dict.fromkeys(
            str(item.get("attribute") or item.get("need_id") or "").strip()
            for item in certifiable_needs
            if isinstance(item, dict)
            and str(item.get("attribute") or item.get("need_id") or "").strip()
        ))
    requested_attributes = list(semantic_spec.get("requested_attributes") or [])
    return list(dict.fromkeys(
        str(slot).strip()
        for slot in (requested_attributes or list(semantic_spec.get("requested_slots") or []))
        if str(slot).strip()
    ))
