"""Query-scoped policy authorization over closed graph evidence.

This stage is intentionally after semantic slot alignment. It lets an LLM
select a policy atom and already aligned SlotNode IDs, while deterministic code
checks every ID, relation, owner, and verbatim policy span before materializing
an ephemeral graph allow edge.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import md5
from typing import Any

from gov_mem.graph.governed_graph import GovernedMemoryGraph, GraphEdge, GraphNode
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


def attach_query_scoped_policy_authorizations(
    *,
    question: str = "",
    graph: GovernedMemoryGraph | None,
    semantic_alignment: dict[str, Any] | None,
    principal_relation_ledger: dict[str, Any],
    owner_id: str | None,
    governance_policy_atom_ids: set[str] | None,
    llm_client: LLMClient | None,
    model_name: str,
    utility_source_message_ids: set[str] | None = None,
    direct_policy_context_message_ids: set[str] | None = None,
) -> dict[str, Any]:
    if graph is None:
        return {"available": False, "reason": "graph_unavailable", "grants": []}
    relation = str(principal_relation_ledger.get("effective_relation") or "").strip()
    requester_id = str(principal_relation_ledger.get("requester_id") or "").strip()
    ledger_owner_id = str(principal_relation_ledger.get("owner_id") or "").strip()
    governed_owner_id = str(owner_id or "").strip()
    relation_proven = (
        str(principal_relation_ledger.get("effective_status") or "") == "proven"
        and bool(relation)
        and bool(requester_id)
        and ledger_owner_id == governed_owner_id
    )
    if not requester_id or not governed_owner_id:
        return {"available": False, "reason": "missing_requester_or_owner", "grants": []}
    aligned_slot_ids = _aligned_slot_ids(semantic_alignment)
    if not aligned_slot_ids:
        return {"available": False, "reason": "no_aligned_slots", "grants": []}
    nodes = {node.node_id: node for node in graph.nodes}
    # A source author can sometimes prove narrow operational stewardship of a
    # record even though the transcript does not establish a global personal
    # relation to its information owner.  This is intentionally evaluated
    # before broad policy authorization and is limited to aligned slots of the
    # exact source record; a role or title can never create this capability.
    if not relation_proven:
        stewardship_grants, stewardship_trace = _scoped_stewardship_grants(
            graph=graph,
            requester_id=requester_id,
            owner_id=governed_owner_id,
            aligned_slot_ids=aligned_slot_ids,
            semantic_alignment=semantic_alignment,
            utility_source_message_ids=utility_source_message_ids,
            llm_client=llm_client,
            model_name=model_name,
        )
        direct_policy_grants, direct_policy_trace = _direct_requester_policy_grants(
            graph=graph,
            question=question,
            requester_id=requester_id,
            owner_id=governed_owner_id,
            aligned_slot_ids=aligned_slot_ids,
            governance_policy_atom_ids=governance_policy_atom_ids,
            direct_policy_context_message_ids=direct_policy_context_message_ids,
            llm_client=llm_client,
            model_name=model_name,
        )
        grants = stewardship_grants + direct_policy_grants
        _materialize_grants(graph=graph, grants=grants)
        return {
            "available": bool(grants),
            "reason": (
                "scoped_stewardship_or_direct_policy_grants_attached"
                if grants else "principal_relation_unproven_no_scoped_stewardship_or_direct_policy"
            ),
            "grants": grants,
            "diagnostics": {
                "scoped_stewardship": stewardship_trace,
                "direct_requester_policy": direct_policy_trace,
            },
        }
    slot_candidates = [
        {
            "slot_node_id": node_id,
            "slot_name": str((nodes[node_id].attributes or {}).get("slot_name") or ""),
            "slot_value": str((nodes[node_id].attributes or {}).get("slot_value") or ""),
            "source_text": _source_text_for_slot(graph, node_id, nodes),
        }
        for node_id in sorted(aligned_slot_ids)
        if node_id in nodes and nodes[node_id].node_type == "SlotNode"
    ]
    policy_candidates = []
    for node in graph.nodes:
        if node.node_type != "PolicyNode":
            continue
        atom_id = str((node.provenance or {}).get("source_atom_id") or "")
        if governance_policy_atom_ids is not None and atom_id not in governance_policy_atom_ids:
            continue
        if str((node.attributes or {}).get("lifecycle") or "active").lower() != "active":
            continue
        policy_candidates.append({
            "policy_atom_id": atom_id,
            "policy_node_id": node.node_id,
            "policy_text": node.label,
        })
    if not slot_candidates or not policy_candidates:
        return {"available": False, "reason": "missing_aligned_slot_or_policy_evidence", "grants": []}
    if llm_client is None or not llm_client.is_available():
        return {"available": False, "reason": "policy_authorization_llm_unavailable", "grants": []}
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You align explicit policy evidence to already aligned memory slots. Do not answer the user, "
                "infer a policy, or expand access. Return JSON only."
            ),
            user_prompt=(
                "Return {\"grants\":[{\"policy_atom_id\":string,\"slot_node_ids\":[string],"
                "\"effect\":\"allow|deny|require_permission|none\",\"relation\":string,"
                "\"governed_owner_id\":string,\"policy_support_span\":string}]}. Select only listed policy "
                "atoms and SlotNode IDs. A grant is valid only if the policy text explicitly governs the selected "
                "field and the exact policy_support_span states the authorization/prohibition. A role/title alone "
                "is never enough. Omit uncertain grants.\n"
                f"Requester-owner relation: {relation}; requester_id: {requester_id}; owner_id: {ledger_owner_id}\n"
                f"Aligned slots: {slot_candidates}\nPolicy evidence: {policy_candidates}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return {"available": False, "reason": f"policy_authorization_unavailable:{type(exc).__name__}", "grants": []}

    policy_by_id = {row["policy_atom_id"]: row for row in policy_candidates if row["policy_atom_id"]}
    grants: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for item in _grant_items(raw):
        if not isinstance(item, dict):
            rejection_counts["non_object_grant"] += 1
            continue
        policy_atom_id = _first_string(item, "policy_atom_id", "policy_id", "atom_id")
        selected_ids = _slot_node_ids(item)
        policy = policy_by_id.get(policy_atom_id)
        support_span = _first_string(item, "policy_support_span", "support_span", "evidence_span")
        effect = _first_string(item, "effect", "authorization_effect").lower()
        item_relation = _canonical_relation(_first_string(item, "relation", "requester_owner_relation"))
        item_owner_id = _first_string(item, "governed_owner_id", "owner_id")
        if policy is None:
            rejection_counts["unknown_policy_atom_id"] += 1
            continue
        if not selected_ids or any(node_id not in aligned_slot_ids for node_id in selected_ids):
            rejection_counts["unknown_or_missing_slot_node_id"] += 1
            continue
        if effect not in {"allow", "deny", "require_permission"}:
            rejection_counts["invalid_effect"] += 1
            continue
        if item_relation != relation:
            rejection_counts["relation_mismatch"] += 1
            continue
        if item_owner_id != ledger_owner_id:
            rejection_counts["owner_mismatch"] += 1
            continue
        if not support_span or support_span not in str(policy.get("policy_text") or ""):
            rejection_counts["policy_span_not_source_grounded"] += 1
            continue
        grants.append({
            "policy_atom_id": policy_atom_id,
            "slot_node_ids": list(dict.fromkeys(selected_ids)),
            "effect": effect,
            "relation": relation,
            "governed_owner_id": ledger_owner_id,
            "policy_support_span": support_span,
            "authority_kind": "policy",
        })
    relation_trace: dict[str, Any] = {"attempted": False}
    if not grants:
        relation_grants, relation_trace = _relation_assignment_grants(
            llm_client=llm_client,
            model_name=model_name,
            relation=relation,
            requester_id=requester_id,
            owner_id=ledger_owner_id,
            ledger=principal_relation_ledger,
            slot_candidates=slot_candidates,
            aligned_slot_ids=aligned_slot_ids,
        )
        grants.extend(relation_grants)
    _materialize_grants(graph=graph, grants=grants)
    return {
        "available": bool(grants),
        "reason": "query_scoped_policy_grants_attached" if grants else "no_valid_query_scoped_policy_grant",
        "grants": grants,
        "diagnostics": {
            "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
            "candidate_policy_count": len(policy_candidates),
            "candidate_slot_count": len(slot_candidates),
            "rejected_grant_count": sum(rejection_counts.values()),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "relation_assignment_fallback": relation_trace,
        },
    }


def _direct_requester_policy_grants(
    *,
    graph: GovernedMemoryGraph,
    question: str,
    requester_id: str,
    owner_id: str,
    aligned_slot_ids: set[str],
    governance_policy_atom_ids: set[str] | None,
    direct_policy_context_message_ids: set[str] | None,
    llm_client: LLMClient | None,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adjudicate a policy that names a requester without inferring a relation.

    This is narrower than family/delegate/staff relation resolution: it can
    allow only the listed aligned slots and only when a visible policy source
    explicitly governs this requester. It cannot create a reusable relation.
    """
    trace: dict[str, Any] = {"attempted": False}
    if llm_client is None or not llm_client.is_available():
        trace["reason"] = "llm_unavailable"
        return [], trace
    nodes = {node.node_id: node for node in graph.nodes}
    slot_candidates = [
        {
            "slot_node_id": node_id,
            "slot_name": str((nodes[node_id].attributes or {}).get("slot_name") or ""),
            "slot_value": str((nodes[node_id].attributes or {}).get("slot_value") or ""),
            "source_text": _source_text_for_slot(graph, node_id, nodes),
        }
        for node_id in sorted(aligned_slot_ids)
        if node_id in nodes and nodes[node_id].node_type == "SlotNode"
    ]
    policy_candidates = []
    context_message_ids = {str(value) for value in set(direct_policy_context_message_ids or set()) if str(value)}
    for node in graph.nodes:
        # A direct requester policy is an authority source, never an ordinary
        # operational fact.  Facts remain available to Stage 1/3 but cannot
        # become an allow edge simply because they describe a current value.
        if node.node_type != "PolicyNode":
            continue
        atom_id = str((node.provenance or {}).get("source_atom_id") or "")
        source_message_ids = {
            str(value) for value in list((node.provenance or {}).get("source_message_ids") or []) if str(value)
        }
        is_selected_policy = governance_policy_atom_ids is None or atom_id in governance_policy_atom_ids
        is_selected_context = bool(source_message_ids & context_message_ids)
        if not (is_selected_policy or is_selected_context):
            continue
        if str((node.attributes or {}).get("lifecycle") or "active").lower() != "active":
            continue
        policy_candidates.append({
            "policy_atom_id": atom_id,
            "policy_text": node.label,
            "source_message_ids": sorted(source_message_ids),
            "selected_as_authorization_context": is_selected_context,
        })
    if not slot_candidates or not policy_candidates:
        trace["reason"] = "missing_aligned_slot_or_policy_evidence"
        return [], trace
    trace.update({"attempted": True, "candidate_policy_count": len(policy_candidates)})
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You verify a direct, requester-specific policy capability over already aligned memory slots. "
                "Return JSON only. Do not infer a family, staff, delegate, or owner relationship. Grant only when "
                "the exact listed policy text explicitly identifies the requester or unambiguously addresses that "
                "requester's request, and explicitly permits the selected operational slot while withholding other "
                "requested content. A role, title, generic category, or merely being related is not sufficient. "
                "A factual status, budget, contract, recap, or record is never itself permission evidence."
            ),
            user_prompt=(
                "Return {\"grants\":[{\"policy_atom_id\":string,\"slot_node_ids\":[string],\"effect\":\"allow\","
                "\"requester_id\":string,\"governed_owner_id\":string,\"requester_reference_span\":string,"
                "\"permission_support_span\":string,\"permission_semantics\":\"explicit_allow|not_allow\"}]}. "
                "Select only listed policy IDs and aligned slot IDs. Return a grant only with permission_semantics="
                "explicit_allow. Use not_allow for every prohibition, restriction, warning, scope boundary, or "
                "non-permission statement and omit it from grants. "
                "Both requester_reference_span and permission_support_span must be distinct, nonempty exact substrings "
                "of the selected policy text. The first must identify or unambiguously address the listed requester; "
                "the second must explicitly permit disclosure of the selected operational slot. A prohibition, "
                "restriction, warning, scope boundary, status fact, or statement that merely describes the field is "
                "not permission. The question is supplied only to resolve references such as 'that "
                "question' or 'the time'; it never grants access. Omit uncertain grants. Never grant an unaligned "
                "attribute.\n"
                f"Question: {question}\n"
                f"Requester ID: {requester_id}; information owner ID: {owner_id}\n"
                f"Aligned slots: {slot_candidates}\nPolicy evidence: {policy_candidates}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        trace["reason"] = f"llm_error:{type(exc).__name__}"
        return [], trace
    policy_by_id = {item["policy_atom_id"]: item for item in policy_candidates if item["policy_atom_id"]}
    grants: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for item in _grant_items(raw):
        policy_atom_id = _first_string(item, "policy_atom_id", "policy_id", "atom_id")
        selected_ids = _slot_node_ids(item)
        requester_span = _first_string(item, "requester_reference_span", "requester_span")
        permission_span = _first_string(item, "permission_support_span", "permission_span")
        policy = policy_by_id.get(policy_atom_id)
        if policy is None:
            rejection_counts["unknown_policy_atom_id"] += 1
            continue
        if not selected_ids or any(node_id not in aligned_slot_ids for node_id in selected_ids):
            rejection_counts["unknown_or_missing_slot_node_id"] += 1
            continue
        if _first_string(item, "effect", "authorization_effect").lower() != "allow":
            rejection_counts["invalid_effect"] += 1
            continue
        if _first_string(item, "permission_semantics", "permission_kind") != "explicit_allow":
            rejection_counts["permission_not_explicit_allow"] += 1
            continue
        if _first_string(item, "requester_id", "principal_id") != requester_id:
            rejection_counts["requester_mismatch"] += 1
            continue
        if _first_string(item, "governed_owner_id", "owner_id") != owner_id:
            rejection_counts["owner_mismatch"] += 1
            continue
        policy_text = str(policy["policy_text"] or "")
        if not requester_span or requester_span not in policy_text:
            rejection_counts["requester_reference_not_source_grounded"] += 1
            continue
        if not permission_span or permission_span not in policy_text:
            rejection_counts["permission_span_not_source_grounded"] += 1
            continue
        grants.append({
            "policy_atom_id": policy_atom_id,
            "slot_node_ids": list(dict.fromkeys(selected_ids)),
            "effect": "allow",
            "relation": "direct_requester_policy",
            "requester_id": requester_id,
            "governed_owner_id": owner_id,
            "policy_support_span": permission_span,
            "requester_reference_span": requester_span,
            "authority_kind": "direct_requester_policy",
        })
    trace.update({
        "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
        "validated_grant_count": len(grants),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    })
    return grants, trace


def _scoped_stewardship_grants(
    *,
    graph: GovernedMemoryGraph,
    requester_id: str,
    owner_id: str,
    aligned_slot_ids: set[str],
    semantic_alignment: dict[str, Any] | None,
    utility_source_message_ids: set[str] | None,
    llm_client: LLMClient | None,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Certify an ephemeral capability over a source-authored record only.

    This is deliberately distinct from requester-owner relation resolution.
    It handles observable operational custody such as an actor creating or
    maintaining one concrete record, while denying access to every other
    record of the same owner unless that record independently carries proof.
    """
    trace: dict[str, Any] = {"attempted": False}
    if llm_client is None or not llm_client.is_available():
        trace["reason"] = "llm_unavailable"
        return [], trace
    candidates = _scoped_stewardship_candidates(
        graph=graph,
        requester_id=requester_id,
        owner_id=owner_id,
        aligned_slot_ids=aligned_slot_ids,
        utility_source_message_ids=utility_source_message_ids,
    )
    if not candidates:
        trace["reason"] = "no_requester_authored_owner_matched_record"
        return [], trace
    trace.update({"attempted": True, "candidate_count": len(candidates)})
    collection_request = any(
        len(list(binding.get("anchor_slot_node_ids") or [])) > 1
        for binding in dict((semantic_alignment or {}).get("bindings") or {}).values()
        if isinstance(binding, dict)
    )
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You verify a narrowly scoped operational-stewardship capability over an observed record. "
                "Return JSON only. A job title, organization, identity prefix, or merely reporting a fact is "
                "not authority. Grant only when the exact source text, spoken by the requester, explicitly "
                "establishes that they create, hold, update, maintain, cancel, confirm, or otherwise operate the "
                "specific listed record. The capability applies only to the listed slots of that same record; do "
                "not infer a general relationship to the information owner or access to other records. Omit doubt."
            ),
            user_prompt=(
                "Return {\"grants\":[{\"candidate_id\":string,\"slot_node_ids\":[string],\"effect\":\"allow\","
                "\"governed_owner_id\":string,\"supports\":[{\"record_node_id\":string,\"support_span\":string}]}]}. "
                "Select only listed candidate IDs and their listed SlotNode IDs. Each support_span must be an exact "
                "substring of its listed source. Grant only if the requester-authored source itself establishes "
                "operational stewardship, OR a requester operation plus the owner's exact confirmation establish "
                "the same record lifecycle. Supporting sources never expand the selected SlotNode scope. "
                + (
                    "This is a record collection request: when one operation source governs multiple listed target "
                    "records, emit one grant for every such target record; do not return an arbitrary partial subset. "
                    if collection_request else ""
                ) + "\n"
                f"Requester ID: {requester_id}; information owner ID: {owner_id}\n"
                f"Record candidates: {candidates}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        trace["reason"] = f"llm_error:{type(exc).__name__}"
        return [], trace
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    grants: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for item in _grant_items(raw):
        candidate = candidate_by_id.get(_first_string(item, "candidate_id", "record_id", "authority_id"))
        selected_ids = _slot_node_ids(item)
        supports = _scoped_supports(item, candidate)
        if candidate is None:
            rejection_counts["unknown_candidate"] += 1
            continue
        if _first_string(item, "effect", "authorization_effect").lower() != "allow":
            rejection_counts["invalid_effect"] += 1
            continue
        if _first_string(item, "governed_owner_id", "owner_id") != owner_id:
            rejection_counts["owner_mismatch"] += 1
            continue
        if not selected_ids or any(node_id not in candidate["slot_node_ids"] for node_id in selected_ids):
            rejection_counts["slot_outside_record_scope"] += 1
            continue
        if not supports:
            rejection_counts["support_not_source_grounded"] += 1
            continue
        support_by_record = {source["record_node_id"]: source for source in candidate["support_sources"]}
        if any(
            support["record_node_id"] not in support_by_record
            or support["support_span"] not in support_by_record[support["record_node_id"]]["source_text"]
            for support in supports
        ):
            rejection_counts["support_not_source_grounded"] += 1
            continue
        if not any(
            support_by_record[support["record_node_id"]]["speaker_id"] == requester_id
            for support in supports
        ):
            rejection_counts["no_requester_operation_support"] += 1
            continue
        grants.append({
            "capability_id": "stewardship::" + md5(
                f"{requester_id}|{owner_id}|{candidate['record_node_id']}|{supports}".encode("utf-8")
            ).hexdigest()[:16],
            "record_node_id": candidate["record_node_id"],
            "source_atom_id": candidate["source_atom_id"],
            "source_message_ids": list(dict.fromkeys(
                message_id
                for support in supports
                for message_id in support_by_record[support["record_node_id"]]["source_message_ids"]
            )),
            "slot_node_ids": list(dict.fromkeys(selected_ids)),
            "effect": "allow",
            "requester_id": requester_id,
            "governed_owner_id": owner_id,
            "supports": [{
                **support,
                "source_atom_id": support_by_record[support["record_node_id"]]["source_atom_id"],
                "source_message_ids": support_by_record[support["record_node_id"]]["source_message_ids"],
                "speaker_id": support_by_record[support["record_node_id"]]["speaker_id"],
            } for support in supports],
            "authority_kind": "scoped_stewardship",
        })
    lifecycle_trace: dict[str, Any] = {"attempted": False}
    # A partial capability set cannot realize a requested record collection.
    # Recheck lifecycle bindings over the same closed candidates rather than
    # treating one record's capability as permission for its siblings.
    granted_record_ids = {str(grant.get("record_node_id") or "") for grant in grants}
    aligned_record_ids = {str(candidate.get("record_node_id") or "") for candidate in candidates}
    needs_collection_completion = collection_request and granted_record_ids != aligned_record_ids
    if not grants or needs_collection_completion:
        lifecycle_grants, lifecycle_trace = _operation_to_state_stewardship_grants(
            candidates=candidates,
            requester_id=requester_id,
            owner_id=owner_id,
            llm_client=llm_client,
            model_name=model_name,
            require_all_candidates=needs_collection_completion,
        )
        existing_records = {str(grant.get("record_node_id") or "") for grant in grants}
        grants.extend(
            grant for grant in lifecycle_grants
            if str(grant.get("record_node_id") or "") not in existing_records
        )
    trace.update({
        "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
        "raw_response": raw if isinstance(raw, dict) else {},
        "validated_grant_count": len(grants),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "operation_to_state_recovery": lifecycle_trace,
    })
    return grants, trace


def _operation_to_state_stewardship_grants(
    *,
    candidates: list[dict[str, Any]],
    requester_id: str,
    owner_id: str,
    llm_client: LLMClient,
    model_name: str,
    require_all_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind an operation source to a later rendition of the same record.

    A record compiler can legally decompose an operation and a current-state
    rendition into distinct atoms. This recovery never merges them by string
    rules: an LLM must select both closed IDs and exact spans, while the final
    grant remains scoped to the selected target record only.
    """
    trace: dict[str, Any] = {"attempted": False}
    operation_sources = {
        source["record_node_id"]: source
        for candidate in candidates
        for source in list(candidate.get("support_sources") or [])
        if isinstance(source, dict) and str(source.get("speaker_id") or "") == requester_id
    }
    if not candidates or not operation_sources:
        trace["reason"] = "missing_target_or_requester_operation_source"
        return [], trace
    trace["attempted"] = True
    target_payload = [{
        "candidate_id": candidate["candidate_id"],
        "record_node_id": candidate["record_node_id"],
        "source_text": candidate["source_text"],
        "slot_node_ids": candidate["slot_node_ids"],
    } for candidate in candidates]
    operation_payload = list(operation_sources.values())
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You verify a narrow operational-stewardship lifecycle binding. Return JSON only. A title, role, "
                "organization, or generic similarity is not proof. Select a target record only when an exact "
                "requester-authored operation source establishes operation over that same concrete record and the "
                "target is a later or current rendition of it. Do not authorize any other record or owner."
            ),
            user_prompt=(
                "Return {\"grants\":[{\"target_candidate_id\":string,\"operation_record_node_id\":string,"
                "\"slot_node_ids\":[string],\"effect\":\"allow\",\"governed_owner_id\":string,"
                "\"operation_support_span\":string}]}. Select only listed IDs. slot_node_ids must belong to the "
                "target candidate. operation_support_span must be an exact substring of the operation source. "
                + (
                    "This is a collection-completion pass. Adjudicate every listed target candidate independently; "
                    "return a grant for each target with proof and omit only targets lacking proof. Do not stop after one. "
                    if require_all_candidates else ""
                ) + "Omit uncertain grants.\n"
                f"Requester ID: {requester_id}; information owner ID: {owner_id}\n"
                f"Target current-record candidates: {target_payload}\n"
                f"Requester operation sources: {operation_payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        trace["reason"] = f"llm_error:{type(exc).__name__}"
        return [], trace
    candidate_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    grants: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = defaultdict(int)
    for item in _grant_items(raw):
        candidate = candidate_by_id.get(_first_string(item, "target_candidate_id", "candidate_id", "record_id"))
        operation = operation_sources.get(_first_string(item, "operation_record_node_id", "operation_source_id"))
        slot_node_ids = _slot_node_ids(item)
        support_span = _first_string(item, "operation_support_span", "support_span", "evidence_span")
        if candidate is None or operation is None:
            rejection_counts["unknown_target_or_operation_source"] += 1
            continue
        if _first_string(item, "effect", "authorization_effect").lower() != "allow":
            rejection_counts["invalid_effect"] += 1
            continue
        if _first_string(item, "governed_owner_id", "owner_id") != owner_id:
            rejection_counts["owner_mismatch"] += 1
            continue
        if not slot_node_ids or any(slot_id not in candidate["slot_node_ids"] for slot_id in slot_node_ids):
            rejection_counts["slot_outside_target_record_scope"] += 1
            continue
        if not support_span or support_span not in operation["source_text"]:
            rejection_counts["operation_support_not_source_grounded"] += 1
            continue
        grants.append({
            "capability_id": "stewardship::" + md5(
                f"{requester_id}|{owner_id}|{candidate['record_node_id']}|{operation['record_node_id']}|{support_span}".encode("utf-8")
            ).hexdigest()[:16],
            "record_node_id": candidate["record_node_id"],
            "source_atom_id": candidate["source_atom_id"],
            "source_message_ids": operation["source_message_ids"],
            "slot_node_ids": list(dict.fromkeys(slot_node_ids)),
            "effect": "allow",
            "requester_id": requester_id,
            "governed_owner_id": owner_id,
            "supports": [{
                "record_node_id": operation["record_node_id"],
                "support_span": support_span,
                "source_atom_id": operation["source_atom_id"],
                "source_message_ids": operation["source_message_ids"],
                "speaker_id": operation["speaker_id"],
            }],
            "authority_kind": "scoped_stewardship",
        })
    trace.update({
        "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
        "raw_response": raw if isinstance(raw, dict) else {},
        "validated_grant_count": len(grants),
        "rejection_counts": dict(sorted(rejection_counts.items())),
    })
    return grants, trace


def _scoped_stewardship_candidates(
    *,
    graph: GovernedMemoryGraph,
    requester_id: str,
    owner_id: str,
    aligned_slot_ids: set[str],
    utility_source_message_ids: set[str] | None,
) -> list[dict[str, Any]]:
    nodes = {node.node_id: node for node in graph.nodes}
    aligned_slots_by_record: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.edge_type != "has_slot" or edge.target_id not in aligned_slot_ids:
            continue
        record = nodes.get(edge.source_id)
        if record is None:
            continue
        if str((record.attributes or {}).get("owner_id") or "") != owner_id:
            continue
        if str((record.provenance or {}).get("speaker") or "") != requester_id:
            continue
        aligned_slots_by_record[record.node_id].append(edge.target_id)
    support_sources = _owner_matched_support_sources(
        graph=graph,
        owner_id=owner_id,
        requester_id=requester_id,
        utility_source_message_ids=utility_source_message_ids,
    )
    candidates: list[dict[str, Any]] = []
    for record_node_id, aligned_slot_ids_for_record in sorted(aligned_slots_by_record.items()):
        record = nodes[record_node_id]
        source_text = str(record.label or "").strip()
        source_atom_id = str((record.provenance or {}).get("source_atom_id") or "").strip()
        source_message_ids = [
            str(value) for value in list((record.provenance or {}).get("source_message_ids") or []) if str(value)
        ]
        if not source_text or not source_atom_id or not source_message_ids:
            continue
        # A collection member is an evidence-local record rather than a
        # detached scalar. The LLM still must explicitly select each field it
        # grants, but it needs the complete record-local closed set to certify
        # an identifying projection (for example, time plus record type).
        record_slot_ids = [
            edge.target_id
            for edge in graph.edges
            if edge.edge_type == "has_slot" and edge.source_id == record_node_id
            and edge.target_id in nodes and nodes[edge.target_id].node_type == "SlotNode"
        ]
        candidate_id = "stewardship_candidate::" + md5(
            f"{requester_id}|{owner_id}|{record_node_id}".encode("utf-8")
        ).hexdigest()[:16]
        candidate_source = {
            "record_node_id": record_node_id,
            "source_atom_id": source_atom_id,
            "source_message_ids": source_message_ids,
            "speaker_id": requester_id,
            "source_text": source_text,
            "source_turn": int((record.provenance or {}).get("source_turn") or 10**9),
        }
        candidate_support_sources = list(support_sources)
        if not any(source["record_node_id"] == record_node_id for source in candidate_support_sources):
            candidate_support_sources.append(candidate_source)
        candidates.append({
            "candidate_id": candidate_id,
            "record_node_id": record_node_id,
            "source_atom_id": source_atom_id,
            "source_message_ids": source_message_ids,
            "source_text": source_text,
            "aligned_slot_node_ids": sorted(set(aligned_slot_ids_for_record)),
            "slot_node_ids": sorted(set(record_slot_ids)),
            "support_sources": candidate_support_sources,
        })
    return candidates


def _owner_matched_support_sources(
    *,
    graph: GovernedMemoryGraph,
    owner_id: str,
    requester_id: str,
    utility_source_message_ids: set[str] | None,
) -> list[dict[str, Any]]:
    """Expose closed, same-owner episode sources for a lifecycle proof."""
    sources: list[dict[str, Any]] = []
    seen_source_messages: set[tuple[str, str, str]] = set()
    for node in graph.nodes:
        # The atom compiler may represent an operational utterance as a role
        # frame. It is admissible only when it is a requester-authored source
        # already selected by this query's utility closure; it never becomes a
        # role/title authorization on its own.
        if node.node_type not in {"FactNode", "EventNode", "RoleNode"}:
            continue
        provenance = dict(node.provenance or {})
        source_text = str(node.label or "").strip()
        source_atom_id = str(provenance.get("source_atom_id") or "").strip()
        speaker_id = str(provenance.get("speaker") or "").strip()
        source_message_ids = [str(value) for value in list(provenance.get("source_message_ids") or []) if str(value)]
        record_owner_id = str((node.attributes or {}).get("owner_id") or "")
        is_owner_attributed = record_owner_id == owner_id
        is_owner_self_assertion = speaker_id == owner_id
        is_requester_utility_source = (
            speaker_id == requester_id
            and utility_source_message_ids is not None
            and bool(set(source_message_ids) & set(utility_source_message_ids))
        )
        if node.node_type == "RoleNode" and not is_requester_utility_source:
            continue
        # Target records carry their own owner attribution. Lifecycle support
        # only needs the requester's query-local operation source and an owner
        # self assertion; including every same-owner graph atom duplicates a
        # long transcript and can make the LLM boundary time out.
        if not (is_owner_self_assertion or is_requester_utility_source):
            continue
        if not source_text or not source_atom_id or not speaker_id or not source_message_ids:
            continue
        source_key = ("|".join(source_message_ids), speaker_id, source_text)
        if source_key in seen_source_messages:
            continue
        seen_source_messages.add(source_key)
        source = {
            "record_node_id": node.node_id,
            "source_atom_id": source_atom_id,
            "source_message_ids": source_message_ids,
            "speaker_id": speaker_id,
            "source_text": source_text,
            "source_turn": int(provenance.get("source_turn") or 10**9),
        }
        if source not in sources:
            sources.append(source)
    return sorted(sources, key=lambda item: (item["source_turn"], item["record_node_id"]))[:8]


def _scoped_supports(item: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, str]]:
    raw_supports = item.get("supports", item.get("stewardship_supports", []))
    if not isinstance(raw_supports, list):
        raw_supports = [raw_supports]
    supports: list[dict[str, str]] = []
    for raw in raw_supports:
        if not isinstance(raw, dict):
            continue
        record_node_id = _first_string(raw, "record_node_id", "source_record_node_id", "source_id")
        support_span = _first_string(raw, "support_span", "evidence_span", "span")
        if record_node_id and support_span:
            support = {"record_node_id": record_node_id, "support_span": support_span}
            if support not in supports:
                supports.append(support)
    # Backward-compatible direct proof shape. It remains restricted to the
    # candidate record and is useful when the source itself is unequivocal.
    if not supports:
        support_span = _first_string(item, "support_span", "evidence_span", "policy_support_span")
        if support_span:
            supports.append({"record_node_id": candidate["record_node_id"], "support_span": support_span})
    return supports


def _relation_assignment_grants(
    *,
    llm_client: LLMClient,
    model_name: str,
    relation: str,
    requester_id: str,
    owner_id: str,
    ledger: dict[str, Any],
    slot_candidates: list[dict[str, str]],
    aligned_slot_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use a proven case-assignment relation as a narrowly scoped authority fallback."""
    trace: dict[str, Any] = {"attempted": False}
    if relation != "authorized_staff":
        return [], trace
    candidates = _relation_assignment_candidates(ledger, requester_id=requester_id, owner_id=owner_id)
    if not candidates:
        return [], trace
    trace["attempted"] = True
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You verify whether a proven, source-grounded staff case assignment authorizes disclosure of "
                "already aligned record slots. Return JSON only. Do not infer a relationship, expand its scope, "
                "or use a title as evidence. Select only a listed assignment evidence ID and slots whose source "
                "facts are relevant to that exact assignment. Omit uncertain grants."
            ),
            user_prompt=(
                "Return {\"grants\":[{\"assignment_id\":string,\"slot_node_ids\":[string],"
                "\"effect\":\"allow\",\"relation\":string,\"governed_owner_id\":string,"
                "\"support_span\":string}]}.\n"
                f"Requester-owner relation: {relation}; requester_id: {requester_id}; owner_id: {owner_id}\n"
                f"Aligned slots: {slot_candidates}\nAssignment evidence: {candidates}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return [], trace
    candidate_by_id = {item["assignment_id"]: item for item in candidates}
    grants: list[dict[str, Any]] = []
    for item in _grant_items(raw):
        assignment_id = _first_string(item, "assignment_id", "authority_id", "policy_atom_id", "atom_id")
        candidate = candidate_by_id.get(assignment_id)
        selected_ids = _slot_node_ids(item)
        support_span = _first_string(item, "support_span", "policy_support_span", "evidence_span")
        if (
            candidate is None
            or not selected_ids
            or any(node_id not in aligned_slot_ids for node_id in selected_ids)
            or _first_string(item, "effect", "authorization_effect").lower() != "allow"
            or _canonical_relation(_first_string(item, "relation", "requester_owner_relation")) != relation
            or _first_string(item, "governed_owner_id", "owner_id") != owner_id
            or support_span != candidate["support_span"]
        ):
            continue
        grants.append({
            "policy_atom_id": assignment_id,
            "slot_node_ids": list(dict.fromkeys(selected_ids)),
            "effect": "allow",
            "relation": relation,
            "governed_owner_id": owner_id,
            "policy_support_span": support_span,
            "authority_kind": "relation_assignment",
        })
    trace.update({
        "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
        "candidate_count": len(candidates),
        "validated_grant_count": len(grants),
    })
    return grants, trace


def _relation_assignment_candidates(
    ledger: dict[str, Any], *, requester_id: str, owner_id: str
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for record in list(ledger.get("records") or []):
        if not isinstance(record, dict) or (
            str(record.get("requester_id") or "") != requester_id
            or str(record.get("owner_id") or "") != owner_id
            or str(record.get("relation") or "") != "authorized_staff"
            or str(record.get("status") or "") != "proven"
        ):
            continue
        for support in list(record.get("supports") or []):
            if not isinstance(support, dict):
                continue
            kind = str(support.get("evidence_kind") or "").lower()
            span = str(support.get("source_span") or "").strip()
            message_id = str(support.get("message_id") or "").strip()
            if kind not in {"explicit_assignment", "explicit_authorization"} or not span or not message_id:
                continue
            assignment_id = "relation_assignment::" + md5(
                f"{requester_id}|{owner_id}|{message_id}|{span}".encode("utf-8")
            ).hexdigest()[:16]
            candidate = {"assignment_id": assignment_id, "support_span": span, "message_id": message_id}
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _grant_items(raw: object) -> list[dict[str, Any]]:
    """Normalize provider-neutral grant containers before closed-set checks."""
    if not isinstance(raw, dict):
        return []
    containers: list[object] = [raw.get("grants"), raw.get("authorizations")]
    for envelope_key in ("query_policy_authorization", "policy_authorization", "result", "data"):
        envelope = raw.get(envelope_key)
        if isinstance(envelope, dict):
            containers.extend([envelope.get("grants"), envelope.get("authorizations")])
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


def _first_string(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, list):
            value = next((entry for entry in value if str(entry).strip()), "")
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _canonical_relation(value: str) -> str:
    """Normalize only provider schema aliases, never transcript wording."""
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "self": "owner",
        "self_owner": "owner",
        "family_member": "family",
        "relative": "family",
        "delegated": "delegate",
        "authorized_staff_member": "authorized_staff",
    }
    return aliases.get(normalized, normalized)


def _slot_node_ids(item: dict[str, Any]) -> list[str]:
    raw_ids = item.get("slot_node_ids", item.get("slot_ids", item.get("slots", [])))
    if not isinstance(raw_ids, list):
        raw_ids = [raw_ids]
    result: list[str] = []
    for value in raw_ids:
        if isinstance(value, dict):
            value = value.get("slot_node_id", value.get("slot_id", ""))
        node_id = str(value or "").strip()
        if node_id and node_id not in result:
            result.append(node_id)
    return result


def _aligned_slot_ids(semantic_alignment: dict[str, Any] | None) -> set[str]:
    result: set[str] = set()
    for binding in dict((semantic_alignment or {}).get("bindings") or {}).values():
        if not isinstance(binding, dict):
            continue
        result.update(str(value) for value in list(binding.get("anchor_slot_node_ids") or []) if str(value))
        if str(binding.get("anchor_slot_node_id") or ""):
            result.add(str(binding["anchor_slot_node_id"]))
    return result


def _source_text_for_slot(graph: GovernedMemoryGraph, slot_node_id: str, nodes: dict[str, GraphNode]) -> str:
    semantic_edge = next((
        edge for edge in graph.edges if edge.edge_type == "has_slot" and edge.target_id == slot_node_id
    ), None)
    semantic_node = nodes.get(semantic_edge.source_id) if semantic_edge is not None else None
    return str(semantic_node.label if semantic_node is not None else "")


def _materialize_grants(*, graph: GovernedMemoryGraph, grants: list[dict[str, Any]]) -> None:
    for grant in grants:
        if grant.get("authority_kind") == "scoped_stewardship":
            _materialize_scoped_stewardship_grant(graph=graph, grant=grant)
            continue
        role_node_id = f"role::{grant['relation']}"
        graph.add_node(GraphNode(
            node_id=role_node_id,
            node_type="RoleNode",
            label=str(grant["relation"]),
            attributes={"scope": str(grant["relation"])},
        ))
        for slot_node_id in grant["slot_node_ids"]:
            edge_type = "allows" if grant["effect"] == "allow" else (
                "denies" if grant["effect"] == "deny" else "requires_permission"
            )
            edge_key = f"query_policy:{grant['policy_atom_id']}:{edge_type}:{role_node_id}:{slot_node_id}"
            graph.add_edge(GraphEdge(
                edge_id=md5(edge_key.encode("utf-8")).hexdigest()[:16],
                edge_type=edge_type,
                source_id=role_node_id,
                target_id=slot_node_id,
                attributes={"query_scoped": True},
                provenance={
                    "source_atom_id": grant["policy_atom_id"],
                    "evidence_span": grant["policy_support_span"],
                    "query_policy_authorization": True,
                    "relation_assignment_authorization": grant.get("authority_kind") == "relation_assignment",
                    "direct_requester_policy_authorization": grant.get("authority_kind") == "direct_requester_policy",
                    "requester_id": grant.get("requester_id"),
                    "governed_owner_id": grant.get("governed_owner_id"),
                },
            ))


def _materialize_scoped_stewardship_grant(*, graph: GovernedMemoryGraph, grant: dict[str, Any]) -> None:
    """Attach a capability node whose graph scope is one source record."""
    capability_id = str(grant["capability_id"])
    requester_id = str(grant["requester_id"])
    owner_id = str(grant["governed_owner_id"])
    record_node_id = str(grant["record_node_id"])
    slot_node_ids = list(dict.fromkeys(str(value) for value in grant["slot_node_ids"] if str(value)))
    supports = [dict(item) for item in list(grant.get("supports") or []) if isinstance(item, dict)]
    provenance = {
        "source_atom_id": str(grant["source_atom_id"]),
        "source_message_ids": list(grant["source_message_ids"]),
        "support_evidence": supports,
        "scoped_stewardship_authorization": True,
    }
    graph.add_node(GraphNode(f"principal::{requester_id}", "PrincipalNode", requester_id))
    graph.add_node(GraphNode(f"principal::{owner_id}", "PrincipalNode", owner_id))
    graph.add_node(GraphNode(
        capability_id,
        "ScopedCapabilityNode",
        "scoped_operational_stewardship",
        attributes={
            "requester_id": requester_id,
            "owner_id": owner_id,
            "record_node_id": record_node_id,
            "slot_node_ids": slot_node_ids,
            "support_record_node_ids": [str(item["record_node_id"]) for item in supports],
            "status": "proven",
        },
        provenance=provenance,
    ))
    for edge_type, source_id, target_id in (
        ("holds_capability", f"principal::{requester_id}", capability_id),
        ("capability_owner", capability_id, f"principal::{owner_id}"),
        ("stewards_record", capability_id, record_node_id),
    ):
        edge_key = f"{edge_type}:{source_id}:{target_id}"
        graph.add_edge(GraphEdge(
            edge_id=md5(edge_key.encode("utf-8")).hexdigest()[:16],
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            provenance=dict(provenance),
        ))
    for slot_node_id in slot_node_ids:
        edge_key = f"scoped_stewardship:{capability_id}:{slot_node_id}"
        graph.add_edge(GraphEdge(
            edge_id=md5(edge_key.encode("utf-8")).hexdigest()[:16],
            edge_type="allows",
            source_id=capability_id,
            target_id=slot_node_id,
            attributes={"query_scoped": True, "scope": "single_record"},
            provenance=dict(provenance),
        ))
