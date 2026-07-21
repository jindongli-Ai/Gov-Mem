"""Closed-set LLM reranking for Stage 2 of the governed memory pipeline.

The model judges relevance and possible redaction only.  It cannot create
evidence or grant access: callers validate IDs/spans and the graph certificate
remains the disclosure authority.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.governance_runtime.claim_adjudicator import (
    _is_collection_attribute,
    _meaningful_field_tokens,
    _slot_matches_attribute,
)


_ADMISSIBLE = {"answer_member", "redactable_member", "safe_projection_member"}
_CLASSES = _ADMISSIBLE | {"policy_only", "deleted_or_historical", "unrelated"}


def semantic_rerank_evidence(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    requester_id: str | None,
    owner_id: str | None,
    relation_to_owner: str | None,
    llm_client: LLMClient | None,
    model_name: str,
) -> tuple[list[RetrievedEvidence], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Return source-validated relevant active records and a complete audit."""
    candidates: dict[str, RetrievedEvidence] = {}
    filtered: list[dict[str, str]] = []
    for row in evidence:
        memory_id = str(row.memory_id or "").strip()
        if not memory_id:
            continue
        if not _is_active(row):
            filtered.append({"memory_id": memory_id, "reason": "nonactive_lifecycle"})
            continue
        # Closed IDs make model output auditable even if two retrieval paths
        # propose the same source record.
        prior = candidates.get(memory_id)
        if prior is None or float(row.score) > float(prior.score):
            candidates[memory_id] = row
    if not candidates:
        return [], [], filtered, {"available": True, "reason": "no_active_candidates", "candidate_count": 0}
    if llm_client is None or not llm_client.is_available():
        return [], [], filtered + [{"memory_id": key, "reason": "stage2_llm_unavailable"} for key in candidates], {
            "available": False, "reason": "stage2_llm_unavailable", "candidate_count": len(candidates)
        }

    payload = [
        {
            "memory_id": memory_id,
            "source_message_ids": list(row.source_message_ids or []),
            "source_text": str(row.content or ""),
            "timestamp": str(row.time or ""),
            "retrieval_score": float(row.score),
            "typed_slots": _typed_slot_payload(row),
        }
        for memory_id, row in candidates.items()
    ]
    requested_attributes = [
        str(value).strip()
        for value in (
            semantic_spec.get("requested_attributes")
            or semantic_spec.get("requested_slots")
            or []
        )
        if str(value).strip()
    ]
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You perform Stage 2 semantic reranking over a closed set of active, source-grounded memory records. "
                "Classify relevance to the question; do not answer, infer facts, create records, or grant access. "
                "Privacy authorization is decided later from graph evidence. Return JSON only."
            ),
            user_prompt=(
                "Return {\"decisions\":[{\"memory_id\":string,\"classification\":"
                "\"answer_member|redactable_member|safe_projection_member|policy_only|deleted_or_historical|unrelated\",\"support_span\":string,"
                "\"served_attributes\":[string],\"safe_projection_slots\":[{\"slot_name\":string,\"value\":string}],"
                "\"typed_fields\":[{\"attribute\":string,\"slot_name\":string,\"value\":string}],"
                "\"authorized_for_requester\":boolean,\"authorization_support_span\":string}]}. "
                "Give exactly one decision for every listed memory_id. answer_member means the record is needed to "
                "answer the question; redactable_member means it is needed but may require partial redaction later; "
                "policy_only is context rather than an answer record. support_span must be a nonempty exact substring "
                "of that record's source_text. For a multi-part question, retain every complementary record needed for "
                "a complete answer, not merely the highest-scoring one. Audit coverage across the whole closed set "
                "before deciding. A current/latest/as-of-now/currently-approved constraint is a state-selection "
                "constraint: for each requested attribute, retain only the final canonical record that satisfies it. "
                "Classify earlier, replaced, preliminary, or narrower corroborating values as deleted_or_historical "
                "for the superseded field even when their storage lifecycle is still active. Supersession is field-level: "
                "do not discard an otherwise active record when a later update replaces only one field; retain its "
                "complementary requested fields unless each has a later replacement. Never include conflicting historical "
                "and current values for the same field merely because both mention the same entity. "
                "When two records state the same current operational attribute, compare their timestamps and explicit "
                "supersession language: a later authoritative record that replaces or confirms a value is canonical over "
                "an earlier status recap. Do not preserve an earlier number merely because its wording says current. "
                "When timestamps are absent, source_message_ids containing tNN are chronological turn identifiers; use "
                "the highest relevant turn as later evidence, while still respecting explicit non-authoritative or "
                "request discourse annotations. "
                "before deciding: a current summary, checkpoint note, recap, or status record is an answer_member when "
                "it states any requested operational value, even if its source uses different field wording or combines "
                "several requested values in one phrase. Preserve the record with the most complete current scope instead "
                "of replacing it with a narrower corroborating record. Do not classify a source-grounded current factual "
                "summary as unrelated just because it also contains non-requested details. For answer_member/redactable_member, decide "
                "whether the requester may receive the record under the visible requester context and source record; "
                "authorization_support_span must be a nonempty exact source_text span explaining the operational/current "
                "record scope. This annotation does not control Stage-2 admission or final disclosure. Use "
                "authorized_for_requester=true only when this exact active record is a normal current "
                "operational fact that the visible episode does not mark as private, deleted, credential-like, or otherwise "
                "restricted for this requester. It is a closed-record, per-attribute disclosure recommendation, not a broad "
                "role grant: it may later authorize only the listed served_attributes on this exact source record. Set it false "
                "for exact customer identities, credentials, incident material, deleted/historical values, or any record with "
                "an explicit access boundary. Do not infer access from a job title alone. A source may state a current value and "
                "A source can contain both a requested operational status and an adjacent protected credential, identity, or "
                "incident detail. In that case, evaluate the listed served_attributes independently: retain and authorize the "
                "safe requested operational attribute on the exact record, while leaving the neighboring protected field out of "
                "served_attributes. Do not reject an otherwise useful requested status solely because another field in the same "
                "source must remain hidden; later stages project only closed selected slots, never the whole record. "
                "If an active source explicitly marks a credential or access artifact as logistics-only for the same current "
                "operation, and the requested attribute is an operational access/contingency field, retain that source as an "
                "answer_member and list only the exact typed attribute it supports. This is source-scoped utility evidence, "
                "not a role-based permission grant; a later closed adjudicator still decides whether the exact value is needed. "
                "also name the value it replaces; retain and authorize the current value when otherwise releasable, but classify "
                "the replaced value as historical rather than dropping the whole source. When a current operational recap supplies "
                "several requested attributes, retain it and list every requested attribute that it directly supports. A question "
                "may mix exact restricted fields with a separately stated broad, scheduling-safe, or public summary. Treat a "
                "negated, prohibitive, scope-setting, or meta-level phrase as governance context rather than as the value of "
                "another requested attribute: do not type phrases saying a field is not shareable, not part of an operation, "
                "unchanged, or unavailable as that field's positive value. Prefer a positive source-grounded claim for the "
                "requested attribute when one exists. Keep the distinction between an object's identity/value and its status "
                "or relation when assigning typed_fields. Also, when a "
                "request targets exact/restricted fields and the closed set contains an explicit safe/public projection for the "
                "same subject, retain that projection even if the user did not separately request the broad summary. In either case, "
                "use classification=safe_projection_member for an active source record that explicitly states the safe summary, "
                "and list only its exact typed slot names and values in safe_projection_slots. Do not list exact requested private, "
                "credential, amount, desk, identity, or wording fields there. The safe projection is a field-level capability for "
                "this request, not a permission grant; every slot_name and value must be copied from the same listed source record. "
                "that asks for a deleted, removed, retired, earlier, or replaced predecessor is not answered by the current replacement: "
                "classify a source that only names the current replacement as unrelated or policy_only for that request, never as an "
                "answer_member for the missing predecessor.\n"
                "For answer_member or redactable_member, typed_fields may record the exact source-grounded slot selected "
                "for a served attribute, including a stable semantic slot label when the extractor did not assign a "
                "conventional slot. For a record collection or logistics summary, emit all complementary requested fields "
                "from the closed records, including fields on an earlier record that were not superseded by a later update. "
                "For a composite schedule or logistics record, type independently useful components such as a weekday/date "
                "anchor, physical location, path, or condition separately from the interval; if a later record changes only "
                "the interval, do not present the older interval as current merely because its date/location remains useful. "
                "An explicitly logistics-only operational access artifact may be typed and retained when the request asks "
                "for that operational contingency; this does not authorize unrelated credentials.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\nRequested attribute IDs: {requested_attributes}\n"
                f"Requester context (not permission): requester_id={requester_id}, owner_id={owner_id}, relation={relation_to_owner}\n"
                f"Candidates: {payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return [], [], filtered + [{"memory_id": key, "reason": "stage2_llm_error"} for key in candidates], {
            "available": False, "reason": f"stage2_llm_error:{type(exc).__name__}", "candidate_count": len(candidates)
        }

    rows = _decision_rows(raw)
    decisions: list[dict[str, Any]] = []
    accepted: list[RetrievedEvidence] = []
    seen: set[str] = set()
    for item in rows:
        memory_id = str(item.get("memory_id") or "").strip()
        row = candidates.get(memory_id)
        classification = str(item.get("classification") or "").strip()
        span = str(item.get("support_span") or "").strip()
        served_attributes = [
            str(value).strip()
            for value in list(item.get("served_attributes") or item.get("attributes") or [])
            if str(value).strip() in requested_attributes
        ]
        authorized_for_requester = bool(item.get("authorized_for_requester"))
        authorization_span = str(item.get("authorization_support_span") or "").strip()
        if memory_id in seen or row is None or classification not in _CLASSES or not span or span not in str(row.content or ""):
            continue
        safe_projection_slots = _validated_safe_projection_slots(item, row)
        typed_fields = _validated_typed_fields(
            item, row, served_attributes, semantic_spec
        )
        if classification in _ADMISSIBLE and (
            not authorization_span or authorization_span not in str(row.content or "")
        ):
            # A relevance label can still be retained for diagnostics, but it
            # cannot become a Stage-2 disclosure capability without a source
            # span that deterministic code can verify.
            authorized_for_requester = False
        # A redaction member is intentionally usable as a certified partial
        # projection. Its classification, not a free-form yes/no label,
        # expresses that the source record is in scope but must not be replayed
        # beyond the graph-approved fields.
        if classification in {"redactable_member", "safe_projection_member"} and authorization_span:
            authorized_for_requester = True
        if classification == "safe_projection_member" and not safe_projection_slots:
            classification = "policy_only"
        seen.add(memory_id)
        decision = {
            "chunk_id": memory_id,
            "classification": classification,
            "support_span": span,
            "served_attributes": list(dict.fromkeys(served_attributes)),
            "safe_projection_slots": safe_projection_slots,
            "typed_fields": typed_fields,
            "authorized_for_requester": authorized_for_requester,
            "authorization_support_span": authorization_span,
            "allowed_for_requester": classification in _ADMISSIBLE,
            "policy_reason": "stage2_semantic_rerank_not_authorization",
        }
        decisions.append(decision)
        if classification in _ADMISSIBLE:
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted.append(replace(row, metadata=metadata))
        else:
            filtered.append({"memory_id": memory_id, "reason": f"stage2_{classification}"})
    # Any omitted or malformed record is not eligible for Stage 3. This keeps
    # the LLM decision closed-set and prevents a transport/parser fallback
    # from silently replaying broad retrieval context.
    for memory_id in candidates:
        if memory_id not in seen:
            filtered.append({"memory_id": memory_id, "reason": "stage2_missing_or_invalid_decision"})
    covered_attributes = {
        attribute
        for decision in decisions
        if str(decision.get("classification") or "") in _ADMISSIBLE
        for attribute in list(decision.get("served_attributes") or [])
    }
    missing_attributes = [attribute for attribute in requested_attributes if attribute not in covered_attributes]
    collection_audit_attributes = (
        requested_attributes
        if str(semantic_spec.get("request_shape") or "").strip().lower() in {"list", "plan"}
        else []
    )
    completion_attributes = list(dict.fromkeys([*missing_attributes, *collection_audit_attributes]))
    if completion_attributes:
        # The first Stage-2 pass is intentionally record-centric. A second,
        # single closed-set audit only runs when it left a requested attribute
        # uncovered, preventing a mixed current/stale record from being
        # discarded merely because it also names a superseded value.
        completion_raw = _request_attribute_coverage_completion(
            llm_client=llm_client,
            model_name=model_name,
            question=question,
            semantic_spec=semantic_spec,
            requested_attributes=completion_attributes,
            candidates=payload,
        )
        for item in _decision_rows(completion_raw):
            memory_id = str(item.get("memory_id") or "").strip()
            classification = str(item.get("classification") or "").strip()
            span = str(item.get("support_span") or "").strip()
            row = candidates.get(memory_id)
            served_attributes = [
                str(value).strip()
                for value in list(item.get("served_attributes") or item.get("attributes") or [])
                if str(value).strip() in completion_attributes
            ]
            authorization_span = str(item.get("authorization_support_span") or "").strip()
            if (
                row is None
                or classification not in _ADMISSIBLE
                or not served_attributes
                or not span
                or span not in str(row.content or "")
                or not authorization_span
                or authorization_span not in str(row.content or "")
            ):
                continue
            decision = {
                "chunk_id": memory_id,
                "classification": classification,
                "support_span": span,
                "served_attributes": list(dict.fromkeys(served_attributes)),
                "typed_fields": _validated_typed_fields(
                    item, row, served_attributes, semantic_spec
                ),
                "authorized_for_requester": bool(item.get("authorized_for_requester")),
                "authorization_support_span": authorization_span,
                "allowed_for_requester": True,
                "policy_reason": "stage2_attribute_coverage_completion_not_authorization",
            }
            decisions = [row for row in decisions if str(row.get("chunk_id") or "") != memory_id]
            decisions.append(decision)
            accepted = [row for row in accepted if str(row.memory_id) != memory_id]
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted.append(replace(row, metadata=metadata))
            filtered = [row for row in filtered if str(row.get("memory_id") or "") != memory_id]
    # LLM coverage is authoritative for semantic relevance, but it can omit a
    # plainly typed field from a multi-part request. Recover only direct,
    # source-grounded slot matches from the same closed candidate set; this is
    # a typed completeness check, not a keyword-specific admission rule.
    covered_attributes = {
        attribute
        for decision in decisions
        if str(decision.get("classification") or "") in _ADMISSIBLE
        for attribute in list(decision.get("served_attributes") or [])
    }
    direct_recovery = _recover_direct_typed_attributes(
        missing_attributes=[attribute for attribute in requested_attributes if attribute not in covered_attributes],
        semantic_spec=semantic_spec,
        candidates=candidates,
        accepted_ids={str(row.memory_id) for row in accepted},
    )
    for item in direct_recovery:
        memory_id = str(item["memory_id"])
        row = candidates.get(memory_id)
        if row is None:
            continue
        decision = {
            "chunk_id": memory_id,
            "classification": "answer_member",
            "support_span": item["support_span"],
            "served_attributes": [item["attribute"]],
            "authorized_for_requester": True,
            "authorization_support_span": item["support_span"],
            "allowed_for_requester": True,
            "policy_reason": "typed_direct_attribute_coverage_recovery",
        }
        decisions.append(decision)
        metadata = dict(row.metadata or {})
        metadata["stage2_semantic_rerank"] = decision
        accepted.append(replace(row, metadata=metadata))
        filtered = [entry for entry in filtered if str(entry.get("memory_id") or "") != memory_id]

    # A current-state query needs the active alternatives for field-level
    # adjudication, even when the record-centric pass labels one of them as
    # historical. The label can be correct for one field while the record
    # still carries a newer complementary field. Recover only records with a
    # direct, source-grounded typed match; the claim adjudicator then resolves
    # same-slot conflicts using chronology and authority evidence.
    if str(semantic_spec.get("temporal_scope") or "").strip().lower() == "current":
        accepted_ids = {str(row.memory_id) for row in accepted}
        for memory_id, row in candidates.items():
            if memory_id in accepted_ids or not _is_active(row):
                continue
            recovered_fields = _recover_direct_typed_attributes(
                missing_attributes=requested_attributes,
                semantic_spec=semantic_spec,
                candidates={memory_id: row},
                accepted_ids=set(),
            )
            if not recovered_fields:
                continue
            served_attributes = list(dict.fromkeys(
                str(field.get("attribute") or "").strip()
                for field in recovered_fields
                if str(field.get("attribute") or "").strip()
            ))
            if not served_attributes:
                continue
            prior = next(
                (item for item in decisions if str(item.get("chunk_id") or "") == memory_id),
                {},
            )
            decision = {
                "chunk_id": memory_id,
                "classification": "answer_member",
                "support_span": str(row.content or ""),
                "served_attributes": served_attributes,
                "safe_projection_slots": [],
                "typed_fields": [
                    {
                        "attribute": str(field["attribute"]),
                        "slot_name": str(field["slot_name"]),
                        "value": str(field["value"]),
                    }
                    for field in recovered_fields
                ],
                "authorized_for_requester": bool(prior.get("authorized_for_requester")),
                "authorization_support_span": str(
                    prior.get("authorization_support_span") or row.content or ""
                ),
                "allowed_for_requester": True,
                "policy_reason": "current_scope_direct_typed_candidate_recovery",
            }
            decisions = [
                item for item in decisions
                if str(item.get("chunk_id") or "") != memory_id
            ]
            decisions.append(decision)
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted.append(replace(row, metadata=metadata))
            accepted_ids.add(memory_id)
            filtered = [
                item for item in filtered
                if str(item.get("memory_id") or "") != memory_id
            ]

    # An admitted record may already be relevant for every requested
    # attribute while the model only typed one of its fields. Complete the
    # record-local typed projection before the next stage makes a field-level
    # chronology decision. This remains closed-set: the recovery can only use
    # a source-grounded slot already present in the admitted record.
    accepted, decisions = _enrich_admitted_typed_fields(
        accepted=accepted,
        decisions=decisions,
        semantic_spec=semantic_spec,
    )
    return accepted, decisions, filtered, {
        "available": True,
        "reason": "stage2_semantic_rerank_complete",
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "accepted_count": len(accepted),
    }


def map_utility_source_attributes(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    llm_client: LLMClient | None,
    model_name: str,
) -> tuple[list[RetrievedEvidence], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Map locator-selected records to source-local requested fields.

    The locator has already selected the source turns. This pass only gives
    claim adjudication a typed, per-record attribute binding; it cannot add a
    source, value, or permission.
    """
    candidates: dict[str, RetrievedEvidence] = {}
    filtered: list[dict[str, str]] = []
    for row in evidence:
        memory_id = str(row.memory_id or "").strip()
        if not memory_id:
            continue
        if not _is_active(row):
            filtered.append({"memory_id": memory_id, "reason": "nonactive_lifecycle"})
            continue
        prior = candidates.get(memory_id)
        if prior is None or float(row.score) > float(prior.score):
            candidates[memory_id] = row
    requested_attributes = [
        str(value).strip()
        for value in (
            semantic_spec.get("requested_attributes")
            or semantic_spec.get("requested_slots")
            or []
        )
        if str(value).strip()
    ]
    if not candidates or not requested_attributes or llm_client is None or not llm_client.is_available():
        return [], [], filtered, {
            "available": False,
            "reason": "utility_source_attribute_mapping_unavailable",
            "candidate_count": len(candidates),
        }
    payload = [
        {
            "memory_id": memory_id,
            "source_message_ids": list(row.source_message_ids or []),
            "source_text": str(row.content or ""),
            "timestamp": str(row.time or ""),
            "typed_slots": _typed_slot_payload(row),
            "typed_claims": [
                {
                    "property_label": str(claim.get("property_label") or ""),
                    "value_span": str(claim.get("value_span") or ""),
                    "claim_span": str(claim.get("claim_span") or ""),
                    "subject_span": str(claim.get("subject_span") or ""),
                }
                for claim in list(
                    ((row.metadata or {}).get("semantic_tags") or {}).get("claims") or []
                )
                if isinstance(claim, dict)
            ],
        }
        for memory_id, row in candidates.items()
    ]
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You map a closed set of locator-selected utility records to typed source-local fields. "
                "Do not resolve current-state conflicts, answer the user, invent values, or grant standing access. "
                "Return JSON only."
            ),
            user_prompt=(
                "Return {\"mappings\":[{\"memory_id\":string,\"support_span\":string,"
                "\"served_attributes\":[string],\"typed_fields\":[{\"attribute\":string,"
                "\"slot_name\":string,\"value\":string}],\"authorized_for_requester\":boolean,"
                "\"authorization_support_span\":string}]}. Give one mapping for every listed record, "
                "including an empty served_attributes list when it does not directly support the request. "
                "Map only requested attributes that the exact source record directly states. For every served "
                "attribute emit at least one typed_field. slot_name may be an existing typed slot or a stable "
                "semantic label when extraction omitted one; value must be copied exactly from that record's "
                "source_text. Do not use a neighboring field merely because it appears in the same sentence. "
                "When a record is a terse source-grounded value, use its typed_slots and claim metadata to map "
                "that value to the requested semantic property; do not discard a direct typed claim merely because "
                "the source_text lacks surrounding prose. A short label or status remains evidence when its typed "
                "field supplies the requested role. "
                "When a current/final summary record contains both retained safe operational fields and retired, "
                "restricted, or access-bearing neighboring fields, split the typed record at field level: map the "
                "safe current fields that directly answer the request and omit only the protected/stale fields. "
                "For an abstract record_collection or summary attribute, use the record's typed_claims as the "
                "field boundary: map each concrete, source-grounded claim that contributes a requested safe "
                "component even when its property_label does not lexically overlap the outer attribute. A concrete "
                "safe wording, schedule, location, or contingency value remains an answer field when its slot name "
                "contains status/state language. Do not map a pure instruction, permission, deletion-boundary, or "
                "policy assertion as an answer member merely because it is semantically related. "
                "Do not discard the entire source record merely because one sibling field is not releasable. "
                "Do not decide whether an older field is superseded; the later claim adjudicator compares all "
                "mapped records. Set authorized_for_requester=true only for a normal current operational field "
                "or an access/contingency value explicitly needed by a logistics-only operational request. "
                "When the question explicitly requests a logistics-only operational contingency and a current "
                "source gives the scoped PIN/code needed for that contingency, type that exact value as the "
                "contingency attribute and authorize it; do not treat it as an unrelated credential. Keep other "
                "credentials, identities, private details, and policy text out of served_attributes. "
                "authorization_support_span must be an exact substring of source_text. An empty mapping is correct "
                "when no requested field is directly stated. Before finalizing, audit every requested attribute "
                "against every listed source record and include all complementary direct fields; do not stop after "
                "the first matching record. A record may map multiple requested attributes when its typed slots or "
                "source-grounded claims support them.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Requested attributes: {requested_attributes}\nClosed locator-selected records: {payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return [], [], filtered + [
            {"memory_id": memory_id, "reason": "utility_source_attribute_mapping_error"}
            for memory_id in candidates
        ], {
            "available": False,
            "reason": f"utility_source_attribute_mapping_error:{type(exc).__name__}",
            "candidate_count": len(candidates),
        }

    rows = raw.get("mappings") if isinstance(raw, dict) else []
    rows = [item for item in rows if isinstance(item, dict)] if isinstance(rows, list) else []
    decisions: list[dict[str, Any]] = []
    accepted: list[RetrievedEvidence] = []
    seen: set[str] = set()
    requested = set(requested_attributes)
    for item in rows:
        memory_id = str(item.get("memory_id") or "").strip()
        row = candidates.get(memory_id)
        support_span = str(item.get("support_span") or "").strip()
        authorization_span = str(item.get("authorization_support_span") or support_span).strip()
        if (
            row is None
            or memory_id in seen
            or not support_span
            or support_span not in str(row.content or "")
            or not authorization_span
            or authorization_span not in str(row.content or "")
        ):
            continue
        typed_fields: list[dict[str, str]] = []
        seen_fields: set[tuple[str, str, str]] = set()
        for field in list(item.get("typed_fields") or []):
            if not isinstance(field, dict):
                continue
            attribute = str(field.get("attribute") or "").strip()
            slot_name = str(field.get("slot_name") or field.get("slot") or "").strip()
            value = str(field.get("value") or "").strip()
            key = (attribute, slot_name, value)
            if (
                attribute not in requested
                or not slot_name
                or not value
                or value not in str(row.content or "")
                or not _slot_matches_attribute(
                    attribute,
                    slot_name,
                    semantic_spec,
                    {
                        "stage2_served_attributes": [attribute],
                        "stage2_typed_fields": [
                            {
                                "attribute": attribute,
                                "slot_name": slot_name,
                                "value": value,
                            }
                        ],
                        "source_text": str(row.content or ""),
                    },
                )
                or key in seen_fields
            ):
                continue
            seen_fields.add(key)
            typed_fields.append({"attribute": attribute, "slot_name": slot_name, "value": value})
        served_attributes = list(dict.fromkeys(field["attribute"] for field in typed_fields))
        if not served_attributes:
            seen.add(memory_id)
            continue
        decision = {
            "chunk_id": memory_id,
            "classification": "answer_member",
            "support_span": support_span,
            "served_attributes": served_attributes,
            "safe_projection_slots": [],
            "typed_fields": typed_fields,
            "authorized_for_requester": bool(item.get("authorized_for_requester")),
            "authorization_support_span": authorization_span,
            "allowed_for_requester": True,
            "policy_reason": "utility_locator_source_local_attribute_mapping",
        }
        decisions.append(decision)
        metadata = dict(row.metadata or {})
        metadata["stage2_semantic_rerank"] = decision
        metadata["utility_source_closure"] = True
        accepted.append(replace(row, metadata=metadata))
        seen.add(memory_id)

    # The locator-selected bridge is a second closed-set path. A model can
    # still under-cover a multi-part utility request after the first mapping
    # call, especially when several complementary records are present. Reuse
    # the shared typed coverage completion pass once for all missing
    # attributes; it cannot expand the candidate set or grant access.
    covered_attributes = {
        str(attribute)
        for decision in decisions
        for attribute in list(decision.get("served_attributes") or [])
        if str(attribute).strip()
    }
    missing_attributes = [
        attribute for attribute in requested_attributes
        if attribute not in covered_attributes
    ]
    if missing_attributes:
        completion_raw = _request_attribute_coverage_completion(
            llm_client=llm_client,
            model_name=model_name,
            question=question,
            semantic_spec=semantic_spec,
            requested_attributes=missing_attributes,
            candidates=payload,
        )
        for item in _decision_rows(completion_raw):
            memory_id = str(item.get("memory_id") or "").strip()
            row = candidates.get(memory_id)
            classification = str(item.get("classification") or "").strip()
            support_span = str(item.get("support_span") or "").strip()
            authorization_span = str(item.get("authorization_support_span") or support_span).strip()
            served_attributes = list(dict.fromkeys(
                str(attribute).strip()
                for attribute in list(item.get("served_attributes") or item.get("attributes") or [])
                if str(attribute).strip() in missing_attributes
            ))
            typed_fields = _validated_typed_fields(
                item,
                row,
                served_attributes,
                semantic_spec,
            ) if row is not None else []
            if (
                row is None
                or classification not in _ADMISSIBLE
                or not support_span
                or support_span not in str(row.content or "")
                or not authorization_span
                or authorization_span not in str(row.content or "")
                or not typed_fields
            ):
                continue
            served_attributes = list(dict.fromkeys(
                str(field.get("attribute") or "").strip()
                for field in typed_fields
                if str(field.get("attribute") or "").strip()
            ))
            decision = {
                "chunk_id": memory_id,
                "classification": classification,
                "support_span": support_span,
                "served_attributes": served_attributes,
                "safe_projection_slots": [],
                "typed_fields": typed_fields,
                "authorized_for_requester": bool(item.get("authorized_for_requester")),
                "authorization_support_span": authorization_span,
                "allowed_for_requester": True,
                "policy_reason": "utility_source_attribute_coverage_completion",
            }
            decisions = [
                prior for prior in decisions
                if str(prior.get("chunk_id") or "") != memory_id
            ]
            accepted = [prior for prior in accepted if str(prior.memory_id) != memory_id]
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            metadata["utility_source_closure"] = True
            accepted.append(replace(row, metadata=metadata))
            seen.add(memory_id)
            covered_attributes.update(served_attributes)
    # A locator-selected source can be a terse typed claim whose value is
    # semantically obvious from its source-local field but was omitted by the
    # mapping model. Recover only those direct typed fields from the same
    # closed candidate set. Current-state adjudication still compares all
    # recovered alternatives chronologically downstream.
    # Do not auto-promote every source-local claim into an outer collection.
    # A record_collection is often a mixed operational bundle; heuristic
    # claims can include policy, neighboring threads, or negative instructions
    # that are source-grounded but not answer fields. The LLM mapping and
    # coverage passes already operate over this same closed set and provide
    # the typed binding needed for collection completion.
    direct_recovery_attributes: list[str] = []
    direct_recovery = _recover_locator_direct_typed_fields(
        requested_attributes=direct_recovery_attributes,
        semantic_spec=semantic_spec,
        candidates=candidates,
        accepted_ids=seen,
    )
    direct_by_memory: dict[str, list[dict[str, str]]] = {}
    for field in direct_recovery:
        direct_by_memory.setdefault(str(field["memory_id"]), []).append(field)
    for memory_id, fields in direct_by_memory.items():
        row = candidates[memory_id]
        typed_fields = [
            {
                "attribute": str(field["attribute"]),
                "slot_name": str(field["slot_name"]),
                "value": str(field["value"]),
            }
            for field in fields
        ]
        served_attributes = list(dict.fromkeys(
            str(field["attribute"]) for field in typed_fields
        ))
        decision = {
            "chunk_id": memory_id,
            "classification": "answer_member",
            "support_span": str(row.content or ""),
            "served_attributes": served_attributes,
            "safe_projection_slots": [],
            "typed_fields": typed_fields,
            "authorized_for_requester": True,
            "authorization_support_span": str(row.content or ""),
            "allowed_for_requester": True,
            "policy_reason": "utility_locator_direct_typed_coverage_recovery",
        }
        decisions.append(decision)
        metadata = dict(row.metadata or {})
        metadata["stage2_semantic_rerank"] = decision
        metadata["utility_source_closure"] = True
        accepted.append(replace(row, metadata=metadata))
        seen.add(memory_id)
        filtered = [
            item for item in filtered
            if str(item.get("memory_id") or "") != memory_id
        ]
    for memory_id in candidates:
        if memory_id not in seen:
            filtered.append({"memory_id": memory_id, "reason": "utility_source_attribute_mapping_missing"})
    return accepted, decisions, filtered, {
        "available": True,
        "reason": "utility_source_attribute_mapping_complete",
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "accepted_count": len(accepted),
        "direct_typed_recovery_count": len(direct_by_memory),
    }


def _validated_safe_projection_slots(item: dict[str, Any], row: RetrievedEvidence) -> list[dict[str, str]]:
    """Accept only exact typed fields from the same active source record."""
    metadata = dict(row.metadata or {})
    available: dict[str, str] = {}
    for container in (
        metadata.get("slots"),
        metadata.get("surface_spans"),
        (metadata.get("semantic_tags") or {}).get("attributes"),
        (metadata.get("semantic_tags") or {}).get("surface_values"),
    ):
        if isinstance(container, dict):
            for key, value in container.items():
                key = str(key or "").strip()
                value = str(value or "").strip()
                if key and value and key not in available:
                    available[key] = value
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in list(item.get("safe_projection_slots") or []):
        if not isinstance(candidate, dict):
            continue
        slot_name = str(candidate.get("slot_name") or candidate.get("slot") or "").strip()
        value = str(candidate.get("value") or "").strip()
        observed = available.get(slot_name)
        if not slot_name or not value or observed != value or value not in str(row.content or ""):
            continue
        key = (slot_name, value)
        if key not in seen:
            seen.add(key)
            result.append({"slot_name": slot_name, "value": value})
    return result


def _validated_typed_fields(
    item: dict[str, Any],
    row: RetrievedEvidence,
    served_attributes: list[str],
    semantic_spec: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Keep model fields source-grounded and aligned to the typed contract."""
    allowed_attributes = {str(value).strip() for value in served_attributes if str(value).strip()}
    source = str(row.content or "")
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in list(item.get("typed_fields") or []):
        if not isinstance(candidate, dict):
            continue
        attribute = str(candidate.get("attribute") or "").strip()
        slot_name = str(candidate.get("slot_name") or candidate.get("slot") or "").strip()
        value = str(candidate.get("value") or "").strip()
        if (
            not attribute
            or attribute not in allowed_attributes
            or not slot_name
            or not value
            or value not in source
        ):
            continue
        if not _slot_matches_attribute(
            attribute,
            slot_name,
            semantic_spec,
            {
                "stage2_served_attributes": [attribute],
                "stage2_typed_fields": [
                    {"attribute": attribute, "slot_name": slot_name, "value": value}
                ],
                "source_text": source,
            },
        ):
            continue
        key = (attribute, slot_name, value)
        if key not in seen:
            seen.add(key)
            result.append({
                "attribute": attribute,
                "slot_name": slot_name,
                "value": value,
            })
    return result


def _typed_slot_payload(row: RetrievedEvidence) -> dict[str, str]:
    metadata = dict(row.metadata or {})
    slots: dict[str, str] = {}
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
            value = str(value or "").strip()
            if key and value:
                slots.setdefault(key, value)
    for field in list((metadata.get("stage2_semantic_rerank") or {}).get("typed_fields") or []):
        if not isinstance(field, dict):
            continue
        slot_name = str(field.get("slot_name") or "").strip()
        value = str(field.get("value") or "").strip()
        if slot_name and value and value in str(row.content or ""):
            slots.setdefault(slot_name, value)
    return slots


def _recover_locator_direct_typed_fields(
    *,
    requested_attributes: list[str],
    semantic_spec: dict[str, Any],
    candidates: dict[str, RetrievedEvidence],
    accepted_ids: set[str],
) -> list[dict[str, str]]:
    """Recover source-local typed claims omitted by the locator mapping pass."""
    bindings = {
        str(item.get("attribute") or "").strip(): item
        for item in list(semantic_spec.get("attribute_bindings") or [])
        if isinstance(item, dict) and str(item.get("attribute") or "").strip()
    }
    recovered: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for memory_id, row in candidates.items():
        if memory_id in accepted_ids or not _is_active(row):
            continue
        source = str(row.content or "")
        if not source:
            continue
        slots = _typed_slot_payload(row)
        for attribute in requested_attributes:
            attr_tokens = _meaningful_field_tokens(attribute)
            if not attr_tokens:
                continue
            binding = bindings.get(attribute) or {}
            hint_tokens = _meaningful_field_tokens(
                str(binding.get("evidence_slot_hint") or "")
            )
            expected_slots = {
                str(binding.get("evidence_slot_hint") or "").strip()
            } - {""}
            for slot_name, value in slots.items():
                value = str(value or "").strip()
                if not value or value not in source:
                    continue
                slot_tokens = _meaningful_field_tokens(slot_name)
                overlap = _semantic_token_overlap(attr_tokens, slot_tokens)
                hint_overlap = _semantic_token_overlap(hint_tokens, slot_tokens)
                if not (
                    slot_name in expected_slots
                    or overlap
                    or hint_overlap >= 2
                ):
                    continue
                synthetic_record = {
                    "source_text": source,
                    "slots": {slot_name: value},
                    "stage2_served_attributes": [attribute],
                    "stage2_typed_fields": [{
                        "attribute": attribute,
                        "slot_name": slot_name,
                        "value": value,
                    }],
                }
                if not _slot_matches_attribute(
                    attribute,
                    slot_name,
                    semantic_spec,
                    synthetic_record,
                ):
                    continue
                key = (memory_id, attribute, slot_name, value)
                if key in seen:
                    continue
                seen.add(key)
                recovered.append({
                    "memory_id": memory_id,
                    "attribute": attribute,
                    "slot_name": slot_name,
                    "value": value,
                })
        # Heuristic extraction may expose a source-grounded claim without a
        # semantically named surface slot.  For open collection attributes,
        # claim_value is still a typed field: recover it from the closed claim
        # metadata and let the later adjudicator choose among complementary
        # fields and resolve chronology.  Instruction/forgetting frames are
        # deliberately excluded here because their claims describe policy or
        # deletion handling rather than the factual answer state.
        if (
            str(getattr(row, "memory_type", "") or "").strip().lower()
            not in {"instruction", "forgetting"}
            and any(_is_collection_attribute(attribute, semantic_spec) for attribute in requested_attributes)
        ):
            claims = list(((row.metadata or {}).get("semantic_tags") or {}).get("claims") or [])
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                slot_name = str(
                    claim.get("property_label") or claim.get("slot_name") or ""
                ).strip()
                value = str(claim.get("value_span") or "").strip()
                claim_span = str(claim.get("claim_span") or "").strip()
                if not slot_name or not value or value not in source:
                    continue
                for attribute in requested_attributes:
                    if not _is_collection_attribute(attribute, semantic_spec):
                        continue
                    synthetic_record = {
                        "source_text": source,
                        "slots": {slot_name: value},
                        "stage2_served_attributes": [attribute],
                        "stage2_typed_fields": [{
                            "attribute": attribute,
                            "slot_name": slot_name,
                            "value": value,
                        }],
                        "claim_slots": [{
                            "slot_name": slot_name,
                            "value": value,
                            "claim_span": claim_span,
                            "subject_span": str(claim.get("subject_span") or ""),
                        }],
                    }
                    if not _slot_matches_attribute(
                        attribute,
                        slot_name,
                        semantic_spec,
                        synthetic_record,
                    ):
                        continue
                    key = (memory_id, attribute, slot_name, value)
                    if key in seen:
                        continue
                    seen.add(key)
                    recovered.append({
                        "memory_id": memory_id,
                        "attribute": attribute,
                        "slot_name": slot_name,
                        "value": value,
                    })
    return recovered


def _semantic_token_overlap(left: set[str], right: set[str]) -> int:
    """Count exact or simple singular/plural typed-token overlap."""
    return sum(
        1
        for token in left
        if token in right
        or (
            len(token) > 3
            and token.endswith("s")
            and token[:-1] in right
        )
        or (
            len(token) > 3
            and token + "s" in right
        )
    )


def _recover_direct_typed_attributes(
    *,
    missing_attributes: list[str],
    semantic_spec: dict[str, Any],
    candidates: dict[str, RetrievedEvidence],
    accepted_ids: set[str],
) -> list[dict[str, str]]:
    """Recover missing fields only when an active candidate has a direct typed match."""
    bindings = {
        str(item.get("attribute") or ""): item
        for item in list(semantic_spec.get("attribute_bindings") or [])
        if isinstance(item, dict) and str(item.get("attribute") or "").strip()
    }
    current_scope = str(semantic_spec.get("temporal_scope") or "").lower() == "current"
    recovered: list[dict[str, str]] = []
    for attribute in missing_attributes:
        attr_tokens = _tokens(attribute)
        if not attr_tokens:
            continue
        expected_tokens = set(attr_tokens)
        binding = bindings.get(attribute) or {}
        expected_tokens.update(_tokens(str(binding.get("evidence_slot_hint") or "")))
        expected_slots: set[str] = set()
        # A generic extractor slot such as `date` is insufficient to realize
        # a qualified request such as `public_coaching_date`. Shared aliases
        # and exact semantic slots remain valid, but weak one-token overlap
        # must not bind a neighboring field.
        hint = str(binding.get("evidence_slot_hint") or "").strip()
        if hint:
            expected_slots.add(hint)
        ranked: list[tuple[tuple[int, int, int, str, int], dict[str, str]]] = []
        for memory_id, row in candidates.items():
            if memory_id in accepted_ids or not _is_active(row):
                continue
            source = str(row.content or "")
            slots = _typed_slot_payload(row)
            for slot_name, value in slots.items():
                value = str(value or "").strip()
                if not value or value not in source:
                    continue
                slot_tokens = _tokens(slot_name)
                overlap = len(attr_tokens & slot_tokens)
                binding_overlap = len(expected_tokens & slot_tokens)
                if overlap == 0 and binding_overlap == 0:
                    continue
                exact = int(slot_tokens == attr_tokens and bool(attr_tokens))
                schema_match = int(slot_name in expected_slots)
                if not exact and not schema_match and max(overlap, binding_overlap) < 2:
                    continue
                aligned = int(
                    exact
                    or schema_match
                    or overlap >= max(2, min(2, len(attr_tokens) // 2))
                    or binding_overlap >= max(2, min(2, len(expected_tokens) // 2))
                )
                if not aligned:
                    continue
                timestamp = str(row.time or "") if current_scope else ""
                ranked.append(((exact, max(overlap, binding_overlap), len(value), timestamp, -len(slot_name)), {
                    "attribute": attribute,
                    "memory_id": memory_id,
                    "slot_name": slot_name,
                    "value": value,
                    "support_span": str(row.content or ""),
                }))
        if ranked:
            recovered.append(max(ranked, key=lambda item: item[0])[1])
    return recovered


def _enrich_admitted_typed_fields(
    *,
    accepted: list[RetrievedEvidence],
    decisions: list[dict[str, Any]],
    semantic_spec: dict[str, Any],
) -> tuple[list[RetrievedEvidence], list[dict[str, Any]]]:
    """Fill omitted typed fields on already admitted records.

    Stage 2 is record-centric, so a model can correctly retain a complete
    current record but emit only one of its requested fields. Keeping the
    record and typed projection in sync lets downstream adjudication compare
    all source-grounded fields without reopening retrieval or authorization.
    """
    decision_by_id = {
        str(item.get("chunk_id") or ""): dict(item)
        for item in decisions
        if str(item.get("chunk_id") or "")
    }
    enriched: list[RetrievedEvidence] = []
    for row in accepted:
        memory_id = str(row.memory_id or "")
        decision = decision_by_id.get(memory_id)
        if decision is None:
            enriched.append(row)
            continue
        served = list(dict.fromkeys(
            str(attribute).strip()
            for attribute in list(decision.get("served_attributes") or [])
            if str(attribute).strip()
        ))
        existing = [
            dict(field)
            for field in list(decision.get("typed_fields") or [])
            if isinstance(field, dict)
            and str(field.get("attribute") or "").strip()
            and str(field.get("slot_name") or "").strip()
            and str(field.get("value") or "").strip()
        ]
        existing_attributes = {
            str(field.get("attribute") or "").strip() for field in existing
        }
        missing = [attribute for attribute in served if attribute not in existing_attributes]
        if missing:
            recovered = _recover_direct_typed_attributes(
                missing_attributes=missing,
                semantic_spec=semantic_spec,
                candidates={memory_id: row},
                accepted_ids=set(),
            )
            existing.extend(
                {
                    "attribute": str(field["attribute"]),
                    "slot_name": str(field["slot_name"]),
                    "value": str(field["value"]),
                }
                for field in recovered
            )
        if existing:
            decision["typed_fields"] = existing
            decision_by_id[memory_id] = decision
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            row = replace(row, metadata=metadata)
        enriched.append(row)
    ordered_decisions = [
        decision_by_id.get(str(item.get("chunk_id") or ""), item)
        for item in decisions
    ]
    return enriched, ordered_decisions


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in str(value or "").lower().replace("-", "_").split("_")
        if token
    }


def _request_attribute_coverage_completion(
    *,
    llm_client: LLMClient,
    model_name: str,
    question: str,
    semantic_spec: dict[str, Any],
    requested_attributes: list[str],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask one closed-set LLM pass to recover missing current attributes."""
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You complete missing attribute coverage over active source-grounded memory records. "
                "Do not answer the user, infer facts, create records, or grant broad access. Return JSON only."
            ),
            user_prompt=(
                "Return {\"decisions\":[{\"memory_id\":string,\"classification\":"
                "\"answer_member|redactable_member|policy_only|deleted_or_historical|unrelated\","
                "\"support_span\":string,\"served_attributes\":[string],\"typed_fields\":[{\"attribute\":string,\"slot_name\":string,\"value\":string}],\"authorized_for_requester\":boolean,"
                "\"authorization_support_span\":string}]}. Consider every listed record and cover every Missing "
                "requested attribute when a current active source directly states it. A record can state both a current "
                "value and an earlier value it replaces: retain the current value's attribute and support span while "
                "preserving other active requested fields from that record that have no later replacement. Do not select deleted, stale, credential-like, exact customer, incident, "
                "or explicitly restricted content. A record_collection may contribute several complementary typed_fields "
                "under one missing attribute; preserve fields that are not superseded. A logistics-only operational access "
                "artifact may be included only when it directly serves the requested operational contingency. typed_fields "
                "must use exact listed slot values or an exact source-grounded value with a stable semantic slot label. "
                "For an abstract record_collection or summary attribute, prefer concrete factual claims and retained "
                "safe projections over instruction, permission, deletion-boundary, or policy assertions. Use the "
                "record's typed claim/value boundary even when the property label does not overlap the outer attribute. "
                "For composite schedule/logistics records, separately type non-superseded weekday/date, location, path, and "
                "condition components while omitting an older interval that a later record replaced. "
                "authorization_support_span must be an exact source substring; it only "
                "recommends this exact record and listed attribute, not a standing permission.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Missing requested attributes: {requested_attributes}\nCandidates: {candidates}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _is_active(row: RetrievedEvidence) -> bool:
    try:
        lifecycle = str(compile_evidence_frame(row).lifecycle_status or "active").lower()
    except Exception:
        lifecycle = str((row.metadata or {}).get("memory_status") or "active").lower()
    return lifecycle not in {"deleted", "superseded", "canceled", "historical"}


def _decision_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for container in (raw, raw.get("result"), raw.get("data")):
        if isinstance(container, dict) and isinstance(container.get("decisions"), list):
            return [dict(item) for item in container["decisions"] if isinstance(item, dict)]
    return []
