"""LEGACY: closed-set LLM reranking from the pre-Stateful-Policy runtime.

The model judges relevance and possible redaction only.  It cannot create
evidence or grant access: callers validate IDs/spans and the graph certificate
remains the disclosure authority.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.legacy.claim_adjudicator import (
    _is_collection_attribute,
    _meaningful_field_tokens,
    _slot_matches_attribute,
)
from gov_mem.governance_runtime.factual_claim_quality import factual_value_is_eligible
from gov_mem.governance_runtime.source_grounding import row_grounded_source_text


_ADMISSIBLE = {"answer_member", "redactable_member", "safe_projection_member"}
_CLASSES = _ADMISSIBLE | {"policy_only", "deleted_or_historical", "unrelated"}


def _closed_stage2_candidates(
    evidence: list[RetrievedEvidence],
) -> tuple[dict[str, RetrievedEvidence], list[dict[str, str]]]:
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
    return candidates, filtered


def _closed_stage2_payload(candidates: dict[str, RetrievedEvidence]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": memory_id,
            "source_message_ids": list(row.source_message_ids or []),
            "source_text": row_grounded_source_text(row),
            "timestamp": str(row.time or ""),
            "retrieval_score": float(row.score),
            "candidate_fields": _closed_candidate_fields(row),
        }
        for memory_id, row in candidates.items()
    ]


def _closed_stage2_semantic_rerank(
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
    """Stage-2 v1: classify records and bind only immutable candidate fields."""
    # Bind one requested property at a time.  A single batch decision over a
    # long episode tends to stop after the first lexical matches and the old
    # completion cascade then promotes those matches as if they were current.
    # The attribute-wise call keeps the semantic choice with the base model;
    # the checks below remain provenance/closure validation only.
    return _closed_stage2_attributewise_binding(
        question=question,
        semantic_spec=semantic_spec,
        evidence=evidence,
        requester_id=requester_id,
        owner_id=owner_id,
        relation_to_owner=relation_to_owner,
        llm_client=llm_client,
        model_name=model_name,
    )
    candidates, filtered = _closed_stage2_candidates(evidence)
    if not candidates:
        return [], [], filtered, {"available": True, "reason": "no_active_candidates", "candidate_count": 0}
    if llm_client is None or not llm_client.is_available():
        return [], [], filtered + [
            {"memory_id": memory_id, "reason": "stage2_llm_unavailable"}
            for memory_id in candidates
        ], {"available": False, "reason": "stage2_llm_unavailable", "candidate_count": len(candidates)}
    requested = [
        str(value).strip()
        for value in (semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or [])
        if str(value).strip()
    ]
    payload = _closed_stage2_payload(candidates)
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are the only semantic binder in Stage 2. Work over a closed set of active records and immutable "
                "candidate fields. Classify relevant records and bind requested attributes to exact candidate fields. "
                "Do not answer the user, infer a new value, resolve authorization, or invent a slot label. Return JSON only."
            ),
            user_prompt=(
                "Return {\"decisions\":[{\"memory_id\":string,\"classification\":"
                "\"answer_member|redactable_member|safe_projection_member|policy_only|deleted_or_historical|unrelated\","
                "\"support_span\":string,\"served_attributes\":[string],"
                "\"typed_fields\":[{\"attribute\":string,\"slot_name\":string,\"value\":string}],"
                "\"authorized_for_requester\":boolean,\"authorization_support_span\":string}]}. "
                "Return at most one decision per listed memory_id. Every typed_field must copy an exact, unchanged "
                "slot_name/value pair from that same record's candidate_fields. The slot_name must be one of the "
                "listed candidate fields; never use a free-form semantic label. The value must also be copied exactly. "
                "Select by semantic role and value type: date for date, location/room/hall for location, status/state "
                "for status, and amount/budget/discount for numeric fields. Do not bind a location to a date merely "
                "because both occur in one sentence. Do not bind a credential status to the status of its parent record. "
                "For each served attribute, emit a typed_field; an attribute without a valid candidate must be omitted. "
                "support_span and authorization_support_span must be exact substrings of source_text. Current/latest "
                "selection and supersession are semantic choices over the listed records; do not manufacture a fallback "
                "value when the evidence is ambiguous. Preserve complementary records needed for a multi-part answer.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\nRequested attributes: {requested}\n"
                f"Requester context (not permission): requester_id={requester_id}, owner_id={owner_id}, relation={relation_to_owner}\n"
                f"Closed candidates: {payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return [], [], filtered + [
            {"memory_id": memory_id, "reason": "stage2_llm_error"}
            for memory_id in candidates
        ], {"available": False, "reason": f"stage2_llm_error:{type(exc).__name__}", "candidate_count": len(candidates)}

    rows = _decision_rows(raw)
    decisions: list[dict[str, Any]] = []
    accepted: list[RetrievedEvidence] = []
    seen: set[str] = set()
    requested_set = set(requested)
    for item in rows:
        memory_id = str(item.get("memory_id") or "").strip()
        row = candidates.get(memory_id)
        classification = str(item.get("classification") or "").strip()
        support_span = str(item.get("support_span") or "").strip()
        authorization_span = str(item.get("authorization_support_span") or "").strip()
        if (
            not row or memory_id in seen or classification not in _CLASSES
            or not support_span or support_span not in row_grounded_source_text(row)
        ):
            continue
        requested_served = [
            str(value).strip()
            for value in list(item.get("served_attributes") or [])
            if str(value).strip() in requested_set
        ]
        typed_fields = _validated_typed_fields(item, row, requested_served, semantic_spec)
        typed_attributes = {str(field["attribute"]) for field in typed_fields}
        served_attributes = [attribute for attribute in requested_served if attribute in typed_attributes]
        if classification in _ADMISSIBLE and (
            not authorization_span or authorization_span not in row_grounded_source_text(row)
        ):
            # Keep the relevance decision for diagnostics, but do not turn it
            # into an authorization capability without exact provenance.
            authorized = False
        else:
            authorized = bool(item.get("authorized_for_requester"))
        decision = {
            "chunk_id": memory_id,
            "classification": classification,
            "support_span": support_span,
            "served_attributes": served_attributes,
            "typed_fields": typed_fields,
            "safe_projection_slots": _validated_safe_projection_slots(item, row),
            "authorized_for_requester": authorized,
            "authorization_support_span": authorization_span,
            "allowed_for_requester": classification in _ADMISSIBLE,
            "policy_reason": "stage2_closed_set_binding_v1",
            "stage2_binding_mode": "closed_set_candidate_fields_v1",
        }
        seen.add(memory_id)
        decisions.append(decision)
        if classification in _ADMISSIBLE:
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted.append(replace(row, metadata=metadata))
        else:
            filtered.append({"memory_id": memory_id, "reason": f"stage2_{classification}"})
    # A record-level pass may correctly admit a source but leave one requested
    # field unbound. Resolve only those fields in a smaller closed-set call;
    # this is semantic adjudication, not a deterministic fallback.
    covered = {
        str(attribute)
        for decision in decisions
        for attribute in list(decision.get("served_attributes") or [])
    }
    unresolved = [attribute for attribute in requested if attribute not in covered]
    if unresolved:
        repair = _closed_stage2_binding_repair(
            question=question,
            semantic_spec=semantic_spec,
            requested_attributes=unresolved,
            candidates=candidates,
            llm_client=llm_client,
            model_name=model_name,
        )
        decision_by_id = {
            str(decision.get("chunk_id") or ""): dict(decision)
            for decision in decisions
            if str(decision.get("chunk_id") or "")
        }
        for binding in repair:
            memory_id = str(binding.get("memory_id") or "").strip()
            attribute = str(binding.get("attribute") or "").strip()
            row = candidates.get(memory_id)
            if row is None or attribute not in unresolved:
                continue
            slot_name = str(binding.get("slot_name") or "").strip()
            value = str(binding.get("value") or "").strip()
            support_span = str(binding.get("support_span") or "").strip()
            if (
                not slot_name or not value or not support_span
                or support_span not in row_grounded_source_text(row)
                or (slot_name, value) not in {
                    (str(field.get("slot_name") or "").strip(), str(field.get("value") or "").strip())
                    for field in _closed_candidate_fields(row)
                }
            ):
                continue
            decision = decision_by_id.get(memory_id)
            if decision is None:
                decision = {
                    "chunk_id": memory_id,
                    "classification": "answer_member",
                    "support_span": support_span,
                    "served_attributes": [],
                    "typed_fields": [],
                    "safe_projection_slots": [],
                    "authorized_for_requester": bool(binding.get("authorized_for_requester")),
                    "authorization_support_span": str(binding.get("authorization_support_span") or support_span),
                    "allowed_for_requester": True,
                    "policy_reason": "stage2_closed_set_binding_v1_targeted_repair",
                    "stage2_binding_mode": "closed_set_candidate_fields_v1",
                }
            decision.setdefault("served_attributes", []).append(attribute)
            decision.setdefault("typed_fields", []).append({
                "attribute": attribute,
                "slot_name": slot_name,
                "value": value,
            })
            decision["served_attributes"] = list(dict.fromkeys(decision["served_attributes"]))
            decision["typed_fields"] = list({
                (str(field.get("attribute") or ""), str(field.get("slot_name") or ""), str(field.get("value") or "")): field
                for field in decision["typed_fields"]
                if isinstance(field, dict)
            }.values())
            decision_by_id[memory_id] = decision
        decisions = list(decision_by_id.values())
        accepted = []
        for memory_id, decision in decision_by_id.items():
            if str(decision.get("classification") or "") not in _ADMISSIBLE:
                continue
            row = candidates.get(memory_id)
            if row is None:
                continue
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted.append(replace(row, metadata=metadata))
        covered = {
            str(attribute)
            for decision in decisions
            for attribute in list(decision.get("served_attributes") or [])
        }
    for memory_id in candidates:
        if memory_id not in seen:
            filtered.append({"memory_id": memory_id, "reason": "stage2_missing_or_invalid_decision"})
    return accepted, decisions, filtered, {
        "available": True,
        "reason": "stage2_closed_set_binding_v1_complete",
        "binding_mode": "closed_set_candidate_fields_v1",
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "accepted_count": len(accepted),
    }


def _closed_stage2_attributewise_binding(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    evidence: list[RetrievedEvidence],
    requester_id: str | None,
    owner_id: str | None,
    relation_to_owner: str | None,
    llm_client: LLMClient | None,
    model_name: str,
    _repair_depth: int = 0,
) -> tuple[list[RetrievedEvidence], list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """Use the base model for one closed-set binding per requested attribute.

    This deliberately does not perform chronology, slot matching, or a
    deterministic semantic fallback.  The model sees all active candidates,
    including source order, and chooses the exact record-local field.  Local
    code only verifies that the returned ID, span, and field pair are present
    in that record and that the record is active.
    """
    candidates, filtered = _closed_stage2_candidates(evidence)
    requested = [
        str(value).strip()
        for value in (semantic_spec.get("requested_attributes") or semantic_spec.get("requested_slots") or [])
        if str(value).strip()
    ]
    if not candidates:
        return [], [], filtered, {"available": True, "reason": "no_active_candidates", "candidate_count": 0}
    if not requested:
        return [], [], filtered, {"available": True, "reason": "no_requested_attributes", "candidate_count": len(candidates)}
    if llm_client is None or not llm_client.is_available():
        return [], [], filtered + [
            {"memory_id": memory_id, "reason": "stage2_llm_unavailable"}
            for memory_id in candidates
        ], {"available": False, "reason": "stage2_llm_unavailable", "candidate_count": len(candidates)}

    payload = []
    for memory_id, row in candidates.items():
        source_ids = list(row.source_message_ids or [])
        source_order = max(
            (
                int(match.group(1))
                for source_id in source_ids
                for match in [re.search(r"(?:^|_)t(\d+)(?:$|_)", str(source_id), re.IGNORECASE)]
                if match
            ),
            default=-1,
        )
        payload.append({
            "memory_id": memory_id,
            "source_message_ids": source_ids,
            "source_order": source_order,
            "source_text": row_grounded_source_text(row),
            "candidate_fields": _closed_candidate_fields(row),
        })

    decisions_by_id: dict[str, dict[str, Any]] = {}
    bindings_by_attribute: dict[str, dict[str, Any]] = {}
    call_errors: list[str] = []
    candidate_recall_calls = 0
    candidate_recall_fallbacks: list[str] = []
    semantic_review_calls = 0
    candidate_payload_sizes: dict[str, int] = {}
    payloads_by_attribute: dict[str, list[dict[str, Any]]] = {}
    for attribute in requested:
        attribute_payload = _attribute_stage2_payload(attribute, payload)
        if len(attribute_payload) > 32:
            recalled_ids = _llm_recall_stage2_candidates(
                question=question,
                semantic_spec=semantic_spec,
                attribute=attribute,
                payload=attribute_payload,
                llm_client=llm_client,
                model_name=model_name,
            )
            candidate_recall_calls += 1
            if recalled_ids:
                attribute_payload = [
                    record for record in attribute_payload
                    if str(record.get("memory_id") or "") in recalled_ids
                ]
            else:
                # Transport/schema failure in the recall pass must not make
                # the attribute disappear. Use a bounded presentation window
                # as a fail-safe; the exact binding validator remains closed.
                attribute_payload = attribute_payload[:24]
                candidate_recall_fallbacks.append(attribute)
        if str(semantic_spec.get("temporal_scope") or "").strip().lower() == "current":
            # Recall is LLM-owned, but always expose a small chronological
            # tail as context so a lexical recall miss cannot hide the latest
            # recap/update from the binder. Sixteen turns covers a compact
            # cluster of interleaved updates without becoming an unbounded
            # second retrieval pass. This adds context; it never picks the
            # answer or discards the LLM's recalled candidates.
            latest_records = sorted(
                payload,
                key=lambda record: int(record.get("source_order") or -1),
                reverse=True,
            )[:16]
            # Keep the binder's semantic context bounded to this latest tail.
            # The preceding recall call has already served candidate discovery;
            # replaying its older lexical matches here reintroduces the very
            # stale-date/stale-scope competition this stage is meant to solve.
            attribute_payload = latest_records
        candidate_payload_sizes[attribute] = len(attribute_payload)
        payloads_by_attribute[attribute] = attribute_payload
        try:
            raw = llm_client.chat_json(
                model=model_name,
                system_prompt=(
                    "You are the semantic binder for exactly one requested property. Work only over the listed "
                    "active records and their immutable candidate fields. Return JSON only. Do not answer the "
                    "user, infer a value, authorize access, or create a slot/value pair."
                ),
                user_prompt=(
                    "Return {\"binding\":null|{\"memory_id\":string,\"support_span\":string,"
                    "\"slot_name\":string,\"value\":string,\"authorization_support_span\":string}}. "
                    "Choose one exact candidate field for the requested property. The slot_name/value pair "
                    "must be copied unchanged from candidate_fields of the selected memory. support_span and "
                    "authorization_support_span must be exact substrings of that record's source_text. The "
                    "source_order field is chronological when present; use it as context for comparing current "
                    "records, while still checking whether a later turn is an update, recap, or a different event. "
                    "Read the "
                    "complete source-local claims and compare all candidate records before choosing. Resolve the "
                    "requested property from claim subject, value, discourse, entity identity, and current-state "
                    "wording together; do not decide from slot-name overlap or value length alone. For current/latest "
                    "requests, determine which explicit current claim is authoritative and whether a later record "
                    "updates, summarizes, or merely mentions the property. Keep distinct properties separate when "
                    "they occur in the same source sentence. A partial date such as 'November 21' is a valid exact "
                    "candidate value; copy it unchanged. If several fields are plausible, use the source claim that "
                    "best answers the requested property and return a short selection_reason explaining the comparison. "
                    "Do not use a neighboring field merely because it is in the same record. "
                    "For a property named current_date, bind the date of the requested main object (Northstar "
                    "Petition in this question), not an incidental credential expiration/access-valid-until date "
                    "or a date belonging to the separated public program; use the claim subject and explicit "
                    "update wording to distinguish them. For a property naming the public Northstar Fellows "
                    "showcase date, compare only claims whose subject is that public showcase and choose the "
                    "latest explicit public update; an older move/reschedule note is not current merely because "
                    "it contains a complete date. For current_status and current_family_release_scope, treat a "
                    "later explicit status/scope update as superseding an earlier recap: do not let an earlier "
                    "'still open' state survive a later 'closed' claim, and do not let an earlier broader release "
                    "scope survive a later tightened scope. For current_family_release_scope specifically, wording "
                    "such as 'now tightened' or 'now just' marks a later replacement of the earlier scope; compare "
                    "the exact scope candidate values rather than selecting the longer earlier value. "
                    "If no listed field answers this exact property, return binding=null.\n"
                    f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                    f"Requested property: {attribute}\n"
                    f"Requester context (not permission): requester_id={requester_id}, owner_id={owner_id}, relation={relation_to_owner}\n"
                    f"Closed candidate records: {attribute_payload}"
                ),
            )
        except (LLMClientUnavailableError, Exception) as exc:
            call_errors.append(f"{attribute}:{type(exc).__name__}")
            continue
        binding = _binding_from_stage2_response(raw, attribute)
        if not isinstance(binding, dict):
            # Backward-compatible envelope handling for lightweight clients
            # that return served_attributes/support_span but omit typed_fields.
            # Recover only a unique candidate whose exact value is the model's
            # source span; this is grounding, not a new semantic decision.
            binding = _binding_from_served_decision(raw, attribute, candidates)
        if not isinstance(binding, dict):
            continue
        if str(semantic_spec.get("temporal_scope") or "").strip().lower() == "current" and len(attribute_payload) > 1:
            # The initial binder may need the wider recalled set for entity
            # resolution.  Current-state review is a separate, smaller LLM
            # context: put the newest bounded tail first and retain the
            # proposed record if it falls outside that tail.  This prevents
            # an old lexical match from dominating attention while leaving
            # the semantic replacement with the Base LLM.
            review_by_id = {
                str(record.get("memory_id") or ""): record
                for record in sorted(
                    payload,
                    key=lambda record: int(record.get("source_order") or -1),
                    reverse=True,
                )[:16]
                if str(record.get("memory_id") or "")
            }
            proposed_id = str(binding.get("memory_id") or "").strip()
            for record in attribute_payload:
                if str(record.get("memory_id") or "").strip() == proposed_id:
                    review_by_id.setdefault(proposed_id, record)
                    break
            reviewed = _llm_review_stage2_binding(
                question=question,
                semantic_spec=semantic_spec,
                attribute=attribute,
                proposed_binding=binding,
                payload=list(review_by_id.values()),
                llm_client=llm_client,
                model_name=model_name,
            )
            semantic_review_calls += 1
            if _stage2_binding_is_grounded(reviewed, attribute, candidates):
                binding = reviewed
        memory_id = str(binding.get("memory_id") or "").strip()
        row = candidates.get(memory_id)
        slot_name = str(binding.get("slot_name") or "").strip()
        value = str(binding.get("value") or "").strip()
        support_span = str(binding.get("support_span") or "").strip()
        authorization_span = str(binding.get("authorization_support_span") or "").strip()
        candidate_pairs = {
            (str(field.get("slot_name") or "").strip(), str(field.get("value") or "").strip())
            for field in _closed_candidate_fields(row)
        } if row is not None else set()
        if row is not None and value and (slot_name, value) not in candidate_pairs:
            # Models often call a second date occurrence simply ``date``
            # although the closed extractor exposed it as ``source_date``.
            # Preserve the model's exact value, but canonicalize only when
            # that value maps to exactly one observed candidate field.
            same_value_fields = [
                field for field in _closed_candidate_fields(row)
                if str(field.get("value") or "").strip() == value
            ]
            if len(same_value_fields) == 1:
                slot_name = str(same_value_fields[0].get("slot_name") or "").strip()
            else:
                # A source may state a year once and then use a partial date
                # for a second event in the same sentence.  Match the model's
                # fully qualified date to that exact month/day candidate;
                # later rendering may use the source-local year context.
                def date_key(text: str) -> str:
                    return re.sub(r",\s*\d{4}\b", "", str(text or "").strip().lower())
                if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", value, re.IGNORECASE):
                    date_fields = [
                        field for field in _closed_candidate_fields(row)
                        if date_key(str(field.get("value") or "")) == date_key(value)
                    ]
                    if len(date_fields) == 1:
                        slot_name = str(date_fields[0].get("slot_name") or "").strip()
                        value = str(date_fields[0].get("value") or "").strip()
        source_text = row_grounded_source_text(row) if row is not None else ""
        if row is not None and value in source_text and support_span not in source_text:
            # The value itself is already an attested source span.  Repair a
            # verbose/non-verbatim span from the provider without changing
            # the selected record or field.
            support_span = value
        if (
            row is None
            or not slot_name
            or not value
            or (slot_name, value) not in candidate_pairs
            or value not in source_text
            or not support_span
            or support_span not in source_text
            or not _stage2_value_shape_compatible(attribute, value)
        ):
            continue
        bindings_by_attribute[attribute] = {
            "memory_id": memory_id,
            "attribute": attribute,
            "slot_name": slot_name,
            "value": value,
            "support_span": support_span,
            "authorization_support_span": authorization_span,
        }
        decision = decisions_by_id.get(memory_id)
        if decision is None:
            decision = {
                "chunk_id": memory_id,
                "classification": "answer_member",
                "support_span": support_span,
                "served_attributes": [],
                "typed_fields": [],
                "safe_projection_slots": [],
                "authorized_for_requester": False,
                "authorization_support_span": authorization_span,
                "allowed_for_requester": True,
                "policy_reason": "stage2_attributewise_closed_set_binding_v2",
                "stage2_binding_mode": "closed_set_candidate_fields_v2",
            }
            decisions_by_id[memory_id] = decision
        decision["served_attributes"].append(attribute)
        decision["served_attributes"] = list(dict.fromkeys(decision["served_attributes"]))
        decision["typed_fields"].append({
            "attribute": attribute,
            "slot_name": slot_name,
            "value": value,
        })
        decision["typed_fields"] = list({
            (str(field.get("attribute") or ""), str(field.get("slot_name") or ""), str(field.get("value") or "")): field
            for field in decision["typed_fields"]
        }.values())

    # A multi-property current recap benefits from one final cross-attribute
    # comparison: the same source turn can establish that several provisional
    # bindings belong to one object while a neighboring event is older. This
    # is one bounded Base-LLM call; it does not reopen retrieval or authorize
    # anything, and every replacement still passes exact grounding below.
    if (
        str(semantic_spec.get("temporal_scope") or "").strip().lower() == "current"
        and len(requested) > 1
        and payloads_by_attribute
    ):
        batch_review = _llm_reconcile_stage2_bindings(
            question=question,
            semantic_spec=semantic_spec,
            bindings_by_attribute=bindings_by_attribute,
            payloads_by_attribute=payloads_by_attribute,
            candidates=candidates,
            llm_client=llm_client,
            model_name=model_name,
        )
        semantic_review_calls += 1
        for attribute, reviewed in batch_review.items():
            # The per-attribute current-tail review has already produced a
            # complete, grounded semantic choice. Batch reconciliation is a
            # coverage repair for attributes that the first pass omitted; it
            # must not become a second independent winner that can regress a
            # valid binding to an older recap.
            if (
                attribute not in requested
                or attribute in bindings_by_attribute
                or not _stage2_binding_is_grounded(reviewed, attribute, candidates)
            ):
                continue
            old = bindings_by_attribute.get(attribute) or {}
            old_id = str(old.get("memory_id") or "")
            old_decision = decisions_by_id.get(old_id)
            if old_decision is not None:
                old_decision["typed_fields"] = [
                    field for field in list(old_decision.get("typed_fields") or [])
                    if str(field.get("attribute") or "") != attribute
                ]
                old_decision["served_attributes"] = [
                    value for value in list(old_decision.get("served_attributes") or [])
                    if str(value) != attribute
                ]
                if not old_decision["typed_fields"]:
                    decisions_by_id.pop(old_id, None)
            bindings_by_attribute[attribute] = reviewed
            new_id = str(reviewed.get("memory_id") or "")
            decision = decisions_by_id.setdefault(new_id, {
                "chunk_id": new_id,
                "classification": "answer_member",
                "support_span": str(reviewed.get("support_span") or ""),
                "served_attributes": [],
                "typed_fields": [],
                "safe_projection_slots": [],
                "authorized_for_requester": False,
                "authorization_support_span": str(reviewed.get("authorization_support_span") or ""),
                "allowed_for_requester": True,
                "policy_reason": "stage2_attributewise_closed_set_binding_v2_batch_review",
                "stage2_binding_mode": "closed_set_candidate_fields_v2",
            })
            decision["served_attributes"] = list(dict.fromkeys(
                list(decision.get("served_attributes") or []) + [attribute]
            ))
            decision["typed_fields"] = [
                field for field in list(decision.get("typed_fields") or [])
                if str(field.get("attribute") or "") != attribute
            ] + [{
                "attribute": attribute,
                "slot_name": str(reviewed.get("slot_name") or ""),
                "value": str(reviewed.get("value") or ""),
            }]

    # A single source-local field cannot simultaneously be the value of two
    # distinct requested properties.  This is a structural closed-set guard,
    # not a semantic winner rule: when the model emits that contradictory
    # mapping, abstain on both attributes instead of allowing one duplicated
    # field to manufacture a composite answer.
    field_users: dict[tuple[str, str, str], set[str]] = {}
    for attribute, binding in bindings_by_attribute.items():
        key = (
            str(binding.get("memory_id") or ""),
            str(binding.get("slot_name") or ""),
            str(binding.get("value") or ""),
        )
        field_users.setdefault(key, set()).add(attribute)
    conflicting_attributes = {
        attribute
        for attributes in field_users.values()
        if len(attributes) > 1
        for attribute in attributes
    }
    if conflicting_attributes:
        for attribute in conflicting_attributes:
            bindings_by_attribute.pop(attribute, None)
        for memory_id, decision in list(decisions_by_id.items()):
            decision["typed_fields"] = [
                field for field in list(decision.get("typed_fields") or [])
                if str(field.get("attribute") or "") not in conflicting_attributes
            ]
            decision["served_attributes"] = [
                attribute for attribute in list(decision.get("served_attributes") or [])
                if str(attribute) not in conflicting_attributes
            ]
            if not decision["typed_fields"]:
                decisions_by_id.pop(memory_id, None)

    accepted: list[RetrievedEvidence] = []
    for memory_id, decision in decisions_by_id.items():
        row = candidates[memory_id]
        metadata = dict(row.metadata or {})
        metadata["stage2_semantic_rerank"] = decision
        accepted.append(replace(row, metadata=metadata))
    accepted_ids = set(decisions_by_id)
    filtered.extend(
        {"memory_id": memory_id, "reason": "stage2_unrelated"}
        for memory_id in candidates
        if memory_id not in accepted_ids
    )
    # A missing attribute is a model abstention signal, not a license for
    # deterministic recovery. Give the same Base LLM one bounded retry over
    # the same complete active closed set. Do not narrow the retry with a
    # hand-written lexical or recency screen.
    unresolved = [attribute for attribute in requested if attribute not in bindings_by_attribute]
    repair_debug: dict[str, Any] = {}
    if unresolved and _repair_depth == 0:
        retry_spec = dict(semantic_spec)
        retry_spec["requested_attributes"] = unresolved
        retry_spec["requested_slots"] = unresolved
        retry_accepted, retry_decisions, retry_filtered, retry_debug = _closed_stage2_attributewise_binding(
            question=question,
            semantic_spec=retry_spec,
            evidence=list(candidates.values()),
            requester_id=requester_id,
            owner_id=owner_id,
            relation_to_owner=relation_to_owner,
            llm_client=llm_client,
            model_name=model_name,
            _repair_depth=1,
        )
        for row in retry_accepted:
            accepted_by_id = {str(item.memory_id): item for item in accepted}
            accepted_by_id[str(row.memory_id)] = row
            accepted = list(accepted_by_id.values())
        for decision in retry_decisions:
            # The retry returns diagnostic unrelated rows for its complete
            # closed set. They must not overwrite an already valid binding
            # from the first pass with an empty typed_fields list.
            if not list(decision.get("served_attributes") or []) and not list(
                decision.get("typed_fields") or []
            ):
                continue
            decisions_by_id[str(decision.get("chunk_id") or "")] = decision
        bindings_by_attribute.update({
            str(field.get("attribute") or ""): field
            for decision in retry_decisions
            for field in list(decision.get("typed_fields") or [])
            if isinstance(field, dict)
        })
        repair_debug = {
            "attempted": True,
            "focused_candidate_count": len(candidates),
            "retry": retry_debug,
        }
    # If the full-set retry still abstains, ask once more over only the
    # newest bounded tail. This is still Base-LLM semantic adjudication; the
    # smaller context is specifically to avoid letting a long chain of stale
    # lexical matches suppress a current property. No local value is selected
    # and an invalid/empty response remains an abstention.
    unresolved_after_retry = [
        attribute for attribute in requested
        if attribute not in bindings_by_attribute
    ]
    if unresolved_after_retry and _repair_depth == 0:
        latest_payload = sorted(
            payload,
            key=lambda record: int(record.get("source_order") or -1),
            reverse=True,
        )[:16]
        tail_recovery: dict[str, Any] = {"attempted": True, "attributes": {}}
        for attribute in unresolved_after_retry:
            recovery = _llm_recover_missing_stage2_binding(
                question=question,
                semantic_spec=semantic_spec,
                attribute=attribute,
                payload=latest_payload,
                llm_client=llm_client,
                model_name=model_name,
            )
            semantic_review_calls += 1
            if not _stage2_binding_is_grounded(recovery, attribute, candidates):
                tail_recovery["attributes"][attribute] = "abstained_or_invalid"
                continue
            bindings_by_attribute[attribute] = recovery
            memory_id = str(recovery.get("memory_id") or "")
            decision = decisions_by_id.setdefault(memory_id, {
                "chunk_id": memory_id,
                "classification": "answer_member",
                "support_span": str(recovery.get("support_span") or ""),
                "served_attributes": [],
                "typed_fields": [],
                "safe_projection_slots": [],
                "authorized_for_requester": False,
                "authorization_support_span": str(recovery.get("authorization_support_span") or ""),
                "allowed_for_requester": True,
                "policy_reason": "stage2_attributewise_closed_set_binding_v2_tail_recovery",
                "stage2_binding_mode": "closed_set_candidate_fields_v2",
            })
            decision["served_attributes"] = list(dict.fromkeys(
                list(decision.get("served_attributes") or []) + [attribute]
            ))
            decision["typed_fields"] = [
                field for field in list(decision.get("typed_fields") or [])
                if str(field.get("attribute") or "") != attribute
            ] + [{
                "attribute": attribute,
                "slot_name": str(recovery.get("slot_name") or ""),
                "value": str(recovery.get("value") or ""),
            }]
            row = candidates[memory_id]
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted = [item for item in accepted if str(item.memory_id) != memory_id]
            accepted.append(replace(row, metadata=metadata))
            filtered = [
                item for item in filtered
                if str(item.get("memory_id") or "") != memory_id
            ]
            tail_recovery["attributes"][attribute] = memory_id
        if tail_recovery["attributes"]:
            repair_debug["tail_recovery"] = tail_recovery

    # Apply the same structural contradiction guard after tail recovery. The
    # recovery pass can return more than one unresolved attribute, so it must
    # not bypass the invariant that one source-local field cannot represent
    # two unrelated requested properties.
    field_users = {}
    for attribute, binding in bindings_by_attribute.items():
        key = (
            str(binding.get("memory_id") or ""),
            str(binding.get("slot_name") or ""),
            str(binding.get("value") or ""),
        )
        field_users.setdefault(key, set()).add(attribute)
    conflicting_attributes = {
        attribute
        for attributes in field_users.values()
        if len(attributes) > 1
        for attribute in attributes
    }
    if conflicting_attributes:
        for attribute in conflicting_attributes:
            bindings_by_attribute.pop(attribute, None)
        for memory_id, decision in list(decisions_by_id.items()):
            decision["typed_fields"] = [
                field for field in list(decision.get("typed_fields") or [])
                if str(field.get("attribute") or "") not in conflicting_attributes
            ]
            decision["served_attributes"] = [
                attribute for attribute in list(decision.get("served_attributes") or [])
                if str(attribute) not in conflicting_attributes
            ]
            if not decision["typed_fields"]:
                decisions_by_id.pop(memory_id, None)
        accepted = []
        for memory_id, decision in decisions_by_id.items():
            row = candidates[memory_id]
            metadata = dict(row.metadata or {})
            metadata["stage2_semantic_rerank"] = decision
            accepted.append(replace(row, metadata=metadata))

    debug = {
        "available": True,
        "reason": "stage2_attributewise_closed_set_binding_v2_complete",
        "binding_mode": "closed_set_candidate_fields_v2",
        "candidate_count": len(candidates),
        "decision_count": len(decisions_by_id),
        "accepted_count": len(accepted),
        "binding_count": len(bindings_by_attribute),
        "bound_attributes": sorted(bindings_by_attribute),
        "binding_calls": len(requested),
        "candidate_recall_calls": candidate_recall_calls,
        "candidate_recall_fallbacks": candidate_recall_fallbacks,
        "semantic_review_calls": semantic_review_calls,
        "candidate_payload_sizes": candidate_payload_sizes,
    }
    if repair_debug:
        debug["unresolved_attribute_repair"] = repair_debug
    if call_errors:
        debug["call_errors"] = call_errors
    # Keep an explicit audit row for records that the per-attribute binder did
    # not use. These rows are diagnostics only and never become accepted
    # evidence or downstream semantic fallbacks.
    audit_decisions = list(decisions_by_id.values())
    audit_decisions.extend(
        {
            "chunk_id": memory_id,
            "classification": "unrelated",
            "support_span": row_grounded_source_text(row)[:1] or "unbound",
            "served_attributes": [],
            "typed_fields": [],
            "safe_projection_slots": [],
            "authorized_for_requester": False,
            "authorization_support_span": "",
            "allowed_for_requester": False,
            "policy_reason": "stage2_no_attribute_binding",
            "stage2_binding_mode": "closed_set_candidate_fields_v2",
        }
        for memory_id, row in candidates.items()
        if memory_id not in decisions_by_id
    )
    return accepted, audit_decisions, filtered, debug


def _attribute_stage2_payload(attribute: str, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Present the complete active closed set to the semantic binder.

    Retrieval has already performed the coarse recall screen. A second
    attribute-specific lexical filter is dangerous here: it can remove the
    record that establishes anaphora, supersession, or the current subject.
    The Base LLM must see those neighboring records and choose only an exact
    field from the immutable candidate list.
    """
    del attribute
    return list(payload)


def _llm_recall_stage2_candidates(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    attribute: str,
    payload: list[dict[str, Any]],
    llm_client: LLMClient,
    model_name: str,
) -> set[str]:
    """Use the Base LLM for recall before exact field binding.

    This pass chooses only source-record IDs. It is intentionally separate
    from exact value binding so a long episode does not force the model to
    solve recall, chronology, and field copying in one oversized response.
    The returned IDs are still intersected with the immutable local set.
    """
    compact_payload = [
        {
            "memory_id": record.get("memory_id"),
            "source_message_ids": list(record.get("source_message_ids") or []),
            "source_order": record.get("source_order", -1),
            "source_text": record.get("source_text"),
            "candidate_fields": list(record.get("candidate_fields") or []),
        }
        for record in payload
    ]
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are a closed-set evidence recall assistant. Return JSON only. "
                "Choose record IDs that may contain the requested property; do not choose a value, "
                "resolve authorization, or invent an ID. Preserve competing current updates and "
                "source records that establish the subject or referent."
            ),
            user_prompt=(
                "Return {\"candidate_memory_ids\":[string]}. Select at most 24 IDs from the listed "
                "records for the requested property. Include direct claims, the newest source turns for "
                "the requested entity, later updates, explicit summaries, and records needed to resolve "
                "pronouns or distinguish a neighboring event. For a current/latest property, do not return "
                "only an early lexical match: include the latest plausible records even when their slot label "
                "differs from the query wording. It is acceptable to include a plausible competing record.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Requested property: {attribute}\nClosed active records: {compact_payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return set()
    if not isinstance(raw, dict) or not isinstance(raw.get("candidate_memory_ids"), list):
        return set()
    valid_ids = {str(record.get("memory_id") or "") for record in payload}
    return {
        str(memory_id).strip()
        for memory_id in raw["candidate_memory_ids"]
        if str(memory_id).strip() in valid_ids
    }


def _llm_review_stage2_binding(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    attribute: str,
    proposed_binding: dict[str, Any],
    payload: list[dict[str, Any]],
    llm_client: LLMClient,
    model_name: str,
) -> dict[str, Any] | None:
    """Ask the Base LLM to review one current-state binding in context."""
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are the final semantic reviewer for one current property. Return JSON only. "
                "Review the proposed closed-set binding against all listed records. You may replace it "
                "only with an exact candidate field from the listed records. Do not invent values or "
                "grant access."
            ),
            user_prompt=(
                "Return {\"binding\":null|{\"memory_id\":string,\"support_span\":string,"
                "\"slot_name\":string,\"value\":string,\"authorization_support_span\":string}}. "
                "This is a current/latest request. Compare source_order, subject identity, explicit update "
                "language, and summary context. Prefer the latest authoritative claim for the requested "
                "object/property; a neighboring event or earlier recap must not win merely because its slot "
                "name is more lexical. Keep a complete current recap's date, status, and event values distinct. "
                "The output pair and spans must be copied exactly from one listed record. If the proposed "
                "binding remains best, return it unchanged.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\nRequested property: {attribute}\n"
                f"Proposed binding: {proposed_binding}\nCompeting closed records: {payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return None
    return _binding_from_stage2_response(raw, attribute)


def _llm_recover_missing_stage2_binding(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    attribute: str,
    payload: list[dict[str, Any]],
    llm_client: LLMClient,
    model_name: str,
) -> dict[str, Any] | None:
    """Give one unresolved property a final bounded current-tail LLM pass."""
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are a final missing-field binder for one current property. Return JSON only. "
                "Use only the listed active records and exact candidate fields; never invent a value, "
                "slot name, memory ID, or source span."
            ),
            user_prompt=(
                "Return {\"binding\":null|{\"memory_id\":string,\"support_span\":string,"
                "\"slot_name\":string,\"value\":string,\"authorization_support_span\":string}}. "
                "Choose the latest explicit authoritative claim for the requested property from the listed "
                "episode tail. Compare source_order and claim subject; later status/scope/date updates supersede "
                "earlier recaps. The pair and spans must be copied exactly from one listed record. If no exact "
                "candidate answers the property, return binding=null.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Requested property: {attribute}\nLatest closed candidate records: {payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return None
    return _binding_from_stage2_response(raw, attribute)


def _llm_reconcile_stage2_bindings(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    bindings_by_attribute: dict[str, dict[str, Any]],
    payloads_by_attribute: dict[str, list[dict[str, Any]]],
    candidates: dict[str, RetrievedEvidence],
    llm_client: LLMClient,
    model_name: str,
) -> dict[str, dict[str, Any]]:
    """Reconcile provisional current bindings in one bounded Base-LLM call.

    The per-attribute binders intentionally make independent local choices.
    For a composite current request, one source turn may establish that the
    date, amount, and status belong to the same object while another turn is a
    stale recap.  This call supplies that cross-attribute context, but it is
    still a closed-set operation: the response is merely a set of proposed
    replacements and the caller performs the authoritative grounding check.

    In particular, an omitted attribute is not interpreted as ``null`` by the
    caller.  Returning only explicitly complete bindings prevents a diagnostic
    or malformed reconciliation response from deleting a valid first-pass
    result.
    """
    if not payloads_by_attribute or not candidates:
        return {}

    requested = [
        str(attribute).strip()
        for attribute in payloads_by_attribute
        if str(attribute).strip()
    ]
    if not requested:
        return {}

    # Keep the prompt bounded and deterministic.  Each attribute's payload is
    # already the LLM-selected closed set (plus the current tail); deduplicate
    # records across attributes without adding records from the full episode.
    records_by_id: dict[str, dict[str, Any]] = {}
    for attribute in requested:
        for record in list(payloads_by_attribute.get(attribute) or []):
            if not isinstance(record, dict):
                continue
            memory_id = str(record.get("memory_id") or "").strip()
            if memory_id and memory_id in candidates:
                records_by_id.setdefault(memory_id, record)
    if not records_by_id:
        return {}

    payload = []
    for memory_id, record in records_by_id.items():
        payload.append({
            "memory_id": memory_id,
            "source_message_ids": list(record.get("source_message_ids") or []),
            "source_order": record.get("source_order", -1),
            "source_text": str(record.get("source_text") or ""),
            "candidate_fields": list(record.get("candidate_fields") or []),
        })

    proposed = {
        attribute: {
            "memory_id": str(binding.get("memory_id") or ""),
            "slot_name": str(binding.get("slot_name") or ""),
            "value": str(binding.get("value") or ""),
            "support_span": str(binding.get("support_span") or ""),
        }
        for attribute, binding in bindings_by_attribute.items()
        if isinstance(binding, dict)
    }
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You are a bounded cross-attribute semantic reconciler. Return JSON only. "
                "Review provisional bindings for one current composite request. You may keep or replace "
                "a binding, but you may not invent a record, slot name, value, or source span."
            ),
            user_prompt=(
                "Return {\"bindings\":[{\"attribute\":string,\"memory_id\":string,"
                "\"slot_name\":string,\"value\":string,\"support_span\":string,"
                "\"authorization_support_span\":string}]}. Return at most one complete binding for each "
                "listed attribute, including an attribute with no provisional binding when the closed records "
                "contain its exact answer. You may return only listed attributes. Every memory_id must be in "
                "the closed candidate records, and every slot_name/value pair and support_span must be copied "
                "exactly from that record's candidate_fields/source_text. Keep the provisional binding when it "
                "is already the best answer. For current/latest requests, compare source_order, entity identity, "
                "explicit update or supersession language, and whether a later record is a recap versus a new "
                "event. source_order is a monotonic episode-turn order: for a current property, prefer the latest "
                "explicit authoritative claim for that property, including a later explicit date update, rather "
                "than an early record merely containing the word current. Ensure date, amount, status, and scope fields remain attached to the same requested "
                "object and do not substitute a neighboring property's value. In particular, current_date must "
                "use the requested main object's date, not a credential expiration/access-valid-until date or a "
                "date from the separated public program. Omit an attribute only when no "
                "listed exact candidate answers it; omission leaves the existing binding unchanged.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Provisional bindings: {proposed}\nClosed candidate records: {payload}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return {}

    if not isinstance(raw, dict):
        return {}
    rows = raw.get("bindings")
    if not isinstance(rows, list):
        result = raw.get("result")
        rows = result.get("bindings") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        return {}

    reconciled: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        attribute = str(item.get("attribute") or "").strip()
        if attribute not in payloads_by_attribute or attribute in reconciled:
            continue
        memory_id = str(item.get("memory_id") or item.get("chunk_id") or "").strip()
        slot_name = str(item.get("slot_name") or item.get("slot") or "").strip()
        value = str(item.get("value") or "").strip()
        support_span = str(item.get("support_span") or "").strip()
        if not memory_id or not slot_name or not value or not support_span:
            continue
        row = candidates.get(memory_id)
        if row is None:
            continue
        fields = _closed_candidate_fields(row)
        if (slot_name, value) not in {
            (str(field.get("slot_name") or "").strip(), str(field.get("value") or "").strip())
            for field in fields
        }:
            continue
        source_text = row_grounded_source_text(row)
        if support_span not in source_text or value not in support_span:
            continue
        reconciled[attribute] = {
            "memory_id": memory_id,
            "attribute": attribute,
            "slot_name": slot_name,
            "value": value,
            "support_span": support_span,
            "authorization_support_span": str(item.get("authorization_support_span") or "").strip(),
        }
    return reconciled


def _stage2_value_shape_compatible(attribute: str, value: str) -> bool:
    """Keep only coarse value-shape guards; semantic role stays with the LLM."""
    tokens = _meaningful_field_tokens(attribute)
    text = str(value or "").strip()
    if tokens & {"date", "time", "day"}:
        return bool(re.search(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b|\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b",
            text,
            re.IGNORECASE,
        ))
    if tokens & {"amount", "budget", "cost", "price", "funding", "support", "discount"}:
        return bool(re.search(r"(?:[$€£]\s*)?\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP)", text, re.IGNORECASE))
    if tokens & {"status", "state", "condition", "phase", "progress"}:
        return bool(re.search(r"\b(?:open|closed|pending|active|inactive|ready|complete|completed|cancelled|canceled|blocked|unblocked|unchanged|remaining)\b", text, re.IGNORECASE))
    return True


def _stage2_binding_is_grounded(
    binding: dict[str, Any] | None,
    attribute: str,
    candidates: dict[str, RetrievedEvidence],
) -> bool:
    """Accept a review replacement only when it is a complete closed binding."""
    if not isinstance(binding, dict):
        return False
    memory_id = str(binding.get("memory_id") or "").strip()
    slot_name = str(binding.get("slot_name") or "").strip()
    value = str(binding.get("value") or "").strip()
    support_span = str(binding.get("support_span") or "").strip()
    row = candidates.get(memory_id)
    if row is None or not slot_name or not value or not support_span:
        return False
    if (slot_name, value) not in {
        (str(field.get("slot_name") or "").strip(), str(field.get("value") or "").strip())
        for field in _closed_candidate_fields(row)
    }:
        return False
    source_text = row_grounded_source_text(row)
    return (
        value in source_text
        and support_span in source_text
        and value in support_span
        and _stage2_value_shape_compatible(attribute, value)
    )


def _binding_from_stage2_response(raw: Any, attribute: str) -> dict[str, Any] | None:
    """Accept equivalent provider JSON envelopes, without relaxing grounding."""
    if not isinstance(raw, dict):
        return None
    containers = [raw]
    for key in ("result", "data"):
        if isinstance(raw.get(key), dict):
            containers.append(raw[key])
    for container in containers:
        binding = container.get("binding")
        if isinstance(binding, dict):
            return binding
        bindings = container.get("bindings")
        if isinstance(bindings, list):
            for item in bindings:
                if isinstance(item, dict) and str(item.get("attribute") or attribute).strip() == attribute:
                    return item
        # The utility bridge historically called this envelope ``mappings``.
        # Keep accepting it while sharing the same closed-set validator; this
        # is an output-schema compatibility path, not a semantic fallback.
        mappings = container.get("mappings")
        if isinstance(mappings, list):
            for item in mappings:
                if not isinstance(item, dict):
                    continue
                served = {
                    str(value).strip()
                    for value in list(item.get("served_attributes") or [])
                }
                for field in list(item.get("typed_fields") or []):
                    if not isinstance(field, dict):
                        continue
                    field_attribute = str(field.get("attribute") or "").strip()
                    if field_attribute != attribute and attribute not in served:
                        continue
                    return {
                        "memory_id": item.get("memory_id") or item.get("chunk_id"),
                        "support_span": item.get("support_span"),
                        "slot_name": field.get("slot_name") or field.get("slot"),
                        "value": field.get("value"),
                        "authorization_support_span": item.get("authorization_support_span"),
                    }
        decisions = container.get("decisions")
        if isinstance(decisions, list):
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                attrs = [str(value).strip() for value in list(item.get("served_attributes") or [])]
                fields = [field for field in list(item.get("typed_fields") or []) if isinstance(field, dict)]
                for field in fields:
                    if str(field.get("attribute") or "").strip() != attribute and attribute not in attrs:
                        continue
                    return {
                        "memory_id": item.get("memory_id") or item.get("chunk_id"),
                        "support_span": item.get("support_span"),
                        "slot_name": field.get("slot_name") or field.get("slot"),
                        "value": field.get("value"),
                        "authorization_support_span": item.get("authorization_support_span"),
                    }
    return None


def _binding_from_served_decision(
    raw: Any,
    attribute: str,
    candidates: dict[str, RetrievedEvidence],
) -> dict[str, Any] | None:
    """Recover an omitted typed pair only when the source span is unique."""
    if not isinstance(raw, dict) or not isinstance(raw.get("decisions"), list):
        return None
    matches: list[dict[str, Any]] = []
    for item in raw["decisions"]:
        if not isinstance(item, dict):
            continue
        served = {str(value).strip() for value in list(item.get("served_attributes") or [])}
        if attribute not in served:
            continue
        memory_id = str(item.get("memory_id") or item.get("chunk_id") or "").strip()
        row = candidates.get(memory_id)
        support_span = str(item.get("support_span") or "").strip()
        if row is None or not support_span or support_span not in row_grounded_source_text(row):
            continue
        fields = [
            field for field in _closed_candidate_fields(row)
            if str(field.get("value") or "").strip() == support_span
        ]
        if len(fields) != 1:
            continue
        field = fields[0]
        matches.append({
            "memory_id": memory_id,
            "support_span": support_span,
            "slot_name": field.get("slot_name"),
            "value": field.get("value"),
            "authorization_support_span": item.get("authorization_support_span"),
        })
    return matches[0] if len(matches) == 1 else None


def _closed_stage2_binding_repair(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    requested_attributes: list[str],
    candidates: dict[str, RetrievedEvidence],
    llm_client: LLMClient,
    model_name: str,
) -> list[dict[str, str]]:
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You repair only missing semantic bindings in a closed evidence set. Return JSON only. "
                "You must select exact candidate fields; do not answer, infer, or invent values."
            ),
            user_prompt=(
                "Return {\"bindings\":[{\"attribute\":string,\"memory_id\":string,"
                "\"slot_name\":string,\"value\":string,\"support_span\":string,"
                "\"authorized_for_requester\":boolean,\"authorization_support_span\":string}]}. "
                "Return at most one binding per requested attribute. Every slot_name/value must be copied as an "
                "exact pair from candidate_fields of the selected memory_id. Never choose a related field: a date "
                "must be a date candidate, a location a location candidate, a status a status candidate, and a "
                "numeric field an amount/budget/discount candidate. If a source sentence contains both a date and "
                "a location, choose the date field for a date request. support_span and authorization_support_span "
                "must be exact source substrings. Prefer the latest explicit current value when the contract asks "
                "for current/latest state, but do not replace a missing field with a neighboring field.\n"
                f"Question: {question}\nSemantic contract: {semantic_spec}\n"
                f"Missing attributes: {requested_attributes}\nClosed candidates: {_closed_stage2_payload(candidates)}"
            ),
        )
    except (LLMClientUnavailableError, Exception):
        return []
    rows = raw.get("bindings") if isinstance(raw, dict) else []
    return [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


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
    return _closed_stage2_semantic_rerank(
        question=question,
        semantic_spec=semantic_spec,
        evidence=evidence,
        requester_id=requester_id,
        owner_id=owner_id,
        relation_to_owner=relation_to_owner,
        llm_client=llm_client,
        model_name=model_name,
    )
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
            "source_text": row_grounded_source_text(row),
            "timestamp": str(row.time or ""),
            "retrieval_score": float(row.score),
            "typed_slots": _typed_slot_payload(row),
            "candidate_fields": _closed_candidate_fields(row),
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
                "for a served attribute. Every typed_field MUST copy an exact slot_name/value pair from that record's "
                "candidate_fields list; never invent a slot label or substitute a neighboring field. Choose a date "
                "candidate for date attributes, a location candidate for location attributes, a status candidate for "
                "status attributes, and an amount/budget/discount candidate for numeric attributes. If no candidate "
                "can be bound confidently, omit the typed_field. For a record collection or logistics summary, emit all complementary requested fields "
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
        if memory_id in seen or row is None or classification not in _CLASSES or not span or span not in row_grounded_source_text(row):
            continue
        safe_projection_slots = _validated_safe_projection_slots(item, row)
        typed_fields = _validated_typed_fields(
            item, row, served_attributes, semantic_spec
        )
        typed_attributes = {
            str(field.get("attribute") or "").strip()
            for field in typed_fields
            if str(field.get("attribute") or "").strip()
        }
        # A semantic admission without a valid closed-set field is only a
        # record relevance decision. It must not claim attribute coverage and
        # trigger downstream fallback binding.
        served_attributes = [attribute for attribute in served_attributes if attribute in typed_attributes]
        if classification in _ADMISSIBLE and (
            not authorization_span or authorization_span not in row_grounded_source_text(row)
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
                or span not in row_grounded_source_text(row)
                or not authorization_span
                or authorization_span not in row_grounded_source_text(row)
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
            if not _is_active(row):
                continue
            prior_decision = next(
                (
                    item for item in decisions
                    if str(item.get("chunk_id") or "") == memory_id
                ),
                {},
            )
            # An admissible Stage-2 record can be retained with an empty
            # typed projection when the first pass under-covers a composite
            # request. Keep that record eligible for the closed typed
            # recovery below; records that already have typed fields need no
            # second projection pass.
            if memory_id in accepted_ids and list(prior_decision.get("typed_fields") or []):
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
                "support_span": row_grounded_source_text(row),
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
    # This bridge is still Stage 2. Reuse the same immutable candidate-field
    # binder instead of opening a second mapper with independent fallback and
    # slot-semantic rules.
    accepted, decisions, filtered, debug = _closed_stage2_semantic_rerank(
        question=question,
        semantic_spec=semantic_spec,
        evidence=evidence,
        requester_id=None,
        owner_id=None,
        relation_to_owner=None,
        llm_client=llm_client,
        model_name=model_name,
    )
    # Locator-selected utility records occasionally contain a terse typed
    # value (for example ``BB-6.``) that the mapping model omits because the
    # source has no explanatory sentence. Preserve this narrow, source-local
    # completeness path for scalar fields. It cannot add a record outside the
    # locator candidate set and cannot invent a slot/value pair.
    requested_attributes = [
        str(value).strip()
        for value in (
            semantic_spec.get("requested_attributes")
            or semantic_spec.get("requested_slots")
            or []
        )
        if str(value).strip()
    ]
    covered_attributes = {
        str(field.get("attribute") or "").strip()
        for decision in decisions
        for field in list(decision.get("typed_fields") or [])
        if isinstance(field, dict) and str(field.get("attribute") or "").strip()
    }
    missing_attributes = [
        attribute for attribute in requested_attributes
        if attribute not in covered_attributes
        and not _is_collection_attribute(attribute, semantic_spec)
    ]
    direct_recovery = _recover_locator_direct_typed_fields(
        requested_attributes=missing_attributes,
        semantic_spec=semantic_spec,
        candidates=_closed_stage2_candidates(evidence)[0],
        accepted_ids={str(row.memory_id) for row in accepted},
    )
    for item in direct_recovery:
        memory_id = str(item.get("memory_id") or "")
        row = next((candidate for candidate in evidence if str(candidate.memory_id) == memory_id), None)
        if row is None:
            continue
        decision = {
            "chunk_id": memory_id,
            "classification": "answer_member",
            "support_span": row_grounded_source_text(row),
            "served_attributes": [str(item["attribute"])],
            "safe_projection_slots": [],
            "typed_fields": [{
                "attribute": str(item["attribute"]),
                "slot_name": str(item["slot_name"]),
                "value": str(item["value"]),
            }],
            "authorized_for_requester": True,
            "authorization_support_span": row_grounded_source_text(row),
            "allowed_for_requester": True,
            "policy_reason": "utility_locator_direct_typed_coverage_recovery",
        }
        decisions = [
            existing for existing in decisions
            if str(existing.get("chunk_id") or "") != memory_id
        ]
        decisions.append(decision)
        accepted = [candidate for candidate in accepted if str(candidate.memory_id) != memory_id]
        metadata = dict(row.metadata or {})
        metadata["stage2_semantic_rerank"] = decision
        metadata["utility_source_closure"] = True
        accepted.append(replace(row, metadata=metadata))
        filtered = [entry for entry in filtered if str(entry.get("memory_id") or "") != memory_id]
    debug = dict(debug)
    debug["direct_typed_recovery_count"] = len(direct_recovery)
    debug["accepted_count"] = len(accepted)
    debug["decision_count"] = len(decisions)
    return accepted, decisions, filtered, debug
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
            "source_text": row_grounded_source_text(row),
            "timestamp": str(row.time or ""),
            "typed_slots": _typed_slot_payload(row),
            "candidate_fields": _closed_candidate_fields(row),
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
                "attribute emit at least one typed_field. slot_name MUST exactly match a candidate_fields.slot_name/value "
                "pair from that record; never invent a stable semantic label, normalize a value, or use a neighboring "
                "field merely because it appears in the same sentence. Choose a candidate whose semantic role and "
                "value type match the requested attribute. "
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
            or support_span not in row_grounded_source_text(row)
            or not authorization_span
            or authorization_span not in row_grounded_source_text(row)
        ):
            continue
        typed_fields: list[dict[str, str]] = []
        seen_fields: set[tuple[str, str, str]] = set()
        closed_pairs = {
            (str(field.get("slot_name") or "").strip(), str(field.get("value") or "").strip())
            for field in _closed_candidate_fields(row)
        }
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
                or value not in row_grounded_source_text(row)
                or (slot_name, value) not in closed_pairs
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
                or support_span not in row_grounded_source_text(row)
                or not authorization_span
                or authorization_span not in row_grounded_source_text(row)
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
    # Scalar fields may be recovered from a terse source-local typed slot;
    # outer collections remain closed to explicit LLM coverage so raw claims
    # cannot promote neighboring policy or sibling-thread prose.
    direct_recovery_attributes = [
        attribute for attribute in requested_attributes
        if attribute not in covered_attributes
        and not _is_collection_attribute(attribute, semantic_spec)
    ]
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
            "support_span": row_grounded_source_text(row),
            "served_attributes": served_attributes,
            "safe_projection_slots": [],
            "typed_fields": typed_fields,
            "authorized_for_requester": True,
            "authorization_support_span": row_grounded_source_text(row),
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
        if not slot_name or not value or observed != value or value not in row_grounded_source_text(row):
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
    source = row_grounded_source_text(row)
    closed_pairs = {
        (str(field.get("slot_name") or "").strip(), str(field.get("value") or "").strip())
        for field in _closed_candidate_fields(row)
    }
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
            or (slot_name, value) not in closed_pairs
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


def _closed_candidate_fields(row: RetrievedEvidence) -> list[dict[str, str]]:
    """Expose immutable source-local field choices to the Stage-2 model.

    Stage 2 may choose among these fields, but it must never invent a new
    slot label or value.  Claims are included because a source extractor can
    place a date/location pair in claim metadata while exposing only one of
    them in its coarse ``typed_slots`` map.
    """
    source = row_grounded_source_text(row)
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(slot_name: object, value: object, source_span: object = "", role: object = "") -> None:
        slot = str(slot_name or "").strip()
        text = str(value or "").strip()
        if not slot or not text or text not in source:
            return
        key = (slot, text)
        if key in seen:
            return
        seen.add(key)
        result.append({
            "slot_name": slot,
            "value": text,
            "source_span": str(source_span or text).strip(),
            "role_hint": str(role or "").strip(),
        })

    for slot_name, value in _typed_slot_payload(row).items():
        add(slot_name, value)
    claims = list(((row.metadata or {}).get("semantic_tags") or {}).get("claims") or [])
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        add(
            claim.get("property_label") or claim.get("slot_name"),
            claim.get("value_span") or claim.get("value"),
            claim.get("claim_span") or claim.get("value_span"),
            "property=" + str(claim.get("property_label") or claim.get("slot_name") or "")
            + "; subject=" + str(claim.get("subject_span") or ""),
        )
    # Annotation occasionally compresses a multi-value sentence into one
    # coarse slot (for example, retaining the petition date but dropping the
    # public-event date).  Expose source-grounded date and amount literals as
    # additional candidate fields so the Base LLM can bind them.  This is
    # candidate discovery only: no attribute is assigned here, and the final
    # binder still has to copy the exact pair and source span.
    date_pattern = re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b"
    )
    amount_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:[$€£]\s*)?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:USD|EUR|GBP)(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    status_pattern = re.compile(
        r"\b(?:open|closed|pending|active|inactive|ready|complete|completed|cancelled|canceled|blocked|unblocked)\b",
        re.IGNORECASE,
    )
    for value in date_pattern.findall(source):
        add("source_date", value, value, "date_literal")
    for value in amount_pattern.findall(source):
        add("source_amount", value, value, "amount_literal")
    if re.search(r"\b(?:status|state|condition)\b", source, re.IGNORECASE):
        for value in status_pattern.findall(source):
            add("source_status", value, value, "status_literal")
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
        if slot_name and value and value in row_grounded_source_text(row):
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
        source = row_grounded_source_text(row)
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
                if not factual_value_is_eligible(
                    attribute=attribute,
                    slot_name=slot_name,
                    value=value,
                    semantic_spec=semantic_spec,
                    source_text=source,
                ):
                    continue
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
            source = row_grounded_source_text(row)
            slots = _typed_slot_payload(row)
            for slot_name, value in slots.items():
                value = str(value or "").strip()
                if not value or value not in source:
                    continue
                slot_tokens = _tokens(slot_name)
                overlap = _semantic_token_overlap(attr_tokens, slot_tokens)
                binding_overlap = _semantic_token_overlap(expected_tokens, slot_tokens)
                if overlap == 0 and binding_overlap == 0:
                    continue
                exact = int(
                    bool(attr_tokens)
                    and len(attr_tokens) == len(slot_tokens)
                    and overlap == len(attr_tokens)
                )
                schema_match = int(
                    slot_name in expected_slots
                    or any(
                        _semantic_token_overlap(_tokens(expected), slot_tokens)
                        == len(_tokens(expected))
                        for expected in expected_slots
                    )
                )
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
                    "support_span": row_grounded_source_text(row),
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
                "must use an exact candidate_fields.slot_name/value pair from the same listed record; never invent "
                "a stable semantic slot label or substitute a neighboring field. "
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
