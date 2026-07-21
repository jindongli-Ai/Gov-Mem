"""LLM-mediated, closed-set utility source selection for one episode."""

from __future__ import annotations

from typing import Any

from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


def locate_utility_source_messages(
    *,
    question: str,
    semantic_spec: dict[str, Any],
    messages: list[dict[str, Any]],
    llm_client: LLMClient | None,
    model_name: str,
    max_sources: int = 20,
    context_radius: int = 5,
) -> dict[str, Any]:
    """Select source turns that may contain requested factual values.

    This is retrieval only. It cannot select an answer, infer access, or make
    a policy source answerable; later atom, ownership, policy, and graph gates
    remain mandatory.
    """
    source_by_id = {
        str(message.get("message_id") or "").strip(): str(message.get("text") or "")
        for message in messages if str(message.get("message_id") or "").strip()
    }

    if not source_by_id or llm_client is None or not llm_client.is_available():
        return {"available": False, "reason": "no_messages_or_llm_unavailable", "source_message_ids": []}
    transcript = [
        {"message_id": message_id, "speaker_id": str(message.get("speaker_id") or ""), "text": source_by_id[message_id]}
        for message_id, message in [(str(row.get("message_id") or "").strip(), row) for row in messages]
        if message_id
    ]
    raw: dict[str, Any] | list[Any] = {}
    request_error = ""
    request_attempts = 0
    # The client already retries transient provider status codes. This second
    # application-level attempt protects the source-selection boundary from a
    # one-off malformed/provider response without broadening the closed set.
    for attempt in range(2):
        request_attempts += 1
        try:
            raw = llm_client.chat_json(
                model=model_name,
                system_prompt=(
                    "You select source turns for factual utility retrieval in a governed memory system. Select only "
                    "listed message IDs whose text contains a concrete value or event needed to answer the requested "
                    "property. Do not select a message merely because it states permission, restriction, role, or the "
                    "question wording. Do not answer, infer missing values, or authorize disclosure. For a current, "
                    "latest, or as-of-now request, cover the visible update chain for each requested property: retain "
                    "the latest explicit value and enough earlier conflicting/update records for a later closed-set "
                    "adjudicator to resolve supersession. Message IDs with a numeric tNN suffix are chronological "
                    "when timestamps are absent. Prefer assertive facts and updates over requester questions or "
                    "storage instructions. If the question mixes exact restricted fields with a broad, public, "
                    "safe-summary, or logistics-safe part, also select independent source turns that state that "
                    "safe projection; later Stage 2 may retain only its typed safe fields. For a redacted or exact-"
                    "field contract, also select an explicit safe/public projection for the same resolved subject when "
                    "one is visible, even if the user did not separately ask for the broad summary. This lets later "
                    "governance return a partial answer without disclosing exact restricted fields. For a current record "
                    "collection or summary, include every visible source turn contributing a complementary operational "
                    "field, including an explicitly logistics-only access artifact when the requested operation depends on "
                    "it; do not omit it merely because it is adjacent to a protected thread. For logistics-only "
                    "operational requests, include exact scoped access artifacts (for example a current code, "
                    "PIN, or fallback method) when they are explicitly part of the requested operation. Do not "
                    "assume that a private fact record itself is a safe projection. The local discourse window "
                    "around a selected fact is also part of the closed candidate set: use it to recover adjacent "
                    "current components of the same operation, while leaving final field selection to the "
                    "adjudicator. Return JSON only."
                ),
                user_prompt=(
                    "Return {\"sources\":[{\"message_id\":string,\"support_span\":string}]}. support_span must be "
                    "a nonempty exact substring of that selected message. For each requested property, include the "
                    "latest visible factual/update turn when it exists, even if an earlier turn has stronger lexical "
                    "similarity. For a redacted/exact-field question, or any mixed disclosure question, reserve at "
                    "least one selection for an explicit safe/public summary when such a source is visible. Select at most "
                    + str(max_sources) + ".\n"
                    f"Question: {question}\nSemantic contract: {semantic_spec}\nVisible transcript: {transcript}"
                ),
            )
            request_error = ""
            break
        except (LLMClientUnavailableError, Exception) as exc:
            request_error = f"utility_source_locator_unavailable:{type(exc).__name__}"
    if request_error:
        return {"available": False, "reason": request_error, "source_message_ids": []}
    rows = _source_rows(raw)
    source_ids: list[str] = []
    rejected = 0
    for row in rows:
        message_id = str(row.get("message_id") or row.get("source_message_id") or "").strip()
        span = str(row.get("support_span") or row.get("source_span") or "").strip()
        if message_id not in source_by_id or not span or span not in source_by_id[message_id]:
            rejected += 1
            continue
        if message_id not in source_ids and len(source_ids) < max_sources:
            source_ids.append(message_id)
    # A source turn can state an update with a pronoun or shared-event
    # reference. Retain its immediate visible dialogue context for later
    # grounding; this is positional discourse structure, not vocabulary.
    ordered_ids = [str(message.get("message_id") or "").strip() for message in messages]
    source_index = {message_id: index for index, message_id in enumerate(ordered_ids) if message_id}
    closure_ids: list[str] = []
    for message_id in source_ids:
        index = source_index.get(message_id)
        if index is None:
            continue
        for neighbor_index in range(max(0, index - max(0, context_radius)), min(len(ordered_ids), index + context_radius + 1)):
            neighbor_id = ordered_ids[neighbor_index]
            if neighbor_id and neighbor_id not in closure_ids:
                closure_ids.append(neighbor_id)
    return {
        "available": bool(source_ids),
        "reason": "source_grounded_utility_messages_selected" if source_ids else "no_valid_utility_source_message",
        "source_message_ids": closure_ids,
        "selected_fact_message_ids": source_ids,
        "diagnostics": {
            "request_attempts": request_attempts,
            "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
            "candidate_count": len(rows),
            "rejected_count": rejected,
        },
    }


def locate_authorization_context_messages(
    *,
    question: str,
    requester_id: str,
    selected_fact_message_ids: set[str],
    messages: list[dict[str, Any]],
    llm_client: LLMClient | None,
    model_name: str,
    max_sources: int = 6,
) -> dict[str, Any]:
    """Retrieve source-grounded operational context for governed reranking.

    This is a separate retrieval closure from answer evidence. It cannot grant
    access: later capability adjudication still proves requester, owner,
    record identity, lifecycle, and selected slot scope.
    """
    source_by_id = {
        str(message.get("message_id") or "").strip(): str(message.get("text") or "")
        for message in messages if str(message.get("message_id") or "").strip()
    }
    selected_ids = {message_id for message_id in selected_fact_message_ids if message_id in source_by_id}
    if not requester_id or not selected_ids or llm_client is None or not llm_client.is_available():
        return {"available": False, "reason": "missing_requester_fact_closure_or_llm", "source_message_ids": []}
    transcript = [
        {"message_id": message_id, "speaker_id": str(message.get("speaker_id") or ""), "text": source_by_id[message_id]}
        for message_id, message in [(str(row.get("message_id") or "").strip(), row) for row in messages]
        if message_id
    ]
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You retrieve source-grounded authorization context for governed reranking. Do not answer the user "
                "or grant permission. Find only exact visible sources that establish a requester's concrete operation, "
                "maintenance, creation, update, cancellation, or owner confirmation for one selected factual record. "
                "A role, title, organization, or generic similarity is never sufficient. Return JSON only."
            ),
            user_prompt=(
                "Return {\"sources\":[{\"message_id\":string,\"support_span\":string}]}. support_span must "
                "be a nonempty exact substring of that source. Select at most " + str(max_sources) + ".\n"
                f"Question: {question}\nRequester ID: {requester_id}\n"
                f"Selected factual messages: {[{'message_id': message_id, 'text': source_by_id[message_id]} for message_id in sorted(selected_ids)]}\n"
                f"Visible transcript: {transcript}"
            ),
        )
    except (LLMClientUnavailableError, Exception) as exc:
        return {"available": False, "reason": f"authorization_context_unavailable:{type(exc).__name__}", "source_message_ids": []}
    source_ids: list[str] = []
    rejected = 0
    for row in _source_rows(raw):
        message_id = str(row.get("message_id") or row.get("source_message_id") or "").strip()
        span = str(row.get("support_span") or row.get("source_span") or "").strip()
        if message_id not in source_by_id or not span or span not in source_by_id[message_id]:
            rejected += 1
            continue
        if message_id not in source_ids and len(source_ids) < max_sources:
            source_ids.append(message_id)
    # Selecting only the fact itself is useful provenance, but cannot prove
    # a separate requester operation over that record. Ask the same LLM for a
    # second, closed-set source only in that degenerate case. This remains
    # retrieval: the returned span is later validated and still cannot grant a
    # capability without the Stage-2 adjudicator and graph checks.
    operation_recovery: dict[str, Any] = {"attempted": False}
    if source_ids and set(source_ids).issubset(selected_ids) and len(source_ids) < max_sources:
        operation_recovery["attempted"] = True
        try:
            repair_raw = llm_client.chat_json(
                model=model_name,
                system_prompt=(
                    "You retrieve source-grounded operational context for governed reranking. Do not answer the "
                    "user or grant permission. The previously selected factual record alone does not establish "
                    "whether its requester operates it. From the listed transcript, find only a DIFFERENT visible "
                    "source spoken by the requester that explicitly creates, updates, maintains, confirms, cancels, "
                    "or otherwise operates the same concrete record. A role, title, organization, or generic "
                    "similarity is never sufficient. Return JSON only."
                ),
                user_prompt=(
                    "Return {\"sources\":[{\"message_id\":string,\"support_span\":string}]}. support_span must "
                    "be a nonempty exact substring of that source. Select at most " + str(max_sources - len(source_ids)) + ".\n"
                    f"Question: {question}\nRequester ID: {requester_id}\n"
                    f"Selected factual messages (not eligible for this repair): "
                    f"{[{'message_id': message_id, 'text': source_by_id[message_id]} for message_id in sorted(selected_ids)]}\n"
                    f"Visible transcript: {transcript}"
                ),
            )
            repair_rows = _source_rows(repair_raw)
            repair_added = 0
            repair_rejected = 0
            for row in repair_rows:
                message_id = str(row.get("message_id") or row.get("source_message_id") or "").strip()
                span = str(row.get("support_span") or row.get("source_span") or "").strip()
                if (
                    message_id not in source_by_id
                    or message_id in selected_ids
                    or not span
                    or span not in source_by_id[message_id]
                ):
                    repair_rejected += 1
                    continue
                if message_id not in source_ids and len(source_ids) < max_sources:
                    source_ids.append(message_id)
                    repair_added += 1
            operation_recovery.update({
                "raw_top_level_keys": sorted(str(key) for key in repair_raw.keys()) if isinstance(repair_raw, dict) else [],
                "candidate_count": len(repair_rows),
                "added_count": repair_added,
                "rejected_count": repair_rejected,
            })
        except (LLMClientUnavailableError, Exception) as exc:
            operation_recovery["reason"] = f"authorization_context_recovery_unavailable:{type(exc).__name__}"
    return {
        "available": bool(source_ids),
        "reason": "source_grounded_authorization_context_selected" if source_ids else "no_authorization_context_source",
        "source_message_ids": source_ids,
        "diagnostics": {
            "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
            "candidate_count": len(_source_rows(raw)),
            "rejected_count": rejected,
            "operation_recovery": operation_recovery,
        },
    }


def _source_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for container in (raw, raw.get("result"), raw.get("data")):
        if not isinstance(container, dict):
            continue
        for key in ("sources", "source_messages", "utility_sources", "evidence"):
            rows = container.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    return []
