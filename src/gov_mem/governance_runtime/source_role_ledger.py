"""Episode-local fact/policy source separation for governed graph evidence."""

from __future__ import annotations

from typing import Any

from gov_mem.llm.client import LLMClient, LLMClientUnavailableError


def classify_source_roles(
    *,
    messages: list[dict[str, Any]],
    candidate_message_ids: set[str],
    required_message_ids: set[str] | None,
    llm_client: LLMClient | None,
    model_name: str,
) -> dict[str, Any]:
    source_by_id = {
        str(message.get("message_id") or "").strip(): str(message.get("text") or "")
        for message in messages if str(message.get("message_id") or "").strip()
    }
    candidate_ids = {message_id for message_id in candidate_message_ids if message_id in source_by_id}
    if not candidate_ids or llm_client is None or not llm_client.is_available():
        return {"available": False, "role_by_message_id": {}, "records": []}

    def request(ids: set[str], repair: bool = False) -> dict[str, Any]:
        transcript = [
            {"message_id": message_id, "speaker_id": str(message.get("speaker_id") or ""), "text": source_by_id[message_id]}
            for message_id, message in [(str(row.get("message_id") or "").strip(), row) for row in messages]
            if message_id
        ]
        try:
            raw = llm_client.chat_json(
                model=model_name,
                system_prompt=(
                    "You classify an observed source turn's evidentiary role for governed memory. A policy says who "
                    "may receive, must not receive, or needs permission for information; a factual source states a "
                    "record value/event. A mixed source contains both. Policy and mixed sources cannot be used as "
                    "answer facts. Do not treat a clinical, operational, educational, household, or project action "
                    "plan as a disclosure policy merely because it contains an instruction, a condition, or a "
                    "recommended response. The policy distinction is about information-release authority, not the "
                    "domain subject matter. A statement that a current value is active, exclusive, replaced, or "
                    "that an old value was deleted is a factual state update unless the same statement also says "
                    "which principal may receive that information. Do not use job titles or domain keyword lists. "
                    "Return JSON only."
                ),
                user_prompt=(
                    "Return {\"source_roles\":[{\"message_id\":string,\"role\":\"factual|policy|mixed|other\","
                    "\"support_span\":string}]}. Select only listed candidate IDs and copy support_span exactly from "
                    "that source message. " + ("Evaluate every listed candidate. " if repair else "") + "\n"
                    f"Candidate source IDs: {sorted(ids)}\nVisible transcript: {transcript}"
                ),
            )
            return raw if isinstance(raw, dict) else {}
        except (LLMClientUnavailableError, Exception):
            return {}

    raw = request(candidate_ids)
    records = _validated_records(raw, candidate_ids, source_by_id)
    required_ids = {str(value) for value in list(required_message_ids or []) if str(value)} & candidate_ids
    unresolved = required_ids - {record["message_id"] for record in records}
    repair_trace: dict[str, Any] = {"attempted": False}
    if unresolved:
        repair_trace["attempted"] = True
        repair_raw = request(unresolved, repair=True)
        records.extend(_validated_records(repair_raw, candidate_ids, source_by_id))
        repair_trace.update({
            "raw_top_level_keys": sorted(str(key) for key in repair_raw.keys()) if isinstance(repair_raw, dict) else [],
            "candidate_record_count": len(_role_rows(repair_raw)),
        })
    # Independent conservative audit: it may only tighten a provisional
    # factual classification into policy/mixed. It can never promote a policy
    # source into answer evidence.
    provisional_factual_ids = {
        record["message_id"] for record in records if record["role"] == "factual"
    }
    policy_audit_trace: dict[str, Any] = {"attempted": False}
    if provisional_factual_ids:
        policy_audit_trace["attempted"] = True
        try:
            audit_raw = llm_client.chat_json(
                model=model_name,
                system_prompt=(
                    "You conservatively audit source roles for a privacy-preserving memory system. Return JSON only. "
                    "If a source states who may receive information, what may be released, a restriction, consent, "
                    "or a permission boundary, label it policy or mixed even if it contains dates or record nouns. "
                    "Return a smallest exact span expressing that policy proposition, not the whole source turn. "
                    "Do not tighten a source that only reports a record value, event, state change, request, or "
                    "domain action plan. The audit concerns authority to disclose information, not ordinary "
                    "instructions or conditions inside a protected record. "
                    "Do not label any source factual in this audit."
                ),
                user_prompt=(
                    "Return {\"source_roles\":[{\"message_id\":string,\"role\":\"policy|mixed\",\"support_span\":string}]}. "
                    "Include only listed IDs that require tightening; support_span must be exact from that message.\n"
                    + _ownerless_transcript_prompt(messages, source_by_id, provisional_factual_ids)
                ),
            )
        except (LLMClientUnavailableError, Exception):
            audit_raw = {}
        # An audit must identify a localized policy proposition. A long span
        # that merely repeats an entire factual turn cannot distinguish a
        # policy claim from the record value it would suppress.
        records.extend(_validated_policy_audit_records(
            audit_raw, provisional_factual_ids, source_by_id
        ))
        policy_audit_trace.update({
            "raw_top_level_keys": sorted(str(key) for key in audit_raw.keys()) if isinstance(audit_raw, dict) else [],
            "candidate_record_count": len(_role_rows(audit_raw)),
        })
    # A locator-selected factual source can be incorrectly labelled policy by
    # a broad first-pass classifier. Verify only those closed message IDs with
    # a second task that distinguishes disclosure authority from the record
    # fact itself. Graph construction independently suppresses any source
    # that actually yields a policy/permission atom.
    provisional_roles = _aggregate_roles(records)
    disputed_required_ids = {
        message_id for message_id in required_ids
        if provisional_roles.get(message_id) in {"policy", "mixed", "other"}
    }
    required_fact_trace: dict[str, Any] = {"attempted": False}
    verified_fact_ids: set[str] = set()
    if disputed_required_ids:
        required_fact_trace["attempted"] = True
        for message_id in sorted(disputed_required_ids):
            try:
                verify_raw = llm_client.chat_json(
                    model=model_name,
                    system_prompt=(
                        "You verify whether one selected source turn is a record fact or a disclosure-authority "
                        "policy. Return JSON only. Label factual only if the turn itself asserts a record value, "
                        "event, state, instruction, or operational plan and does not establish who may receive "
                        "information. Current-state, replacement, exclusivity, and deletion statements are factual "
                        "unless they also establish a recipient or release boundary. Label policy/mixed if it "
                        "establishes release authority. Do not use titles or domain keyword lists."
                    ),
                    user_prompt=(
                        "Return {\"source_roles\":[{\"message_id\":string,\"role\":\"factual|policy|mixed|other\","
                        "\"support_span\":string}]}. Return exactly one row for the fixed candidate ID.\n"
                        f"Candidate source ID: {message_id}\nSource text: {source_by_id[message_id]}"
                    ),
                )
            except (LLMClientUnavailableError, Exception):
                verify_raw = {}
            verified = _validated_records(verify_raw, {message_id}, source_by_id)
            if any(record["role"] == "factual" for record in verified):
                verified_fact_ids.add(message_id)
            required_fact_trace.setdefault("per_message_attempts", []).append({
                "message_id": message_id,
                "raw_top_level_keys": sorted(str(key) for key in verify_raw.keys())
                if isinstance(verify_raw, dict) else [],
                "verified_factual": message_id in verified_fact_ids,
            })
    role_sets: dict[str, set[str]] = {}
    for record in records:
        role_sets.setdefault(record["message_id"], set()).add(record["role"])
    role_by_message_id = {}
    for message_id, roles in role_sets.items():
        if message_id in verified_fact_ids:
            role_by_message_id[message_id] = "factual"
            continue
        # policy/mixed dominates factual; contradictory conservative labels
        # remain mixed rather than accidentally admitting a fact.
        if "mixed" in roles or ("policy" in roles and "factual" in roles):
            role_by_message_id[message_id] = "mixed"
        elif "policy" in roles:
            role_by_message_id[message_id] = "policy"
        elif roles == {"factual"}:
            role_by_message_id[message_id] = "factual"
        elif roles == {"other"}:
            role_by_message_id[message_id] = "other"
    return {
        "available": bool(role_by_message_id),
        "role_by_message_id": role_by_message_id,
        "records": records,
        "resolution_trace": {
            "raw_top_level_keys": sorted(str(key) for key in raw.keys()) if isinstance(raw, dict) else [],
            "candidate_record_count": len(_role_rows(raw)),
            "resolved_message_count": len(role_by_message_id),
            "priority_role_repair": repair_trace,
            "policy_role_audit": policy_audit_trace,
            "required_fact_verification": required_fact_trace,
        },
    }


def _aggregate_roles(records: list[dict[str, str]]) -> dict[str, str]:
    """Aggregate provisional roles with the same fail-closed dominance rule."""
    role_sets: dict[str, set[str]] = {}
    for record in records:
        role_sets.setdefault(record["message_id"], set()).add(record["role"])
    result: dict[str, str] = {}
    for message_id, roles in role_sets.items():
        if "mixed" in roles or ("policy" in roles and "factual" in roles):
            result[message_id] = "mixed"
        elif "policy" in roles:
            result[message_id] = "policy"
        elif roles == {"factual"}:
            result[message_id] = "factual"
        elif roles == {"other"}:
            result[message_id] = "other"
    return result


def _ownerless_transcript_prompt(
    messages: list[dict[str, Any]], source_by_id: dict[str, str], candidate_ids: set[str]
) -> str:
    transcript = [
        {"message_id": message_id, "text": source_by_id[message_id]}
        for message_id, message in [(str(row.get("message_id") or "").strip(), row) for row in messages]
        if message_id
    ]
    return f"Candidate source IDs: {sorted(candidate_ids)}\nVisible transcript: {transcript}"


def _role_rows(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for container in (raw, raw.get("result"), raw.get("data")):
        if isinstance(container, dict):
            rows = container.get("source_roles", container.get("records"))
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _validated_records(raw: object, candidate_ids: set[str], source_by_id: dict[str, str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for row in _role_rows(raw):
        message_id = str(row.get("message_id") or row.get("source_message_id") or "").strip()
        role = str(row.get("role") or row.get("source_role") or "").strip().lower()
        span = str(row.get("support_span") or row.get("source_span") or row.get("evidence_span") or "").strip()
        if message_id in candidate_ids and role in {"factual", "policy", "mixed", "other"} and span and span in source_by_id[message_id]:
            record = {"message_id": message_id, "role": role, "support_span": span}
            if record not in records:
                records.append(record)
    return records


def _validated_policy_audit_records(
    raw: object,
    candidate_ids: set[str],
    source_by_id: dict[str, str],
) -> list[dict[str, str]]:
    """Validate audit-only tightenings with a non-vacuous policy span."""
    records = _validated_records(raw, candidate_ids, source_by_id)
    return [
        record for record in records
        if record["role"] in {"policy", "mixed"}
        and not _audit_span_is_overbroad(record["support_span"], source_by_id[record["message_id"]])
    ]


def _audit_span_is_overbroad(span: str, source: str) -> bool:
    """Reject non-localized audit claims without inspecting domain vocabulary."""
    source_tokens = source.split()
    span_tokens = span.split()
    return len(source_tokens) >= 9 and len(span_tokens) / len(source_tokens) >= 0.8
