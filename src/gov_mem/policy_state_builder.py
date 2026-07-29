"""Build an observable PolicyState from an episode prefix.

The builder is intentionally conservative: it records only facts supported by
visible turns and keeps every extracted operation tied to source provenance.
It never reads evaluator metadata.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import re
from typing import Any

from gov_mem.data.schema import MemoryInstance
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.entity_resolution import principal_mentions
from gov_mem.general_lexicon import topics_from_text
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.policy_schema import (
    MemoryItemState,
    MemoryStatus,
    OperationKind,
    OperationState,
    PermissionState,
    PolicyState,
    PrincipalState,
    Provenance,
    schema_to_dict,
)


_SCOPE_TERMS = (
    "scheduling", "logistics", "clinical", "medical", "budget", "contract",
    "project", "room", "location", "financial", "public", "broad", "exact",
)

def _topics(text: str) -> tuple[str, ...]:
    lowered = str(text or "").lower()
    topics = set(topics_from_text(text))
    # Generic medication/action language catches unseen drug names without
    # enumerating dataset-specific products or patient facts.
    if re.search(r"\b(?:start|stop|take|taking|continue|avoid|refill|use)\s+[a-z][a-z0-9-]*\b", lowered):
        topics.add("medication")
    return tuple(sorted(topics))


def _safe_disclosure_projections(text: str) -> tuple[str, ...]:
    """Extract explicitly quoted broad/public phrases as separate records.

    A lifecycle instruction can remove an exact value while preserving a
    broad projection in the same message. Keeping the projection separate
    prevents the lifecycle transition from deleting both semantic layers.
    This only extracts verbatim text; authorization remains a policy decision.
    """
    lowered = text.lower()
    if not re.search(r"\b(?:safe|broad|public|sponsor-safe|sponsor-ready|household-safe|mixed-audience|helper-facing)\b", lowered):
        return ()
    result: list[str] = []
    for match in re.finditer(r"(['\"])([^'\"]{2,100})\1", text):
        phrase = match.group(2).strip()
        context = lowered[max(0, match.start() - 140):match.end() + 80]
        if re.search(r"\b(?:safe|broad|public|sponsor-safe|sponsor-ready|household-safe|mixed-audience|helper-facing)\b", context):
            result.append(phrase)
    # Also preserve an unquoted, explicitly named broad value.  This covers
    # ordinary prose such as "the broad wording is after 6:20 AM" without
    # copying the surrounding restricted boundary into the projection.
    for match in re.finditer(
        r"\b(?:safe|broad|public)(?:\s+safe)?\s+(?:label|wording|summary|phrase)\s*"
        r"(?:is|of|:)?\s+([^,.;]{2,100}?)(?=\s*(?:,\s*(?:but|while|and)|;|\.|$))",
        text,
        re.IGNORECASE,
    ):
        phrase = match.group(1).strip(" \t:-")
        if phrase and not re.search(
            r"\b(?:exact|private|restricted|resident-only|credential|badge|pin|password|diagnosis|medical)\b",
            phrase,
            re.IGNORECASE,
        ):
            result.append(phrase)
    # Boundary prose often uses ``refer to the case only as X`` instead of a
    # noun phrase such as ``safe label is X``. Keep X as the explicit safe
    # carrier while excluding quoted exact/private wording.
    for match in re.finditer(
        r"\b(?:only\s+as|refer(?:red)?\s+to\s+(?:the\s+)?(?:case|file|record)?\s*only\s+as|"
        r"described\s+only\s+as)\s+([^,.;]+)",
        text,
        re.IGNORECASE,
    ):
        phrase = match.group(1).strip(" \t:-'")
        if phrase and not re.search(
            r"\b(?:exact|private|restricted|credential|badge|pin|password|diagnosis|medical)\b",
            phrase,
            re.IGNORECASE,
        ):
            result.append(phrase)
    return tuple(dict.fromkeys(result))


def stable_memory_id(instance_id: str, message_id: str) -> str:
    digest = sha256(f"{instance_id}:{message_id}".encode("utf-8")).hexdigest()[:20]
    return f"mem_{digest}"


def _provenance(message: dict[str, Any], index: int) -> Provenance:
    return Provenance(
        source_message_ids=(str(message.get("message_id") or f"turn_{index}"),),
        timestamp=str(message.get("timestamp") or "") or None,
        turn_index=index,
        evidence_text=str(message.get("text") or ""),
    )


def _scope(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"\b(?:helper|household|public|safe|broad)[-\s]*(?:safe\s*)?(?:summary|wording|statement|recap)\b", lowered):
        return "safe_summary"
    # Observable conversation often marks a disclosure boundary with the
    # audience or direction of the projection rather than the literal words
    # ``safe summary``.  Treat those explicit outward-facing summaries and
    # snapshots as safe projections, while leaving ordinary internal records
    # and exact credentials untouched.
    if re.search(
        r"\b(?:outward|guest[- ]facing|technician[- ]facing|technician|"
        r"resident[- ]facing|household[- ]facing|desk[- ]facing|helper[- ]facing)\b"
        r"[^.!?]{0,60}\b(?:summary|snapshot|wording|statement|recap|status)\b",
        lowered,
    ):
        return "safe_summary"
    # Colors/labels are ordinary operational attributes when explicitly
    # marked current/active.  This keeps a benign field from disappearing
    # merely because the surrounding record does not say ``summary``.
    if re.search(
        r"\b(?:current(?:ly)?|active|latest)\b[^.!?]{0,40}\b(?:tag|label)\s+color\b",
        lowered,
    ):
        return "safe_summary"
    for term in _SCOPE_TERMS:
        if term in lowered:
            return term
    return None


def _subject(text: str) -> str | None:
    targeted = re.search(
        r"\b(?:deleted|delete|forgotten|forget|current|old|active)\s+(?:place|record|memory|contact|line|number)?\s*([A-Z][a-z0-9&'-]+(?:\s+[A-Z][a-z0-9&'-]+){1,4})",
        text,
        re.IGNORECASE,
    )
    if targeted:
        return targeted.group(1).strip()
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z0-9&'-]+){1,5}\b", text)
    filtered = [item for item in matches if item.split()[0].lower() not in {"yes", "no", "delete", "forget", "current", "temporary", "for", "then"}]
    return max(filtered or matches, key=len) if matches else None


def _is_lifecycle_instruction(kind: OperationKind, text: str) -> bool:
    """Return whether text explicitly mutates memory lifecycle state.

    Mentions of deleted/forgotten values inside recovery questions or denial
    boundaries are not lifecycle operations.  This keeps policy-state replay
    from deleting safe-summary carrier records merely because they name an
    omitted exact subfield.
    """
    lowered = text.lower()
    if kind not in {OperationKind.DELETE, OperationKind.FORGET}:
        return True
    if "?" in text and not re.match(r"\s*(?:please\s+)?(?:delete|remove|forget)\b", lowered):
        return False
    if re.search(
        r"\b(?:do\s+not|don't|never|no\s+one\s+should|should\s+not|cannot|can't)\s+"
        r"(?:confirm|reveal|share|recover|provide|return)\b",
        lowered,
    ):
        return False
    if re.search(r"\b(?:was|were|used\s+to|before|did|does)\b", lowered) and re.search(
        r"\b(?:deleted|removed|forgotten|old|former|previous|prior)\b", lowered
    ) and not re.search(r"\b(?:delete|remove|forget)\s+(?:the|my|our|this|that|old|previous|exact)\b", lowered):
        return False
    if kind == OperationKind.DELETE:
        return bool(
            re.search(r"\b(?:delete|remove)\b", lowered)
            or re.search(r"\b(?:deleted|removed)\b[^.]{0,100}\b(?:from|active|memory|retriev|retired)\b", lowered)
        )
    return bool(
        re.search(r"\bforget\b", lowered)
        or re.search(r"\bforgotten\b[^.]{0,100}\b(?:from|active|memory|retriev|retired)\b", lowered)
    )


def _operation_kind(text: str) -> OperationKind | None:
    lowered = text.lower()
    # Agent audit lines describe a prior update; without an explicit target
    # they are not themselves lifecycle transitions.
    if re.search(r"\bshared memory (?:was )?updated\b", lowered) and not re.search(
        r"\b(delete|remove|forget|revoke|grant|replace|supersede)\w*\b", lowered
    ):
        return None
    if re.search(r"\b(forget|forgotten|no longer remember)\b", lowered):
        kind = OperationKind.FORGET
        return kind if _is_lifecycle_instruction(kind, text) else None
    if re.search(r"\b(delete|deleted|remove|removed)\b", lowered):
        kind = OperationKind.DELETE
        return kind if _is_lifecycle_instruction(kind, text) else None
    if re.search(r"\b(revoke|revoked|withdraw|withdrawn)\b", lowered):
        return OperationKind.REVOKE
    if re.search(r"\b(grant|granted|authorize|authorized|permit|permitted)\b", lowered) or re.search(r"\b(?:allow|allowed)\s+(?:access|to|for)\b", lowered):
        return OperationKind.GRANT
    if re.search(r"\b(share|shared|release|released)\b", lowered):
        return OperationKind.SHARE
    if re.search(r"\b(update|updates|updated|replace|replaces|replaced|supersede|supersedes|superseded)\b", lowered):
        return OperationKind.UPDATE
    return None


def _is_state_replacing_update(text: str) -> bool:
    """Distinguish a value transition from an annotation about a record.

    A later summary, wording edit, reminder, or anchor can restate an older
    fact without retiring its detailed provenance.  Only explicit replacement
    language should make an earlier memory superseded.
    """
    lowered = text.lower()
    if re.search(
        r"\b(?:summary|summaries|recap|anchor|reminder|wording|label|note|phrase)\b"
        r"[^.]{0,40}\b(?:updated|saved|preserved|checked|converted|simplified)\b",
        lowered,
    ):
        return False
    if re.search(
        r"\b(?:plan|care plan|treatment plan|medication plan|summary|snapshot|state|status)\s+update\b",
        lowered,
    ) and not re.search(
        r"\b(?:replace|replaced|supersede|superseded|retire|retired|switch|switched)\b",
        lowered,
    ):
        return False
    # Ordinary updates are field-level deltas.  The source record must remain
    # available as a carrier for unchanged fields (for example, a date in an
    # earlier schedule record when only its time window moves).  The current
    # field-aware retrieval path selects the newest value for the changed
    # field, so retaining the carrier does not reintroduce stale values.
    # Only explicit whole-record lifecycle language may supersede it.
    return bool(
        re.search(r"\b(?:replace|replaced|supersede|supersedes|superseded|retire|retired)\b", lowered)
    )


def _has_field_continuity_language(text: str) -> bool:
    """Whether an update explicitly carries unspecified fields forward.

    Phrases such as "same entrance" and "rooms remain" are field-level deltas.
    Retiring the entire source record would erase the concrete value that the
    later message intentionally carries forward. This is independent of
    authorization and never revives deleted or forgotten records.
    """
    lowered = str(text or "").lower()
    return bool(re.search(
        r"\b(?:same|unchanged|remains?|still|continues?|as before|carry(?:ing)? forward)\b"
        r"[^.]{0,100}\b(?:entrance|entry|room|rooms|window|schedule|plan|route|address|"
        r"location|access|rule|contact|vehicle|parking|status|details?|fields?)\b",
        lowered,
    ))


def _permission_effect(kind: OperationKind) -> tuple[str, bool] | None:
    if kind in {OperationKind.GRANT, OperationKind.SHARE}:
        return "allow", False
    if kind in {OperationKind.REVOKE}:
        return "deny", True
    return None


_OPERATION_STOPWORDS = {
    "delete", "deleted", "remove", "removed", "forget", "forgotten", "erase", "erased", "purge",
    "update", "updated", "replace", "replaced", "supersede", "superseded", "grant", "granted",
    "allow", "allowed", "authorize", "authorized", "share", "shared", "release", "released",
    "revoke", "revoked", "withdraw", "withdrawn", "access", "memory", "record", "information",
    "please", "could", "would", "should", "the", "this", "that", "old", "current", "previous",
}


def _semantic_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())
        if token not in _OPERATION_STOPWORDS
    }


def _resolve_operation_targets(
    operation: OperationState,
    *,
    memories: list[MemoryItemState],
    source_memory_id: str,
) -> tuple[str, ...]:
    """Resolve a governance operation to prior observable records.

    Operation turns are control evidence, not automatically the object being
    deleted or updated.  Resolution uses only text, subject, provenance order,
    and explicit IDs returned by the semantic extractor.
    """
    explicit = tuple(dict.fromkeys(
        memory_id for memory_id in operation.target_memory_ids
        if memory_id and memory_id != source_memory_id
    ))
    source_text = operation.provenance.evidence_text
    operation_terms = _semantic_tokens(" ".join([source_text, operation.target_subject or "", operation.scope or ""]))
    if operation.kind == OperationKind.UPDATE and not operation.target_subject:
        # Generic update notes often contain incidental words such as
        # "side", "earlier", or "now". They are not object identity. Keep
        # explicit named entities and distinctive object terms for matching.
        operation_terms -= {
            "from", "than", "now", "may", "first", "earlier", "little",
            "closer", "ready", "planned", "ends", "side", "time", "current",
            "new", "old", "week",
        }
    target_subject = str(operation.target_subject or "").strip().lower()
    source_topics = set(_topics(source_text))
    candidates = [
        item for item in memories
        if item.memory_id != source_memory_id
        and (operation.provenance.turn_index is None or item.provenance.turn_index is None or item.provenance.turn_index < operation.provenance.turn_index)
    ]
    lifecycle_text = operation.provenance.evidence_text.lower()
    protects_broad_projection = bool(
        operation.kind in {OperationKind.DELETE, OperationKind.FORGET}
        and re.search(r"\b(?:exact|private|confidential|customer mapping|account name|credential|token)\b", lifecycle_text)
    )
    if protects_broad_projection:
        candidates = [
            item for item in candidates
            if not (item.scope in {"public", "broad", "safe_summary"} or "public_projection" in item.topics)
        ]
    scored: list[tuple[float, int, MemoryItemState]] = []

    def support(item: MemoryItemState) -> tuple[float, bool]:
        text = " ".join([item.subject or "", item.scope or "", item.provenance.evidence_text or ""])
        lowered = text.lower()
        item_topics = set(item.topics)
        if source_topics and item_topics and not source_topics.intersection(item_topics):
            return 0.0, False
        score = float(len(operation_terms & _semantic_tokens(text)))
        subject_hit = bool(target_subject and target_subject in lowered)
        if subject_hit:
            score += 8.0
        if item.subject and target_subject and target_subject in item.subject.lower():
            score += 5.0
        return score, subject_hit

    # An opaque ID returned by the semantic extractor is only a hint.  Apply
    # it after checking that the referenced record is lexically supported by
    # the operation; otherwise a bad ID can retire an unrelated memory.
    by_id = {item.memory_id: item for item in candidates}
    explicit_scores = [
        (support(by_id[memory_id])[0], memory_id)
        for memory_id in explicit
        if memory_id in by_id
    ]
    best_explicit = max((score for score, _ in explicit_scores), default=0.0)
    validated_explicit = tuple(
        memory_id for score, memory_id in explicit_scores
        if best_explicit >= 1.0 and score >= max(1.0, best_explicit - 1.0)
    )
    for item in candidates:
        score, _ = support(item)
        # Prefer the most recent matching record when several records describe
        # the same object, while leaving the semantic score dominant.
        recency = int(item.provenance.turn_index or 0)
        if score > 0:
            scored.append((score, recency, item))
    if explicit and validated_explicit:
        # The LLM target IDs are hints. Keep only IDs with lexical support
        # from the operation's own subject/text; otherwise one noisy batch
        # can delete an unrelated current record alongside the intended old
        # record.
        return validated_explicit
    if not scored:
        return validated_explicit
    best_score = max(score for score, _, _ in scored)
    # A target is determinate only when it has observable lexical/subject
    # support. Include tied records so a compound operation can cover all
    # explicitly described historical representations.
    # A generic update must not retire neighboring records merely because
    # they share a broad topic such as logistics or medication. Keep only
    # strongly supported maxima for updates; lifecycle operations retain the
    # wider tie window because they may name several representations.
    score_margin = 0.0 if operation.kind == OperationKind.UPDATE else 1.0
    minimum_score = max(2.0 if operation.kind == OperationKind.UPDATE else 1.0, best_score - score_margin)
    selected = [item for score, _, item in scored if score >= minimum_score]
    if validated_explicit and operation.kind == OperationKind.REVOKE:
        selected.extend(by_id[memory_id] for memory_id in validated_explicit if memory_id in by_id)
    selected.sort(key=lambda item: int(item.provenance.turn_index or 0), reverse=True)
    return tuple(item.memory_id for item in selected)


def _display_principals(instance: MemoryInstance) -> dict[str, str]:
    result: dict[str, str] = {}
    raw = dict(instance.metadata.get("raw_sample") or {})
    episode = dict(raw.get("episode") or {})
    entities = dict(episode.get("entities") or {})
    for principal in list(entities.get("principals") or []):
        if not isinstance(principal, dict):
            continue
        principal_id = str(principal.get("principal_id") or "")
        display = str(principal.get("display_name") or "")
        if principal_id and display:
            result[display.lower()] = principal_id
    return result


def _infer_observable_subject_relations(
    *,
    messages: list[dict[str, Any]],
    aliases: dict[str, str],
    principals: dict[str, PrincipalState],
) -> list[dict[str, Any]]:
    """Recover explicit case-subject links from visible conversation text.

    A patient/client can legitimately access records authored by a clinician
    when the conversation explicitly establishes that the person is the
    subject of the case. This relation is evidence-bound; a role by itself is
    never treated as a blanket grant.
    """
    subject_roles = {"patient", "client", "customer", "student", "resident"}
    cue = re.compile(
        r"\b(?:is|was|has been)\s+(?:here|seen|scheduled|assigned|enrolled|registered)\b"
        r"|\b(?:follow[- ]?up|case|chart|appointment|visit)\b",
        re.IGNORECASE,
    )
    relations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for message in messages:
        text = str(message.get("text") or "")
        if not cue.search(text):
            continue
        lowered = text.lower()
        message_id = str(message.get("message_id") or "")
        for display, principal_id in aliases.items():
            principal = principals.get(principal_id)
            if principal is None or str(principal.role or "").lower() not in subject_roles:
                continue
            if display not in lowered:
                continue
            key = (principal_id, message_id)
            if key in seen:
                continue
            seen.add(key)
            relations.append({
                "type": "case_subject",
                "subject_id": principal_id,
                "source_message_id": message_id,
                "evidence_text": text,
            })
    return relations


def _observable_entities(instance: MemoryInstance) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = dict(instance.metadata.get("raw_sample") or {})
    episode = dict(raw.get("episode") or {})
    entities = dict(episode.get("entities") or {})
    principals = [item for item in list(entities.get("principals") or []) if isinstance(item, dict)]
    relationships = [item for item in list(entities.get("relationships") or []) if isinstance(item, dict)]
    return ({"principals": principals}, relationships)


def _infer_grantee(text: str, aliases: dict[str, str]) -> str | None:
    lowered = text.lower()
    for display, principal_id in aliases.items():
        if display in lowered:
            return principal_id
        first_name = display.split()[0]
        if len(first_name) > 2 and re.search(rf"\b{re.escape(first_name)}\b", lowered):
            return principal_id
    match = re.search(r"\b(?:for|to|with)\s+([a-z][a-z0-9_.-]+)", lowered)
    return match.group(1) if match else None


def _is_question_turn(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return "?" in lowered and bool(re.search(
        r"\b(?:what|which|who|where|when|why|how|do|does|did|is|are|was|were|"
        r"can|could|should|would|will|whether|confirm)\b",
        lowered,
    ))


def _safe_direct_response_projection(question: str, response: str) -> str | None:
    """Keep a safe operational projection from an explicit direct reply.

    A resident's direct answer can authorize a narrow logistics projection,
    while the same reply may also repeat a credential or other private exact
    value. Split the reply into source-local sentences and retain only the
    non-sensitive declarative clauses for a safe/logistics-only question.
    This creates a new provenance-bound projection; it does not rewrite the
    original memory or infer a value from evaluator metadata.
    """
    lowered_question = str(question or "").lower()
    if not re.search(
        r"\b(?:logistics|safe|broad|public|helper[- ]facing|summary|only)\b",
        lowered_question,
    ):
        return None
    sensitive = re.compile(
        r"\b(?:credential|password|passcode|pin|token|secret|phrase|keypad|"
        r"code|private|exact)\b|\b\d{3,}\b",
        re.IGNORECASE,
    )
    useful = []
    for clause in re.split(r"(?<=[.!?])\s+", str(response or "").strip()):
        clause = clause.strip(" \t\"'")
        if not clause or clause.lower() in {"yes", "no", "correct", "confirmed", "understood"}:
            continue
        if sensitive.search(clause):
            continue
        if re.search(
            r"\b(?:after|before|window|approved|allowed|fallback|route|entry|"
            r"arrival|desk|buzz|release|only|summary|current|remains?)\b",
            clause,
            re.IGNORECASE,
        ):
            useful.append(clause)
    projection = " ".join(useful).strip()
    return projection or None


_POLICY_SYSTEM_PROMPT = """
Extract explicit governance operations from observable conversation turns.
Return JSON only: {\"operations\": [...]}. Each operation may contain
source_message_id, kind (grant/revoke/update/delete/forget/share/access),
actor, grantee, target_message_ids, target_subject, scope, effect, and
effective_at. Copy IDs from the supplied messages. Do not infer hidden labels,
benchmark categories, answers, or permissions that are not stated. Questions
and ordinary factual statements are not operations.
""".strip()


def _llm_policy_operations(
    *,
    instance: MemoryInstance,
    llm_client: LLMClient | None,
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if llm_client is None or not llm_client.is_available() or not instance.messages:
        return []
    payload = {
        "messages": [
            {
                "message_id": str(message.get("message_id") or ""),
                "speaker_id": str(message.get("speaker_id") or ""),
                "timestamp": message.get("timestamp"),
                "text": str(message.get("text") or ""),
            }
            for message in instance.messages
        ]
    }
    assert_runtime_payload_safe(payload, context="policy_state_extraction_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config or {}, "reasoning"),
            system_prompt=_POLICY_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    operations = raw.get("operations")
    if not isinstance(operations, list):
        return []
    allowed = {item.value for item in OperationKind}
    result: list[dict[str, Any]] = []
    known_ids = {str(message.get("message_id") or "") for message in instance.messages}
    for item in operations:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        source_id = str(item.get("source_message_id") or "").strip()
        if kind not in allowed or source_id not in known_ids:
            continue
        target_ids = item.get("target_message_ids") or []
        if isinstance(target_ids, str):
            target_ids = [target_ids]
        result.append(
            {
                "source_message_id": source_id,
                "kind": kind,
                "actor": str(item.get("actor") or "").strip() or None,
                "grantee": (
                    str(instance.asking_user_id or "").strip()
                    if str(item.get("grantee") or "").strip().lower().rstrip(".") in {"you", "your", "the requester", "requester"}
                    else str(item.get("grantee") or "").strip() or None
                ),
                "target_message_ids": [str(value) for value in target_ids if str(value) in known_ids],
                "target_subject": str(item.get("target_subject") or "").strip() or None,
                "scope": str(item.get("scope") or "").strip() or None,
                "effect": str(item.get("effect") or "").strip().lower() or None,
                "effective_at": str(item.get("effective_at") or "").strip() or None,
            }
        )
    return result


def build_policy_state(
    instance: MemoryInstance,
    *,
    llm_client: LLMClient | None = None,
    config: dict[str, Any] | None = None,
) -> PolicyState:
    messages = list(instance.messages or [])
    aliases = _display_principals(instance)
    principals: dict[str, PrincipalState] = {}
    memories: list[MemoryItemState] = []
    operations: list[OperationState] = []
    permissions: list[PermissionState] = []
    statuses: dict[str, MemoryStatus] = {}
    ownership: list[dict[str, Any]] = []
    scope_constraints: list[dict[str, Any]] = []
    provenance: list[Provenance] = []
    direct_response_permissions: list[PermissionState] = []

    entity_payload, relationships = _observable_entities(instance)
    for principal in entity_payload.get("principals", []):
        principal_id = str(principal.get("principal_id") or "").strip()
        if not principal_id:
            continue
        display_name = str(principal.get("display_name") or "").strip() or None
        aliases_from_data = principal.get("aliases") or ()
        if isinstance(aliases_from_data, str):
            aliases_from_data = (aliases_from_data,)
        principal_aliases = tuple(dict.fromkeys(
            str(value).strip()
            for value in (display_name, *aliases_from_data)
            if str(value or "").strip()
        ))
        principals.setdefault(
            principal_id,
            PrincipalState(
                principal_id=principal_id,
                role=str(principal.get("role") or "") or None,
                display_name=display_name,
                aliases=principal_aliases,
                entity_type=str(principal.get("entity_type") or principal.get("type") or "person"),
            ),
        )
    for relation in relationships:
        safe_relation = {
            str(key): value for key, value in relation.items()
            if str(key) not in {"expected_action", "judge_spec", "leak_targets", "query_type", "oracle_evidence", "rationale"}
        }
        if safe_relation:
            scope_constraints.append(safe_relation)
    # Keep relationship categories attached to principals as a compact graph
    # index.  Authorization still requires the concrete relation object below;
    # this field is for identity disambiguation and audit only.
    for relation in relationships:
        relation_type = str(relation.get("type") or "").strip()
        if not relation_type:
            continue
        for key, value in relation.items():
            if not str(key).endswith("_id"):
                continue
            principal_id = str(value or "").strip()
            if principal_id not in principals:
                continue
            current = principals[principal_id]
            principals[principal_id] = replace(
                current,
                relations=tuple(dict.fromkeys((*current.relations, relation_type))),
            )

    for index, message in enumerate(messages):
        message_id = str(message.get("message_id") or f"turn_{index}")
        speaker = str(message.get("speaker_id") or "unknown")
        role = str(message.get("speaker_role") or "") or None
        prov = _provenance(message, index)
        provenance.append(prov)
        principals.setdefault(
            speaker,
            PrincipalState(principal_id=speaker, role=role, provenance=prov),
        )
        memory_id = stable_memory_id(instance.instance_id, message_id)
        message_text = str(message.get("text") or "")
        memories.append(
            MemoryItemState(
                memory_id=memory_id,
                owner=speaker,
                subject=_subject(message_text),
                scope=_scope(message_text),
                content_ref=message_id,
                created_at=str(message.get("timestamp") or "") or None,
                provenance=prov,
                topics=_topics(message_text),
                subject_principal_ids=principal_mentions(message_text, principals.values()),
            )
        )
        ownership.append({"owner": speaker, "memory_id": memory_id, "provenance": prov})
        # A direct answer from an observable authority to the requester's
        # immediately preceding question is an explicit disclosure event. It
        # should authorize that response record (or a safe projection of it)
        # without turning the speaker's role into a blanket grant.
        requester = str(instance.asking_user_id or "").strip()
        speaker_roles = {
            "owner", "account_owner", "case_owner", "primary_resident",
            "household_manager", "manager", "clinician", "nurse",
            "pharmacist", "caregiver", "agent",
        }
        prior_question = next(
            (
                prior for prior in reversed(messages[max(0, index - 3):index])
                if str(prior.get("speaker_id") or "") == requester
                and _is_question_turn(str(prior.get("text") or ""))
            ),
            None,
        )
        response_text = str(message.get("text") or "")
        if (
            requester
            and prior_question is not None
            and speaker != requester
            and str(role or "").lower() in speaker_roles
        ):
            projection_text = _safe_direct_response_projection(
                str(prior_question.get("text") or ""),
                response_text,
            )
            target_id = memory_id
            if projection_text:
                projection_id = stable_memory_id(
                    instance.instance_id,
                    f"{message_id}:direct_response_projection",
                )
                projection_prov = Provenance(
                    source_message_ids=(message_id,),
                    timestamp=str(message.get("timestamp") or "") or None,
                    turn_index=index,
                    evidence_text=projection_text,
                )
                memories.append(
                    MemoryItemState(
                        memory_id=projection_id,
                        owner=speaker,
                        subject=_subject(projection_text),
                        scope="safe_summary",
                        content_ref=f"{message_id}:direct_response_projection",
                        created_at=str(message.get("timestamp") or "") or None,
                        provenance=projection_prov,
                        topics=tuple(dict.fromkeys((*_topics(projection_text), "public_projection"))),
                        subject_principal_ids=principal_mentions(projection_text, principals.values()),
                    )
                )
                ownership.append({"owner": speaker, "memory_id": projection_id, "provenance": projection_prov})
                statuses[projection_id] = MemoryStatus.ACTIVE
                target_id = projection_id
            direct_response_permissions.append(
                PermissionState(
                    policy_id=f"policy_direct_response_{message_id}",
                    grantor=speaker,
                    grantee=requester,
                    operation="access",
                    target_memory_ids=(target_id,),
                    scope=_scope(str(prior_question.get("text") or "")) or _scope(response_text),
                    valid_from=str(message.get("timestamp") or "") or None,
                    effect="allow",
                    specificity=4,
                    provenance=prov,
                )
            )
        # A safe/public wording reminder is itself a disclosure projection,
        # even when it is not a grant or revoke operation. This keeps explicit
        # safe carriers retrievable for partial disclosure requests.
        for projection_index, projection_text in enumerate(_safe_disclosure_projections(str(message.get("text") or ""))):
            projection_id = stable_memory_id(instance.instance_id, f"{message_id}:public_projection:{projection_index}")
            projection_prov = Provenance(
                source_message_ids=(message_id,),
                timestamp=str(message.get("timestamp") or "") or None,
                turn_index=index,
                evidence_text=projection_text,
            )
            memories.append(
                MemoryItemState(
                    memory_id=projection_id,
                    owner=speaker,
                    subject=projection_text,
                    scope="public",
                    content_ref=f"{message_id}:public_projection:{projection_index}",
                    created_at=str(message.get("timestamp") or "") or None,
                    provenance=projection_prov,
                    topics=tuple(dict.fromkeys((*_topics(projection_text), "public_projection"))),
                    subject_principal_ids=principal_mentions(projection_text, principals.values()),
                )
            )
            ownership.append({"owner": speaker, "memory_id": projection_id, "provenance": projection_prov})
        kind = _operation_kind(str(message.get("text") or ""))
        if kind is None:
            continue
        operation_id = f"op_{message_id}_{kind.value}"
        operations.append(
            OperationState(
                operation_id=operation_id,
                kind=kind,
                actor=speaker,
                target_memory_ids=(memory_id,),
                scope=_scope(str(message.get("text") or "")),
                effective_at=str(message.get("timestamp") or "") or None,
                provenance=prov,
            )
        )
        effect = _permission_effect(kind)
        if effect is not None:
            effect_name, revoked = effect
            grantee = _infer_grantee(str(message.get("text") or ""), aliases)
            permissions.append(
                PermissionState(
                    policy_id=f"policy_{message_id}",
                    grantor=speaker,
                    grantee=grantee,
                    operation="access",
                    target_memory_ids=(memory_id,),
                    scope=_scope(str(message.get("text") or "")),
                    valid_from=str(message.get("timestamp") or "") or None,
                    revoked=revoked,
                    effect=effect_name,
                    specificity=2 if _scope(str(message.get("text") or "")) else 1,
                    provenance=prov,
                )
            )

    # Entity tables may list principals without relationships. Recover only
    # explicit case-subject links from the observable conversation prefix.
    scope_constraints.extend(
        _infer_observable_subject_relations(
            messages=messages,
            aliases=aliases,
            principals=principals,
        )
    )

    # LLM extraction enriches policy semantics but cannot invent state. The
    # deterministic records above remain the safe fallback.
    message_by_id = {str(message.get("message_id") or ""): message for message in messages}
    memory_by_source = {item.content_ref: item for item in memories}
    for item in _llm_policy_operations(instance=instance, llm_client=llm_client, config=config):
        source_id = str(item["source_message_id"])
        source = message_by_id[source_id]
        prov = _provenance(source, messages.index(source))
        kind = OperationKind(item["kind"])
        source_text = str(source.get("text") or "")
        if kind in {OperationKind.DELETE, OperationKind.FORGET} and not _is_lifecycle_instruction(kind, source_text):
            # LLM extraction may interpret a phrase such as "stop taking"
            # as a memory deletion, or a deleted-value probe as a new delete.
            # Lifecycle transitions require an explicit observable mutation.
            continue
        target_ids = tuple(
            memory_by_source[target].memory_id
            for target in item.get("target_message_ids") or []
            if target in memory_by_source
        )
        operation_id = f"llm_op_{source_id}_{kind.value}"
        # Prefer the LLM's richer target/grantee parse over the shallow
        # keyword record for the same observable source message.
        operations = [op for op in operations if op.provenance.source_message_ids != (source_id,)]
        permissions = [p for p in permissions if p.provenance.source_message_ids != (source_id,)]
        operations.append(
            OperationState(
                operation_id=operation_id,
                kind=kind,
                actor=item.get("actor") or str(source.get("speaker_id") or "") or None,
                target_memory_ids=target_ids,
                target_subject=item.get("target_subject"),
                scope=item.get("scope") or _scope(str(source.get("text") or "")),
                effective_at=item.get("effective_at") or str(source.get("timestamp") or "") or None,
                provenance=prov,
            )
        )
        lowered_source = source_text.lower()
        # Only explicit access-governance operations create PermissionState.
        # An update note may contain words such as stop/hold/avoid as content
        # instructions; those are not permission revocations.
        effect = (
            item.get("effect")
            if kind in {OperationKind.GRANT, OperationKind.SHARE, OperationKind.REVOKE}
            else None
        ) or (
            "deny" if kind == OperationKind.REVOKE
            else "allow" if kind in {OperationKind.GRANT, OperationKind.SHARE}
            else None
        )
        if effect is not None and re.search(r"\b(cannot|can't|can not|do not|don't|no|not|without|neither)\b", lowered_source):
            effect = "deny"
        elif kind in {OperationKind.GRANT, OperationKind.SHARE}:
            effect = "allow"
        if effect in {"allow", "deny"}:
            permissions.append(
                PermissionState(
                    policy_id=f"llm_policy_{source_id}",
                    grantor=str(source.get("speaker_id") or "") or None,
                    grantee=item.get("grantee"),
                    operation="access",
                    target_memory_ids=target_ids,
                    target_subject=item.get("target_subject"),
                    scope=item.get("scope") or _scope(str(source.get("text") or "")),
                    valid_from=item.get("effective_at") or str(source.get("timestamp") or "") or None,
                    revoked=effect == "deny",
                    effect=effect,
                    specificity=3 if item.get("target_subject") or target_ids else 1,
                    provenance=prov,
                )
            )
        effective_subject = item.get("target_subject") or _subject(str(source.get("text") or ""))
        if effective_subject:
            target_ids = tuple(
                dict.fromkeys(
                    list(target_ids)
                    + [
                        memory.memory_id
                        for memory in memories
                        if effective_subject.lower() in memory.provenance.evidence_text.lower()
                    ]
                )
            )
        # Lifecycle targets are resolved below after all observable memories
        # and extracted operations are available.

    # Add direct-response disclosure records after LLM operation replacement so
    # a semantic extractor cannot accidentally discard this source-bound grant.
    permissions.extend(direct_response_permissions)

    # Rebind operations and permissions to the observable records they refer
    # to.  This prevents a delete/forget sentence from marking only its own
    # control message as deleted and leaves unresolved references uncertain.
    source_to_memory = {item.content_ref: item.memory_id for item in memories}
    resolved_operations: list[OperationState] = []
    for operation in operations:
        source_id = operation.provenance.source_message_ids[0] if operation.provenance.source_message_ids else ""
        resolved_ids = _resolve_operation_targets(
            operation,
            memories=memories,
            source_memory_id=source_to_memory.get(source_id, ""),
        )
        resolved = replace(operation, target_memory_ids=resolved_ids)
        resolved_operations.append(resolved)
        if operation.kind == OperationKind.DELETE:
            for target_id in resolved_ids:
                statuses[target_id] = MemoryStatus.DELETED
        elif operation.kind == OperationKind.FORGET:
            for target_id in resolved_ids:
                statuses[target_id] = MemoryStatus.FORGOTTEN
        elif (
            operation.kind == OperationKind.UPDATE
            and _is_state_replacing_update(operation.provenance.evidence_text)
            and not _has_field_continuity_language(operation.provenance.evidence_text)
        ):
            for target_id in resolved_ids:
                statuses[target_id] = MemoryStatus.SUPERSEDED
    operations = resolved_operations

    operation_targets_by_source = {
        operation.provenance.source_message_ids[0]: operation.target_memory_ids
        for operation in operations
        if operation.provenance.source_message_ids
    }
    permissions = [
        replace(
            permission,
            target_memory_ids=operation_targets_by_source.get(
                permission.provenance.source_message_ids[0], permission.target_memory_ids
            ),
        )
        for permission in permissions
    ]

    requester = str(instance.asking_user_id or "")
    if requester and requester not in principals:
        principals[requester] = PrincipalState(
            principal_id=requester,
            role=str((instance.metadata.get("requester") or {}).get("role") or "") or None,
        )

    normalized_memories = [
        replace(item, status=statuses.get(item.memory_id, item.status))
        for item in memories
    ]
    delegation = tuple(item for item in permissions if item.grantee is not None)
    revocations = tuple(item for item in permissions if item.revoked)
    return PolicyState(
        principals=tuple(principals.values()),
        memory_items=tuple(normalized_memories),
        ownership_relations=tuple(ownership),
        permission_relations=tuple(permissions),
        delegation_relations=delegation,
        revocation_relations=revocations,
        memory_status={item.memory_id: item.status for item in normalized_memories},
        operation_history=tuple(operations),
        temporal_constraints=(),
        scope_constraints=tuple(scope_constraints),
        provenance=tuple(provenance),
        as_of_turn_id=str((instance.metadata.get("observable") or {}).get("as_of_turn_id") or "") or None,
    )


def policy_state_audit_dict(state: PolicyState, *, blocked_memory_ids: set[str] | None = None) -> dict[str, Any]:
    """Serialize state without writing blocked memory plaintext to artifacts."""
    payload = schema_to_dict(state)
    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("[blocked_memory_content_redacted]" if key == "evidence_text" else redact(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    payload = redact(payload)
    return payload
