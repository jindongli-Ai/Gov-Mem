"""Natural-language query intent parsing for policy reasoning."""

from __future__ import annotations

import json
import re
from typing import Any

from gov_mem.data.schema import MemoryInstance
from gov_mem.entity_resolution import EntityResolution, resolve_query_entities
from gov_mem.general_lexicon import topics_from_text
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.policy_schema import PolicyState, Provenance, QueryIntent


SYSTEM_PROMPT = """
You parse a natural-language request for a shared-memory governance engine.
Return JSON only with: target_subject, target_scope, requested_operation,
mentioned_entities, requested_topics, requested_attributes, confidence,
uncertainty, sensitivity_topics, disclosure_mode.
requested_operation must be one of access, share, update, delete, forget,
grant, revoke, unknown. Do not answer the request and do not infer permission.
Use only the supplied query and observable principal summary.
target_subject is only a named entity or concrete record subject. Do not put
descriptions such as "current medicine plan" or "pickup time for the patient"
in target_subject; put those concepts in requested_topics or attributes.
Use broad reusable topic names such as logistics, scheduling, callback,
medication, clinical, laboratory, imaging, safety, or contact. A question
asking what information can be used in an update/share is an access request;
only an explicit command to perform the governance operation is update/share.
requested_attributes must contain one short label for every distinct fact the
question asks for. Do not leave it empty for a multi-part request. Use the
question's wording, not evaluator labels. Mark a request as safe_summary when
it asks for a broad, public, sponsor-safe, household-safe, mixed-audience, or
helper-facing summary. sensitivity_topics should contain semantic categories
that may require narrower disclosure than ordinary operational facts: health,
medication, laboratory, imaging, identity, private_location, private_contact,
credential, finance, legal, access_control, privacy, or other_sensitive. Use
an empty list for ordinary facts. disclosure_mode must be one of exact,
broad, yes_no, historical, or unknown. Classify the requested information,
not merely words copied from the query. An exact budget, visit reason,
private room, diagnosis, or whether-a-fact-is-true request is sensitive even
when the user's pretext sounds ordinary. A broad public summary is broad.
""".strip()


FIELD_EXTRACTION_SYSTEM_PROMPT = """
Extract the requested fact fields from the user's question. Return JSON only:
{"requested_attributes": ["short field label", ...]}. Include one label for
each independently answerable fact. Split comma- and conjunction-separated
lists, including lists introduced by "including", "with", or "covering".
Preserve meaningful qualifiers such as current, approved, exact, public, or
safe. Do not return an aggregate label such as "all requested fields",
"everything", "details", or "the plan" when the question names its parts.
Do not answer the question, choose evidence, infer authorization, or use any
dataset labels. If the question asks for one broad fact without named parts,
return one concise label for that fact.
When a question asks for a plan, summary, instructions, or details and then
names subparts, include both the overall requested item and each named
subpart in the checklist.
""".strip()


_SENSITIVITY_CATEGORIES = {
    "health", "medication", "laboratory", "imaging", "identity", "private_location",
    "private_contact", "credential", "finance", "legal", "access_control", "privacy",
    "other_sensitive",
}

_SENSITIVE_TOPIC_HINTS = {
    "health": ("diagnosis", "condition", "symptom", "visit reason", "hormone", "gender clinic", "clinical", "care"),
    "medication": ("medication", "medicine", "drug", "dose", "prescription", "started", "start", "stop", "treatment"),
    "laboratory": ("lab", "laboratory", "bloodwork", "test result", "normal", "cbc", "cmp"),
    "imaging": ("imaging", "scan", "ultrasound", "mri", "biopsy", "pathology", "x-ray"),
    "identity": ("identity", "who", "same person", "customer", "patient", "account holder"),
    "private_location": ("private room", "private bay", "private site", "exact room", "exact bay", "address", "where"),
    "private_contact": ("phone", "mobile", "number", "voicemail", "contact", "callback", "email"),
    "credential": ("credential", "badge", "code", "password", "pin", "token", "key", "login"),
    "finance": ("budget", "amount", "discount", "price", "payment", "invoice", "billing", "salary", "expense", "financial"),
    "legal": ("contract", "nda", "legal", "counsel", "settlement", "agreement"),
    "access_control": ("permission", "authorization", "access", "share", "private file", "internal"),
    "privacy": ("private", "confidential", "restricted", "sensitive", "not for sharing"),
}


def _fallback_sensitivity(query: str) -> tuple[str, ...]:
    lowered = str(query or "").lower()
    result = [
        category for category, hints in _SENSITIVE_TOPIC_HINTS.items()
        if any(hint in lowered for hint in hints)
    ]
    if re.search(r"\b(?:exact|precise|specific|whether|if|yes\s+or\s+no|confirm|tell me if)\b", lowered) and result:
        result.append("other_sensitive")
    return tuple(dict.fromkeys(result))


def _fallback_disclosure_mode(query: str) -> str:
    lowered = str(query or "").lower()
    if re.search(r"\b(?:yes\s+or\s+no|whether|if|confirm|is\s+this|was\s+this|did\s+it)\b", lowered):
        return "yes_no"
    if re.search(r"\b(?:exact|precise|specific|full|complete)\b", lowered):
        return "exact"
    if re.search(r"\b(?:safe|broad|public|sponsor-safe|household-safe|mixed-audience|helper-facing)\b", lowered):
        return "broad"
    if re.search(r"\b(?:deleted|old|former|previous|used\s+to|before)\b", lowered):
        return "historical"
    return "unknown"


def _topics_from_text(text: str) -> tuple[str, ...]:
    return topics_from_text(text)


def _normalize_topic(value: object) -> str:
    topic = str(value or "").strip().lower().replace("-", "_")
    return {
        "callback": "communication",
        "contact": "communication",
        "medical": "health",
        "clinical": "health",
        "transport": "logistics",
        "transportation": "logistics",
        "calendar": "scheduling",
    }.get(topic, topic)


def _operation_from_text(query: str) -> str:
    lowered = query.lower()
    if re.search(r"^\s*(please\s+)?(?:forget|erase|purge)\b", lowered) or re.search(r"\b(?:forget|erase|purge)\s+(?:this|that|the)\b", lowered):
        return "forget"
    if re.search(r"^\s*(please\s+)?(?:delete|remove)\b", lowered) or re.search(r"\b(?:delete|remove)\s+(?:this|that|the)\b", lowered):
        return "delete"
    if re.search(r"^\s*(please\s+)?(?:update|change|replace)\b", lowered) or re.search(r"\b(?:update|change|replace)\s+(?:this|that|the)\b", lowered):
        return "update"
    if re.search(r"^\s*(please\s+)?(?:grant|authorize|allow)\b", lowered):
        return "grant"
    if re.search(r"^\s*(please\s+)?(?:revoke|withdraw)\b", lowered):
        return "revoke"
    if re.search(r"^\s*(please\s+)?(?:share|send|release)\b", lowered):
        return "share"
    if query.strip():
        return "access"
    return "unknown"


def _scope_from_text(query: str) -> str | None:
    lowered = query.lower()
    # A request that explicitly asks for a snapshot using only the current
    # state is a projection request.  Treating it as a safe projection keeps
    # stale/exact historical layers out of the answering context while still
    # allowing the policy-approved broad state to be retrieved.  This is a
    # linguistic constraint, not a dataset or query-type rule.
    if _is_current_state_projection_request(query):
        return "safe_summary"
    if re.search(
        r"\b(?:safe|broad|public|sponsor-safe|sponsor-ready|household-safe|mixed-audience|helper-facing)\s+"
        r"(?:wording|summary|sentence|update|recap|brief|status|one-line\s+status)\b"
        r"|\bhigh[- ]level\s+(?:status|summary|update|recap)\b",
        lowered,
    ) or re.search(r"\bkeep\s+(?:it|this)\s+high[- ]level\b", lowered):
        return "safe_summary"
    for value in ("scheduling", "logistics", "clinical", "medical", "budget", "contract", "project", "public", "private", "broad"):
        if value in lowered:
            return value
    return None


def _is_current_state_projection_request(query: str) -> bool:
    """Recognize explicit current-state-only projection language."""
    lowered = re.sub(r"\s+", " ", str(query or "").lower()).strip()
    has_projection_noun = bool(
        re.search(r"\b(?:summary|snapshot|recap|overview|state|status)\b", lowered)
    )
    has_current_only_boundary = bool(
        re.search(
            r"\b(?:using|with|showing|report(?:ing)?|keep(?:ing)?)\s+"
            r"(?:only\s+)?(?:the\s+)?(?:current|active|retained)\s+state\b",
            lowered,
        )
        or re.search(r"\b(?:only|just)\s+(?:the\s+)?(?:current|active|retained)\s+values?\b", lowered)
    )
    return has_projection_noun and has_current_only_boundary


def _subject_from_text(query: str) -> str | None:
    # Generic entity grounding for proper-name subjects. This is only a
    # fallback when the semantic parser leaves the subject unresolved.
    # Prefer case-preserving named entities before the broader lifecycle/date
    # pattern below.  The latter is intentionally case-insensitive and can
    # otherwise mistake ``current target date`` for the requested object
    # ``Project Maple``.
    named_matches = re.findall(r"\b[A-Z][a-z0-9&'-]+(?:\s+[A-Z][a-z0-9&'-]+){1,5}\b", query)
    named_filtered = [
        item for item in named_matches
        if item.split()[0].lower() not in {"then", "what", "which", "where", "when", "give", "tell", "for", "as"}
    ]
    if named_filtered:
        return max(named_filtered, key=len).strip().removesuffix("'s")
    targeted = re.search(
        r"\b(?:deleted|delete|forgotten|forget|current|old|active)\s+(?:place|record|memory|contact|line|number)?\s*([A-Z][a-z0-9&'-]+(?:\s+[A-Z][a-z0-9&'-]+){1,4})",
        query,
        re.IGNORECASE,
    )
    if targeted:
        return targeted.group(1).strip()
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z0-9&'-]+){1,5}\b", query)
    if matches:
        filtered = [item for item in matches if item.split()[0].lower() not in {"then", "what", "which", "where", "when", "give", "tell", "is", "are"}]
        if filtered:
            return max(filtered, key=len)
        return max(matches, key=len)
    match = re.search(r"\b(?:deleted|current|old|active)\s+(?:place|record|memory|contact|line|number)\s+([^,?.]+)", query, re.IGNORECASE)
    return match.group(1).strip() if match else None


_GENERIC_SUBJECT_WORDS = {
    "what", "which", "current", "latest", "summary", "recap", "timing",
    "description", "plan", "right", "now", "can", "use", "remind",
    "tell", "whether", "because", "the", "active", "callback", "number",
    "return", "customer", "account", "mapping", "scope", "family", "release",
}

_GENERIC_SUBJECT_SUFFIXES = {
    "plan", "summary", "recap", "thread", "record", "file",
    "status", "details", "information", "request", "question", "matter",
    "note", "update", "answer", "memory",
}


def _ground_subject(candidate: str | None, query: str) -> str | None:
    named_subject = _subject_from_text(query)
    candidate_words = set(re.findall(r"[a-z0-9]+", (candidate or "").lower()))
    if named_subject and (not candidate or candidate_words & _GENERIC_SUBJECT_WORDS):
        return named_subject
    return candidate or named_subject


def _subject_matches_query_object(candidate: str | None, named_subject: str, query: str) -> bool:
    """Reject a state term that is merely an attribute of the named object."""
    if not candidate:
        return False
    candidate_words = set(re.findall(r"[a-z0-9]+", str(candidate).lower()))
    named_words = set(re.findall(r"[a-z0-9]+", str(named_subject).lower()))
    if named_words and named_words.issubset(candidate_words):
        return True
    return str(candidate).lower() in str(query).lower() and str(named_subject).lower() in str(query).lower()


def _observable_subject_catalog(state: PolicyState) -> tuple[str, ...]:
    values: list[str] = []
    for item in state.memory_items:
        values.extend(value for value in (item.subject, item.scope) if value)
    values.extend(principal.display_name for principal in state.principals if principal.display_name)
    return tuple(str(value).strip().lower() for value in values if str(value).strip())


def _ground_subject_to_state(candidate: str | None, *, state: PolicyState) -> str | None:
    """Keep only a subject grounded in observable state.

    An intent model may turn an entity plus a generic noun, such as
    ``Juniper Parcel plan``, into one subject string. That paraphrase must not
    narrow policy matching. Strip generic suffixes when the remaining phrase
    is observable; otherwise leave the subject unresolved.
    """
    value = str(candidate or "").strip()
    if not value:
        return None
    catalog = _observable_subject_catalog(state)
    lowered = value.lower()
    parts = value.split()
    stripped_generic_suffix = False
    while len(parts) > 1 and parts[-1].lower().strip(".,:;?!") in _GENERIC_SUBJECT_SUFFIXES:
        stripped_generic_suffix = True
        parts.pop()
        shortened = " ".join(parts).strip()
        if any(shortened.lower() in item or item in shortened.lower() for item in catalog):
            return shortened
    candidate_words = {word.lower().strip(".,:;?!") for word in parts}
    if candidate_words & _GENERIC_SUBJECT_WORDS:
        named_chunks = re.findall(
            r"\b[A-Z][a-z0-9&'-]+(?:\s+[A-Z][a-z0-9&'-]+){0,4}\b",
            value,
        )
        for chunk in named_chunks:
            if any(chunk.lower() in item or item in chunk.lower() for item in catalog):
                return chunk.strip()
        return None
    if lowered in catalog:
        return value
    if not stripped_generic_suffix and not any(part.lower().strip(".,:;?!") in _GENERIC_SUBJECT_SUFFIXES for part in parts):
        if any(lowered in item or item in lowered for item in catalog):
            return value
    return None


def _grounded_named_objects_in_query(query: str, *, state: PolicyState) -> tuple[str, ...]:
    """Return distinct observable non-person objects named by a query.

    A query can legitimately span two shared-memory lanes, for example a
    private project record plus its public logistics thread.  Selecting the
    last proper noun as the singular ``target_subject`` silently drops the
    first lane.  This helper only detects names already visible in the
    observable state; it does not invent entities or expand authorization.
    """
    principal_names = {
        str(principal.display_name or "").strip().casefold()
        for principal in state.principals
        if str(principal.display_name or "").strip()
    }
    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){1,5}\b",
        str(query or ""),
    )
    visible_text = " ".join(
        " ".join((str(item.subject or ""), str(item.provenance.evidence_text or "")))
        for item in state.memory_items
    ).casefold()
    result: list[str] = []
    for candidate in candidates:
        value = re.sub(r"[,.?!:;]+$", "", candidate).strip()
        lowered = value.casefold().removesuffix("'s")
        if not lowered or lowered in principal_names or lowered in {item.casefold() for item in result}:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])", visible_text):
            result.append(value.removesuffix("'s"))
    return tuple(result)


def _is_multi_object_compound_query(query: str, *, state: PolicyState, attributes: tuple[str, ...]) -> bool:
    """Identify benign compound reads that span multiple named objects.

    This is deliberately conservative.  Identity-mapping, impersonation, and
    exact cross-project probes must retain a narrow subject so the policy
    engine can deny them; ordinary multi-field utility reads may use the full
    already-authorized set and let field-level retrieval bind each value.
    """
    lowered = str(query or "").casefold()
    if not re.search(r"[,;]|\band\b|\b(?:including|with|covering)\b", lowered):
        return False
    if re.search(
        r"\b(?:customer|account|identity|mapping|impersonat|cross[- ]project|exact\s+behind|"
        r"credential|password|pin|token|secret)\b",
        lowered,
    ):
        return False
    if len(attributes) < 2:
        return False
    if len(_grounded_named_objects_in_query(query, state=state)) >= 2:
        return True
    # The second lane is often elliptical: ``my private bay`` refers to the
    # case already established in state, while ``public Hollow Coaches room``
    # names only the public lane.  The private/public contrast is enough to
    # avoid binding the whole request to the final named lane.
    return bool(
        re.search(r"\bprivate\b", lowered)
        and re.search(r"\bpublic\b", lowered)
    )


def _fallback_attributes(query: str) -> tuple[str, ...]:
    """Extract a conservative field checklist when the semantic call fails.

    This parser only separates the query's own grammatical list. It never
    creates answer values, authorization, or dataset-specific field names.
    """
    text = re.sub(r"\s+", " ", str(query or "").strip())
    if not text:
        return ()
    lowered = text.lower()
    list_match = re.search(
        r"\b(?:including|with|covering|consisting of|as follows)\b(.+?)(?:\?|$)",
        text,
        flags=re.IGNORECASE,
    )
    candidate = list_match.group(1) if list_match else text
    candidate = re.sub(
        r"^(?:as\s+of\s+now\s*,?\s*)?(?:what\s+(?:are|is)|which|tell\s+me|give\s+me)\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    # Questions such as "what I should start, what I should stop, and what
    # changes today" are already independently answerable clauses.
    candidate = re.sub(r"\b(?:what|which)\s+(?=(?:i|we|they|the|my|our)\b)", "", candidate, flags=re.IGNORECASE)
    parts = re.split(r"\s*(?:,|;|\band\b|\bor\b)\s*", candidate, flags=re.IGNORECASE)
    labels: list[str] = []
    for part in parts:
        value = part.strip(" .:()? ")
        value = re.sub(
            r"^(?:what\s+are|what\s+is|which|tell\s+me|give\s+me|the|my|our|their)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\s+", " ", value).strip(" .:()? ")
        if not value or len(value.split()) > 12:
            continue
        if value.lower() in {"all requested fields", "everything", "details", "the plan"}:
            continue
        labels.append(value.lower())
    if labels:
        return tuple(dict.fromkeys(labels))
    # A single broad request still needs a field label for the answer contract.
    single = re.sub(
        r"^(?:what\s+(?:are|is)|which|tell\s+me|give\s+me)\s+",
        "",
        lowered,
        flags=re.IGNORECASE,
    ).strip(" ?.")
    single = re.sub(r"^(?:my|the|our|their)\s+", "", single, flags=re.IGNORECASE)
    return (single,) if single and len(single.split()) <= 12 else ()


def _is_non_fact_query_fragment(label: object, *, confirmation_query: bool = False) -> bool:
    """Recognize grammatical wrappers that are not independently requested facts.

    The intent model occasionally echoes a whole conditional/confirmation
    clause as an attribute (for example if ... or right). These fragments
    pollute retrieval and later compete with the typed fields the same model
    produced. The test is deliberately grammatical: it does not mention any
    domain entity, name, number, or benchmark vocabulary.
    """
    text = re.sub(r"\s+", " ", str(label or "").strip().lower()).strip(" .:()? ")
    if not text:
        return True
    if text in {"right", "correct", "is that so", "isn't it", "is it"}:
        return True
    if re.match(r"^(?:if|assuming|provided that|given that|that means|which means)\b", text):
        return True
    if not confirmation_query:
        return False
    # In a confirmation question, a first-person/quantified proposition is a
    # premise to verify, not a separate memory slot. Noun phrases such as
    # current departure time do not match this clause shape.
    if re.match(r"^(?:i|we|you|they|he|she|it|no|nobody|nothing)\b", text) and re.search(
        r"\b(?:am|are|is|was|were|have|has|had|can|could|will|would|should|"
        r"leave|arrive|arrives|expected|means|need|needs|want|wants|"
        r"remain|remains|go|goes|come|comes)\b",
        text,
    ):
        return True
    return False


def _filter_requested_attributes(
    question: str,
    attributes: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Remove query wrappers while preserving concrete requested slots."""
    lowered = re.sub(r"\s+", " ", str(question or "").strip().lower())
    confirmation_query = bool(re.search(
        r"(?:\?|\s)(?:right|correct|isn't it|is that so)\s*[?.]*$",
        lowered,
    )) or bool(re.match(r"^\s*(?:if|assuming|given that)\b", lowered))
    return tuple(dict.fromkeys(
        str(value).strip().lower()
        for value in attributes
        if str(value).strip()
        and not _is_non_fact_query_fragment(value, confirmation_query=confirmation_query)
    ))


def _is_aggregate_attributes(attributes: tuple[str, ...] | list[str]) -> bool:
    values = {str(item).strip().lower() for item in attributes if str(item).strip()}
    return not values or values.issubset({
        "all requested fields", "all fields", "everything", "details", "the details",
        "the plan", "current plan", "requested information",
    })


def _expand_requested_attributes(
    *,
    question: str,
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> tuple[str, ...]:
    if llm_client is not None and llm_client.is_available():
        payload = {"query": question}
        assert_runtime_payload_safe(payload, context="requested_field_extraction_prompt")
        try:
            raw = llm_client.chat_json(
                model=resolve_llm_model(config, "reasoning"),
                system_prompt=FIELD_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            values = raw.get("requested_attributes") if isinstance(raw, dict) else []
            if isinstance(values, str):
                values = [values]
            normalized = tuple(dict.fromkeys(
                str(item).strip().lower()
                for item in values or []
                if str(item).strip()
                and str(item).strip().lower() not in {"all requested fields", "everything", "details"}
            ))
            if normalized:
                return _add_overall_request_field(
                    question,
                    _filter_requested_attributes(question, normalized),
                )
        except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return _add_overall_request_field(
        question,
        _filter_requested_attributes(question, _fallback_attributes(question)),
    )


def _add_overall_request_field(
    question: str,
    attributes: tuple[str, ...],
) -> tuple[str, ...]:
    """Keep an aggregate request and make vague action clauses explicit.

    The aggregate is a semantic completeness anchor, not a dataset field. It
    prevents an answer model from treating "what should I start" as the whole
    request while still leaving concrete values to the evidence.
    """
    normalized: list[str] = []
    vague_replacements = {
        "i should start": "medications to start",
        "what i should start": "medications to start",
        "i should stop": "medications to stop",
        "what i should stop": "medications to stop",
        "what changes today": "medication changes today",
    }
    for value in attributes:
        label = re.sub(r"\s+", " ", str(value).strip().lower())
        normalized.append(vague_replacements.get(label, label))

    text = re.sub(r"\s+", " ", str(question or "").strip())
    overall: str | None = None
    match = re.search(
        r"\bwhat\s+(?:is|are)\s+(.+?)(?:,\s*|\s+including\b|\s+with\b|\s+covering\b|:\s*)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .:?")
        if re.search(r"\b(?:plan|summary|instructions?|details?|changes?)\b", candidate, re.IGNORECASE):
            overall = candidate.lower()

    lowered = text.lower()
    medication_request = bool(re.search(r"\b(?:medications?|medicines?|drugs?|treatment)\b", lowered))
    action_request = bool(re.search(r"\b(?:should|start|stop|continue|change|use|take)\b", lowered))
    if overall is None and medication_request and action_request and re.search(r"\band\b|,", lowered):
        overall = "current medication instructions"

    if overall is None and re.search(r"\b(?:treatment|care)\s+plan\b", lowered):
        overall = "current treatment plan"

    # A time-bounded plan is incomplete without its calendar anchor, even when
    # the user lists only the window and operational fields.  Keep this as a
    # semantic completeness invariant for plan-state queries; it creates no
    # value and does not consult evaluator metadata.
    if (
        overall
        and re.search(r"\bplan\b", overall)
        and re.search(r"\b(?:window|schedule|scheduled|arrival|departure|appointment|day|date|time)\b", lowered)
        and not any(re.search(r"\b(?:date|day)\b", str(value).lower()) for value in normalized)
    ):
        normalized.append("current date")

    return tuple(dict.fromkeys(((overall,) if overall else ()) + tuple(normalized)))


def _normalize(raw: Any, *, instance: MemoryInstance) -> QueryIntent:
    payload = dict(raw) if isinstance(raw, dict) else {}
    operation = str(payload.get("requested_operation") or "unknown").strip().lower()
    if operation not in {"access", "share", "update", "delete", "forget", "grant", "revoke", "unknown"}:
        operation = "unknown"
    uncertainty = payload.get("uncertainty") or []
    if isinstance(uncertainty, str):
        uncertainty = [uncertainty]
    entities = payload.get("mentioned_entities") or []
    if isinstance(entities, str):
        entities = [entities]
    topics = payload.get("requested_topics") or []
    if isinstance(topics, str):
        topics = [topics]
    attributes = payload.get("requested_attributes") or []
    if isinstance(attributes, str):
        attributes = [attributes]
    normalized_topics = tuple(dict.fromkeys(
        _normalize_topic(item)
        for item in list(topics) + list(_topics_from_text(instance.question))
        if str(item).strip()
    ))
    normalized_attributes = tuple(dict.fromkeys(
        str(item).strip().lower() for item in attributes if str(item).strip()
    ))
    normalized_attributes = _filter_requested_attributes(instance.question, normalized_attributes)
    sensitivity = tuple(dict.fromkeys(
        str(item).strip().lower()
        for item in (payload.get("sensitivity_topics") or [])
        if str(item).strip().lower() in _SENSITIVITY_CATEGORIES
    )) or _fallback_sensitivity(instance.question)
    disclosure_mode = str(payload.get("disclosure_mode") or "unknown").strip().lower()
    if disclosure_mode not in {"exact", "broad", "yes_no", "historical", "unknown"}:
        disclosure_mode = _fallback_disclosure_mode(instance.question)
    requester = str(instance.asking_user_id or "").strip() or None
    confidence = float(payload.get("confidence") or 0.0)
    target_scope = str(payload.get("target_scope") or "").strip() or _scope_from_text(instance.question)
    if _scope_from_text(instance.question) == "safe_summary":
        target_scope = "safe_summary"
    target_subject = str(payload.get("target_subject") or "").strip()
    target_subject = re.sub(r"^(?:if|whether|that|the)\s+", "", target_subject, flags=re.IGNORECASE).strip()
    if not target_subject or target_subject.lower() in {"current", "the current record", "the memory"}:
        target_subject = _subject_from_text(instance.question)
    if not normalized_attributes and re.search(
        r"\b(?:what are|what is|which|give me|tell me|list|summary|recap)\b.*\b(?:and|,|;|current|latest|active|scope|status|date|site|amount|phone|extension|dose|plan)\b",
        instance.question.lower(),
    ):
        # The semantic parser is authoritative when it provides attributes;
        # this fallback only preserves the fact that a compound request needs
        # coverage rather than a one-clause answer.
        normalized_attributes = _fallback_attributes(instance.question)
    return QueryIntent(
        requester=requester,
        target_subject=_ground_subject(target_subject, instance.question),
        target_scope=target_scope or None,
        requested_operation=operation,
        mentioned_entities=tuple(str(item) for item in entities if str(item).strip()),
        confidence=max(0.0, min(1.0, confidence)),
        uncertainty=tuple(str(item) for item in uncertainty if str(item).strip()),
        requested_topics=normalized_topics,
        requested_attributes=normalized_attributes,
        sensitivity_topics=sensitivity,
        disclosure_mode=disclosure_mode,
        provenance=(Provenance(evidence_text=instance.question),),
    )


def _attach_entity_resolution(
    intent: QueryIntent,
    *,
    question: str,
    state: PolicyState,
) -> QueryIntent:
    """Bind person mentions before policy selection and content retrieval."""
    resolution: EntityResolution = resolve_query_entities(question, state.principals)
    target_subject = intent.target_subject
    if resolution.ambiguous:
        target_subject = None
    elif resolution.resolved_principal_id:
        principal = next(
            (item for item in state.principals if item.principal_id == resolution.resolved_principal_id),
            None,
        )
        if principal is not None and principal.display_name:
            target_subject = principal.display_name
    uncertainty = list(intent.uncertainty)
    if resolution.ambiguous and "principal_identity_ambiguous" not in uncertainty:
        uncertainty.append("principal_identity_ambiguous")
    return QueryIntent(
        **{
            **intent.__dict__,
            "target_subject": target_subject,
            "mentioned_principal_ids": resolution.mentioned_principal_ids,
            "identity_ambiguous": resolution.ambiguous,
            "identity_resolution_reason": resolution.reason,
            "uncertainty": tuple(uncertainty),
        }
    )


def _with_expanded_attributes(
    intent: QueryIntent,
    *,
    question: str,
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> QueryIntent:
    attributes = tuple(intent.requested_attributes)
    question_lower = str(question or "").lower()
    # Compound-field extraction is a general query operation.  Restricting it
    # to words such as ``plan`` or ``summary`` caused ordinary requests like
    # "bay, badge, scope, and blocker" to collapse into one model field.
    # Re-ask the semantic field extractor whenever the surface grammar clearly
    # contains a list; it still sees only the user query and never chooses
    # evidence or authorization.
    list_marker = bool(re.search(r"[,;]|\band\b|\b(?:including|with|covering)\b", question_lower))
    interrogative = bool(re.search(r"\b(?:what|which|where|when|who|give|tell|list|provide)\b", question_lower))
    needs_compound_completion = list_marker and interrogative
    if not _is_aggregate_attributes(attributes) and not needs_compound_completion:
        return intent
    expanded = _expand_requested_attributes(
        question=question,
        llm_client=llm_client,
        config=config,
    )
    merged = _add_overall_request_field(
        question,
        _filter_requested_attributes(question, tuple(dict.fromkeys((*attributes, *expanded)))),
    )
    return QueryIntent(
        **{
            **intent.__dict__,
            "requested_attributes": merged,
        }
    )


def parse_query_intent(
    *,
    instance: MemoryInstance,
    state: PolicyState,
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> QueryIntent:
    """Parse query semantics; never use dataset query labels."""
    if not instance.asking_user_id:
        return QueryIntent(
            requester=None,
            target_subject=None,
            target_scope=_scope_from_text(instance.question),
            requested_operation=_operation_from_text(instance.question),
            confidence=0.0,
            uncertainty=("requester_identity_missing",),
            requested_topics=_topics_from_text(instance.question),
            provenance=(Provenance(evidence_text=instance.question),),
        )

    if llm_client is not None and llm_client.is_available():
        principals = [
            {
                "principal_id": item.principal_id,
                "role": item.role,
                "display_name": item.display_name,
                "aliases": list(item.aliases),
                "relations": list(item.relations),
                "entity_type": item.entity_type,
            }
            for item in state.principals
        ]
        payload = {"query": instance.question, "requester_id": instance.asking_user_id, "principals": principals}
        assert_runtime_payload_safe(payload, context="query_intent_prompt")
        try:
            raw = llm_client.chat_json(
                model=resolve_llm_model(config, "reasoning"),
                system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps(payload, ensure_ascii=False),
            )
            result = _normalize(raw, instance=instance)
            result = _with_expanded_attributes(
                result,
                question=instance.question,
                llm_client=llm_client,
                config=config,
            )
            result = _attach_entity_resolution(
                result,
                question=instance.question,
                state=state,
            )
            # Reject fluent but ungrounded subject descriptions returned by a
            # model and fall back to the named entity in the query. This keeps
            # relevance broad without allowing a paraphrase to become state.
            if not result.identity_ambiguous:
                grounded_subject = _ground_subject(result.target_subject, instance.question)
                grounded_subject = _ground_subject_to_state(grounded_subject, state=state)
                named_subject = _ground_subject_to_state(_subject_from_text(instance.question), state=state)
                if named_subject and (
                    grounded_subject is None
                    or not _subject_matches_query_object(grounded_subject, named_subject, instance.question)
                ):
                    grounded_subject = named_subject
                if grounded_subject != result.target_subject:
                    result = QueryIntent(**{**result.__dict__, "target_subject": grounded_subject})
            lowered_query = instance.question.lower().strip()
            interrogative = lowered_query.startswith(("what ", "which ", "how ", "can ", "could ", "where ", "when ", "who "))
            explicit_command = bool(re.match(r"^(?:please\s+)?(?:update|change|replace|delete|remove|forget|erase|purge|grant|authorize|allow|revoke|withdraw|share|send|release)\b", lowered_query))
            if interrogative and not explicit_command and result.requested_operation in {"update", "delete", "forget", "grant", "revoke", "share"}:
                result = QueryIntent(
                    **{**result.__dict__, "requested_operation": "access"}
                )
            if _is_multi_object_compound_query(
                instance.question,
                state=state,
                attributes=tuple(result.requested_attributes),
            ):
                # Keep policy authorization over the requester's observable
                # authorized set, while retrieval binds each requested field
                # to its own named lane. A singular last-name binding is
                # unsafe for a legitimate private/public compound read.
                result = QueryIntent(
                    **{
                        **result.__dict__,
                        "target_subject": None,
                        "uncertainty": tuple(dict.fromkeys((*result.uncertainty, "compound_named_objects"))),
                    }
                )
            if result.requested_operation != "unknown":
                return result
        except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
            pass

    result = QueryIntent(
        requester=instance.asking_user_id,
        target_subject=_ground_subject_to_state(_ground_subject(None, instance.question), state=state),
        target_scope=_scope_from_text(instance.question),
        requested_operation=_operation_from_text(instance.question),
        confidence=0.35,
        uncertainty=("llm_intent_parse_unavailable",),
        requested_topics=_topics_from_text(instance.question),
        requested_attributes=_fallback_attributes(instance.question),
        sensitivity_topics=_fallback_sensitivity(instance.question),
        disclosure_mode=_fallback_disclosure_mode(instance.question),
        provenance=(Provenance(evidence_text=instance.question),),
    )
    result = _attach_entity_resolution(
        result,
        question=instance.question,
        state=state,
    )
    result = _with_expanded_attributes(
        result,
        question=instance.question,
        llm_client=llm_client,
        config=config,
    )
    if _is_multi_object_compound_query(
        instance.question,
        state=state,
        attributes=tuple(result.requested_attributes),
    ):
        result = QueryIntent(
            **{
                **result.__dict__,
                "target_subject": None,
                "uncertainty": tuple(dict.fromkeys((*result.uncertainty, "compound_named_objects"))),
            }
        )
    return result
