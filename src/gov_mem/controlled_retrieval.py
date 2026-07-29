"""Content retrieval constrained by an already-issued PolicyDecision."""

from __future__ import annotations

import re
from typing import Any

from gov_mem.backbones.common import RAGChunk
from gov_mem.data.schema import MemoryItem, RetrievedEvidence
from gov_mem.general_lexicon import GENERAL_VALUE_HEADS
from gov_mem.llm.client import LLMClient
from gov_mem.memory.dense_index import DenseMemoryIndex
from gov_mem.policy_schema import MemoryItemState, PolicyAction, PolicyDecision, PolicyState
from gov_mem.field_state_projection import (
    _is_aggregate_safe_summary_request,
    query_contract_from_dict,
)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _is_question_or_echo_source(content: str) -> bool:
    """Identify conversational questions that are not fact carriers.

    Shared-memory episodes often preserve a user's question alongside the
    later answer. Dense retrieval can prefer the question because it repeats
    the requested nouns and a proposed value. The question remains visible to
    policy-state construction, but it must not become answering evidence when
    a declarative source is available. This is a linguistic source-quality
    filter, not a dataset-value rule.
    """
    text = str(content or "").strip()
    if "?" not in text:
        return False
    first_clause = re.split(r"[?]", text, maxsplit=1)[0].strip().lower()
    return bool(re.search(
        r"\b(?:what|which|who|where|when|why|how|do|does|did|is|are|was|were|"
        r"can|could|should|would|will|may|shall|have|has|had|whether|confirm|"
        r"please\s+confirm|can\s+you|do\s+I|should\s+I)\b",
        first_clause,
        re.IGNORECASE,
    ))


def _source_kind(content: str) -> str:
    if _is_question_or_echo_source(content):
        return "question_or_echo"
    lowered = str(content or "").strip().lower()
    if re.search(r"\b(?:do not|don't|should not|must not|not entitled|only needs?)\b", lowered):
        return "boundary_or_instruction"
    if re.search(r"\b(?:i can|i could|i will|please move|then please|my .* may)\b", lowered):
        return "proposal_or_request"
    return "declarative_fact"


def _safe_partial_content(content: str, *, public_projection: bool, query: str) -> bool:
    """Reject mixed records whose text still carries a restricted field."""
    lowered_query = str(query or "").lower()
    lowered_content = str(content or "").lower()
    # A public projection is safe only at the requested disclosure granularity.
    # If the same record also carries the exact field being denied, hide the
    # whole mixed record; otherwise an answer model can re-bind the denied
    # value from a supposedly public sentence.
    restricted_surface_terms: list[str] = []
    if re.search(r"\bexact\b", lowered_query):
        if re.search(r"\b(?:room|bay|site|location|address|desk|hall|suite)\b", lowered_query):
            restricted_surface_terms.extend(["room", "bay", "site", "location", "address", "desk", "hall", "suite"])
        if re.search(r"\b(?:badge|code|credential|password|pin|token|key)\b", lowered_query):
            restricted_surface_terms.extend(["badge", "code", "credential", "password", "pin", "token", "key"])
        if re.search(r"\b(?:amount|budget|price|cost|fee|discount|payment)\b", lowered_query):
            restricted_surface_terms.extend(["amount", "budget", "price", "cost", "fee", "discount", "payment", "$", "€", "£"])
    if restricted_surface_terms and any(
        (
            (
                not (
                    term in {"room", "bay", "site", "location", "address", "desk", "hall", "suite"}
                    and re.search(rf"\bpublic\s+(?:{re.escape(term)})\b", lowered_content)
                    and not re.search(rf"\b(?:private|exact|restricted)\s+(?:{re.escape(term)})\b", lowered_content)
                )
                and (re.search(rf"\b{re.escape(term)}\b", lowered_content) if term.isalpha() else term in lowered_content)
            )
        )
        for term in restricted_surface_terms
    ):
        return False
    if public_projection:
        if re.search(r"\bexact\b", lowered_query) and re.search(
            r"\b(?:room|bay|badge|amount|code|pin|customer|token)\b",
            lowered_query,
        ):
            return not bool(re.search(
                r"\b(?:room|bay|badge|amount|code|pin|customer|token)\b",
                str(content or "").lower(),
            ))
        return True
    # A safe-summary boundary may mention that an exact/private value is
    # restricted without carrying that value itself. Preserve the boundary
    # sentence as a safe carrier; quoted exact wording or concrete sensitive
    # values remain excluded by the checks above.
    if re.search(r"\b(?:safe[- ]summary|safe\s+label|only\s+as|may\s+refer\s+to)\b", lowered_content):
        if not re.search(r"(['\"])[^'\"]{2,}\b(?:exact|private|restricted)\b[^'\"]{0,}\1", lowered_content):
            return True
    lowered = lowered_content
    return not bool(re.search(
        r"\b(?:private|exact|restricted|resident-only|credential|badge|pin|"
        r"password|diagnosis|medical|clinical|health|condition|disease|"
        r"amount|access code|active code|helper phrase|customer identity)\b",
        lowered,
    ))


def _memory_items(
    state: PolicyState,
    allowed: set[str],
    *,
    query: str = "",
    partial_disclosure: bool = False,
) -> list[MemoryItem]:
    result: list[MemoryItem] = []
    for item in state.memory_items:
        if item.memory_id not in allowed:
            continue
        content = item.provenance.evidence_text
        # The query is not evidence.  Some episodes also contain a prior
        # question or a policy-boundary prompt that repeats sensitive terms;
        # never let that echo become answer context on a partial-disclosure
        # path.
        if query and _normalize_text(content) == _normalize_text(query):
            continue
        # A prior question/confirmation echo is observable conversation, but
        # it is not an answer-bearing fact. Keep it in PolicyState provenance
        # and exclude it from the content channel.
        if _is_question_or_echo_source(content):
            continue
        if partial_disclosure and "?" in content:
            continue
        if partial_disclosure and not _safe_partial_content(
            content,
            public_projection="public_projection" in set(item.topics),
            query=query,
        ):
            continue
        result.append(
            MemoryItem(
                memory_id=item.memory_id,
                instance_id="stateful_policy",
                user_id=item.owner,
                scope=item.scope or "observable",
                content=content,
                memory_type="policy_allowed_memory",
                entities=[item.subject] if item.subject else [],
                time=item.created_at,
                source_message_ids=list(item.provenance.source_message_ids),
                confidence=1.0,
                privacy_level=None,
                tags=["policy_allowed"],
                memory_status=item.status.value,
                metadata={
                    "content_ref": item.content_ref,
                    "policy_scope": item.scope,
                    "policy_topics": list(item.topics),
                    "source_turn_index": item.provenance.turn_index,
                    "source_kind": _source_kind(content),
                },
            )
        )
    return result


def _projection_candidates(items: list[MemoryItem], query: str, *, limit: int = 8) -> list[MemoryItem]:
    """Keep authorized broad/public projections visible to safe-summary answers.

    Dense retrieval is useful for ordinary relevance, but a short projection
    such as ``the bank pilot`` or ``international exchange review`` can have
    almost no lexical overlap with ``generic customer description`` or ``safe
    wording``. These records are already policy-allowed; this is a recall
    safeguard inside the allowed set, not an authorization rule.
    """
    query_tokens = {
        token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", query.lower())
        if token not in {"what", "what", "current", "safe", "summary", "wording", "generic", "exact"}
    }
    # Safe-summary requests need the latest explicit recap even when the
    # record is not tagged with a broad/public scope.  These are recall hints
    # inside the already authorized set; they do not expand the policy set.
    summary_cue = re.compile(
        r"\b(?:summary|summaries|recap|broad(?:\s+wording|\s+statement)?|"
        r"safe\s+(?:wording|summary)|helper(?:s)?|no\s+exact|final\s+state|"
        r"current\s+(?:state|summary|anchor))\b",
        re.IGNORECASE,
    )
    # A current operational carrier is often a short declarative update and
    # does not call itself a summary ("Current arrival window is ...").
    # Safe-summary retrieval must retain those records; otherwise only
    # boundary prose and unrelated public summaries survive the top-k cut.
    current_carrier_cue = re.compile(
        r"\b(?:current|latest|active|approved|updated)\b[^.!?;]{0,100}"
        r"\b(?:window|arrival|entry|method|route|zone|setup|schedule|"
        r"handling|release|scope|state)\b",
        re.IGNORECASE,
    )
    candidates = [
        item for item in items
        if item.scope in {"public", "broad", "safe_summary"}
        or "public_projection" in set(item.metadata.get("policy_topics") or ())
        or summary_cue.search(item.content or "")
        or current_carrier_cue.search(item.content or "")
    ]

    def rank(item: MemoryItem) -> tuple[float, str, str]:
        tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", item.content.lower()))
        overlap = len(query_tokens & tokens) / max(1, len(query_tokens))
        # Prefer actual projections and explicit summaries, then preserve
        # deterministic source ordering for equal scores.  Recency is a
        # tiebreaker only; it cannot override lexical support.
        projection = 1.0 if "public_projection" in set(item.metadata.get("policy_topics") or ()) else 0.0
        cue = 0.5 if summary_cue.search(item.content or "") else 0.0
        current = 0.4 if current_carrier_cue.search(item.content or "") else 0.0
        return (projection + cue + current + overlap, item.time or "", item.memory_id)

    chosen: list[MemoryItem] = []
    seen_content: set[str] = set()
    # Explicit safe-summary carriers are easy to drown out when an episode
    # repeats many public logistics notes. Reserve a small recent frontier
    # for those declarative summaries, then fill the remaining slots by the
    # normal relevance ranking. This changes recall only inside the already
    # policy-approved set.
    summary_candidates = sorted(
        (item for item in candidates if summary_cue.search(item.content or "")),
        key=lambda item: (
            item.metadata.get("source_turn_index", -1)
            if isinstance(item.metadata.get("source_turn_index", -1), int) else -1,
            item.time or "",
            item.memory_id,
        ),
        reverse=True,
    )[: min(4, limit)]
    current_candidates = sorted(
        (item for item in candidates if current_carrier_cue.search(item.content or "")),
        key=lambda item: (
            item.metadata.get("source_turn_index", -1)
            if isinstance(item.metadata.get("source_turn_index", -1), int) else -1,
            item.time or "",
            item.memory_id,
        ),
        reverse=True,
    )[: min(16, limit)]
    ranked_candidates = [*current_candidates, *summary_candidates, *sorted(candidates, key=rank, reverse=True)]
    for item in ranked_candidates:
        content_key = re.sub(r"\s+", " ", item.content.strip().lower())
        if not content_key or content_key in seen_content:
            continue
        seen_content.add(content_key)
        chosen.append(item)
        if len(chosen) >= limit:
            break
    return chosen


def _operational_snapshot_candidates(
    items: list[MemoryItem],
    *,
    query: str,
    requested_attributes: list[str],
) -> list[MemoryItem]:
    """Recall current non-sensitive label/color carriers for snapshots."""
    surface = " ".join([str(query or ""), *(str(value) for value in requested_attributes)])
    if not re.search(r"\b(?:calibration|tag|label|color)\b", surface, re.IGNORECASE):
        return []
    pattern = re.compile(
        r"\b(?:current(?:ly)?|active|latest)\b[^.!?;]{0,50}"
        r"\b(?:tag|label)\s+color\b",
        re.IGNORECASE,
    )
    return [
        item for item in items
        if item.scope in {"public", "broad", "safe_summary"}
        and pattern.search(str(item.content or ""))
    ]


def _lexical_field_candidates(
    items: list[MemoryItem],
    *,
    query: str,
    requested_attributes: list[str],
    limit: int,
) -> list[MemoryItem]:
    """Recover direct field evidence inside the already authorized set.

    This is a content-recall complement to dense retrieval, never an access
    rule.  It is deliberately field-agnostic: it scores words from the user
    query/checklist against source text and does not contain dataset values or
    answer templates.
    """
    terms = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", query.lower()))
    for attribute in requested_attributes:
        terms.update(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", attribute.lower()))
    stop = {"current", "latest", "approved", "private", "public", "exact", "what", "are", "the", "and", "for", "with", "room", "amount"}
    terms.difference_update(stop)
    scored: list[tuple[float, str, MemoryItem]] = []
    for item in items:
        text = str(item.content or "").lower()
        tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text))
        overlap = len(terms.intersection(tokens))
        if not overlap:
            continue
        field_phrase = max(
            (len(set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", attribute.lower())).intersection(tokens)) for attribute in requested_attributes),
            default=0,
        )
        scored.append((float(overlap) + 0.5 * float(field_phrase), item.time or "", item))
    scored.sort(key=lambda row: (row[0], row[1], row[2].memory_id), reverse=True)
    return [item for _, _, item in scored[:max(1, limit)]]


def _latest_field_candidates(
    items: list[MemoryItem],
    *,
    requested_attributes: list[str],
    target_subject: str | None = None,
    query: str = "",
) -> list[MemoryItem]:
    """Select the latest direct source for each requested field.

    This remains inside the policy-approved set and only controls content
    recall.  It prevents a long episode from presenting ten conflicting
    values all labelled ``current`` to the answer model when a later source
    explicitly carries the same field forward.
    """
    stop = {
        "current", "latest", "active", "approved", "exact", "what", "are", "the", "and",
        "for", "with", "my", "our", "their", "status", "details", "information",
        "private", "public", "broad", "safe", "summary", "only",
    }
    # Current-state retrieval is a temporal lane, not a similarity contest.
    # Once a source is bound to the requested field and does not carry an
    # explicit competing time/object anchor, a later effective source must
    # outrank an older source with more repeated query words.  Otherwise a
    # continuation such as "the exact note remains ..." loses to an older
    # sentence that repeats the weekday and object noun.
    chosen: dict[str, tuple[int, int, MemoryItem]] = {}

    target_tokens = {
        token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", str(target_subject or "").lower())
        if token not in {"current", "latest", "summary", "snapshot", "record", "state"}
    }

    def explicit_time_anchor_mismatch(item: MemoryItem) -> bool:
        """Keep a source in the same explicit time lane as the request.

        Conversation updates often omit the object name, so recency alone is
        not enough to bind a field.  A directly stated weekday/month anchor is
        stronger evidence of lane identity and is safe to use as a retrieval
        constraint inside the already authorized set.
        """
        query_text = str(query or "").lower()
        source_text = str(item.content or "").lower()
        weekdays = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        query_anchors = weekdays.intersection(re.findall(r"[a-z]+", query_text))
        source_anchors = weekdays.intersection(re.findall(r"[a-z]+", source_text))
        return bool(query_anchors and source_anchors and query_anchors.isdisjoint(source_anchors))

    def explicitly_other_subject(item: MemoryItem) -> bool:
        """Reject a source explicitly owned by a different named object.

        Conversation updates commonly omit the object name because the
        preceding turns already established the thread. Such an item remains
        eligible and is ordered by its later source turn. Only an explicit
        competing entity is disqualifying here.
        """
        if not target_tokens:
            return False
        labels = [str(value).strip() for value in item.entities if str(value).strip()]
        if not labels:
            subject = str(item.metadata.get("subject") or "").strip()
            if subject:
                labels.append(subject)
        for label in labels:
            label_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", label.lower()))
            # A label with no target token may be a shorthand continuation;
            # only a sibling label sharing part of the target identity is an
            # explicit competing object (Laurel Locker vs Laurel Lift).
            if (
                label_tokens
                and target_tokens.intersection(label_tokens)
                and not target_tokens.issubset(label_tokens)
            ):
                return True
        # Some memory builders expose only a generic subject such as
        # ``For Sunday``. Recover an explicitly named competing object from
        # the observable source text when it is present.
        for label in re.findall(
            r"\b[A-Z][A-Za-z0-9&'-]+(?:\s+[A-Z][A-Za-z0-9&'-]+){1,4}\b",
            str(item.content or ""),
        ):
            label_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", label.lower()))
            # Proper-noun extraction also sees qualified values such as
            # ``Hollow Studio Bay``.  Those are field values, not sibling
            # objects, and must not make a legitimate ``Hollow Capstone``
            # source ineligible.  Subject metadata remains the preferred
            # competing-object signal; this fallback ignores common value
            # heads when inspecting raw prose.
            if (
                label_tokens
                and target_tokens.intersection(label_tokens)
                and not target_tokens.issubset(label_tokens)
                and not label_tokens.intersection(GENERAL_VALUE_HEADS)
            ):
                return True
        return False

    def token_matches(term: str, tokens: set[str]) -> bool:
        """Match ordinary inflectional variants without a word list."""
        if term in tokens:
            return True
        if term.endswith("s") and term[:-1] in tokens:
            return True
        return f"{term}s" in tokens

    for attribute in requested_attributes:
        terms = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", attribute.lower())) - stop
        if not terms:
            continue
        # If the requested label includes the already grounded object, remove
        # that object from the value-bearing requirement.  For example,
        # ``blocker for Hollow Capstone`` must match a blocker statement, not
        # any later statement that merely repeats ``Hollow Capstone``.
        target_terms = set(target_tokens)
        field_terms = terms - target_terms
        if not field_terms:
            field_terms = terms
        for item in items:
            if explicitly_other_subject(item):
                continue
            if explicit_time_anchor_mismatch(item):
                continue
            text = str(item.content or "")
            # A later coordination note can repeat the requested noun while
            # explicitly saying that it changes nothing.  It is useful policy
            # provenance, but it is not a value carrier and must not replace a
            # prior explicit budget/date/contract record during latest-field
            # selection.  Keep this semantic and value-agnostic: a note with
            # an actual number or an explicit ``is/are`` assignment remains
            # eligible.
            non_value_boundary = bool(re.search(
                r"\b(?:nothing|no\s+part|no\s+detail|without)\b"
                r"[^.!?;]{0,100}\b(?:change|changes|changed|alter|altered|"
                r"update|updated|supersede|superseded)\b"
                r"|\b(?:does|do)\s+not\s+(?:change|alter|update|supersede)\b",
                text,
                re.IGNORECASE,
            )) and not bool(re.search(r"\d", text))
            non_value_boundary = non_value_boundary or bool(re.match(
                r"^\s*(?:no\s+change|unchanged|same|still\s+the\s+same)\b",
                text,
                re.IGNORECASE,
            )) and not bool(re.search(r"\d", text))
            if non_value_boundary:
                continue
            # Disclosure/retrieval instructions are policy evidence, not
            # carriers of the requested value. Excluding them from the
            # latest-value lane prevents a late housekeeping sentence from
            # outranking the concrete record it describes.
            if re.search(
                r"\b(?:should\s+not|must\s+not|do\s+not|don't|not\s+receive|"
                r"not\s+entitled|housekeeping\s+note|retrieval\s+note)\b",
                text,
                re.IGNORECASE,
            ):
                continue
            # Policy-boundary sentences can mention a requested field without
            # supplying its value. They are useful for governance reasoning,
            # but should not outrank a direct factual source during recall.
            if re.search(
                r"\b(?:do not|don't|should not|must not|not need|no need|not entitled)\b",
                text,
                re.IGNORECASE,
            ) and not re.search(
                r"\b(?:current|new|updated|changed)\b|\b(?:is|are)\s+(?!not\b)|\bfrom\b.{0,80}\bto\b|\d",
                text,
                re.IGNORECASE,
            ):
                continue
            tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower()))
            # Some memory records use a compact value-only surface (for
            # example ``LH-318``) while their governed scope/topics carry the
            # field head. Keep these records eligible inside the policy
            # approved set; this is semantic carrier recall, not permission.
            scope_tokens = set(re.findall(
                r"[a-z0-9][a-z0-9_-]{2,}",
                " ".join([
                    str(item.scope or ""),
                    *(str(topic) for topic in item.metadata.get("policy_topics") or ()),
                ]).lower(),
            ))
            core_matches = {
                term for term in field_terms
                if token_matches(term, tokens) or token_matches(term, scope_tokens)
            }
            calendar_anchor = bool(
                field_terms.intersection({"date", "day"})
                and re.search(
                    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+"
                    r"(?:january|february|march|april|may|june|july|august|september|october|"
                    r"november|december)\s+\d{1,2}\b"
                    r"|\b(?:january|february|march|april|may|june|july|august|september|october|"
                    r"november|december)\s+\d{1,2}(?:,\s*\d{4})?\b",
                    text,
                    re.IGNORECASE,
                )
            )
            # Qualifiers and object names identify a lane but do not carry a
            # value. Require at least one field-head match so a badge cannot
            # satisfy a bay field, and a release note cannot satisfy blocker.
            if not core_matches and not calendar_anchor:
                continue
            overlap = len({term for term in terms if token_matches(term, tokens)})
            # Calendar anchors are often expressed as ``Saturday, September
            # 19`` rather than with the word "date". Treat a source date as
            # evidence for a requested date field without inventing a value.
            if not overlap and calendar_anchor:
                overlap = 1
            if not overlap:
                continue
            turn = item.metadata.get("source_turn_index")
            turn_index = int(turn) if isinstance(turn, int) else -1
            key = str(attribute).lower()
            candidate = (turn_index, overlap, item)
            previous = chosen.get(key)
            if previous is None or (candidate[0], candidate[1]) > (previous[0], previous[1]):
                chosen[key] = candidate
    unique: dict[str, MemoryItem] = {}
    for _, _, item in sorted(chosen.values(), key=lambda row: (row[0], row[2].memory_id), reverse=True):
        unique.setdefault(item.memory_id, item)
    return list(unique.values())


def _is_aggregate_requested_attribute(value: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if lowered in {
        "all requested fields", "everything", "details", "the details", "the plan",
        "current plan", "snapshot", "current snapshot", "calibration snapshot",
        "current calibration snapshot", "overview", "current overview",
    }:
        return True
    # Collection nouns can occur before or after their qualifiers, e.g.
    # "final ... snapshot ... current state". Treating them as scalar fields
    # activates latest-value compaction and drops the other members.
    # ``window`` is intentionally not in this collection list.  In natural
    # language, ``setup window`` and ``helper window`` are scalar requested
    # fields (usually one time range), while ``plan`` or ``snapshot`` denotes
    # an aggregate request. Treating every window as aggregate disables the
    # field-level latest-source pass and can drop the current time range.
    return bool(re.search(
        r"\b(?:plan|summary|recap|details|information|snapshot|overview|"
        r"calibration|order|routing|status|state|instructions?|arrangement|"
        r"schedule)\b",
        lowered,
    ))


def retrieve_allowed_memory(
    *,
    state: PolicyState,
    decision: PolicyDecision,
    query: str,
    embedding_client: LLMClient | None,
    embedding_model: str,
    top_k: int = 12,
    retrieval_strategy: str = "global_authorized_topk",
) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    """Only memory IDs explicitly allowed by policy reach the index/prompt."""
    if decision.action != PolicyAction.ALLOW or not decision.allowed_memory_ids:
        return [], {
            "policy_filtered": True,
            "allowed_memory_ids": list(decision.allowed_memory_ids),
            "blocked_memory_ids": list(decision.blocked_memory_ids),
            "selected_memory_ids": [],
        }
    allowed = set(decision.allowed_memory_ids)
    items = _memory_items(
        state,
        allowed,
        query=query,
        partial_disclosure=bool(decision.state_snapshot.get("partial_disclosure")),
    )
    if not items:
        return [], {"policy_filtered": True, "allowed_memory_ids": [], "blocked_memory_ids": list(decision.blocked_memory_ids), "selected_memory_ids": []}
    query_contract = query_contract_from_dict(decision.state_snapshot.get("query_contract"))
    requested_attributes = [
        field.label
        for field in (query_contract.fields if query_contract is not None else ())
        if str(field.label).strip()
    ]
    if not requested_attributes:
        # Compatibility for pre-contract decisions and hand-built unit-test
        # decisions.  New production runs always use the policy-owned
        # structured contract above.
        requested_attributes = [
            str(value).strip()
            for value in decision.state_snapshot.get("requested_attributes") or []
            if str(value).strip()
            and str(value).strip().lower() not in {"all requested fields", "everything", "details"}
        ]
    target_subject = str(decision.target_subject or "").strip() or None
    # A request may be classified as an ordinary logistics request by the
    # policy layer while the field contract correctly identifies its delivery
    # surface as a safe summary (for example, "send the current logistics
    # summary" ).  Keep the public/current carrier recall path enabled for the
    # latter as well; otherwise the aggregate projector receives only the
    # dense top-k slice and cannot recover the current state carrier.
    safe_summary_contract = bool(
        query_contract is not None
        and any(field.disclosure_scope == "safe_summary" for field in query_contract.fields)
    ) or _is_aggregate_safe_summary_request(query)
    # A helper/summary query often omits the requester's name even though the
    # source records use it as the only stable lane anchor (for example,
    # ``Carmen porch-only ...``).  Add the already-resolved principal identity
    # as a retrieval hint.  This stays inside the authorized set and changes
    # recall only; policy authorization and field binding remain downstream.
    requester_identity: list[str] = []
    if decision.requester:
        requester_id = str(decision.requester).strip()
        for principal in state.principals:
            if str(principal.principal_id).strip() != requester_id:
                continue
            requester_identity.extend(
                value
                for value in (
                    principal.display_name,
                    principal.principal_id,
                    *principal.aliases,
                )
                if str(value or "").strip()
            )
            break
        if not requester_identity:
            requester_identity.append(requester_id)
    requester_hint = " ".join(dict.fromkeys(str(value).strip() for value in requester_identity if str(value).strip()))
    helper_lane_request = bool(re.search(
        r"\b(?:helper|helpers|allowed|logistics[- ]only|signoff[- ]safe|coordination)\b",
        " ".join([query, *requested_attributes]),
        re.IGNORECASE,
    ))
    retrieval_query = (
        f"{query}\nResolved requester identity: {requester_hint}"
        if requester_hint and helper_lane_request
        else query
    )
    retrieval_fields: dict[str, set[str]] = {}
    use_field_local_retrieval = str(retrieval_strategy or "").strip().lower() in {
        "field_local",
        "per_field",
        "per_field_union",
    }
    if embedding_client is not None:
        index = DenseMemoryIndex.build(items=items, llm_client=embedding_client, embedding_model=embedding_model)
        # The stable path creates one policy-filtered candidate set. A field
        # local union is available for controlled experiments, but it is not
        # the default because it introduces multiple competing retrieval and
        # binding authorities before answer projection.
        field_budget = min(len(items), max(3, min(8, top_k))) if use_field_local_retrieval else max(top_k, 1)
        rows: list[tuple[str, float]] = []
        query_views = [("__query__", retrieval_query)]
        if target_subject and use_field_local_retrieval:
            query_views.append(("__subject__", f"Target object: {target_subject}\n{retrieval_query}"))
        if use_field_local_retrieval:
            for field_index, attribute in enumerate(requested_attributes):
                field_id = (
                    query_contract.fields[field_index].field_id
                    if query_contract is not None and field_index < len(query_contract.fields)
                    else str(attribute)
                )
                query_views.append((field_id, f"{retrieval_query}\nRequested field: {attribute}"))
        for field_id, query_view in query_views:
            field_rows = index.query(
                query_texts=[query_view],
                top_k=field_budget,
                llm_client=embedding_client,
                embedding_model=embedding_model,
            )
            for memory_id, score in field_rows:
                rows.append((memory_id, score))
                retrieval_fields.setdefault(memory_id, set()).add(field_id)
    else:
        rows = [(item.memory_id, 0.0) for item in items[:max(top_k, 1)]]
    by_id = {item.memory_id: item for item in items}
    if (
        decision.state_snapshot.get("target_scope") == "safe_summary"
        or decision.state_snapshot.get("partial_disclosure")
        or safe_summary_contract
    ):
        # Add only records that are already in the policy-approved set. This
        # makes broad projections recallable without exposing blocked content
        # or changing the Stage 2 decision.
        projection_rows = _projection_candidates(items, query, limit=max(24, top_k))
        existing_ids = {memory_id for memory_id, _ in rows}
        rows.extend(
            (item.memory_id, 0.0)
            for item in projection_rows
            if item.memory_id not in existing_ids
        )
        for item in projection_rows:
            retrieval_fields.setdefault(item.memory_id, set()).add("safe_summary")
        if decision.state_snapshot.get("partial_disclosure"):
            # Partial answers need the complete safe carrier frontier, not
            # just the top few public rows.  These rows have already passed
            # policy filtering and the partial-content guard above; adding
            # them cannot expose a blocked/private record.
            safe_rows = [
                item for item in items
                if item.scope in {"public", "broad", "safe_summary"}
                or "public_projection" in set(item.metadata.get("policy_topics") or ())
            ]
            for item in safe_rows:
                if item.memory_id not in existing_ids:
                    rows.append((item.memory_id, 0.0))
                    existing_ids.add(item.memory_id)
                retrieval_fields.setdefault(item.memory_id, set()).add("safe_carrier_frontier")
    aggregate_requested = any(
        _is_aggregate_requested_attribute(attribute)
        for attribute in requested_attributes
    )
    if aggregate_requested:
        operational_rows = _operational_snapshot_candidates(
            items,
            query=query,
            requested_attributes=requested_attributes,
        )
        existing_ids = {memory_id for memory_id, _ in rows}
        rows.extend(
            (item.memory_id, 0.0)
            for item in operational_rows
            if item.memory_id not in existing_ids
        )
        for item in operational_rows:
            retrieval_fields.setdefault(item.memory_id, set()).add("operational_snapshot")
    # The baseline lexical fallback is also global. The per-field variant is
    # retained behind an explicit strategy flag for later ablations.
    lexical_candidates: list[MemoryItem] = []
    if use_field_local_retrieval:
        lexical_requests = [
            (
                query_contract.fields[field_index].field_id
                if query_contract is not None and field_index < len(query_contract.fields)
                else str(attribute),
                [attribute],
                max(3, min(8, top_k)),
            )
            for field_index, attribute in enumerate(requested_attributes)
        ]
    else:
        lexical_requests = [("lexical_global_recall", requested_attributes, max(8, min(24, top_k * 2)))]
    for field_id, attributes, limit in lexical_requests:
        field_rows = _lexical_field_candidates(items, query=retrieval_query, requested_attributes=attributes, limit=limit)
        existing_ids = {memory_id for memory_id, _ in rows}
        rows.extend(
            (item.memory_id, 0.0)
            for item in field_rows
            if item.memory_id not in existing_ids
        )
        for item in field_rows:
            lexical_candidates.append(item)
            retrieval_fields.setdefault(item.memory_id, set()).add(field_id)
    # Preserve the broad lexical signal for compatibility when no structured
    # contract exists.
    if not requested_attributes and use_field_local_retrieval:
        lexical_candidates = _lexical_field_candidates(
            items, query=retrieval_query, requested_attributes=[], limit=max(8, min(24, top_k * 2))
        )
        existing_ids = {memory_id for memory_id, _ in rows}
        rows.extend((item.memory_id, 0.0) for item in lexical_candidates if item.memory_id not in existing_ids)
        for item in lexical_candidates:
            retrieval_fields.setdefault(item.memory_id, set()).add("lexical_field_recall")
    latest_field_carriers: list[MemoryItem] = []
    concrete_attributes: list[str] = []
    if decision.state_snapshot.get("requested_attributes") and re.search(
        r"\b(?:current|latest|active|now|as\s+of\s+now|still|remains?)\b", query.lower()
    ):
        concrete_attributes = [
            item for item in requested_attributes
            if not _is_aggregate_requested_attribute(item)
        ]
        latest_fields = _latest_field_candidates(
            items,
            requested_attributes=concrete_attributes,
            target_subject=target_subject,
            query=retrieval_query,
        )
        latest_field_carriers = latest_fields
        # For one scalar current field, the latest carrier is the complete
        # field-local answer context.  For compound requests, latest-source
        # recall is augmentation only: replacing the union here would let one
        # field evict a lower-similarity carrier for another field.
        if len(concrete_attributes) == 1 and latest_fields:
            rows = [(item.memory_id, 0.0) for item in latest_fields]
        else:
            existing_ids = {memory_id for memory_id, _ in rows}
            rows.extend(
                (item.memory_id, 0.0)
                for item in latest_fields
                if item.memory_id not in existing_ids
            )
        for item in latest_fields:
            retrieval_fields.setdefault(item.memory_id, set()).add("latest_field_recall")
    # A continuation turn often omits the object and field nouns entirely
    # (for example, "Wednesday November 25 at 8:00 AM" after a repeat-lab
    # question). For temporal/current requests, retain a small tail of the
    # requester-owned authorized records so source lineage can recover the
    # omitted qualifiers. This remains policy-bounded and field closure still
    # decides which records carry the requested fact.
    if re.search(
        r"\b(?:repeat|before|after|next|upcoming|due)\b",
        query.lower(),
    ):
        owner_tail = sorted(
            (
                item for item in items
                if item.user_id == decision.requester
                and state.memory_status.get(item.memory_id, item.memory_status).value == "active"
            ),
            key=lambda item: (
                item.metadata.get("source_turn_index", -1)
                if isinstance(item.metadata.get("source_turn_index", -1), int) else -1,
                item.memory_id,
            ),
            reverse=True,
        )[: max(4, min(8, top_k))]
        existing_ids = {memory_id for memory_id, _ in rows}
        rows.extend(
            (item.memory_id, 0.0)
            for item in owner_tail
            if item.memory_id not in existing_ids
        )
        for item in owner_tail:
            retrieval_fields.setdefault(item.memory_id, set()).add("requester_owned_temporal_tail")
    # Multiple semantic query views can return the same authorized memory more
    # than once. Deduplicate before constructing the answer context so a
    # repeated boundary note cannot crowd out a distinct field carrier.
    deduped_rows: dict[str, float] = {}
    for memory_id, score in rows:
        if memory_id not in allowed:
            continue
        deduped_rows[memory_id] = max(float(score), deduped_rows.get(memory_id, float("-inf")))
    rows = list(deduped_rows.items())
    evidence = [
        RetrievedEvidence(
            memory_id=memory_id,
            content=by_id[memory_id].content,
            score=float(score),
            retrieval_source=(
                "policy_allowed_projection"
                if "public_projection" in set(by_id[memory_id].metadata.get("policy_topics") or ())
                else "policy_allowed_dense"
            ),
            reason=(
                "authorized broad/public projection recall"
                if "public_projection" in set(by_id[memory_id].metadata.get("policy_topics") or ())
                else "content relevance within allowed memory set"
            ),
            user_id=by_id[memory_id].user_id,
            scope=by_id[memory_id].scope,
            entities=list(by_id[memory_id].entities),
            memory_type=by_id[memory_id].memory_type,
            source_message_ids=by_id[memory_id].source_message_ids,
            time=by_id[memory_id].time,
            metadata={
                "policy_allowed": True,
                "policy_scope": by_id[memory_id].metadata.get("policy_scope"),
                "subject": (by_id[memory_id].entities[0] if by_id[memory_id].entities else None),
                "policy_topics": list(by_id[memory_id].metadata.get("policy_topics") or ()),
                "source_turn_index": by_id[memory_id].metadata.get("source_turn_index"),
                "disclosure_role": (
                    "PUBLIC_PROJECTION"
                    if "public_projection" in set(by_id[memory_id].metadata.get("policy_topics") or ())
                    else "POLICY_ALLOWED_EVIDENCE"
                ),
                "retrieval_fields": sorted(retrieval_fields.get(memory_id, set())),
            },
        )
        for memory_id, score in rows
        if memory_id in allowed and memory_id in by_id
    ]
    return evidence, {
        "policy_filtered": True,
        "allowed_memory_ids": sorted(allowed),
        "blocked_memory_ids": list(decision.blocked_memory_ids),
        "selected_memory_ids": [row.memory_id for row in evidence],
        "retrieval_mode": (
            "per_field_controlled_union"
            if use_field_local_retrieval
            else "global_authorized_topk"
        ),
        "retrieval_fields": {
            memory_id: sorted(values)
            for memory_id, values in retrieval_fields.items()
            if memory_id in {row.memory_id for row in evidence}
        },
    }
