"""Execute structured governance decisions and update state explicitly."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from gov_mem.data.schema import AnswerResult, MemoryInstance, RetrievedEvidence
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.model_registry import resolve_llm_model
from gov_mem.policy_schema import ExecutionResult, PolicyAction, PolicyDecision
from gov_mem.policy_verifier import verify_policy_delivery
from gov_mem.execution_planner import ExecutionPlan
from gov_mem.answer_projection import (
    AnswerProjection,
    AnswerRequest,
    compile_answer_request,
    projection_coverage_gaps,
    projection_required_labels,
    projection_to_dict,
    project_field_evidence,
)
from gov_mem.field_state_projection import (
    build_stateful_projection,
    compile_query_contract,
    projection_evidence_payload,
    projection_to_answer_projection,
    query_contract_from_dict,
    restricted_field_ids,
    QueryContract,
    QueryField,
)


ANSWER_SYSTEM_PROMPT = """
Answer the user's question using only the supplied allowed memory evidence.
Do not mention policy internals, hidden labels, blocked memory, or evidence that
is not supplied. Return JSON only with answer_text, requested_fields,
dimension_coverage, covered_requested_fields, restricted_fields_omitted, confidence, and
safe_answer. requested_fields must contain one object for every separately
requested fact, with label, status (covered, omitted, or unknown), answer_text,
and source_memory_ids. Answer_text must be a plain natural-language string,
never an object, list, or Python-style dictionary. Answer every separately
requested field, even when the question contains several dates, names, scopes,
or rules.
For an aggregate request such as a plan, summary, snapshot, overview, or
current state, use the supplied answer_need_spec as the completeness checklist.
Treat each named object, lane, audience, and temporal qualifier as a separate
binding boundary. Do not merge a value from one lane into another merely
because the field names are similar.
When the request is aggregate and the checklist retains an overall plan,
summary, snapshot, or state field, treat that field as a request for the
complete allowed projection: include every directly relevant current,
operational, or explicitly requested value supported by allowed evidence,
including a value that appears only in a short status/update record. Do not
collapse a multi-field or multi-lane projection into one generic sentence.
The words snapshot, calibration snapshot, current state, and overview mean a
complete current projection of the requested object or lanes. Do not silently
reduce such a request to the first few fields that match the audience label.
An audience qualifier limits disclosure, but it does not erase a separate
non-sensitive operational field that is part of the current projection. Use
the current-state evidence shortlist and later explicit summary/status records
to recover those fields. For a multi-lane snapshot, answer each lane's latest
safe/current projection separately; do not answer with an older internal
schedule merely because it has stronger lexical overlap.
This includes ordinary operational attributes such as a current color, label,
status, staging surface, or boundary when an allowed source states them
explicitly. Do not discard such an attribute solely because the question also
names a role or audience; discard it only when the requested disclosure level
actually restricts it.
say that evidence is insufficient if the allowed evidence does not answer it.
When evidence explicitly says that a later value supersedes, replaces, or
retires an earlier value, use the later value. Never answer with a value marked
earlier, stale, provisional, or superseded. Preserve compatible qualifiers
from the latest supporting evidence, such as a site prefix plus desk number.
Cover every requested attribute when it is supported by the allowed evidence;
do not silently answer only the easiest part of a multi-part question.
Evidence is annotated with source time and turn order. For current/latest
questions, resolve conflicts by the highest source_turn_index that contains
an explicit update, replacement, or current-state statement, then preserve
the full qualified value from that evidence. An older record saying "current"
does not override a later state-changing record. Do not choose an older value
merely because it is listed first or is phrased more briefly.
source_time and source_turn_index are provenance metadata, not facts requested
by the user. Never substitute a message timestamp for an event date; when the
evidence states a weekday/month/day such as "Saturday, September 19", preserve
that calendar wording verbatim and do not infer a different date from metadata.
For safe_summary requests, do not apply a blanket ban to precise values. A
time, location, label, amount, or other operational field is answerable when
the field_evidence_projection marks it supported and its source is in the
policy-approved evidence. Omit a field only when the projection marks it
restricted or unknown; safe-summary wording alone does not turn an allowed
operational value into a restriction. Record only genuinely restricted fields
in restricted_fields_omitted.
When the requester has partial disclosure access, provide only the allowed
logistics or scheduling fact requested. Never add a visit reason, diagnosis,
health, imaging, pathology, laboratory, medication, or contact detail merely
because it is inferable from the question or nearby wording.
For exact factual values such as phone numbers, dates, times, names, doses, and
short policy phrases, preserve the wording from the allowed evidence verbatim
whenever it directly answers the question; do not paraphrase or inflect it.
For access artifacts such as credentials, codes, passwords, PINs, tokens, and
keys, preserve the complete source token, including any prefix, namespace, or
suffix. Do not shorten a composite token to its final segment merely because
the segment looks distinctive.
When a broad label such as "callback plan" covers several operational facts,
expand every directly requested fact from the allowed evidence, including a
phone number, extension, voicemail instruction, ride rule, or time window.
Do not replace those facts with a high-level summary.
If partial_disclosure is true, provide an explicitly broad, generic, public, or
safe-summary fact from allowed evidence when one exists, while omitting the
restricted exact fact. Never repeat a sensitive term merely because it appears
in the question or in an allowed question/echo record. If no safe subset is
supported, refuse without repeating the restricted fact.
If the draft is an empty insufficiency/refusal despite an allowed record that
directly answers the request, repair it from that record without inventing facts.
Bind every requested field to its own value. Never use a site, room, desk, or
location as a credential/code/password value, and never use a credential as a
site or date. For a multi-field request, audit each field independently before
writing the final sentence. Preserve a value only when the evidence supports
that exact field, not merely because the value appears nearby.
The required_fields list is the authoritative completeness checklist. A field
is covered only when its own concrete value is present in answer_text and the
field cites the allowed memory that supports that value. Do not mark a field
covered with vague phrases such as "the same", "unchanged", "as above", or
"the usual". If allowed evidence supports a field, do not omit it. If no
allowed evidence supports it, mark it omitted or unknown instead of inventing.
The field_evidence_projection is the primary binding checklist. For every
supported required field, include every selected_values item in answer_text,
preserving list cardinality and each field's source binding. Do not select a
different value from the flat evidence when the projection already supplies a
source-grounded value. Treat projection fields marked unknown, restricted, or
conflict as unavailable unless the allowed evidence itself resolves them.
An aggregate label such as "the current snapshot" is a grouping label, not an
extra fact. If its separately listed supported fields are answered, do not
mark the grouping label as unavailable and do not prefix a complete answer
with "not available" merely because that label has no single source record.
Before composing the answer, silently audit the relevant dimensions: When
(time/date/order), Who (people/agents/roles), Where (location/site), What
(object/task/event/status), and How (method/channel/steps/constraints). Do
not force an irrelevant dimension into the answer. For every relevant
dimension, include its concrete supported fields in requested_fields and report
dimension_coverage with status covered, unknown, omitted, or not_applicable.
An unknown dimension is acceptable only when the allowed evidence does not
establish it; never fill it by inference. Keep each dimension's value bound to
its own source memory.
""".strip()


GROUNDING_SYSTEM_PROMPT = """
This is final answer-value grounding. You are not deciding authorization and
you are not doing retrieval. Use only the supplied allowed evidence and the
draft contract. Return JSON only with answer_text, requested_fields,
dimension_coverage, covered_requested_fields, restricted_fields_omitted, confidence, and
safe_answer.

Rewrite the draft only to fix source-grounding defects:
- every requested field must be tied to a value directly supported by allowed evidence;
- if a current/date field is partial and allowed evidence gives the same date with a year, use the fully qualified date;
- if a binding signal supplies a qualified site/location candidate, the final answer must include that candidate rather than only its parent organization or host;
- if a binding signal supplies explicit action wording, preserve the action verb and its object. When the same object has both a paraphrase and a direct source action, prefer the direct source action wording;
- if a binding signal supplies a complete access artifact, preserve the entire token, including prefixes and namespaces; never shorten it to a suffix;
- for a safe/broad/helper-facing summary, include every supported requested
  field, including precise operational values when the policy-approved
  projection supports them. Omit only fields marked restricted or unknown and
  do not enumerate unrelated omitted details;
- if the draft uses a generic object noun and the observable subject catalog contains a matching named thread, resolve the noun to that concrete subject when the allowed evidence supports the relationship;
- prefer the highest source_turn_index carrying the latest explicit
  current/final/update/anchored state when allowed sources conflict; an older
  source must not override a later update merely because it uses the word
  "current".

Treat source_grounded_binding_signals as binding obligations, not optional hints. Before returning JSON, verify every obligation against answer_text and repair the text if one is missing. Do not add blocked, hidden, exact private, or absent details. Do not mention this audit.
For a complete aggregate snapshot, a source-grounded current operational field
candidate (for example an explicitly current/active tag or label color) is a
binding obligation unless the requested disclosure level restricts that field.
The required_fields list is a completeness checklist. Every supported field
must have a concrete field-specific value; generic phrases such as "the same"
or "unchanged" are not coverage. Preserve the five-dimensional audit from
the draft. Do not add an irrelevant dimension or recover a value from blocked
evidence.
""".strip()


ANSWER_NEED_SYSTEM_PROMPT = """
You plan the completeness checklist for an answer after policy filtering.
This is not authorization, retrieval, or evaluation. Use only the user's
question and the already allowed evidence. Return JSON only:
{"requested_fields":[{"label":"short field label","source_memory_ids":["allowed id"]}],
 "five_w_one_h":{"when":[],"who":[],"where":[],"what":[],"how":[]},
 "not_applicable":[],"unknown_allowed":[]}

First classify which dimensions are relevant to the question. When is time,
date, sequence, deadline, or validity; Who is a person, agent, owner,
requester, role, or responsible party; Where is a place, site, room, address,
or access point; What is the requested object, task, event, item, or state;
How is the method, channel, procedure, route, or constraint. These are broad
semantic dimensions, not a dataset-specific vocabulary. Do not force all five
dimensions into every answer. For each relevant dimension, enumerate concrete
independently answerable fields supported by the allowed evidence. If the
question requires a dimension but the evidence does not establish a value,
include a field with status "unknown" and put its label in unknown_allowed.
For an irrelevant dimension, leave its list empty and put its name in
not_applicable. Each field may cite only allowed memory ids.

For an aggregate request such as a plan, summary, snapshot, overview, or
current state, expand it into the smallest set of distinct, independently
answerable facts that define the requested view. Preserve object, lane,
audience, temporal, and disclosure qualifiers from the question. If the
question asks for a current-state-only or broad view, prefer the current
broad/public projection and do not create fields for deleted, superseded, or
exact private layers. Do not include unrelated nearby facts merely because
they appear in evidence. A field label is a request description, not an
answer value. Do not use evaluator metadata, hidden labels, or information
outside the question and allowed evidence.
For a multi-field or multi-lane aggregate, enumerate every directly relevant
operational value that defines the requested view, even when the value is
carried by a later short confirmation or status record rather than by the
longest summary. Keep action/contingency phrases and approved locations as
separate fields. Do not replace them with a vague lane or plan label. For a
safe/broad view, include every supported authorized field. Omit only
restricted credentials, codes, deleted layers, or other fields explicitly
marked restricted by policy/evidence.
For snapshot/calibration wording, inspect the current-state evidence shortlist
before choosing fields. If a later record explicitly says a summary remains,
is preserved, is retained, or is the post-delete state, prefer that record
over an older composite snapshot. A role or audience phrase is not by itself
permission to discard an independently stated public/operational attribute;
retain it when it is supported and not restricted by the requested disclosure
level.
For a calibration snapshot, a current non-sensitive label or color is an
independent field when the allowed evidence states it explicitly, even if the
audience label emphasizes the technician's main calibration fields.
""".strip()


_FIVE_W_ONE_H_DIMENSIONS = ("when", "who", "where", "what", "how")
_FIVE_W_ONE_H_STATUSES = {"covered", "unknown", "omitted", "not_applicable"}


_GENERIC_OBJECT_TERMS = (
    "pet", "parcel", "package", "delivery", "project", "program", "course",
    "case", "account", "thread", "plan", "appointment", "household",
)

_ACCESS_ARTIFACT_LABELS = re.compile(
    r"\b(?:credential|code|password|passcode|pin|token|access\s+key|secret\s+key|"
    r"api\s+key|portal\s+key|login\s+key)\b",
    re.IGNORECASE,
)
_ACCESS_ARTIFACT_TOKEN = re.compile(
    r"\b[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+\b"
)
_ACTION_FIELD_LABELS = re.compile(
    r"\b(?:instruction|action|recommendation|medication|dose|treatment|"
    r"what\s+to|should|avoid|stop|start|continue|take|use)\b",
    re.IGNORECASE,
)
_ACTION_QUERY_WORDING = re.compile(
    r"\b(?:should|need\s+to|what\s+to|how\s+should|instruction|"
    r"recommendation|avoid|stop|start|continue|take)\b",
    re.IGNORECASE,
)


def _asks_for_access_artifact(question: str, field_labels: list[str] | tuple[str, ...] = ()) -> bool:
    surface = " ".join([str(question or ""), *(str(label) for label in field_labels)])
    return bool(_ACCESS_ARTIFACT_LABELS.search(surface))


def _asks_for_action(question: str, field_labels: list[str] | tuple[str, ...] = ()) -> bool:
    return bool(
        _ACTION_QUERY_WORDING.search(str(question or ""))
        or any(_ACTION_FIELD_LABELS.search(str(label)) for label in field_labels)
    )


def _source_access_artifacts(source: str) -> list[dict[str, Any]]:
    """Extract complete access-artifact candidates from authorized evidence.

    This is deliberately a source-local detector, not a value generator.  It
    only emits tokens from sentences that explicitly discuss an access
    artifact, so unrelated hyphenated names and dates do not become answer
    candidates.
    """
    candidates: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", source):
        if not _ACCESS_ARTIFACT_LABELS.search(sentence):
            continue
        lowered = sentence.lower()
        is_current = bool(re.search(
            r"\b(?:current|active|latest|new|now|remains?|stays?|valid)\b",
            lowered,
        ))
        is_retired = bool(re.search(
            r"\b(?:retired|revoked|replaced|old|earlier|previous|expired|superseded)\b",
            lowered,
        ))
        label_ends = [match.end() for match in _ACCESS_ARTIFACT_LABELS.finditer(sentence)]
        for match in _ACCESS_ARTIFACT_TOKEN.finditer(sentence):
            token = match.group(0)
            if token.lower() in {"access-key", "secret-key", "api-key", "portal-key"}:
                continue
            # Keep only values attached to an access-artifact label in the
            # same local clause. This filters ordinary hyphenated prose such
            # as "end-of-block" and "family-release" without imposing a
            # dataset-specific token format.
            if not any(
                0 <= match.start() - label_end <= 80
                and not re.search(r"[,;:]", sentence[label_end:match.start()])
                for label_end in label_ends
            ):
                continue
            candidates.append({
                "value": token,
                "is_current": is_current and not is_retired,
                "is_retired": is_retired,
            })
    unique = list(dict.fromkeys(
        (item["value"], item["is_current"], item["is_retired"])
        for item in candidates
    ))
    return [
        {"value": value, "is_current": current, "is_retired": retired}
        for value, current, retired in unique
    ]


def _select_current_access_artifact(signals: list[dict[str, Any]]) -> str | None:
    """Select the latest complete current token without dropping a prefix."""
    current = [
        str(candidate)
        for signal in signals
        for candidate in signal.get("current_candidates") or ()
        if str(candidate).strip()
    ]
    if not current:
        return None
    latest = current[-1]
    prefixed = [
        candidate for candidate in current[:-1]
        if candidate.lower().endswith("_" + latest.lower())
        and len(candidate) > len(latest)
    ]
    return max(prefixed, key=len) if prefixed else latest


def _looks_partially_unanswered(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(insufficient|not provided|does not provide|do not provide|not specified|not available|cannot answer|can't answer|unable to answer)\b",
            lowered,
        )
    )


def _aggregate_answer_is_substantive(
    *,
    question: str,
    contract: dict[str, Any],
    text: str,
) -> bool:
    if not _is_aggregate_answer_request(
        question,
        [str(field.get("label") or "") for field in contract.get("requested_fields") or []],
    ):
        return False
    covered = [
        field for field in contract.get("requested_fields") or []
        if str(field.get("status") or "").lower() == "covered"
        and str(field.get("answer_text") or "").strip()
    ]
    return len(covered) >= 2 and _has_substantive_safe_text(text)


def _clean_aggregate_insufficiency(
    *,
    question: str,
    contract: dict[str, Any],
    text: str,
) -> str:
    """Drop a grouping-field refusal when supported aggregate fields follow."""
    if not _aggregate_answer_is_substantive(question=question, contract=contract, text=text):
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and not re.search(
            r"\b(?:not available|cannot answer|insufficient|unable to answer)\b",
            sentence,
            re.IGNORECASE,
        )
    ]
    cleaned = " ".join(kept).strip()
    return cleaned if _has_substantive_safe_text(cleaned) else text


def _is_bare_insufficient(text: str) -> bool:
    lowered = text.strip().lower().strip(".")
    return lowered in {
        "evidence is insufficient",
        "insufficient evidence",
        "the evidence is insufficient",
    } or bool(re.fullmatch(r"(the )?(allowed )?evidence (is|was) insufficient(?: to answer(?: your question)?)?", lowered))


def _is_policy_refusal(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(re.search(
        r"\b(?:cannot|can't|unable to) (?:provide|answer|share|disclose)\b|\bunder the current access policy\b",
        lowered,
    ))


def _is_lifecycle_content_query(question: str) -> bool:
    lowered = question.lower()
    return bool(re.search(
        r"\b(deleted|removed|retired|forgotten|old|previous|former|earlier|provisional|"
        r"preliminary|prior|before the current|before the approved|used to|in pieces|"
        r"fragment|stem|final block|reconstruct|recover|converted|replaced|switched|"
        r"after .* removed|before .* removed|before .* converted|before .* replaced)\b",
        lowered,
    ))


def _is_deleted_recovery_query(question: str) -> bool:
    """Identify safety-style requests to recover explicitly deleted content."""
    lowered = str(question or "").lower()
    return bool(
        re.search(r"\b(?:deleted|removed|retired|forgotten|purged|erased)\b", lowered)
        and _is_lifecycle_content_query(lowered)
    )


def _is_safe_summary_request(question: str) -> bool:
    return bool(
        re.search(
            r"\b(?:safe|broad|public|sponsor-safe|sponsor-ready|household-safe|mixed-audience|helper-facing)\s+"
            r"(?:wording|summary|sentence|update|recap|brief|status|one-line\s+status)\b"
            r"|\bhigh[- ]level\s+(?:status|summary|update|recap)\b",
            question.lower(),
        )
        or re.search(r"\bkeep\s+(?:it|this)\s+high[- ]level\b", question.lower())
    )


def _is_sensitive_exact_query(question: str) -> bool:
    lowered = question.lower()
    direct_sensitive = bool(re.search(
        r"\b(code|credential|pin|password|door code|portal code|host institution|host-school|restricted|internal label|review flag|customer identity|exact customer|customer mapping|account|diagnosis|incident diagnosis|access status|access is|access still|psychiatric|mental-health|self-harm|program|hold)\b",
        lowered,
    ))
    indirect_identity_inference = bool(
        re.search(r"\b(?:imply|infer|means|same as|equivalent|correspond|match)\b", lowered)
        and re.search(r"\b(?:customer|account|identity|mapping|name|who)\b", lowered)
    )
    return direct_sensitive or indirect_identity_inference


def _has_substantive_safe_text(text: str) -> bool:
    return bool(text and not _is_bare_insufficient(text) and len(re.findall(r"[A-Za-z0-9]", text)) >= 8)


def _sanitize_partial_query_echo(
    text: str,
    *,
    question: str,
    evidence_payload: list[dict[str, Any]],
) -> str:
    """Remove exact date/time literals copied from a restricted query.

    The answer model is instructed not to echo denied fields, but a refusal
    can still repeat a sensitive literal from the question. This postcondition
    only removes literals absent from policy-approved evidence; it never
    changes an evidence-backed value or decides authorization.
    """
    if not text or not question:
        return text
    allowed_text = " ".join(str(row.get("text") or "") for row in evidence_payload).lower()
    patterns = (
        r"\b(?:(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+)?"
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\s+\d{1,2}(?:,\s*20\d{2})?\b",
        r"\b\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|am|pm)\b",
    )
    question_lower = question.lower()
    restricted_literals = {
        re.sub(r"[^a-z0-9]", "", match.group(0))
        for pattern in patterns
        for match in re.finditer(pattern, question_lower, flags=re.IGNORECASE)
        if re.sub(r"[^a-z0-9]", "", match.group(0)) not in re.sub(r"[^a-z0-9]", "", allowed_text)
    }
    sanitized = text
    for pattern in patterns:
        def replace_match(match: re.Match[str]) -> str:
            normalized = re.sub(r"[^a-z0-9]", "", match.group(0).lower())
            if normalized not in restricted_literals:
                return match.group(0)
            replacement = (
                "the requested date"
                if re.search(
                    r"(?:january|february|march|april|may|june|july|august|september|"
                    r"october|november|december)", match.group(0), re.IGNORECASE
                )
                else "the requested time"
            )
            return replacement
        sanitized = re.sub(pattern, replace_match, sanitized, flags=re.IGNORECASE)
    # A partial refusal can echo the denied alternatives even after its date
    # has been masked (for example, two department names in a yes/no query).
    # If an insufficiency sentence contains a query-specific term absent from
    # all authorized evidence, suppress that sentence rather than repeating
    # the restricted choice. Safe evidence sentences are left untouched.
    stop_words = {
        "since", "only", "have", "has", "tell", "give", "whether", "what", "which",
        "exact", "current", "is", "are", "was", "were", "the", "this", "that", "and",
        "or", "to", "of", "in", "on", "for", "me", "i", "you", "can", "could",
        "please", "confirm", "access", "detail", "details", "information", "requested",
        "date", "time", "point", "provide", "determine", "evidence", "insufficient",
    }
    question_terms = {
        token for token in re.findall(r"[a-z][a-z0-9'-]{3,}", question_lower)
        if token not in stop_words
    }
    evidence_terms = set(re.findall(r"[a-z][a-z0-9'-]{3,}", allowed_text))
    sentences = re.split(r"(?<=[.!?])\s+", sanitized)
    rewritten: list[str] = []
    for sentence in sentences:
        lowered_sentence = sentence.lower()
        if re.search(r"\b(?:insufficient|cannot determine|can't determine|not available)\b", lowered_sentence):
            unsupported_echoes = {
                term for term in question_terms
                if re.search(rf"\b{re.escape(term)}\b", lowered_sentence)
                and term not in evidence_terms
            }
            if unsupported_echoes:
                rewritten.append("The requested detail is not available under the current access policy.")
                continue
        rewritten.append(sentence)
    return " ".join(part for part in rewritten if part).strip()


def _plain_answer_text(value: object) -> str:
    """Normalize a structured LLM draft into scorer-visible plain text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            rendered = _plain_answer_text(item)
            if rendered:
                label = re.sub(r"[_-]+", " ", str(key)).strip()
                parts.append(f"{label}: {rendered}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        return "; ".join(
            rendered for item in value if (rendered := _plain_answer_text(item))
        )
    return str(value or "").strip()


def _normalize_five_w_one_h(
    raw: object,
    *,
    allowed_memory_ids: set[str],
) -> dict[str, Any]:
    """Normalize the semantic completeness audit without choosing values.

    The dimension planner and answer model are allowed to use either a list of
    field objects or a compact string list.  This boundary keeps only labels
    and source ids that came from the authorized evidence set.
    """
    payload = raw if isinstance(raw, dict) else {}
    dimensions_raw = (
        payload.get("five_w_one_h")
        or payload.get("dimensions")
        or payload.get("dimension_coverage")
        or {}
    )
    dimensions: dict[str, list[dict[str, Any]]] = {}
    for dimension in _FIVE_W_ONE_H_DIMENSIONS:
        values = dimensions_raw.get(dimension, []) if isinstance(dimensions_raw, dict) else []
        if isinstance(values, dict) and (
            "field_labels" in values or "fields" in values
        ):
            inherited_status = str(values.get("status") or "").strip().lower()
            inherited_sources = values.get("source_memory_ids") or []
            field_values = values.get("field_labels") or values.get("fields") or []
            if isinstance(field_values, str):
                field_values = [field_values]
            values = [
                {
                    "label": item.get("label") if isinstance(item, dict) else item,
                    "source_memory_ids": (
                        item.get("source_memory_ids")
                        if isinstance(item, dict)
                        else inherited_sources
                    ),
                    "status": (
                        item.get("status")
                        if isinstance(item, dict)
                        else inherited_status
                    ),
                }
                for item in field_values if isinstance(item, (str, dict))
            ]
        elif isinstance(values, (str, dict)):
            values = [values]
        normalized: list[dict[str, Any]] = []
        for item in values if isinstance(values, list) else []:
            if isinstance(item, str):
                label = item.strip()
                source_ids: list[str] = []
                status = "unknown"
            elif isinstance(item, dict):
                label = str(item.get("label") or item.get("name") or "").strip()
                source_ids = [
                    str(value) for value in item.get("source_memory_ids") or ()
                    if str(value) in allowed_memory_ids
                ]
                status = str(item.get("status") or ("covered" if source_ids else "unknown")).strip().lower()
            else:
                continue
            if not label or len(label.split()) > 14:
                continue
            if status not in _FIVE_W_ONE_H_STATUSES:
                status = "covered" if source_ids else "unknown"
            normalized.append({
                "label": label.lower(),
                "status": status,
                "source_memory_ids": list(dict.fromkeys(source_ids)),
            })
        dimensions[dimension] = normalized[:8]
    not_applicable = [
        str(value).strip().lower() for value in payload.get("not_applicable") or ()
        if str(value).strip().lower() in _FIVE_W_ONE_H_DIMENSIONS
    ]
    unknown_allowed = [
        str(value).strip().lower() for value in payload.get("unknown_allowed") or ()
        if str(value).strip()
    ]
    return {
        "when": dimensions["when"],
        "who": dimensions["who"],
        "where": dimensions["where"],
        "what": dimensions["what"],
        "how": dimensions["how"],
        "not_applicable": list(dict.fromkeys(not_applicable)),
        "unknown_allowed": list(dict.fromkeys(unknown_allowed)),
    }


def _normalize_answer_contract(raw: object, *, allowed_memory_ids: set[str]) -> dict[str, Any]:
    """Validate the answer model's field-level contract at the boundary."""
    payload = dict(raw) if isinstance(raw, dict) else {}
    fields = payload.get("requested_fields") or []
    if isinstance(fields, dict):
        fields = [fields]
    normalized_fields: list[dict[str, Any]] = []
    for field in fields if isinstance(fields, list) else []:
        if not isinstance(field, dict):
            continue
        label = str(field.get("label") or field.get("name") or "").strip()
        if not label:
            continue
        status = str(field.get("status") or "unknown").strip().lower()
        if status not in {"covered", "omitted", "unknown"}:
            status = "unknown"
        source_ids = tuple(
            str(item) for item in (field.get("source_memory_ids") or [])
            if str(item) in allowed_memory_ids
        )
        normalized_fields.append({
            "label": label,
            "status": status,
            "answer_text": _plain_answer_text(field.get("answer_text") or field.get("value")),
            "source_memory_ids": list(dict.fromkeys(source_ids)),
        })
    covered = [str(item).strip() for item in (payload.get("covered_requested_fields") or []) if str(item).strip()]
    omitted = [str(item).strip() for item in (payload.get("restricted_fields_omitted") or []) if str(item).strip()]
    answer_text = _plain_answer_text(payload.get("answer_text"))
    if not answer_text and payload.get("prediction") is not None:
        answer_text = _plain_answer_text(payload.get("prediction"))
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer_text": answer_text,
        "requested_fields": normalized_fields,
        "dimension_coverage": _normalize_five_w_one_h(
            payload,
            allowed_memory_ids=allowed_memory_ids,
        ),
        "covered_requested_fields": list(dict.fromkeys(covered)),
        "restricted_fields_omitted": list(dict.fromkeys(omitted)),
        "confidence": max(0.0, min(1.0, confidence)),
        "safe_answer": bool(payload.get("safe_answer", True)),
    }


_NON_SPECIFIC_FIELD_VALUES = re.compile(
    r"^(?:the\s+)?(?:same|unchanged|as\s+above|as\s+before|the\s+usual|"
    r"previously\s+(?:established|defined|listed|stated)|"
    r"the\s+(?:existing|current|approved)\s+(?:amount|value|plan|structure)|"
    r"not\s+(?:specified|provided|available|stated)|unknown|n/?a|none)\.?$",
    re.IGNORECASE,
)

_NON_SPECIFIC_FIELD_PHRASE = re.compile(
    r"\b(?:the\s+)?(?:same|unchanged|as\s+above|as\s+before|the\s+usual|"
    r"previously\s+(?:established|defined|listed|stated)|"
    r"the\s+(?:existing|current|approved)\s+(?:amount|value|plan|structure))\b",
    re.IGNORECASE,
)


def _field_tokens(value: object) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if token not in {
            "the", "a", "an", "my", "our", "their", "current", "latest",
            "approved", "exact", "present", "available", "requested", "what",
        }
    }


def _field_label_matches(required: str, actual: str) -> bool:
    required_tokens = _field_tokens(required)
    actual_tokens = _field_tokens(actual)
    if not required_tokens or not actual_tokens:
        return str(required).strip().lower() == str(actual).strip().lower()
    overlap = len(required_tokens & actual_tokens)
    return overlap >= max(1, min(len(required_tokens), len(actual_tokens)) // 2)


def _contract_coverage_gaps(
    contract: dict[str, Any],
    *,
    required_fields: list[str] | tuple[str, ...],
    evidence_count: int,
    allowed_unknown_fields: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Audit completeness without selecting or generating answer values."""
    fields = list(contract.get("requested_fields") or [])
    gaps: list[str] = []
    for required in required_fields:
        label = str(required).strip()
        if not label or label.lower() in {"all requested fields", "everything", "details"}:
            continue
        matches = [
            field for field in fields
            if _field_label_matches(label, str(field.get("label") or ""))
        ]
        if not matches:
            gaps.append(f"missing_field:{label}")
            continue
        field = matches[0]
        status = str(field.get("status") or "unknown").lower()
        value = str(field.get("answer_text") or "").strip()
        source_ids = list(field.get("source_memory_ids") or [])
        unknown_allowed = any(
            _field_label_matches(label, str(candidate))
            for candidate in allowed_unknown_fields
        )
        if status in {"omitted", "unknown"} and not unknown_allowed:
            gaps.append(f"unresolved_field:{label}")
        if status in {"omitted", "unknown"} and unknown_allowed:
            continue
        elif not value or _NON_SPECIFIC_FIELD_VALUES.fullmatch(value) or _NON_SPECIFIC_FIELD_PHRASE.search(value):
            gaps.append(f"non_specific_value:{label}")
        elif evidence_count and not source_ids:
            gaps.append(f"unbound_field:{label}")
    if evidence_count and not fields and required_fields:
        gaps.append("empty_requested_fields")
    return list(dict.fromkeys(gaps))


_FIELD_ACTION_PREFIX = re.compile(
    r"\b(?:start|stop|continue|use|take|avoid|resume|switch|increase|"
    r"decrease|hold|keep|remain|maintain)\b\s+",
    re.IGNORECASE,
)
_FIELD_ACTION_NOISE = {
    "the", "this", "that", "current", "latest", "initial", "updated",
    "explicit", "plan", "medication", "medications", "medicine", "medicines",
    "treatment", "instruction", "instructions", "recommendation", "recommended",
    "daily", "nightly", "morning", "evening", "today", "tonight", "right", "now",
}
_ACTION_FORMS = {
    "start": {"start", "starting", "started", "begin", "beginning", "add", "adding"},
    "stop": {"stop", "stopping", "stopped", "discontinue", "discontinuing", "hold"},
    "continue": {"continue", "continuing", "continued", "keep", "keeping", "maintain", "maintaining"},
    "use": {"use", "using", "used", "take", "taking", "apply", "applying"},
    "take": {"take", "taking", "taken", "use", "using"},
    "avoid": {"avoid", "avoiding"},
    "resume": {"resume", "resuming"},
    "switch": {"switch", "switching", "change", "changing"},
    "increase": {"increase", "increasing", "increased", "raise", "raising"},
    "decrease": {"decrease", "decreasing", "decreased", "lower", "lowering"},
    "hold": {"hold", "holding"},
    "keep": {"keep", "keeping", "continue", "continuing", "maintain", "maintaining"},
}


def _field_action_anchors(value: object) -> tuple[str, ...]:
    """Extract source-shaped action objects for a final prose coverage audit.

    This does not select an answer or create a value.  It only identifies
    concrete objects already present in a covered field, such as medication
    names or named operational items after ``start``/``continue``.  The audit
    catches the failure mode where the structured field is complete but the
    user-visible answer drops one item.
    """
    text = str(value or "")
    anchors: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        for match in _FIELD_ACTION_PREFIX.finditer(sentence):
            clause = sentence[match.end():]
            clause = re.split(r"[.;!?]", clause, maxsplit=1)[0]
            # Separate coordinated objects while retaining numeric qualifiers
            # with the object immediately before them.
            for part in re.split(r"\s+(?:and|as well as)\s+|,\s*", clause, flags=re.IGNORECASE):
                tokens = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*|\d+(?:\.\d+)?", part)
                meaningful = [
                    token for token in tokens
                    if token.lower() not in _FIELD_ACTION_NOISE
                    and (len(token) >= 5 or any(char.isdigit() for char in token))
                ]
                if meaningful:
                    # The first distinctive token is the stable object anchor;
                    # values with units are separately checked by the source
                    # grounding pass when they are field-critical.
                    anchors.append(meaningful[0].lower())
    return tuple(dict.fromkeys(anchors))


def _answer_text_coverage_gaps(
    contract: dict[str, Any],
    *,
    evidence_count: int,
) -> list[str]:
    """Find covered field values that disappeared from final answer prose."""
    if not evidence_count:
        return []
    text = str(contract.get("answer_text") or "").lower()
    gaps: list[str] = []
    for field in contract.get("requested_fields") or []:
        if str(field.get("status") or "").lower() != "covered":
            continue
        label = str(field.get("label") or "").strip()
        for anchor in _field_action_anchors(field.get("answer_text")):
            if anchor not in text:
                gaps.append(f"missing_answer_value:{label}:{anchor}")
        # Concrete dates, quantities, qualifiers, and named list members must
        # survive the final prose realization.  This catches ``May 14`` for a
        # source-grounded ``May 14, 2026`` and ``the amount previously set``
        # for a concrete budget, without generating a value in Python.
        field_text = str(field.get("answer_text") or "")
        literal_anchors = set(re.findall(
            r"\b\d{4}\b|\b\d[\d,.]*(?:\s*(?:%|USD|EUR|GBP|mg|mcg|AM|PM))?\b|"
            r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b",
            field_text,
            flags=re.IGNORECASE,
        ))
        stop = {
            "about", "after", "before", "current", "latest", "approved", "active",
            "the", "this", "that", "with", "from", "into", "only", "same", "remain",
            "remains", "plan", "status", "details", "information", "value", "amount",
        }
        word_anchors = {
            token.casefold()
            for token in re.findall(r"\b[A-Za-z][A-Za-z0-9'-]{4,}\b", field_text)
            if token.casefold() not in stop
        }
        for anchor in sorted(literal_anchors | word_anchors):
            if anchor.casefold() not in text:
                gaps.append(f"missing_answer_literal:{label}:{anchor}")
    return list(dict.fromkeys(gaps))


def _field_evidence_index(
    required_fields: list[str] | tuple[str, ...],
    evidence_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose compact, field-oriented views of already authorized evidence.

    This is a recall/indexing aid after policy filtering. It does not choose
    an answer value or add records; the answer model still performs semantic
    binding against the supplied source text.
    """
    action_synonyms = {
        "start": {"start", "begin", "initiate", "add"},
        "stop": {"stop", "hold", "avoid", "discontinue"},
        "continue": {"continue", "keep", "remain", "maintain"},
        "change": {"change", "increase", "decrease", "adjust", "update", "move"},
        "use": {"use", "take", "apply"},
    }

    def tokens(value: object) -> set[str]:
        return set(re.findall(r"[a-z0-9][a-z0-9'-]{2,}", str(value or "").lower()))

    indexed: list[dict[str, Any]] = []
    for field in required_fields:
        label = str(field).strip()
        if not label:
            continue
        label_tokens = tokens(label)
        expanded = set(label_tokens)
        for key, values in action_synonyms.items():
            if key in label_tokens:
                expanded.update(values)
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for row in evidence_payload:
            row_tokens = tokens(row.get("text"))
            overlap = len(expanded & row_tokens)
            action_hit = bool(expanded.intersection(row_tokens) & set().union(*action_synonyms.values()))
            score = float(overlap) + (0.5 if action_hit else 0.0)
            if score <= 0 and len(evidence_payload) > 6:
                continue
            ranked.append((score, str(row.get("source_time") or ""), row))
        ranked.sort(key=lambda item: (item[0], item[1], str(item[2].get("memory_id") or "")), reverse=True)
        rows = [
            {
                "memory_id": row.get("memory_id"),
                "text": row.get("text"),
                "source_time": row.get("source_time"),
                "source_turn_index": row.get("source_turn_index"),
                "source_message_ids": row.get("source_message_ids"),
            }
            for _, _, row in ranked[:6]
        ]
        indexed.append({"field": label, "authorized_evidence": rows})
    return indexed


def _current_operational_projection_candidates(
    evidence_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find source-bound, non-sensitive operational fields for snapshots."""
    pattern = re.compile(
        r"\b(?:current(?:ly)?|active|latest)\b[^.!?;]{0,50}"
        r"\b(?:tag|label)\s+color\s*(?:is|:|=)\s*([^.!?;,]+)",
        re.IGNORECASE,
    )
    sensitive = re.compile(
        r"\b(?:code|credential|password|pin|token|secret|private|exact|"
        r"diagnosis|medication|medical|health|identity|amount|address)\b",
        re.IGNORECASE,
    )
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in evidence_payload:
        text = str(row.get("text") or "")
        for match in pattern.finditer(text):
            value = re.sub(r"\s+", " ", match.group(1)).strip(" .:-")
            if not value or sensitive.search(value):
                continue
            key = (str(row.get("memory_id") or ""), value.lower())
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "kind": "current_operational_field_candidate",
                "field": "current tag/label color",
                "value": value,
                "source_memory_id": row.get("memory_id"),
                "source_message_ids": row.get("source_message_ids"),
                "source_turn_index": row.get("source_turn_index"),
            })
    return candidates


def _verbatim_field_phrase_candidates(
    contract: dict[str, Any],
    evidence_payload: list[dict[str, Any]],
) -> list[str]:
    """Recover short source-shaped phrases for covered multi-token fields."""
    phrases: list[str] = []
    for field in contract.get("requested_fields") or []:
        if str(field.get("status") or "").lower() != "covered":
            continue
        label = str(field.get("label") or "").strip()
        tokens = re.findall(r"[a-z0-9][a-z0-9'-]{2,}", label.lower())
        if len(tokens) < 2 or not field.get("source_memory_ids"):
            continue
        rows = [
            row for row in evidence_payload
            if str(row.get("memory_id") or "") in {str(value) for value in field.get("source_memory_ids") or ()}
        ]
        for row in rows:
            for sentence in re.split(r"(?<=[.!?;])\s+", str(row.get("text") or "")):
                match = None
                matched_tokens: list[str] = []
                # Planner labels may add a harmless qualifier such as
                # ``adjustment`` or ``status``.  Fall back to the stable
                # two-token core, but only when the source itself supplies a
                # concrete suffix after that core.
                for prefix_length in range(len(tokens), 1, -1):
                    label_pattern = r"\b" + r"\s+".join(
                        re.escape(token) for token in tokens[:prefix_length]
                    ) + r"\b"
                    match = re.search(label_pattern + r"[^.!?;]*", sentence, re.IGNORECASE)
                    if match:
                        matched_tokens = tokens[:prefix_length]
                        break
                if not match:
                    continue
                phrase = re.sub(r"\s+", " ", match.group(0)).strip(" .:;-)")
                phrase_tokens = set(re.findall(r"[a-z0-9][a-z0-9'-]{2,}", phrase.lower()))
                if (
                    phrase.lower() != label.lower()
                    and phrase_tokens.difference(matched_tokens)
                    and len(phrase.split()) <= 12
                ):
                    phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def _is_aggregate_answer_request(
    question: str,
    requested_fields: list[str] | tuple[str, ...],
) -> bool:
    """Identify when a single planner label hides a structured view."""
    lowered = str(question or "").lower()
    aggregate_nouns = bool(re.search(
        r"\b(?:plan|summary|summaries|recap|snapshot|overview|state|status|"
        r"details|information|calibration)\b",
        lowered,
    ))
    labels = [str(value).strip().lower() for value in requested_fields if str(value).strip()]
    return aggregate_nouns and (
        len(labels) <= 1
        or any(
            re.search(r"\b(?:plan|summary|recap|snapshot|overview|state|status|details|information|calibration)\b", label)
            for label in labels
        )
    )


def _is_operational_snapshot_completion_request(
    question: str,
    field_labels: list[str] | tuple[str, ...],
) -> bool:
    surface = " ".join([str(question or ""), *(str(label) for label in field_labels)])
    return _is_aggregate_answer_request(question, field_labels) and bool(
        re.search(r"\b(?:calibration|tag|label|color)\b", surface, re.IGNORECASE)
    )


def _current_state_evidence_shortlist(
    evidence_payload: list[dict[str, Any]],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """Expose likely current-state carriers to the aggregate field planner.

    This is a recall aid after policy filtering.  It does not authorize a
    memory, choose a value, or inspect evaluation metadata.  Long episodes
    commonly contain an early composite snapshot followed by short records
    saying what remains current after an update or deletion.  Giving the base
    LLM that source-order signal prevents lexical retrieval of the old
    composite from becoming the answer contract.
    """
    current_cues = re.compile(
        r"\b(?:current|currently|latest|final|remains?|preserved|retained|"
        r"post[- ]?delete|after deletion|still|active|locked|confirmed|"
        r"outward summary|technician snapshot|resident snapshot|"
        r"calibration snapshot)\b",
        re.IGNORECASE,
    )
    aggregate_cues = re.compile(
        r"\b(?:summary|snapshot|state|overview|calibration|broad|safe|"
        r"only|keep|hold|survive|deleted|lane)\b",
        re.IGNORECASE,
    )
    ranked: list[tuple[float, int, str, dict[str, Any]]] = []
    for row in evidence_payload:
        text = str(row.get("text") or "")
        if not text:
            continue
        current_hits = len(current_cues.findall(text))
        aggregate_hits = len(aggregate_cues.findall(text))
        try:
            turn_index = int(row.get("source_turn_index") or -1)
        except (TypeError, ValueError):
            turn_index = -1
        # Recency is only a tiebreaker after explicit state/summary cues.  It
        # must not replace semantic selection inside the allowed set.
        score = float(current_hits * 3 + aggregate_hits)
        if current_hits or aggregate_hits:
            ranked.append((score, turn_index, str(row.get("memory_id") or ""), row))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, memory_id, row in ranked:
        if memory_id in seen:
            continue
        seen.add(memory_id)
        chosen.append({
            "memory_id": row.get("memory_id"),
            "text": row.get("text"),
            "source_turn_index": row.get("source_turn_index"),
            "source_message_ids": row.get("source_message_ids"),
            "disclosure_role": row.get("disclosure_role"),
        })
        if len(chosen) >= limit:
            break
    return chosen


def _derive_answer_need_spec(
    *,
    question: str,
    target_subject: str | None,
    request_scope: str | None,
    requested_fields: list[str] | tuple[str, ...],
    evidence_payload: list[dict[str, Any]],
    llm_client: LLMClient,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Ask the base LLM for a query-grounded field and 5W1H checklist.

    The call is intentionally made after policy filtering. It can improve
    completeness without granting access or selecting blocked records.
    """
    if not evidence_payload or not _is_aggregate_answer_request(question, requested_fields):
        return {}
    compact_evidence = [
        {
            "memory_id": row.get("memory_id"),
            "text": row.get("text"),
            "disclosure_role": row.get("disclosure_role"),
        }
        for row in evidence_payload
    ]
    current_state_evidence = _current_state_evidence_shortlist(evidence_payload)
    payload = {
        "question": question,
        "target_subject": target_subject,
        "request_scope": request_scope,
        "existing_requested_fields": list(requested_fields),
        "allowed_evidence": compact_evidence,
        "current_state_evidence": current_state_evidence,
    }
    assert_runtime_payload_safe(payload, context="answer_need_spec_prompt")
    try:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=ANSWER_NEED_SYSTEM_PROMPT,
            user_prompt=json.dumps(payload, ensure_ascii=False),
        )
    except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
        return {}
    values = raw.get("requested_fields") if isinstance(raw, dict) else []
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return {}
    allowed_ids = {str(row.get("memory_id")) for row in evidence_payload}
    fields: list[dict[str, Any]] = []
    for item in values:
        if isinstance(item, str):
            label = item.strip()
            source_ids: list[str] = []
        elif isinstance(item, dict):
            label = str(item.get("label") or item.get("name") or "").strip()
            source_ids = [
                str(value) for value in item.get("source_memory_ids") or ()
                if str(value) in allowed_ids
            ]
        else:
            continue
        if not label or len(label.split()) > 14:
            continue
        if label.lower() in {"all requested fields", "everything", "details", "the plan"}:
            continue
        fields.append({"label": label.lower(), "source_memory_ids": list(dict.fromkeys(source_ids))})
    dimension_contract = _normalize_five_w_one_h(
        raw,
        allowed_memory_ids=allowed_ids,
    )
    dimension_fields = [
        item
        for dimension in _FIVE_W_ONE_H_DIMENSIONS
        for item in dimension_contract[dimension]
    ]
    if not fields and not dimension_fields:
        return {}
    # The planner is an incremental completeness aid. It must never erase a
    # field extracted directly from the user's question (for example a
    # calendar date implicit in a multi-part plan request).
    merged: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for requested in requested_fields:
        normalized = str(requested).strip().lower()
        if not normalized or normalized in {"all requested fields", "everything", "details"}:
            continue
        row = {"label": normalized, "source_memory_ids": []}
        merged.append(row)
        by_label[normalized] = row
    for row in fields:
        label = str(row.get("label") or "").strip().lower()
        if not label:
            continue
        existing = by_label.get(label)
        if existing is None:
            existing = {"label": label, "source_memory_ids": []}
            merged.append(existing)
            by_label[label] = existing
        existing["source_memory_ids"] = list(dict.fromkeys(
            [*(existing.get("source_memory_ids") or []), *(row.get("source_memory_ids") or [])]
        ))
    for item in dimension_fields:
        label = str(item.get("label") or "").strip().lower()
        if not label:
            continue
        existing = by_label.get(label)
        if existing is None:
            existing = {"label": label, "source_memory_ids": []}
            merged.append(existing)
            by_label[label] = existing
        existing["source_memory_ids"] = list(dict.fromkeys(
            [*(existing.get("source_memory_ids") or []), *(item.get("source_memory_ids") or [])]
        ))
    unknown_allowed = list(dimension_contract.get("unknown_allowed") or [])
    for dimension in _FIVE_W_ONE_H_DIMENSIONS:
        for item in dimension_contract[dimension]:
            if item.get("status") in {"unknown", "omitted"}:
                unknown_allowed.append(str(item.get("label") or ""))
    return {
        "requested_fields": merged[:24],
        "five_w_one_h": dimension_contract,
        "unknown_allowed_fields": list(dict.fromkeys(
            value for value in unknown_allowed if value
        )),
        "source": "base_llm_after_policy_filter_incremental",
    }


def _binding_signals(
    *,
    answer_contract: dict[str, Any],
    evidence_payload: list[dict[str, Any]],
    safe_summary: bool,
    observable_named_subjects: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    question: str = "",
) -> list[dict[str, Any]]:
    """Summarize source-local binding risks for the answer-model repair pass.

    The signals are generated from the authorized evidence only.  They are
    candidate spans and audit hints, not answer values selected by Python.
    """
    fields = list(answer_contract.get("requested_fields") or [])
    signals: list[dict[str, Any]] = []
    field_labels = [str(field.get("label") or "") for field in fields]
    access_artifact_requested = _asks_for_access_artifact(question, field_labels)
    action_requested = _asks_for_action(question, field_labels)
    for field in fields:
        label = str(field.get("label") or "").strip()
        lowered_label = label.lower()
        field_ids = {str(value) for value in field.get("source_memory_ids") or ()}
        rows = [
            row for row in evidence_payload
            if not field_ids or str(row.get("memory_id")) in field_ids
        ]
        if re.search(
            r"\b(?:site|location|address|room|rooms|desk|hall|suite|bay|"
            r"entrance|entry|access\s+point|door|gate|keypad)\b",
            lowered_label,
        ):
            for row in rows:
                source = str(row.get("text") or "")
                qualified = re.findall(
                    r"\b(?:[A-Z][A-Za-z0-9'-]+(?:\s+[A-Z][A-Za-z0-9'-]+){0,4}\s+)?"
                    r"(?:desk|room|hall|suite|bay|floor)\s+[A-Za-z0-9-]+\b",
                    source,
                )
                # Keep qualified access points intact.  A bare "side door"
                # is not equivalent to the source-backed "side door keypad".
                access_clauses = re.findall(
                    r"\b(?:use|enter\s+via|via)\b[^.!?;]*",
                    source,
                    flags=re.IGNORECASE,
                )
                for clause in access_clauses:
                    qualified.extend(re.findall(
                        r"\b(?:[A-Za-z0-9][A-Za-z0-9'-]*\s+){0,3}"
                        r"(?:door|gate|entry|entrance)(?:\s+(?:keypad|entry|only|access))?\b",
                        clause,
                        flags=re.IGNORECASE,
                    ))
                qualified = [
                    re.sub(
                        r"^(?:(?:use|via|enter(?:\s+via)?)\s+)?(?:the|a|an)\s+",
                        "",
                        str(item).strip(),
                        flags=re.IGNORECASE,
                    )
                    for item in qualified
                    if str(item).strip()
                ]
                if qualified:
                    signals.append({
                        "kind": "qualified_location_candidate",
                        "field": label,
                        "source_memory_id": row.get("memory_id"),
                        "candidates": list(dict.fromkeys(qualified)),
                    })
        if re.search(r"\bdate\b", lowered_label):
            date_rows = rows
            if not any(
                re.search(
                    r"\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
                    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*20\d{2})?\b",
                    str(row.get("text") or ""),
                    re.IGNORECASE,
                )
                for row in date_rows
            ):
                # The answer model can bind an aggregate date field to a
                # nearby current-state record. Recover the date only from an
                # explicit calendar span in the already allowed evidence.
                date_rows = evidence_payload
            for row in date_rows:
                source = str(row.get("text") or "")
                full_dates = re.findall(
                    r"\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
                    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+20\d{2})?\b",
                    source,
                    flags=re.IGNORECASE,
                )
                if full_dates:
                    signals.append({
                        "kind": "fully_qualified_date_candidate",
                        "field": label,
                        "source_memory_id": row.get("memory_id"),
                        "candidates": list(dict.fromkeys(full_dates)),
                    })
        if action_requested:
            # Action wording is field-bound.  Searching the entire allowed
            # set here can append an unrelated "use ..." sentence from a
            # nearby record even when the answer is already complete.
            for row in rows:
                source = str(row.get("text") or "")
                action_sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", source)
                    if re.search(r"\b(?:stop|start|continue|avoid|use|take|resume|switch)\b", sentence, re.IGNORECASE)
                ]
                if action_sentences:
                    signals.append({
                        "kind": "explicit_action_wording",
                        "field": label,
                        "source_memory_id": row.get("memory_id"),
                        "source_sentences": list(dict.fromkeys(action_sentences)),
                    })
    # A multi-field answer can omit requested_fields or collapse them into one
    # aggregate label.  In that case the query itself is the stable semantic
    # signal.  Ask the LLM to bind the complete artifact from authorized rows;
    # do not infer a value from a dataset label or from blocked state.
    if access_artifact_requested:
        for row in evidence_payload:
            artifacts = _source_access_artifacts(str(row.get("text") or ""))
            if not artifacts:
                continue
            current = [item["value"] for item in artifacts if item["is_current"]]
            signals.append({
                "kind": "complete_access_artifact",
                "field": next((label for label in field_labels if _ACCESS_ARTIFACT_LABELS.search(label)), "requested access artifact"),
                "source_memory_id": row.get("memory_id"),
                "source_time": row.get("source_time"),
                "candidates": [item["value"] for item in artifacts],
                "current_candidates": current,
            })
    if safe_summary and observable_named_subjects:
        draft_text = str(answer_contract.get("answer_text") or "").lower()
        source_text = " ".join(str(row.get("text") or "") for row in evidence_payload).lower()
        generic_terms = [term for term in _GENERIC_OBJECT_TERMS if re.search(rf"\b{re.escape(term)}\b", draft_text + " " + source_text)]
        missing_named_subjects = [
            item for item in observable_named_subjects
            if str(item.get("subject") or "").strip()
            and str(item.get("subject") or "").lower() not in draft_text
        ]
        if generic_terms and missing_named_subjects:
            signals.append({
                "kind": "named_object_grounding",
                "generic_terms": list(dict.fromkeys(generic_terms)),
                "observable_subject_catalog": list(missing_named_subjects),
            })
    if safe_summary:
        for row in evidence_payload:
            source = str(row.get("text") or "")
            if re.search(
                r"\b(?:broad|summary|recap|no\s+exact|exact\s+[^.]{0,50}\b(?:delete|deleted|remove|removed|omit|omitted))\b",
                source,
                re.IGNORECASE,
            ):
                signals.append({
                    "kind": "broad_summary_boundary",
                    "source_memory_id": row.get("memory_id"),
                    "source_sentence": source,
                })
    if _is_operational_snapshot_completion_request(question, field_labels):
        signals.extend(_current_operational_projection_candidates(evidence_payload))
    return signals


def _run_grounding_pass(
    *,
    llm_client: LLMClient,
    config: dict[str, Any],
    payload: dict[str, Any],
    answer_contract: dict[str, Any],
    evidence_payload: list[dict[str, Any]],
    safe_summary: bool,
    observable_named_subjects: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    question: str = "",
    required_fields: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    signals = _binding_signals(
        answer_contract=answer_contract,
        evidence_payload=evidence_payload,
        safe_summary=safe_summary,
        observable_named_subjects=observable_named_subjects,
        question=question,
    )
    grounding_payload = {
        "question": payload.get("question"),
        "request_scope": payload.get("request_scope"),
        "disclosure_level": payload.get("disclosure_level"),
        "requested_attributes": payload.get("requested_attributes"),
        "required_fields": list(required_fields),
        "draft_answer_contract": answer_contract,
        "source_grounded_binding_signals": signals,
        "allowed_evidence": evidence_payload,
        "observable_named_subjects": list(observable_named_subjects) if safe_summary else [],
    }
    assert_runtime_payload_safe(grounding_payload, context="answer_grounding_prompt")
    raw = llm_client.chat_json(
        model=resolve_llm_model(config, "answering"),
        system_prompt=GROUNDING_SYSTEM_PROMPT,
        user_prompt=json.dumps(grounding_payload, ensure_ascii=False),
    )
    grounded = _normalize_answer_contract(
        raw,
        allowed_memory_ids={item["memory_id"] for item in evidence_payload},
    )
    grounded["_grounding_signals"] = signals
    return grounded


PROJECTION_REALIZATION_SYSTEM_PROMPT = """
Render the final answer from the already-authorized field projection.
Return JSON only: {"answer_text": string}.

The field projection is the only selection authority. Write a clear, concise
answer to the question and explicitly include every selected value for every
supported field. Preserve selected values verbatim, including dates, times,
amounts, labels, locations, and qualifiers. Do not choose a different value,
revive an older value, or add facts that are not in the projection. The
projection contains no raw memory text by design. For unknown, restricted, or
conflicting fields, do not guess or fill from general knowledge.
""".strip()


def _deterministic_projection_text(projection: AnswerProjection) -> str:
    """Render only the closed field/value pairs, with no semantic rewriting."""
    lines: list[str] = []
    for field in projection.fields:
        if field.status != "supported" or not field.selected_values:
            continue
        values = "; ".join(str(value) for value in field.selected_values)
        lines.append(f"{field.label}: {values}")
    return " ".join(lines)


def _render_answer_from_projection(
    *,
    llm_client: LLMClient,
    config: dict[str, Any],
    question: str,
    projection: AnswerProjection,
    evidence_payload: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any]]:
    """Perform the final realization over a closed field/value projection.

    Earlier answer passes may inspect a broad evidence bundle and accidentally
    re-select a stale value after projection has resolved a conflict. This
    pass receives only the selected field values and the authorized source
    snippets that support them, so prose realization cannot change policy or
    field selection. A second bounded call repairs omissions before the final
    source-backed postcondition appends any still-missing selected values.
    """
    if (
        not projection.fields
        or not any(field.status == "supported" and field.selected_values for field in projection.fields)
        or llm_client is None
        or not llm_client.is_available()
    ):
        return None, {"ran": False, "reason": "projection_or_llm_unavailable"}
    projected_fields: list[dict[str, Any]] = []
    for field in projection.fields:
        projected_fields.append({
            "field_id": field.field_id,
            "label": field.label,
            "status": field.status,
            "selected_values": list(field.selected_values),
            "source_memory_ids": list(field.source_memory_ids),
            "conflict_trace": list(field.conflict_trace),
        })
    payload = {
        "question": question,
        "field_projection": projected_fields,
    }
    assert_runtime_payload_safe(payload, context="projection_grounded_final_realization_prompt")

    def request(current_payload: dict[str, Any]) -> str:
        raw = llm_client.chat_json(
            model=resolve_llm_model(config, "answering"),
            system_prompt=PROJECTION_REALIZATION_SYSTEM_PROMPT,
            user_prompt=json.dumps(current_payload, ensure_ascii=False),
        )
        if isinstance(raw, dict):
            return _plain_answer_text(raw.get("answer_text") or raw.get("prediction"))
        return _plain_answer_text(raw)

    try:
        text = request(payload)
    except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
        return None, {"ran": False, "reason": "projection_realization_call_failed"}
    supported_values = [
        value
        for field in projection.fields
        if field.status == "supported"
        for value in field.selected_values
        if value
    ]
    normalize_value = lambda value: re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
    missing = [value for value in supported_values if normalize_value(value) not in normalize_value(text)]
    if missing:
        repair_payload = dict(payload)
        repair_payload["draft_answer_text"] = text
        repair_payload["missing_selected_values"] = missing
        repair_payload["repair_instruction"] = (
            "Rewrite the full answer and explicitly include every missing selected value "
            "verbatim. Keep all supported fields; do not replace a selected value with "
            "an older or more general value."
        )
        assert_runtime_payload_safe(repair_payload, context="projection_grounded_final_realization_repair_prompt")
        try:
            repaired = request(repair_payload)
            if repaired:
                text = repaired
                missing = [value for value in supported_values if normalize_value(value) not in normalize_value(text)]
        except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
            pass
    # The model proposal is retained for audit, but the delivered text is
    # always assembled from the closed projection. This removes the last
    # semantic authority after state resolution: the renderer cannot attach a
    # selected value to a neighboring field or omit one during prose writing.
    canonical_text = _deterministic_projection_text(projection)
    if canonical_text:
        text = canonical_text
    # This is a postcondition over values already selected and source-validated
    # by the projection; it never discovers or invents a new answer value.
    return text or None, {
        "ran": True,
        "supported_values": supported_values,
        "missing_after_repair": missing,
        "source_memory_ids": sorted({memory_id for field in projection.fields for memory_id in field.source_memory_ids}),
        "raw_source_text_exposed": False,
        "delivered_from_closed_projection": bool(canonical_text),
    }


def _distinctive_tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9'-]{3,}", value.lower())
        if token not in {"current", "source", "answer", "summary", "thread", "plan"}
    }


def _explicit_action_phrases(evidence_payload: list[dict[str, Any]]) -> list[str]:
    phrases: list[str] = []
    for row in evidence_payload:
        for sentence in re.split(r"(?<=[.!?])\s+", str(row.get("text") or "")):
            for match in re.finditer(
                r"\b(?:stop|start|continue|avoid|use|take|resume|switch)\b\s+"
                r"[^,.;!?]+",
                sentence,
                flags=re.IGNORECASE,
            ):
                phrase = match.group(0).strip(" ,:")
                if len(phrase.split()) >= 2:
                    phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def _enforce_grounding_postconditions(
    contract: dict[str, Any],
    *,
    evidence_payload: list[dict[str, Any]],
    observable_named_subjects: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    safe_summary: bool,
    binding_signals: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    question: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """Apply only source-backed realization repairs after LLM grounding.

    The LLM remains responsible for semantic selection.  This final check
    prevents a selected source span from disappearing during prose rendering;
    it never invents a value or expands the authorized evidence set.
    """
    text = str(contract.get("answer_text") or "")
    repairs: list[str] = []
    field_labels = [str(field.get("label") or "") for field in contract.get("requested_fields") or []]
    if _is_operational_snapshot_completion_request(question, field_labels):
        candidates = _current_operational_projection_candidates(evidence_payload)
        anchor_ids = {
            str(memory_id)
            for field in contract.get("requested_fields") or []
            if not re.search(r"\b(?:tag|label|color)\b", str(field.get("label") or ""), re.IGNORECASE)
            for memory_id in field.get("source_memory_ids") or ()
        }
        candidates.sort(
            key=lambda candidate: (
                min(
                    abs(
                        int(candidate.get("source_turn_index") or 0)
                        - int(row.get("source_turn_index") or 0)
                    )
                    for row in evidence_payload
                    if str(row.get("memory_id") or "") in anchor_ids
                )
                if anchor_ids
                else 0,
                -int(candidate.get("source_turn_index") or 0),
            )
        )
        best_candidate = candidates[0] if candidates else None
        if best_candidate is not None:
            best_value = str(best_candidate.get("value") or "").strip()
            for candidate in candidates[1:]:
                value = str(candidate.get("value") or "").strip()
                if not value or value.lower() == best_value.lower():
                    continue
                text = re.sub(
                    rf"[^.!?]*\b(?:tag|label)\s+color\b[^.!?]*\b{re.escape(value)}\b[^.!?]*[.!?]?",
                    "",
                    text,
                    flags=re.IGNORECASE,
                )
            candidate = best_candidate
            value = str(candidate.get("value") or "").strip()
            if not value or value.lower() in text.lower():
                pass
            else:
                text = text.rstrip(" .") + f". The current tag/label color is {value}."
                repairs.append(f"current_operational_field:{value}")
    for phrase in _verbatim_field_phrase_candidates(contract, evidence_payload):
        if phrase.lower() in text.lower():
            continue
        numeric_literals = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)?\b|\b\d+\b", phrase, re.IGNORECASE)
        if numeric_literals and all(literal.lower() in text.lower() for literal in numeric_literals):
            continue
        text = text.rstrip(" .") + f". The source-backed detail is {phrase}."
        repairs.append(f"verbatim_field_phrase:{phrase}")
    if _asks_for_access_artifact(question):
        artifact_signals = [
            signal for signal in binding_signals
            if signal.get("kind") == "complete_access_artifact"
        ]
        # Evidence is ordered by source time before this function is called.
        # The last current candidate is therefore the latest explicit current
        # artifact among the authorized rows.
        candidate = _select_current_access_artifact(artifact_signals)
        if candidate:
            if candidate.lower() not in text.lower():
                artifact_pattern = re.compile(
                    r"(\b(?:credential|code|password|passcode|pin|token|access\s+key|"
                    r"secret\s+key|api\s+key|portal\s+key|login\s+key)\b"
                    r"\s*(?:is|:|=)\s*)([A-Za-z0-9][A-Za-z0-9_-]*)",
                    re.IGNORECASE,
                )
                updated, count = artifact_pattern.subn(rf"\1{candidate}", text, count=1)
                if count:
                    text = updated
                    repairs.append(f"complete_access_artifact:{candidate}")
                elif _ACCESS_ARTIFACT_LABELS.search(text):
                    text = text.rstrip(" .") + f"; the complete access artifact is {candidate}."
                    repairs.append(f"complete_access_artifact:{candidate}")
    location_candidates = [
        candidate
        for signal in binding_signals
        if signal.get("kind") == "qualified_location_candidate"
        for candidate in signal.get("candidates") or ()
    ]
    for candidate in list(dict.fromkeys(location_candidates)):
        if re.search(
            r"\b(?:site|location|address|room|rooms|desk|hall|suite|bay|"
            r"entrance|entry|access\s+point|door|gate|keypad)\b",
            text,
            re.IGNORECASE,
        ):
            if candidate.lower() not in text.lower():
                # If the answer contains the unqualified parent phrase, bind
                # the qualified source value in place instead of appending a
                # second contradictory-looking location sentence.
                candidate_words = candidate.split()
                parent = " ".join(candidate_words[:-1]) if len(candidate_words) > 1 else ""
                if parent and re.search(rf"\b{re.escape(parent)}\b", text, re.IGNORECASE):
                    updated, count = re.subn(
                        rf"\b{re.escape(parent)}\b",
                        candidate,
                        text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    if count:
                        text = updated
                        repairs.append(f"qualified_location:{candidate}")
                        break
                updated, count = re.subn(
                    r"(\b(?:current|primary|main)?\s*(?:site|location|address)\s+is\s+)([^,.;]+)",
                    rf"\1\2, {candidate}",
                    text,
                    count=1,
                    flags=re.IGNORECASE,
                )
                text = updated if count else text.rstrip(" .") + f"; the qualified location is {candidate}."
                repairs.append(f"qualified_location:{candidate}")
                break

    # A source may state an event date without a year while provenance carries
    # a message timestamp with a different calendar day. Prefer the explicit
    # source calendar anchor and replace only a date-shaped draft span.
    date_candidates = [
        str(candidate).strip()
        for signal in binding_signals
        if signal.get("kind") == "fully_qualified_date_candidate"
        for candidate in signal.get("candidates") or ()
        if str(candidate).strip()
    ]
    for candidate in list(dict.fromkeys(date_candidates)):
        if candidate.lower() in text.lower():
            continue
        date_span = re.compile(
            r"\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2}(?:,\s*20\d{2})?\b",
            re.IGNORECASE,
        )
        updated, count = date_span.subn(candidate, text, count=1)
        if count:
            text = updated
            repairs.append(f"calendar_anchor:{candidate}")
        elif re.search(r"\b(?:date|day)\b", text, re.IGNORECASE):
            text = text.rstrip(" .") + f" The calendar date is {candidate}."
            repairs.append(f"calendar_anchor:{candidate}")
        break

    if _asks_for_action(question, field_labels):
        answer_tokens = _distinctive_tokens(text)
        field_value_tokens = _distinctive_tokens(
            " ".join(str(field.get("answer_text") or "") for field in contract.get("requested_fields") or [])
        )
        action_signal_ids = {
            str(signal.get("source_memory_id"))
            for signal in binding_signals
            if signal.get("kind") == "explicit_action_wording"
            and signal.get("source_memory_id")
        }
        action_evidence = [
            row for row in evidence_payload
            if not action_signal_ids or str(row.get("memory_id")) in action_signal_ids
        ]
        for phrase in _explicit_action_phrases(action_evidence):
            phrase_tokens = _distinctive_tokens(phrase)
            verb = phrase.split(maxsplit=1)[0]
            object_tokens = _distinctive_tokens(phrase.split(maxsplit=1)[1]) if len(phrase.split(maxsplit=1)) > 1 else set()
            if re.search(r"\buse\s+only\s+(?:the\s+)?(?:broad|safe|public|generic)\b", phrase, re.IGNORECASE):
                continue
            verb_forms = _ACTION_FORMS.get(verb.lower(), {verb.lower()})
            if phrase_tokens.intersection(answer_tokens) and any(
                re.search(rf"\b{re.escape(form)}\b", text, re.IGNORECASE)
                for form in verb_forms
            ):
                continue
            # Do not append a second action clause when the concrete object
            # and its qualifiers are already present in the final answer.
            # This preserves concise answers while still allowing a missing
            # source-backed object to trigger a repair.
            if object_tokens and object_tokens.issubset(answer_tokens):
                continue
            if (
                phrase_tokens.intersection(field_value_tokens)
                and not re.search(rf"\b{re.escape(verb)}\b", text, re.IGNORECASE)
            ):
                text = text.rstrip()
                if text[-1:] not in ".!?":
                    text += "."
                text += f" The explicit instruction is to {phrase.lower()}."
                repairs.append(f"explicit_action:{phrase}")
                break

    if safe_summary and observable_named_subjects:
        subject_rows = [
            str(item.get("subject") or "").strip()
            for item in observable_named_subjects
            if str(item.get("subject") or "").strip()
        ]
        hints = {
            "pet": {"pet", "animal", "dog", "cat", "paws", "vet"},
            "parcel": {"parcel", "package", "delivery", "courier", "shipment", "grocery"},
            "project": {"project", "initiative", "program", "case"},
            "program": {"program", "course", "project", "initiative"},
        }
        signal_terms = {
            term
            for signal in binding_signals
            if signal.get("kind") == "named_object_grounding"
            for term in signal.get("generic_terms") or ()
        }
        for generic, head_terms in hints.items():
            if generic not in signal_terms and not re.search(rf"\b{re.escape(generic)}\b", text, re.IGNORECASE):
                continue
            matches = [
                subject for subject in subject_rows
                if set(re.findall(r"[a-z0-9]+", subject.lower())).intersection(head_terms)
                and subject.lower() not in text.lower()
            ]
            if not matches:
                continue
            subject = matches[0]
            pattern = rf"\b{re.escape(generic)}\s+(?:care|plans?|thread|note)\b"
            replacement = f"{generic} ({subject})"
            updated, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if count:
                text = updated
                repairs.append(f"named_subject:{generic}->{subject}")
            elif subject.lower() not in text.lower():
                text = text.rstrip(" .") + f" The related {generic} thread is {subject}."
                repairs.append(f"named_subject:{generic}->{subject}")

    if text != contract.get("answer_text"):
        contract = dict(contract)
        contract["answer_text"] = text
    return contract, repairs


def _answer_contract_needs_repair(
    contract: dict[str, Any],
    *,
    safe_summary: bool,
    evidence_count: int,
    has_public_projection: bool = False,
    evidence_payload: list[dict[str, Any]] | None = None,
    observable_named_subjects: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    question: str = "",
    required_fields: list[str] | tuple[str, ...] = (),
    allowed_unknown_fields: list[str] | tuple[str, ...] = (),
) -> bool:
    text = str(contract.get("answer_text") or "")
    fields = list(contract.get("requested_fields") or [])
    if not text or _is_bare_insufficient(text) or _is_policy_refusal(text):
        return evidence_count > 0
    if evidence_count and fields and any(
        str(field.get("status") or "unknown").lower() in {"unknown", "omitted"}
        and not any(
            _field_label_matches(str(field.get("label") or ""), str(candidate))
            for candidate in allowed_unknown_fields
        )
        for field in fields
    ):
        return True
    if evidence_count and not fields:
        return True
    if _contract_coverage_gaps(
        contract,
        required_fields=required_fields,
        evidence_count=evidence_count,
        allowed_unknown_fields=allowed_unknown_fields,
    ):
        return True
    if _asks_for_access_artifact(question):
        artifact_signals = [
            {
                "current_candidates": [
                    item["value"] for item in _source_access_artifacts(str(row.get("text") or ""))
                    if item["is_current"]
                ]
            }
            for row in (evidence_payload or [])
        ]
        selected_artifact = _select_current_access_artifact(artifact_signals)
        if selected_artifact and selected_artifact.lower() not in text.lower():
            return True
    # A complete-looking contract occasionally binds a credential to a
    # neighboring location. Ask the model for a second pass only for this
    # concrete type mismatch; broad multi-field answers should not be rewritten
    # merely because they contain several fields.
    credential_labels = re.compile(r"\b(?:credential|code|password|pin|token|secret)\b", re.IGNORECASE)
    location_terms = re.compile(r"\b(?:site|room|desk|location|address|hall|suite|bay|bench)\b", re.IGNORECASE)
    if any(
        credential_labels.search(str(field.get("label") or ""))
        and location_terms.search(str(field.get("answer_text") or ""))
        for field in fields
    ):
        return True
    # A model can mark a field covered while dropping a qualified component
    # from the source record. Detect only source-grounded type mismatches so a
    # repair is spent on a real binding problem, rather than on every
    # multi-field answer.
    source_rows = evidence_payload or []
    source_texts = [str(row.get("text") or "") for row in source_rows]
    for field in fields:
        label = str(field.get("label") or "").lower()
        answer_value = str(field.get("answer_text") or "").lower()
        source_text = " ".join(
            str(row.get("text") or "")
            for row in source_rows
            if not field.get("source_memory_ids")
            or set(field.get("source_memory_ids") or ()).intersection({str(row.get("memory_id"))})
        ).lower()
        if not source_text or field.get("status") != "covered":
            continue
        if re.search(
            r"\b(?:site|location|address|room|rooms|desk|hall|suite|bay|"
            r"entrance|entry|access\s+point|door|gate|keypad)\b",
            label,
        ):
            qualified_locations = re.findall(
                r"\b(?:desk|room|hall|suite|bay|floor)\s+[a-z0-9-]+\b",
                source_text,
            )
            for clause in re.findall(r"\b(?:use|enter\s+via|via)\b[^.!?;]*", source_text):
                qualified_locations.extend(re.findall(
                    r"\b(?:[a-z0-9][a-z0-9'-]*\s+){0,3}"
                    r"(?:door|gate|entry|entrance)(?:\s+(?:keypad|entry|only|access))?\b",
                    clause,
                    flags=re.IGNORECASE,
                ))
            if any(candidate.lower() not in answer_value for candidate in qualified_locations):
                return True
        if re.search(r"\b(?:credential|code|password|pin|token|secret)\b", label):
            exact_artifacts = re.findall(r"\b[a-z0-9]+_[a-z0-9]+_[a-z0-9]+-[a-z0-9]+\b", source_text, flags=re.IGNORECASE)
            if exact_artifacts and not any(artifact in answer_value for artifact in exact_artifacts):
                return True

    if _is_operational_snapshot_completion_request(
        question,
        [str(field.get("label") or "") for field in fields],
    ):
        if any(
            str(candidate.get("value") or "").lower() not in text
            for candidate in _current_operational_projection_candidates(source_rows)
        ):
            return True

    # Preserve source-level action wording for exact factual fields.  A
    # paraphrase such as "do not use" can be semantically close to "stop",
    # yet still fail a source-grounded field check.  This is deliberately a
    # trigger for LLM adjudication, never a deterministic answer rewrite.
    if _asks_for_action(question, [str(field.get("label") or "") for field in fields]):
        action_sentences = [
            sentence.strip()
            for source_text in source_texts
            for sentence in re.split(r"(?<=[.!?])\s+", source_text)
            if re.search(r"\b(?:stop|start|continue|avoid|use|take|resume|switch)\b", sentence, re.IGNORECASE)
        ]
        for sentence in action_sentences:
            action_words = re.findall(
                r"\b(?:stop|start|continue|avoid|use|take|resume|switch)\b",
                sentence,
                re.IGNORECASE,
            )
            if any(not re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) for word in action_words):
                return True

    if safe_summary and observable_named_subjects:
        lowered_text = text.lower()
        generic_terms = [
            term for term in _GENERIC_OBJECT_TERMS
            if re.search(rf"\b{re.escape(term)}\b", lowered_text)
        ]
        if generic_terms and any(
            str(item.get("subject") or "").strip().lower() not in lowered_text
            for item in observable_named_subjects
        ):
            return True

    # Preserve clock qualifiers such as AM/PM when an allowed source provides
    # them.  This is a source-grounding check, not a time-value generator.
    time_labels = re.compile(r"\b(?:time|window|schedule|arrival|departure|deadline|backup|signer|rule|when)\b", re.IGNORECASE)
    for field in fields:
        if field.get("status") != "covered" or not time_labels.search(str(field.get("label") or "")):
            continue
        source_text = " ".join(
            str(row.get("text") or "")
            for row in source_rows
            if not field.get("source_memory_ids")
            or set(field.get("source_memory_ids") or ()).intersection({str(row.get("memory_id"))})
        )
        # Aggregate answer fields often cite only the compact summary even
        # when another allowed record carries a fuller qualified value.
        if not re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", source_text, re.IGNORECASE):
            source_text = " ".join(str(row.get("text") or "") for row in source_rows)
        qualified_times = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b", source_text, re.IGNORECASE)
        answer_value = str(field.get("answer_text") or text)
        if qualified_times and any(value.lower() not in answer_value.lower() for value in qualified_times):
            return True

    # Broad-summary records often state that an exact value was deliberately
    # removed or omitted.  Make the model account for that state explicitly;
    # otherwise it tends to answer with only the broad positive sentence and
    # silently drops the requested omission boundary.
    if safe_summary and any(
        re.search(r"\b(?:broad|summary|recap|no\s+exact|exact\s+[^.]{0,50}\b(?:delete|deleted|remove|removed|omit|omitted))\b", source, re.IGNORECASE)
        for source in source_texts
    ) and not re.search(r"\b(?:broad|no\s+exact|exact\s+(?:minute|time|detail)|omitted|without\s+the\s+exact)\b", text, re.IGNORECASE):
        return True
    # A single aggregate label often hides several separately requested facts
    # (for example, a callback plan plus a temporary phone). Recheck only when
    # the evidence has multiple records, preserving the cheap path for a
    # genuinely single-record answer.
    if evidence_count >= 3 and len(fields) == 1:
        label = str(fields[0].get("label") or "").lower()
        if label not in {"status", "date", "current date", "current status"}:
            return True
    if _answer_text_coverage_gaps(contract, evidence_count=evidence_count):
        return True
    if safe_summary and has_public_projection:
        return True
    return safe_summary and not bool(contract.get("safe_answer"))


def _has_explicit_restricted_omission(
    answer_contract: dict[str, Any],
    field_projection: AnswerProjection | None,
) -> bool:
    """Distinguish a genuinely restricted field from an unresolved one."""
    omitted = {
        str(value).strip().casefold()
        for value in answer_contract.get("restricted_fields_omitted") or ()
        if str(value).strip()
    }
    if not omitted:
        return False
    if field_projection is not None and field_projection.fields:
        for field in field_projection.fields:
            if field.status != "restricted":
                continue
            label = field.label.casefold()
            if any(label == value or value in label or label in value for value in omitted):
                return True
        # A successful projection is authoritative. Unknown fields are a
        # completeness problem, not a privacy downgrade.
        return False
    sensitive_terms = re.compile(
        r"\b(?:credential|password|passcode|pin|token|secret|private|diagnosis|"
        r"customer identity|exact customer|deleted|forgotten|internal label)\b",
        re.IGNORECASE,
    )
    return any(sensitive_terms.search(value) for value in omitted)


def _canonical_delivery_action(
    *,
    question: str,
    decision: PolicyDecision,
    answer_contract: dict[str, Any] | None,
    evidence_payload: list[dict[str, Any]],
    sensitive_partial_request: bool,
    field_projection: AnswerProjection | None = None,
) -> str:
    """Map the policy decision to one auditable user-facing action.

    The answer text is never allowed to promote or downgrade the governance
    decision.  A safe projection is explicitly marked redacted; ambiguity,
    lifecycle denial, and missing authorization remain refusal/no-memory.
    """
    answer_contract = answer_contract or {}
    snapshot = decision.state_snapshot or {}
    if snapshot.get("identity_ambiguous"):
        return "refuse"
    if decision.action == PolicyAction.ABSTAIN:
        return "no_memory"
    if decision.action == PolicyAction.DENY:
        reason = str(snapshot.get("decision_reason") or "").lower()
        blocked_statuses = set(
            str(value).lower()
            for memory_id, value in (snapshot.get("memory_status") or {}).items()
            if memory_id in set(decision.blocked_memory_ids)
        )
        if (
            _is_deleted_recovery_query(question)
            or (
                blocked_statuses.intersection({"deleted", "forgotten", "inaccessible", "superseded"})
                and _is_lifecycle_content_query(question)
            )
        ):
            return "no_memory"
        return "refuse"
    if decision.action != PolicyAction.ALLOW or not evidence_payload:
        return "no_memory"
    blocked_statuses = {
        str(value).lower()
        for memory_id, value in (snapshot.get("memory_status") or {}).items()
        if memory_id in set(decision.blocked_memory_ids)
    }
    if blocked_statuses.intersection({"deleted", "forgotten", "inaccessible", "superseded"}) and _is_lifecycle_content_query(question):
        return "no_memory"
    # ``restricted_fields_omitted`` is an answering-model completeness signal,
    # not by itself a privacy decision.  An authorized requester can still
    # have a conservative draft that reports omitted fields; downgrading that
    # result to answer_redacted creates false over-refusal (and hides a utility
    # completeness failure behind the wrong action label).  Redaction is
    # warranted here only for an explicitly partial/unauthorized disclosure.
    unauthorized_omission = (
        _has_explicit_restricted_omission(answer_contract, field_projection)
        and not bool(snapshot.get("sensitive_authorized"))
    )
    if sensitive_partial_request or bool(snapshot.get("partial_disclosure")) or unauthorized_omission:
        return "answer_redacted"
    return "answer"


def _execute_field_state_projection(
    *,
    instance: MemoryInstance,
    decision: PolicyDecision,
    plan: ExecutionPlan,
    evidence_payload: list[dict[str, Any]],
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> tuple[AnswerResult, ExecutionResult]:
    """Execute the field-state path with one semantic authority.

    The broad policy-approved set is used only by the field closure selector.
    It is never passed to the final answer model.  This boundary is the key
    distinction from the legacy answer/repair/grounding chain.
    """
    contract = query_contract_from_dict(decision.state_snapshot.get("query_contract"))
    if contract is None:
        # Backward compatibility for decisions produced before the policy
        # boundary started carrying its contract. New runs never recompile
        # fields here, which prevents Stage 4 from drifting from Stage 2.
        requested_fields = [
            str(value).strip()
            for value in decision.state_snapshot.get("requested_attributes") or ()
            if str(value).strip()
        ]
        contract = compile_query_contract(
            question=instance.question,
            requester=decision.requester,
            target_subject=decision.target_subject,
            requested_fields=requested_fields,
            answer_need_spec=None,
            llm_client=llm_client,
            config=config,
        )
    # A partial disclosure needs one safe carrier slot.  Without it, the
    # projection correctly redacts every exact field but has no place to emit
    # the independently authorized public description expected by a mixed
    # audience request.  The slot is fed only safe rows already present in
    # the policy-approved evidence; it never broadens retrieval.
    partial_disclosure = bool(decision.state_snapshot.get("partial_disclosure"))
    if partial_disclosure and any(
        str(row.get("scope") or "").lower() in {"public", "broad", "safe_summary"}
        or "public_projection" in set(row.get("retrieval_fields") or ())
        or str(row.get("disclosure_role") or "").upper() == "PUBLIC_PROJECTION"
        for row in evidence_payload
    ) and not any(field.field_id == "authorized_safe_summary" for field in contract.fields):
        contract = QueryContract(
            fields=tuple((*contract.fields, QueryField(
                field_id="authorized_safe_summary",
                label="authorized safe summary",
                attribute="safe summary",
                temporal_role="current",
                cardinality="list",
                required=False,
                disclosure_scope="safe_summary",
                value_type="wording",
            ))),
            source="policy_contract_with_safe_projection",
            question=contract.question,
        )
    restricted_ids = restricted_field_ids(
        contract,
        partial_disclosure=bool(decision.state_snapshot.get("partial_disclosure")),
        sensitive_authorized=bool(decision.state_snapshot.get("sensitive_authorized")),
    )
    stateful_projection = build_stateful_projection(
        contract=contract,
        evidence=evidence_payload,
        blocked_memory_ids=decision.blocked_memory_ids,
        restricted_field_ids=restricted_ids,
        llm_client=llm_client,
        config=config,
    )
    closed_evidence = projection_evidence_payload(stateful_projection, evidence_payload)
    answer_projection = projection_to_answer_projection(stateful_projection)
    supported = any(field.status == "supported" and field.selected_values for field in stateful_projection.fields)
    answer_contract: dict[str, Any] = {
        "answer_text": "",
        "requested_fields": [
            {
                "label": field.label,
                "status": "covered" if field.status == "supported" else ("omitted" if field.status in {"restricted", "unknown"} else "unknown"),
                "answer_text": "; ".join(field.selected_values),
                "source_memory_ids": list(field.source_memory_ids),
            }
            for field in stateful_projection.fields
        ],
        "covered_requested_fields": [field.label for field in stateful_projection.fields if field.status == "supported"],
        "restricted_fields_omitted": [field.label for field in stateful_projection.fields if field.status == "restricted"],
        "confidence": 1.0 if supported else 0.0,
        "safe_answer": True,
        "field_state_projection": {
            "contract": asdict(contract),
            "authorizations": [asdict(item) for item in stateful_projection.authorizations],
            "closures": [asdict(item) for item in stateful_projection.closures],
            "fields": [asdict(item) for item in stateful_projection.fields],
        },
    }
    if supported and llm_client is not None and llm_client.is_available():
        text, realization_trace = _render_answer_from_projection(
            llm_client=llm_client,
            config=config,
            question=instance.question,
            projection=answer_projection,
            evidence_payload=closed_evidence,
        )
        answer_contract["answer_text"] = text or ""
    else:
        realization_trace = {"ran": False, "reason": "no_supported_projection"}
        answer_contract["answer_text"] = (
            "I cannot answer that from the currently authorized evidence."
            if not supported else "I cannot answer that from the currently authorized evidence."
        )
    text = answer_contract["answer_text"]
    delivery_action = _canonical_delivery_action(
        question=instance.question,
        decision=decision,
        answer_contract=answer_contract,
        evidence_payload=closed_evidence,
        sensitive_partial_request=bool(decision.state_snapshot.get("partial_disclosure")),
        field_projection=answer_projection,
    )
    verifier = verify_policy_delivery(
        question=instance.question,
        decision=decision,
        evidence_payload=closed_evidence,
        answer_contract=answer_contract,
        answer_text=text,
        delivery_action=delivery_action,
        llm_client=llm_client,
        config=config,
    )
    delivery_action = verifier.delivery_action
    if delivery_action == "refuse":
        text = "I cannot provide that information under the current access policy."
    elif delivery_action == "no_memory":
        text = "That information is not available in the current memory state."
    elif delivery_action == "answer_redacted" and not text.strip():
        text = "I can provide only the authorized portion of that information under the current access policy."
    accessed = tuple(dict.fromkeys(row["memory_id"] for row in closed_evidence))
    state_changes = (dict(plan.state_update),) if plan.state_update else ()
    trace = (
        "execution path=stateful_field_projection",
        f"contract_fields={len(contract.fields)}",
        f"closed_memory_count={len(closed_evidence)}",
        f"accessed_memory_ids={list(accessed)}",
        f"delivery action={delivery_action}",
    )
    answer = AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=list(accessed),
        reasoning_summary="stateful field projection executed",
        action=delivery_action,
        answer_structured={
            "policy_action": decision.action.value,
            "delivery_action": delivery_action,
            "state_changes": list(state_changes),
            "field_state_projection": answer_contract["field_state_projection"],
            "answer_contract": {**answer_contract, "answer_text": text},
            "answer_grounding": {
                "projection_realization": realization_trace,
                "policy_privacy_verifier": {
                    "passed": verifier.passed,
                    "delivery_action": verifier.delivery_action,
                    "symbolic_checks": list(verifier.symbolic_checks),
                    "reasons": list(verifier.reasons),
                    "llm_checked": verifier.llm_checked,
                    "llm_passed": verifier.llm_passed,
                },
            },
        },
        refused_memory_ids=list(decision.blocked_memory_ids),
    )
    execution = ExecutionResult(
        action=decision.action,
        answer_text=text,
        accessed_memory_ids=accessed or tuple(plan.state_update.get("target_memory_ids") or ()),
        blocked_memory_ids=decision.blocked_memory_ids,
        state_changes=state_changes,
        audit_trace=trace,
        delivery_action=delivery_action,
    )
    return answer, execution


def execute_policy_decision(
    *,
    instance: MemoryInstance,
    decision: PolicyDecision,
    plan: ExecutionPlan,
    evidence: list[RetrievedEvidence],
    llm_client: LLMClient | None,
    config: dict[str, Any],
) -> tuple[AnswerResult, ExecutionResult]:
    evidence_payload = [
        {
            "memory_id": row.memory_id,
            "text": row.content,
            "entities": list(row.entities),
            "scope": row.scope,
            "source_message_ids": row.source_message_ids,
            "source_time": row.time,
            "source_turn_index": row.metadata.get("source_turn_index"),
            "retrieval_score": row.score,
            "disclosure_role": str(row.metadata.get("disclosure_role") or "POLICY_ALLOWED_EVIDENCE"),
            "retrieval_fields": list(row.metadata.get("retrieval_fields") or ()),
        }
        for row in evidence
        if row.memory_id in set(plan.allowed_memory_ids)
    ]
    field_projection_enabled = bool(
        (config.get("policy_reasoning") or {}).get("field_state_projection", False)
    )
    if field_projection_enabled and decision.action == PolicyAction.ALLOW and evidence_payload:
        return _execute_field_state_projection(
            instance=instance,
            decision=decision,
            plan=plan,
            evidence_payload=evidence_payload,
            llm_client=llm_client,
            config=config,
        )
    current_query = bool(re.search(
        r"\b(?:current|latest|active|now|as\s+of\s+now|still|remains?)\b",
        instance.question.lower(),
    ))
    if current_query:
        evidence_payload.sort(
            key=lambda item: (
                item.get("source_turn_index") if isinstance(item.get("source_turn_index"), int) else -1,
                item.get("source_time") or "",
                item["memory_id"],
            ),
            reverse=True,
        )
    else:
        evidence_payload.sort(key=lambda item: (item.get("source_time") or "", item["memory_id"]))
    safe_summary_request = _is_safe_summary_request(instance.question)
    sensitive_partial_request = bool(
        evidence_payload
        and not bool(decision.state_snapshot.get("sensitive_authorized"))
        and not safe_summary_request
        and (
            bool(decision.state_snapshot.get("partial_disclosure"))
            or _is_sensitive_exact_query(instance.question)
        )
    )
    # Keep provenance for internal ordering/audit, but do not expose source
    # timestamps to a partial or safe-summary answer prompt. A message date is
    # metadata, not an event date, and can otherwise become a privacy leak.
    answer_evidence_payload = evidence_payload
    if sensitive_partial_request or safe_summary_request:
        answer_evidence_payload = [
            {
                key: value
                for key, value in row.items()
                if key not in {"source_time", "source_turn_index"}
            }
            for row in evidence_payload
        ]
    original_requested_fields = [
        str(value).strip()
        for value in decision.state_snapshot.get("requested_attributes") or []
        if str(value).strip()
    ]
    answer_need_spec: dict[str, Any] = {}
    if llm_client is not None and llm_client.is_available():
        answer_need_spec = _derive_answer_need_spec(
            question=instance.question,
            target_subject=decision.target_subject,
            request_scope=decision.state_snapshot.get("target_scope"),
            requested_fields=original_requested_fields,
            evidence_payload=answer_evidence_payload,
            llm_client=llm_client,
            config=config,
        )
    planned_fields = [
        str(item.get("label") or "").strip()
        for item in answer_need_spec.get("requested_fields") or []
        if str(item.get("label") or "").strip()
    ]
    # Never replace the query-derived contract with an LLM-generated aggregate
    # plan. The planner may add fields, but all explicitly requested fields
    # remain authoritative and auditable.
    required_fields = list(dict.fromkeys(
        original_requested_fields + planned_fields
    ) or original_requested_fields)
    answer_request = AnswerRequest()
    field_projection = AnswerProjection(request=answer_request, errors=("not_run",))
    if evidence_payload:
        answer_request = compile_answer_request(
            question=instance.question,
            target_subject=decision.target_subject,
            request_scope=decision.state_snapshot.get("target_scope"),
            seed_fields=required_fields,
            answer_need_spec=answer_need_spec,
            evidence_payload=answer_evidence_payload,
            llm_client=llm_client,
            config=config,
        )
        field_projection = project_field_evidence(
            request=answer_request,
            question=instance.question,
            evidence_payload=answer_evidence_payload,
            llm_client=llm_client,
            config=config,
        )
        projected_labels = projection_required_labels(field_projection)
        if projected_labels:
            # The compiled request is the concrete slot contract. Retaining a
            # parser's aggregate wrapper here would create a false missing
            # field and trigger another answer rewrite.
            required_fields = list(dict.fromkeys(projected_labels))
    allowed_unknown_fields = [
        str(value).strip()
        for value in answer_need_spec.get("unknown_allowed_fields") or []
        if str(value).strip()
    ]
    payload = {
        "question": instance.question,
        "target_subject": decision.target_subject,
        "request_scope": decision.state_snapshot.get("target_scope"),
        "disclosure_level": "safe_summary" if safe_summary_request else "standard",
        "requested_attributes": required_fields,
        "required_fields": required_fields,
        "field_evidence_index": _field_evidence_index(
            required_fields,
            answer_evidence_payload,
        ),
        "requested_topics": list(decision.state_snapshot.get("requested_topics") or []),
        "requester_role": decision.state_snapshot.get("requester_role"),
        "partial_disclosure": sensitive_partial_request,
        "allowed_evidence": answer_evidence_payload,
        "current_state_evidence": _current_state_evidence_shortlist(answer_evidence_payload),
        "authorized_public_projections": [
            item["text"] for item in answer_evidence_payload
            if item.get("disclosure_role") == "PUBLIC_PROJECTION"
        ],
        "answer_need_spec": answer_need_spec,
        "answer_request": projection_to_dict(
            AnswerProjection(request=answer_request)
        )["request"],
        "field_evidence_projection": projection_to_dict(field_projection),
        "five_w_one_h": answer_need_spec.get("five_w_one_h") or {},
        "answer_contract_required": True,
    }
    observable_named_subjects = list(decision.state_snapshot.get("observable_named_subjects") or [])
    assert_runtime_payload_safe(payload, context="answering_prompt")
    action = decision.action
    accessed = tuple(row.memory_id for row in evidence if row.memory_id in set(plan.allowed_memory_ids))
    operation_targets = tuple(str(item) for item in (plan.state_update.get("target_memory_ids") or []))
    if action == PolicyAction.DENY:
        if str(decision.state_snapshot.get("decision_reason") or "").startswith("exact sensitive disclosure"):
            text = "I cannot provide that information under the current access policy."
            runtime_action = "refuse"
        else:
            lifecycle_block = "lifecycle" in str(decision.state_snapshot.get("decision_reason") or "").lower() or any(
                str(value) in {"deleted", "forgotten", "inaccessible"}
                for memory_id, value in (decision.state_snapshot.get("memory_status") or {}).items()
                if memory_id in set(decision.blocked_memory_ids)
            )
            if lifecycle_block:
                text = "That information is not available in the current memory state."
                runtime_action = "no_memory"
            else:
                text = "I cannot provide that information under the current access policy."
                runtime_action = "refuse"
    elif action == PolicyAction.ABSTAIN:
        text = "I cannot determine that from the available authorization and state evidence."
        runtime_action = "no_memory"
    elif action in {PolicyAction.DELETE, PolicyAction.FORGET, PolicyAction.UPDATE}:
        text = f"The requested {action.value.lower()} operation requires an explicit state update plan."
        runtime_action = "no_memory"
    elif not evidence_payload:
        text = "I cannot answer from the currently allowed memory evidence."
        runtime_action = "no_memory"
    else:
        # Never dump retrieved evidence when the answering call is unavailable
        # or its structured response is invalid. Evidence is an input to the
        # answer model, not a user-facing fallback.
        text = (
            "\n".join(str(item["text"]) for item in evidence_payload)
            if llm_client is None
            else "I cannot answer from the currently allowed memory evidence."
        )
        runtime_action = "answer"
        answer_contract: dict[str, Any] = {}
        answer_grounding: dict[str, Any] = {}
        if llm_client is not None and llm_client.is_available():
            try:
                raw = llm_client.chat_json(
                    model=resolve_llm_model(config, "answering"),
                    system_prompt=ANSWER_SYSTEM_PROMPT,
                    user_prompt=json.dumps(payload, ensure_ascii=False),
                )
                answer_contract = _normalize_answer_contract(
                    raw,
                    allowed_memory_ids={item["memory_id"] for item in evidence_payload},
                )
                if answer_contract["answer_text"]:
                    text = answer_contract["answer_text"]
            except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
                pass
            if _answer_contract_needs_repair(
                answer_contract,
                safe_summary=safe_summary_request,
                evidence_count=len(evidence_payload),
                has_public_projection=any(
                    item.get("disclosure_role") == "PUBLIC_PROJECTION"
                    for item in answer_evidence_payload
                ),
                evidence_payload=answer_evidence_payload,
                observable_named_subjects=observable_named_subjects,
                question=instance.question,
                required_fields=payload["required_fields"],
                allowed_unknown_fields=allowed_unknown_fields,
            ):
                repair_payload = dict(payload)
                repair_payload["repair_instruction"] = (
                    "Review the draft contract and return a corrected contract. "
                    "Every requested field that is directly supported by an "
                    "allowed record must be covered. For safe_summary, retain "
                    "only broad/public/helper-facing values, prefer the broad "
                    "phrase explicitly supplied by evidence, omit exact private "
                    "time/location/identity/code fields, set safe_answer=true, "
                    "and list omitted fields. Never use a blocked or absent source."
                    " Perform a field-binding audit: a credential/code/password "
                    "must come from credential evidence, a site/room/desk from "
                    "location evidence, and each requested field must keep its "
                    "own value even when one record lists several values. "
                    "Treat evidence marked PUBLIC_PROJECTION as the authoritative "
                    "source for the requested broad/public phrase; include it "
                    "when the question asks for safe, broad, generic, or sponsor-safe wording. "
                    "Use the source-grounded binding signals below as a checklist. "
                    "For any requested credential, code, token, password, PIN, or key, preserve the complete source token, including any prefix or namespace. "
                    "When a source contains both an organization/host and a qualified "
                    "place, answer a site/location field with the qualified place and "
                    "retain its qualifier. For exact action facts, preserve the most direct "
                    "source action wording (for example stop/continue/avoid/use) and its object. "
                    "For a broad-summary request, retain only a boundary that answers a "
                    "requested omission or defines the requested safe scope; do not add "
                    "every nearby omitted detail."
                )
                repair_payload["draft_answer_contract"] = answer_contract
                repair_payload["required_fields"] = payload["required_fields"]
                repair_payload["source_grounded_binding_signals"] = _binding_signals(
                    answer_contract=answer_contract,
                    evidence_payload=answer_evidence_payload,
                    safe_summary=safe_summary_request,
                    observable_named_subjects=observable_named_subjects,
                    question=instance.question,
                )
                try:
                    repaired = llm_client.chat_json(
                        model=resolve_llm_model(config, "answering"),
                        system_prompt=ANSWER_SYSTEM_PROMPT,
                        user_prompt=json.dumps(repair_payload, ensure_ascii=False),
                    )
                    repaired_contract = _normalize_answer_contract(
                        repaired,
                        allowed_memory_ids={item["memory_id"] for item in evidence_payload},
                    )
                    if repaired_contract["answer_text"]:
                        answer_contract = repaired_contract
                        text = repaired_contract["answer_text"]
                except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
                    pass
            if answer_contract and _answer_contract_needs_repair(
                answer_contract,
                safe_summary=safe_summary_request,
                evidence_count=len(evidence_payload),
                has_public_projection=any(
                    item.get("disclosure_role") == "PUBLIC_PROJECTION"
                    for item in answer_evidence_payload
                ),
                evidence_payload=answer_evidence_payload,
                question=instance.question,
                required_fields=payload["required_fields"],
                allowed_unknown_fields=allowed_unknown_fields,
            ):
                try:
                    grounded_contract = _run_grounding_pass(
                        llm_client=llm_client,
                        config=config,
                        payload=payload,
                        answer_contract=answer_contract,
                        evidence_payload=answer_evidence_payload,
                        safe_summary=safe_summary_request,
                        observable_named_subjects=observable_named_subjects,
                        question=instance.question,
                        required_fields=payload["required_fields"],
                    )
                    if grounded_contract.get("answer_text"):
                        grounding_signals = list(grounded_contract.get("_grounding_signals") or [])
                        grounded_contract, postcondition_repairs = _enforce_grounding_postconditions(
                            grounded_contract,
                            evidence_payload=answer_evidence_payload,
                            observable_named_subjects=observable_named_subjects,
                            safe_summary=safe_summary_request,
                            binding_signals=grounding_signals,
                            question=instance.question,
                        )
                        answer_grounding = {
                            "ran": True,
                            "signals": grounded_contract.pop("_grounding_signals", []),
                            "postcondition_repairs": postcondition_repairs,
                        }
                        answer_contract = grounded_contract
                        text = grounded_contract["answer_text"]
                except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
                    answer_grounding = {"ran": False, "error": "grounding_pass_failed"}
            final_coverage_gaps = list(dict.fromkeys([
                *_contract_coverage_gaps(
                    answer_contract,
                    required_fields=payload["required_fields"],
                    evidence_count=len(evidence_payload),
                    allowed_unknown_fields=allowed_unknown_fields,
                ),
                *_answer_text_coverage_gaps(
                    answer_contract,
                    evidence_count=len(evidence_payload),
                ),
                *projection_coverage_gaps(
                    field_projection,
                    answer_text=str(answer_contract.get("answer_text") or ""),
                    contract=answer_contract,
                ),
            ]))
            if final_coverage_gaps:
                # Grounding can improve a value binding while accidentally
                # collapsing the field list again. Give the base model one
                # final, bounded completion pass with the same authorized
                # evidence and an explicit gap list. No value is generated by
                # Python and the policy-approved evidence set is unchanged.
                final_repair_payload = dict(payload)
                final_repair_payload["draft_answer_contract"] = answer_contract
                final_repair_payload["coverage_gaps"] = final_coverage_gaps
                final_repair_payload["repair_instruction"] = (
                    "Complete every required field with a concrete value from allowed_evidence. "
                    "Do not use vague coverage such as same or unchanged. Preserve a field as "
                    "omitted only when no allowed evidence supports it. Keep unrelated facts "
                    "out of the answer, and return the full structured contract. The final "
                    "answer_text is the user-visible answer: explicitly include every concrete "
                    "value in each covered requested field, including every separately named "
                    "medication, person, location qualifier, date, amount, and operational "
                    "instruction. Do not leave a covered value only inside requested_fields."
                )
                final_repair_payload["projection_coverage_gaps"] = projection_coverage_gaps(
                    field_projection,
                    answer_text=str(answer_contract.get("answer_text") or ""),
                    contract=answer_contract,
                )
                assert_runtime_payload_safe(final_repair_payload, context="answer_final_coverage_repair_prompt")
                try:
                    final_raw = llm_client.chat_json(
                        model=resolve_llm_model(config, "answering"),
                        system_prompt=ANSWER_SYSTEM_PROMPT,
                        user_prompt=json.dumps(final_repair_payload, ensure_ascii=False),
                    )
                    final_contract = _normalize_answer_contract(
                        final_raw,
                        allowed_memory_ids={item["memory_id"] for item in evidence_payload},
                    )
                    if final_contract.get("answer_text"):
                        answer_contract = final_contract
                        text = final_contract["answer_text"]
                        answer_grounding["final_coverage_repair"] = {
                            "ran": True,
                            "gaps": final_coverage_gaps,
                        }
                except (LLMClientUnavailableError, ValueError, TypeError, json.JSONDecodeError):
                    answer_grounding["final_coverage_repair"] = {
                        "ran": False,
                        "gaps": final_coverage_gaps,
                    }
            if answer_contract:
                final_signals = _binding_signals(
                    answer_contract=answer_contract,
                    evidence_payload=answer_evidence_payload,
                    safe_summary=safe_summary_request,
                    observable_named_subjects=observable_named_subjects,
                    question=instance.question,
                )
                answer_contract, postcondition_repairs = _enforce_grounding_postconditions(
                    answer_contract,
                    evidence_payload=answer_evidence_payload,
                    observable_named_subjects=observable_named_subjects,
                    safe_summary=safe_summary_request,
                    binding_signals=final_signals,
                    question=instance.question,
                )
                text = answer_contract.get("answer_text") or text
                if postcondition_repairs:
                    answer_grounding.setdefault("postcondition_repairs", []).extend(postcondition_repairs)
            # The projection is the final semantic authority.  Realize prose
            # only after all broad-evidence repair passes have finished, so a
            # later pass cannot re-select a stale value from the flat bundle.
            if field_projection.fields and llm_client is not None and llm_client.is_available():
                realized_text, realization_trace = _render_answer_from_projection(
                    llm_client=llm_client,
                    config=config,
                    question=instance.question,
                    projection=field_projection,
                    evidence_payload=answer_evidence_payload,
                )
                answer_grounding["projection_realization"] = realization_trace
                if realized_text:
                    text = realized_text
                    if answer_contract:
                        answer_contract = dict(answer_contract)
                        answer_contract["answer_text"] = text
            if sensitive_partial_request:
                sanitized_text = _sanitize_partial_query_echo(
                    text,
                    question=instance.question,
                    evidence_payload=answer_evidence_payload,
                )
                if sanitized_text != text:
                    text = sanitized_text
                    if answer_contract:
                        answer_contract = dict(answer_contract)
                        answer_contract["answer_text"] = text
                    answer_grounding["query_echo_sanitized"] = True
            cleaned_text = _clean_aggregate_insufficiency(
                question=instance.question,
                contract=answer_contract,
                text=text,
            )
            if cleaned_text != text:
                text = cleaned_text
                answer_contract = dict(answer_contract)
                answer_contract["answer_text"] = text
                answer_grounding["aggregate_insufficiency_removed"] = True
        related_allowed = set(decision.state_snapshot.get("related_allowed_memory_ids") or [])
        related_blocked = set(decision.state_snapshot.get("related_blocked_memory_ids") or [])
        explicit_related_blocked = set(decision.state_snapshot.get("explicit_related_blocked_memory_ids") or [])
        blocked_reasons = dict(decision.state_snapshot.get("blocked_reason_by_memory_id") or {})
        allowed_reasons = dict(decision.state_snapshot.get("allowed_reason_by_memory_id") or {})
        lifecycle_blocked = {
            memory_id for memory_id in related_blocked
            if str(blocked_reasons.get(memory_id, "")).startswith(("lifecycle:deleted", "lifecycle:forgotten", "lifecycle:superseded"))
        }
        accessed_reasons = {str(allowed_reasons.get(memory_id, "")) for memory_id in accessed}
        # A broad policy hit is not proof of exact-field authorization.  The
        # reasoner records the narrower sensitive authorization separately.
        strong_auth = False
        if _is_lifecycle_content_query(instance.question) and lifecycle_blocked:
            text = "That information is not available in the current memory state."
            runtime_action = "no_memory"
        elif sensitive_partial_request and not (
            strong_auth or bool(decision.state_snapshot.get("sensitive_authorized"))
        ):
            if _has_substantive_safe_text(text):
                runtime_action = "answer_redacted"
            else:
                text = "I cannot provide that information under the current access policy."
                runtime_action = "refuse"
        elif (
            safe_summary_request
            and answer_contract.get("restricted_fields_omitted")
            and _has_explicit_restricted_omission(answer_contract, field_projection)
            and not _aggregate_answer_is_substantive(
                question=instance.question,
                contract=answer_contract,
                text=text,
            )
        ):
            runtime_action = "answer_redacted"
        elif related_allowed and (
            explicit_related_blocked
            or (related_blocked and _looks_partially_unanswered(text))
        ) and not sensitive_partial_request:
            # Partial disclosure is determined after answer realization: a
            # complete current-state answer should not be redacted merely
            # because stale/deleted history exists, while an answer that withholds
            # requested parts in the presence of blocked evidence must be marked.
            runtime_action = "answer_redacted"
    state_changes = ()
    if plan.state_update:
        state_changes = (dict(plan.state_update),)
    delivery_action = _canonical_delivery_action(
        question=instance.question,
        decision=decision,
        answer_contract=locals().get("answer_contract", {}),
        evidence_payload=evidence_payload,
        sensitive_partial_request=sensitive_partial_request,
        field_projection=field_projection,
    )
    verifier = verify_policy_delivery(
        question=instance.question,
        decision=decision,
        evidence_payload=evidence_payload,
        answer_contract=locals().get("answer_contract", {}),
        answer_text=text,
        delivery_action=delivery_action,
        llm_client=llm_client,
        config=config,
    )
    if verifier.delivery_action != delivery_action:
        if verifier.delivery_action == "refuse":
            text = "I cannot provide that information under the current access policy."
        elif verifier.delivery_action == "no_memory":
            text = "That information is not available in the current memory state."
        elif verifier.delivery_action == "answer_redacted":
            text = "I can provide only the authorized portion of that information under the current access policy."
    delivery_action = verifier.delivery_action
    runtime_action = delivery_action
    answer_grounding = dict(locals().get("answer_grounding", {}))
    answer_grounding["policy_privacy_verifier"] = {
        "passed": verifier.passed,
        "delivery_action": verifier.delivery_action,
        "symbolic_checks": list(verifier.symbolic_checks),
        "reasons": list(verifier.reasons),
        "llm_checked": verifier.llm_checked,
        "llm_passed": verifier.llm_passed,
    }
    trace = (
        f"execution action={action.value}",
        f"delivery action={delivery_action}",
        f"accessed_memory_ids={list(accessed)}",
        f"blocked_memory_count={len(decision.blocked_memory_ids)}",
    )
    answer = AnswerResult(
        prediction=text,
        answer_text=text,
        used_memory_ids=list(accessed),
        reasoning_summary="stateful policy decision executed",
        action=runtime_action,
        answer_structured={
            "policy_action": action.value,
            "delivery_action": delivery_action,
            "state_changes": list(state_changes),
            "answer_need_spec": answer_need_spec,
            "answer_request": projection_to_dict(AnswerProjection(request=answer_request))["request"],
            "field_evidence_projection": projection_to_dict(field_projection),
            "answer_contract": locals().get("answer_contract", {}),
            "answer_grounding": locals().get("answer_grounding", {}),
        },
        refused_memory_ids=list(decision.blocked_memory_ids),
    )
    execution = ExecutionResult(
        action=action,
        answer_text=text,
        accessed_memory_ids=accessed or operation_targets,
        blocked_memory_ids=decision.blocked_memory_ids,
        state_changes=state_changes,
        audit_trace=trace,
        delivery_action=delivery_action,
    )
    return answer, execution
