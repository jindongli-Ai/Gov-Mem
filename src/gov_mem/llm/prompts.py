from __future__ import annotations

from textwrap import dedent


MEMORY_INGESTION_SYSTEM_PROMPT = dedent(
    """
    You are a memory extraction engine for a multi-party agent memory system.
    Extract long-term useful structured memory from observable messages only.
    Do not infer hidden labels, oracle answers, or benchmark targets.
    Also identify access control, sensitivity, redaction needs, and forgetting/update signals.
    """
).strip()


QUERY_PLANNER_SYSTEM_PROMPT = dedent(
    """
    You are a query planner for a memory-reasoning system.
    Produce a reasoning-aware retrieval plan from the user question and visible metadata.
    Do not answer the question directly.
    """
).strip()


ANSWERING_SYSTEM_PROMPT = dedent(
    """
    You are the final answering component of Gov-Mem.
    Use only the provided evidence and reasoning summary.
    Do not invent missing facts.
    Return compact JSON only.
    """
).strip()


SKILL_UPDATE_SYSTEM_PROMPT = dedent(
    """
    You summarize repeated failure lessons into concise reusable skill instructions.
    """
).strip()


def build_memory_ingestion_user_prompt(messages: list[dict], skill_text: str) -> str:
    return dedent(
        f"""
        Skill instructions:
        {skill_text or "None"}

        Extract a JSON object with field "memory_items", where each item contains:
        user_id, scope, content, memory_type, entities, time, source_message_ids,
        confidence, privacy_level, tags, memory_status, metadata.

        In metadata, include when possible:
        access_scope, authorized_users, forbidden_users, forget_after,
        is_deleted, redaction_required, sensitive_entities, supersedes_memory_ids.

        Messages:
        {messages}
        """
    ).strip()


def build_query_planner_user_prompt(
    *,
    question: str,
    asking_user_id: str | None,
    choices: list[str] | None,
    observable_metadata: dict,
    skill_text: str,
    retrieved_lessons: list[str],
) -> str:
    return dedent(
        f"""
        Skill instructions:
        {skill_text or "None"}

        Prior lessons:
        {retrieved_lessons or []}

        Return JSON with fields:
        query_type, target_users, target_entities, required_memory_types,
        symbolic_filters, dense_queries, reasoning_ops, semantic_spec.

        semantic_spec must be an object with:
        - requested_slots: zero or more canonical snake_case evidence
          attributes. This is an open schema: use a new attribute name when
          the question requests a property not covered by an existing name.
        - requested_attributes: canonical snake_case property names for every
          explicitly requested field, including fields also represented by requested_slots. This
          is an open, domain-independent attribute set; preserve distinctions
          between different requested properties and do not invent fields.
        - attribute_bindings: one object per requested attribute with fields
          attribute, support_span, semantic_role, evidence_slot_hint, and
          need_kind. need_kind is scalar or record_collection. A collection
          request asks for one or more complete observed records and must not
          be represented as an invented scalar slot. A request for a named
          recap, summary, status entry, reminder, update, or record uses
          record_collection when the observed fields jointly express that
          named item, even if the wording is singular.
          evidence_slot_hint is an optional canonical snake_case name for the
          intrinsic evidence field expected to answer that attribute; it is a
          retrieval hint, not an answer or an authorization. support_span must be
          copied verbatim from the question. semantic_role must be
          requested_property; spans whose role is target_entity,
          request_object, or temporal_modifier are not requested attributes.
          support_span must be the minimal property phrase, never an entire
          question sentence or a multi-property conjunction.
          Urgency, deadline, meeting/review purpose, audience, and other
          discourse context explain why the requester asks; they are not
          requested properties unless the question explicitly asks for their
          value. Do not create an evidence attribute for such context.
        - disclosure_constraints: zero or more objects with constraint_kind
          (temporal_access_boundary, access_boundary, redaction_boundary, or
          exclusion) and support_span copied verbatim from the question. These
          constrain what may be disclosed, but are never answer attributes or
          evidence slots. Put phrases such as authorization time windows and
          "only"-style limits here rather than requested_attributes.
        - temporal_scope: one of [current, historical, comparison, unspecified]
        - disclosure_scope: one of [full, redacted, public_only, unspecified]
        - state_domain: one of [project, research, operational, unspecified]
        - request_shape: one of [fact, list, plan, policy, comparison, mixed]
        - requires_entity_resolution: boolean

        Infer these fields from meaning, not from a fixed phrase list. Use only
        the question and observable metadata; never infer an answer.

        Question: {question}
        Asking user id: {asking_user_id}
        Choices: {choices}
        Observable metadata: {observable_metadata}
        """
    ).strip()


def build_semantic_spec_repair_prompt(
    *,
    question: str,
    asking_user_id: str | None,
    observable_metadata: dict,
) -> str:
    """Request only the missing semantic contract, never an answer."""
    return dedent(
        f"""
        Return JSON only with one field, semantic_spec. Do not answer the
        question and do not infer any fact from it.

        semantic_spec must be an object with every field below:
        - requested_slots: zero or more canonical snake_case evidence
          attributes. This is an open schema: use a new attribute name when
          the question requests a property not covered by an existing name.
        - requested_attributes: canonical snake_case property names for every
          explicitly requested field, including fields also represented by requested_slots. This
          is an open, domain-independent attribute set; preserve distinctions
          between different requested properties and do not invent fields.
        - attribute_bindings: one object per requested attribute with fields
          attribute, support_span, semantic_role, evidence_slot_hint, and
          need_kind (scalar or record_collection).
          evidence_slot_hint is an optional canonical snake_case intrinsic
          evidence field used only for retrieval routing, never authorization.
          support_span must be
          copied verbatim from the question. semantic_role must be
          requested_property; spans whose role is target_entity,
          request_object, or temporal_modifier are not requested attributes.
          Each support_span must be the minimal property phrase, never an
          entire question sentence or a multi-property conjunction.
          Urgency, deadline, meeting/review purpose, audience, and other
          discourse context are not requested properties unless their value is
          explicitly requested. Do not create evidence attributes for them.
        - disclosure_constraints: zero or more objects with constraint_kind
          (temporal_access_boundary, access_boundary, redaction_boundary, or
          exclusion) and a verbatim support_span. Constraints are not answer
          attributes and must not be mapped to evidence slots.
        - temporal_scope: one of [current, historical, comparison, unspecified]
        - disclosure_scope: one of [full, redacted, public_only, unspecified]
        - state_domain: one of [project, research, operational, unspecified]
        - request_shape: one of [fact, list, plan, policy, comparison, mixed]
        - requires_entity_resolution: boolean

        Infer the contract from meaning, including indirect wording. Use only
        the question and observable metadata; never use hidden labels or an
        expected answer.

        Question: {question}
        Asking user id: {asking_user_id}
        Observable metadata: {observable_metadata}
        """
    ).strip()


def build_semantic_contract_audit_prompt(
    *,
    question: str,
    asking_user_id: str | None,
    observable_metadata: dict,
    target_entities: list[str],
) -> str:
    """Request an independent role audit, not a second answer plan."""
    return dedent(
        f"""
        Return JSON only with one field, semantic_spec. You are auditing the
        minimal semantic contract of a question, not answering it.

        First separate every relevant query span into exactly one of these
        roles: target_entity, requested_property, request_object,
        temporal_modifier, or disclosure_constraint. A name/person/group or
        the thing being discussed is a target_entity/request_object, never a
        requested_property. A phrase that limits who may receive information,
        what may be released, or the time range of authorization is a
        disclosure_constraint, never a requested_property or evidence slot.
        A requested_property is only information whose value must appear in an
        answer. When the question asks for a set of entries/details/steps,
        represent it as one requested_property with need_kind
        record_collection rather than inventing a synthetic scalar field.
        Treat a named recap, summary, status entry, reminder, update, or
        record the same way when its observed fields jointly express the
        requested item, even if the wording is singular.
        An urgency, deadline, meeting/review purpose, audience, justification,
        or other discourse context is neither a requested_property nor a
        disclosure constraint unless the question explicitly asks for its
        value. Do not turn "the review is soon" into a review attribute when
        the question asks for a budget or another fact.
        In "schedule after the injection visit", "schedule" is the requested
        property; "after the injection visit" is a temporal boundary, and the
        injection visit is not another requested property.
        Before responding, enumerate every independently requested answer or
        inference in the question. Coordinated requests, conditional requests,
        yes/no requests, and requests for an interpretation are separate
        requested_properties when each asks for a distinct answer, even if one
        will later be denied or cannot be grounded in memory. Never silently
        drop such a property because it is sensitive, speculative, or unlikely
        to be answerable; privacy is decided after this contract.

        semantic_spec must contain:
        - requested_slots and requested_attributes: only requested properties;
          open-schema snake_case names are allowed.
        - attribute_bindings: attribute, support_span (verbatim minimal query
          span), semantic_role=requested_property, evidence_slot_hint, and
          need_kind (scalar or record_collection).
        - disclosure_constraints: objects with constraint_kind
          (temporal_access_boundary, access_boundary, redaction_boundary, or
          exclusion) and verbatim support_span. Do not repeat these in
          requested_attributes or attribute_bindings.
        - temporal_scope: current, historical, comparison, or unspecified.
        - disclosure_scope: full, redacted, public_only, or unspecified.
        - state_domain: project, research, operational, or unspecified.
        - request_shape: fact, list, plan, policy, comparison, or mixed.
        - requires_entity_resolution: boolean.

        Never infer an answer, policy, or hidden label. Use only the supplied
        question and observable metadata.

        Question: {question}
        Asking user id: {asking_user_id}
        Candidate target entities from an independent planner: {target_entities}
        Observable metadata: {observable_metadata}
        """
    ).strip()


def build_semantic_contract_verification_prompt(
    *,
    question: str,
    asking_user_id: str | None,
    observable_metadata: dict,
    target_entities: list[str],
    candidate_spec: dict,
) -> str:
    return dedent(
        f"""
        Return JSON only with one corrected semantic_spec. You verify a query
        contract before a privacy certificate uses it; do not answer the
        question.

        Reject and correct these category errors: a person/name/request object
        represented as a requested property; a time, authorization, scope,
        redaction, exclusion, or other disclosure limitation represented as a
        requested property; and a collection request represented as an invented
        scalar slot. A requested-property support_span must be the smallest
        verbatim phrase naming information to return. Any qualifier that limits
        permission or authorized range must be removed from that span and added
        as disclosure_constraints with a verbatim support_span. Do not invent
        properties, values, constraints, or hidden policy.
        An urgency, deadline, meeting/review purpose, audience, justification,
        or other discourse context is not a requested property unless its
        value is explicitly requested; remove it from requested_attributes and
        do not create an evidence slot for it.
        For an ordered request such as "schedule after the injection visit",
        retain only "schedule" as a requested property and classify the full
        "after the injection visit" phrase as a temporal boundary.
        A named recap, summary, status entry, reminder, update, or record is
        record_collection when its observed fields jointly constitute the
        requested item; do not reduce it to one representative scalar merely
        because the question uses singular grammar.
        Preserve every independently requested answer or inference from the
        original question. Do not delete a sensitive, speculative, conditional,
        or yes/no requested property merely because it may later be denied. It
        still needs its own minimal verbatim requested-property support_span so
        downstream governance can redact it instead of treating a partial
        answer as the complete request.
        When a requested-property phrase coordinates multiple noun phrases with
        "and", commas, or another list separator, split it into one binding per
        independently answerable property and preserve each exact minimal
        surface span. Do not collapse two properties into one synthetic field
        merely because the question uses a shared qualifier such as "current"
        or "revised". A surrounding record/state request may remain a
        record_collection, but its explicitly enumerated fields must still be
        represented separately.

        Return semantic_spec with requested_slots, requested_attributes,
        attribute_bindings (attribute, support_span, semantic_role,
        evidence_slot_hint, need_kind), disclosure_constraints
        (constraint_kind, support_span), temporal_scope, disclosure_scope,
        state_domain, request_shape, and requires_entity_resolution.

        Question: {question}
        Asking user id: {asking_user_id}
        Candidate target entities: {target_entities}
        Candidate contract: {candidate_spec}
        Observable metadata: {observable_metadata}
        """
    ).strip()


def build_answering_user_prompt(
    *,
    question: str,
    asking_user_id: str | None,
    choices: list[str] | None,
    selected_evidence: list[dict],
    reasoning_trace: list[str],
    conclusion_hint: str | None,
    skill_text: str,
    retrieved_lessons: list[str],
) -> str:
    return dedent(
        f"""
        Skill instructions:
        {skill_text or "None"}

        Prior lessons:
        {retrieved_lessons or []}

        Return JSON only.
        Required fields:
        - prediction
        - answer_text
        - used_memory_ids
        - reasoning_summary
        - action

        Valid action values:
        answer, answer_redacted, refuse, no_memory

        Question: {question}
        Asking user id: {asking_user_id}
        Choices: {choices}
        Selected evidence: {selected_evidence}
        Reasoning trace: {reasoning_trace}
        Conclusion hint: {conclusion_hint}
        """
    ).strip()


def build_skill_update_user_prompt(stage: str, failure_lessons: list[str]) -> str:
    return dedent(
        f"""
        Stage: {stage}
        Failure lessons:
        {failure_lessons}

        Return JSON with:
        instruction, summary
        """
    ).strip()
