from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import MemoryInstance, QueryPlan
from gov_mem.experience.experience_bank import ExperienceBank
from gov_mem.governance_runtime.leakage_guard import strip_hidden_eval_fields
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_semantic_contract_audit_prompt,
    build_semantic_contract_verification_prompt,
    build_semantic_spec_repair_prompt,
    build_query_planner_user_prompt,
)
from gov_mem.query_semantics import infer_current_state_slots, infer_household_slots


SEMANTIC_SCOPE_VALUES = {
    "temporal_scope": {"current", "historical", "comparison", "unspecified"},
    "disclosure_scope": {"full", "redacted", "public_only", "unspecified"},
    "state_domain": {"project", "research", "operational", "unspecified"},
    "request_shape": {"fact", "list", "plan", "policy", "comparison", "mixed"},
}

DISCLOSURE_CONSTRAINT_KINDS = {
    "temporal_access_boundary",
    "access_boundary",
    "redaction_boundary",
    "exclusion",
}
NEED_KINDS = {"scalar", "record_collection"}


class QueryUnderstandingAgent:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model_name: str,
        skill_text: str = "",
        use_asking_user_id: bool = True,
        experience_bank: ExperienceBank | None = None,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.skill_text = skill_text
        self.use_asking_user_id = use_asking_user_id
        self.experience_bank = experience_bank

    def plan(self, instance: MemoryInstance) -> QueryPlan:
        asking_user_id = instance.asking_user_id if self.use_asking_user_id else None
        lessons = (
            self.experience_bank.retrieve_lessons(
                question=instance.question,
                top_k=3,
                stage="query_planning",
                domain=instance.domain,
            )
            if self.experience_bank is not None
            else []
        )

        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=QUERY_PLANNER_SYSTEM_PROMPT,
                user_prompt=build_query_planner_user_prompt(
                    question=instance.question,
                    asking_user_id=asking_user_id,
                    choices=instance.choices,
                    observable_metadata=strip_hidden_eval_fields(
                        instance.metadata.get("observable_metadata", instance.metadata.get("observable", {}))
                    ),
                    skill_text=self.skill_text,
                    retrieved_lessons=lessons,
                ),
            )
            if isinstance(raw, dict):
                plan = self._normalize_plan(instance, raw, asking_user_id=asking_user_id)
                audited = self._audit_semantic_contract(
                    instance,
                    asking_user_id=asking_user_id,
                    target_entities=plan.target_entities,
                )
                verified = self._verify_semantic_contract(
                    instance,
                    asking_user_id=asking_user_id,
                    target_entities=plan.target_entities,
                    candidate_spec=audited,
                )
                if _semantic_contract_is_usable(verified, plan):
                    plan.semantic_spec = _merge_query_grounded_slot_contract(
                        verified,
                        plan.semantic_spec,
                        instance.question,
                        target_entities=plan.target_entities,
                    )
                    plan.planning_trace = {
                        "semantic_contract_source": "verified_llm",
                        "semantic_repair_attempted": True,
                    }
                    return plan
                if _semantic_contract_is_usable(audited, plan):
                    plan.semantic_spec = _merge_query_grounded_slot_contract(
                        audited,
                        plan.semantic_spec,
                        instance.question,
                        target_entities=plan.target_entities,
                    )
                    plan.planning_trace = {
                        "semantic_contract_source": "audited_llm",
                        "semantic_repair_attempted": True,
                    }
                    return plan
                if _semantic_contract_is_usable(plan.semantic_spec, plan):
                    if str(plan.semantic_spec.get("request_shape") or "") in {"list", "mixed"}:
                        audited = self._repair_semantic_spec(
                            instance,
                            asking_user_id=asking_user_id,
                            target_entities=plan.target_entities,
                        )
                        merged_audit = dict(audited)
                        combined_bindings = _consensus_attribute_bindings(
                            plan.semantic_spec.get("attribute_bindings"),
                            audited.get("attribute_bindings"),
                        )
                        if combined_bindings:
                            merged_audit["requested_attributes"] = list(
                                plan.semantic_spec.get("raw_requested_attributes") or []
                            ) + list(audited.get("raw_requested_attributes") or [])
                            merged_audit["attribute_bindings"] = combined_bindings
                            merged_audit = _normalize_certifiable_contract(
                                merged_audit,
                                instance.question,
                                target_entities=plan.target_entities,
                            )
                            # A provenance-bound open contract supersedes the
                            # legacy closed slot vocabulary. Keeping both lets
                            # an over-complete LLM slot list broaden governance.
                            merged_audit["requested_slots"] = []
                            if len(merged_audit.get("requested_attributes") or []) > 1:
                                merged_audit["request_shape"] = "mixed"
                        if (
                            _semantic_contract_is_complete(merged_audit, plan.target_entities)
                            and (
                                bool(merged_audit.get("attribute_bindings_valid"))
                                or _semantic_contract_quality(merged_audit)
                                >= _semantic_contract_quality(plan.semantic_spec)
                            )
                        ):
                            plan.semantic_spec = merged_audit
                            plan.planning_trace = {
                                "semantic_contract_source": "audited_llm",
                                "semantic_repair_attempted": True,
                            }
                            return plan
                    plan.planning_trace = {
                        "semantic_contract_source": "initial_llm",
                        "semantic_repair_attempted": True,
                    }
                    return plan
                repaired = self._repair_semantic_spec(
                    instance,
                    asking_user_id=asking_user_id,
                    target_entities=plan.target_entities,
                )
                if _semantic_contract_is_usable(repaired, plan):
                    plan.semantic_spec = repaired
                    plan.planning_trace = {
                        "semantic_contract_source": "repair_llm",
                        "semantic_repair_attempted": True,
                    }
                else:
                    recovered = self._verify_exact_surface_recovery(
                        instance,
                        asking_user_id=asking_user_id,
                        target_entities=plan.target_entities,
                        candidate_spec=plan.semantic_spec,
                    )
                    if not _semantic_contract_is_complete(recovered, plan.target_entities):
                        recovered = _recover_exact_slot_surface_contract(
                            plan.semantic_spec,
                            instance.question,
                            target_entities=plan.target_entities,
                        )
                    if not _semantic_contract_is_complete(recovered, plan.target_entities) and str(plan.query_type or "").lower() != "safety":
                        recovered = _recover_exact_target_collection_contract(
                            plan.target_entities,
                            instance.question,
                        )
                    if _semantic_contract_is_complete(recovered, plan.target_entities):
                        plan.semantic_spec = recovered
                        # The property name remains LLM-proposed.  This
                        # structural recovery only restores its verbatim query
                        # span after the contract calls omitted that field.
                        plan.planning_trace = {
                            "semantic_contract_source": "repair_llm",
                            "semantic_repair_attempted": True,
                            "exact_surface_recovery": True,
                        }
                    else:
                        plan.planning_trace = {
                            "semantic_contract_source": "missing_after_repair",
                            "semantic_repair_attempted": True,
                        }
                return plan
        except LLMClientUnavailableError:
            return self._heuristic_plan(
                instance,
                asking_user_id=asking_user_id,
                planning_trace={"semantic_contract_source": "heuristic", "reason": "llm_unavailable"},
            )
        except Exception as exc:
            return self._heuristic_plan(
                instance,
                asking_user_id=asking_user_id,
                planning_trace={"semantic_contract_source": "heuristic", "reason": f"llm_error:{type(exc).__name__}"},
            )

        return self._heuristic_plan(
            instance,
            asking_user_id=asking_user_id,
            planning_trace={"semantic_contract_source": "heuristic", "reason": "unusable_llm_output"},
        )

    def _audit_semantic_contract(
        self,
        instance: MemoryInstance,
        *,
        asking_user_id: str | None,
        target_entities: list[str],
    ) -> dict[str, Any]:
        """Independently classify query spans before they can reach governance."""
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=QUERY_PLANNER_SYSTEM_PROMPT,
                user_prompt=build_semantic_contract_audit_prompt(
                    question=instance.question,
                    asking_user_id=asking_user_id,
                    observable_metadata=strip_hidden_eval_fields(
                        instance.metadata.get("observable_metadata", instance.metadata.get("observable", {}))
                    ),
                    target_entities=target_entities,
                ),
            )
        except (LLMClientUnavailableError, Exception):
            return {}
        payload = _semantic_spec_from_response(raw)
        if not payload:
            return {}
        audited = _normalize_semantic_spec(payload)
        return _normalize_certifiable_contract(
            audited,
            instance.question,
            target_entities=target_entities,
        )

    def _verify_semantic_contract(
        self,
        instance: MemoryInstance,
        *,
        asking_user_id: str | None,
        target_entities: list[str],
        candidate_spec: dict[str, Any],
    ) -> dict[str, Any]:
        if not candidate_spec:
            return {}
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=QUERY_PLANNER_SYSTEM_PROMPT,
                user_prompt=build_semantic_contract_verification_prompt(
                    question=instance.question,
                    asking_user_id=asking_user_id,
                    observable_metadata=strip_hidden_eval_fields(
                        instance.metadata.get("observable_metadata", instance.metadata.get("observable", {}))
                    ),
                    target_entities=target_entities,
                    candidate_spec=candidate_spec,
                ),
            )
        except (LLMClientUnavailableError, Exception):
            return {}
        payload = _semantic_spec_from_response(raw)
        if not payload:
            return {}
        return _normalize_certifiable_contract(
            _normalize_semantic_spec(payload),
            instance.question,
            target_entities=target_entities,
        )

    def _repair_semantic_spec(
        self,
        instance: MemoryInstance,
        *,
        asking_user_id: str | None,
        target_entities: list[str],
    ) -> dict[str, Any]:
        """Repair an omitted contract with a narrower LLM-only schema request."""
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=QUERY_PLANNER_SYSTEM_PROMPT,
                user_prompt=build_semantic_spec_repair_prompt(
                    question=instance.question,
                    asking_user_id=asking_user_id,
                    observable_metadata=strip_hidden_eval_fields(
                        instance.metadata.get("observable_metadata", instance.metadata.get("observable", {}))
                    ),
                )
                + f"\nPlanner-proposed target attributes: {target_entities}"
                + "\nAlso return semantic_spec.requested_attributes for every explicitly requested "
                "property, including properties covered by requested_slots. Use domain-independent canonical "
                "snake_case names and do not invent properties absent from the query.",
            )
        except (LLMClientUnavailableError, Exception):
            return {}
        payload = _semantic_spec_from_response(raw)
        if not payload:
            return {}
        repaired = _normalize_semantic_spec(payload)
        return _normalize_certifiable_contract(
            repaired,
            instance.question,
            target_entities=target_entities,
        )

    def _verify_exact_surface_recovery(
        self,
        instance: MemoryInstance,
        *,
        asking_user_id: str | None,
        target_entities: list[str],
        candidate_spec: dict[str, Any],
    ) -> dict[str, Any]:
        """Narrowly repair an unbound LLM slot that appears verbatim in a query."""
        candidate_slots = [
            slot for slot in _normalize_open_attributes(candidate_spec.get("requested_slots"))
            if _find_exact_attribute_surface(slot, instance.question)
        ]
        if not candidate_slots:
            return {}
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=(
                    "You verify a minimal query contract before privacy authorization. Return JSON only and do not "
                    "answer the question. You receive fixed candidate phrases that occur verbatim in the question. "
                    "Identify the smallest requested-property span and separately identify any access, disclosure, "
                    "or temporal constraint. Do not invent a property or use information outside the question."
                ),
                user_prompt=(
                    "Return {\"semantic_spec\":{requested_slots,requested_attributes,attribute_bindings,"
                    "disclosure_constraints,temporal_scope,disclosure_scope,state_domain,request_shape,"
                    "requires_entity_resolution}}. attribute_bindings must use an exact minimal question span and "
                    "semantic_role=requested_property. A candidate phrase can combine a property with a constraint; "
                    "then split them rather than treating the combined phrase as one property.\n"
                    f"Question: {instance.question}\nAsking user id: {asking_user_id}\n"
                    f"Target entities: {target_entities}\nFixed candidate slots: {candidate_slots}"
                ),
            )
        except (LLMClientUnavailableError, Exception):
            return {}
        payload = _semantic_spec_from_response(raw)
        if not payload:
            return {}
        return _normalize_certifiable_contract(
            _normalize_semantic_spec(payload),
            instance.question,
            target_entities=target_entities,
        )

    def _heuristic_plan(
        self,
        instance: MemoryInstance,
        *,
        asking_user_id: str | None,
        planning_trace: dict[str, Any] | None = None,
    ) -> QueryPlan:
        question = instance.question
        lowered = question.lower()
        entities = _extract_title_entities(question)
        target_users: list[str] = []
        if _should_ground_requester(
            question=question,
            asking_user_id=asking_user_id,
            target_entities=entities,
        ):
            target_users.append(asking_user_id)


        required_memory_types = []
        if any(token in lowered for token in ["prefer", "favorite", "like"]):
            query_type = "preference"
            required_memory_types.append("preference")
        elif any(token in lowered for token in ["before", "after", "latest", "current", "when"]):
            query_type = "temporal"
            required_memory_types.extend(["event", "factual"])
        elif any(token in lowered for token in ["difference", "compare", "both"]):
            query_type = "multi-hop"
            required_memory_types.extend(["relation", "factual"])
        elif any(token in lowered for token in ["conflict", "contradict", "still"]):
            query_type = "conflict"
            required_memory_types.extend(["constraint", "factual"])
        else:
            query_type = "factual"
            required_memory_types.append("factual")

        reasoning_ops = []
        if any(token in lowered for token in [" and ", "all", "both"]):
            reasoning_ops.append("AND")
        if any(token in lowered for token in [" or ", "either"]):
            reasoning_ops.append("OR")
        if any(token in lowered for token in [" not ", "except", "other than"]):
            reasoning_ops.append("NOT")
        if query_type == "temporal":
            reasoning_ops.append("temporal_order")
        if query_type == "conflict":
            reasoning_ops.append("conflict_check")
        if query_type in {"multi-hop", "group-dynamics"} or "compare" in lowered:
            reasoning_ops.append("compare")
        if not reasoning_ops:
            reasoning_ops.append("AND")

        symbolic_filters = {
            "user_ids": target_users,
            "entities": entities,
            "memory_types": required_memory_types,
        }
        dense_queries = [question]
        if _should_ground_requester(
            question=question,
            asking_user_id=asking_user_id,
            target_entities=entities,
        ):
            dense_queries.append(f"{asking_user_id} {question}")
        if entities:
            dense_queries.append(" ".join(entities + required_memory_types))

        inferred_slots = (
            infer_current_state_slots(question)
            + infer_household_slots(question)
            + _infer_security_relevant_slots(question)
        )
        semantic_spec = {}
        if _is_record_bundle_request(instance.question):
            # Collection requests ask for a coherent set of records, not a
            # single typed field. LLM-proposed slots must not narrow them.
            semantic_spec["requested_slots"] = []
            semantic_spec["request_shape"] = "mixed"
        elif inferred_slots:
            semantic_spec = {
                "requested_slots": _normalize_requested_slots(inferred_slots),
                "temporal_scope": "current" if any(token in lowered for token in ["current", "latest", "now", "still"]) else "unspecified",
                "disclosure_scope": "unspecified",
                "state_domain": "unspecified",
                "request_shape": "fact",
                "requires_entity_resolution": bool(entities),
            }
        return QueryPlan(
            query_type=query_type,
            target_users=target_users,
            target_entities=entities,
            required_memory_types=required_memory_types,
            symbolic_filters=symbolic_filters,
            dense_queries=dense_queries,
            reasoning_ops=reasoning_ops,
            semantic_spec=semantic_spec,
            planning_trace=dict(planning_trace or {"semantic_contract_source": "heuristic"}),
        )

    def _normalize_plan(self, instance: MemoryInstance, raw: dict, *, asking_user_id: str | None) -> QueryPlan:
        question = instance.question
        lowered = question.lower()
        raw_query_type = str(raw.get("query_type") or "")
        query_type = _normalize_query_type(raw_query_type, question)
        target_users = _as_string_list(raw.get("target_users"))
        target_entities = _normalize_entities(_as_string_list(raw.get("target_entities")), question)
        requester_grounding_allowed = _should_ground_requester(
            question=question,
            asking_user_id=asking_user_id,
            target_entities=target_entities,
        )
        if requester_grounding_allowed and asking_user_id:
            if asking_user_id not in target_users:
                target_users.insert(0, asking_user_id)
        if asking_user_id and target_users == [asking_user_id] and not requester_grounding_allowed and target_entities:
            target_users = []
        required_memory_types = _normalize_memory_types(
            _as_string_list(raw.get("required_memory_types")),
            question=question,
            query_type=query_type,
        )
        dense_queries = _as_string_list(raw.get("dense_queries"))
        if not dense_queries:
            dense_queries = [question]
        if asking_user_id and any(token in f" {lowered} " for token in [" i ", " my ", " me ", " we ", " our "]):
            if all(asking_user_id not in row for row in dense_queries):
                dense_queries.append(f"{asking_user_id} {question}")
        reasoning_ops = _normalize_reasoning_ops(
            _as_string_list(raw.get("reasoning_ops")),
            question=question,
            query_type=query_type,
        )
        raw_symbolic_filters = raw.get("symbolic_filters")
        symbolic_filters = dict(raw_symbolic_filters) if isinstance(raw_symbolic_filters, dict) else {}
        symbolic_filters["user_ids"] = list(target_users)
        symbolic_filters["entities"] = list(target_entities)
        symbolic_filters["memory_types"] = list(required_memory_types)
        semantic_spec = _normalize_semantic_spec(_semantic_spec_from_response(raw))
        semantic_spec = _normalize_certifiable_contract(
            semantic_spec,
            question,
            target_entities=target_entities,
        )
        inferred_slots = (
            infer_current_state_slots(instance.question)
            + infer_household_slots(instance.question)
            + _infer_security_relevant_slots(instance.question)
        )
        if _is_record_bundle_request(instance.question):
            semantic_spec["request_shape"] = "mixed"
        elif inferred_slots:
            # Deterministic semantic grounding is anchored in the query text.
            # Do not let unsupported LLM-proposed slots broaden authorization.
            semantic_spec["requested_slots"] = _normalize_requested_slots(
                list(semantic_spec.get("requested_slots") or []) + list(inferred_slots)
            )
            # The slot surface is part of the typed contract. Rebuild the
            # certifiable needs after adding deterministic query slots so a
            # planner/auditor disagreement cannot silently drop a requested
            # field after the initial normalization pass.
            semantic_spec = _normalize_certifiable_contract(
                semantic_spec,
                question,
                target_entities=target_entities,
            )
        if semantic_spec.get("attribute_bindings_valid") and len(semantic_spec.get("requested_attributes") or []) > 1:
            semantic_spec["request_shape"] = "mixed"
            semantic_spec["requested_slots"] = []
        return QueryPlan(
            query_type=query_type,
            target_users=target_users,
            target_entities=target_entities,
            required_memory_types=required_memory_types,
            symbolic_filters=symbolic_filters,
            dense_queries=dense_queries,
            reasoning_ops=reasoning_ops,
            semantic_spec=semantic_spec,
        )


def _normalize_semantic_spec(raw: object) -> dict:
    if not isinstance(raw, dict):
        return {}
    # Slot names are an open schema. Restricting this field to a benchmark-era
    # vocabulary prevents a policy/fact contract from representing new domains.
    slots = _normalize_open_attributes(raw.get("requested_slots"))
    normalized = {
        "requested_slots": _normalize_requested_slots(slots),
        "requested_attributes": _normalize_open_attributes(raw.get("requested_attributes")),
        # Raw planner labels can help retrieval and debugging, but never form
        # a governance obligation until a minimal property binding validates it.
        "raw_requested_attributes": _normalize_open_attributes(raw.get("requested_attributes")),
        "attribute_bindings": _normalize_attribute_bindings(raw.get("attribute_bindings")),
        "disclosure_constraints": _normalize_disclosure_constraints(raw.get("disclosure_constraints")),
    }
    for key, allowed_values in SEMANTIC_SCOPE_VALUES.items():
        value = str(raw.get(key) or "").strip().lower()
        normalized[key] = value if value in allowed_values else "unspecified"
    normalized["requires_entity_resolution"] = bool(raw.get("requires_entity_resolution"))
    return normalized


def _recover_exact_slot_surface_contract(
    semantic_spec: dict[str, Any],
    question: str,
    *,
    target_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Recover omitted bindings for LLM slots with an exact query surface.

    This is intentionally narrower than semantic parsing: an LLM must already
    have supplied the open-schema slot, and every generated support span is a
    contiguous, verbatim sequence of question tokens.  It does not introduce
    attributes from a vocabulary or decide what a phrase means.
    """
    slots = _normalize_open_attributes(dict(semantic_spec or {}).get("requested_slots"))
    bindings: list[dict[str, str]] = []
    for slot in slots:
        span = _find_exact_attribute_surface(slot, question)
        if not span:
            continue
        candidate = {
            "attribute": slot,
            "support_span": span,
            "semantic_role": "requested_property",
            "evidence_slot_hint": "",
            "need_kind": (
                "record_collection"
                if str(semantic_spec.get("request_shape") or "").lower() in {"list", "mixed"}
                else "scalar"
            ),
        }
        if _binding_is_dynamic_entity(candidate, target_entities):
            continue
        bindings.append(candidate)
    if not bindings:
        return {}
    recovered = dict(semantic_spec or {})
    recovered["requested_attributes"] = [binding["attribute"] for binding in bindings]
    recovered["raw_requested_attributes"] = [binding["attribute"] for binding in bindings]
    recovered["attribute_bindings"] = bindings
    return _normalize_certifiable_contract(recovered, question, target_entities=target_entities)


def _recover_exact_target_collection_contract(target_entities: list[str], question: str) -> dict[str, Any]:
    """Use an LLM-extracted query object as a collection only when textually grounded."""
    for entity in target_entities:
        span = _find_exact_attribute_surface(_span_attribute_key(entity), question)
        if not span or any(char.isupper() for char in str(entity)):
            continue
        attribute = _span_attribute_key(span)
        return _normalize_certifiable_contract({
            "requested_attributes": [attribute],
            "attribute_bindings": [{
                "attribute": attribute, "support_span": span, "semantic_role": "requested_property",
                "need_kind": "record_collection",
            }],
            "request_shape": "list",
        }, question, target_entities=target_entities)
    return {}


def _find_exact_attribute_surface(attribute: str, question: str) -> str:
    """Return a verbatim query span whose normalized token sequence is `attribute`."""
    attribute_tokens = re.findall(r"[a-z0-9]+", str(attribute or "").lower())
    question_tokens = list(re.finditer(r"[a-z0-9]+", str(question or ""), flags=re.IGNORECASE))
    if not attribute_tokens or len(attribute_tokens) > len(question_tokens):
        return ""
    lowered_tokens = [match.group(0).lower() for match in question_tokens]
    width = len(attribute_tokens)
    for start in range(len(lowered_tokens) - width + 1):
        if lowered_tokens[start:start + width] == attribute_tokens:
            return str(question)[question_tokens[start].start():question_tokens[start + width - 1].end()]
    return ""


def _semantic_spec_from_response(raw: object) -> dict[str, Any]:
    """Read a semantic contract from provider-neutral structured envelopes.

    The contract stages use JSON mode, but compatible providers can preserve
    the requested object under a generic result/data wrapper or name it a
    semantic/query contract. This adapter changes only transport envelopes:
    subsequent normalization still requires verbatim query bindings and does
    not infer any property from vocabulary.
    """
    if not isinstance(raw, dict):
        return {}
    envelope_keys = ("semantic_spec", "semantic_contract", "query_contract", "contract")
    for key in envelope_keys:
        payload = raw.get(key)
        if isinstance(payload, dict):
            return payload
    semantic_fields = {
        "requested_slots", "requested_attributes", "attribute_bindings",
        "disclosure_constraints", "temporal_scope", "disclosure_scope",
        "state_domain", "request_shape", "requires_entity_resolution",
    }
    if semantic_fields & set(raw):
        return raw
    for wrapper_key in ("result", "data", "response", "output"):
        wrapped = raw.get(wrapper_key)
        if not isinstance(wrapped, dict):
            continue
        for key in envelope_keys:
            payload = wrapped.get(key)
            if isinstance(payload, dict):
                return payload
        if semantic_fields & set(wrapped):
            return wrapped
    return {}


def _normalize_certifiable_contract(
    semantic_spec: dict[str, Any],
    question: str,
    *,
    target_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Separate raw planning hints from the closed set of certifiable needs."""
    normalized = _augment_exact_slot_bindings(
        dict(semantic_spec or {}),
        question,
        target_entities=target_entities,
    )
    raw_attributes = _normalize_open_attributes(
        normalized.get("raw_requested_attributes", normalized.get("requested_attributes"))
    )
    normalized["raw_requested_attributes"] = raw_attributes
    grounded, bindings_valid = _ground_attribute_contract(
        raw_attributes,
        normalized.get("attribute_bindings"),
        question,
        target_entities=target_entities,
    )
    bindings = _normalize_attribute_bindings(normalized.get("attribute_bindings"))
    constraints = _normalize_disclosure_constraints(normalized.get("disclosure_constraints"))
    # A planner can express a boundary either directly or by assigning a
    # temporal/disclosure role to a span. Both are query-local and never become
    # evidence attributes merely because they occur beside a requested fact.
    for binding in bindings:
        if binding.get("semantic_role") == "disclosure_constraint":
            constraints.append({
                "constraint_kind": "access_boundary",
                "support_span": binding["support_span"],
            })
    constraints = _deduplicate_disclosure_constraints(constraints)
    constraint_spans = {str(item.get("support_span") or "").strip().lower() for item in constraints}
    certifiable_needs: list[dict[str, str]] = []
    for attribute in grounded:
        matching = [
            binding
            for binding in bindings
            if binding.get("semantic_role") == "requested_property"
            and (
                binding.get("attribute") == attribute
                or _span_attribute_key(binding.get("support_span", "")) == attribute
            )
        ]
        if not matching:
            continue
        binding = matching[0]
        if _binding_is_dynamic_entity(binding, target_entities):
            continue
        if str(binding.get("support_span") or "").strip().lower() in constraint_spans:
            continue
        need = {
            "need_id": attribute,
            "attribute": attribute,
            "query_support_span": binding["support_span"],
            "evidence_slot_hint": binding.get("evidence_slot_hint", ""),
            "temporal_scope": str(normalized.get("temporal_scope") or "unspecified"),
        }
        if binding.get("need_kind") == "record_collection":
            need["need_kind"] = "record_collection"
        certifiable_needs.append(need)
    # These compatibility fields are now derived views, never independently
    # merged planner output. This prevents an entity/modifier from poisoning a
    # certificate after the binding set has already been validated.
    normalized["certifiable_needs"] = certifiable_needs
    if any(need.get("need_kind") == "record_collection" for need in certifiable_needs):
        normalized["request_shape"] = "list"
    normalized["requested_attributes"] = [need["attribute"] for need in certifiable_needs]
    normalized["attribute_bindings"] = [
        {
            "attribute": need["attribute"],
            "support_span": need["query_support_span"],
            "semantic_role": "requested_property",
            "evidence_slot_hint": need["evidence_slot_hint"],
            **({"need_kind": "record_collection"} if need.get("need_kind") == "record_collection" else {}),
        }
        for need in certifiable_needs
    ]
    normalized["disclosure_constraints"] = _deduplicate_disclosure_constraints(constraints)
    normalized["attribute_bindings_valid"] = bool(bindings_valid and certifiable_needs) and _bindings_cover_attributes(
        attributes=normalized["requested_attributes"],
        bindings=normalized["attribute_bindings"],
    )
    return normalized


def _augment_exact_slot_bindings(
    semantic_spec: dict[str, Any],
    question: str,
    *,
    target_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Recover omitted properties from LLM slots only when textually grounded.

    ``requested_slots`` and the open attribute contract are two views of the
    same query.  A planner can emit a slot for a property while a later audit
    omits the corresponding open attribute.  Preserve recall by adding only
    slots whose canonical name occurs as a contiguous surface span in the
    question; this never introduces a dataset vocabulary or invents a field.
    """
    result = dict(semantic_spec or {})
    slots = _normalize_open_attributes(result.get("requested_slots"))
    bindings = _normalize_attribute_bindings(result.get("attribute_bindings"))
    existing = {
        str(item.get("attribute") or "").strip()
        for item in bindings
        if str(item.get("attribute") or "").strip()
    }
    existing_spans = {
        " ".join(re.findall(r"[a-z0-9]+", str(item.get("support_span") or "").lower()))
        for item in bindings
        if str(item.get("support_span") or "").strip()
    }
    added: list[dict[str, str]] = []
    need_kind = (
        "record_collection"
        if str(result.get("request_shape") or "").lower() in {"list", "mixed"}
        else "scalar"
    )
    for slot in slots:
        span = _find_exact_attribute_surface(slot, question)
        if not span:
            continue
        normalized_span = " ".join(re.findall(r"[a-z0-9]+", span.lower()))
        if slot in existing or normalized_span in existing_spans:
            continue
        candidate = {
            "attribute": slot,
            "support_span": span,
            "semantic_role": "requested_property",
            "evidence_slot_hint": "",
            "need_kind": need_kind,
        }
        if _binding_is_dynamic_entity(candidate, target_entities):
            continue
        bindings.append(candidate)
        existing.add(slot)
        existing_spans.add(normalized_span)
        added.append(candidate)
    if added:
        result["attribute_bindings"] = bindings
        result["raw_requested_attributes"] = list(dict.fromkeys([
            *_normalize_open_attributes(result.get("raw_requested_attributes")),
            *_normalize_open_attributes(result.get("requested_attributes")),
            *[item["attribute"] for item in added],
        ]))
    return result


def _merge_query_grounded_slot_contract(
    candidate: dict[str, Any],
    fallback: dict[str, Any],
    question: str,
    *,
    target_entities: list[str] | None = None,
) -> dict[str, Any]:
    """Preserve query-grounded slots when an audit replaces the plan.

    Independent planner views may disagree on the open-schema field list.
    Their closed contract must retain the union of slots explicitly present
    in the query; normalization still admits only contiguous query surfaces.
    """
    merged = dict(candidate or {})
    merged["requested_slots"] = _normalize_requested_slots(
        list(merged.get("requested_slots") or [])
        + list((fallback or {}).get("requested_slots") or [])
    )
    return _normalize_certifiable_contract(
        merged,
        question,
        target_entities=target_entities,
    )


def _binding_is_dynamic_entity(binding: dict[str, str], target_entities: list[str] | None) -> bool:
    span_tokens = set(re.findall(r"[a-z0-9]+", str(binding.get("support_span") or "").lower()))
    return bool(span_tokens) and any(
        any(char.isupper() for char in str(entity or ""))
        and span_tokens == set(re.findall(r"[a-z0-9]+", str(entity or "").lower()))
        for entity in list(target_entities or [])
        if str(entity or "").strip()
    )


def _span_attribute_key(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _as_string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            item = (
                item.get("identification") or item.get("display_name") or item.get("name")
                or item.get("label") or item.get("value") or ""
            )
        value = str(item).strip()
        if value:
            values.append(value)
    return values


def _ground_requested_attributes(raw: object, question: str) -> list[str]:
    attributes = _normalize_open_attributes(raw)
    question_tokens = set(re.findall(r"[a-z0-9]+", str(question or "").lower()))
    return [
        attribute
        for attribute in attributes
        if set(re.findall(r"[a-z0-9]+", attribute.lower())).issubset(question_tokens)
    ]


def _ground_attribute_contract(
    raw_attributes: object,
    raw_bindings: object,
    question: str,
    *,
    target_entities: list[str] | None = None,
) -> tuple[list[str], bool]:
    bindings = _normalize_attribute_bindings(raw_bindings)
    question_text = str(question or "").lower()
    entity_token_sets = [
        set(re.findall(r"[a-z0-9]+", entity.lower()))
        for entity in list(target_entities or [])
        if str(entity).strip()
    ]
    title_entity_sets = [
        set(re.findall(r"[a-z0-9]+", entity.lower()))
        for entity in _extract_title_entities(question)
        if str(entity).strip()
    ]
    if title_entity_sets:
        entity_token_sets = [
            entity_tokens
            for entity_tokens in entity_token_sets
            if any(
                title_tokens.issubset(entity_tokens)
                or entity_tokens.issubset(title_tokens)
                for title_tokens in title_entity_sets
            )
        ]
    valid_bindings: list[dict[str, str]] = []
    for binding in bindings:
        attribute = binding["attribute"]
        support_span = binding["support_span"]
        if support_span.lower() not in question_text:
            continue
        if _property_span_is_overbroad(support_span=support_span, question=question):
            # A whole-question span is not a provenance-bearing property
            # reference. Reject it so the LLM repair path must decompose the
            # request into minimal semantic needs.
            continue
        if binding.get("semantic_role") != "requested_property":
            continue
        if _property_span_is_disclosure_or_temporal_boundary(support_span):
            # A contract may not turn a range qualifier into a value to
            # render. This validates the role boundary structurally; the LLM
            # still determines the actual requested property.
            continue
        support_tokens = set(re.findall(r"[a-z0-9]+", support_span.lower()))
        # The LLM's explicit role, rather than token overlap with a dynamic
        # entity list, determines whether this span is a requested property.
        # A queried property can itself name the target (for example a list of
        # alerts); list/mixed requests receive a second independent audit.
        valid_bindings.append(binding)

    # A property phrase may be mapped to both a legacy slot name and a more
    # precise open attribute. Keep the binding best aligned with its evidence
    # span so contract size cannot be inflated by duplicate aliases.
    best_by_span: dict[str, tuple[set[str], tuple[float, int, str]]] = {}
    for index, binding in enumerate(valid_bindings):
        attribute = binding["attribute"]
        span_key = " ".join(re.findall(r"[a-z0-9]+", binding["support_span"].lower()))
        attribute_tokens = set(re.findall(r"[a-z0-9]+", attribute.lower()))
        span_tokens = set(span_key.split())
        alignment = len(attribute_tokens & span_tokens) / max(1, len(attribute_tokens))
        if alignment < 0.8 and span_key:
            # The verbatim property span is the provenance-bearing schema
            # source when an LLM canonical key drifts to a different concept.
            attribute = span_key.replace(" ", "_")
            attribute_tokens = set(span_tokens)
            alignment = 1.0
        candidate = (alignment, -index, attribute)
        family_key = span_key
        for existing_key, (existing_tokens, _) in best_by_span.items():
            smaller = min(len(span_tokens), len(existing_tokens))
            if smaller >= 2 and len(span_tokens & existing_tokens) == smaller:
                family_key = existing_key
                break
        if family_key not in best_by_span or candidate > best_by_span[family_key][1]:
            best_by_span[family_key] = (span_tokens, candidate)
    grounded_from_bindings = [
        value[1][2]
        for value in sorted(best_by_span.values(), key=lambda item: -item[1][1])
    ]
    if grounded_from_bindings:
        return grounded_from_bindings, True
    return _ground_requested_attributes(raw_attributes, question), False


def _property_span_is_overbroad(*, support_span: str, question: str) -> bool:
    """Reject a sentence-sized support span without assuming domain vocabulary."""
    question_tokens = re.findall(r"[a-z0-9]+", str(question or "").lower())
    span_tokens = re.findall(r"[a-z0-9]+", str(support_span or "").lower())
    if len(question_tokens) < 6 or not span_tokens:
        return False
    return len(span_tokens) / len(question_tokens) >= 0.8


def _property_span_is_disclosure_or_temporal_boundary(support_span: str) -> bool:
    """Reject a standalone range/access qualifier mislabeled as a property."""
    lowered = " ".join(re.findall(r"[a-z0-9]+", str(support_span or "").lower()))
    return bool(re.match(r"^(?:through|until|before|after|during|within|for)\b", lowered))


def _bindings_cover_attributes(*, attributes: list[str], bindings: object) -> bool:
    """A certificate contract cannot contain unbound raw planner fields."""
    bound: set[str] = set()
    for binding in _normalize_attribute_bindings(bindings):
        if str(binding.get("semantic_role") or "") != "requested_property":
            continue
        bound.add(str(binding.get("attribute") or "").strip())
        span_key = "_".join(re.findall(r"[a-z0-9]+", str(binding.get("support_span") or "").lower()))
        if span_key:
            bound.add(span_key)
    return bool(attributes) and set(attributes).issubset(bound)


def _consensus_attribute_bindings(
    first_raw: object,
    second_raw: object,
) -> list[dict[str, str]]:
    """Keep only independently repeated property spans from two LLM views."""
    first = _normalize_attribute_bindings(first_raw)
    second = _normalize_attribute_bindings(second_raw)

    def tokens(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower()))

    def same_span_family(left: set[str], right: set[str]) -> bool:
        if not left or not right:
            return False
        if left == right:
            return True
        smaller = min(len(left), len(right))
        return smaller >= 2 and len(left & right) == smaller

    consensus: list[dict[str, str]] = []
    paired_views = (
        [(binding, second) for binding in first]
        + [(binding, first) for binding in second]
    )
    for binding, peers in paired_views:
        if binding.get("semantic_role") != "requested_property":
            continue
        span_tokens = tokens(binding["support_span"])
        cross_view = any(
            peer.get("semantic_role") == "requested_property"
            and same_span_family(span_tokens, tokens(peer["support_span"]))
            for peer in peers
        )
        if cross_view and binding not in consensus:
            consensus.append(binding)
    return consensus


def _semantic_contract_quality(semantic_spec: dict[str, Any]) -> tuple[int, int]:
    attributes = list(dict.fromkeys(semantic_spec.get("requested_attributes") or []))
    return len(attributes), len(set(attributes))


def _normalize_requested_slots(slots: list[str]) -> list[str]:
    normalized = list(dict.fromkeys(str(slot) for slot in slots if str(slot)))
    if "date" in normalized and any(
        slot in normalized for slot in ("target_date", "public_event_date")
    ):
        normalized = [slot for slot in normalized if slot != "date"]
    if "family_release_scope" in normalized:
        normalized = [slot for slot in normalized if slot != "policy_scope"]
    if any(slot in normalized for slot in ("access_room", "public_room")):
        normalized = [
            slot for slot in normalized if slot not in {"location", "approved_areas"}
        ]
    return normalized


def _normalize_open_attributes(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    for item in raw:
        if item is None:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", str(item).strip().lower()).strip("_")
        if key and len(key) <= 80 and key not in normalized:
            normalized.append(key)
    return normalized


def _normalize_attribute_bindings(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        attributes = _normalize_open_attributes([item.get("attribute")])
        support_span = str(item.get("support_span") or "").strip()
        semantic_role = str(item.get("semantic_role") or "").strip().lower()
        if semantic_role not in {
            "",
            "requested_property",
            "target_entity",
            "request_object",
            "temporal_modifier",
            "disclosure_constraint",
        }:
            semantic_role = ""
        if attributes and support_span:
            normalized.append({
                "attribute": attributes[0],
                "support_span": support_span,
                "semantic_role": semantic_role,
                "evidence_slot_hint": _normalize_open_attributes([item.get("evidence_slot_hint")])[0]
                if _normalize_open_attributes([item.get("evidence_slot_hint")]) else "",
                "need_kind": str(item.get("need_kind") or "scalar").strip().lower()
                if str(item.get("need_kind") or "scalar").strip().lower() in NEED_KINDS else "scalar",
            })
    return normalized


def _normalize_disclosure_constraints(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    constraints: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("constraint_kind") or "").strip().lower()
        span = str(item.get("support_span") or "").strip()
        if kind in DISCLOSURE_CONSTRAINT_KINDS and span:
            constraints.append({"constraint_kind": kind, "support_span": span})
    return _deduplicate_disclosure_constraints(constraints)


def _deduplicate_disclosure_constraints(constraints: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in constraints:
        kind = str(item.get("constraint_kind") or "").strip().lower()
        span = str(item.get("support_span") or "").strip()
        key = (kind, span.lower())
        if kind in DISCLOSURE_CONSTRAINT_KINDS and span and key not in seen:
            seen.add(key)
            result.append({"constraint_kind": kind, "support_span": span})
    return result


def _infer_security_relevant_slots(question: str) -> list[str]:
    lowered = str(question or "").lower()
    slots: list[str] = []
    if any(token in lowered for token in [
        "phone", "phone number", "mobile", "direct line", "callback number",
        "contact number", "contact details",
    ]):
        slots.append("phone")
    return slots


def _is_record_bundle_request(question: str) -> bool:
    lowered = str(question or "").lower()
    collection_noun = re.search(
        r"\b(?:items?|plans?|bookings?|appointments?|routes?|triggers?|steps?|"
        r"instructions?|schedule|recap|summary|details?)\b",
        lowered,
    )
    collection_signal = bool(re.search(
        r"\b(?:what|which|list|give|show|tell|current|active|remain|booked|"
        r"three|two|all|both|next[- ]week)\b",
        lowered,
    ))
    return bool(collection_noun and collection_signal)


def _has_semantic_signal(semantic_spec: object) -> bool:
    if not isinstance(semantic_spec, dict):
        return False
    if any(str(slot).strip() for slot in list(semantic_spec.get("requested_slots") or [])):
        return True
    if any(str(slot).strip() for slot in list(semantic_spec.get("requested_attributes") or [])):
        return True
    return any(
        str(semantic_spec.get(key) or "unspecified") != "unspecified"
        for key in ("temporal_scope", "disclosure_scope", "state_domain", "request_shape")
    )


def _semantic_contract_is_complete(
    semantic_spec: object,
    target_entities: list[str],
) -> bool:
    if not _has_semantic_signal(semantic_spec):
        return False
    if not isinstance(semantic_spec, dict):
        return False
    requested_slots = [
        str(slot).strip()
        for slot in list(semantic_spec.get("requested_slots") or [])
        if str(slot).strip()
    ]
    requested_attributes = [
        str(attribute).strip()
        for attribute in list(semantic_spec.get("requested_attributes") or [])
        if str(attribute).strip()
    ]
    raw_attributes = [
        str(attribute).strip()
        for attribute in list(semantic_spec.get("raw_requested_attributes") or [])
        if str(attribute).strip()
    ]
    if raw_attributes and not bool(semantic_spec.get("attribute_bindings_valid")):
        return False
    request_shape = str(semantic_spec.get("request_shape") or "unspecified").lower()
    requires_field_contract = request_shape in {"list", "mixed"} or len(target_entities) > 1
    if not requires_field_contract:
        return True
    if "requested_attributes" in semantic_spec:
        return bool(requested_attributes)
    return bool(requested_slots)


def _semantic_contract_is_usable(semantic_spec: object, plan: QueryPlan) -> bool:
    """Require an evidence obligation for non-safety queries before certification."""
    if not _semantic_contract_is_complete(semantic_spec, plan.target_entities):
        return False
    if str(plan.query_type or "").lower() == "safety":
        return True
    return bool(list((semantic_spec or {}).get("certifiable_needs") or []))


def _should_ground_requester(question: str, asking_user_id: str | None, target_entities: list[str]) -> bool:
    if not asking_user_id:
        return False
    lowered = f" {question.lower()} "
    if not any(token in lowered for token in [" i ", " my ", " me ", " we ", " our "]):
        return False
    requester_core = _normalize_entityish(asking_user_id)
    explicit_non_requester = [
        entity
        for entity in target_entities
        if _normalize_entityish(entity)
        and _normalize_entityish(entity) != requester_core
        and entity.lower() not in {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    ]
    if explicit_non_requester:
        return False
    return True


def _normalize_entityish(value: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", str(value or "").strip().lower())
    tokens = [tok for tok in re.split(r"\s+", cleaned) if tok and tok not in {"adult", "child", "resident", "student", "patient", "prof", "professor", "manager", "contact"}]
    if len(tokens) >= 2:
        return " ".join(tokens[-2:])
    return " ".join(tokens)


def _extract_title_entities(text: str) -> list[str]:
    singletons = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]+\b", text)
    phrases = re.findall(r"\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)+)(?:'s)?\b", text)
    blocked = {"As", "Of", "Now", "What", "Which", "Who", "When", "Where", "Why", "How"}
    out = []
    for item in list(phrases) + list(singletons):
        item = str(item).strip()
        if not item or item in blocked:
            continue
        if item not in out:
            out.append(item)
    return out[:10]


def _normalize_query_type(raw_query_type: str, question: str) -> str:
    lowered = raw_query_type.lower()
    q = question.lower()
    if any(token in lowered for token in ["temporal", "update", "latest", "current", "follow_up"]):
        return "temporal"
    if any(token in lowered for token in ["conflict", "revoked", "no longer", "active setting"]):
        return "conflict"
    if any(token in lowered for token in ["compare", "multi-hop", "group"]):
        return "multi-hop"
    if any(token in lowered for token in ["preference"]):
        return "preference"
    if any(token in q for token in ["before", "after", "current", "currently", "latest", "updated", "still needs", "no longer active"]):
        return "temporal"
    if any(token in q for token in ["revoked", "currently active for", "conflict", "contradict"]):
        return "conflict"
    return "factual"


def _normalize_entities(entities: list[str], question: str) -> list[str]:
    blocked = {"what", "which", "who", "when", "where", "why", "how", "as", "of", "now"}
    extracted = _extract_title_entities(question)
    normalized = []
    for entity in list(entities) + list(extracted):
        item = str(entity).strip()
        if not item or item.lower() in blocked:
            continue
        if item not in normalized:
            normalized.append(item)
    return normalized[:10]


def _normalize_memory_types(raw_types: list[str], *, question: str, query_type: str) -> list[str]:
    allowed = {"factual", "preference", "intention", "constraint", "event", "relation", "experience", "task"}
    normalized = []
    for value in raw_types:
        lowered = value.lower().strip()
        if lowered in allowed and lowered not in normalized:
            normalized.append(lowered)
        elif "follow" in lowered or "appointment" in lowered or "schedule" in lowered:
            if "task" not in normalized:
                normalized.append("task")
        elif "medical_event" in lowered or "timeline" in lowered:
            if "event" not in normalized:
                normalized.append("event")
        elif "communication" in lowered:
            if "constraint" not in normalized:
                normalized.append("constraint")
    if normalized:
        return normalized
    q = question.lower()
    defaults = []
    if any(token in q for token in ["appointment", "schedule", "plan", "follow-up", "follow up"]):
        defaults.append("task")
    if query_type == "temporal":
        defaults.extend(["event", "constraint"])
    if any(token in q for token in ["access", "allowed", "authorized", "revoked", "privacy"]):
        defaults.append("constraint")
    if not defaults:
        defaults.append("factual")
    return list(dict.fromkeys(defaults))


def _normalize_reasoning_ops(raw_ops: list[str], *, question: str, query_type: str) -> list[str]:
    supported = {"AND", "OR", "NOT", "temporal_order", "compare", "conflict_check"}
    normalized: list[str] = []
    lowered_question = question.lower()
    for value in raw_ops:
        lowered = value.lower()
        if value in supported and value not in normalized:
            normalized.append(value)
            continue
        if "and" in lowered and "AND" not in normalized:
            normalized.append("AND")
        if "or" in lowered and "OR" not in normalized:
            normalized.append("OR")
        if "not" in lowered or "exclude" in lowered or "ground pronoun" in lowered:
            if "NOT" not in normalized:
                normalized.append("NOT")
        if any(token in lowered for token in ["temporal", "latest", "updated", "after", "before"]):
            if "temporal_order" not in normalized:
                normalized.append("temporal_order")
        if any(token in lowered for token in ["compare", "difference"]):
            if "compare" not in normalized:
                normalized.append("compare")
        if any(token in lowered for token in ["conflict", "revise", "revision", "canceled", "revoked"]):
            if "conflict_check" not in normalized:
                normalized.append("conflict_check")
    if query_type == "temporal" and "temporal_order" not in normalized:
        normalized.append("temporal_order")
    if query_type == "conflict" and "conflict_check" not in normalized:
        normalized.append("conflict_check")
    if any(token in lowered_question for token in [" and ", " both ", "still needs", "what prior"]):
        if "AND" not in normalized:
            normalized.append("AND")
    if any(token in lowered_question for token in ["compare", "difference"]):
        if "compare" not in normalized:
            normalized.append("compare")
    if any(token in lowered_question for token in ["no longer active", "canceled", "revoked", "replaced"]):
        if "conflict_check" not in normalized:
            normalized.append("conflict_check")
    return normalized or ["AND"]
