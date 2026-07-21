from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import MemoryInstance, QueryPlan
from gov_mem.experience.experience_bank import ExperienceBank
from gov_mem.governance_runtime.leakage_guard import strip_hidden_eval_fields
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_semantic_spec_repair_prompt,
    build_query_planner_user_prompt,
)
from gov_mem.query_semantics import infer_current_state_slots, infer_household_slots


SEMANTIC_SLOT_VOCABULARY = {
    "target_date", "public_event_date", "approved_budget", "approved_discount_cap",
    "monthly_stipend", "safe_wording", "blocker", "access_room", "access_badge",
    "operational_result", "date", "time", "location", "visit_window",
    "entry_method", "package_rule", "approved_areas", "parking_pass",
    "arrival_contact_rule", "medication", "dosage", "instruction", "condition",
    "policy_scope", "phone", "contact_method",
    "contract_structure", "selected_vendor", "family_release_scope",
    "public_room", "coordination_label", "access_token",
}
SEMANTIC_SCOPE_VALUES = {
    "temporal_scope": {"current", "historical", "comparison", "unspecified"},
    "disclosure_scope": {"full", "redacted", "public_only", "unspecified"},
    "state_domain": {"project", "research", "operational", "unspecified"},
    "request_shape": {"fact", "list", "plan", "policy", "comparison", "mixed"},
}


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
                if _semantic_contract_is_complete(plan.semantic_spec, plan.target_entities):
                    if (
                        str(plan.semantic_spec.get("request_shape") or "") in {"list", "mixed"}
                        or _semantic_contract_needs_slot_audit(plan.semantic_spec)
                    ):
                        audited = self._repair_semantic_spec(
                            instance,
                            asking_user_id=asking_user_id,
                            target_entities=plan.target_entities,
                        )
                        merged_audit = dict(plan.semantic_spec)
                        merged_audit.update(audited)
                        # The repair call fills coverage gaps; it must not
                        # erase fields already grounded by the first call.
                        merged_audit["requested_attributes"] = list(dict.fromkeys(
                            list(plan.semantic_spec.get("requested_attributes") or [])
                            + list(audited.get("requested_attributes") or [])
                        ))
                        if plan.semantic_spec.get("attribute_bindings") and not audited.get("attribute_bindings"):
                            merged_audit["attribute_bindings"] = list(
                                plan.semantic_spec.get("attribute_bindings") or []
                            )
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
                if _semantic_contract_is_complete(repaired, plan.target_entities):
                    plan.semantic_spec = repaired
                    plan.planning_trace = {
                        "semantic_contract_source": "repair_llm",
                        "semantic_repair_attempted": True,
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
            return self._recover_after_planner_error(
                instance,
                asking_user_id=asking_user_id,
                reason=f"llm_error:{type(exc).__name__}",
            )

        return self._heuristic_plan(
            instance,
            asking_user_id=asking_user_id,
            planning_trace={"semantic_contract_source": "heuristic", "reason": "unusable_llm_output"},
        )

    def _recover_after_planner_error(
        self,
        instance: MemoryInstance,
        *,
        asking_user_id: str | None,
        reason: str,
    ) -> QueryPlan:
        """Recover a typed contract after a transport/provider planner error.

        A planner failure should not erase the question's requested fields.
        Keep the deterministic retrieval fallback, then spend one bounded LLM
        repair call on the semantic contract only. If that repair also fails,
        the original fail-closed heuristic plan remains the fallback.
        """
        plan = self._heuristic_plan(
            instance,
            asking_user_id=asking_user_id,
            planning_trace={"semantic_contract_source": "heuristic", "reason": reason},
        )
        repaired = self._repair_semantic_spec(
            instance,
            asking_user_id=asking_user_id,
            target_entities=plan.target_entities,
        )
        merged = dict(plan.semantic_spec or {})
        merged.update(repaired)
        if _semantic_contract_is_complete(merged, plan.target_entities):
            plan.semantic_spec = merged
            plan.planning_trace = {
                "semantic_contract_source": "repair_llm_after_planner_error",
                "semantic_repair_attempted": True,
                "planner_error": reason,
            }
        return plan

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
        if not isinstance(raw, dict):
            return {}
        payload = raw.get("semantic_spec")
        if not isinstance(payload, dict):
            return {}
        repaired = _normalize_semantic_spec(payload)
        grounded, bindings_valid = _ground_attribute_contract(
            repaired.get("requested_attributes"),
            repaired.get("attribute_bindings"),
            instance.question,
            target_entities=target_entities,
        )
        repaired["requested_attributes"] = grounded
        repaired["attribute_bindings_valid"] = bindings_valid
        return repaired

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
        semantic_spec = _normalize_semantic_spec(raw.get("semantic_spec"))
        grounded, bindings_valid = _ground_attribute_contract(
            semantic_spec.get("requested_attributes"),
            semantic_spec.get("attribute_bindings"),
            question,
            target_entities=target_entities,
        )
        semantic_spec["requested_attributes"] = grounded
        semantic_spec["attribute_bindings_valid"] = bindings_valid
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
            semantic_spec["requested_slots"] = _normalize_requested_slots(inferred_slots)
        # Temporal scope is a contract invariant when the question explicitly
        # asks for current/latest state. Preserve the model's historical or
        # comparison choice, but repair an omitted scope from the query cue.
        if (
            str(semantic_spec.get("temporal_scope") or "unspecified") == "unspecified"
            and any(token in lowered for token in ["current", "currently", "latest", "now", "right now", "as of now", "still"])
        ):
            semantic_spec["temporal_scope"] = "current"
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
    slots = [
        str(slot).strip()
        for slot in list(raw.get("requested_slots") or [])
        if str(slot).strip() in SEMANTIC_SLOT_VOCABULARY
    ]
    normalized = {
        "requested_slots": _normalize_requested_slots(slots),
        "requested_attributes": _normalize_open_attributes(raw.get("requested_attributes")),
        "attribute_bindings": _normalize_attribute_bindings(raw.get("attribute_bindings")),
    }
    for key, allowed_values in SEMANTIC_SCOPE_VALUES.items():
        value = str(raw.get(key) or "").strip().lower()
        normalized[key] = value if value in allowed_values else "unspecified"
    normalized["requires_entity_resolution"] = bool(raw.get("requires_entity_resolution"))
    return normalized


def _as_string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


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
    valid_bindings: list[dict[str, str]] = []
    for binding in bindings:
        attribute = binding["attribute"]
        support_span = binding["support_span"]
        if support_span.lower() not in question_text:
            continue
        attribute_tokens = set(re.findall(r"[a-z0-9]+", attribute.lower()))
        # Entity identity and requested properties have separate semantic roles.
        # Reject an open attribute that merely renames a planner-resolved entity.
        if attribute_tokens and any(
            len(attribute_tokens & entity_tokens) / len(attribute_tokens) >= 0.8
            for entity_tokens in entity_token_sets
            if entity_tokens
        ):
            continue
        valid_bindings.append(binding)

    # A property phrase may be mapped to both a legacy slot name and a more
    # precise open attribute. Keep the binding best aligned with its evidence
    # span so contract size cannot be inflated by duplicate aliases.
    best_by_span: dict[str, tuple[float, int, str]] = {}
    for index, binding in enumerate(valid_bindings):
        attribute = binding["attribute"]
        span_key = " ".join(re.findall(r"[a-z0-9]+", binding["support_span"].lower()))
        attribute_tokens = set(re.findall(r"[a-z0-9]+", attribute.lower()))
        span_tokens = set(span_key.split())
        alignment = len(attribute_tokens & span_tokens) / max(1, len(attribute_tokens))
        candidate = (alignment, -index, attribute)
        if span_key not in best_by_span or candidate > best_by_span[span_key]:
            best_by_span[span_key] = candidate
    grounded_from_bindings = [
        value[2]
        for value in sorted(best_by_span.values(), key=lambda item: -item[1])
    ]
    if grounded_from_bindings:
        return grounded_from_bindings, True
    return _ground_requested_attributes(raw_attributes, question), False


def _semantic_contract_quality(semantic_spec: dict[str, Any]) -> tuple[int, int]:
    attributes = list(dict.fromkeys(semantic_spec.get("requested_attributes") or []))
    return len(attributes), len(set(attributes))


def _semantic_contract_needs_slot_audit(semantic_spec: object) -> bool:
    """Detect an incomplete slot-to-attribute contract independent of domain."""
    if not isinstance(semantic_spec, dict):
        return False
    slots = list(dict.fromkeys(
        str(slot).strip()
        for slot in list(semantic_spec.get("requested_slots") or [])
        if str(slot).strip()
    ))
    attributes = list(dict.fromkeys(
        str(attribute).strip()
        for attribute in list(semantic_spec.get("requested_attributes") or [])
        if str(attribute).strip()
    ))
    return bool(slots and len(attributes) < len(slots))


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
        if attributes and support_span:
            normalized.append({"attribute": attributes[0], "support_span": support_span})
    return normalized


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
    request_shape = str(semantic_spec.get("request_shape") or "unspecified").lower()
    requires_field_contract = request_shape in {"list", "mixed"} or len(target_entities) > 1
    if not requires_field_contract:
        return True
    if "requested_attributes" in semantic_spec:
        return bool(requested_attributes)
    return bool(requested_slots)


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
