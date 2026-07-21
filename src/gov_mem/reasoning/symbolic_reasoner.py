from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from gov_mem.reasoning.governance_decision import GovernanceDecision, governance_decision_to_dict
from gov_mem.reasoning.predicates import Predicate, predicate_to_dict
from gov_mem.reasoning.proof_trace import ProofStep, proof_step_to_dict
from gov_mem.retrieval.retrieval_bundle import GovernedRetrievalBundle


LOGISTICS_SLOTS = {
    "date",
    "time",
    "arrival_time",
    "location",
    "provider",
    "procedure",
    "visit_type",
    "status",
    "visit_window",
    "entry_method",
    "parking_pass",
    "arrival_contact_rule",
    "approved_areas",
    "package_rule",
}

SENSITIVE_SLOTS = {
    "medication",
    "instruction",
    "timing",
    "dosage",
    "condition",
    "phone",
    "contact_method",
    "backup_contact",
    "approved_budget",
    "approved_discount_cap",
    "monthly_stipend",
    "safe_wording",
    "blocker",
    "target_date",
    "consent_scope",
}

ROLE_ALLOWANCES = {
    "owner": {"*"},
    "patient": {"*"},
    "self": {"*"},
    "clinician": {"*"},
    "nurse": {"*"},
    "pharmacist": {"*"},
    "authorized_staff": {"*"},
    "family": set(LOGISTICS_SLOTS),
    "caregiver": set(LOGISTICS_SLOTS),
    "delegate": set(LOGISTICS_SLOTS),
    "guest": set(LOGISTICS_SLOTS),
}

RELATION_ALLOWANCES = {
    "owner": {"*"},
    "authorized_staff": {"*"},
    "family": set(LOGISTICS_SLOTS),
    "caregiver": set(LOGISTICS_SLOTS),
    "delegate": set(LOGISTICS_SLOTS),
    "guest": set(LOGISTICS_SLOTS),
    None: set(),
}

TRUSTED_INTERNAL_RELATIONS = {"owner", "authorized_staff"}
EDUCATION_PARTIAL_ACCESS_SAFE_SLOTS = {"safe_wording", "public_event_date"}
EDUCATION_FINANCE_SLOTS = {"monthly_stipend", "approved_budget", "approved_discount_cap"}


class SymbolicGovernanceReasoner:
    def decide(
        self,
        *,
        query: str,
        requester_id: str | None,
        owner_id: str | None,
        requester_role: str | None,
        relation_to_owner: str | None,
        required_slot_plan: dict[str, Any],
        retrieval_bundle: GovernedRetrievalBundle,
        runtime_skill_bundle: dict[str, Any] | None = None,
        explicit_deleted_request: bool | None = None,
        explicit_historical_request: bool | None = None,
        explicit_current_request: bool | None = None,
    ) -> dict[str, Any]:
        predicates: list[Predicate] = []
        proof_steps: list[ProofStep] = []
        rules_fired: list[str] = []
        activated_rules = list((runtime_skill_bundle or {}).get("activated_rules") or [])
        explicit_deleted_request = (
            bool((runtime_skill_bundle or {}).get("explicit_deleted_request"))
            if explicit_deleted_request is None
            else bool(explicit_deleted_request)
        )
        explicit_historical_request = (
            bool((runtime_skill_bundle or {}).get("explicit_historical_request"))
            if explicit_historical_request is None
            else bool(explicit_historical_request)
        )
        explicit_current_request = (
            bool((runtime_skill_bundle or {}).get("explicit_current_request"))
            if explicit_current_request is None
            else bool(explicit_current_request)
        )
        required_slots = list(required_slot_plan.get("required_slots") or [])
        required_slots = list(dict.fromkeys(required_slots))

        utility_atoms = list(retrieval_bundle.utility_evidence.atoms or [])
        governance = retrieval_bundle.governance_evidence
        graph_paths = list(retrieval_bundle.graph_paths or [])
        utility_facts = list(retrieval_bundle.utility_evidence.facts or [])
        lowered_query = str(query or "").lower()

        if requester_id:
            predicates.append(self._predicate("requester", (requester_id,), [f"requester:{requester_id}"], 1.0))
        if owner_id:
            predicates.append(self._predicate("owner", (owner_id,), [f"owner:{owner_id}"], 1.0))
        if requester_role:
            predicates.append(self._predicate("requester_has_role", (requester_id, requester_role), [f"role:{requester_role}"], 0.95))
        if relation_to_owner:
            predicates.append(
                self._predicate(
                    "owner_requester_relation",
                    (requester_id, owner_id, relation_to_owner),
                    [f"relation:{relation_to_owner}"],
                    0.95,
                )
            )

        fact_slots: dict[str, set[str]] = defaultdict(set)
        fact_state: dict[str, str] = {}
        fact_sources: dict[str, list[str]] = defaultdict(list)
        for atom in utility_atoms:
            fact_id = str(atom.get("atom_id") or atom.get("text") or "fact")
            text = str(atom.get("text") or "")
            lifecycle = str(atom.get("lifecycle") or "active")
            predicates.append(self._predicate("fact", (fact_id,), [fact_id], float(atom.get("confidence") or 0.7)))
            fact_state[fact_id] = lifecycle
            fact_sources[fact_id].append(fact_id)
            if lifecycle == "active":
                predicates.append(self._predicate("fact_active", (fact_id,), [fact_id], float(atom.get("confidence") or 0.7)))
            elif lifecycle == "deleted":
                predicates.append(self._predicate("fact_deleted", (fact_id,), [fact_id], float(atom.get("confidence") or 0.7)))
            elif lifecycle == "superseded":
                predicates.append(self._predicate("fact_superseded", (fact_id, fact_id), [fact_id], float(atom.get("confidence") or 0.7)))
            elif lifecycle == "canceled":
                predicates.append(self._predicate("fact_canceled", (fact_id,), [fact_id], float(atom.get("confidence") or 0.7)))
            for slot_name, slot_value in dict(atom.get("slots") or {}).items():
                predicates.append(
                    self._predicate(
                        "slot",
                        (fact_id, str(slot_name), slot_value),
                        [fact_id],
                        float(atom.get("confidence") or 0.7),
                    )
                )
                fact_slots[fact_id].add(str(slot_name))

        for slot_name in required_slots:
            predicates.append(self._predicate("query_requires", (slot_name,), [f"required:{slot_name}"], 1.0))

        policy_denies: dict[str, list[str]] = defaultdict(list)
        policy_allows: dict[str, list[str]] = defaultdict(list)
        policy_projections: dict[str, set[str]] = defaultdict(set)
        scope_denies: dict[str, set[str]] = defaultdict(set)
        scope_allows: dict[str, set[str]] = defaultdict(set)

        for row in governance.roles:
            for scope in row.get("access_scope") or []:
                predicates.append(self._predicate("requester_has_role", (requester_id, scope), [str(row.get("atom_id"))], float(row.get("confidence") or 0.7)))

        for row in governance.policies:
            atom_id = str(row.get("atom_id") or row.get("text") or "policy")
            text = str(row.get("text") or "")
            binding = dict((row.get("provenance") or {}).get("policy_binding") or {})
            # Policy extraction can carry incidental event slots. Never let those
            # hide an explicitly named protected slot in the policy text.
            bound_slots = [str(slot) for slot in list(binding.get("slots") or []) if str(slot)]
            policy_slots = bound_slots or list(dict.fromkeys([
                *list((row.get("slots") or {}).keys()),
                *self._infer_slot_names_from_text(text),
            ]))
            scopes = list(binding.get("scopes") or row.get("access_scope") or []) or ([relation_to_owner] if relation_to_owner else [])
            binding_effect = str(binding.get("effect") or "").lower()
            denies = binding_effect == "deny" if binding_effect else self._text_denies(text)
            safe_projection = bool(binding.get("minimal_projection")) or self._text_defines_safe_projection(text)
            principal_boundary = self._text_is_principal_boundary(text)
            if denies and relation_to_owner != "owner" and (safe_projection or principal_boundary):
                scopes = list(dict.fromkeys([
                    *scopes,
                    *([requester_role] if requester_role else []),
                    *([relation_to_owner] if relation_to_owner else []),
                ]))
            for scope in scopes:
                for slot_name in policy_slots or required_slots:
                    if safe_projection and not denies:
                        policy_projections[scope].add(slot_name)
                        policy_allows[scope].append(slot_name)
                        predicates.append(self._predicate("policy_allows", (scope, slot_name), [atom_id], float(row.get("confidence") or 0.7)))
                    elif denies:
                        policy_denies[scope].append(slot_name)
                        predicates.append(self._predicate("policy_denies", (scope, slot_name), [atom_id], float(row.get("confidence") or 0.7)))
                    else:
                        policy_allows[scope].append(slot_name)
                        predicates.append(self._predicate("policy_allows", (scope, slot_name), [atom_id], float(row.get("confidence") or 0.7)))

        for scope, allowed in ROLE_ALLOWANCES.items():
            if requester_role and scope == requester_role:
                for slot_name in (required_slots or []):
                    if "*" in allowed or slot_name in allowed:
                        scope_allows[scope].add(slot_name)
                        predicates.append(self._predicate("scope_allows", (scope, slot_name), [f"role_default:{scope}"], 0.8))
                    else:
                        scope_denies[scope].add(slot_name)
                        predicates.append(self._predicate("scope_denies", (scope, slot_name), [f"role_default:{scope}"], 0.8))

        relation_allow = RELATION_ALLOWANCES.get(relation_to_owner, set())
        for slot_name in required_slots:
            if "*" in relation_allow or slot_name in relation_allow:
                predicates.append(self._predicate("scope_allows", (relation_to_owner, slot_name), [f"relation_default:{relation_to_owner}"], 0.85))
                if relation_to_owner:
                    scope_allows[relation_to_owner].add(slot_name)
            elif relation_to_owner is not None:
                predicates.append(self._predicate("scope_denies", (relation_to_owner, slot_name), [f"relation_default:{relation_to_owner}"], 0.85))
                scope_denies[relation_to_owner].add(slot_name)

        deleted_fact_ids: set[str] = set()
        superseded_fact_ids: set[str] = set()
        deleted_provenance: dict[str, list[str]] = defaultdict(list)
        superseded_provenance: dict[str, list[str]] = defaultdict(list)
        for path in graph_paths:
            edge_type = str(path.get("edge_type") or "")
            source_id = str(path.get("source_id") or "")
            target_id = str(path.get("target_id") or "")
            if edge_type == "deletes":
                deleted_fact_ids.add(target_id)
                deleted_provenance[target_id].append(source_id)
                predicates.append(self._predicate("block_reconstruction", (target_id,), [source_id], 0.95))
                rules_fired.append("graph_deletion_blocks_reconstruction")
                proof_steps.append(
                    ProofStep(
                        rule_name="graph_deletion_blocks_reconstruction",
                        conclusion=f"block_reconstruction({target_id})",
                        supporting_predicates=[f"graph_path:{edge_type}", f"target:{target_id}"],
                        evidence=[source_id, target_id],
                        confidence=0.95,
                    )
                )
            if edge_type == "supersedes":
                superseded_fact_ids.add(target_id)
                superseded_provenance[target_id].append(source_id)
                predicates.append(self._predicate("fact_superseded", (target_id, source_id), [source_id, target_id], 0.92))
                rules_fired.append("graph_supersession_prefers_latest")
                proof_steps.append(
                    ProofStep(
                        rule_name="graph_supersession_prefers_latest",
                        conclusion=f"prefer_latest({source_id}, {target_id})",
                        supporting_predicates=[f"graph_path:{edge_type}", f"target:{target_id}"],
                        evidence=[source_id, target_id],
                        confidence=0.92,
                    )
                )
            if edge_type == "denies":
                slot_name = str((path.get("attributes") or {}).get("slot_name") or "")
                role_name = str(path.get("source_label") or "")
                if slot_name:
                    policy_denies[role_name].append(slot_name)
                    predicates.append(self._predicate("policy_denies", (role_name, slot_name), [source_id], 0.9))
            if edge_type == "allows":
                slot_name = str((path.get("attributes") or {}).get("slot_name") or "")
                role_name = str(path.get("source_label") or "")
                if slot_name:
                    policy_allows[role_name].append(slot_name)
                    predicates.append(self._predicate("policy_allows", (role_name, slot_name), [source_id], 0.9))

        available_fact_ids = {str(row.get("memory_id") or row.get("content") or f"fact_{idx}") for idx, row in enumerate(utility_facts)}
        if not available_fact_ids:
            available_fact_ids = set(fact_slots.keys())

        allowed_slots: set[str] = set()
        denied_slots: set[str] = set()
        redacted_slots: set[str] = set()

        effective_role = requester_role or relation_to_owner or "unknown"
        for slot_name in required_slots:
            denied_by_policy = slot_name in set(policy_denies.get(effective_role, [])) or slot_name in set(policy_denies.get(relation_to_owner or "", []))
            allowed_by_policy = slot_name in set(policy_allows.get(effective_role, [])) or slot_name in set(policy_allows.get(relation_to_owner or "", []))
            denied_by_scope = slot_name in scope_denies.get(effective_role, set()) or slot_name in scope_denies.get(relation_to_owner, set())
            allowed_by_scope = slot_name in scope_allows.get(effective_role, set()) or slot_name in scope_allows.get(relation_to_owner, set())
            if denied_by_policy:
                denied_slots.add(slot_name)
                predicates.append(self._predicate("access_denied", (requester_id, slot_name), [f"policy_denies:{slot_name}"], 0.95))
                rules_fired.append("policy_denial_overrides_allow")
                proof_steps.append(
                    ProofStep(
                        rule_name="policy_denial_overrides_allow",
                        conclusion=f"access_denied({requester_id}, {slot_name})",
                        supporting_predicates=[f"policy_denies({slot_name})"],
                        evidence=[slot_name],
                        confidence=0.95,
                    )
                )
                continue
            if denied_by_scope and not allowed_by_policy:
                denied_slots.add(slot_name)
                predicates.append(self._predicate("access_denied", (requester_id, slot_name), [f"scope_denies:{slot_name}"], 0.88))
                rules_fired.append("relation_scope_restricts_sensitive_slots")
                proof_steps.append(
                    ProofStep(
                        rule_name="relation_scope_restricts_sensitive_slots",
                        conclusion=f"access_denied({requester_id}, {slot_name})",
                        supporting_predicates=[f"scope_denies({slot_name})"],
                        evidence=[slot_name],
                        confidence=0.88,
                    )
                )
                continue
            if allowed_by_policy or allowed_by_scope or relation_to_owner == "owner":
                allowed_slots.add(slot_name)
                predicates.append(self._predicate("access_allowed", (requester_id, slot_name), [f"allow:{slot_name}"], 0.9))
                if relation_to_owner == "owner":
                    rules_fired.append("owner_scope_default_allow")
            else:
                denied_slots.add(slot_name)
                predicates.append(self._predicate("access_denied", (requester_id, slot_name), [f"default_deny:{slot_name}"], 0.8))

        for slot_name in sorted(allowed_slots & denied_slots):
            denied_slots.add(slot_name)
            redacted_slots.add(slot_name)
            allowed_slots.discard(slot_name)
            predicates.append(self._predicate("must_redact", (slot_name,), [f"mixed_access:{slot_name}"], 0.9))
            rules_fired.append("mixed_access_requires_redaction")
            proof_steps.append(
                ProofStep(
                    rule_name="mixed_access_requires_redaction",
                    conclusion=f"must_redact({slot_name})",
                    supporting_predicates=[f"query_requires({slot_name})"],
                    evidence=[slot_name],
                    confidence=0.9,
                )
            )

        effective_projection_slots = set(policy_projections.get(effective_role, set())) | set(
            policy_projections.get(relation_to_owner or "", set())
        )

        non_owner_requester = relation_to_owner not in TRUSTED_INTERNAL_RELATIONS | {None} and requester_id != owner_id
        if "skill_reduce_sensitive_medical_slots" in activated_rules and non_owner_requester:
            affected = set(required_slots) & set(SENSITIVE_SLOTS)
            if affected:
                denied_slots.update(affected)
                allowed_slots.difference_update(affected)
                redacted_slots.update(affected)
                rules_fired.append("skill_reduce_sensitive_medical_slots")

        if "skill_household_logistics_scope_only" in activated_rules and relation_to_owner in {"family", "caregiver", "delegate", "guest"}:
            non_logistics = {slot for slot in required_slots if slot not in LOGISTICS_SLOTS}
            if non_logistics:
                denied_slots.update(non_logistics)
                allowed_slots.difference_update(non_logistics)
                redacted_slots.update(non_logistics)
                rules_fired.append("skill_household_logistics_scope_only")

        if "skill_education_financial_scope_guard" in activated_rules and non_owner_requester:
            affected = set(required_slots) & EDUCATION_FINANCE_SLOTS
            if affected:
                denied_slots.update(affected)
                allowed_slots.difference_update(affected)
                redacted_slots.update(affected)
                rules_fired.append("skill_education_financial_scope_guard")

        if (
            "skill_education_partial_access_redaction" in activated_rules
            and relation_to_owner in {"delegate", "family", "caregiver"}
        ):
            allowed_projection_slots = set(required_slots) & EDUCATION_PARTIAL_ACCESS_SAFE_SLOTS
            if allowed_projection_slots:
                allowed_slots.update(allowed_projection_slots)
                denied_slots.difference_update(allowed_projection_slots)
                redacted_slots.update(allowed_projection_slots)
                rules_fired.append("skill_education_partial_access_redaction")

        active_facts = sorted(fact_id for fact_id, lifecycle in fact_state.items() if lifecycle == "active")
        deleted_facts = sorted(set(deleted_fact_ids) | {fact_id for fact_id, lifecycle in fact_state.items() if lifecycle == "deleted"})
        superseded_facts = sorted(set(superseded_fact_ids) | {fact_id for fact_id, lifecycle in fact_state.items() if lifecycle == "superseded"})
        blocked_facts = sorted(set(deleted_facts))

        slot_has_active_support = {
            slot_name: any(slot_name in slots and fact_id not in blocked_facts for fact_id, slots in fact_slots.items())
            for slot_name in required_slots
        }

        if not utility_facts and not utility_atoms:
            action_constraint = "no_memory"
            rules_fired.append("no_utility_evidence_after_projection")
            proof_steps.append(
                ProofStep(
                    rule_name="no_utility_evidence_after_projection",
                    conclusion="action_constraint=no_memory",
                    supporting_predicates=[],
                    evidence=[],
                    confidence=0.98,
                )
            )
        else:
            deleted_required = {slot for slot in required_slots if not slot_has_active_support.get(slot, False) and blocked_facts}
            fully_denied = set(required_slots) and all(
                slot in denied_slots or slot in deleted_required for slot in required_slots
            )
            some_allowed = any(slot in allowed_slots for slot in required_slots)
            some_denied = any(slot in denied_slots or slot in deleted_required for slot in required_slots)
            denied_requested_slots = set(required_slots) & (denied_slots | deleted_required)
            allowed_requested_slots = set(required_slots) & allowed_slots
            deleted_reconstruction_query = (
                explicit_deleted_request and explicit_historical_request
            ) or bool(blocked_facts) and (
                explicit_deleted_request or explicit_historical_request
            )
            if deleted_reconstruction_query or (
                "skill_force_deleted_reconstruction_block" in activated_rules
                and blocked_facts
                and (explicit_deleted_request or explicit_historical_request)
            ):
                action_constraint = "no_memory"
                rules_fired.append(
                    "skill_force_deleted_reconstruction_block"
                    if "skill_force_deleted_reconstruction_block" in activated_rules
                    else "deleted_reconstruction_query_no_memory"
                )
                proof_steps.append(
                    ProofStep(
                        rule_name=(
                            "skill_force_deleted_reconstruction_block"
                            if "skill_force_deleted_reconstruction_block" in activated_rules
                            else "deleted_reconstruction_query_no_memory"
                        ),
                        conclusion="action_constraint=no_memory",
                        supporting_predicates=["block_reconstruction(f)"],
                        evidence=list(blocked_facts),
                        confidence=0.95,
                    )
                )
            elif fully_denied or (denied_requested_slots and not allowed_requested_slots):
                has_explicit_safe_projection = bool(effective_projection_slots) or self._has_policy_backed_safe_projection(
                    policies=governance.policies,
                    denied_slots=denied_requested_slots,
                )
                action_constraint = "answer_redacted" if has_explicit_safe_projection else "refuse"
                rules_fired.append(
                    "policy_backed_safe_projection"
                    if has_explicit_safe_projection
                    else "all_required_slots_denied_or_deleted"
                )
                proof_steps.append(
                    ProofStep(
                        rule_name="all_required_slots_denied_or_deleted",
                        conclusion=f"action_constraint={action_constraint}",
                        supporting_predicates=[f"query_requires({slot})" for slot in required_slots],
                        evidence=sorted(list(denied_slots | deleted_required)),
                        confidence=0.94,
                    )
                )
            elif some_allowed and some_denied:
                action_constraint = "answer_redacted"
                rules_fired.append("mixed_access_requires_redaction")
            elif (
                "skill_education_partial_access_redaction" in activated_rules
                and relation_to_owner in {"delegate", "family", "caregiver"}
                and any(slot in allowed_slots for slot in EDUCATION_PARTIAL_ACCESS_SAFE_SLOTS)
            ):
                action_constraint = "answer_redacted"
                rules_fired.append("skill_education_partial_access_redaction")
            elif all(slot in allowed_slots for slot in required_slots):
                action_constraint = "answer"
                rules_fired.append("all_required_slots_allowed")
                proof_steps.append(
                    ProofStep(
                        rule_name="all_required_slots_allowed",
                        conclusion="action_constraint=answer",
                        supporting_predicates=[f"access_allowed({slot})" for slot in required_slots],
                        evidence=sorted(allowed_slots),
                        confidence=0.92,
                    )
                )
            else:
                action_constraint = "no_memory"

        confidence = self._decision_confidence(
            action_constraint=action_constraint,
            required_slots=required_slots,
            allowed_slots=allowed_slots,
            denied_slots=denied_slots,
            blocked_facts=blocked_facts,
            graph_paths=graph_paths,
        )
        decision = GovernanceDecision(
            action_constraint=action_constraint,
            allowed_slots=sorted(allowed_slots),
            denied_slots=sorted(denied_slots),
            redacted_slots=sorted(redacted_slots or denied_slots),
            explicit_deleted_request=explicit_deleted_request,
            explicit_historical_request=explicit_historical_request,
            explicit_current_request=explicit_current_request,
            blocked_facts=blocked_facts,
            active_facts=active_facts,
            superseded_facts=superseded_facts,
            deleted_facts=deleted_facts,
            rules_fired=list(dict.fromkeys(rules_fired)),
            proof_trace=[proof_step_to_dict(step) for step in proof_steps],
            confidence=confidence,
        )
        decision_payload = governance_decision_to_dict(decision)
        decision_payload["projection_slots"] = sorted(effective_projection_slots)
        return {
            "predicates": [predicate_to_dict(predicate) for predicate in predicates],
            "decision": decision_payload,
        }

    @staticmethod
    def _predicate(name: str, args: tuple[Any, ...], provenance: list[str], confidence: float) -> Predicate:
        return Predicate(name=name, args=args, truth=True, provenance=list(provenance), confidence=float(confidence))

    @staticmethod
    def _infer_slot_names_from_text(text: str) -> list[str]:
        lowered = text.lower()
        slot_names: list[str] = []
        mapping = {
            "medication": ["medication", "medicine", "drug", "dose"],
            "instruction": ["instruction", "plan", "details", "results", "notes"],
            "phone": ["phone", "number", "callback", "contact detail", "direct mobile"],
            "target_date": ["target date", "launch date", "pilot target", "date"],
            "approved_budget": ["budget"],
            "approved_discount_cap": ["discount", "cap"],
            "monthly_stipend": ["stipend", "support amount"],
            "safe_wording": ["wording", "sponsor name", "sponsor"],
            "blocker": ["blocker", "hold", "remaining issue"],
            "family_release_scope": ["release scope", "shareable", "shared summary"],
            "access_room": ["private room", "private suite"],
            "access_badge": ["badge", "private code"],
            "visit_window": ["window", "time"],
            "location": ["location", "suite", "room"],
        }
        for slot_name, tokens in mapping.items():
            if any(token in lowered for token in tokens):
                slot_names.append(slot_name)
        return list(dict.fromkeys(slot_names))

    @staticmethod
    def _text_denies(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in [
            "do not", "not authorized", "not my", "without naming",
            "logistics only", "unavailable", "cannot share", "not exact",
            "only is shareable", "summary is shareable",
        ])

    @staticmethod
    def _text_defines_safe_projection(text: str) -> bool:
        lowered = text.lower()
        has_projection = any(token in lowered for token in [
            "shareable", "public summary", "safe summary", "logistics only",
            "schedule only", "may be described only as",
        ])
        has_boundary = any(token in lowered for token in [
            "only", "not exact", "cannot share", "do not share", "private",
        ])
        return has_projection and has_boundary

    @staticmethod
    def _text_is_principal_boundary(text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in [
            "without assignment", "not assigned", "across services",
            "not authorized", "outside the care team", "without consent",
        ])

    def _has_policy_backed_safe_projection(
        self,
        *,
        policies: list[dict[str, Any]],
        denied_slots: set[str],
    ) -> bool:
        """Redaction needs a policy-backed substitute for the requested slot."""
        for row in policies:
            text = str(row.get("text") or "")
            policy_slots = set(self._infer_slot_names_from_text(text))
            if self._text_defines_safe_projection(text) and policy_slots & denied_slots:
                return True
        return False

    @staticmethod
    def _decision_confidence(
        *,
        action_constraint: str,
        required_slots: list[str],
        allowed_slots: set[str],
        denied_slots: set[str],
        blocked_facts: list[str],
        graph_paths: list[dict[str, Any]],
    ) -> float:
        confidence = 0.72
        confidence += min(len(required_slots), 5) * 0.03
        confidence += min(len(graph_paths), 5) * 0.025
        confidence += min(len(blocked_facts), 3) * 0.03
        if action_constraint == "answer":
            confidence += 0.08 if required_slots and all(slot in allowed_slots for slot in required_slots) else 0.0
        if action_constraint in {"refuse", "answer_redacted"} and denied_slots:
            confidence += 0.08
        if action_constraint == "no_memory" and blocked_facts:
            confidence += 0.1
        return min(confidence, 0.98)
