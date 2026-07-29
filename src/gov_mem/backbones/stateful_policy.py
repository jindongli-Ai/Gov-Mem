"""Stateful Policy Reasoning backbone.

The backbone has one governance boundary: PolicyDecision.  Content retrieval
is downstream of that decision and therefore cannot alter permissions.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from gov_mem.backbones.common import BackboneRunResult, build_reasoning_state
from gov_mem.data.schema import AnswerResult, GovernedActionDecision, MemoryInstance, QueryPlan
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient
from gov_mem.policy_schema import PolicyAction, schema_to_dict
from gov_mem.policy_state_builder import build_policy_state, policy_state_audit_dict
from gov_mem.policy_reasoner import StatefulPolicyReasoner
from gov_mem.controlled_retrieval import retrieve_allowed_memory
from gov_mem.execution_planner import build_execution_plan
from gov_mem.governance_executor import execute_policy_decision
from gov_mem.state_transition_engine import apply_execution_state_update


class StatefulPolicyBackbone:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        embedding_client: LLMClient,
        config: dict[str, Any],
        output_dir: Path,
        dataset_name: str,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = dataset_name

    def run_instance(self, instance: MemoryInstance) -> BackboneRunResult:
        # Stage 1: observable state construction.
        state = build_policy_state(instance, llm_client=self.llm_client, config=self.config)
        # The query plan is semantic telemetry, not a dataset label.
        plan = QueryPlan(
            query_type="stateful_policy_request",
            target_users=[instance.asking_user_id] if instance.asking_user_id else [],
            target_entities=[],
            required_memory_types=["policy_allowed_memory"],
            symbolic_filters={},
            dense_queries=[instance.question],
            reasoning_ops=["policy_state_transition"],
            semantic_spec={},
            planning_trace={"stage": "policy_state_construction"},
        )

        # Stage 2: the only authorization/reasoning decision boundary.
        reasoner = StatefulPolicyReasoner(llm_client=self.llm_client, config=self.config)
        decision = reasoner.decide(instance=instance, state=state)
        # Preserve the policy reasoner's query-grounded field contract in
        # telemetry.  The contract is derived from the natural-language query
        # and observable state; it is not an evaluator label or an answer.
        plan.semantic_spec = {
            "requested_attributes": list(decision.state_snapshot.get("requested_attributes") or []),
            "requested_topics": list(decision.state_snapshot.get("requested_topics") or []),
            "target_scope": decision.state_snapshot.get("target_scope"),
            "request_shape": "aggregate"
            if len(decision.state_snapshot.get("requested_attributes") or []) <= 1
            else "mixed",
        }
        plan.planning_trace = {
            "stage": "policy_state_construction",
            "semantic_contract_source": "stateful_query_intent",
        }

        # Stage 3: retrieval is a constrained operation over allowed IDs only.
        evidence, retrieval_debug = retrieve_allowed_memory(
            state=state,
            decision=decision,
            query=instance.question,
            embedding_client=self.embedding_client,
            embedding_model=str((self.config.get("embedding") or {}).get("model") or "text-embedding-3-large"),
            top_k=int((self.config.get("policy_reasoning") or {}).get("retrieval_top_k", 12)),
            retrieval_strategy=str(
                (self.config.get("policy_reasoning") or {}).get(
                    "retrieval_strategy", "global_authorized_topk"
                )
            ),
        )

        # Stage 4: execute structured action and record explicit state changes.
        execution_plan = build_execution_plan(decision)
        answer_result, execution = execute_policy_decision(
            instance=instance,
            decision=decision,
            plan=execution_plan,
            evidence=evidence,
            llm_client=self.llm_client,
            config=self.config,
        )
        next_state = apply_execution_state_update(
            state,
            action=execution.action,
            memory_ids=execution.accessed_memory_ids,
            evidence_text=execution.answer_text,
        )
        reasoning_state = build_reasoning_state(
            evidence,
            trace=[
                "stage1: policy state constructed from observable episode prefix",
                "stage2: stateful policy decision completed before content retrieval",
                f"stage3: retrieved {len(evidence)} items from allowed set only",
                "stage4: execution result recorded as structured state update",
                *decision.transition_trace,
            ],
        )
        reasoning_state.current_state_ledger = policy_state_audit_dict(
            state,
            blocked_memory_ids=set(decision.blocked_memory_ids),
        )
        reasoning_state.conflicts = [
            {"policy_ids": list(decision.applicable_policy_ids), "trace": list(decision.transition_trace)}
        ] if decision.applicable_policy_ids else []
        action_decision = GovernedActionDecision(
            action=answer_result.action,
            answer_mode="direct" if answer_result.action == "answer" else answer_result.action,
            privacy_decision="allowed" if decision.action == PolicyAction.ALLOW else "denied",
            forgetting_decision=decision.action.value.lower(),
            evidence_memory_ids=list(execution.accessed_memory_ids),
            rationale_summary=decision.abstain_reason or f"stateful policy action={decision.action.value}",
        )
        debug_payload = {
            "experiment_mode": "stateful_policy_reasoning",
            "workflow": [
                "policy_state_construction",
                "stateful_policy_reasoning",
                "policy_grounded_memory_execution",
                "execution_and_state_update",
            ],
            "policy_state": policy_state_audit_dict(state, blocked_memory_ids=set(decision.blocked_memory_ids)),
            "policy_state_after_execution": policy_state_audit_dict(next_state, blocked_memory_ids=set(decision.blocked_memory_ids)),
            "policy_decision": schema_to_dict(decision),
            "execution_plan": schema_to_dict(execution_plan),
            "execution_result": schema_to_dict(execution),
            "retrieval": retrieval_debug,
            "selected_evidence": [asdict(row) for row in evidence],
        }
        assert_runtime_payload_safe(debug_payload, context="stateful_policy_debug_payload")
        return BackboneRunResult(
            query_plan=plan,
            retrieval_result=retrieval_debug,
            reasoning_state=reasoning_state,
            action_decision=action_decision,
            answer_result=answer_result,
            debug_payload=debug_payload,
        )
