"""Turn a PolicyDecision into an explicit execution plan."""

from __future__ import annotations

from dataclasses import dataclass

from gov_mem.policy_schema import PolicyAction, PolicyDecision


@dataclass(frozen=True)
class ExecutionPlan:
    action: PolicyAction
    allowed_memory_ids: tuple[str, ...]
    blocked_memory_ids: tuple[str, ...]
    answer_allowed: bool
    state_update: dict[str, object]
    reason: str | None


def build_execution_plan(decision: PolicyDecision) -> ExecutionPlan:
    answer_allowed = decision.action == PolicyAction.ALLOW
    state_update: dict[str, object] = {}
    if decision.action in {PolicyAction.UPDATE, PolicyAction.DELETE, PolicyAction.FORGET}:
        state_update = {
            "operation": decision.action.value.lower(),
            "target_memory_ids": list(decision.allowed_memory_ids),
        }
    target_ids = decision.allowed_memory_ids if decision.action in {PolicyAction.ALLOW, PolicyAction.UPDATE, PolicyAction.DELETE, PolicyAction.FORGET} else ()
    return ExecutionPlan(
        action=decision.action,
        allowed_memory_ids=target_ids if answer_allowed else (),
        blocked_memory_ids=decision.blocked_memory_ids,
        answer_allowed=answer_allowed,
        state_update={**state_update, "target_memory_ids": list(target_ids)} if state_update else {},
        reason=decision.abstain_reason,
    )
