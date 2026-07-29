"""Applicable-policy selection over explicit PolicyState."""

from __future__ import annotations

from dataclasses import dataclass

from gov_mem.policy_schema import PermissionState, PolicyState, QueryIntent


@dataclass(frozen=True)
class ApplicablePolicies:
    permissions: tuple[PermissionState, ...]
    policy_ids: tuple[str, ...]
    trace: tuple[str, ...]


def _matches(permission: PermissionState, intent: QueryIntent) -> bool:
    if permission.operation not in {"access", intent.requested_operation, "*"}:
        return False
    if permission.grantee and intent.requester and permission.grantee != intent.requester:
        return False
    if permission.scope and intent.target_scope and permission.scope != intent.target_scope:
        return False
    if permission.target_subject and intent.target_subject:
        return permission.target_subject.lower() in intent.target_subject.lower() or intent.target_subject.lower() in permission.target_subject.lower()
    return True


def select_applicable_policies(state: PolicyState, intent: QueryIntent) -> ApplicablePolicies:
    selected = tuple(item for item in state.permission_relations if _matches(item, intent))
    return ApplicablePolicies(
        permissions=selected,
        policy_ids=tuple(item.policy_id for item in selected),
        trace=(f"selected {len(selected)} applicable permission records",),
    )
