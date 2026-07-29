"""Configurable, deterministic policy conflict resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from gov_mem.policy_schema import MemoryStatus, PermissionState


@dataclass(frozen=True)
class ConflictResolution:
    effect: str
    winning_policy_id: str | None
    trace: tuple[str, ...]


def _time_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (0, "")
    try:
        return (1, datetime.fromisoformat(value).isoformat())
    except ValueError:
        return (1, value)


def resolve_permission(
    permissions: tuple[PermissionState, ...],
    *,
    memory_status: MemoryStatus,
    requester: str | None,
    default_effect: str = "unknown",
) -> ConflictResolution:
    """Apply status > revoke/deny > specific > recency priority."""
    trace: list[str] = []
    if memory_status in {MemoryStatus.DELETED, MemoryStatus.FORGOTTEN, MemoryStatus.INACCESSIBLE}:
        trace.append(f"memory status {memory_status.value} overrides ordinary permissions")
        return ConflictResolution("deny", None, tuple(trace))
    if not requester:
        return ConflictResolution("unknown", None, ("requester is not resolved",))
    candidates = [item for item in permissions if item.grantee in {None, requester}]
    if not candidates:
        return ConflictResolution(default_effect, None, ("no applicable permission chain",))
    # Specificity wins first; within one scope, the latest effective policy
    # wins. A deny/revoke breaks an exact timestamp tie only. This permits a
    # later valid grant to replace an earlier revoke at the same scope while a
    # specific deny still overrides a general allow.
    candidates.sort(
        key=lambda item: (
            item.specificity,
            _time_key(item.valid_from),
            1 if item.revoked or item.effect == "deny" else 0,
        ),
        reverse=True,
    )
    winner = candidates[0]
    effect = "deny" if winner.revoked or winner.effect == "deny" else "allow"
    trace.append(f"winner={winner.policy_id} specificity={winner.specificity} effect={effect}")
    if len(candidates) > 1:
        trace.append(f"resolved {len(candidates)} conflicting permission records")
    return ConflictResolution(effect, winner.policy_id, tuple(trace))
