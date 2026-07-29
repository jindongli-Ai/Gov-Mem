"""Audit records for the stateful governance pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gov_mem.policy_schema import Provenance, schema_to_dict


@dataclass
class AuditTrace:
    stage: str
    event: str
    details: dict[str, Any] = field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return schema_to_dict(self)


def audit(stage: str, event: str, **details: Any) -> AuditTrace:
    return AuditTrace(stage=stage, event=event, details=details)
