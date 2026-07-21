from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SymbolicRule:
    name: str
    description: str
    evaluator: Callable[..., Any]


DEFAULT_RULE_NAMES = [
    "graph_deletion_blocks_reconstruction",
    "graph_supersession_prefers_latest",
    "owner_scope_default_allow",
    "policy_denial_overrides_allow",
    "relation_scope_restricts_sensitive_slots",
    "mixed_access_requires_redaction",
    "all_required_slots_denied_or_deleted",
    "all_required_slots_allowed",
]
