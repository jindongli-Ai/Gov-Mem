from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UtilityEvidenceBundle:
    facts: list[dict[str, Any]] = field(default_factory=list)
    atoms: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    selected_memory_ids: list[str] = field(default_factory=list)
    selected_atom_ids: list[str] = field(default_factory=list)
    selected_source_message_ids: list[str] = field(default_factory=list)
    selection_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GovernanceEvidenceBundle:
    roles: list[dict[str, Any]] = field(default_factory=list)
    policies: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    deletions: list[dict[str, Any]] = field(default_factory=list)
    supersessions: list[dict[str, Any]] = field(default_factory=list)
    denied_scopes: list[str] = field(default_factory=list)
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    selected_policy_atom_ids: list[str] = field(default_factory=list)
    selection_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GovernedRetrievalBundle:
    query: str
    requester: str | None
    owner: str | None
    relation: str | None
    utility_evidence: UtilityEvidenceBundle
    governance_evidence: GovernanceEvidenceBundle
    graph_paths: list[dict[str, Any]] = field(default_factory=list)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)


def retrieval_bundle_to_dict(bundle: GovernedRetrievalBundle) -> dict[str, Any]:
    return asdict(bundle)
