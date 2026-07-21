from __future__ import annotations

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.memory.governed_atom import GovernedMemoryAtom
from gov_mem.retrieval.governance_retriever import GovernanceRetriever
from gov_mem.retrieval.retrieval_bundle import GovernedRetrievalBundle, retrieval_bundle_to_dict
from gov_mem.retrieval.utility_retriever import UtilityRetriever


class DualChannelRetriever:
    def __init__(self) -> None:
        self.utility_retriever = UtilityRetriever()
        self.governance_retriever = GovernanceRetriever()

    def retrieve(
        self,
        *,
        query: str,
        requester: str | None,
        owner: str | None,
        relation: str | None,
        evidence: list[RetrievedEvidence],
        governed_atoms: list[GovernedMemoryAtom],
        graph_paths: list[dict],
        requested_attributes: list[str] | None = None,
        semantic_utility_source_message_ids: set[str] | None = None,
    ) -> GovernedRetrievalBundle:
        utility_evidence = self.utility_retriever.retrieve(
            evidence=evidence,
            governed_atoms=governed_atoms,
            requested_attributes=requested_attributes,
            semantic_source_message_ids=semantic_utility_source_message_ids,
        )
        governance_evidence = self.governance_retriever.retrieve(
            governed_atoms=governed_atoms,
            graph_paths=graph_paths,
            requested_attributes=requested_attributes,
            requester_relation=relation,
            owner_id=owner,
        )
        return GovernedRetrievalBundle(
            query=query,
            requester=requester,
            owner=owner,
            relation=relation,
            utility_evidence=utility_evidence,
            governance_evidence=governance_evidence,
            graph_paths=list(graph_paths or []),
            retrieval_trace={
                "utility_fact_count": len(utility_evidence.facts),
                "utility_atom_count": len(utility_evidence.atoms),
                "governance_policy_count": len(governance_evidence.policies),
                "governance_relation_count": len(governance_evidence.relations),
                "governance_deletion_count": len(governance_evidence.deletions),
                "governance_supersession_count": len(governance_evidence.supersessions),
                "graph_path_count": len(graph_paths or []),
                "utility_selected_count": len(utility_evidence.selected_memory_ids),
                "governance_selected_policy_count": len(governance_evidence.selected_policy_atom_ids),
            },
        )


__all__ = ["DualChannelRetriever", "retrieval_bundle_to_dict"]
