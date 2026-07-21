from __future__ import annotations

from gov_mem.memory.governed_atom import GovernedMemoryAtom, governed_atom_to_dict
from gov_mem.retrieval.retrieval_bundle import GovernanceEvidenceBundle


class GovernanceRetriever:
    def retrieve(
        self,
        *,
        governed_atoms: list[GovernedMemoryAtom],
        graph_paths: list[dict],
        requested_attributes: list[str] | None = None,
        requester_relation: str | None = None,
        owner_id: str | None = None,
    ) -> GovernanceEvidenceBundle:
        requested = {str(slot).strip() for slot in (requested_attributes or []) if str(slot).strip()}
        roles = [governed_atom_to_dict(atom) for atom in governed_atoms if atom.atom_type == "role_atom"]
        policies = [governed_atom_to_dict(atom) for atom in governed_atoms if atom.atom_type in {"policy_atom", "permission_atom"}]
        relations = [governed_atom_to_dict(atom) for atom in governed_atoms if atom.atom_type == "relation_atom"]
        deletions = [governed_atom_to_dict(atom) for atom in governed_atoms if atom.atom_type == "deletion_atom"]
        supersessions = [governed_atom_to_dict(atom) for atom in governed_atoms if atom.atom_type == "supersession_atom"]
        ranked_policies: list[tuple[float, str, dict]] = []
        for atom in policies:
            binding = dict((atom.get("provenance") or {}).get("policy_binding") or {})
            bound_slots = {str(slot) for slot in list(binding.get("slots") or []) if str(slot)}
            scopes = {str(scope) for scope in list(binding.get("scopes") or []) if str(scope)}
            score = 2.0 * len(bound_slots & requested)
            if requester_relation and requester_relation in scopes:
                score += 1.0
            if owner_id and str(atom.get("owner_id") or "") == str(owner_id):
                score += 0.5
            ranked_policies.append((score, str(atom.get("atom_id") or ""), atom))
        ranked_policies.sort(key=lambda item: (item[0], item[1]), reverse=True)
        policies = [atom for _, _, atom in ranked_policies]
        selection_trace = [
            {
                "atom_id": atom_id,
                "selection_score": score,
                "matched_requested_attributes": sorted(
                    set((atom.get("provenance") or {}).get("policy_binding", {}).get("slots") or []) & requested
                ),
            }
            for score, atom_id, atom in ranked_policies
        ]
        return GovernanceEvidenceBundle(
            roles=roles,
            policies=policies,
            relations=relations,
            deletions=deletions,
            supersessions=supersessions,
            denied_scopes=[],
            graph_paths=list(graph_paths or []),
            selected_policy_atom_ids=[atom_id for _, atom_id, _ in ranked_policies],
            selection_trace=selection_trace,
        )
