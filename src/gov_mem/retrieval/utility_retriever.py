from __future__ import annotations

from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.memory.governed_atom import GovernedMemoryAtom, governed_atom_to_dict
from gov_mem.retrieval.retrieval_bundle import UtilityEvidenceBundle


class UtilityRetriever:
    def retrieve(
        self,
        *,
        evidence: list[RetrievedEvidence],
        governed_atoms: list[GovernedMemoryAtom],
        requested_attributes: list[str] | None = None,
        semantic_source_message_ids: set[str] | None = None,
        max_items: int = 16,
    ) -> UtilityEvidenceBundle:
        requested = {str(slot).strip() for slot in (requested_attributes or []) if str(slot).strip()}
        utility_atoms = [
            atom
            for atom in governed_atoms
            if atom.atom_type in {"fact_atom", "event_atom"}
        ]
        chunks = [
            {
                "memory_id": row.memory_id,
                "content": row.content,
                "memory_type": row.memory_type,
                "score": float(row.score),
                "source": row.retrieval_source,
                "metadata": dict(row.metadata or {}),
                "source_message_ids": list(row.source_message_ids or []),
            }
            for row in evidence
            if row.scope != "atomic_memory"
        ]
        ranked_rows: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for row in evidence:
            frame_slots = dict((row.metadata or {}).get("slots") or {})
            atom_slot_overlap = set(frame_slots) & requested
            # Retrieval scores and schema alignment are general evidence
            # signals; neither uses domain labels or policy wording.
            score = float(row.score) + (2.0 * len(atom_slot_overlap))
            payload = {
                "memory_id": row.memory_id,
                "content": row.content,
                "memory_type": row.memory_type,
                "score": score,
                "slots": frame_slots,
                "source_message_ids": list(row.source_message_ids or []),
            }
            trace = {
                "memory_id": row.memory_id,
                "base_score": float(row.score),
                "matched_requested_attributes": sorted(atom_slot_overlap),
                "selection_score": score,
            }
            ranked_rows.append((score, payload, trace))
        ranked_rows.sort(key=lambda item: (item[0], str(item[1]["memory_id"])), reverse=True)
        # With a typed contract, channel selection is bounded. Without one, do
        # not discard evidence merely because the planner was incomplete.
        selected_rows = ranked_rows[:max_items] if requested else ranked_rows
        fact_rows = [row for _, row, _ in selected_rows]
        selection_trace = [trace for _, _, trace in selected_rows]
        selected_source_message_ids = {
            str(message_id)
            for row in selected_rows
            for message_id in list(row[1].get("source_message_ids") or [])
            if str(message_id)
        }
        # Chunk retrieval selects source messages before atoms are produced.
        # Preserve that evidence-local proposal when open claim labels differ
        # from a query's canonical property name; it is not a lexical or
        # domain-specific inference and cannot itself authorize disclosure.
        if not selected_source_message_ids:
            selected_source_message_ids = {
                str(message_id)
                for row in evidence
                if str(row.memory_id) in {str(item["memory_id"]) for item in fact_rows}
                for message_id in list(row.source_message_ids or [])
                if str(message_id)
            }
        semantic_source_ids = {str(value) for value in list(semantic_source_message_ids or []) if str(value)}
        ranked_atoms = []
        for atom in utility_atoms:
            slots = {str(name) for name in dict(atom.slots or {})}
            source_message_ids = {
                str(message_id)
                for message_id in list((atom.provenance or {}).get("source_message_ids") or [])
                if str(message_id)
            }
            provenance_overlap = len(source_message_ids & selected_source_message_ids)
            score = (
                float(atom.confidence or 0.0)
                + (2.0 * len(slots & requested))
                + (3.0 * provenance_overlap)
            )
            ranked_atoms.append((score, str(atom.atom_id), atom))
        ranked_atoms.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return UtilityEvidenceBundle(
            facts=fact_rows,
            atoms=[governed_atom_to_dict(atom) for atom in utility_atoms],
            chunks=chunks,
            scores={
                "fact_count": len(fact_rows),
                "atom_count": len(utility_atoms),
                "chunk_count": len(chunks),
                "requested_attribute_count": len(requested),
            },
            selected_memory_ids=[str(row["memory_id"]) for row in fact_rows],
            selected_atom_ids=[atom_id for _, atom_id, _ in (ranked_atoms[:max_items] if requested else ranked_atoms)],
            selected_source_message_ids=sorted(selected_source_message_ids | semantic_source_ids),
            selection_trace=selection_trace,
        )
