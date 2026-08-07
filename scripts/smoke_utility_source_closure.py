#!/usr/bin/env python3
"""Selected source turns admit their facts, but not unrelated or policy facts."""

from __future__ import annotations

from gov_mem.legacy.semantic_alignment import _candidate_slots
from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.memory.governed_atom import GovernedMemoryAtom


def _atom(atom_id, atom_type, text, slots, message_id, source_memory_id):
    return GovernedMemoryAtom(
        atom_id=atom_id, atom_type=atom_type, text=text, slots=slots,
        owner_id="owner", subject_id="record", speaker_id="owner", source_turn=1,
        timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
        access_scope=[], related_entities=[], confidence=1.0,
        provenance={
            "source_memory_id": source_memory_id,
            "source_message_ids": [message_id],
            "evidence_span": text,
        },
    )


def main() -> None:
    selected_fact = _atom(
        "selected-fact", "fact_atom", "The record value is alpha.", {"field": "alpha"}, "m-selected", "fact-a"
    )
    unrelated_fact = _atom(
        "unrelated-fact", "fact_atom", "The record value is beta.", {"field": "beta"}, "m-other", "fact-b"
    )
    policy = _atom(
        "policy", "permission_atom", "A collaborator may receive the field.", {"field": "policy-text"}, "m-policy", "policy-source"
    )
    graph = GovernedGraphBuilder().build(
        graph_id="source-closure", instance_id="source-closure",
        atoms=[selected_fact, unrelated_fact, policy],
    )
    rows = _candidate_slots(
        graph=graph,
        owner_id="owner",
        utility_atom_ids=set(),
        utility_source_message_ids={"m-selected"},
    )
    assert len(rows) == 1, rows
    assert rows[0]["source_atom_id"] == "selected-fact", rows
    assert rows[0]["source_message_ids"] == ["m-selected"], rows
    print("utility_source_closure_smoke=PASS")


if __name__ == "__main__":
    main()
