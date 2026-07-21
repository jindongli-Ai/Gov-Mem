#!/usr/bin/env python3
"""A policy source cannot become answer evidence through a derived atom."""

from __future__ import annotations

from gov_mem.data.schema import MemoryInstance
from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.memory.amem_memory import AtomicMemory
from gov_mem.memory.governed_atom import GovernedMemoryAtom, adapt_atomic_memories_to_governed_atoms


def _atom(atom_id, atom_type, slots, source_memory_id):
    return GovernedMemoryAtom(
        atom_id=atom_id, atom_type=atom_type, text="Source text.", slots=slots,
        owner_id="owner", subject_id="subject", speaker_id="owner", source_turn=1,
        timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
        access_scope=[], related_entities=[],
        provenance={"source_memory_id": source_memory_id, "evidence_span": "Source text."}, confidence=1.0,
    )


def main() -> None:
    graph = GovernedGraphBuilder().build(
        graph_id="separation", instance_id="separation",
        atoms=[
            _atom("policy", "permission_atom", {"policy_scope": "limited"}, "shared-policy-source"),
            GovernedMemoryAtom(
                **{
                    **_atom("policy-derived-event", "event_atom", {"synthetic_field": "limited"}, "shared-policy-source").__dict__,
                    "provenance": {
                        "source_memory_id": "shared-policy-source", "evidence_span": "Source text.",
                        "policy_binding": {"effect": "allow"},
                    },
                }
            ),
            GovernedMemoryAtom(
                **{
                    **_atom("bound-fact", "event_atom", {"current_field": "value"}, "fact-source").__dict__,
                    "provenance": {
                        "source_memory_id": "fact-source", "evidence_span": "Source text.",
                        "source_role": "factual", "policy_binding": {"effect": "allow"},
                    },
                }
            ),
            _atom("fact", "fact_atom", {"actual_field": "value"}, "fact-source"),
        ],
    )
    slot_names = {
        node.attributes.get("slot_name")
        for node in graph.nodes if node.node_type == "SlotNode"
    }
    assert {"actual_field", "current_field"} <= slot_names, slot_names
    assert "policy_scope" not in slot_names and "synthetic_field" not in slot_names, slot_names

    mixed_record = AtomicMemory(
        memory_id="mixed-record", instance_id="separation", owner_user="operator",
        memory_type="logistics", content="Current recap: endpoint-A and portal only.",
        entities=[], slots={"endpoint": "endpoint-A", "channel": "portal only"},
        source_message_ids=["m-mixed"], timestamp="2026-01-01T00:00:00",
        lifecycle_status="active", access_tags={}, confidence=1.0,
    )
    instance = MemoryInstance(
        "separation", "synthetic", None,
        [{"message_id": "m-mixed", "speaker_id": "operator", "text": mixed_record.content}],
        "What is the current recap?", None, None, None,
    )
    mixed_atoms = adapt_atomic_memories_to_governed_atoms(
        instance=instance, atomic_memories=[mixed_record], information_owner_by_message_id={},
    )
    assert {atom.atom_type for atom in mixed_atoms} == {"event_atom", "policy_atom"}, mixed_atoms
    print("policy_fact_separation_smoke=PASS")


if __name__ == "__main__":
    main()
