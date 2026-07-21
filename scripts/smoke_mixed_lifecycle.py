"""Regression check: deleting one field must not retire an unrelated active field."""

from gov_mem.data.schema import MemoryInstance
from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.memory.amem_memory import AtomicMemory, _materialize_record_atom
from dataclasses import replace

from gov_mem.memory.governed_atom import adapt_atomic_memories_to_governed_atoms


def main() -> None:
    text = "Current callback is channel-A only; the former backup has been deleted."
    instance = MemoryInstance(
        instance_id="mixed-lifecycle", domain="generic", conversation_id=None,
        messages=[{"message_id": "m1", "speaker_id": "owner", "timestamp": "2026-01-01T00:00:00", "text": text}],
        question="What is my current callback?", asking_user_id="owner", choices=None, answer=None,
    )
    memory = AtomicMemory(
        memory_id="mixed", instance_id="mixed-lifecycle", owner_user="owner", memory_type="forgetting",
        content=text, entities=[], slots={"callback": "channel-A", "backup_deletion": "former backup"},
        source_message_ids=["m1"], timestamp="2026-01-01T00:00:00", lifecycle_status="deleted",
        access_tags={"semantic_tags": {
            "discourse_act": "update", "assertion_confidence": 0.9,
            "attributes": {"callback": "channel-A", "backup_deletion": "former backup"},
            "surface_values": {"callback": "channel-A", "backup_deletion": "former backup"},
            "state_delta": {"operation": "set", "changed_fields": {"callback": "channel-A"}},
            "evidence_span": text,
        }}, confidence=1.0,
    )
    atoms = adapt_atomic_memories_to_governed_atoms(
        instance=instance, atomic_memories=[memory], information_owner_by_message_id={"m1": "owner"},
    )
    assert atoms and all(atom.lifecycle == "active" for atom in atoms), atoms
    assert all("backup_deletion" not in atom.slots for atom in atoms), atoms
    graph = GovernedGraphBuilder().build(graph_id="mixed", instance_id=instance.instance_id, atoms=atoms)
    observed = {
        (node.attributes or {}).get("slot_name"): (node.attributes or {}).get("slot_value")
        for node in graph.nodes if node.node_type == "SlotNode"
    }
    assert observed == {"callback": "channel-A"}, observed

    # One source can contain both a current record and deletion of a sibling.
    # Dynamic record decomposition must classify them independently.
    current = _materialize_record_atom(item=memory, semantic_tags={
        "discourse_act": "update", "assertion_confidence": 0.9,
        "event_identity": {"entity_key": "current_callback", "entity_surface_span": "Current callback"},
        "attributes": {"callback": "channel-A"}, "surface_values": {"callback": "channel-A"},
        "state_delta": {"operation": "set", "changed_fields": {"callback": "channel-A"}},
        "evidence_span": "Current callback is channel-A only",
    })
    removed = _materialize_record_atom(item=memory, semantic_tags={
        "discourse_act": "update", "assertion_confidence": 0.9,
        "event_identity": {"entity_key": "former_backup", "entity_surface_span": "former backup"},
        "attributes": {"backup": "former backup"}, "surface_values": {"backup": "former backup"},
        "state_delta": {"operation": "remove", "changed_fields": {"backup_removed": "former backup"}},
        "evidence_span": "the former backup has been deleted",
    })
    record_atoms = adapt_atomic_memories_to_governed_atoms(
        instance=instance, atomic_memories=[current, removed], information_owner_by_message_id={"m1": "owner"},
    )
    assert {(atom.lifecycle, atom.atom_type) for atom in record_atoms} == {
        ("active", "fact_atom"), ("deleted", "deletion_atom"),
    }, record_atoms

    # A policy atom sharing the same source turn must not erase an independent
    # fact record. Only the policy atom's own pseudo-slots stay unobserved.
    policy_atom = replace(
        record_atoms[1], atom_id="m1-policy", atom_type="policy_atom", lifecycle="active",
        slots={"callback": "policy-only"},
    )
    mixed_graph = GovernedGraphBuilder().build(
        graph_id="mixed-policy", instance_id=instance.instance_id,
        atoms=[record_atoms[0], policy_atom],
    )
    mixed_slots = {
        (node.attributes or {}).get("slot_name"): (node.attributes or {}).get("slot_value")
        for node in mixed_graph.nodes if node.node_type == "SlotNode"
    }
    assert mixed_slots == {"callback": "channel-A"}, mixed_slots

    asserted_current = _materialize_record_atom(item=memory, semantic_tags={
        "discourse_act": "assertion", "assertion_confidence": 0.9,
        "event_identity": {"entity_key": "current_callback_label", "entity_surface_span": "Current callback label"},
        "attributes": {"callback_label": "channel-A"}, "surface_values": {"callback_label": "channel-A"},
        "state_delta": {"operation": "supersede", "changed_fields": {"callback_label": "channel-A"}},
        "evidence_span": "Current callback label is channel-A and replaces the deleted earlier label",
    })
    asserted_atoms = adapt_atomic_memories_to_governed_atoms(
        instance=instance, atomic_memories=[asserted_current], information_owner_by_message_id={"m1": "owner"},
    )
    assert {(atom.lifecycle, atom.atom_type) for atom in asserted_atoms} == {("active", "fact_atom")}, asserted_atoms
    print("mixed_lifecycle_smoke=PASS")


if __name__ == "__main__":
    main()
