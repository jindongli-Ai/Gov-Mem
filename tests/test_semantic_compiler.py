from gov_mem.backbones.semantic_compiler import compile_semantics, verify_grounded_atom
from gov_mem.data.schema import RetrievedEvidence


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def chat_json(self, **kwargs):
        return next(self.responses)


def evidence(text="Alice's current office is Room 301."):
    return [RetrievedEvidence(
        memory_id="m1", content=text, score=1.0, retrieval_source="dense",
        reason="test", metadata={"structured_record": {"text": text, "turn_id": "t1"}},
    )]


def test_compiler_grounding_and_coverage():
    llm = FakeLLM([
        {"target_entities": [], "requested_slots": [{"slot_id": "qslot_0", "slot_name": "current office", "required_for_answer": True}]},
        {"atoms": [{"slot_id": "qslot_0", "slot_name": "current office", "value": "Room 301", "source": {"chunk_id": "m1", "span": "current office is Room 301"}}]},
    ])
    result = compile_semantics(
        question="What is Alice's current office?", requester="alice", requester_role="member",
        evidence=evidence(), llm_client=llm, model_name="gpt-4o-mini",
        config={"semantic_compiler": {"enabled": True, "repair": {"enabled": False}}},
    )
    assert result.coverage_after_repair["coverage_ratio"] == 1.0
    assert len(result.final_grounded_atoms) == 1


def test_unrelated_source_is_rejected():
    atom = __import__("gov_mem.backbones.semantic_compiler", fromlist=["GroundedMemoryAtom"]).GroundedMemoryAtom(
        atom_id="a", target_entity=None, slot_id="qslot_0", slot_name="office",
        source={"chunk_id": "future", "span": "Room 301"},
    )
    check = verify_grounded_atom(atom, evidence(), {"qslot_0"})
    assert not check.accepted_for_symbolic_reasoning
    assert "source_not_in_stage1_evidence" in check.reasons


def test_mismatched_source_turn_is_rejected():
    atom = __import__("gov_mem.backbones.semantic_compiler", fromlist=["GroundedMemoryAtom"]).GroundedMemoryAtom(
        atom_id="a", target_entity=None, slot_id="qslot_0", slot_name="office",
        source={"chunk_id": "m1", "turn_id": "future", "span": "Room 301"},
    )
    check = verify_grounded_atom(atom, evidence(), {"qslot_0"})
    assert not check.accepted_for_symbolic_reasoning
    assert "turn_id_mismatch" in check.reasons


def test_malformed_source_is_safely_dropped():
    llm = FakeLLM([
        {"requested_slots": [{"slot_id": "qslot_0", "slot_name": "office"}]},
        {"atoms": [{"slot_id": "qslot_0", "slot_name": "office", "source": "m1"}]},
    ])
    result = compile_semantics(
        question="What is the office?", requester="alice", requester_role=None,
        evidence=evidence(), llm_client=llm, model_name="gpt-4o-mini",
        config={"semantic_compiler": {"enabled": True, "repair": {"enabled": False}}},
    )
    assert result.final_grounded_atoms == []


def test_contract_slot_name_recovers_a_structurally_complete_atom():
    llm = FakeLLM([
        {"requested_slots": [{"slot_id": "qslot_0", "slot_name": "current office"}]},
        {"atoms": [{
            "slot_id": "qslot_0", "value": "Room 301",
            "source": {"chunk_id": "m1", "span": "current office is Room 301"},
        }]},
    ])
    result = compile_semantics(
        question="What is Alice's current office?", requester="alice", requester_role=None,
        evidence=evidence(), llm_client=llm, model_name="gpt-4o-mini",
        config={"semantic_compiler": {"enabled": True, "repair": {"enabled": False}}},
    )
    assert result.final_grounded_atoms[0]["slot_name"] == "current office"


def test_flattened_provenance_still_requires_closed_evidence_and_span():
    llm = FakeLLM([
        {"requested_slots": [{"slot_id": "qslot_0", "slot_name": "current office"}]},
        {"atoms": [{
            "slot_id": "qslot_0", "value": "Room 301", "chunk_id": "m1",
            "span": "current office is Room 301",
        }]},
    ])
    result = compile_semantics(
        question="What is Alice's current office?", requester="alice", requester_role=None,
        evidence=evidence(), llm_client=llm, model_name="gpt-4o-mini",
        config={"semantic_compiler": {"enabled": True, "repair": {"enabled": False}}},
    )
    assert len(result.final_grounded_atoms) == 1


def test_atom_value_must_be_grounded_in_its_cited_span():
    atom = __import__(
        "gov_mem.backbones.semantic_compiler", fromlist=["GroundedMemoryAtom"]
    ).GroundedMemoryAtom(
        atom_id="a",
        target_entity=None,
        slot_id="qslot_0",
        slot_name="current office",
        value="Cinder Annex Room 301",
        source={
            "chunk_id": "m1",
            "turn_id": "t1",
            "span": "Alice's current office is Room 301.",
        },
    )

    check = verify_grounded_atom(atom, evidence(), {"qslot_0"})

    assert not check.accepted_for_symbolic_reasoning
    assert not check.value_valid
    assert "value_not_present_in_source_span" in check.reasons


def test_missing_slot_triggers_one_repair_round():
    llm = FakeLLM([
        {"requested_slots": [
            {"slot_id": "qslot_0", "slot_name": "office", "required_for_answer": True},
            {"slot_id": "qslot_1", "slot_name": "status", "required_for_answer": True},
        ]},
        {"atoms": [{"slot_id": "qslot_0", "slot_name": "office", "value": "Room 301", "source": {"chunk_id": "m1", "span": "office is Room 301"}}]},
        {"atoms": [{"slot_id": "qslot_1", "slot_name": "status", "value": "open", "source": {"chunk_id": "m1", "span": "status is open"}}]},
    ])
    result = compile_semantics(
        question="What are the office and status?", requester="alice", requester_role="member",
        evidence=evidence("Alice's office is Room 301 and status is open."), llm_client=llm,
        model_name="gpt-4o-mini", config={"semantic_compiler": {"enabled": True, "repair": {"enabled": True, "max_rounds": 1}}},
    )
    assert result.repair_triggered
    assert result.coverage_before_repair["coverage_ratio"] == 0.5
    assert result.coverage_after_repair["coverage_ratio"] == 1.0


def test_current_slot_repair_is_triggered_by_state_resolvability():
    llm = FakeLLM([
        {"requested_slots": [{
            "slot_id": "qslot_0",
            "slot_name": "current office",
            "temporal_requirement": "current",
            "required_for_answer": True,
        }]},
        {"atoms": [{
            "slot_id": "qslot_0",
            "value": "Room 301",
            "temporal": {"state": "historical"},
            "source": {
                "chunk_id": "m1", "turn_id": "t1",
                "span": "Alice's current office is Room 301.",
            },
        }]},
        {"atoms": [{
            "slot_id": "qslot_0",
            "value": "Room 301",
            "temporal": {"state": "current"},
            "source": {
                "chunk_id": "m1", "turn_id": "t1",
                "span": "Alice's current office is Room 301.",
            },
        }]},
    ])

    result = compile_semantics(
        question="What is Alice's current office?",
        requester="alice",
        requester_role="member",
        evidence=evidence(),
        llm_client=llm,
        model_name="gpt-5.4-mini",
        config={"semantic_compiler": {
            "enabled": True, "repair": {"enabled": True, "max_rounds": 1},
        }},
    )

    assert result.coverage_before_repair["coverage_ratio"] == 1.0
    assert result.resolvability_before_repair["resolvability_ratio"] == 0.0
    assert result.repair_triggered
    assert result.resolvability_after_repair["resolvability_ratio"] == 1.0


def test_malformed_nested_contract_fields_degrade_without_crashing():
    llm = FakeLLM([
        {
            "target_entities": "Alice",
            "requested_slots": [{
                "slot_id": "qslot_0",
                "slot_name": "current office",
            }],
            "requester_relations": {"relation": "self"},
            "temporal_intent": "current",
            "answer_shape": ["single_field"],
        },
        {"atoms": []},
    ])

    result = compile_semantics(
        question="What is Alice's current office?",
        requester="alice",
        requester_role="member",
        evidence=evidence(),
        llm_client=llm,
        model_name="gpt-5.4-mini",
        config={"semantic_compiler": {
            "enabled": True, "repair": {"enabled": False},
        }},
    )

    assert result.query_contract.target_entities == []
    assert result.query_contract.requester_relations == []
    assert result.query_contract.temporal_intent == {
        "mode": "unknown", "reference": None,
    }
    assert result.query_contract.answer_shape == {}


def test_malformed_nested_atom_fields_degrade_without_crashing():
    llm = FakeLLM([
        {"requested_slots": [{
            "slot_id": "qslot_0",
            "slot_name": "current office",
        }]},
        {"atoms": [{
            "slot_id": "qslot_0",
            "value": "Room 301",
            "temporal": "current",
            "authorization_semantics": ["allow"],
            "lifecycle_semantics": "assert",
            "sensitivity_semantics": None,
            "source": {
                "chunk_id": "m1",
                "turn_id": "t1",
                "span": "Alice's current office is Room 301.",
            },
        }]},
    ])

    result = compile_semantics(
        question="What is Alice's current office?",
        requester="alice",
        requester_role="member",
        evidence=evidence(),
        llm_client=llm,
        model_name="gpt-5.4-mini",
        config={"semantic_compiler": {
            "enabled": True, "repair": {"enabled": False},
        }},
    )

    atom = result.final_grounded_atoms[0]
    assert atom["temporal"] == {}
    assert atom["authorization_semantics"] == {}
    assert atom["lifecycle_semantics"] == {}
    assert atom["sensitivity_semantics"] == {}


def test_non_list_atoms_field_degrades_to_no_candidates():
    llm = FakeLLM([
        {"requested_slots": [{
            "slot_id": "qslot_0",
            "slot_name": "current office",
        }]},
        {"atoms": {"slot_id": "qslot_0"}},
    ])

    result = compile_semantics(
        question="What is Alice's current office?",
        requester="alice",
        requester_role="member",
        evidence=evidence(),
        llm_client=llm,
        model_name="gpt-5.4-mini",
        config={"semantic_compiler": {
            "enabled": True, "repair": {"enabled": False},
        }},
    )

    assert result.initial_atoms == []
    assert result.final_grounded_atoms == []
