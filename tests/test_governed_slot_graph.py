import json

from gov_mem.backbones.v4_support import RAGChunk
from gov_mem.memory.governed_slot_graph import (
    MemoryGovernedSlotGraph,
    build_memory_governed_slot_graph,
)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat_json(self, *, model, system_prompt, user_prompt):
        self.calls.append((model, system_prompt, user_prompt))
        return self.payload

    def embed_texts(self, *, model, texts):
        # A stable toy embedding is sufficient to exercise the independent
        # graph index without making a provider call.
        del model
        return [[float(len(text)), float(sum(ch.isdigit() for ch in text))] for text in texts]


def _chunks():
    return [
        RAGChunk(
            chunk_id="chunk_1",
            instance_id="episode",
            text="[owner:alice] Alice's current meeting room is Room 301.",
            source_message_ids=["m1"],
            speaker_ids=["alice"],
            timestamp_range=(None, None),
            metadata={"structured_record": {"turn_index": 1, "speaker": {"principal_id": "alice"}}},
        ),
    ]


def test_graph_is_source_grounded_and_permission_edges_are_directed():
    llm = FakeLLM({
        "records": [{
            "source_chunk_id": "chunk_1",
            "source_message_id": "m1",
            "source_span": "Alice's current meeting room is Room 301",
            "target_entity": "Alice",
            "slots": [{
                "slot_name": "current meeting room",
                "value": "Room 301",
                "value_span": "Room 301",
            }],
            "policies": [{
                "effect": "allow",
                "subject": "Alice",
                "resource": "current meeting room",
                "span": "Alice's current meeting room is Room 301",
            }],
        }],
    })
    result = build_memory_governed_slot_graph(
        chunks=_chunks(), llm_client=llm, model_name="test-model",
        config={"embedding": {"model": "test-embedding"}},
    )
    graph = result.graph
    assert result.llm_calls == 1
    assert len(graph.slots) == 1
    assert len(graph.policies) == 1
    assert any(edge["edge_type"] == "allow" for edge in graph.edges)
    allow_edge = next(edge for edge in graph.edges if edge["edge_type"] == "allow")
    assert allow_edge["source_id"].startswith("policy::")
    assert allow_edge["target_id"].startswith("slot::")
    assert graph.audit["evidence_boundary"] == "observable_prefix"


def test_ungrounded_graph_records_are_rejected_without_lexical_fallback():
    llm = FakeLLM({
        "records": [{
            "source_chunk_id": "chunk_1",
            "source_span": "not in the source",
            "slots": [{"slot_name": "room", "value": "Room 999", "value_span": "Room 999"}],
        }],
    })
    result = build_memory_governed_slot_graph(
        chunks=_chunks(), llm_client=llm, model_name="test-model", config={},
    )
    assert result.graph.slots == []
    assert result.graph.policies == []


def test_graph_llm_failure_degrades_to_empty_optional_channel():
    class Broken(FakeLLM):
        def chat_json(self, **kwargs):
            raise RuntimeError("provider unavailable")

    result = build_memory_governed_slot_graph(
        chunks=_chunks(), llm_client=Broken({}), model_name="test-model", config={},
    )
    assert result.parse_failure is True
    assert result.graph.nodes == []
    assert result.graph.audit["fallback"] == "empty_graph"


def test_entity_relation_list_preserves_permission_changes_in_order():
    chunks = [
        *_chunks(),
        RAGChunk(
            chunk_id="chunk_2",
            instance_id="episode",
            text="Alice's access to the current meeting room is revoked.",
            source_message_ids=["m2"],
            speaker_ids=["alice"],
            timestamp_range=(None, None),
            metadata={"structured_record": {"turn_index": 2}},
        ),
    ]
    llm = FakeLLM({
        "records": [
            {
                "source_chunk_id": "chunk_1",
                "source_message_id": "m1",
                "source_span": "Alice's current meeting room is Room 301",
                "target_entity": "Alice",
                "slots": [{
                    "slot_name": "current meeting room",
                    "value": "Room 301",
                    "value_span": "Room 301",
                }],
                "policies": [{
                    "effect": "allow",
                    "subject": "Alice",
                    "resource": "current meeting room",
                    "span": "Alice's current meeting room is Room 301",
                }],
            },
            {
                "source_chunk_id": "chunk_2",
                "source_message_id": "m2",
                "source_span": "Alice's access to the current meeting room is revoked",
                "target_entity": "Alice",
                "slots": [],
                "policies": [{
                    "effect": "revoke",
                    "subject": "Alice",
                    "resource": "current meeting room",
                    "span": "Alice's access to the current meeting room is revoked",
                }],
            },
        ],
    })
    result = build_memory_governed_slot_graph(
        chunks=chunks, llm_client=llm, model_name="test-model", config={},
    )
    graph = result.graph
    history = graph.permission_history("entity::alice", resource="current meeting room")
    assert [row["effect"] for row in history] == ["allow", "revoke"]
    assert graph.current_permission("entity::alice", resource="current meeting room") == "deny"
    assert len(graph.to_dict()["entity_lists"]["entity::alice"]) >= 2
    retrieved_lists = graph.retrieve_entity_lists(
        question="Alice meeting room access",
        embedding_client=llm,
        embedding_model="test-embedding",
        top_k=20,
    )
    assert any(item["entity_id"] == "entity::alice" for item in retrieved_lists)


def test_entity_relation_list_can_append_only_new_history_chunks():
    first = _chunks()[0]
    second = RAGChunk(
        chunk_id="chunk_2",
        instance_id="episode",
        text="Alice's access to the current meeting room is revoked.",
        source_message_ids=["m2"],
        speaker_ids=["alice"],
        timestamp_range=(None, None),
        metadata={"structured_record": {"turn_index": 2}},
    )

    class IncrementalLLM(FakeLLM):
        def chat_json(self, *, model, system_prompt, user_prompt):
            self.calls.append((model, system_prompt, user_prompt))
            payload = json.loads(user_prompt)
            message_id = payload["messages"][0]["source_message_id"]
            if message_id == "m1":
                return {"records": [{
                    "source_chunk_id": "chunk_1",
                    "source_message_id": "m1",
                    "source_span": "Alice's current meeting room is Room 301",
                    "target_entities": ["Alice"],
                    "slots": [],
                    "policies": [{
                        "effect": "allow", "subject": "Alice",
                        "resource": "current meeting room",
                        "span": "Alice's current meeting room is Room 301",
                    }],
                }]}
            return {"records": [{
                "source_chunk_id": "chunk_2",
                "source_message_id": "m2",
                "source_span": "Alice's access to the current meeting room is revoked",
                "target_entities": ["Alice"],
                "slots": [],
                "policies": [{
                    "effect": "revoke", "subject": "Alice",
                    "resource": "current meeting room",
                    "span": "Alice's access to the current meeting room is revoked",
                }],
            }]}

    client = IncrementalLLM({})
    first_build = build_memory_governed_slot_graph(
        chunks=[first], llm_client=client, model_name="test-model", config={},
    )
    processed = {first.chunk_id}
    second_build = build_memory_governed_slot_graph(
        chunks=[first, second], llm_client=client, model_name="test-model", config={},
        existing_graph=first_build.graph, processed_chunk_ids=processed,
    )
    assert len(client.calls) == 2
    assert second_build.graph.audit["incremental"] is True
    assert second_build.graph.audit["new_chunk_count"] == 1
    assert [row["effect"] for row in second_build.graph.permission_history(
        "entity::alice", resource="current meeting room",
    )] == ["allow", "revoke"]


def test_future_conditional_revocation_is_not_current_permission_change():
    chunks = [
        RAGChunk(
            chunk_id="chunk_1",
            instance_id="episode",
            text=(
                "Approved for Leo: logs-only incident access through Friday. "
                "Revocation expected after the repro window."
            ),
            source_message_ids=["t1"],
            speaker_ids=[],
            timestamp_range=(None, None),
            metadata={"structured_record": {"turn_index": 1}},
        )
    ]
    client = FakeLLM(
        {
            "records": [
                {
                    "source_chunk_id": "chunk_1",
                    "source_message_id": "t1",
                    "source_span": chunks[0].text,
                    "target_entities": ["logs-only incident access"],
                    "slots": [],
                    "policies": [
                        {
                            "effect": "allow",
                            "subject": "Leo",
                            "resource": "logs-only incident access",
                            "span": "Approved for Leo: logs-only incident access through Friday",
                        },
                        {
                            "effect": "revoke",
                            "subject": "Leo",
                            "resource": "logs-only incident access",
                            "span": "Revocation expected after the repro window",
                        },
                    ],
                    "lifecycle": [],
                }
            ]
        }
    )
    result = build_memory_governed_slot_graph(
        chunks=chunks,
        llm_client=client,
        model_name="test",
        config={"memory_governed_slot_graph": {"batch_size": 8}, "embedding": {}},
    )
    effects = [
        row["effect"]
        for values in result.graph.to_dict()["entity_lists"].values()
        for row in values
        if row["relation_type"] == "permission"
    ]
    assert effects == ["allow"]


def test_abstract_entity_list_owns_permission_history_and_keeps_subject():
    chunks = [RAGChunk(
        chunk_id="chunk_1",
        instance_id="episode",
        text="Alice may access the work system.",
        source_message_ids=["m1"],
        speaker_ids=["alice"],
        timestamp_range=(None, None),
        metadata={"structured_record": {"turn_index": 1}},
    )]
    llm = FakeLLM({
        "records": [{
            "source_chunk_id": "chunk_1",
            "source_message_id": "m1",
            "source_span": "Alice may access the work system.",
            "target_entities": ["work system"],
            "slots": [],
            "policies": [{
                "effect": "allow",
                "subject": "Alice",
                "resource": "work system",
                "span": "Alice may access the work system",
            }],
            "lifecycle": [],
        }],
    })
    graph = build_memory_governed_slot_graph(
        chunks=chunks, llm_client=llm, model_name="test-model", config={},
    ).graph
    rows = graph.permission_history("entity::work system")
    assert len(rows) == 1
    assert rows[0]["effect"] == "allow"
    assert rows[0]["subject"] == "Alice"


def test_permission_subject_can_be_grounded_in_source_chunk_outside_short_span():
    chunks = [RAGChunk(
        chunk_id="chunk_1",
        instance_id="episode",
        text="Approved for Leo: logs-only work system access through Friday.",
        source_message_ids=["m1"],
        speaker_ids=["security"],
        timestamp_range=(None, None),
        metadata={"structured_record": {"turn_index": 1}},
    )]
    llm = FakeLLM({"records": [{
        "source_chunk_id": "chunk_1",
        "source_message_id": "m1",
        "source_span": "logs-only work system access",
        "target_entities": ["work system access"],
        "slots": [],
        "policies": [{
            "effect": "allow",
            "subject": "Leo",
            "resource": "work system access",
            "span": "logs-only work system access",
        }],
        "lifecycle": [],
    }]})
    graph = build_memory_governed_slot_graph(
        chunks=chunks, llm_client=llm, model_name="test-model", config={},
    ).graph
    assert graph.permission_history("entity::work system access")[0]["subject"] == "Leo"


def test_structured_caller_can_append_without_replacing_history():
    graph = MemoryGovernedSlotGraph()
    graph.append_entity_change(
        entity_id="entity::appointment",
        relation_type="state",
        target="Tuesday 10:00",
        source_message_id="m1",
    )
    graph.append_entity_change(
        entity_id="entity::appointment",
        relation_type="state",
        target="Wednesday 11:00",
        source_message_id="m2",
    )
    assert [row["target"] for row in graph.relation_list("entity::appointment")] == [
        "Tuesday 10:00", "Wednesday 11:00",
    ]


def test_entity_list_retrieval_keeps_governance_history_but_filters_fact_rows():
    graph = MemoryGovernedSlotGraph()
    for index in range(20):
        graph.append_entity_change(
            entity_id="entity::maple",
            relation_type="has_slot",
            target=f"unrelated field {index}=value {index}",
            source_chunk_id=f"chunk_{index}",
        )
    graph.append_entity_change(
        entity_id="entity::maple",
        relation_type="has_slot",
        target="current access scope=scheduling only",
        source_chunk_id="chunk_scope",
    )
    graph.append_entity_change(
        entity_id="entity::maple",
        relation_type="permission",
        target="work system",
        effect="allow",
        subject="Alice",
        source_chunk_id="chunk_policy_1",
        source_span="Alice may access the work system",
    )
    graph.append_entity_change(
        entity_id="entity::maple",
        relation_type="permission",
        target="work system",
        effect="revoke",
        subject="Alice",
        source_chunk_id="chunk_policy_2",
        source_span="Alice's access is revoked",
    )

    rows = graph.retrieve_entity_lists(
        question="Maple work system access scope",
        embedding_client=FakeLLM({}),
        embedding_model="test-embedding",
        top_k=20,
        max_relations_per_entity=3,
    )
    assert len(rows) == 1
    relations = rows[0]["relations"]
    assert [item["effect"] for item in relations if item["relation_type"] == "permission"] == [
        "allow", "revoke",
    ]
    assert any("access scope" in str(item["target"]) for item in relations)
    assert len(relations) == 3
