"""Regression check: final alignment cannot anchor a non-certifiable source."""

from gov_mem.graph.graph_builder import GovernedGraphBuilder
from gov_mem.governance_runtime.semantic_alignment import _is_record_collection_attribute, align_requested_attributes
from gov_mem.memory.governed_atom import GovernedMemoryAtom


def main() -> None:
    mixed_list_spec = {
        "request_shape": "list",
        "attribute_bindings": [
            {"attribute": "open_items", "need_kind": "record_collection"},
            {"attribute": "current_date", "need_kind": "scalar"},
        ],
    }
    assert _is_record_collection_attribute("open_items", mixed_list_spec)
    assert not _is_record_collection_attribute("current_date", mixed_list_spec)
    def fact(atom_id: str, value: str, evidence_span: str | None) -> GovernedMemoryAtom:
        text = f"Observed record value is {value}."
        return GovernedMemoryAtom(
            atom_id=atom_id, atom_type="fact_atom", text=text, slots={"record_value": value},
            owner_id="owner", subject_id=atom_id, speaker_id="owner", source_turn=1,
            timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private",
            access_scope=["self"], related_entities=[], confidence=1.0,
            provenance={"source_message_ids": [atom_id], "evidence_span": evidence_span},
        )

    ungrounded = fact("ungrounded", "old", None)
    grounded = fact("grounded", "current", "Observed record value is current.")
    graph = GovernedGraphBuilder().build(graph_id="attestation", instance_id="smoke", atoms=[ungrounded, grounded])

    class AlignmentLLM:
        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            candidates = str(kwargs["user_prompt"]).split("Candidates:", 1)[-1]
            assert "old" not in candidates, candidates
            slot_id = next(node.node_id for node in graph.nodes if node.node_type == "SlotNode" and "grounded" in node.node_id)
            return {"audit": [
                {"source_atom_id": "record_a", "classification": "answer_member"},
                {"source_atom_id": "record_b", "classification": "answer_member"},
                {"source_atom_id": "unrelated", "classification": "unrelated"},
            ], "bindings": [{
                "attribute": "record_summary", "slot_node_ids": [slot_id],
                "query_support_span": "record summary",
                "fact_support_spans": [{"slot_node_id": slot_id, "fact_support_span": "Observed record value is current."}],
            }]}

    result = align_requested_attributes(
        question="What is the record summary?",
        semantic_spec={"requested_attributes": ["record_summary"], "attribute_bindings": [{
            "attribute": "record_summary", "support_span": "record summary", "semantic_role": "requested_property",
        }]},
        graph=graph, owner_id="owner", utility_atom_ids=None, llm_client=AlignmentLLM(), model_name="stub",
        semantic_contract_certifiable=True, require_attested_evidence_span=True,
    )
    assert result["available"], result
    assert result["bindings"]["record_summary"]["slot_name"] == "record_value", result

    # A closed Stage-2 authorization may retain a value that is verbatim in
    # its source record when a heuristic extractor omitted redundant span
    # metadata. Without the Stage-2 atom ID, the preceding assertion proves
    # that this same source remains unavailable under attestation mode.
    class Stage2AuthorizedLLM:
        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            slot_id = next(node.node_id for node in graph.nodes if node.node_type == "SlotNode" and "ungrounded" in node.node_id)
            return {"bindings": [{
                "attribute": "current_record", "slot_node_ids": [slot_id],
                "query_support_span": "current record",
                "fact_support_spans": [{"slot_node_id": slot_id, "fact_support_span": "Observed record value is old."}],
            }]}

    stage2_authorized = align_requested_attributes(
        question="What is the current record?",
        semantic_spec={"requested_attributes": ["current_record"], "attribute_bindings": [{
            "attribute": "current_record", "support_span": "current record", "semantic_role": "requested_property",
        }]},
        graph=graph, owner_id="owner", utility_atom_ids={"ungrounded"},
        stage2_authorized_atom_ids={"ungrounded"}, llm_client=Stage2AuthorizedLLM(), model_name="stub",
        semantic_contract_certifiable=True, require_attested_evidence_span=True,
    )
    assert stage2_authorized["available"], stage2_authorized
    assert stage2_authorized["bindings"]["current_record"]["anchor_slot_node_id"].startswith("slot::ungrounded::"), stage2_authorized

    class ExpandedQuestionSpanLLM:
        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            slot_id = next(node.node_id for node in graph.nodes if node.node_type == "SlotNode" and "grounded" in node.node_id)
            return {"bindings": [{
                "attribute": "current_record", "slot_node_ids": [slot_id],
                "query_support_span": "current record after the prior value was removed",
                "fact_support_spans": [{"slot_node_id": slot_id, "fact_support_span": "Observed record value is current."}],
            }]}

    expanded_span = align_requested_attributes(
        question="What is the current record after the prior value was removed?",
        semantic_spec={"requested_attributes": ["current_record"], "attribute_bindings": [{
            "attribute": "current_record", "support_span": "current record", "semantic_role": "requested_property",
        }]},
        graph=graph, owner_id="owner", utility_atom_ids=None, llm_client=ExpandedQuestionSpanLLM(), model_name="stub",
        semantic_contract_certifiable=True, require_attested_evidence_span=True,
    )
    assert expanded_span["available"], expanded_span

    class CollectionAlignmentLLM:
        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            slot_ids = [node.node_id for node in graph.nodes if node.node_type == "SlotNode"]
            spans = {
                slot_id: "Observed record value is current." if "slot::grounded::" in slot_id else "Observed record value is old."
                for slot_id in slot_ids
            }
            return {"bindings": [{
                "attribute": "two_records", "binding_kind": "collection", "slot_node_ids": slot_ids,
                "query_support_span": "two records",
                "fact_support_spans": [
                    {"slot_node_id": slot_id, "fact_support_span": spans[slot_id]} for slot_id in slot_ids
                ],
            }]}

    collection = align_requested_attributes(
        question="Which are the two records?",
        semantic_spec={"requested_attributes": ["two_records"], "attribute_bindings": [{
            "attribute": "two_records", "support_span": "two records", "semantic_role": "requested_property",
        }], "request_shape": "unspecified"},
        graph=graph, owner_id="owner", utility_atom_ids=None, llm_client=CollectionAlignmentLLM(), model_name="stub",
        semantic_contract_certifiable=True, require_attested_evidence_span=False,
    )
    assert collection["available"], collection
    assert len(collection["bindings"]["two_records"]["anchor_slot_node_ids"]) == 2, collection

    # A full-set audit replaces a partial collection proposal with every LLM-
    # selected record whose closed ID and complete source span validate.
    records = [
        fact("record_a", "alpha", "Observed record value is alpha."),
        fact("record_b", "bravo", "Observed record value is bravo."),
        fact("unrelated", "charlie", "Observed record value is charlie."),
    ]
    collection_graph = GovernedGraphBuilder().build(graph_id="collection", instance_id="smoke", atoms=records)
    slot_ids_by_atom = {}
    for node in collection_graph.nodes:
        if node.node_type != "SlotNode":
            continue
        atom_id = str((node.provenance or {}).get("source_atom_id") or "")
        slot_ids_by_atom[atom_id] = node.node_id

    class CompletionLLM:
        def __init__(self, include_unrelated=False):
            self.calls = 0
            self.include_unrelated = include_unrelated

        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                slot_id = slot_ids_by_atom["record_a"]
                return {"bindings": [{
                    "attribute": "records", "binding_kind": "collection", "slot_node_ids": [slot_id],
                    "query_support_span": "records", "fact_support_spans": [{
                        "slot_node_id": slot_id, "fact_support_span": "Observed record value is alpha.",
                    }],
                }]}
            if self.calls == 2:
                return {"audit": [
                    {"source_atom_id": "record_a", "classification": "answer_member"},
                    {"source_atom_id": "record_b", "classification": "unrelated" if self.include_unrelated else "answer_member"},
                    {"source_atom_id": "unrelated", "classification": "unrelated"},
                ], "bindings": [{
                "attribute": "records", "binding_kind": "collection", "source_atom_ids": (["record_a"] if self.include_unrelated else ["record_a", "record_b"]),
                    "query_support_span": "records", "fact_support_spans": [{
                        "source_atom_id": "record_a", "fact_support_span": "Observed record value is alpha.",
                    }, *([] if self.include_unrelated else [{
                        "source_atom_id": "record_b", "fact_support_span": "Observed record value is bravo.",
                    }])],
                }]}
            record_ids = ["record_b"] + (["unrelated"] if self.include_unrelated else [])
            return {"bindings": [{
                "attribute": "records", "binding_kind": "collection", "source_atom_ids": record_ids,
                "query_support_span": "records", "fact_support_spans": [
                    {"source_atom_id": "record_b", "fact_support_span": "Observed record value is bravo."},
                    *([{"source_atom_id": "unrelated", "fact_support_span": "not a source span"}] if self.include_unrelated else []),
                ],
            }]}

    collection_spec = {"requested_attributes": ["records"], "attribute_bindings": [{
        "attribute": "records", "support_span": "records", "need_kind": "record_collection",
    }], "request_shape": "list"}
    completed = align_requested_attributes(
        question="Which records?", semantic_spec=collection_spec, graph=collection_graph, owner_id="owner",
        utility_atom_ids=None, llm_client=CompletionLLM(), model_name="stub", semantic_contract_certifiable=True,
    )
    assert set(completed["bindings"]["records"]["anchor_slot_node_ids"]) == {
        slot_ids_by_atom["record_a"], slot_ids_by_atom["record_b"],
    }, completed
    rejected_completion = align_requested_attributes(
        question="Which records?", semantic_spec=collection_spec, graph=collection_graph, owner_id="owner",
        utility_atom_ids=None, llm_client=CompletionLLM(include_unrelated=True), model_name="stub",
        semantic_contract_certifiable=True,
    )
    assert set(rejected_completion["bindings"]["records"]["anchor_slot_node_ids"]) == {slot_ids_by_atom["record_a"]}, rejected_completion

    # The auditor, rather than a slot-name or date rule, excludes a record
    # that merely mentions one collection member.
    schedule_records = [
        GovernedMemoryAtom(
            atom_id="appointment_a", atom_type="event_atom", text="Tue Mar 17 1:00 PM follow-up.",
            slots={"event_date": "Tue Mar 17", "event_time": "1:00 PM", "event_type": "follow-up"},
            owner_id="owner", subject_id="appointment_a", speaker_id="staff", source_turn=1,
            timestamp="2026-01-01T00:00:00", lifecycle="active", sensitivity="private", access_scope=["self"],
            related_entities=[], confidence=1.0, provenance={"source_message_ids": ["appointment_a"], "evidence_span": "Tue Mar 17 1:00 PM follow-up."},
        ),
        GovernedMemoryAtom(
            atom_id="appointment_b", atom_type="event_atom", text="Mon Jun 8 8:30 AM repeat RPR.",
            slots={"event_date": "Mon Jun 8", "event_time": "8:30 AM", "event_type": "repeat RPR"},
            owner_id="owner", subject_id="appointment_b", speaker_id="staff", source_turn=2,
            timestamp="2026-01-01T00:01:00", lifecycle="active", sensitivity="private", access_scope=["self"],
            related_entities=[], confidence=1.0, provenance={"source_message_ids": ["appointment_b"], "evidence_span": "Mon Jun 8 8:30 AM repeat RPR."},
        ),
        GovernedMemoryAtom(
            atom_id="treatment_plan", atom_type="event_atom", text="No sex for 7 days; keep Tuesday March 17 follow-up.",
            slots={"restriction_days": "7 days", "follow_up_date": "Tuesday March 17"},
            owner_id="owner", subject_id="treatment_plan", speaker_id="staff", source_turn=3,
            timestamp="2026-01-01T00:02:00", lifecycle="active", sensitivity="private", access_scope=["self"],
            related_entities=[], confidence=1.0, provenance={"source_message_ids": ["treatment_plan"], "evidence_span": "No sex for 7 days; keep Tuesday March 17 follow-up."},
        ),
    ]
    schedule_graph = GovernedGraphBuilder().build(graph_id="schedule", instance_id="smoke", atoms=schedule_records)
    schedule_slots = {
        str((node.provenance or {}).get("source_atom_id")): node.node_id
        for node in schedule_graph.nodes
        if node.node_type == "SlotNode" and str((node.attributes or {}).get("slot_name")) == "event_date"
    }

    class ScheduleCompletionLLM:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                slot_id = schedule_slots["appointment_a"]
                return {"bindings": [{
                    "attribute": "schedule", "binding_kind": "collection", "slot_node_ids": [slot_id],
                    "query_support_span": "future schedule", "fact_support_spans": [{
                        "slot_node_id": slot_id, "fact_support_span": "Tue Mar 17 1:00 PM follow-up.",
                    }],
                }]}
            if self.calls == 2:
                return {"audit": [
                    {"source_atom_id": "appointment_a", "classification": "answer_member"},
                    {"source_atom_id": "appointment_b", "classification": "answer_member"},
                    {"source_atom_id": "treatment_plan", "classification": "mere_mention"},
                ], "bindings": [{
                "attribute": "schedule", "binding_kind": "collection", "source_atom_ids": ["appointment_a", "appointment_b"],
                    "query_support_span": "future schedule", "fact_support_spans": [{
                        "source_atom_id": "appointment_a", "fact_support_span": "Tue Mar 17 1:00 PM follow-up.",
                    }, {
                        "source_atom_id": "appointment_b", "fact_support_span": "Mon Jun 8 8:30 AM repeat RPR.",
                    }],
                }]}
            candidate_text = str(_kwargs["user_prompt"])
            assert "No sex for 7 days" in candidate_text, candidate_text
            return {"audit": [
                {"source_atom_id": "appointment_b", "classification": "answer_member"},
                {"source_atom_id": "treatment_plan", "classification": "mere_mention"},
            ], "bindings": [{
                "attribute": "schedule", "binding_kind": "collection", "source_atom_ids": ["appointment_b"],
                "query_support_span": "future schedule", "fact_support_spans": [{
                    "source_atom_id": "appointment_b", "fact_support_span": "Mon Jun 8 8:30 AM repeat RPR.",
                }],
            }]}

    schedule = align_requested_attributes(
        question="What is my future schedule?", semantic_spec={"requested_attributes": ["schedule"], "attribute_bindings": [{
            "attribute": "schedule", "support_span": "future schedule", "need_kind": "record_collection",
        }], "request_shape": "list"}, graph=schedule_graph, owner_id="owner", utility_atom_ids=None,
        llm_client=ScheduleCompletionLLM(), model_name="stub", semantic_contract_certifiable=True,
    )
    assert {str(node_id).split("::")[1] for node_id in schedule["bindings"]["schedule"]["anchor_slot_node_ids"]} == {
        "appointment_a", "appointment_b",
    }, schedule

    class BoundaryScheduleLLM:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                slot_id = schedule_slots["appointment_a"]
                return {"bindings": [{
                    "attribute": "schedule", "binding_kind": "collection", "slot_node_ids": [slot_id],
                    "query_support_span": "schedule after injection visit", "fact_support_spans": [{
                        "slot_node_id": slot_id, "fact_support_span": "Tue Mar 17 1:00 PM follow-up.",
                    }],
                }]}
            assert "completeness audit over the entire closed candidate set" in str(kwargs["user_prompt"]), kwargs["user_prompt"]
            return {"audit": [
                {"source_atom_id": "appointment_a", "classification": "reference_boundary"},
                {"source_atom_id": "appointment_b", "classification": "answer_member"},
                {"source_atom_id": "treatment_plan", "classification": "mere_mention"},
            ], "bindings": [{
                "attribute": "schedule", "binding_kind": "collection", "source_atom_ids": ["appointment_b"],
                "query_support_span": "schedule after injection visit", "fact_support_spans": [{
                    "source_atom_id": "appointment_b", "fact_support_span": "Mon Jun 8 8:30 AM repeat RPR.",
                }],
            }]}

    after_schedule = align_requested_attributes(
        question="What is my schedule after injection visit?", semantic_spec={"requested_attributes": ["schedule"], "attribute_bindings": [{
            "attribute": "schedule", "support_span": "schedule after injection visit", "need_kind": "record_collection",
        }], "disclosure_constraints": [{
            "constraint_kind": "temporal_access_boundary", "support_span": "after injection visit",
        }], "request_shape": "list"}, graph=schedule_graph, owner_id="owner", utility_atom_ids=None,
        llm_client=BoundaryScheduleLLM(), model_name="stub", semantic_contract_certifiable=True,
    )
    assert {str(node_id).split("::")[1] for node_id in after_schedule["bindings"]["schedule"]["anchor_slot_node_ids"]} == {
        "appointment_b",
    }, after_schedule
    assert after_schedule["diagnostics"]["collection_completion"]["full_set_audits"]["schedule"]["selected_record_count"] == 1

    class OpenSchemaCollectionLLM:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                return {"bindings": []}
            return {"bindings": [{
                "attribute": "records", "binding_kind": "collection",
                "source_atom_ids": ["record_a", "record_b"], "query_support_span": "records",
                "fact_support_spans": [
                    {"source_atom_id": "record_a", "fact_support_span": "Observed record value is alpha."},
                    {"source_atom_id": "record_b", "fact_support_span": "Observed record value is bravo."},
                ],
            }]}

    fallback = align_requested_attributes(
        question="Which records?", semantic_spec=collection_spec, graph=collection_graph, owner_id="owner",
        utility_atom_ids=None, llm_client=OpenSchemaCollectionLLM(), model_name="stub",
        semantic_contract_certifiable=True,
    )
    assert set(fallback["bindings"]["records"]["anchor_slot_node_ids"]) == {
        slot_ids_by_atom["record_a"], slot_ids_by_atom["record_b"],
    }, fallback

    # Two independently requested scalar properties must not silently share
    # one source slot. The repair turn receives the duplicate property and
    # binds it to its distinct evidence-local field.
    paired = GovernedMemoryAtom(
        atom_id="paired", atom_type="fact_atom", text="Budget is 214 USD and discount cap is 6%.",
        slots={"budget": "214 USD", "discount_cap": "6%"}, owner_id="owner", subject_id="paired",
        speaker_id="owner", source_turn=4, timestamp="2026-01-01T00:03:00", lifecycle="active",
        sensitivity="private", access_scope=["self"], related_entities=[], confidence=1.0,
        provenance={"source_message_ids": ["paired"], "evidence_span": "Budget is 214 USD and discount cap is 6%."},
    )
    paired_graph = GovernedGraphBuilder().build(graph_id="paired", instance_id="smoke", atoms=[paired])
    paired_slots = {
        str((node.attributes or {}).get("slot_name")): node.node_id
        for node in paired_graph.nodes if node.node_type == "SlotNode"
    }

    class DuplicateScalarLLM:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"bindings": [
                    {"attribute": "budget", "slot_node_ids": [paired_slots["budget"]],
                     "query_support_span": "budget", "fact_support_spans": [{"slot_node_id": paired_slots["budget"], "fact_support_span": "Budget is 214 USD"}]},
                    {"attribute": "discount", "slot_node_ids": [paired_slots["budget"]],
                     "query_support_span": "discount", "fact_support_spans": [{"slot_node_id": paired_slots["budget"], "fact_support_span": "Budget is 214 USD"}]},
                ]}
            return {"bindings": [{
                "attribute": "discount", "slot_node_ids": [paired_slots["discount_cap"]],
                "query_support_span": "discount", "fact_support_spans": [{"slot_node_id": paired_slots["discount_cap"], "fact_support_span": "discount cap is 6%"}],
            }]}

    paired_result = align_requested_attributes(
        question="What are the budget and discount?",
        semantic_spec={"requested_attributes": ["budget", "discount"], "attribute_bindings": [
            {"attribute": "budget", "support_span": "budget", "semantic_role": "requested_property"},
            {"attribute": "discount", "support_span": "discount", "semantic_role": "requested_property"},
        ], "temporal_scope": "current"},
        graph=paired_graph, owner_id="owner", utility_atom_ids=None, llm_client=DuplicateScalarLLM(),
        model_name="stub", semantic_contract_certifiable=True,
    )
    assert paired_result["available"], paired_result
    assert paired_result["bindings"]["budget"]["anchor_slot_node_id"] == paired_slots["budget"], paired_result
    assert paired_result["bindings"]["discount"]["anchor_slot_node_id"] == paired_slots["discount_cap"], paired_result
    assert paired_result["diagnostics"]["rejection_counts"].get("duplicate_scalar_slot_across_attributes") == 1, paired_result

    # A typed claim can expose contextual and value slots together. The claim
    # audit receives only this source's closed slots and must select the value
    # that realizes the requested property.
    contract = GovernedMemoryAtom(
        atom_id="contract", atom_type="fact_atom",
        text="Contract term is fixed twelve months plus mutual written renewal. That is the current approved structure.",
        slots={
            "contract_term_structure": "fixed twelve months plus mutual written renewal",
            "contract_term_structure_is_current_approved": "the current approved structure",
        },
        owner_id="owner", subject_id="contract", speaker_id="owner", source_turn=5,
        timestamp="2026-01-01T00:05:00", lifecycle="active", sensitivity="private", access_scope=["self"],
        related_entities=[], confidence=1.0,
        provenance={"source_message_ids": ["contract"], "evidence_span": "Contract term is fixed twelve months plus mutual written renewal. That is the current approved structure."},
    )
    contract_graph = GovernedGraphBuilder().build(graph_id="contract", instance_id="smoke", atoms=[contract])
    contract_slots = {
        str((node.attributes or {}).get("slot_name")): node.node_id
        for node in contract_graph.nodes if node.node_type == "SlotNode"
    }

    class StatusPredicateRepairLLM:
        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def chat_json(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                slot_id = contract_slots["contract_term_structure_is_current_approved"]
                return {"bindings": [{
                    "attribute": "contract_structure", "slot_node_ids": [slot_id],
                    "query_support_span": "current contract structure",
                    "fact_support_spans": [{"slot_node_id": slot_id, "fact_support_span": "That is the current approved structure."}],
                }]}
            assert "complete typed claim" in str(kwargs["user_prompt"]), kwargs["user_prompt"]
            slot_id = contract_slots["contract_term_structure"]
            return {"bindings": [{
                "attribute": "contract_structure", "slot_node_ids": [slot_id],
                "query_support_span": "current contract structure",
                "fact_support_spans": [{"slot_node_id": slot_id, "fact_support_span": "Contract term is fixed twelve months plus mutual written renewal."}],
            }]}

    contract_result = align_requested_attributes(
        question="What is the current contract structure?",
        semantic_spec={"requested_attributes": ["contract_structure"], "attribute_bindings": [{
            "attribute": "contract_structure", "support_span": "current contract structure", "semantic_role": "requested_property",
        }], "temporal_scope": "current"},
        graph=contract_graph, owner_id="owner", utility_atom_ids=None, llm_client=StatusPredicateRepairLLM(),
        model_name="stub", semantic_contract_certifiable=True,
    )
    assert contract_result["available"], contract_result
    assert contract_result["bindings"]["contract_structure"]["anchor_slot_node_id"] == contract_slots["contract_term_structure"], contract_result

    # A source-grounded empty-set state is a fact, not a missing field. It
    # must remain alignable without replaying neighboring sensitive slots.
    empty_state = GovernedMemoryAtom(
        atom_id="empty_state", atom_type="fact_atom", text="There are no open blockers now.",
        slots={"open_blockers_state": "There are no open blockers now."}, owner_id="owner",
        subject_id="empty_state", speaker_id="owner", source_turn=5, timestamp="2026-01-01T00:04:00",
        lifecycle="active", sensitivity="private", access_scope=["self"], related_entities=[], confidence=1.0,
        provenance={"source_message_ids": ["empty_state"], "evidence_span": "There are no open blockers now."},
    )
    empty_graph = GovernedGraphBuilder().build(graph_id="empty", instance_id="smoke", atoms=[empty_state])
    empty_slot = next(node.node_id for node in empty_graph.nodes if node.node_type == "SlotNode")

    class EmptyStateLLM:
        def is_available(self):
            return True

        def chat_json(self, **_kwargs):
            return {"bindings": [{
                "attribute": "open_blockers", "slot_node_ids": [empty_slot],
                "query_support_span": "blockers remain open",
                "fact_support_spans": [{"slot_node_id": empty_slot, "fact_support_span": "There are no open blockers now."}],
            }]}

    empty_result = align_requested_attributes(
        question="Which blockers remain open?",
        semantic_spec={"requested_attributes": ["open_blockers"], "attribute_bindings": [{
            "attribute": "open_blockers", "support_span": "blockers remain open", "semantic_role": "requested_property",
            "need_kind": "record_collection",
        }]}, graph=empty_graph, owner_id="owner", utility_atom_ids=None, llm_client=EmptyStateLLM(),
        model_name="stub", semantic_contract_certifiable=True,
        stage2_record_atom_ids_by_attribute={"open_blockers": {"empty_state"}},
    )
    assert empty_result["available"], empty_result
    assert empty_result["bindings"]["open_blockers"]["anchor_slot_node_id"] == empty_slot, empty_result
    assert empty_result["bindings"]["open_blockers"]["binding_kind"] == "scalar", empty_result
    assert empty_result["diagnostics"]["stage2_empty_state_attributes"] == ["open_blockers"], empty_result
    print("semantic_alignment_attestation_smoke=PASS")


if __name__ == "__main__":
    main()
