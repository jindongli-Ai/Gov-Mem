import unittest

from gov_mem.answer_projection import (
    AnswerField,
    AnswerRequest,
    compile_answer_request,
    projection_coverage_gaps,
    project_field_evidence,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def is_available(self):
        return True

    def chat_json(self, **kwargs):
        return self.responses.pop(0)


EVIDENCE = [
    {
        "memory_id": "m-window",
        "text": "The current appointment is Friday, September 19, 2026, at Harbor Clinic.",
        "source_message_ids": ["msg-window"],
        "source_turn_index": 4,
    },
    {
        "memory_id": "m-method",
        "text": "Use the north entrance and bring the referral form.",
        "source_message_ids": ["msg-method"],
        "source_turn_index": 5,
    },
]


class AnswerProjectionTests(unittest.TestCase):
    def test_compiler_preserves_seed_and_adds_concrete_fields(self):
        client = FakeLLM([{
            "fields": [
                {"field_id": "date", "label": "appointment date", "temporal_role": "event"},
                {"field_id": "location", "label": "appointment location", "entity": "appointment"},
                {"field_id": "method", "label": "entry method", "cardinality": "list"},
            ],
        }])
        request = compile_answer_request(
            question="What is the appointment date, location, and entry method?",
            target_subject="appointment",
            request_scope="standard",
            seed_fields=["appointment date"],
            answer_need_spec=None,
            evidence_payload=EVIDENCE,
            llm_client=client,
            config={"llm": {"answering_model": "test-model"}},
        )
        labels = [field.label for field in request.fields]
        self.assertIn("appointment date", labels)
        self.assertIn("appointment location", labels)
        self.assertIn("entry method", labels)
        self.assertEqual(next(field for field in request.fields if field.label == "entry method").cardinality, "list")

    def test_projection_rejects_unallowed_and_unlocated_values(self):
        request = AnswerRequest(fields=(
            AnswerField(field_id="location", label="appointment location"),
        ))
        client = FakeLLM([{
            "fields": [{
                "field_id": "location",
                "status": "supported",
                "candidate_values": ["Harbor Clinic", "Hidden Clinic"],
                "selected_values": ["Harbor Clinic", "Hidden Clinic"],
                "source_memory_ids": ["m-window", "blocked"],
            }],
        }])
        projection = project_field_evidence(
            request=request,
            question="Where is the appointment?",
            evidence_payload=EVIDENCE,
            llm_client=client,
            config={"llm": {"answering_model": "test-model"}},
        )
        field = projection.fields[0]
        self.assertEqual(field.source_memory_ids, ("m-window",))
        self.assertEqual(field.selected_values, ("Harbor Clinic",))
        self.assertNotIn("Hidden Clinic", field.selected_values)

    def test_projection_coverage_detects_missing_list_member(self):
        request = AnswerRequest(fields=(
            AnswerField(field_id="method", label="entry method", cardinality="list"),
        ))
        client = FakeLLM([{
            "fields": [{
                "field_id": "method",
                "status": "supported",
                "candidate_values": ["north entrance", "referral form"],
                "selected_values": ["north entrance", "referral form"],
                "source_memory_ids": ["m-method"],
            }],
        }])
        projection = project_field_evidence(
            request=request,
            question="What entry method and materials are required?",
            evidence_payload=EVIDENCE,
            llm_client=client,
            config={"llm": {"answering_model": "test-model"}},
        )
        gaps = projection_coverage_gaps(
            projection,
            answer_text="Use the north entrance.",
            contract={"requested_fields": [{"label": "entry method", "status": "covered"}]},
        )
        self.assertTrue(any("referral form" in gap for gap in gaps))

    def test_projection_coverage_passes_when_all_selected_values_are_rendered(self):
        request = AnswerRequest(fields=(
            AnswerField(field_id="date", label="appointment date"),
        ))
        client = FakeLLM([{
            "fields": [{
                "field_id": "date",
                "status": "supported",
                "selected_values": ["Friday, September 19, 2026"],
                "source_memory_ids": ["m-window"],
            }],
        }])
        projection = project_field_evidence(
            request=request,
            question="When is the appointment?",
            evidence_payload=EVIDENCE,
            llm_client=client,
            config={"llm": {"answering_model": "test-model"}},
        )
        gaps = projection_coverage_gaps(
            projection,
            answer_text="The appointment is Friday, September 19, 2026.",
            contract={"requested_fields": [{"label": "appointment date", "status": "covered"}]},
        )
        self.assertEqual(gaps, [])

    def test_schedule_projection_preserves_qualified_weekday_span(self):
        from gov_mem.answer_projection import AnswerField, AnswerRequest, _normalize_projection

        request = AnswerRequest(fields=(AnswerField(field_id="window", label="delivery window"),))
        rows = [{
            "memory_id": "m1",
            "text": "The current delivery window is Sunday, November 29, 10:45 AM to 11:00 AM at the front desk.",
            "source_turn_index": 58,
        }]
        fields = _normalize_projection(
            {"fields": [{"field_id": "window", "status": "supported", "selected_values": ["10:45 AM to 11:00 AM"], "source_memory_ids": ["m1"]}]},
            request=request,
            evidence_payload=rows,
        )
        assert fields[0].selected_values == ("Sunday, November 29, 10:45 AM to 11:00 AM",)


if __name__ == "__main__":
    unittest.main()
