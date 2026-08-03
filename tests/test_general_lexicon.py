from gov_mem.general_lexicon import (
    GENERAL_OBJECT_LEXICON,
    GENERAL_TOPIC_LEXICON,
    GENERAL_VALUE_HEAD_LEXICON,
    topics_from_text,
)
from gov_mem.query_semantics import (
    HOUSEHOLD_DELIVERY_SLOT_ALIASES,
    infer_household_delivery_slots,
)


def test_topic_lexicon_contains_atomic_general_categories():
    expected = {
        "work", "medical", "education", "academic", "research", "finance",
        "economics", "legal", "privacy", "technology", "document", "environment",
    }
    assert expected.issubset(GENERAL_TOPIC_LEXICON)
    assert "finance_economics" not in GENERAL_VALUE_HEAD_LEXICON
    assert "finance" in GENERAL_VALUE_HEAD_LEXICON
    assert "economics" in GENERAL_VALUE_HEAD_LEXICON


def test_topics_from_text_is_shared_and_word_boundary_aware():
    topics = set(topics_from_text("My research course starts after the lab appointment."))
    assert {"research", "education", "laboratory", "scheduling"}.issubset(topics)
    assert "education" not in topics_from_text("Please classify this record.")
    assert "laboratory" not in topics_from_text("The collaboration is ordinary.")
    assert "medication" not in topics_from_text("Start the project and stop the meeting.")


def test_object_lexicon_is_not_dataset_specific():
    all_terms = {
        term for values in GENERAL_OBJECT_LEXICON.values() for term in values
    }
    assert "Project Maple" not in all_terms
    assert "Harbor State Exchange" not in all_terms
    assert {"project", "course", "appointment", "account", "address"}.issubset(all_terms)


def test_ordinary_operational_records_get_general_topics():
    topics = set(topics_from_text(
        "Current pantry delivery is 10:45 AM to 11:00 AM at the front desk cold cubby."
    ))
    assert {"logistics", "scheduling", "location"}.issubset(topics)


def test_delivery_vocabulary_is_not_benchmark_phrase_specific():
    terms = {
        term
        for aliases in HOUSEHOLD_DELIVERY_SLOT_ALIASES.values()
        for term in aliases
    }
    assert not terms.intersection({
        "meal-drop window", "floral window", "watering window",
        "watering route", "watering areas", "desk-buzz rule",
        "tart box handling",
    })


def test_delivery_slot_shapes_accept_unseen_operational_modifiers():
    slots = set(infer_household_delivery_slots(
        "What is the collection window, secure route, approved work areas, and weather contingency?"
    ))
    assert {"visit_window", "entry_method", "approved_areas", "fallback_rule"}.issubset(slots)
