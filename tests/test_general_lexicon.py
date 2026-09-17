from gov_mem.general_lexicon import (
    GENERAL_OBJECT_LEXICON,
    GENERAL_OBJECT_PREFIXES,
    GENERAL_TOPIC_LEXICON,
    GOVMEM_GOVERNANCE_ONTOLOGY,
    governance_ontology_terms,
    governance_ontology_word_count,
    lexicon_terms_match,
    topics_from_text,
)


def test_release_path_contains_no_general_semantic_vocabularies():
    assert GENERAL_OBJECT_LEXICON == {}
    assert GENERAL_TOPIC_LEXICON == {}
    assert GENERAL_OBJECT_PREFIXES == frozenset()
    assert GOVMEM_GOVERNANCE_ONTOLOGY == {}
    assert governance_ontology_terms() == ()
    assert governance_ontology_word_count() == 0


def test_topics_are_not_deterministically_inferred_from_text():
    assert topics_from_text("A private appointment has a current address.") == ()


def test_caller_provided_phrase_matching_is_not_a_builtin_ontology():
    assert lexicon_terms_match("The current number is 42.", "number")
    assert not lexicon_terms_match("The current number is 42.", "address")
