"""Compatibility surface after removing deterministic semantic vocabularies.

The paper-facing Gov-Mem-v4 pipeline does not classify topics, fields, or
protected scopes from a hand-written word list. Dynamic query contracts and
source-grounded atoms are produced by the semantic compiler instead.
"""

from __future__ import annotations

import re


# Keep import compatibility for legacy modules without retaining a semantic
# vocabulary that could be mistaken for a benchmark-tailored ontology.
GENERAL_OBJECT_LEXICON: dict[str, tuple[str, ...]] = {}
GENERAL_TOPIC_LEXICON: dict[str, tuple[str, ...]] = {}
GENERAL_OBJECT_PREFIXES = frozenset()
GOVMEM_GOVERNANCE_ONTOLOGY: dict[str, tuple[str, ...]] = {}


def lexicon_terms_match(text: str, term: str) -> bool:
    """Compare caller-provided text without consulting a built-in vocabulary."""
    value = str(text or "").casefold()
    phrase = str(term or "").strip().casefold()
    if not phrase:
        return False
    pattern = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", value) is not None


def topics_from_text(text: str) -> tuple[str, ...]:
    """Return no deterministic topics; semantic compilation owns this task."""
    del text
    return ()


def governance_ontology_terms() -> tuple[str, ...]:
    return ()


def governance_ontology_word_count() -> int:
    return 0
