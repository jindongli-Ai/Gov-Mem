"""Deterministic, provenance-free entity resolution for policy routing.

This module only resolves observable principal names and role categories.  It
does not authorize access and it never uses benchmark metadata.  The policy
engine treats an ambiguous person reference as a safety boundary rather than
guessing which person's record the query means.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from gov_mem.policy_schema import PrincipalState


_IDENTITY_CUES = re.compile(
    r"\b(?:same|other|another|which|who|whose|identity|mix(?:ed|ing)?|"
    r"confus(?:e|ed|ion)|different|one\s+of|or\s+is|is\s+.+\s+the)\b",
    re.IGNORECASE,
)
_GENERIC_NAME_WORDS = {
    "the", "other", "another", "same", "person", "patient", "client",
    "customer", "student", "resident", "owner", "account", "holder",
}


@dataclass(frozen=True)
class EntityResolution:
    mentioned_principal_ids: tuple[str, ...] = ()
    candidate_principal_ids: tuple[str, ...] = ()
    resolved_principal_id: str | None = None
    ambiguous: bool = False
    reason: str | None = None


def _tokens(value: object) -> tuple[str, ...]:
    return tuple(
        token.casefold()
        for token in re.findall(r"[a-z0-9][a-z0-9'_-]*", str(value or ""))
        if token not in _GENERIC_NAME_WORDS
    )


def _principal_aliases(principal: PrincipalState) -> tuple[str, ...]:
    values = [principal.display_name or "", *principal.aliases]
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\([^)]*\)", "", str(value or "")).strip()
        if cleaned and cleaned.casefold() not in {item.casefold() for item in result}:
            result.append(cleaned)
    return tuple(result)


def _ordered_name_match(query_tokens: tuple[str, ...], name_tokens: tuple[str, ...]) -> bool:
    if not name_tokens:
        return False
    for start in range(len(query_tokens)):
        if query_tokens[start] != name_tokens[0]:
            continue
        cursor = start
        matched = True
        for token in name_tokens[1:]:
            positions = [
                index for index in range(cursor + 1, min(len(query_tokens), cursor + 4))
                if query_tokens[index] == token
            ]
            if not positions:
                matched = False
                break
            cursor = positions[0]
        if matched:
            return True
    return False


def _surname(principal: PrincipalState) -> str | None:
    name_tokens = _tokens(principal.display_name or "")
    return name_tokens[-1] if len(name_tokens) >= 2 else None


def resolve_query_entities(
    query: str,
    principals: Iterable[PrincipalState],
) -> EntityResolution:
    """Resolve person references without collapsing same-name principals."""
    principal_list = tuple(principals)
    query_tokens = _tokens(query)
    query_lower = str(query or "").casefold()
    full_matches: list[PrincipalState] = []
    first_matches: list[PrincipalState] = []
    for principal in principal_list:
        aliases = _principal_aliases(principal)
        if any(_ordered_name_match(query_tokens, _tokens(alias)) for alias in aliases if len(_tokens(alias)) >= 2):
            full_matches.append(principal)
            continue
        first = (_tokens(principal.display_name or "") or ("",))[0]
        if first and re.search(rf"\b{re.escape(first)}\b", query_lower):
            first_matches.append(principal)

    matched = full_matches or first_matches
    surnames = {
        _surname(principal)
        for principal in principal_list
        if _surname(principal) and _surname(principal) in query_tokens
    }
    same_surname = tuple(
        principal for principal in principal_list
        if _surname(principal) in surnames
    )
    identity_query = bool(_IDENTITY_CUES.search(query_lower))
    # A fully qualified name is normally unambiguous.  Add the surname cohort
    # only when the query explicitly compares/contrasts identities.
    if not full_matches and same_surname:
        matched = list(dict.fromkeys((*matched, *same_surname)))
    elif identity_query and same_surname and len(same_surname) > 1:
        matched = list(dict.fromkeys((*matched, *same_surname)))

    candidate_ids = tuple(dict.fromkeys(principal.principal_id for principal in matched))
    roles = {
        str(principal.role or "").casefold()
        for principal in matched
        if principal.principal_id in candidate_ids
    }
    same_category = len(roles) == 1 and bool(roles)
    ambiguous = len(candidate_ids) > 1 and (identity_query or same_category or bool(surnames))
    if ambiguous:
        return EntityResolution(
            mentioned_principal_ids=candidate_ids,
            candidate_principal_ids=candidate_ids,
            ambiguous=True,
            reason="multiple observable principals match the person reference",
        )
    if len(candidate_ids) == 1:
        return EntityResolution(
            mentioned_principal_ids=candidate_ids,
            candidate_principal_ids=candidate_ids,
            resolved_principal_id=candidate_ids[0],
        )
    return EntityResolution()


def principal_mentions(text: str, principals: Iterable[PrincipalState]) -> tuple[str, ...]:
    """Return explicitly name-grounded principal IDs for a visible record."""
    resolution = resolve_query_entities(text, principals)
    return resolution.candidate_principal_ids
