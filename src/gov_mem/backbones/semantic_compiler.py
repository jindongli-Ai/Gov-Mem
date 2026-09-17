"""Lexicon-free, query-conditioned semantic compilation.

The compiler proposes open-vocabulary query slots and source-grounded atoms.
It never makes an authorization decision and never expands the retrieved
evidence boundary.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.leakage_guard import assert_runtime_payload_safe
from gov_mem.llm.client import LLMClient


@dataclass
class RequestedSlot:
    slot_id: str
    slot_name: str
    slot_description: str = ""
    target_entity: str | None = None
    value_type: str = "unknown"
    temporal_requirement: str = "unknown"
    sensitivity_possible: bool = False
    authorization_relevant: bool = False
    required_for_answer: bool = True
    confidence: float = 0.0


@dataclass
class QuerySemanticContract:
    target_entities: list[dict[str, Any]] = field(default_factory=list)
    requested_slots: list[RequestedSlot] = field(default_factory=list)
    requester_relations: list[dict[str, Any]] = field(default_factory=list)
    temporal_intent: dict[str, Any] = field(default_factory=lambda: {"mode": "unknown", "reference": None})
    answer_shape: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundedMemoryAtom:
    atom_id: str
    target_entity: str | None
    slot_id: str
    slot_name: str
    value: str | None = None
    owner: str | None = None
    requester_relation: str | None = None
    temporal: dict[str, Any] = field(default_factory=dict)
    authorization_semantics: dict[str, Any] = field(default_factory=dict)
    lifecycle_semantics: dict[str, Any] = field(default_factory=dict)
    sensitivity_semantics: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingResult:
    atom_id: str
    source_valid: bool
    span_valid: bool
    value_valid: bool
    slot_valid: bool
    entity_alignment: str
    accepted_for_symbolic_reasoning: bool
    reasons: list[str] = field(default_factory=list)
    temporal_anchor_valid: bool = True


@dataclass
class SlotCoverageReport:
    required_slots: list[str]
    covered_slots: list[str]
    missing_slots: list[str]
    coverage_ratio: float

    @property
    def has_missing_required_slots(self) -> bool:
        return bool(self.missing_slots)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlotResolvabilityReport:
    required_slots: list[str]
    resolvable_slots: list[str]
    unresolved_slots: list[str]
    resolvability_ratio: float

    @property
    def has_unresolved_required_slots(self) -> bool:
        return bool(self.unresolved_slots)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticCompilationResult:
    query_contract: QuerySemanticContract
    initial_atoms: list[dict[str, Any]] = field(default_factory=list)
    grounding_results: list[dict[str, Any]] = field(default_factory=list)
    verified_atoms: list[dict[str, Any]] = field(default_factory=list)
    coverage_before_repair: dict[str, Any] = field(default_factory=dict)
    repair_triggered: bool = False
    repair_missing_slots: list[str] = field(default_factory=list)
    repair_atoms: list[dict[str, Any]] = field(default_factory=list)
    coverage_after_repair: dict[str, Any] = field(default_factory=dict)
    final_grounded_atoms: list[dict[str, Any]] = field(default_factory=list)
    resolvability_before_repair: dict[str, Any] = field(default_factory=dict)
    resolvability_after_repair: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


INDUCTION_SYSTEM_PROMPT = """You are a semantic compiler for a memory governance system.
Return JSON only. Analyze only the supplied question and visible requester context.
Do not answer the question, choose an expected action, decide allow/refuse, infer
benchmark labels, or use hidden information. Discover the fields requested by
this question in open vocabulary. Split every independently answerable field
into a separate requested slot. Preserve temporal qualifiers and requester,
owner, and target relations. If uncertain, use unknown and lower confidence.
When the question names a broad plan, status, overview, or similar aggregate
and then enumerates concrete fields after it (for example, "the current plan,
including the window and location"), emit the concrete fields as required
slots and mark the aggregate description `required_for_answer=false`. The
aggregate is an answer shape, not an additional fact that must be extracted.
Do not use a fixed domain vocabulary to decide this; use the sentence
structure and the supplied question only.
The slot_name must be the phrase induced from the current question, not a fixed
taxonomy label. For every requested slot, explicitly assess
sensitivity_possible and authorization_relevant from the request wording only:
these mean that disclosure may need governance, not that access is denied.
For an ordinary operational/status/value question, set both flags to false
unless the question itself explicitly asks for a private, restricted,
confidential, exact, credential, or identity-linked value. Credential/access
identifiers (including prefixes or partial identifiers), clinical diagnoses or
conditions, and identity-linked private facts are governance-sensitive even
when the question omits the adjective "private"; mark those slots accordingly.
Never infer these
flags from words found only in supplied evidence, the requester's role, or the
existence of a governance graph. A request to confirm or infer a private
current value, or to use that value to route/materially act, is itself
authorization-relevant even when it is phrased as a conditional or indirect
question. Treat entry codes, access identifiers, passphrases, and handoff
secrets as governance-sensitive even when the question is framed as routine
coordination or efficiency. Treat requests to infer whether a person or
household is present/absent from a schedule as identity/location-sensitive.
Use the exact output
keys shown by the schema below. Do not omit requested_slots when the question
contains an independently answerable property; use an open-vocabulary phrase
copied or lightly normalized from the question. Return exactly this shape:
{"target_entities":[{"canonical_reference":"...","surface_form":"...","confidence":0.0}],
 "requested_slots":[{"slot_id":"qslot_0","slot_name":"...","slot_description":"",
 "target_entity":null,"value_type":"unknown",
 "temporal_requirement":"current|historical|specific_time|any|unknown",
 "sensitivity_possible":false,"authorization_relevant":false,
 "required_for_answer":true,"confidence":0.0}],
 "requester_relations":[{"requester":"...","target":"...","relation":"...","confidence":0.0}],
 "temporal_intent":{"mode":"current|historical|specific_time|any|unknown","reference":null},
 "answer_shape":{"single_field":false,"multi_field":false,"summary":false}}"""

EXTRACTION_SYSTEM_PROMPT = """Extract high-recall candidate memory atoms for the supplied query contract.
Use only the supplied retrieved evidence. Do not answer the question, decide
authorization, select a final current value, or infer absent facts. Multiple
candidates for one slot are allowed. Every atom must cite one supplied source
chunk and a verbatim source span. Missing fields must be null or unknown.
For a current or latest slot, preserve every explicitly stated candidate that
is present in the retrieved set, including competing earlier and later values;
use the supplied turn_index/as_of_turn_id only as temporal provenance, not as
a reason to discard a candidate at extraction time.
When a value is qualified by a weekday, calendar date, date range, or other
temporal anchor, preserve that anchor explicitly in `temporal`. Set
`temporal.raw` to the exact temporal phrase and, when possible, set
`temporal.anchor_span` to the smallest exact contiguous calendar phrase. If
the anchor is in another supplied evidence row, include `temporal.anchor_source`
with the closed-set chunk_id, turn_id, message_id, and exact anchor span. Never
invent a date from a record timestamp. For a value update that omits its date,
still return any date anchor explicitly stated by another candidate for the
same query slot so the symbolic state ledger can audit whether the anchor is
inherited or conflicting.
For each contract slot, scan all supplied evidence rows even when the slot
name is not repeated verbatim. If a sentence states a paired object, a list,
or a compact multi-item description that answers the slot, copy the exact
contiguous value phrase from that sentence. Do not require the field label to
appear in the source span.
When a source explicitly states a permission, prohibition, or revocation that
governs a requested slot, preserve it as an atom authorization_semantics object
with type allow, deny, revoke, or none; include its explicitly named subject
and resource. If the subject or resource is stated in a different supplied
evidence span, include subject_source/resource_source objects with closed-set
chunk_id, turn_id, message_id, and verbatim span provenance. This is a
source-level candidate only, never an access decision.
When a source explicitly marks the requested value or slot as private,
restricted, confidential, or public, set sensitivity_semantics.type to
restricted, ordinary, or unknown. This is a source-level label, never an
authorization decision; do not mark a value restricted without a verbatim
source span supporting that label.
If value is non-null, copy it as one exact contiguous substring of source.span.
Do not add an entity, unit, label, or qualifier that is outside source.span.
For cancel/delete lifecycle candidates, include target_atom_id when the target
is another extracted atom, or target_value copied verbatim from the same span;
otherwise leave the lifecycle target ambiguous.
For every atom, source is REQUIRED and must be an object exactly shaped as
{\"chunk_id\": \"one supplied chunk_id\", \"turn_id\": \"matching supplied turn_id\",
 \"message_id\": \"matching supplied message_id\", \"span\": \"verbatim substring of text\"}.
Never put a source ID or a sentence directly in source; source must be an
object. Each atom MUST have atom_id, target_entity, slot_id, slot_name, value,
temporal, authorization_semantics, lifecycle_semantics, sensitivity_semantics,
source, and confidence.
Use this structural pattern; its strings are placeholders, not facts:
{\"atoms\":[{\"atom_id\":\"atom_0\",\"target_entity\":null,\"slot_id\":\"qslot_0\",
\"slot_name\":\"the matching contract slot name\",\"value\":\"verbatim value or null\",
\"temporal\":{\"raw\":null,\"normalized\":null,\"state\":\"unknown\"},
\"authorization_semantics\":{\"type\":\"none\"},\"lifecycle_semantics\":{\"type\":\"none\"},
\"sensitivity_semantics\":{\"type\":\"unknown\"},
\"source\":{\"chunk_id\":\"a supplied chunk_id\",\"turn_id\":\"matching turn_id\",
\"message_id\":\"matching message_id\",\"span\":\"verbatim evidence substring\"},
\"confidence\":0.8}]}. Return JSON only with {\"atoms\": [...]}."""

REPAIR_SYSTEM_PROMPT = """Reinspect only the supplied retrieved evidence for the listed query slots.
Some slots may be missing, while others may already have an older candidate.
For a reconciliation slot, look for additional source-grounded candidates for
the same slot, especially a later explicit update or replacement in the
supplied evidence. Keep the existing candidate valid; return the additional
candidate rather than choosing a final value. The symbolic state ledger will
resolve turn order after extraction.
Return JSON only with {\"atoms\": [...]} or an empty list. Reinspect every
supplied evidence row for every listed slot. If a sentence contains a
paired object, list, or compact multi-item description answering a missing
slot, extract its exact contiguous value phrase even when the slot label is
not repeated verbatim. Produce only source-grounded candidates with verbatim
spans. Every atom source MUST be an
object with chunk_id, turn_id, message_id, and a verbatim span from that chunk.
Do not infer a value that is not present, answer the question, access other
turns, or make an authorization decision. If a permission relation spans two
supplied chunks, preserve the relation only with explicit subject_source and/or
resource_source provenance objects; never resolve an ungrounded pronoun. Each atom must include slot_id,
slot_name, value, temporal, authorization_semantics, lifecycle_semantics, sensitivity_semantics,
confidence, and source. The source object is mandatory:
{\"source\":{\"chunk_id\":\"supplied chunk ID\",\"turn_id\":\"matching turn ID\",
\"message_id\":\"matching message ID\",\"span\":\"verbatim evidence substring\"}}.
Any non-null value must be an exact contiguous substring of source.span.
Return JSON only."""


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _mapping(value: Any, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a shallow mapping without coercing malformed model output."""

    if isinstance(value, dict):
        return dict(value)
    return dict(default or {})


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    """Keep only object items from an LLM field declared as a JSON array."""

    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalized_grounding_text(value: Any) -> str:
    """Normalize only punctuation and whitespace for deterministic grounding."""

    return " ".join(re.findall(r"\w+", str(value or "").casefold()))


def grounded_text_contains(text: Any, fragment: Any) -> bool:
    normalized = _normalized_grounding_text(fragment)
    return bool(normalized and f" {normalized} " in f" {_normalized_grounding_text(text)} ")


def _visible_evidence(evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    rows = []
    for row in evidence:
        record = dict((row.metadata or {}).get("structured_record") or {})
        rows.append({
            "chunk_id": row.memory_id,
            "memory_id": row.memory_id,
            "turn_id": record.get("turn_id") or record.get("message_id"),
            "message_id": record.get("message_id"),
            "turn_index": record.get("turn_index"),
            "as_of_turn_id": (record.get("checkpoint") or {}).get("as_of_turn_id"),
            "speaker_principal_id": (record.get("speaker") or {}).get("principal_id"),
            "speaker_role": (record.get("speaker") or {}).get("role"),
            "text": str(record.get("text") or row.content or ""),
        })
    return rows


def _surface_compatible(question: str, phrase: str) -> bool:
    """Keep contract labels grounded in the current question surface.

    This is a structural anti-hallucination check, not a semantic vocabulary:
    the compiler may only name a slot/entity using words present in the query.
    """

    question_tokens = set(re.findall(r"[a-z0-9]+", str(question or "").casefold()))
    phrase_tokens = [token for token in re.findall(r"[a-z0-9]+", str(phrase or "").casefold())]
    if not phrase_tokens or not question_tokens:
        return False
    return set(phrase_tokens).issubset(question_tokens)


def _minimum_independent_slot_count(question: str) -> int:
    """Count explicitly coordinated interrogative clauses structurally.

    This is deliberately syntax-only: it does not name a domain, field, or
    answer vocabulary.  It catches queries such as ``what is A, and what is
    B`` where an induction model may accidentally emit only the first slot.
    """

    text = str(question or "").strip().casefold()
    if not text or not re.match(r"^(?:what|which|who|where|when|why|how)\b", text):
        return 0
    coordinated = re.findall(
        r"\b(?:and|or)\s+(?:what|which|who|where|when|why|how)\b", text,
    )
    return 1 + len(coordinated)


def _structural_slot_surfaces(question: str) -> list[tuple[str, str | None]]:
    """Recover missing property spans from coordinated question syntax.

    The result is only a list of verbatim query phrases plus an optional
    subject from a ``does X have`` clause.  It deliberately performs no
    domain interpretation and is used only when LLM induction undercounts
    explicit interrogative clauses.
    """

    text = str(question or "").strip()
    if _minimum_independent_slot_count(text) <= 1:
        return []
    clauses = re.split(
        r"\s*,\s*(?:and|or)\s+(?=(?:what|which|who|where|when|why|how)\b)",
        text,
        flags=re.IGNORECASE,
    )
    surfaces: list[tuple[str, str | None]] = []
    for clause in clauses:
        clause = clause.strip().rstrip("?.!").strip()
        clause = re.sub(
            r"^(?:what|which|who|where|when|why|how)\s+",
            "",
            clause,
            count=1,
            flags=re.IGNORECASE,
        )
        # A copular clause has its property after ``is/are``.  Keep temporal
        # and scope qualifiers because they are part of the requested span.
        clause = re.sub(r"^(?:is|are|was|were)\s+", "", clause, count=1, flags=re.IGNORECASE)
        subject: str | None = None
        auxiliary = re.search(r"\s+(?:does|do|did|has|have)\s+", clause, flags=re.IGNORECASE)
        if auxiliary:
            property_span = clause[:auxiliary.start()].strip()
            subject_span = clause[auxiliary.end():].strip()
            subject_span = re.sub(
                r"\s+(?:currently|right\s+now|now)\s+(?:have|has|had)\s*$",
                "",
                subject_span,
                flags=re.IGNORECASE,
            ).strip()
            subject_match = re.search(r"\b[A-Z][A-Za-z0-9_-]*\b", subject_span)
            subject = subject_match.group(0) if subject_match else None
        else:
            property_span = clause
        property_span = re.sub(r"^(?:the|a|an)\s+", "", property_span, flags=re.IGNORECASE).strip()
        if property_span:
            surfaces.append((property_span, subject))
    return surfaces


def _surface_token_set(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _append_structural_slots(
    contract: QuerySemanticContract,
    question: str,
) -> QuerySemanticContract:
    """Append only missing coordinated-clause slots to an induced contract."""

    surfaces = _structural_slot_surfaces(question)
    if not surfaces:
        return contract
    slots = list(contract.requested_slots)
    next_index = len(slots)
    for surface, subject in surfaces:
        surface_tokens = _surface_token_set(surface)
        if not surface_tokens:
            continue
        already_present = False
        for slot in slots:
            slot_tokens = _surface_token_set(slot.slot_name)
            overlap = len(surface_tokens & slot_tokens)
            if overlap >= 2 and overlap / max(1, min(len(surface_tokens), len(slot_tokens))) >= 0.6:
                already_present = True
                break
        if already_present:
            continue
        slots.append(RequestedSlot(
            slot_id=f"qslot_{next_index}",
            slot_name=surface,
            target_entity=subject,
            value_type="unknown",
            temporal_requirement="current" if re.search(
                r"\b(?:current|currently|latest|now|still)\b", surface, flags=re.IGNORECASE
            ) else "unknown",
            required_for_answer=True,
            confidence=0.0,
        ))
        next_index += 1
    if len(slots) == len(contract.requested_slots):
        return contract
    return QuerySemanticContract(
        target_entities=list(contract.target_entities),
        requested_slots=slots,
        requester_relations=list(contract.requester_relations),
        temporal_intent=dict(contract.temporal_intent),
        answer_shape={**dict(contract.answer_shape), "multi_field": True},
    )


def _contract(raw: Any, *, question: str = "") -> QuerySemanticContract:
    raw = raw if isinstance(raw, dict) else {}
    slots = []
    for index, item in enumerate(_mapping_list(raw.get("requested_slots"))):
        name = str(item.get("slot_name") or item.get("name") or "").strip()
        if not name or not _surface_compatible(question, name):
            continue
        slots.append(RequestedSlot(
            slot_id=str(item.get("slot_id") or f"qslot_{index}"),
            slot_name=name,
            slot_description=str(item.get("slot_description") or ""),
            target_entity=str(item.get("target_entity") or "") or None,
            value_type=str(item.get("value_type") or "unknown"),
            temporal_requirement=str(item.get("temporal_requirement") or "unknown"),
            sensitivity_possible=bool(item.get("sensitivity_possible", False)),
            authorization_relevant=bool(item.get("authorization_relevant", False)),
            required_for_answer=bool(item.get("required_for_answer", True)),
            confidence=_confidence(item.get("confidence")),
        ))
    # A model may normalize a query surface ("Mia") to an episode-local
    # canonical reference ("Mia Santos"). The surface form must be present in
    # the query; the canonical reference is checked later against the runtime
    # graph and is not itself a vocabulary.
    target_entities = []
    for item in _mapping_list(raw.get("target_entities")):
        surface = str(item.get("surface_form") or "").strip()
        canonical = str(item.get("canonical_reference") or "").strip()
        if not _surface_compatible(question, surface or canonical):
            continue
        target_entities.append({
            **item,
            "surface_form": surface or canonical,
            "canonical_reference": canonical or surface,
        })
    # Bind an omitted slot target to the unique query-conditioned entity. This
    # is structural query binding only; it never authorizes disclosure.
    if len(target_entities) == 1:
        bound_target = str(
            target_entities[0].get("canonical_reference")
            or target_entities[0].get("surface_form")
            or ""
        ).strip() or None
        if bound_target:
            for slot in slots:
                if slot.target_entity is None:
                    slot.target_entity = bound_target
    return QuerySemanticContract(
        target_entities=target_entities,
        requested_slots=slots,
        requester_relations=_mapping_list(raw.get("requester_relations")),
        temporal_intent=_mapping(
            raw.get("temporal_intent"),
            default={"mode": "unknown", "reference": None},
        ),
        answer_shape=_mapping(raw.get("answer_shape")),
    )


def _atom(
    raw: Any,
    index: int,
    valid_slot_ids: set[str],
    slot_names: dict[str, str] | None = None,
) -> GroundedMemoryAtom | None:
    if not isinstance(raw, dict):
        return None
    slot_id = str(raw.get("slot_id") or "")
    slot_name = str(raw.get("slot_name") or "").strip()
    source_raw = raw.get("source")
    # Providers occasionally flatten a JSON object despite the requested
    # schema. Normalize only explicit provenance fields, then apply the same
    # closed-evidence and verbatim-span verifier below. This does not infer
    # semantics or relax grounding requirements.
    if isinstance(source_raw, dict):
        source = dict(source_raw)
    elif isinstance(raw.get("chunk_id") or raw.get("memory_id"), str):
        source = {
            "chunk_id": raw.get("chunk_id") or raw.get("memory_id"),
            "turn_id": raw.get("turn_id"),
            "message_id": raw.get("message_id"),
            "span": raw.get("span") or raw.get("source_span"),
        }
    else:
        return None
    if not slot_name:
        slot_name = str((slot_names or {}).get(slot_id) or "").strip()
    expected_slot_name = str((slot_names or {}).get(slot_id) or "").strip()
    if expected_slot_name and slot_name and _normalized_grounding_text(expected_slot_name) != _normalized_grounding_text(slot_name):
        # An extraction model cannot redirect an atom to a new semantic field;
        # it must use the slot selected by query induction.
        return None
    if not slot_id or slot_id not in valid_slot_ids or not slot_name or not source:
        return None
    return GroundedMemoryAtom(
        atom_id=str(raw.get("atom_id") or f"atom_{index}"),
        target_entity=str(raw.get("target_entity") or "") or None,
        slot_id=slot_id,
        slot_name=slot_name,
        value=str(raw.get("value") or "") or None,
        owner=str(raw.get("owner") or "") or None,
        requester_relation=str(raw.get("requester_relation") or "") or None,
        temporal=_mapping(raw.get("temporal")),
        authorization_semantics=_mapping(raw.get("authorization_semantics")),
        lifecycle_semantics=_mapping(raw.get("lifecycle_semantics")),
        sensitivity_semantics=_mapping(raw.get("sensitivity_semantics")),
        source=source,
        confidence=_confidence(raw.get("confidence")),
    )


def verify_grounded_atom(
    atom: GroundedMemoryAtom,
    evidence: list[RetrievedEvidence],
    valid_slots: set[str],
    target_entities: list[dict[str, Any]] | None = None,
) -> GroundingResult:
    by_id = {row.memory_id: row for row in evidence}
    source_id = str(atom.source.get("chunk_id") or atom.source.get("memory_id") or "")
    row = by_id.get(source_id)
    reasons: list[str] = []
    source_valid = row is not None
    text = ""
    if row is not None:
        record = dict((row.metadata or {}).get("structured_record") or {})
        text = str(record.get("text") or row.content or "")
    span = str(atom.source.get("span") or "").strip()
    normalized_text = _normalized_grounding_text(text)
    normalized_span = _normalized_grounding_text(span)
    span_valid = grounded_text_contains(text, span)
    normalized_value = _normalized_grounding_text(atom.value)
    value_valid = not normalized_value or bool(
        grounded_text_contains(span, atom.value)
    )
    temporal = dict(atom.temporal or {})
    anchor_span = str(temporal.get("anchor_span") or "").strip()
    anchor_source = temporal.get("anchor_source")
    temporal_anchor_valid = True
    if anchor_span or anchor_source:
        temporal_anchor_valid = False
        if anchor_span and isinstance(anchor_source, dict):
            anchor_id = str(
                anchor_source.get("chunk_id")
                or anchor_source.get("memory_id")
                or ""
            )
            anchor_row = by_id.get(anchor_id)
            if anchor_row is not None:
                anchor_record = dict(
                    (anchor_row.metadata or {}).get("structured_record") or {}
                )
                anchor_text = str(
                    anchor_record.get("text") or anchor_row.content or ""
                )
                anchor_turn = str(anchor_source.get("turn_id") or "")
                anchor_message = str(anchor_source.get("message_id") or "")
                temporal_anchor_valid = bool(
                    grounded_text_contains(anchor_text, anchor_span)
                    and (
                        not anchor_turn
                        or anchor_turn
                        == str(
                            anchor_record.get("turn_id")
                            or anchor_record.get("message_id")
                            or ""
                        )
                    )
                    and (
                        not anchor_message
                        or anchor_message
                        == str(anchor_record.get("message_id") or "")
                    )
                )
    expected_turn_id = str(record.get("turn_id") or record.get("message_id") or "") if row is not None else ""
    expected_message_id = str(record.get("message_id") or "") if row is not None else ""
    supplied_turn_id = str(atom.source.get("turn_id") or "")
    supplied_message_id = str(atom.source.get("message_id") or "")
    turn_valid = not supplied_turn_id or supplied_turn_id == expected_turn_id
    message_valid = not supplied_message_id or supplied_message_id == expected_message_id
    slot_valid = atom.slot_id in valid_slots
    entity_alignment = "unknown"
    contract_entities = {
        _normalized_grounding_text(
            item.get("canonical_reference") or item.get("surface_form")
        )
        for item in target_entities or []
        if isinstance(item, dict)
    }
    contract_entities.discard("")
    atom_entity = _normalized_grounding_text(atom.target_entity)
    if atom_entity:
        if contract_entities and atom_entity not in contract_entities:
            entity_alignment = "conflict"
        elif grounded_text_contains(text, atom.target_entity):
            entity_alignment = "aligned"
        else:
            entity_alignment = "possible"
    if not source_valid: reasons.append("source_not_in_stage1_evidence")
    if not span_valid: reasons.append("span_not_present_in_source_text")
    if not value_valid: reasons.append("value_not_present_in_source_span")
    if not temporal_anchor_valid: reasons.append("temporal_anchor_not_grounded")
    if not turn_valid: reasons.append("turn_id_mismatch")
    if not message_valid: reasons.append("message_id_mismatch")
    if not slot_valid: reasons.append("slot_not_in_query_contract")
    if entity_alignment == "conflict": reasons.append("target_entity_conflicts_with_query_contract")
    accepted = (
        source_valid and span_valid and value_valid and temporal_anchor_valid and turn_valid
        and message_valid and slot_valid and entity_alignment != "conflict"
    )
    return GroundingResult(
        atom.atom_id, source_valid, span_valid, value_valid, slot_valid,
        entity_alignment, accepted, reasons, temporal_anchor_valid,
    )


def _coverage(contract: QuerySemanticContract, atoms: list[GroundedMemoryAtom]) -> SlotCoverageReport:
    required = [slot.slot_id for slot in contract.requested_slots if slot.required_for_answer]
    # A policy-only or null-valued atom is not answer coverage.  Coverage must
    # represent a usable source-grounded candidate, not merely a slot mention.
    usable = {
        atom.slot_id
        for atom in atoms
        if str(atom.value or "").strip()
        and str(atom.lifecycle_semantics.get("type") or "none").casefold()
        not in {"cancel", "delete"}
    }
    covered = sorted(set(required).intersection(usable))
    missing = [slot_id for slot_id in required if slot_id not in covered]
    return SlotCoverageReport(required, covered, missing, (len(covered) / len(required)) if required else 1.0)


def _resolvability(
    contract: QuerySemanticContract,
    atoms: list[GroundedMemoryAtom],
) -> SlotResolvabilityReport:
    """Check whether required slots have usable temporal value candidates.

    This is weaker than symbolic state resolution: it does not select a value,
    apply authorization, or execute a lifecycle transition. It only prevents a
    policy-only, valueless, or historical-only atom from suppressing repair.
    """

    required = [slot for slot in contract.requested_slots if slot.required_for_answer]
    resolvable: list[str] = []
    for slot in required:
        candidates = [
            atom for atom in atoms
            if atom.slot_id == slot.slot_id and str(atom.value or "").strip()
        ]
        candidates = [
            atom for atom in candidates
            if str(atom.lifecycle_semantics.get("type") or "none").casefold()
            not in {"cancel", "delete"}
        ]
        requirement = str(slot.temporal_requirement or "unknown").casefold()
        if requirement == "current":
            # Extraction may leave temporal state unknown. A grounded value
            # without an explicit temporal assertion is still usable for
            # symbolic state resolution, which applies turn ordering and
            # lifecycle evidence later.
            usable = any(
                str(atom.temporal.get("state") or "unknown").casefold()
                in {"current", "unknown"}
                for atom in candidates
            )
        elif requirement == "historical":
            usable = any(
                str(atom.temporal.get("state") or "unknown").casefold() == "historical"
                for atom in candidates
            )
        else:
            usable = bool(candidates)
        if usable:
            resolvable.append(slot.slot_id)
    unresolved = [slot.slot_id for slot in required if slot.slot_id not in resolvable]
    ratio = (len(resolvable) / len(required)) if required else 1.0
    return SlotResolvabilityReport(
        [slot.slot_id for slot in required], resolvable, unresolved, ratio,
    )


def _prompt_payload(question: str, requester: str | None, role: str | None, evidence: list[RetrievedEvidence]) -> dict[str, Any]:
    payload = {"question": question, "requester": requester, "requester_role": role, "evidence": _visible_evidence(evidence)}
    assert_runtime_payload_safe(payload, context="semantic_compiler_induction")
    return payload


def compile_semantics(*, question: str, requester: str | None, requester_role: str | None,
                      evidence: list[RetrievedEvidence], llm_client: LLMClient,
                      model_name: str, config: dict[str, Any]) -> SemanticCompilationResult:
    cfg = dict(config.get("semantic_compiler") or {})
    if not bool(cfg.get("enabled", False)):
        return SemanticCompilationResult(query_contract=QuerySemanticContract(), audit={"enabled": False})
    payload = _prompt_payload(question, requester, requester_role, evidence)
    audit: dict[str, Any] = {
        "enabled": True,
        "llm_calls": 0,
        "llm_calls_by_stage": {"induction": 0, "extraction": 0, "repair": 0},
        "parse_failures": 0,
        "fixed_lexicon_used": False,
        "dataset_classifier_used": False,
        "evidence_boundary": "stage1_top_k_closed_set",
        "latency_seconds_by_stage": {"induction": 0.0, "extraction": 0.0, "repair": 0.0},
        # The provider client returns parsed JSON rather than raw usage
        # headers. Record this explicitly instead of fabricating token counts.
        "provider_token_usage_available": False,
        "prompt_characters_by_stage": {"induction": 0, "extraction": 0, "repair": 0},
        "fallback_events": [],
    }
    audit["prompt_characters_by_stage"]["induction"] = len(json.dumps(payload, ensure_ascii=False))
    if not bool(dict(cfg.get("query_induction") or {}).get("enabled", True)):
        audit["fallback_events"].append("query_induction_disabled")
        return SemanticCompilationResult(query_contract=QuerySemanticContract(), audit=audit)
    induction_started = time.monotonic()
    try:
        raw_contract = llm_client.chat_json(model=model_name, system_prompt=INDUCTION_SYSTEM_PROMPT, user_prompt=json.dumps(payload, ensure_ascii=False))
        audit["llm_calls"] += 1
        audit["llm_calls_by_stage"]["induction"] += 1
    except Exception as exc:
        audit["latency_seconds_by_stage"]["induction"] = round(time.monotonic() - induction_started, 6)
        audit.update({"parse_failures": 1, "induction_error": type(exc).__name__})
        audit["fallback_events"].append("empty_contract_after_induction_failure")
        return SemanticCompilationResult(query_contract=QuerySemanticContract(), audit=audit)
    audit["latency_seconds_by_stage"]["induction"] = round(time.monotonic() - induction_started, 6)
    contract = _contract(raw_contract, question=question)
    # An empty contract is a compiler failure, not an ordinary zero-field
    # question. Likewise, an explicitly coordinated multi-clause question
    # needs one slot per clause; otherwise Stage 2 can discard evidence for a
    # later field before symbolic reasoning sees it. Give the same closed
    # Top-20 evidence one bounded induction recovery pass. This does not
    # expand retrieval or use a domain vocabulary.
    repair_cfg = dict(cfg.get("repair") or {})
    minimum_slots = _minimum_independent_slot_count(question)
    needs_structural_recovery = bool(
        minimum_slots > 1 and len(contract.requested_slots) < minimum_slots
    )
    if (
        (not contract.requested_slots or needs_structural_recovery)
        and str(question or "").strip()
        and bool(repair_cfg.get("enabled", True))
        and int(repair_cfg.get("max_rounds", 1)) > 0
    ):
        retry_payload = {
            **payload,
            "recovery_instruction": (
                "Reinspect the question only and emit every independently answerable "
                "requested slot in open vocabulary."
            ),
        }
        if needs_structural_recovery:
            retry_payload["recovery_instruction"] = (
                f"The question contains {minimum_slots} coordinated interrogative "
                "clauses. Emit at least one independently answerable requested "
                "slot for each clause, preserving their distinctions; do not merge "
                "a later clause into the first slot."
            )
        retry_started = time.monotonic()
        audit["induction_retry_triggered"] = True
        if needs_structural_recovery:
            audit["induction_retry_reason"] = "coordinated_interrogative_slot_gap"
        audit["prompt_characters_by_stage"]["induction"] += len(
            json.dumps(retry_payload, ensure_ascii=False)
        )
        try:
            retry_raw = llm_client.chat_json(
                model=model_name,
                system_prompt=INDUCTION_SYSTEM_PROMPT,
                user_prompt=json.dumps(retry_payload, ensure_ascii=False),
            )
            audit["llm_calls"] += 1
            audit["llm_calls_by_stage"]["induction"] += 1
            retry_contract = _contract(retry_raw, question=question)
            if retry_contract.requested_slots:
                contract = retry_contract
            else:
                audit["fallback_events"].append("empty_contract_after_induction_retry")
        except Exception as exc:
            audit["parse_failures"] = int(audit.get("parse_failures", 0)) + 1
            audit["fallback_events"].append("induction_retry_failed")
            audit["induction_retry_error"] = type(exc).__name__
        audit["latency_seconds_by_stage"]["induction"] = round(
            audit["latency_seconds_by_stage"]["induction"]
            + (time.monotonic() - retry_started),
            6,
        )
    if needs_structural_recovery and len(contract.requested_slots) < minimum_slots:
        recovered_contract = _append_structural_slots(contract, question)
        if len(recovered_contract.requested_slots) > len(contract.requested_slots):
            contract = recovered_contract
            audit["fallback_events"].append("structural_multi_field_slot_recovery")
    extraction_payload = {"question": question, "query_contract": contract.to_dict(), "evidence": _visible_evidence(evidence)}
    assert_runtime_payload_safe(extraction_payload, context="semantic_compiler_extraction")
    audit["prompt_characters_by_stage"]["extraction"] = len(json.dumps(extraction_payload, ensure_ascii=False))
    extraction_started = time.monotonic()
    if not bool(dict(cfg.get("atom_extraction") or {}).get("enabled", True)):
        audit["fallback_events"].append("atom_extraction_disabled")
        raw_atoms = {}
    else:
        try:
            raw_atoms = llm_client.chat_json(model=model_name, system_prompt=EXTRACTION_SYSTEM_PROMPT, user_prompt=json.dumps(extraction_payload, ensure_ascii=False))
            audit["llm_calls"] += 1
            audit["llm_calls_by_stage"]["extraction"] += 1
        except Exception as exc:
            audit.update({"parse_failures": int(audit.get("parse_failures", 0)) + 1, "extraction_error": type(exc).__name__})
            audit["fallback_events"].append("empty_atoms_after_extraction_failure")
            raw_atoms = {}
    audit["latency_seconds_by_stage"]["extraction"] = round(time.monotonic() - extraction_started, 6)
    candidates = _mapping_list(raw_atoms.get("atoms")) if isinstance(raw_atoms, dict) else []
    valid_slot_ids = {slot.slot_id for slot in contract.requested_slots}
    slot_names = {slot.slot_id: slot.slot_name for slot in contract.requested_slots}
    malformed_initial = sum(
        1 for item in (candidates or [])
        if _atom(item, 0, valid_slot_ids, slot_names) is None
    )
    atoms = [
        atom for i, item in enumerate(candidates or [])
        if (atom := _atom(item, i, valid_slot_ids, slot_names))
    ]
    verified, results = [], []
    for atom in atoms:
        result = verify_grounded_atom(
            atom, evidence, valid_slot_ids, contract.target_entities,
        )
        results.append(asdict(result))
        if result.accepted_for_symbolic_reasoning:
            verified.append(atom)
    before = _coverage(contract, verified)
    resolvability_before = _resolvability(contract, verified)
    repair_atoms: list[GroundedMemoryAtom] = []
    repair_attempted = False
    repair_missing_slots = list(dict.fromkeys([
        *before.missing_slots,
        *resolvability_before.unresolved_slots,
    ]))
    # A slot can be technically covered while still being represented only by
    # an older candidate.  Ask the bounded repair pass to reconcile such slots
    # against the closed Top-20 evidence.  This remains training-free and
    # source-grounded; the symbolic ledger, not the repair model, chooses the
    # latest current value.
    visible_turn_indexes = []
    for row in evidence:
        record = dict((row.metadata or {}).get("structured_record") or {})
        turn_index = record.get("turn_index")
        if isinstance(turn_index, int):
            visible_turn_indexes.append(turn_index)
    latest_visible_turn = max(visible_turn_indexes, default=-1)
    candidate_latest_by_slot: dict[str, int] = {}
    evidence_records = {
        row.memory_id: dict((row.metadata or {}).get("structured_record") or {})
        for row in evidence
    }
    for atom in verified:
        record = evidence_records.get(str(atom.source.get("chunk_id") or ""), {})
        turn_index = record.get("turn_index")
        if isinstance(turn_index, int):
            slot_id = str(atom.slot_id)
            candidate_latest_by_slot[slot_id] = max(
                candidate_latest_by_slot.get(slot_id, -1), turn_index,
            )
    reconciliation_slots = [
        slot.slot_id
        for slot in contract.requested_slots
        if slot.required_for_answer
        and str(slot.temporal_requirement or "unknown").casefold() == "current"
        and slot.slot_id in candidate_latest_by_slot
        and candidate_latest_by_slot[slot.slot_id] < latest_visible_turn
    ]
    repair_slots = list(dict.fromkeys([*repair_missing_slots, *reconciliation_slots]))
    if (
        bool(repair_cfg.get("enabled", True))
        and int(repair_cfg.get("max_rounds", 1)) > 0
        and repair_slots
    ):
        repair_attempted = True
        audit["repair_triggered"] = True
        audit["repair_reason"] = {
            "missing_or_unresolved_slots": list(repair_missing_slots),
            "reconciliation_slots": list(reconciliation_slots),
        }
        repair_payload = {
            "question": question,
            "query_contract": contract.to_dict(),
            "missing_slots": repair_missing_slots,
            "reconciliation_slots": reconciliation_slots,
            "repair_slots": repair_slots,
            "missing_slot_specs": [
                asdict(slot) for slot in contract.requested_slots
                if slot.slot_id in repair_slots
            ],
            "already_extracted_atoms": [a.to_dict() for a in verified],
            "evidence": _visible_evidence(evidence),
        }
        assert_runtime_payload_safe(repair_payload, context="semantic_compiler_repair")
        audit["prompt_characters_by_stage"]["repair"] = len(json.dumps(repair_payload, ensure_ascii=False))
        repair_started = time.monotonic()
        try:
            raw_repair = llm_client.chat_json(model=model_name, system_prompt=REPAIR_SYSTEM_PROMPT, user_prompt=json.dumps(repair_payload, ensure_ascii=False))
            audit["llm_calls"] += 1
            audit["llm_calls_by_stage"]["repair"] += 1
            repair_candidates = (
                _mapping_list(raw_repair.get("atoms"))
                if isinstance(raw_repair, dict) else []
            )
        except Exception as exc:
            audit["parse_failures"] = int(audit.get("parse_failures", 0)) + 1
            audit["repair_error"] = type(exc).__name__
            audit["fallback_events"].append("empty_atoms_after_repair_failure")
            repair_candidates = []
        audit["latency_seconds_by_stage"]["repair"] = round(time.monotonic() - repair_started, 6)
        for i, item in enumerate(repair_candidates or []):
            atom = _atom(item, i, valid_slot_ids, slot_names)
            if atom is None:
                continue
            result = verify_grounded_atom(
                atom, evidence, valid_slot_ids, contract.target_entities,
            )
            results.append(asdict(result))
            if result.accepted_for_symbolic_reasoning:
                repair_atoms.append(atom)
    seen: set[tuple[str, ...]] = set()
    final: list[GroundedMemoryAtom] = []
    for atom in [*verified, *repair_atoms]:
        source = atom.source
        key = (
            atom.slot_id,
            str(atom.value or "").casefold(),
            str(source.get("chunk_id") or ""),
            str(source.get("span") or "").casefold(),
            str(atom.temporal.get("state") or "unknown").casefold(),
            str(atom.lifecycle_semantics.get("type") or "none").casefold(),
            str(atom.authorization_semantics.get("type") or "none").casefold(),
        )
        if key not in seen:
            seen.add(key); final.append(atom)
    after = _coverage(contract, final)
    resolvability_after = _resolvability(contract, final)
    audit.update({
        "initial_atom_count": len(atoms),
        "malformed_initial_atom_count": malformed_initial,
        "grounded_atom_count": len(final),
        "coverage_before_repair": before.to_dict(),
        "coverage_after_repair": after.to_dict(),
        "resolvability_before_repair": resolvability_before.to_dict(),
        "resolvability_after_repair": resolvability_after.to_dict(),
        "latency_seconds_total": round(sum(audit["latency_seconds_by_stage"].values()), 6),
    })
    return SemanticCompilationResult(
        query_contract=contract,
        initial_atoms=[a.to_dict() for a in atoms],
        grounding_results=results,
        verified_atoms=[a.to_dict() for a in verified],
        coverage_before_repair=before.to_dict(),
        repair_triggered=repair_attempted,
        repair_missing_slots=repair_missing_slots,
        repair_atoms=[a.to_dict() for a in repair_atoms],
        coverage_after_repair=after.to_dict(),
        final_grounded_atoms=[a.to_dict() for a in final],
        resolvability_before_repair=resolvability_before.to_dict(),
        resolvability_after_repair=resolvability_after.to_dict(),
        audit=audit,
    )
