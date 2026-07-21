"""LLM-derived, provenance-preserving policy bindings for governed atoms."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.memory.governed_atom import GovernedMemoryAtom


POLICY_SCOPES = {"self", "family", "clinician", "collaborator", "researcher", "external"}


def compile_policy_frames(
    *,
    atoms: list[GovernedMemoryAtom],
    llm_client: LLMClient | None,
    model_name: str,
) -> list[GovernedMemoryAtom]:
    """Attach explicit policy bindings; untouched atoms remain conservative."""
    candidates = _policy_candidate_atoms(atoms)
    if not candidates:
        return atoms
    # Authorization semantics are intentionally delegated to the LLM compiler.
    # A lexical fallback would turn benchmark phrasing into runtime policy.
    if llm_client is None or not llm_client.is_available():
        return atoms
    payload = [{"atom_id": atom.atom_id, "text": atom.text, "slots": atom.slots} for atom in candidates]
    # Policy prose often names a property differently from its observed fact
    # slot.  The compiler may resolve that semantics, but only by selecting a
    # name from this closed evidence inventory.
    governance_source_memory_ids = {
        str((atom.provenance or {}).get("source_memory_id") or "")
        for atom in atoms
        if atom.atom_type in {"policy_atom", "permission_atom"}
        and str((atom.provenance or {}).get("source_memory_id") or "")
    }
    governance_source_message_ids = {
        str(message_id)
        for atom in atoms
        if str((atom.provenance or {}).get("source_role") or "") in {"policy", "mixed"}
        for message_id in list((atom.provenance or {}).get("source_message_ids") or [])
        if str(message_id)
    }
    observed_slot_names = sorted({
        _normalize_slot_name(slot_name)
        for atom in atoms
        if atom.atom_type not in {"policy_atom", "permission_atom"}
        and str((atom.provenance or {}).get("source_memory_id") or "") not in governance_source_memory_ids
        and not (set(str(value) for value in list((atom.provenance or {}).get("source_message_ids") or [])) & governance_source_message_ids)
        for slot_name in dict(atom.slots or {})
        if _normalize_slot_name(slot_name)
    })
    observed_slot_examples = _observed_slot_examples(atoms, governance_source_memory_ids, governance_source_message_ids)
    raw = _request_policy_bindings(
        llm_client=llm_client,
        model_name=model_name,
        payload=payload,
        observed_slot_names=observed_slot_names,
        observed_slot_examples=observed_slot_examples,
    )
    bindings_by_id = _validated_bindings(
        raw=raw,
        candidates=candidates,
        observed_slot_names=observed_slot_names,
        source="llm_policy_frame",
    )
    # A second independent pass checks slot coverage, not the policy decision
    # itself. It may only retain the same effect/scope and add observed slots
    # that are explicitly covered by the same source-grounded policy span.
    audited_raw = _request_policy_coverage_audit(
        llm_client=llm_client,
        model_name=model_name,
        payload=payload,
        observed_slot_names=observed_slot_names,
        observed_slot_examples=observed_slot_examples,
        proposed_bindings=bindings_by_id,
    )
    audited_bindings = _validated_bindings(
        raw=audited_raw,
        candidates=candidates,
        observed_slot_names=observed_slot_names,
        source="llm_policy_coverage_audit",
    )
    for atom_id, audited in audited_bindings.items():
        prior = bindings_by_id.get(atom_id)
        if prior is None:
            continue
        if audited["effect"] != prior["effect"] or audited["scopes"] != prior["scopes"]:
            continue
        if not set(prior["slots"]).issubset(audited["slots"]):
            continue
        bindings_by_id[atom_id] = audited
    enriched: list[GovernedMemoryAtom] = []
    for atom in atoms:
        binding = bindings_by_id.get(atom.atom_id)
        if binding is None:
            enriched.append(atom)
            continue
        provenance = dict(atom.provenance or {})
        provenance["policy_binding"] = binding
        enriched.append(replace(atom, access_scope=list(binding["scopes"]), provenance=provenance))
    return enriched


def _request_policy_bindings(
    *,
    llm_client: LLMClient,
    model_name: str,
    payload: list[dict[str, Any]],
    observed_slot_names: list[str],
    observed_slot_examples: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt=(
                "You extract policy bindings from governed-memory evidence. Do not answer any user question, "
                "do not invent permissions, and return JSON only."
            ),
            user_prompt=(
                "Return {\"bindings\":[{\"atom_id\":string,\"effect\":\"allow|deny|require_permission|none\","
                "\"scopes\":[\"self|family|clinician|collaborator|researcher|external\"],"
                "\"slots\":[exact observed slot names],"
                "\"support_spans\":[exact verbatim evidence spans]}]}. A binding is present only if the "
                "evidence explicitly states an authorization, prohibition, or permission requirement. "
                "For slots, select only exact names from Observed fact slot inventory; do not create a new "
                "schema name or copy an entity, value, or sentence. Every binding must cite one or more exact "
                "spans from its own evidence atom that support the effect, scope, and governed attribute. "
                "When a policy authorizes a category of information, include every observed slot that the policy "
                "semantically covers, rather than a single representative field. Do not bind an atom when support "
                f"spans are unavailable. Observed fact slot inventory: {observed_slot_names}\n"
                f"One source-grounded example per observed slot: {observed_slot_examples}\nEvidence: {payload}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _request_policy_coverage_audit(
    *,
    llm_client: LLMClient,
    model_name: str,
    payload: list[dict[str, Any]],
    observed_slot_names: list[str],
    observed_slot_examples: list[dict[str, str]],
    proposed_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not proposed_bindings:
        return {}
    try:
        raw = llm_client.chat_json(
            model=model_name,
            system_prompt="You verify source-grounded policy-to-evidence coverage. Return JSON only.",
            user_prompt=(
                "Return the same bindings schema. For each proposed binding, preserve atom_id, effect, and scopes. "
                "Select a superset of its slots only when the exact cited policy span explicitly authorizes or "
                "restricts that additional observed field. Do not infer a policy from a role/title. Do not add a "
                "binding, change effect/scope, or use a slot outside the closed inventory.\n"
                f"Observed fact slot inventory: {observed_slot_names}\n"
                f"One source-grounded example per observed slot: {observed_slot_examples}\n"
                f"Proposed bindings: {[{'atom_id': atom_id, **binding} for atom_id, binding in proposed_bindings.items()]}\n"
                f"Evidence: {payload}"
            ),
        )
        return raw if isinstance(raw, dict) else {}
    except (LLMClientUnavailableError, Exception):
        return {}


def _validated_bindings(
    *,
    raw: dict[str, Any],
    candidates: list[GovernedMemoryAtom],
    observed_slot_names: list[str],
    source: str,
) -> dict[str, dict[str, Any]]:
    bindings_by_id: dict[str, dict[str, Any]] = {}
    for binding in list(raw.get("bindings") or []) if isinstance(raw, dict) else []:
        if not isinstance(binding, dict):
            continue
        atom_id = str(binding.get("atom_id") or "")
        effect = str(binding.get("effect") or "none").lower()
        scopes = sorted({str(scope) for scope in list(binding.get("scopes") or []) if str(scope) in POLICY_SCOPES})
        slots = sorted({_normalize_slot_name(slot) for slot in list(binding.get("slots") or [])})
        slots = [slot for slot in slots if slot in observed_slot_names]
        support_spans = _validated_support_spans(
            binding.get("support_spans"),
            next((atom.text for atom in candidates if atom.atom_id == atom_id), ""),
        )
        if atom_id and effect in {"allow", "deny", "require_permission"} and scopes and slots and support_spans:
            bindings_by_id[atom_id] = {
                "effect": effect,
                "scopes": scopes,
                "slots": slots,
                "support_spans": support_spans,
                "source": source,
            }
    return bindings_by_id


def _observed_slot_examples(
    atoms: list[GovernedMemoryAtom], governance_source_memory_ids: set[str], governance_source_message_ids: set[str]
) -> list[dict[str, str]]:
    """Give policy alignment semantic context without opening the slot set."""
    examples: dict[str, dict[str, str]] = {}
    for atom in atoms:
        if atom.atom_type in {"policy_atom", "permission_atom"}:
            continue
        if str((atom.provenance or {}).get("source_memory_id") or "") in governance_source_memory_ids:
            continue
        if set(str(value) for value in list((atom.provenance or {}).get("source_message_ids") or [])) & governance_source_message_ids:
            continue
        for slot_name, value in dict(atom.slots or {}).items():
            normalized = _normalize_slot_name(slot_name)
            if not normalized or normalized in examples:
                continue
            examples[normalized] = {
                "slot_name": normalized,
                "source_text": str(atom.text or ""),
                "slot_value": str(value or ""),
            }
    return [examples[name] for name in sorted(examples)]


def _policy_candidate_atoms(atoms: list[GovernedMemoryAtom]) -> list[GovernedMemoryAtom]:
    """Keep one source-grounded representative per raw memory turn.

    Memory ingestion may label an explicit permission as an event/fact atom.
    The LLM policy compiler, rather than a lexical fallback, decides whether
    that source actually expresses a policy. Prefer an existing policy type
    when one is available; otherwise retain the first source representative.
    """
    by_source: dict[str, GovernedMemoryAtom] = {}
    for atom in atoms:
        source = str((atom.provenance or {}).get("source_memory_id") or atom.atom_id)
        current = by_source.get(source)
        if current is None or (
            atom.atom_type in {"policy_atom", "permission_atom"}
            and current.atom_type not in {"policy_atom", "permission_atom"}
        ):
            by_source[source] = atom
    return list(by_source.values())


def _normalize_slot_name(value: Any) -> str:
    """Accept a schema-shaped slot name without imposing a domain vocabulary."""
    slot = str(value or "").strip().lower()
    if not slot or len(slot) > 80:
        return ""
    normalized = "_".join(slot.replace("-", " ").split())
    if not normalized or not normalized.replace("_", "").isalnum():
        return ""
    return normalized


def _validated_support_spans(value: Any, source_text: str) -> list[str]:
    """Keep only non-empty verbatim citations from the same policy atom."""
    source = str(source_text or "")
    spans: list[str] = []
    for item in list(value or []) if isinstance(value, list) else []:
        span = str(item or "").strip()
        if span and span in source and span not in spans:
            spans.append(span)
    return spans
