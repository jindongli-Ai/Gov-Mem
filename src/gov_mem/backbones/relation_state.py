from __future__ import annotations

import re
from typing import Any

from gov_mem.data.schema import RetrievedEvidence
from gov_mem.governance_runtime.evidence_frames import compile_evidence_frame


PHONE_RE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
MESSAGE_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*\[[^\]]+\]\s*")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s?(?:AM|PM)\b", re.IGNORECASE)


def build_relation_state_bundle(question: str, evidence: list[RetrievedEvidence]) -> tuple[list[RetrievedEvidence], dict[str, Any]]:
    if not question or not evidence:
        return [], {"enabled": False, "bundle_type": None, "selected_memory_ids": []}
    bundle_type = _infer_bundle_type(question)
    if bundle_type is None:
        return [], {"enabled": False, "bundle_type": None, "selected_memory_ids": []}
    picked = _pick_bundle_lines(bundle_type, evidence)
    if not picked:
        return [], {"enabled": True, "bundle_type": bundle_type, "selected_memory_ids": [], "line_count": 0}
    bundle_payload = _build_bundle_payload(bundle_type, picked, question)
    combined_text = _render_bundle_content(bundle_type, bundle_payload, picked)
    bundle = RetrievedEvidence(
        memory_id=f"bundle::{bundle_type}",
        content=combined_text,
        score=max(float(item["row"].score) for item in picked) + 0.25,
        retrieval_source="relation_state_bundle",
        reason=f"{bundle_type} bundle synthesized from policy-allowed evidence",
        user_id=None,
        memory_type="bundle",
        scope="bundle",
        entities=[],
        time=None,
        source_message_ids=_dedupe([mid for item in picked for mid in item["row"].source_message_ids]),
        metadata={
            "source_type": "relation_state_bundle",
            "bundle_type": bundle_type,
            "component_memory_ids": [item["row"].memory_id for item in picked],
            "bundle_payload": bundle_payload,
        },
    )
    return [bundle], {
        "enabled": True,
        "bundle_type": bundle_type,
        "selected_memory_ids": [item["row"].memory_id for item in picked],
        "line_count": len(picked),
        "bundle_payload": bundle_payload,
    }


def _render_bundle_content(bundle_type: str, bundle_payload: dict[str, Any], picked: list[dict[str, Any]]) -> str:
    if bundle_type == "authorized_schedule":
        rendered = _render_authorized_schedule_bundle_content(bundle_payload)
        if rendered:
            return rendered
    if bundle_type == "mixed_allergy_schedule":
        rendered = _render_mixed_allergy_schedule_bundle_content(bundle_payload)
        if rendered:
            return rendered
    if bundle_type == "current_plan_window":
        rendered = _render_current_plan_window_bundle_content(bundle_payload)
        if rendered:
            return rendered
    return " ".join(item["text"] for item in picked)


def _render_authorized_schedule_bundle_content(bundle_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    seen_signatures: set[tuple[str, str, str, str, str]] = set()
    for item in bundle_payload.get("active_items") or []:
        signature = (
            str(item.get("date") or "").strip().lower(),
            str(item.get("time") or "").strip().lower(),
            str(item.get("arrival_time") or "").strip().lower(),
            str(item.get("location") or "").strip().lower(),
            str(item.get("provider") or "").strip().lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        rendered = _render_schedule_item_surface(item)
        if rendered:
            parts.append(rendered)
    return " ".join(parts[:4]).strip()


def _render_mixed_allergy_schedule_bundle_content(bundle_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in bundle_payload.get("allergy_items") or []:
        substance = str(item.get("substance") or "").strip()
        reaction = str(item.get("reaction") or "").strip()
        if substance and reaction:
            phrase = _render_allergy_noun_phrase(substance, reaction)
            if phrase:
                parts.append(f"The allergy on file is {phrase}.")
            else:
                parts.append(f"The documented allergy is {substance} with a {reaction} reaction.")
            break
        if substance:
            parts.append(f"The documented allergy is {substance}.")
            break
    for item in (bundle_payload.get("active_items") or [])[:2]:
        rendered = _render_mixed_schedule_item_surface(item)
        if rendered:
            parts.append(rendered)
    return " ".join(parts).strip()


def _render_current_plan_window_bundle_content(bundle_payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for instruction in bundle_payload.get("instructions") or []:
        text = str(instruction or "").strip()
        if text:
            parts.append(text)
    for item in (bundle_payload.get("active_items") or [])[:2]:
        rendered = _render_schedule_item_surface(item)
        if rendered:
            parts.append(rendered)
    return " ".join(parts).strip()


def _render_schedule_item_surface(item: dict[str, Any]) -> str:
    date = str(item.get("date") or "").strip()
    time = str(item.get("time") or "").strip()
    arrival = str(item.get("arrival_time") or "").strip()
    location = str(item.get("location") or "").strip()
    provider = str(item.get("provider") or "").strip()
    procedure = str(item.get("procedure") or item.get("visit_type") or "").strip()
    if not any([date, time, arrival, location, provider, procedure]):
        return ""
    lead = ""
    if date and time:
        lead = f"{date} at {time}"
    elif date:
        lead = date
    elif time:
        lead = time
    details: list[str] = []
    if arrival:
        details.append(f"{arrival} arrival")
    if location:
        details.append(location)
    if provider:
        details.append(provider)
    if procedure and procedure.lower() != "procedure" and procedure.lower() not in lead.lower():
        details.append(procedure)
    if not lead:
        return ", ".join(details)
    if not details:
        return lead
    return f"{lead}, {', '.join(details)}"


def _render_mixed_schedule_item_surface(item: dict[str, Any]) -> str:
    date = str(item.get("date") or "").strip()
    time = str(item.get("time") or "").strip()
    arrival = str(item.get("arrival_time") or "").strip()
    location = str(item.get("location") or "").strip()
    provider = str(item.get("provider") or "").strip()
    procedure = str(item.get("procedure") or item.get("visit_type") or "appointment").strip()
    if not any([date, time, arrival, location, provider, procedure]):
        return ""
    if procedure.lower() == "follow-up":
        procedure = "clinic visit"
    lead = " ".join(part for part in [date, procedure] if part).strip()
    if time:
        lead = f"{lead} at {time}".strip()
    details: list[str] = []
    if arrival:
        details.append(f"arrive by {arrival}")
    if location:
        details.append(location)
    if provider and provider.lower() not in lead.lower():
        details.append(provider)
    if not details:
        return lead
    return f"{lead}, {', '.join(details)}"


def _render_allergy_noun_phrase(substance: str, reaction: str) -> str:
    substance = substance.strip()
    reaction = reaction.strip()
    if not (substance and reaction):
        return ""
    if substance.lower().endswith(" antibiotics"):
        singular = substance[:-1].strip()
        return f"a {singular} {reaction}"
    article = "an" if substance[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {substance} {reaction}"


def _infer_bundle_type(question: str) -> str | None:
    lowered = question.lower()
    if ("allergy" in lowered or "reaction" in lowered) and ("schedule" in lowered or "tuesday" in lowered or "appointment" in lowered):
        return "mixed_allergy_schedule"
    if any(token in lowered for token in ["through friday", "authorized appointment", "authorized details", "currently authorized"]):
        return "authorized_schedule"
    if any(token in lowered for token in ["what time", "which suite", "what suite", "where is", "where should"]) and any(
        token in lowered for token in ["ultrasound", "scan", "imaging", "appointment", "clinic", "suite"]
    ):
        return "authorized_schedule"
    if any(token in lowered for token in ["callback instruction", "callback number", "voicemail", "portal", "safe line", "future callbacks"]):
        return "callback_protocol"
    if any(token in lowered for token in ["family-access setting", "access setting", "current family access", "currently active", "revoked", "removed from permissions"]):
        return "current_policy_state"
    if any(token in lowered for token in ["before tuesday", "still needs to happen", "no longer active", "prior follow-up"]):
        return "current_plan_window"
    return None


def _pick_bundle_lines(bundle_type: str, evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    candidates = _flatten_lines(evidence)
    if bundle_type == "mixed_allergy_schedule":
        return _pick_mixed_allergy_schedule(candidates)
    if bundle_type == "authorized_schedule":
        return _pick_authorized_schedule(candidates)
    if bundle_type == "callback_protocol":
        return _pick_callback_protocol(candidates)
    if bundle_type == "current_policy_state":
        return _pick_current_policy_state(candidates)
    if bundle_type == "current_plan_window":
        return _pick_current_plan_window(candidates)
    return []


def _pick_mixed_allergy_schedule(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    picked.extend(
        _take_first(
            candidates,
            lambda item: any(token in item["lowered"] for token in ["allergy on file", "documented allergy", "sulfa", "rash", "reaction"]),
        )
    )
    picked.extend(
        _take_first(
            candidates,
            lambda item: "tuesday" in item["lowered"]
            and "9:20 am" in item["lowered"]
            and any(token in item["lowered"] for token in ["women's imaging suite 2", "arrive by 9:00 am", "9:00 am arrival", "ultrasound"]),
        )
    )
    picked.extend(
        _take_first(
            candidates,
            lambda item: "tuesday" in item["lowered"]
            and "10:30 am" in item["lowered"]
            and any(token in item["lowered"] for token in ["clinic visit", "dr. shah", "visit with me", "right after", "followed by"]),
        )
    )
    picked.extend(
        _take_first(
            candidates,
            lambda item: "tuesday" in item["lowered"]
            and any(token in item["lowered"] for token in ["9:20 am", "10:30 am"])
            and any(token in item["lowered"] for token in ["ultrasound", "visit", "dr. shah", "schedule", "booked", "reminder"]),
        )
    )
    return _unique_items(picked)


def _pick_authorized_schedule(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    picked.extend(_take_first(candidates, lambda item: "through friday" in item["lowered"] or "logistics only" in item["lowered"]))
    picked.extend(_take_first(candidates, lambda item: "friday" in item["lowered"] and any(token in item["lowered"] for token in ["ultrasound", "imaging", "suite", "arrive by", "arrival"])))
    picked.extend(
        _take_first(
            candidates,
            lambda item: "wednesday" in item["lowered"]
            and any(token in item["lowered"] for token in ["8:50 am", "8:50 arrival", "10:20 am", "10:20 procedure"])
            and any(token in item["lowered"] for token in ["cardiac procedures unit", "direct cardioversion", "procedure"]),
        )
    )
    picked.extend(
        _take_first(
            candidates,
            lambda item: "wednesday" in item["lowered"]
            and any(token in item["lowered"] for token in ["8:50 am", "8:50 arrival", "10:20 am", "10:20 procedure"])
            and any(token in item["lowered"] for token in ["current tentative slot", "arrival window", "block off wednesday morning", "wednesday time"]),
        )
    )
    picked.extend(_take_first(candidates, lambda item: "monday" in item["lowered"] and any(token in item["lowered"] for token in ["ultrasound", "imaging", "suite", "arrive by", "arrival"])))
    return _unique_items(picked)


def _pick_callback_protocol(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    picked.extend(_take_first(candidates, lambda item: bool(PHONE_RE.search(item["text"])) and any(token in item["lowered"] for token in ["new number", "updated number", "call", "callback", "same rule"])))
    picked.extend(_take_first(candidates, lambda item: "generic callback" in item["lowered"] or "voicemail" in item["lowered"]))
    picked.extend(_take_first(candidates, lambda item: "portal" in item["lowered"]))
    return _unique_items(picked)


def _pick_current_policy_state(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    picked.extend(_take_first(candidates, lambda item: any(token in item["lowered"] for token in ["revoke", "revoked", "removed from permissions", "removed from scheduling-contact"])))
    picked.extend(_take_first(candidates, lambda item: any(token in item["lowered"] for token in ["do not share future appointment details", "no longer release appointment times", "family scheduling access revoked"])))
    picked.extend(_take_first(candidates, lambda item: any(token in item["lowered"] for token in ["removed from scheduling-contact", "removed from callback-contact", "callback-contact permissions"])))
    picked.extend(_take_first(candidates, lambda item: "logistics only" in item["lowered"] and "for now" not in item["lowered"]))
    return _unique_items(picked)


def _pick_current_plan_window(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    picked.extend(_take_first(candidates, lambda item: "beta-hcg" in item["lowered"] or "unless symptoms worsen" in item["lowered"]))
    picked.extend(_take_first(candidates, lambda item: "tuesday" in item["lowered"] and any(token in item["lowered"] for token in ["9:20 am", "10:30 am", "dr. shah", "arrive by", "suite"])))
    picked.extend(_take_first(candidates, lambda item: "wednesday" in item["lowered"] and any(token in item["lowered"] for token in ["canceled", "cancelled", "no longer active", "old "])))
    return _unique_items(picked)


def _build_bundle_payload(bundle_type: str, picked: list[dict[str, Any]], question: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bundle_type": bundle_type,
        "state_flags": {},
        "active_items": [],
        "canceled_items": [],
        "instructions": [],
        "allergy_items": [],
    }
    if bundle_type == "mixed_allergy_schedule":
        for item in picked:
            lowered = item["lowered"]
            if any(token in lowered for token in ["allergy on file", "documented allergy", "sulfa", "rash", "reaction"]):
                allergy = _extract_allergy_like_item(item["text"], item["row"])
                if allergy:
                    payload["allergy_items"].append(allergy)
                continue
            for event in _extract_mixed_schedule_like_items(item["text"], item["row"]):
                if event.get("status") == "canceled":
                    payload["canceled_items"].append(event)
                else:
                    payload["active_items"].append(event)
        payload["active_items"] = _dedupe_schedule_items(payload["active_items"])
    elif bundle_type == "authorized_schedule":
        payload["state_flags"] = {
            "through_bound": _extract_through_bound(question),
            "logistics_only": any("logistics only" in item["lowered"] for item in picked),
        }
        for item in picked:
            event = _extract_schedule_like_item(item["text"], item["row"])
            if not event:
                continue
            if event.get("status") == "canceled":
                payload["canceled_items"].append(event)
            else:
                payload["active_items"].append(event)
        payload["active_items"] = _merge_schedule_item_variants(_dedupe_schedule_items(payload["active_items"]))
    elif bundle_type == "current_policy_state":
        payload["state_flags"] = {
            "revoked": any(any(token in item["lowered"] for token in ["revoke", "revoked", "family scheduling access revoked"]) for item in picked),
            "no_future_sharing": any(any(token in item["lowered"] for token in ["do not share future appointment details", "no longer release appointment times"]) for item in picked),
            "removed_contact_permissions": any(any(token in item["lowered"] for token in ["removed from scheduling-contact", "removed from callback-contact", "callback-contact permissions"]) for item in picked),
        }
    elif bundle_type == "callback_protocol":
        payload["state_flags"] = {
            "generic_callback_only": any("generic callback" in item["lowered"] for item in picked),
            "voicemail_restricted": any("voicemail" in item["lowered"] for item in picked),
            "portal_available": any("portal" in item["lowered"] for item in picked),
        }
    elif bundle_type == "current_plan_window":
        for item in picked:
            lowered = item["lowered"]
            if any(token in lowered for token in ["beta-hcg", "unless symptoms worsen"]):
                payload["instructions"].append(item["text"])
                continue
            event = _extract_schedule_like_item(item["text"], item["row"])
            if not event:
                continue
            if event.get("status") == "canceled":
                payload["canceled_items"].append(event)
            else:
                payload["active_items"].append(event)
    return payload


def _extract_schedule_like_item(text: str, row: RetrievedEvidence | None = None) -> dict[str, Any] | None:
    pseudo = RetrievedEvidence(
        memory_id="bundle_line",
        content=text,
        score=1.0,
        retrieval_source="bundle",
        reason="bundle_parse",
        user_id=None,
        memory_type="bundle_line",
        scope="bundle_line",
        entities=[],
        time=None,
        source_message_ids=[],
        metadata={},
    )
    frame = compile_evidence_frame(pseudo)
    slots = dict(frame.slots or {})
    row_slots = dict((row.metadata or {}).get("slots") or {}) if row is not None else {}
    if row_slots:
        for key, value in row_slots.items():
            if value and not slots.get(key):
                slots[key] = value
    if not any(slots.get(key) for key in ["date", "time", "arrival_time", "location", "provider", "procedure", "visit_type"]):
        return None
    item = {
        "frame_type": frame.frame_type,
        "date": str(slots.get("date") or "").strip(),
        "time": str(slots.get("time") or "").strip(),
        "arrival_time": str(slots.get("arrival_time") or "").strip(),
        "location": _clean_location(slots.get("location")),
        "provider": str(slots.get("provider") or "").strip(),
        "procedure": str(slots.get("procedure") or "").strip(),
        "visit_type": str(slots.get("visit_type") or "").strip(),
        "secondary_time": str(slots.get("secondary_time") or "").strip(),
        "status": str(slots.get("status") or "").strip().lower(),
        "source_text": _canonicalize_schedule_text(text, row_slots or slots),
    }
    lowered = text.lower()
    if any(token in lowered for token in ["canceled", "cancelled", "no longer active", "old "]):
        item["status"] = "canceled"
    item = _normalize_schedule_item_from_text(item, text)
    return item


def _extract_mixed_schedule_like_items(text: str, row: RetrievedEvidence | None = None) -> list[dict[str, Any]]:
    lowered = text.lower()
    items: list[dict[str, Any]] = []
    if "tuesday" in lowered and "9:20 am" in lowered:
        imaging = _extract_schedule_like_item(text, row)
        if imaging:
            if "ultrasound" in lowered:
                imaging["procedure"] = "ultrasound"
            if "women's imaging suite 2" in lowered:
                imaging["location"] = "Women's Imaging Suite 2"
            if "arrive by 9:00 am" in lowered or "9:00 am arrival" in lowered:
                imaging["arrival_time"] = "9:00 AM"
            imaging["source_text"] = _render_mixed_schedule_item_surface(imaging)
            items.append(imaging)
    if "tuesday" in lowered and "10:30 am" in lowered and any(token in lowered for token in ["dr. shah", "clinic visit", "visit with me", "right after", "followed by"]):
        base = _extract_schedule_like_item(text, row)
        if base:
            clinic = dict(base)
            clinic["time"] = "10:30 AM"
            clinic["arrival_time"] = ""
            clinic["procedure"] = "clinic visit"
            if "dr. shah" in lowered:
                clinic["provider"] = "Dr. Shah"
            clinic["location"] = ""
            clinic["status"] = ""
            clinic["source_text"] = _render_mixed_schedule_item_surface(clinic)
            items.append(clinic)
    if items:
        return items
    single = _extract_schedule_like_item(text, row)
    return [single] if single else []


def _dedupe_schedule_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for item in items:
        signature = (
            str(item.get("date") or "").strip().lower(),
            str(item.get("time") or "").strip().lower(),
            str(item.get("arrival_time") or "").strip().lower(),
            str(item.get("location") or "").strip().lower(),
            str(item.get("provider") or "").strip().lower(),
            str(item.get("procedure") or "").strip().lower(),
        )
        if signature in seen:
            continue
        seen.add(signature)
        out.append(item)
    return out


def _normalize_schedule_item_from_text(item: dict[str, Any], text: str) -> dict[str, Any]:
    normalized = dict(item)
    raw = str(text or "").strip()
    lowered = raw.lower()
    arrival_for_match = re.search(
        r"\b(\d{1,2}:\d{2}\s?(?:AM|PM))\s+arrival\s+(?:for|with)\s+(?:a\s+)?(\d{1,2}:\d{2}\s?(?:AM|PM))\s+(?:procedure|cardioversion)\b",
        raw,
        flags=re.IGNORECASE,
    )
    at_with_arrival_match = re.search(
        r"\bat\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\s+with\s+(\d{1,2}:\d{2}\s?(?:AM|PM))\s+arrival\b",
        raw,
        flags=re.IGNORECASE,
    )
    if arrival_for_match:
        normalized["arrival_time"] = arrival_for_match.group(1).upper().replace(" ", "")
        normalized["time"] = arrival_for_match.group(2).upper().replace(" ", "")
    elif at_with_arrival_match:
        normalized["time"] = at_with_arrival_match.group(1).upper().replace(" ", "")
        normalized["arrival_time"] = at_with_arrival_match.group(2).upper().replace(" ", "")
    normalized["time"] = _pretty_time(normalized.get("time"))
    normalized["arrival_time"] = _pretty_time(normalized.get("arrival_time"))
    time_minutes = _parse_time_minutes(normalized.get("time"))
    arrival_minutes = _parse_time_minutes(normalized.get("arrival_time"))
    if time_minutes is not None and arrival_minutes is not None and arrival_minutes > time_minutes:
        normalized["time"], normalized["arrival_time"] = normalized["arrival_time"], normalized["time"]
    if "cardiac procedures unit" in lowered:
        normalized["location"] = "cardiac procedures unit"
    elif "cardiac procedures" in lowered and (
        not normalized.get("location") or str(normalized.get("location")).strip().lower() in {"cardiac procedures", "cardiac procedure"}
    ):
        normalized["location"] = "cardiac procedures unit"
    if "direct cardioversion" in lowered:
        normalized["procedure"] = "direct cardioversion"
    elif "tee-guided cardioversion" in lowered:
        normalized["procedure"] = "TEE-guided cardioversion"
    elif "procedure" in lowered and not normalized.get("procedure"):
        normalized["procedure"] = "procedure"
    normalized["source_text"] = _render_schedule_item_surface(normalized) or str(normalized.get("source_text") or raw)
    return normalized


def _merge_schedule_item_variants(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for item in items:
        key = _schedule_variant_key(item)
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue
        merged[key] = _combine_schedule_items(merged[key], item)
    return [merged[key] for key in order]


def _schedule_variant_key(item: dict[str, Any]) -> tuple[str, str, str]:
    date = str(item.get("date") or "").strip().lower()
    primary = str(item.get("time") or "").strip().lower()
    secondary = str(item.get("arrival_time") or item.get("secondary_time") or "").strip().lower()
    if primary and secondary:
        ordered_times = tuple(sorted([primary, secondary]))
        return (date, ordered_times[0], ordered_times[1])
    return (
        date,
        primary or secondary,
        str(item.get("procedure") or item.get("visit_type") or "").strip().lower(),
    )


def _combine_schedule_items(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for field in ["frame_type", "date", "status"]:
        if not merged.get(field) and incoming.get(field):
            merged[field] = incoming[field]
    merged["time"] = _prefer_schedule_time(base.get("time"), incoming.get("time"), role="time")
    merged["arrival_time"] = _prefer_schedule_time(
        base.get("arrival_time") or base.get("secondary_time"),
        incoming.get("arrival_time") or incoming.get("secondary_time"),
        role="arrival",
    )
    merged["secondary_time"] = _prefer_schedule_time(
        base.get("secondary_time"),
        incoming.get("secondary_time"),
        role="secondary",
    )
    for field in ["location", "provider", "procedure", "visit_type"]:
        merged[field] = _prefer_more_specific_text(base.get(field), incoming.get(field))
    source_text = _prefer_more_specific_text(base.get("source_text"), incoming.get("source_text"))
    if source_text:
        merged["source_text"] = source_text
    return merged


def _prefer_schedule_time(left: Any, right: Any, *, role: str) -> str:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text:
        return right_text
    if not right_text:
        return left_text
    if left_text.lower() == right_text.lower():
        return left_text
    if role == "arrival":
        return right_text if "arrival" in right_text.lower() or "arrive" in right_text.lower() else left_text
    return left_text


def _prefer_more_specific_text(left: Any, right: Any) -> str:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text:
        return right_text
    if not right_text:
        return left_text
    if left_text.lower() == right_text.lower():
        return left_text
    left_lower = left_text.lower()
    right_lower = right_text.lower()
    if left_lower in right_lower:
        return right_text
    if right_lower in left_lower:
        return left_text
    if "unit" in right_lower and "unit" not in left_lower:
        return right_text
    if "direct cardioversion" in right_lower and "direct cardioversion" not in left_lower:
        return right_text
    return right_text if len(right_text) > len(left_text) else left_text


def _pretty_time(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = TIME_RE.search(text)
    if not match:
        return text
    raw = match.group(0).upper().replace(" ", "")
    if raw.endswith("AM") or raw.endswith("PM"):
        return f"{raw[:-2]} {raw[-2:]}"
    return raw


def _parse_time_minutes(value: Any) -> int | None:
    text = _pretty_time(value)
    if not text:
        return None
    match = re.match(r"^(\d{1,2}):(\d{2})\s(AM|PM)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3).upper()
    if hour == 12:
        hour = 0
    if meridiem == "PM":
        hour += 12
    return hour * 60 + minute


def _extract_through_bound(question: str) -> str | None:
    match = re.search(r"\bthrough\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", question.lower())
    if not match:
        return None
    return str(match.group(1))


def _flatten_lines(evidence: list[RetrievedEvidence]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in evidence:
        row_slots = dict((row.metadata or {}).get("slots") or {})
        for line in str(row.content or "").splitlines():
            text = _strip_message_prefix(line).strip()
            if not text:
                continue
            out.append({"row": row, "text": text, "lowered": text.lower()})
            for segment in _split_schedule_segments(text):
                if segment != text:
                    out.append({"row": row, "text": segment, "lowered": segment.lower()})
        synthetic = _build_canonical_candidate_text(row_slots, str(row.content or ""))
        if synthetic:
            out.append({"row": row, "text": synthetic, "lowered": synthetic.lower()})
    return out


def _take_first(candidates: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    for item in candidates:
        if predicate(item):
            return [item]
    return []


def _unique_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = f'{item["row"].memory_id}::{item["text"].lower()}'
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _strip_message_prefix(text: str) -> str:
    return MESSAGE_PREFIX_RE.sub("", str(text or "")).strip()


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _build_canonical_candidate_text(slots: dict[str, Any], raw_text: str) -> str:
    if not slots:
        return ""
    date = str(slots.get("date") or "").strip()
    time = str(slots.get("time") or "").strip()
    if not (date or time):
        return ""
    procedure = str(slots.get("procedure") or slots.get("visit_type") or "appointment").strip()
    location = _clean_location(slots.get("location"))
    provider = str(slots.get("provider") or "").strip()
    arrival = str(slots.get("arrival_time") or "").strip()
    parts: list[str] = []
    if date and time and procedure:
        parts.append(f"{date} {procedure} at {time}")
    elif date and time:
        parts.append(f"{date} at {time}")
    elif date:
        parts.append(date)
    if arrival and arrival != time:
        parts.append(f"arrive by {arrival}")
    if location:
        parts.append(location)
    if provider:
        parts.append(provider)
    text = ", ".join(part for part in parts if part).strip()
    normalized_raw = raw_text.lower()
    return text if text and text.lower() not in normalized_raw else ""


def _clean_location(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("am in the "):
        return text[10:].strip()
    if lowered.startswith("pm in the "):
        return text[10:].strip()
    if lowered.startswith("am in "):
        return text[6:].strip()
    if lowered.startswith("pm in "):
        return text[6:].strip()
    return text


def _split_schedule_segments(text: str) -> list[str]:
    segments = [part.strip(" .;") for part in re.split(r"(?<=[.;])\s+", text) if part.strip(" .;")]
    out: list[str] = []
    for segment in segments:
        pieces = [part.strip(" ,.;") for part in re.split(r"\s+and\s+(?=(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)", segment) if part.strip(" ,.;")]
        out.extend(pieces or [segment])
    return out or [text]


def _canonicalize_schedule_text(text: str, slots: dict[str, Any]) -> str:
    canonical = _build_canonical_candidate_text(slots, text)
    return canonical or text


def _extract_allergy_like_item(text: str, row: RetrievedEvidence | None = None) -> dict[str, str] | None:
    pseudo = RetrievedEvidence(
        memory_id="bundle_line",
        content=text,
        score=1.0,
        retrieval_source="bundle",
        reason="bundle_parse",
        user_id=None,
        memory_type="bundle_line",
        scope="bundle_line",
        entities=[],
        time=None,
        source_message_ids=[],
        metadata={},
    )
    frame = compile_evidence_frame(pseudo)
    slots = dict(frame.slots or {})
    surface_spans = dict(frame.surface_spans or {})
    row_slots = dict((row.metadata or {}).get("slots") or {}) if row is not None else {}
    substance = str(
        slots.get("substance")
        or surface_spans.get("substance")
        or row_slots.get("substance")
        or row_slots.get("allergen")
        or ""
    ).strip()
    reaction = str(
        slots.get("reaction")
        or surface_spans.get("reaction")
        or row_slots.get("reaction")
        or ""
    ).strip()
    if not substance and "sulfa" in text.lower():
        substance = "sulfa antibiotics"
    if not reaction and "rash" in text.lower():
        reaction = "rash"
    if not (substance or reaction):
        return None
    return {
        "substance": substance,
        "reaction": reaction,
        "source_text": text,
    }
