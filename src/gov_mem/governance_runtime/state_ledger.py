from __future__ import annotations

from dataclasses import asdict

from gov_mem.data.schema import CurrentStateLedger, EvidenceFrame, EventState, StateSlot


def build_current_state_ledger(frames: list[EvidenceFrame]) -> CurrentStateLedger:
    owner_user = next((frame.owner_user for frame in frames if frame.owner_user), None)
    ledger = CurrentStateLedger(
        owner_user=owner_user,
        active_events={},
        canceled_events={},
        superseded_events={},
        deleted_events={},
        active_slots={},
        canceled_slots={},
        deleted_slots={},
        superseded_slots={},
        trace=[],
    )
    for frame in frames:
        event = _frame_to_event(frame)
        bucket = _bucket_for_frame(frame)
        getattr(ledger, bucket)[event.event_key] = event
        _merge_event_slots(ledger, event, bucket)
        if frame.frame_type in {"cancellation", "update"}:
            link_cancellation_or_update(frame, ledger)
    return ledger


def link_cancellation_or_update(frame: EvidenceFrame, ledger: CurrentStateLedger) -> None:
    if frame.frame_type == "cancellation":
        candidate = _frame_to_event(frame)
        matched = _match_prior_event(candidate, ledger.active_events)
        if matched:
            ledger.canceled_events[matched.event_key] = matched
            ledger.active_events.pop(matched.event_key, None)
            ledger.trace.append(f"canceled event linked: {matched.event_key}")
        else:
            ledger.trace.append(f"cancellation unlinked: {candidate.event_key}")
    elif frame.frame_type == "update":
        candidate = _frame_to_event(frame)
        matched = _match_prior_event(candidate, ledger.active_events)
        if matched:
            ledger.superseded_events[matched.event_key] = matched
            ledger.active_events.pop(matched.event_key, None)
            ledger.active_events[candidate.event_key] = candidate
            ledger.trace.append(f"updated event linked: {matched.event_key} -> {candidate.event_key}")
        else:
            ledger.active_events[candidate.event_key] = candidate
            ledger.trace.append(f"update unlinked: {candidate.event_key}")


def ledger_to_dict(ledger: CurrentStateLedger) -> dict:
    return asdict(ledger)


def _frame_to_event(frame: EvidenceFrame) -> EventState:
    return EventState(
        event_key=_event_key(frame),
        frame_type=frame.frame_type,
        subject_entity=frame.subject_entity,
        lifecycle_status=frame.lifecycle_status,
        slots=dict(frame.slots),
        surface_spans=dict(frame.surface_spans),
        frame_ids=[frame.frame_id],
        memory_ids=[frame.memory_id],
        effective_time=frame.effective_time,
        confidence=frame.confidence,
    )


def _bucket_for_frame(frame: EvidenceFrame) -> str:
    status = (frame.lifecycle_status or "unknown").lower()
    if status == "deleted":
        return "deleted_events"
    if status == "superseded":
        return "superseded_events"
    if status == "canceled" or frame.frame_type == "cancellation":
        return "canceled_events"
    return "active_events"


def _merge_event_slots(ledger: CurrentStateLedger, event: EventState, bucket: str) -> None:
    slot_bucket = {
        "active_events": ledger.active_slots,
        "canceled_events": ledger.canceled_slots,
        "superseded_events": ledger.superseded_slots,
        "deleted_events": ledger.deleted_slots,
    }[bucket]
    for slot_name, slot_value in event.slots.items():
        if not slot_value:
            continue
        key = f"{ledger.owner_user or 'unknown_owner'}::{event.frame_type}::{event.subject_entity or 'general'}::{slot_name}"
        candidate = StateSlot(
            key=key,
            value=str(slot_value),
            frame_ids=list(event.frame_ids),
            memory_ids=list(event.memory_ids),
            lifecycle_status=event.lifecycle_status,
            effective_time=event.effective_time,
            confidence=event.confidence,
        )
        existing = slot_bucket.get(key)
        if existing is None or _prefer_candidate(candidate, existing):
            slot_bucket[key] = candidate
            ledger.trace.append(f"{bucket}:{key} <- {candidate.value}")


def _match_prior_event(candidate: EventState, events: dict[str, EventState]) -> EventState | None:
    if not events:
        return None
    candidate_tokens = _event_signature_tokens(candidate)
    best = None
    best_score = -1
    for event in events.values():
        score = len(candidate_tokens & _event_signature_tokens(event))
        if score > best_score:
            best = event
            best_score = score
    return best if best_score > 0 else None


def _event_signature_tokens(event: EventState) -> set[str]:
    tokens = {event.frame_type.lower(), (event.subject_entity or "").lower()}
    for value in event.slots.values():
        if not value:
            continue
        for chunk in str(value).lower().split():
            tokens.add(chunk.strip(",.;"))
    return {token for token in tokens if token}


def _event_key(frame: EvidenceFrame) -> str:
    owner = frame.owner_user or "unknown_owner"
    frame_type = frame.frame_type or "unknown_frame"
    subject = frame.subject_entity or frame.slots.get("procedure") or frame.slots.get("provider") or "general"
    date = frame.slots.get("date") or "unknown_date"
    time = frame.slots.get("time") or frame.slots.get("arrival_time") or "unknown_time"
    return f"{owner}::{frame_type}::{subject}::{date}::{time}"


def _prefer_candidate(left: StateSlot, right: StateSlot) -> bool:
    left_time = left.effective_time or ""
    right_time = right.effective_time or ""
    if left.lifecycle_status == "active" and right.lifecycle_status != "active":
        return True
    if left.lifecycle_status != "active" and right.lifecycle_status == "active":
        return False
    if left_time != right_time:
        return left_time > right_time
    return left.confidence >= right.confidence
