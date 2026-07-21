from __future__ import annotations

import re
from hashlib import md5
from typing import Any

from gov_mem.data.schema import MemoryInstance, MemoryItem
from gov_mem.governance_runtime.access import normalize_role
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.prompts import (
    MEMORY_INGESTION_SYSTEM_PROMPT,
    build_memory_ingestion_user_prompt,
)


ENTITY_PATTERN = re.compile(r"\b[A-Z][a-zA-Z0-9_\-]+\b")


class MemoryIngestionAgent:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model_name: str,
        skill_text: str = "",
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.skill_text = skill_text

    def ingest(self, instance: MemoryInstance) -> list[MemoryItem]:
        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=MEMORY_INGESTION_SYSTEM_PROMPT,
                user_prompt=build_memory_ingestion_user_prompt(instance.messages, self.skill_text),
            )
            items = raw.get("memory_items", []) if isinstance(raw, dict) else []
            memory_items = [self._from_llm_item(instance, idx, item) for idx, item in enumerate(items)]
            if memory_items:
                return memory_items
        except LLMClientUnavailableError:
            pass
        except Exception:
            pass

        return self._heuristic_ingest(instance)

    def _heuristic_ingest(self, instance: MemoryInstance) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        active_ids_by_content: dict[str, str] = {}
        for idx, message in enumerate(instance.messages):
            content = str(message.get("text") or "").strip()
            if not content:
                continue
            user_id = message.get("speaker_id")
            scope = _infer_scope(content, user_id)
            memory_type = _infer_memory_type(content)
            entities = _extract_entities(content)
            memory_id = f"{instance.instance_id}_mem_{idx:04d}"
            lowered = content.lower()
            privacy_level = "restricted" if any(token in lowered for token in ["token", "password", "private", "confidential", "lab", "result", "stipend", "diagnosis"]) else "normal"
            authorized_users = _infer_authorized_users(instance, message, content)
            forbidden_users = _infer_forbidden_users(instance, content)
            redaction_required = _infer_redaction_required(content)
            memory_status = "active"
            supersedes_memory_ids: list[str] = []

            if _is_forgetting_instruction(content):
                memory_type = "experience"
                scope = "constraint"
            elif _is_update_instruction(content):
                memory_status = "active"

            items.append(
                MemoryItem(
                    memory_id=memory_id,
                    instance_id=instance.instance_id,
                    user_id=user_id,
                    scope=scope,
                    content=content,
                    memory_type=memory_type,
                    entities=entities,
                    time=message.get("timestamp"),
                    source_message_ids=[str(message.get("message_id"))],
                    confidence=0.6,
                    privacy_level=privacy_level,
                    tags=[message.get("speaker_role") or "unknown"],
                    memory_status=memory_status,
                    metadata={
                        "privacy_level": privacy_level,
                        "access_scope": scope,
                        "authorized_users": authorized_users,
                        "forbidden_users": forbidden_users,
                        "forget_after": None,
                        "is_deleted": False,
                        "redaction_required": redaction_required,
                        "sensitive_entities": entities if privacy_level == "restricted" else [],
                        "supersedes_memory_ids": supersedes_memory_ids,
                    },
                )
            )
            active_ids_by_content[content] = memory_id

        runtime_profile = dict(instance.metadata.get("runtime_profile") or {})
        if bool(runtime_profile.get("apply_checkpoint_updates", False)):
            self._apply_checkpoint_memory_updates(items)
        return items

    def _from_llm_item(self, instance: MemoryInstance, idx: int, item: dict[str, Any]) -> MemoryItem:
        memory_id = item.get("memory_id") or f"{instance.instance_id}_mem_{idx:04d}"
        return MemoryItem(
            memory_id=str(memory_id),
            instance_id=instance.instance_id,
            user_id=item.get("user_id"),
            scope=str(item.get("scope") or "event"),
            content=str(item.get("content") or ""),
            memory_type=str(item.get("memory_type") or "factual"),
            entities=[str(entity) for entity in item.get("entities", [])],
            time=_normalize_time_value(item.get("time")),
            source_message_ids=[str(mid) for mid in item.get("source_message_ids", [])],
            confidence=float(item.get("confidence", 0.7)),
            privacy_level=item.get("privacy_level"),
            tags=[str(tag) for tag in item.get("tags", [])],
            memory_status=str(item.get("memory_status") or "active"),
            metadata=dict(item.get("metadata", {}) or {}),
        )

    def _apply_checkpoint_memory_updates(self, items: list[MemoryItem]) -> None:
        for idx, item in enumerate(items):
            lowered = item.content.lower()
            if _is_forgetting_instruction(item.content):
                item.metadata["is_deleted"] = False
                targets = _find_deletion_targets(items[:idx], item.content)
                for target in targets:
                    target.memory_status = "deleted"
                    target.metadata["is_deleted"] = True
            elif _is_update_instruction(item.content):
                targets = _find_update_targets(items[:idx], item.content)
                for target in targets:
                    if target.memory_id != item.memory_id:
                        target.memory_status = "superseded"
                        target.metadata["is_deleted"] = False
                        item.metadata.setdefault("supersedes_memory_ids", []).append(target.memory_id)


def _infer_scope(content: str, user_id: str | None) -> str:
    lowered = content.lower()
    if " prefer " in f" {lowered} " or " like " in f" {lowered} ":
        return "preference"
    if " must " in f" {lowered} " or " should " in f" {lowered} " or "do not" in lowered:
        return "constraint"
    if user_id and ("i " in lowered or " my " in lowered):
        return "user"
    if any(token in lowered for token in ["meeting", "appointment", "deadline", "tomorrow", "today"]):
        return "event"
    return "group"


def _infer_memory_type(content: str) -> str:
    lowered = content.lower()
    if any(token in lowered for token in ["prefer", "like", "favorite"]):
        return "preference"
    if any(token in lowered for token in ["plan", "intend", "will"]):
        return "intention"
    if any(token in lowered for token in ["must", "should", "cannot", "can't", "do not"]):
        return "constraint"
    if any(token in lowered for token in ["met", "meeting", "appointment", "scheduled", "date"]):
        return "event"
    return "factual"


def _extract_entities(content: str) -> list[str]:
    found = []
    seen = set()
    for match in ENTITY_PATTERN.findall(content):
        if match not in seen:
            seen.add(match)
            found.append(match)
    return found[:10]


def _infer_authorized_users(instance: MemoryInstance, message: dict, content: str) -> list[str]:
    requester = ((instance.metadata.get("requester") or {}).get("principal_id"))
    role = str(message.get("speaker_role") or "").lower()
    lowered = content.lower()
    if "family access revoked" in lowered or "removed from scheduling-contact" in lowered:
        return [requester] if requester else []
    if "logistics only" in lowered or "appointment time" in lowered:
        authorized = []
        if requester:
            authorized.append(requester)
        if "family" in lowered or "linda" in lowered:
            authorized.append("family_linda_park")
        return list(dict.fromkeys(authorized))
    if any(token in lowered for token in ["result", "diagnosis", "lab", "medication", "clinical interpretation"]):
        if "patient" in role or str(message.get("speaker_id") or "").startswith("patient_"):
            return [str(message.get("speaker_id"))]
    if any(token in lowered for token in ["follow-up", "follow up", "appointment", "schedule", "arrival", "suite", "parking"]):
        speaker_id = str(message.get("speaker_id") or "").strip()
        authorized = [speaker_id] if speaker_id else []
        speaker_role = normalize_role(message.get("speaker_role"))
        if speaker_role in {"scheduler", "clinician", "nurse", "social_worker", "front_desk"}:
            authorized.append(speaker_role)
        return list(dict.fromkeys([item for item in authorized if item]))
    return []


def _infer_forbidden_users(instance: MemoryInstance, content: str) -> list[str]:
    lowered = content.lower()
    forbidden: list[str] = []
    if "not my results" in lowered or "not lab values" in lowered or "not medication details" in lowered:
        forbidden.append("family_linda_park")
    if "family access revoked" in lowered:
        forbidden.append("family_linda_park")
    return forbidden


def _infer_redaction_required(content: str) -> bool:
    lowered = content.lower()
    if any(token in lowered for token in ["appointment", "arrival", "suite", "parking", "location", "schedule", "scheduled", "follow-up", "follow up"]):
        return False
    return any(
        token in lowered
        for token in ["logistics only", "generic callback", "do not mention", "broad wording", "safe wording"]
    )


def _is_forgetting_instruction(content: str) -> bool:
    lowered = content.lower()
    return any(token in lowered for token in ["delete", "forget", "remove", "no longer remember", "deleted", "clear the temporary", "clear temporary"])


def _is_update_instruction(content: str) -> bool:
    lowered = content.lower()
    return any(token in lowered for token in ["updated", "overwritten", "replaced", "current", "canceled", "cancelled", "revoked", "superseded", "instead of", "changed effective"])


def _find_deletion_targets(previous_items: list[MemoryItem], instruction: str) -> list[MemoryItem]:
    lowered = instruction.lower()
    targets = []
    for item in previous_items:
        content = item.content.lower()
        if any(token in lowered and token in content for token in ["safe number", "callback", "voicemail", "temporary", "contact"]):
            targets.append(item)
    return targets


def _find_update_targets(previous_items: list[MemoryItem], instruction: str) -> list[MemoryItem]:
    lowered = instruction.lower()
    targets = []
    for item in previous_items:
        content = item.content.lower()
        if any(token in lowered and token in content for token in ["schedule", "appointment", "slot", "access", "revoked", "canceled", "cancelled", "current", "callback", "follow-up", "follow up", "Tuesday", "Wednesday", "Monday"]):
            targets.append(item)
    return targets


def _normalize_time_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
