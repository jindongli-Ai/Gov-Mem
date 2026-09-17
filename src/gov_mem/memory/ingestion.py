from __future__ import annotations

import re
from hashlib import md5
from typing import Any

from gov_mem.data.schema import MemoryInstance, MemoryItem
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
        """Schema-only emergency fallback when the ingestion LLM is unavailable.

        The fallback deliberately does not classify content with domain words
        or trigger tables. A live paper-facing run uses the LLM extractor;
        this path preserves only message identity, provenance, and neutral
        container metadata so an API outage cannot silently reintroduce a
        benchmark-specific lexicon.
        """
        items: list[MemoryItem] = []
        for idx, message in enumerate(instance.messages):
            content = str(message.get("text") or "").strip()
            if not content:
                continue
            user_id = message.get("speaker_id")
            scope = "observable"
            memory_type = "factual"
            entities = _extract_entities(content)
            memory_id = f"{instance.instance_id}_mem_{idx:04d}"

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
                    privacy_level="unknown",
                    tags=[message.get("speaker_role") or "unknown"],
                    memory_status="active",
                    metadata={
                        "privacy_level": "unknown",
                        "access_scope": scope,
                        "authorized_users": [],
                        "forbidden_users": [],
                        "forget_after": None,
                        "is_deleted": False,
                        "redaction_required": False,
                        "sensitive_entities": [],
                        "supersedes_memory_ids": [],
                    },
                )
            )
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
        # Lifecycle transitions are supplied by the source-grounded semantic
        # compiler and symbolic state layer. Never infer them from raw words.
        del items


def _infer_scope(content: str, user_id: str | None) -> str:
    """Compatibility stub; semantic scope comes from the LLM extractor."""
    del content, user_id
    return "observable"


def _infer_memory_type(content: str) -> str:
    """Compatibility stub; memory type comes from the LLM extractor."""
    del content
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
    del instance, message, content
    return []


def _infer_forbidden_users(instance: MemoryInstance, content: str) -> list[str]:
    del instance, content
    return []


def _infer_redaction_required(content: str) -> bool:
    del content
    return False


def _is_forgetting_instruction(content: str) -> bool:
    del content
    return False


def _is_update_instruction(content: str) -> bool:
    del content
    return False


def _find_deletion_targets(previous_items: list[MemoryItem], instruction: str) -> list[MemoryItem]:
    del previous_items, instruction
    return []


def _find_update_targets(previous_items: list[MemoryItem], instruction: str) -> list[MemoryItem]:
    del previous_items, instruction
    return []


def _normalize_time_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
