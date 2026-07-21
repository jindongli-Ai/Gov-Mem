from __future__ import annotations

from collections import defaultdict

from gov_mem.data.schema import MemoryItem


class SymbolicMemoryStore:
    def __init__(self, items: list[MemoryItem]):
        self.items = items
        self._by_memory_id = {item.memory_id: item for item in items}
        self._by_user_id = defaultdict(list)
        self._by_entity = defaultdict(list)
        self._by_memory_type = defaultdict(list)
        self._by_scope = defaultdict(list)
        self._by_source_message_id = defaultdict(list)
        self._by_time = defaultdict(list)

        for item in items:
            if item.user_id:
                self._by_user_id[item.user_id].append(item)
            for entity in item.entities:
                self._by_entity[entity].append(item)
            self._by_memory_type[item.memory_type].append(item)
            self._by_scope[item.scope].append(item)
            for source_message_id in item.source_message_ids:
                self._by_source_message_id[source_message_id].append(item)
            if item.time:
                self._by_time[item.time].append(item)

    def filter(
        self,
        *,
        user_ids: list[str] | None = None,
        entities: list[str] | None = None,
        memory_types: list[str] | None = None,
        scopes: list[str] | None = None,
        source_message_ids: list[str] | None = None,
        time_values: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[MemoryItem]:
        candidates = list(self.items)

        def _apply(current: list[MemoryItem], allowed_ids: set[str]) -> list[MemoryItem]:
            return [item for item in current if item.memory_id in allowed_ids]

        if user_ids:
            allowed = {
                item.memory_id
                for user_id in user_ids
                for item in self._by_user_id.get(user_id, [])
            }
            candidates = _apply(candidates, allowed)
        if entities:
            allowed = {
                item.memory_id
                for entity in entities
                for item in self._by_entity.get(entity, [])
            }
            candidates = _apply(candidates, allowed)
        if memory_types:
            allowed = {
                item.memory_id
                for memory_type in memory_types
                for item in self._by_memory_type.get(memory_type, [])
            }
            candidates = _apply(candidates, allowed)
        if scopes:
            allowed = {
                item.memory_id
                for scope in scopes
                for item in self._by_scope.get(scope, [])
            }
            candidates = _apply(candidates, allowed)
        if source_message_ids:
            allowed = {
                item.memory_id
                for source_message_id in source_message_ids
                for item in self._by_source_message_id.get(source_message_id, [])
            }
            candidates = _apply(candidates, allowed)
        if time_values:
            allowed = {
                item.memory_id
                for time_value in time_values
                for item in self._by_time.get(time_value, [])
            }
            candidates = _apply(candidates, allowed)

        if top_k is not None:
            return candidates[:top_k]
        return candidates

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._by_memory_id.get(memory_id)

