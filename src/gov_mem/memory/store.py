from __future__ import annotations

from dataclasses import dataclass, field

from gov_mem.data.schema import MemoryItem


@dataclass
class MemoryStoreBundle:
    items: list[MemoryItem] = field(default_factory=list)

    def by_id(self) -> dict[str, MemoryItem]:
        return {item.memory_id: item for item in self.items}

