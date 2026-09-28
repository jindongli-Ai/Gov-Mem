"""Build a closed principal registry from checkpoint-observable episode data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gov_mem.data.schema import MemoryInstance


@dataclass
class V8Principal:
    principal_id: str
    role: str
    display_name: str
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_id": self.principal_id,
            "role": self.role,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
        }


@dataclass
class V8PrincipalRegistry:
    principals: dict[str, V8Principal]
    static_relationships: list[dict[str, Any]]

    def candidate_payload(self) -> list[dict[str, Any]]:
        return [self.principals[key].to_dict() for key in sorted(self.principals)]

    def contains(self, principal_id: str | None) -> bool:
        return bool(principal_id and principal_id in self.principals)

    def resolve(self, mention: str | None) -> str | None:
        value = _normalize(mention)
        if not value:
            return None
        matches = []
        for principal in self.principals.values():
            aliases = [principal.principal_id, principal.display_name, *principal.aliases]
            if any(_normalize(alias) == value for alias in aliases if alias):
                matches.append(principal.principal_id)
        return matches[0] if len(set(matches)) == 1 else None


def _normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _name_aliases(principal_id: str, display_name: str) -> list[str]:
    aliases = [principal_id, display_name]
    cleaned = re.sub(r"\([^)]*\)", "", display_name).strip()
    if cleaned:
        aliases.append(cleaned)
        parts = cleaned.split()
        if parts:
            aliases.append(parts[0])
            aliases.append(parts[-1])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def build_v8_principal_registry(instance: MemoryInstance) -> V8PrincipalRegistry:
    episode = dict((instance.metadata.get("raw_sample") or {}).get("episode") or {})
    entities = dict(episode.get("entities") or {})
    principals: dict[str, V8Principal] = {}
    for raw in entities.get("principals") or []:
        if not isinstance(raw, dict):
            continue
        principal_id = str(raw.get("principal_id") or "").strip()
        if not principal_id:
            continue
        display_name = str(raw.get("display_name") or principal_id).strip()
        principals[principal_id] = V8Principal(
            principal_id=principal_id,
            role=str(raw.get("role") or "unknown").strip(),
            display_name=display_name,
            aliases=_name_aliases(principal_id, display_name),
        )

    # Identity endpoints and relation type are observable episode structure.
    # Scope/access attributes can summarize later transitions, so v8 never
    # imports them into the current authorization state.
    relationships = []
    for raw in entities.get("relationships") or []:
        if not isinstance(raw, dict):
            continue
        relation = {
            str(key): value
            for key, value in raw.items()
            if key == "type" or str(key).endswith("_id")
        }
        if relation.get("type"):
            relationships.append(relation)
    return V8PrincipalRegistry(principals=principals, static_relationships=relationships)

