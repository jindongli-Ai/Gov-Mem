"""Published, domain-independent structural vocabulary for Gov-Mem V8.

These are schema types, not keyword classifiers. No benchmark names, values,
role-to-permission tables, or domain-specific examples belong here.
"""
from __future__ import annotations
from typing import Any

COMMON_ENTITY_TYPES = (
    "person", "organization", "scene", "resource", "information_field",
    "event", "location", "credential", "record",
)
COMMON_RELATION_TYPES = (
    "about", "participates_in", "has_field", "assigned_to", "related_to",
    "separate_from", "supersedes", "updates", "deletes", "operational_duty",
)
COMMON_GOVERNANCE_EFFECTS = ("allow", "deny", "revoke", "require_permission")


def v8_scene_schema(domain: str | None = None) -> dict[str, Any]:
    # The same ontology applies to every domain, including unseen domains.
    return {
        "schema_version": "govmem-v8-general-ontology-1",
        "entity_types": list(COMMON_ENTITY_TYPES),
        "relation_types": list(COMMON_RELATION_TYPES),
        "governance_effects": list(COMMON_GOVERNANCE_EFFECTS),
        "lifecycle_effects": ["assert", "update", "supersede", "cancel", "delete"],
        "temporal_states": ["current", "historical", "unknown"],
        "duty_attributes": ["role", "action", "resource_category", "subject_id"],
    }
