from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import md5
from typing import Any

from gov_mem.experience.pattern_inducer import FailurePattern
from gov_mem.skills.governance_skill import GovernanceSkill


@dataclass
class AuditableUpdate:
    update_id: str
    source_pattern: str
    update_type: str
    before: Any
    after: Any
    applied_to: list[str]
    created_from_dev_only: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class RuleUpdater:
    def build(
        self,
        *,
        patterns: list[FailurePattern],
        skills: list[GovernanceSkill],
    ) -> list[AuditableUpdate]:
        pattern_by_id = {pattern.pattern_id: pattern for pattern in patterns}
        updates: list[AuditableUpdate] = []
        for skill in skills:
            for pattern_id in skill.source_patterns:
                pattern = pattern_by_id.get(pattern_id)
                if pattern is None:
                    continue
                for patch_name in skill.symbolic_rule_patch:
                    updates.append(
                        AuditableUpdate(
                            update_id=md5(f"rule:{pattern_id}:{patch_name}".encode("utf-8")).hexdigest()[:12],
                            source_pattern=pattern_id,
                            update_type="rule_patch",
                            before=[],
                            after=patch_name,
                            applied_to=["symbolic_reasoner"],
                            metadata={
                                "skill_id": skill.skill_id,
                                "failure_type": pattern.failure_type,
                                "domain": pattern.affected_domains,
                            },
                        )
                    )
        return _dedupe_updates(updates)


def auditable_update_to_dict(update: AuditableUpdate) -> dict[str, Any]:
    return asdict(update)


def _dedupe_updates(updates: list[AuditableUpdate]) -> list[AuditableUpdate]:
    deduped: list[AuditableUpdate] = []
    seen: set[tuple[str, str, str]] = set()
    for update in updates:
        key = (update.source_pattern, update.update_type, str(update.after))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(update)
    return deduped
