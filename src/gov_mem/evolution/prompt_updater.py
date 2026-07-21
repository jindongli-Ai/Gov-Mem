from __future__ import annotations

from hashlib import md5

from gov_mem.experience.pattern_inducer import FailurePattern
from gov_mem.evolution.rule_updater import AuditableUpdate
from gov_mem.skills.governance_skill import GovernanceSkill


class PromptUpdater:
    def build(
        self,
        *,
        patterns: list[FailurePattern],
        skills: list[GovernanceSkill],
    ) -> list[AuditableUpdate]:
        pattern_by_id = {pattern.pattern_id: pattern for pattern in patterns}
        updates: list[AuditableUpdate] = []
        for skill in skills:
            if not skill.prompt_patch.strip():
                continue
            for pattern_id in skill.source_patterns:
                pattern = pattern_by_id.get(pattern_id)
                if pattern is None:
                    continue
                updates.append(
                    AuditableUpdate(
                        update_id=md5(f"prompt:{pattern_id}:{skill.skill_id}".encode("utf-8")).hexdigest()[:12],
                        source_pattern=pattern_id,
                        update_type="prompt_patch",
                        before="",
                        after=skill.prompt_patch.strip(),
                        applied_to=["renderer"],
                        metadata={
                            "skill_id": skill.skill_id,
                            "failure_type": pattern.failure_type,
                            "domain": pattern.affected_domains,
                        },
                    )
                )
        return _dedupe(updates)


def _dedupe(updates: list[AuditableUpdate]) -> list[AuditableUpdate]:
    deduped: list[AuditableUpdate] = []
    seen: set[tuple[str, str]] = set()
    for update in updates:
        key = (update.source_pattern, str(update.after))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(update)
    return deduped
