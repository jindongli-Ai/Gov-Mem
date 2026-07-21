from __future__ import annotations

from hashlib import md5

from gov_mem.experience.pattern_inducer import FailurePattern
from gov_mem.evolution.rule_updater import AuditableUpdate
from gov_mem.skills.governance_skill import GovernanceSkill


class PolicyUpdater:
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
                policy_patch = {
                    "skill_id": skill.skill_id,
                    "domains": skill.applicable_domains,
                    "roles": skill.applicable_roles,
                    "slots": skill.applicable_slots,
                    "verifier_patch": skill.verifier_patch,
                }
                updates.append(
                    AuditableUpdate(
                        update_id=md5(f"policy:{pattern_id}:{skill.skill_id}".encode("utf-8")).hexdigest()[:12],
                        source_pattern=pattern_id,
                        update_type="policy_patch",
                        before={},
                        after=policy_patch,
                        applied_to=["slot_authorization_map", "verifier"],
                        metadata={
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
