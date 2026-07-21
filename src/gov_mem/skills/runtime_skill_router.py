from __future__ import annotations

from pathlib import Path

from gov_mem.evolution.dev_guard import has_valid_embedded_dev_attestation
from gov_mem.skills.skill_context import SkillQueryContext
from gov_mem.skills.skill_executor import GovernanceSkillExecutor, RetrievedSkillBundle
from gov_mem.skills.skill_library import GovernanceSkillLibrary
from gov_mem.skills.skill_retriever import GovernanceSkillRetriever
from gov_mem.utils.io import read_jsonl


class RuntimeSkillRouter:
    def __init__(
        self,
        *,
        skill_library_path: str | Path | None,
        rule_patches_path: str | Path | None = None,
        prompt_patches_path: str | Path | None = None,
        policy_patches_path: str | Path | None = None,
    ):
        self.skill_library_path = Path(skill_library_path) if skill_library_path else None
        self.rule_patches_path = Path(rule_patches_path) if rule_patches_path else None
        self.prompt_patches_path = Path(prompt_patches_path) if prompt_patches_path else None
        self.policy_patches_path = Path(policy_patches_path) if policy_patches_path else None
        self.skill_retriever = GovernanceSkillRetriever()
        self.skill_executor = GovernanceSkillExecutor()

    def route(self, *, context: SkillQueryContext) -> RetrievedSkillBundle:
        if self.skill_library_path is None or not self.skill_library_path.exists():
            return RetrievedSkillBundle(skill_trace=["skill_library_unavailable"])
        skills = GovernanceSkillLibrary(self.skill_library_path).all()
        skills = [
            skill for skill in skills
            if has_valid_embedded_dev_attestation({"metadata": dict(skill.metadata or {})})
        ]
        if not skills:
            return RetrievedSkillBundle(skill_trace=["skill_library_missing_dev_attestation"])
        selected = self.skill_retriever.retrieve(
            context=context,
            skills=skills,
        )
        if not selected:
            return RetrievedSkillBundle(skill_trace=["no_runtime_skill_selected"])
        bundle = self.skill_executor.execute(
            context=context,
            selected_skills=selected,
        )
        return self._merge_auditable_updates(context=context, bundle=bundle)

    def _merge_auditable_updates(self, *, context: SkillQueryContext, bundle: RetrievedSkillBundle) -> RetrievedSkillBundle:
        selected_patterns = {
            pattern_id
            for row in bundle.selected_skills
            for pattern_id in list((row.get("source_patterns") or []))
        }
        selected_skill_ids = {
            str(row.get("skill_id") or row.get("name") or "")
            for row in bundle.selected_skills
            if str(row.get("skill_id") or row.get("name") or "").strip()
        }
        if self.rule_patches_path and self.rule_patches_path.exists():
            for row in read_jsonl(self.rule_patches_path):
                if (
                    has_valid_embedded_dev_attestation(dict(row))
                    and
                    row.get("source_pattern") in selected_patterns
                    and "symbolic_reasoner" in list(row.get("applied_to") or [])
                    and _update_applicable(row=row, context=context, selected_skill_ids=selected_skill_ids)
                ):
                    bundle.activated_rules.append(str(row.get("after")))
                    bundle.loaded_rule_updates.append(str(row.get("update_id") or ""))
                    bundle.skill_trace.append(f"loaded_rule_update:{row.get('update_id')}")
        if self.prompt_patches_path and self.prompt_patches_path.exists():
            for row in read_jsonl(self.prompt_patches_path):
                if (
                    has_valid_embedded_dev_attestation(dict(row))
                    and
                    row.get("source_pattern") in selected_patterns
                    and "renderer" in list(row.get("applied_to") or [])
                    and _update_applicable(row=row, context=context, selected_skill_ids=selected_skill_ids)
                ):
                    bundle.prompt_patches.append(str(row.get("after")))
                    bundle.loaded_prompt_updates.append(str(row.get("update_id") or ""))
                    bundle.skill_trace.append(f"loaded_prompt_update:{row.get('update_id')}")
        if self.policy_patches_path and self.policy_patches_path.exists():
            for row in read_jsonl(self.policy_patches_path):
                if has_valid_embedded_dev_attestation(dict(row)) and row.get("source_pattern") in selected_patterns and _update_applicable(
                    row=row,
                    context=context,
                    selected_skill_ids=selected_skill_ids,
                ):
                    after = dict(row.get("after") or {})
                    for verifier_patch in list(after.get("verifier_patch") or []):
                        bundle.verifier_patches.append(str(verifier_patch))
                    bundle.loaded_policy_updates.append(str(row.get("update_id") or ""))
                    bundle.skill_trace.append(f"loaded_policy_update:{row.get('update_id')}")
        bundle.activated_rules = list(dict.fromkeys(bundle.activated_rules))
        bundle.prompt_patches = list(dict.fromkeys(bundle.prompt_patches))
        bundle.verifier_patches = list(dict.fromkeys(bundle.verifier_patches))
        bundle.loaded_rule_updates = list(dict.fromkeys(bundle.loaded_rule_updates))
        bundle.loaded_prompt_updates = list(dict.fromkeys(bundle.loaded_prompt_updates))
        bundle.loaded_policy_updates = list(dict.fromkeys(bundle.loaded_policy_updates))
        return bundle


def _update_applicable(
    *,
    row: dict,
    context: SkillQueryContext,
    selected_skill_ids: set[str],
) -> bool:
    metadata = dict(row.get("metadata") or {})
    after = row.get("after")
    after_map = dict(after) if isinstance(after, dict) else {}

    skill_id = str(metadata.get("skill_id") or after_map.get("skill_id") or "").strip()
    if skill_id and selected_skill_ids and skill_id not in selected_skill_ids:
        return False

    domains = _normalize_list(after_map.get("domains"))
    if not domains:
        domains = _normalize_list(metadata.get("domain"))
    if domains and context.domain not in domains:
        return False

    roles = _normalize_list(after_map.get("roles"))
    if roles:
        active_roles = {context.requester_role, context.owner_relation}
        active_roles.discard("")
        if not active_roles.intersection(roles):
            return False

    slots = _normalize_list(after_map.get("slots"))
    if slots:
        active_slots = set(context.required_slots) | set(context.detected_sensitive_slots)
        if not active_slots.intersection(slots):
            return False

    return True


def _normalize_list(value) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if value in {None, ""}:
        return set()
    return {str(value).strip()}
