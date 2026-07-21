from __future__ import annotations

from pathlib import Path

import yaml

from gov_mem.experience.experience_bank import ExperienceBank
from gov_mem.llm.client import LLMClient, LLMClientUnavailableError
from gov_mem.llm.prompts import (
    SKILL_UPDATE_SYSTEM_PROMPT,
    build_skill_update_user_prompt,
)
from gov_mem.skills.registry import SkillRegistry


class SkillUpdater:
    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model_name: str,
        skill_registry: SkillRegistry,
        enable_skill_update: bool,
        update_every_n_failures: int,
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.skill_registry = skill_registry
        self.enable_skill_update = enable_skill_update
        self.update_every_n_failures = update_every_n_failures
        self._last_update_bucket_by_stage: dict[str, int] = {}

    def maybe_update(self, *, stage: str, experience_bank: ExperienceBank) -> None:
        if not self.enable_skill_update:
            return
        failure_count = experience_bank.failure_count()
        if failure_count < self.update_every_n_failures:
            return
        bucket = failure_count // self.update_every_n_failures
        if self._last_update_bucket_by_stage.get(stage) == bucket:
            return

        lessons = experience_bank.retrieve_lessons(
            question=stage,
            top_k=min(4, self.update_every_n_failures + 1),
            stage=stage,
        )
        if not lessons:
            return

        try:
            raw = self.llm_client.chat_json(
                model=self.model_name,
                system_prompt=SKILL_UPDATE_SYSTEM_PROMPT,
                user_prompt=build_skill_update_user_prompt(stage=stage, failure_lessons=lessons),
            )
            instruction = _normalize_instruction((raw or {}).get("instruction"))
        except LLMClientUnavailableError:
            instruction = ""
        except Exception:
            instruction = ""

        if not instruction:
            instruction = f"Recent failures suggest extra caution for stage {stage}: " + "; ".join(lessons[:3])

        path = self.skill_registry.skill_dir / f"{stage}.yaml"
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        skills = list(data.get("skills", []))
        normalized_instruction = _normalize_instruction(instruction)
        if any(_normalize_instruction(skill.get("instruction")) == normalized_instruction for skill in skills):
            self._last_update_bucket_by_stage[stage] = bucket
            self.skill_registry.reload()
            return
        version = max((int(skill.get("version", 1)) for skill in skills), default=1) + 1
        skills.append(
            {
                "skill_id": f"{stage}_evolved_skill_v{version}",
                "name": f"{stage}_evolved_skill_v{version}",
                "stage": stage,
                "instruction": normalized_instruction,
                "version": version,
                "source": "evolved",
            }
        )
        data["skills"] = skills
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)
        self._last_update_bucket_by_stage[stage] = bucket
        self.skill_registry.reload()


def _normalize_instruction(value) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts)
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text.strip("[]")
    return " ".join(text.split())

