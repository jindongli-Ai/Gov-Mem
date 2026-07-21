from __future__ import annotations

from pathlib import Path

import yaml

from gov_mem.data.schema import Skill


DEFAULT_SKILLS = {
    "ingestion": [
        Skill(
            skill_id="memory_extraction_skill",
            name="memory_extraction_skill",
            stage="ingestion",
            instruction="Extract durable user, group, task, event, and constraint memories without leaking labels.",
            version=1,
            source="initial",
        ),
    ],
    "query_planning": [
        Skill(
            skill_id="speaker_grounding_skill",
            name="speaker_grounding_skill",
            stage="query_planning",
            instruction="Ground I, my, and we to the asking user when that user id is available.",
            version=1,
            source="initial",
        ),
    ],
    "retrieval": [
        Skill(
            skill_id="hybrid_retrieval_skill",
            name="hybrid_retrieval_skill",
            stage="retrieval",
            instruction="Combine symbolic filters and semantic retrieval, then favor evidence aligned with user, entity, and memory type.",
            version=1,
            source="initial",
        ),
    ],
    "reasoning": [
        Skill(
            skill_id="temporal_reasoning_skill",
            name="temporal_reasoning_skill",
            stage="reasoning",
            instruction="Prefer recent evidence when the question asks for current or latest state, and detect conflicts before concluding.",
            version=1,
            source="initial",
        ),
        Skill(
            skill_id="conflict_resolution_skill",
            name="conflict_resolution_skill",
            stage="reasoning",
            instruction="When memories conflict, track the conflict explicitly and prefer the most aligned or recent evidence.",
            version=1,
            source="initial",
        ),
    ],
    "action_decision": [
        Skill(
            skill_id="action_governance_skill",
            name="action_governance_skill",
            stage="action_decision",
            instruction="Separate answer, answer_redacted, refuse, and no_memory using access scope, deletion constraints, and logistics-safe delegate allowances.",
            version=1,
            source="initial",
        ),
    ],
    "answering": [
        Skill(
            skill_id="answer_calibration_skill",
            name="answer_calibration_skill",
            stage="answering",
            instruction="Answer in the required JSON format, cite used memory ids, and stay conservative when evidence is weak.",
            version=1,
            source="initial",
        ),
    ],
}


class SkillRegistry:
    def __init__(self, skill_dir: str | Path):
        self.skill_dir = Path(skill_dir)
        self.skills_by_stage: dict[str, list[Skill]] = {}
        self._load()

    def _load(self) -> None:
        self.skills_by_stage = {}
        if not self.skill_dir.exists():
            self.skill_dir.mkdir(parents=True, exist_ok=True)
            self._write_defaults()
        for path in self.skill_dir.glob("*.yaml"):
            with path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            stage = str(raw.get("stage") or path.stem)
            skills = [Skill(**row) for row in raw.get("skills", [])]
            self.skills_by_stage[stage] = skills

    def reload(self) -> None:
        self._load()

    def _write_defaults(self) -> None:
        stage_file_map = {
            "ingestion": "ingestion.yaml",
            "query_planning": "query_planning.yaml",
            "retrieval": "retrieval.yaml",
            "reasoning": "reasoning.yaml",
            "action_decision": "action_decision.yaml",
            "answering": "answering.yaml",
        }
        for stage, skills in DEFAULT_SKILLS.items():
            file_name = stage_file_map[stage]
            path = self.skill_dir / file_name
            payload = {
                "stage": stage,
                "skills": [skill.__dict__ for skill in skills],
            }
            with path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)

    def get_stage_text(self, stage: str) -> str:
        skills = self.skills_by_stage.get(stage, [])
        return "\n".join(f"- {skill.name}: {skill.instruction}" for skill in skills)

