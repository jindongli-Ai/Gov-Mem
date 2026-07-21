from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from gov_mem.utils.io import ensure_dir, write_jsonl


class ExperienceStore:
    def __init__(self, *, output_dir: str | Path):
        self.output_dir = ensure_dir(output_dir)
        self.failure_case_path = self.output_dir / "failure_cases.jsonl"
        self.experience_path = self.output_dir / "experience_memory.jsonl"

    def save_failure_cases(self, failure_cases: list[Any]) -> None:
        write_jsonl(self.failure_case_path, [asdict(case) for case in failure_cases])

    def save_experiences(self, experiences: list[Any]) -> None:
        write_jsonl(self.experience_path, [asdict(item) for item in experiences])

    def artifact_paths(self) -> dict[str, Any]:
        return {
            "failure_cases": str(self.failure_case_path),
            "experience_memory": str(self.experience_path),
        }
