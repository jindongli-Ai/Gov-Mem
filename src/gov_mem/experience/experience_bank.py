from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gov_mem.data.schema import ExperienceItem
from gov_mem.utils.io import append_jsonl, read_jsonl


class ExperienceBank:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self.items: list[ExperienceItem] = []
        if self.path is not None and self.path.exists():
            self.items = [ExperienceItem(**row) for row in read_jsonl(self.path)]

    def add(self, item: ExperienceItem) -> None:
        self.items.append(item)
        if self.path is not None:
            append_jsonl(self.path, asdict(item))

    def retrieve_lessons(
        self,
        *,
        question: str,
        top_k: int = 3,
        stage: str | None = None,
        query_type: str | None = None,
        domain: str | None = None,
        prefer_failures: bool = True,
    ) -> list[str]:
        # Runtime lesson retrieval must not depend on dataset-provided query labels.
        _ = query_type
        query_tokens = _tokenize(question)
        scored: list[tuple[float, str]] = []
        seen: set[str] = set()
        for item in self.items:
            lesson_text = (item.suggested_skill_update or item.lesson or "").strip()
            if not lesson_text:
                continue
            lesson_tokens = _tokenize(lesson_text)
            score = float(len(query_tokens & lesson_tokens))
            if domain and getattr(item, "domain", None) == domain:
                score += 3.0
            elif domain and getattr(item, "domain", None) and getattr(item, "domain", None) != domain:
                score -= 2.5
            if stage and item.failure_type:
                score += _stage_failure_alignment(stage, item.failure_type)
            if prefer_failures:
                score += 1.5 if not item.success else 0.3
            elif item.success:
                score += 0.8
            if score <= 0 and not item.failure_type:
                continue
            if lesson_text in seen:
                continue
            seen.add(lesson_text)
            scored.append((score, lesson_text))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [lesson for _, lesson in scored[:top_k]]

    def failure_count(self) -> int:
        return sum(1 for item in self.items if not item.success)


def _tokenize(text: str) -> set[str]:
    return {token for token in str(text or "").lower().replace("_", " ").replace("-", " ").split() if token}


def _stage_failure_alignment(stage: str, failure_type: str) -> float:
    stage = str(stage or "").lower()
    failure_type = str(failure_type or "").lower()
    stage_to_failures = {
        "query_planning": {"wrong_user_grounding", "retrieval_miss", "slot_selection_error"},
        "action_decision": {"access_filter_error", "official_match_miss", "retrieval_miss"},
        "answering": {"renderer_omission", "official_match_miss", "temporal_error", "conflict_unresolved"},
        "retrieval": {"retrieval_miss", "slot_selection_error"},
        "reasoning": {"temporal_error", "conflict_unresolved", "state_tracking_error", "slot_selection_error"},
        "ingestion": {"slot_selection_error", "state_tracking_error"},
    }
    matches = stage_to_failures.get(stage, set())
    return 2.5 if failure_type in matches else 0.0
