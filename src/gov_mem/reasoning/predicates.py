from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Predicate:
    name: str
    args: tuple[Any, ...]
    truth: bool
    provenance: list[str]
    confidence: float


def predicate_to_dict(predicate: Predicate) -> dict[str, Any]:
    return asdict(predicate)
