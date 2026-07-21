from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    import sys

    deps_root = Path(__file__).resolve().parents[3] / "third_party" / "python_deps"
    fallback_dep = next((path for path in deps_root.iterdir() if path.is_dir()), None) if deps_root.exists() else None
    if fallback_dep is not None and str(fallback_dep) not in sys.path:
        sys.path.insert(0, str(fallback_dep))
    import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_random_seed(seed: int) -> None:
    random.seed(seed)
