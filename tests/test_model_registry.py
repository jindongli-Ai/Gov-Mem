from pathlib import Path

import pytest

from gov_mem.llm.model_registry import build_resolved_llm_settings, resolve_llm_model
from gov_mem.utils.config import load_yaml_config


def test_repository_default_uses_gemini_for_all_memory_roles():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / "configs" / "govmem_default.yaml")
    settings = build_resolved_llm_settings(config)

    assert settings["provider"] == "openlux"
    assert settings["base_model"] == "gemini-2.5-flash-lite"
    assert set(settings["role_models_resolved"].values()) == {"gemini-2.5-flash-lite"}


@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5.4-mini"])
def test_openlux_cost_safe_mini_aliases_are_preserved(model):
    assert resolve_llm_model(
        {"llm": {"provider": "openlux", "base_model": model}}, "reasoning"
    ) == model


@pytest.mark.parametrize(
    "model", ["gpt-5-mini-2025-08-07", "gpt-5.4-mini-2025-08-07"]
)
def test_openlux_rejects_date_pinned_mini_models(model):
    with pytest.raises(ValueError, match="date-pinned"):
        resolve_llm_model(
            {"llm": {"provider": "openlux", "base_model": model}}, "answering"
        )


def test_non_openlux_historical_snapshot_is_unmodified():
    model = "gpt-5-mini-2025-08-07"
    assert resolve_llm_model(
        {"llm": {"provider": "yunwu", "base_model": model}}, "answering"
    ) == model
