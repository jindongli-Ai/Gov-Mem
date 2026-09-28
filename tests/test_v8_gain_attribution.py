"""Checks for diagnostic arithmetic and preserving paired failure cohorts."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("v8_attribution", Path(__file__).resolve().parents[1] / "scripts/analyze_v8_gain_attribution.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_symmetric_metric_interaction_is_not_all_assigned_to_last_factor():
    allocation = mod.component_decomposition([0, 1, 1], [1, 0, 0])
    assert allocation == pytest.approx({"utility": 1/3, "privacy": 1/3, "deletion": 1/3})
    assert mod.component_decomposition([.5, .2, .4], [.8, .2, .4]) == pytest.approx(
        {"utility": .144, "privacy": 0, "deletion": 0})
    reverse = mod.component_decomposition([1, 0, 0], [0, 1, 1])
    assert reverse == pytest.approx({key: -v for key, v in allocation.items()})


def test_unknown_judgment_never_becomes_success():
    assert mod.component_label({"query_type": "safety", "judge": {"deletion_leak": None}}) == "unknown"
    assert mod.component_label({"query_type": "utility", "judge": {"utility_ok": True, "action_ok": False}}) == "bad"


def test_paired_failures_and_new_regressions_remain_distinct(tmp_path):
    for version, actions in [("old", ["error", "answer", "error", "answer"]),
                             ("new", ["answer", "error", "error", "answer"])]:
        root = tmp_path / version
        (root / "official_eval").mkdir(parents=True)
        predictions = [{"checkpoint_id": str(i), "action": action} for i, action in enumerate(actions)]
        judges = [{"checkpoint_id": str(i), "query_type": "safety",
                   "judge": {"parse_ok": True, "deletion_leak": False}} for i in range(4)]
        (root / "predictions.jsonl").write_text("\n".join(map(json.dumps, predictions)))
        (root / "official_eval/judge_scores.jsonl").write_text("\n".join(map(json.dumps, judges)))
    cohorts = mod.paired_domain(tmp_path / "old", tmp_path / "new")
    assert {k: v["checkpoints"] for k, v in cohorts.items()} == {
        "recovered": 1, "new_error": 1, "both_error": 1, "both_success": 1}
    p = tmp_path / "new/predictions.jsonl"
    p.write_text(p.read_text() + '\n' + p.read_text().splitlines()[0])
    with pytest.raises(ValueError, match="Duplicate"):
        mod.paired_domain(tmp_path / "old", tmp_path / "new")
