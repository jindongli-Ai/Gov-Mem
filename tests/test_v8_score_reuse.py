import hashlib
import json
from pathlib import Path

import pytest

import importlib.util
spec = importlib.util.spec_from_file_location("v8_scoring", Path(__file__).resolve().parents[1] / "scripts/score_v8_paired_episode_suite.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    bench = repo / "third_party/GateMem-official/bench"
    bench.mkdir(parents=True)
    (bench / "scorer.py").write_text("# pinned scorer\n")
    monkeypatch.setattr(scoring, "ROOT", repo)
    old, new = tmp_path / "old", tmp_path / "new"
    pred = {"checkpoint_id": "q1", "answer": "Original baseline", "action": "answer"}
    for root in (old, new):
        path = root / "rag_naive/synthetic"
        path.mkdir(parents=True)
        (path / "predictions.jsonl").write_text(json.dumps(pred) + "\n")
    evaldir = old / "rag_naive/synthetic/official_eval"
    evaldir.mkdir()
    (evaldir / "judge_scores.jsonl").write_text(json.dumps({"checkpoint_id": "q1", "judge": {"parse_ok": True}}) + "\n")
    (evaldir / "summary.json").write_text(json.dumps({"gated_by_action": False,
        "llm_judge": {"llm": {"model": "gpt-4o", "provider": "openlux"}}}))
    (old / "official_source_identity.json").write_text(json.dumps({"files": {
        "scorer.py": hashlib.sha256((bench / "scorer.py").read_bytes()).hexdigest()}}))
    manifest = {"domains": {"synthetic": {}}, "entries": [{"domain": "synthetic", "checkpoint_id": "q1"}]}
    return old, new, manifest, bench


def test_reuse_requires_exact_baseline_predictions_and_preserves_provenance(tmp_path, monkeypatch):
    old, new, manifest, _ = fixture(tmp_path, monkeypatch)
    scoring.reuse_baseline_scores(old, new, manifest)
    assert (new / "rag_naive/synthetic/official_eval/judge_scores.jsonl").exists()
    assert scoring.read(new / "reused_baseline_scores.json")["synthetic"]["checkpoints"] == 1
    assert not list(new.rglob("judge_http_telemetry.jsonl"))


@pytest.mark.parametrize("change", ["prediction", "source", "judge"])
def test_invalid_reuse_is_rejected(tmp_path, monkeypatch, change):
    old, new, manifest, bench = fixture(tmp_path, monkeypatch)
    if change == "prediction":
        (new / "rag_naive/synthetic/predictions.jsonl").write_text('{"checkpoint_id":"q1","answer":"Different"}\n')
    elif change == "source":
        (bench / "scorer.py").write_text("# changed scorer\n")
    else:
        (old / "rag_naive/synthetic/official_eval/judge_scores.jsonl").write_text('{"checkpoint_id":"q1","judge":{"parse_ok":false}}\n')
    with pytest.raises(ValueError):
        scoring.reuse_baseline_scores(old, new, manifest)
