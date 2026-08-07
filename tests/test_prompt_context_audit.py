from types import SimpleNamespace
from pathlib import Path

from gov_mem.evaluation.prompt_context_audit import audit_prompt_contexts
from gov_mem.pipeline import GovMemRunner


def _instance(checkpoint_id, query_type, leak_target):
    return SimpleNamespace(
        instance_id=checkpoint_id,
        metadata={
            "evaluation": {
                "query_type": query_type,
                "leak_targets": [leak_target],
                "judge_spec": {"not_include": [leak_target] if query_type == "privacy" else []},
            }
        },
    )


def test_prompt_context_audit_scans_runtime_context_and_reports_coverage(tmp_path):
    audit_dir = tmp_path / "prompt_audit"
    audit_dir.mkdir()
    (audit_dir / "privacy_01.json").write_text(
        '{"schema_version":1,"answer_prompt":{"context_text":"private name"},'
        '"stage2_rerank_prompt":{"context_text":"private name"}}',
        encoding="utf-8",
    )
    (audit_dir / "safety_01.json").write_text(
        '{"schema_version":1,"answer_prompt":{"context_text":"safe summary"}}',
        encoding="utf-8",
    )

    summary = audit_prompt_contexts(
        audit_dir=audit_dir,
        instances=[
            _instance("privacy_01", "privacy", "private name"),
            _instance("safety_01", "safety", "deleted address"),
        ],
    )

    assert summary["n_audited_checkpoints"] == 2
    assert summary["audit_coverage_rate"] == 1.0
    assert summary["privacy_context_leakage_rate"] == 1.0
    assert summary["privacy_stage2_prompt_leakage_rate"] == 1.0
    assert summary["deletion_context_leakage_rate"] == 0.0


def _runner_config(*, clean_benchmark, allow_gold_feedback):
    return {
        "llm": {
            "provider": "stub",
            "temperature": 0.0,
            "max_retries": 0,
            "allow_fallback": True,
            "request_timeout": 1,
            "base_model": "stub",
            "memory_ingestion_model": "stub",
            "query_planner_model": "stub",
            "answering_model": "stub",
        },
        "embedding": {"provider": "stub", "model": "stub", "allow_fallback": True},
        "pipeline": {"use_asking_user_id": True},
        "self_evolving": {
            "enable_experience": True,
            "enable_skill_update": False,
            "update_every_n_failures": 20,
        },
        "evaluation": {
            "clean_benchmark": clean_benchmark,
            "allow_gold_feedback": allow_gold_feedback,
        },
        "ablation": {},
    }


def test_clean_benchmark_blocks_gold_feedback_by_default(tmp_path):
    runner = GovMemRunner(
        dataset_name="checkpoint_benchmark",
        data_path=Path("dataset/GateMem/gatemem/data"),
        output_dir=tmp_path / "clean",
        config=_runner_config(clean_benchmark=True, allow_gold_feedback=True),
        stage="answer",
        run_official_benchmark_eval=False,
        experiment_mode="rag_naive_v3_typed_rerank",
    )
    assert runner.experience_bank is None
    assert runner.clean_benchmark is True


def test_adaptation_feedback_requires_explicit_nonbenchmark_mode(tmp_path):
    runner = GovMemRunner(
        dataset_name="checkpoint_benchmark",
        data_path=Path("dataset/GateMem/gatemem/data"),
        output_dir=tmp_path / "adaptation",
        config=_runner_config(clean_benchmark=False, allow_gold_feedback=True),
        stage="answer",
        run_official_benchmark_eval=False,
        experiment_mode="govmem_structured_old",
    )
    assert runner.experience_bank is not None
