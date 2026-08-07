"""Regression check: GateMem paper scoring defaults remain GPT-4o based."""

from gov_mem.eval import benchmark_official


def main() -> None:
    captured = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return None

    original = benchmark_official.subprocess.run
    benchmark_official.subprocess.run = fake_run
    try:
        benchmark_official.run_official_scorer(
            domain="medical",
            data_dir=None,
            predictions_path=benchmark_official.PROJECT_ROOT / "2026-07-23-07-23-21_2026-07-23-07-23-21_report.txt",
            out_dir=benchmark_official.PROJECT_ROOT / "outputs" / "smoke_official_judge_defaults",
            judge_provider="yunwu",
            judge_model="gpt-4o",
            judge_temperature=0.0,
            judge_max_output_tokens=4096,
        )
    finally:
        benchmark_official.subprocess.run = original
    command = captured["cmd"]
    assert "--use_llm_judge" in command, command
    assert command[command.index("--judge_model") + 1] == "gpt-4o", command
    assert command[command.index("--judge_temperature") + 1] == "0.0", command
    assert command[command.index("--judge_max_output_tokens") + 1] == "4096", command
    print("official_judge_defaults_smoke=PASS")


if __name__ == "__main__":
    main()
