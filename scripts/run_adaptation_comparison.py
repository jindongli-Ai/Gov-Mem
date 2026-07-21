from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ModuleNotFoundError:
    VENDORED_YAML = ROOT / "third_party" / "python_deps" / "gatemem_eval"
    if str(VENDORED_YAML) not in sys.path:
        sys.path.insert(0, str(VENDORED_YAML))
    import yaml

from gov_mem.utils.config import deep_update, load_yaml_config


RUN_GOVMEM = ROOT / "src" / "gov_mem" / "run_checkpoint_benchmark.py"
BUILD_OFFICIAL_SCORE = ROOT / "scripts" / "build_official_score.py"
ANALYZE_CLOSURE = ROOT / "scripts" / "analyze_adaptation_closure.py"
COMPARE_SCORES = ROOT / "scripts" / "compare_scores.py"


@dataclass
class Variant:
    name: str
    label: str
    overrides: dict[str, Any]


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[Variant]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    variants = [
        Variant(
            name=str(row["name"]),
            label=str(row.get("label") or row["name"]),
            overrides=dict(row.get("overrides") or {}),
        )
        for row in payload.get("variants") or []
    ]
    if not variants:
        raise ValueError(f"No variants found in manifest: {path}")
    return payload, variants


def _materialize_config(
    *,
    base_config: dict[str, Any],
    variant: Variant,
    suite_manifest: str,
    output_root: Path,
) -> tuple[Path, Path]:
    run_dir = output_root / "runs" / variant.name
    resolved = deep_update(base_config, variant.overrides)
    resolved = deep_update(
        resolved,
        {
            "runner": {
                "suite_manifest": suite_manifest,
                "output_dir": str(run_dir.relative_to(ROOT)),
                "stage": "all",
                "skip_official_eval": False,
            }
        },
    )
    config_path = output_root / "configs" / f"{variant.name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path, run_dir


def _run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), check=check, text=True, capture_output=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, *, manifest_path: Path, output_root: Path, baseline: Path, rows: list[dict[str, Any]], executed: bool) -> None:
    lines = [
        "# Phase 14 Adaptation Comparison Report",
        "",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Output root: `{output_root}`",
        f"- Baseline: `{baseline}`",
        "",
        "## Variants",
        "",
        "| Variant | MGS | Delta vs v0 | Trigger Rate | Enabled Rate | Accepted |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['MGS']:.4f} | {row['delta_mgs']:+.4f} | {row['trigger_rate']:.4f} | {row['enabled_rate']:.4f} | {row['accepted']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Materialized configs: `{output_root / 'configs'}`",
            f"- Run directories: `{output_root / 'runs'}`",
            f"- Summary JSON: `{output_root / 'comparison_summary.json'}`",
            f"- Summary MD: `{output_root / 'comparison_summary.md'}`",
            "",
            "## Note",
            "",
            (
                "- This workflow executed the suite runs, built official score artifacts, and aggregated adaptation closure summaries."
                if executed
                else "- This workflow was prepared and validated locally without executing new suite runs."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the Gov-Mem frozen adaptation comparison.")
    parser.add_argument("--manifest", default="configs/adaptation/gov_mem_smoke40_closure.yaml")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--baseline", default="outputs/gov_mem_v0_strong/official_score.json")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    payload, variants = _load_manifest(manifest_path)
    base_config_path = (ROOT / str(payload["base_config"])).resolve()
    base_config = load_yaml_config(base_config_path)
    suite_manifest = str(payload.get("suite_manifest") or (base_config.get("runner") or {}).get("suite_manifest"))
    if not suite_manifest:
        raise ValueError("No suite manifest specified.")

    selected = {item.strip() for item in args.only if item.strip()}
    if selected:
        variants = [row for row in variants if row.name in selected]
        if not variants:
            raise ValueError(f"No variants matched --only={sorted(selected)}")

    default_root = ROOT / str(payload.get("output_root") or "outputs/adaptation")
    output_root = Path(args.output_root).resolve() if args.output_root else (default_root / _timestamp())
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = (ROOT / args.baseline).resolve()

    commands: list[dict[str, str]] = []
    summary_rows: list[dict[str, Any]] = []
    executed = not args.prepare_only
    for variant in variants:
        config_path, run_dir = _materialize_config(
            base_config=base_config,
            variant=variant,
            suite_manifest=suite_manifest,
            output_root=output_root,
        )
        run_cmd = [sys.executable, str(RUN_GOVMEM), "--config", str(config_path)]
        commands.append({"variant": variant.name, "command": " ".join(run_cmd)})
        if args.prepare_only:
            continue

        run_proc = _run(run_cmd, cwd=ROOT)
        (run_dir / "run_stdout.txt").write_text(run_proc.stdout, encoding="utf-8")
        (run_dir / "run_stderr.txt").write_text(run_proc.stderr, encoding="utf-8")

        official_score_path = run_dir / "official_score.json"
        score_cmd = [
            sys.executable,
            str(BUILD_OFFICIAL_SCORE),
            "--run_dir",
            str(run_dir),
            "--config",
            str(config_path),
            "--output",
            str(official_score_path),
            "--run_name",
            variant.name,
            "--tag",
            "adaptation_comparison",
        ]
        score_proc = _run(score_cmd, cwd=ROOT)
        (run_dir / "official_score_stdout.txt").write_text(score_proc.stdout, encoding="utf-8")
        (run_dir / "official_score_stderr.txt").write_text(score_proc.stderr, encoding="utf-8")

        closure_json = run_dir / "adaptation_closure.json"
        closure_md = run_dir / "adaptation_closure.md"
        closure_cmd = [
            sys.executable,
            str(ANALYZE_CLOSURE),
            "--run_dir",
            str(run_dir),
            "--output_json",
            str(closure_json),
            "--output_md",
            str(closure_md),
        ]
        closure_proc = _run(closure_cmd, cwd=ROOT)
        (run_dir / "adaptation_closure_stdout.txt").write_text(closure_proc.stdout, encoding="utf-8")
        (run_dir / "adaptation_closure_stderr.txt").write_text(closure_proc.stderr, encoding="utf-8")

        compare_cmd = [
            sys.executable,
            str(COMPARE_SCORES),
            "--baseline",
            str(baseline),
            "--candidate",
            str(official_score_path),
        ]
        compare_proc = _run(compare_cmd, cwd=ROOT, check=False)
        (run_dir / "vs_v0.txt").write_text(compare_proc.stdout + compare_proc.stderr, encoding="utf-8")

        score = _read_json(official_score_path)
        closure = _read_json(closure_json)
        summary_rows.append(
            {
                "variant": variant.name,
                "label": variant.label,
                "run_dir": str(run_dir),
                "config_path": str(config_path),
                "official_score": str(official_score_path),
                "adaptation_closure_json": str(closure_json),
                "MGS": float(dict(score.get("overall") or {}).get("MGS") or 0.0),
                "delta_mgs": float(dict(score.get("overall") or {}).get("MGS") or 0.0) - float(
                    dict(_read_json(baseline).get("overall") or {}).get("MGS") or 0.0
                ),
                "trigger_rate": float(closure.get("trigger_rate") or 0.0),
                "enabled_rate": float(closure.get("enabled_rate") or 0.0),
                "accepted": compare_proc.returncode == 0,
            }
        )

    commands_path = output_root / "commands.json"
    commands_path.write_text(json.dumps(commands, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.prepare_only:
        print(f"prepared_output_root={output_root}")
        print(f"commands={commands_path}")
    else:
        summary_json = output_root / "comparison_summary.json"
        summary_md = output_root / "comparison_summary.md"
        summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        lines = [
            "# Adaptation Comparison Summary",
            "",
            f"- Output root: `{output_root}`",
            f"- Baseline: `{baseline}`",
            "",
            "| Variant | MGS | Delta vs v0 | Trigger Rate | Enabled Rate | Accepted |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for row in summary_rows:
            lines.append(
                f"| {row['label']} | {row['MGS']:.4f} | {row['delta_mgs']:+.4f} | {row['trigger_rate']:.4f} | {row['enabled_rate']:.4f} | {row['accepted']} |"
            )
        summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report_path = ROOT / "outputs" / "reports" / "phase14_adaptation_comparison_report.md"
        _write_report(
            report_path,
            manifest_path=manifest_path,
            output_root=output_root,
            baseline=baseline,
            rows=summary_rows,
            executed=executed,
        )
        print(f"output_root={output_root}")
        print(f"summary_json={summary_json}")
        print(f"summary_md={summary_md}")
        print(f"report={report_path}")


if __name__ == "__main__":
    main()
