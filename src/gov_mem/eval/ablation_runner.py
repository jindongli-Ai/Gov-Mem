from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ModuleNotFoundError:
    deps_root = ROOT / "third_party" / "python_deps"
    fallback_dep = next((path for path in deps_root.iterdir() if path.is_dir()), None) if deps_root.exists() else None
    if fallback_dep is not None and str(fallback_dep) not in sys.path:
        sys.path.insert(0, str(fallback_dep))
    import yaml

from gov_mem.utils.config import deep_update, load_yaml_config

RUN_GOVMEM = ROOT / "src" / "gov_mem" / "run_checkpoint_benchmark.py"


@dataclass
class AblationVariant:
    name: str
    label: str
    overrides: dict[str, Any]


@dataclass
class AblationResult:
    variant: str
    label: str
    checkpoints: int
    U: float
    A: float
    F: float
    MGS: float
    action_accuracy: float
    config_path: str
    run_dir: str
    delta_mgs: float = 0.0


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[AblationVariant]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    variants = [
        AblationVariant(
            name=str(row["name"]),
            label=str(row.get("label") or row["name"]),
            overrides=dict(row.get("overrides") or {}),
        )
        for row in payload.get("variants") or []
    ]
    if not variants:
        raise ValueError(f"No ablation variants found in {path}")
    return payload, variants


def _materialize_config(
    *,
    base_config: dict[str, Any],
    variant: AblationVariant,
    suite_manifest: str,
    output_root: Path,
) -> tuple[Path, Path]:
    run_dir = output_root / "runs" / variant.name
    resolved = deep_update(base_config, variant.overrides)
    runner_updates = {
        "runner": {
            "suite_manifest": suite_manifest,
            "output_dir": str(run_dir.relative_to(ROOT)),
            "stage": "all",
            "skip_official_eval": False,
        }
    }
    resolved = deep_update(resolved, runner_updates)
    config_path = output_root / "configs" / f"{variant.name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path, run_dir


def _run_variant(*, config_path: Path) -> None:
    cmd = [sys.executable, str(RUN_GOVMEM), "--config", str(config_path)]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _aggregate_suite_summary(suite_summary: dict[str, Any]) -> tuple[int, float, float, float, float]:
    total_checkpoints = 0
    total_action = 0.0
    total_utility_scored = 0
    total_utility_correct = 0.0
    total_privacy_scored = 0
    total_privacy_leaks = 0.0
    total_deletion_scored = 0
    total_deletion_leaks = 0.0

    for row in (suite_summary.get("domains") or {}).values():
        summary = dict(row.get("summary") or {})
        checkpoints = int(summary.get("n_checkpoints") or row.get("n_entries") or 0)
        total_checkpoints += checkpoints
        total_action += float(summary.get("action_accuracy") or 0.0) * checkpoints

        utility_scored = int(summary.get("n_utility_scored") or 0)
        privacy_scored = int(summary.get("n_privacy_scored") or 0)
        deletion_scored = int(summary.get("n_safety_scored") or 0)

        total_utility_scored += utility_scored
        total_privacy_scored += privacy_scored
        total_deletion_scored += deletion_scored

        total_utility_correct += float(summary.get("utility_accuracy") or 0.0) * utility_scored
        total_privacy_leaks += float(summary.get("privacy_leakage_rate") or 0.0) * privacy_scored
        total_deletion_leaks += float(summary.get("deletion_leakage_rate") or 0.0) * deletion_scored

    U = (total_utility_correct / total_utility_scored) if total_utility_scored else 0.0
    A = (total_privacy_leaks / total_privacy_scored) if total_privacy_scored else 0.0
    F = (total_deletion_leaks / total_deletion_scored) if total_deletion_scored else 0.0
    MGS = U * (1.0 - A) * (1.0 - F)
    action_accuracy = (total_action / total_checkpoints) if total_checkpoints else 0.0
    return total_checkpoints, U, A, F, action_accuracy


def _write_csv(path: Path, rows: list[AblationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "label", "checkpoints", "U", "A", "F", "MGS", "action_accuracy", "delta_mgs", "config_path", "run_dir"])
        for row in rows:
            writer.writerow(
                [
                    row.variant,
                    row.label,
                    row.checkpoints,
                    f"{row.U:.4f}",
                    f"{row.A:.4f}",
                    f"{row.F:.4f}",
                    f"{row.MGS:.4f}",
                    f"{row.action_accuracy:.4f}",
                    f"{row.delta_mgs:+.4f}",
                    row.config_path,
                    row.run_dir,
                ]
            )


def _write_tex(path: Path, rows: list[AblationResult]) -> None:
    lines = [
        "% Auto-generated by gov_mem.eval.ablation_runner",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Variant & U & A & F & MGS & Action Acc. & $\\Delta$MGS \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_escape_tex(row.label)} & {row.U:.4f} & {row.A:.4f} & {row.F:.4f} & {row.MGS:.4f} & {row.action_accuracy:.4f} & {row.delta_mgs:+.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    *,
    path: Path,
    manifest_path: Path,
    suite_manifest: str,
    output_root: Path,
    rows: list[AblationResult],
    executed: bool,
) -> None:
    lines = [
        "# Phase 12 Ablation Report",
        "",
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        "Phase 12 only.",
        f"Ablation manifest: `{manifest_path}`",
        f"Suite manifest: `{suite_manifest}`",
        f"Output root: `{output_root}`",
        "",
        "## Variants",
        "",
        "| Variant | U | A | F | MGS | Action Acc. | Delta MGS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.label} | {row.U:.4f} | {row.A:.4f} | {row.F:.4f} | {row.MGS:.4f} | {row.action_accuracy:.4f} | {row.delta_mgs:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- CSV: `{output_root / 'ablation_results.csv'}`",
            f"- TeX: `{output_root / 'ablation_table.tex'}`",
            f"- Materialized configs: `{output_root / 'configs'}`",
            "",
            "## Note",
            "",
            (
                "- This run executed the ablation variants and read the official checkpoint benchmark suite summaries."
                if executed
                else "- This report was generated without executing new API runs; metrics were read from existing ablation run directories when available."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_tex(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or summarize Gov-Mem ablations.")
    parser.add_argument("--manifest", default="configs/ablations/gov_mem_smoke40.yaml")
    parser.add_argument("--output_root", default=None, help="Optional explicit output root. Default uses manifest output_root + timestamp.")
    parser.add_argument("--suite_manifest_override", default=None)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--skip_run", action="store_true")
    args = parser.parse_args()

    manifest_path = (ROOT / args.manifest).resolve()
    payload, variants = _load_manifest(manifest_path)
    base_config_path = (ROOT / str(payload["base_config"])).resolve()
    base_config = load_yaml_config(base_config_path)
    suite_manifest = str(args.suite_manifest_override or payload.get("suite_manifest") or (base_config.get("runner") or {}).get("suite_manifest"))
    if not suite_manifest:
        raise ValueError("No suite manifest specified in ablation manifest or CLI.")
    selected = {item.strip() for item in args.only if item.strip()}
    if selected:
        variants = [row for row in variants if row.name in selected]
        if not variants:
            raise ValueError(f"No ablation variants matched --only={sorted(selected)}")

    default_root = ROOT / str(payload.get("output_root") or "outputs/ablation")
    output_root = Path(args.output_root).resolve() if args.output_root else (default_root / _timestamp())
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[AblationResult] = []
    executed = not args.skip_run
    for variant in variants:
        config_path, run_dir = _materialize_config(
            base_config=base_config,
            variant=variant,
            suite_manifest=suite_manifest,
            output_root=output_root,
        )
        if not args.skip_run:
            _run_variant(config_path=config_path)
        suite_summary_path = run_dir / "suite_summary.json"
        if not suite_summary_path.exists():
            raise FileNotFoundError(f"Missing suite summary for variant {variant.name}: {suite_summary_path}")
        suite_summary = yaml.safe_load(suite_summary_path.read_text(encoding="utf-8")) or {}
        checkpoints, U, A, F, action_accuracy = _aggregate_suite_summary(suite_summary)
        results.append(
            AblationResult(
                variant=variant.name,
                label=variant.label,
                checkpoints=checkpoints,
                U=U,
                A=A,
                F=F,
                MGS=U * (1.0 - A) * (1.0 - F),
                action_accuracy=action_accuracy,
                config_path=str(config_path),
                run_dir=str(run_dir),
            )
        )

    full_row = next((row for row in results if row.variant == "full_gov_mem"), None)
    if full_row is not None:
        for row in results:
            row.delta_mgs = row.MGS - full_row.MGS

    csv_path = output_root / "ablation_results.csv"
    tex_path = output_root / "ablation_table.tex"
    report_path = ROOT / "outputs" / "reports" / "phase12_ablation_report.md"
    _write_csv(csv_path, results)
    _write_tex(tex_path, results)
    _write_report(
        path=report_path,
        manifest_path=manifest_path,
        suite_manifest=suite_manifest,
        output_root=output_root,
        rows=results,
        executed=executed,
    )
    print(f"output_root={output_root}")
    print(f"csv={csv_path}")
    print(f"tex={tex_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
