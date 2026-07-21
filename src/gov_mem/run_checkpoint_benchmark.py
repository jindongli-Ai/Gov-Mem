from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gov_mem.utils.config import load_yaml_config

RUN_GOVMEM = ROOT / "run_govmem.py"


def _load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValueError(f"Suite manifest must be an object with entries: {path}")
    return payload


def _group_entries_by_domain(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["domain"])].append(entry)
    return dict(grouped)


def _write_domain_manifest(
    *,
    suite_name: str,
    version: int,
    domain: str,
    entries: list[dict],
    output_dir: Path,
) -> Path:
    manifest = {
        "suite_name": f"{suite_name}:{domain}",
        "dataset_name": "checkpoint_benchmark",
        "version": version,
        "domain": domain,
        "entries": entries,
    }
    path = output_dir / "suite_manifests" / f"{domain}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_command(
    *,
    dataset_name: str,
    data_root: Path,
    domain: str,
    domain_output_dir: Path,
    config_path: Path,
    experiment_mode: str,
    stage: str,
    domain_manifest: Path,
    skip_official_eval: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_GOVMEM),
        "--dataset_name",
        dataset_name,
        "--data_path",
        str(data_root / domain),
        "--output_dir",
        str(domain_output_dir),
        "--config",
        str(config_path),
        "--experiment_mode",
        experiment_mode,
        "--stage",
        stage,
        "--checkpoint_manifest",
        str(domain_manifest),
    ]
    if skip_official_eval:
        cmd.append("--skip_official_eval")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen Gov-Mem checkpoint benchmark suite from a single config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    runner_cfg = dict(config.get("runner") or {})
    if not runner_cfg:
        raise ValueError("Config is missing runner settings required by gov_mem.run_checkpoint_benchmark.")

    dataset_name = str(runner_cfg.get("dataset_name") or "checkpoint_benchmark")
    suite_manifest = (ROOT / str(runner_cfg["suite_manifest"])).resolve()
    data_root = (ROOT / str(runner_cfg["data_root"])).resolve()
    output_dir = (ROOT / str(runner_cfg["output_dir"])).resolve()
    stage = str(runner_cfg.get("stage") or "all")
    experiment_mode = str(runner_cfg.get("experiment_mode") or (config.get("experiment") or {}).get("mode") or "govmem_structured_old")
    skip_official_eval = bool(runner_cfg.get("skip_official_eval", False))

    payload = _load_manifest(suite_manifest)
    suite_name = str(payload.get("suite_name") or suite_manifest.stem)
    version = int(payload.get("version") or 1)
    grouped = _group_entries_by_domain(payload.get("entries", []))
    suite_summary: dict[str, object] = {
        "suite_name": suite_name,
        "version": version,
        "domains": {},
    }

    commands: list[list[str]] = []
    for domain, entries in grouped.items():
        domain_output_dir = output_dir / domain
        domain_manifest = _write_domain_manifest(
            suite_name=suite_name,
            version=version,
            domain=domain,
            entries=entries,
            output_dir=output_dir,
        )
        cmd = _build_command(
            dataset_name=dataset_name,
            data_root=data_root,
            domain=domain,
            domain_output_dir=domain_output_dir,
            config_path=config_path,
            experiment_mode=experiment_mode,
            stage=stage,
            domain_manifest=domain_manifest,
            skip_official_eval=skip_official_eval,
        )
        commands.append(cmd)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        suite_summary["domains"][domain] = {
            "n_entries": len(entries),
            "summary": _read_json(domain_output_dir / "official_eval" / dataset_name / domain / "summary.json"),
            "paper_metrics": _read_json(domain_output_dir / "official_eval" / dataset_name / domain / "paper_metrics.json"),
        }

    if args.dry_run:
        return

    (output_dir / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
