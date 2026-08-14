"""Storage controls for high-concurrency experiment runs.

The helpers in this module inspect mount metadata only. They never enumerate
the contents of a directory, which is important when the project checkout is
on NFS.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def filesystem_type(path: str | Path) -> str:
    candidate = Path(path).resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    try:
        result = subprocess.run(
            ["findmnt", "-T", str(candidate), "-n", "-o", "FSTYPE"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def require_local_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    fs_type = filesystem_type(resolved)
    if fs_type.startswith("nfs") or fs_type in {"", "unknown"}:
        raise RuntimeError(
            f"Refusing high-concurrency Gov-Mem run: {label}={resolved} is on "
            f"{fs_type or 'unknown'}; use the local runtime root."
        )
    return resolved


def runtime_root(run_id: str) -> Path:
    configured = os.environ.get("GOVMEM_RUNTIME_ROOT", "").strip()
    base = Path(configured) if configured else Path(tempfile.gettempdir()) / "govmem-runtime"
    token = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    root = base / token
    root.mkdir(parents=True, exist_ok=True)
    require_local_path(root, label="runtime_root")
    return root


def configure_local_environment(root: Path) -> dict[str, str]:
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    values = {
        "TMPDIR": str(root / "tmp"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "datasets"),
        "TRANSFORMERS_CACHE": str(cache / "transformers"),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for value in values.values():
        if value.startswith("/"):
            Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(values)
    return values


def stage_explicit_dataset(source_root: Path, target_root: Path, domains: tuple[str, ...]) -> Path:
    """Copy only the two known GateMem JSONL files per requested domain."""
    source_root = Path(source_root).resolve()
    target_root = require_local_path(target_root, label="dataset_runtime_root")
    for domain in domains:
        source_dir = source_root / domain
        target_dir = target_root / domain
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("episodes.jsonl", "checkpoints.jsonl"):
            source = source_dir / name
            if not source.is_file():
                raise FileNotFoundError(f"Missing dataset file: {source}")
            shutil.copy2(source, target_dir / name)
    return target_root


def stage_runtime_code(project_root: Path, target_root: Path) -> Path:
    """Stage tracked runtime code using Git's index, without walking checkout."""
    project_root = Path(project_root).resolve()
    target_root = require_local_path(target_root, label="code_runtime_root")
    listing = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z", "--", "run_govmem.py", "src/gov_mem"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    items = [item for item in listing.split("\0") if item]
    storage_module = project_root / "src" / "gov_mem" / "utils" / "storage.py"
    if storage_module.exists() and str(storage_module.relative_to(project_root)) not in items:
        items.append(str(storage_module.relative_to(project_root)))
    # Keep newly developed runtime modules available before they are committed.
    # This explicit list preserves the no-checkout-walk invariant of staging.
    for relative in ("src/gov_mem/backbones/symbolic_evidence.py",):
        candidate = project_root / relative
        if candidate.exists() and relative not in items:
            items.append(relative)
    for item in items:
        if not item or item.endswith(".pyc") or "__pycache__" in item:
            continue
        source = project_root / item
        target = target_root / item
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return target_root


def stage_tracked_tree(project_root: Path, target_root: Path) -> Path:
    """Stage a Git repository from its index without traversing its checkout."""
    project_root = Path(project_root).resolve()
    target_root = require_local_path(target_root, label="tracked_runtime_root")
    listing = subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    for item in listing.split("\0"):
        if not item or item.endswith(".pyc") or "__pycache__" in item:
            continue
        source = project_root / item
        target = target_root / item
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return target_root


def storage_audit(paths: dict[str, str | Path]) -> dict[str, str]:
    result = {label: filesystem_type(path) for label, path in paths.items()}
    print("[Storage Audit]")
    for label, fs_type in result.items():
        print(f"{label}: {Path(paths[label]).resolve()} filesystem={fs_type}")
    return result
