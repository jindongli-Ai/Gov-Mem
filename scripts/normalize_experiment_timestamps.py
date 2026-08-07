#!/usr/bin/env python3
"""Normalize experiment artifact names to second-resolution timestamps.

This is intentionally limited to experiment artifacts.  Dataset identifiers,
checkpoint IDs, and explicit ``seed``/``hashseed`` values are not timestamps
and are preserved verbatim.

The default mode is a dry run.  Use ``--apply`` to rename artifacts and update
textual references outside the immutable output trees.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SUITES = ROOT / "experiments" / "gatemem_suites"
RESULTS = ROOT / "experiments" / "result"

CANONICAL = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})(?!\d)")
LEGACY_HYPHEN_DATE = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?![-\d])")
COMPACT = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d-%H-%M-%S")


def _is_seed_token(name: str, start: int) -> bool:
    prefix = name[:start].lower()
    return prefix.endswith("seed") or prefix.endswith("hashseed")


def _strip_run_timestamp(name: str) -> tuple[str, str | None]:
    """Return (name without run timestamp, embedded timestamp if available)."""
    match = CANONICAL.search(name)
    if match:
        base = name[: match.start()] + name[match.end() :]
        return re.sub(r"_+", "_", base).strip("_-"), match.group(1)

    match = LEGACY_HYPHEN_DATE.search(name)
    if match:
        base = name[: match.start()] + name[match.end() :]
        return re.sub(r"_+", "_", base).strip("_-"), None

    match = COMPACT.search(name)
    while match:
        if not _is_seed_token(name, match.start()):
            # A compact date may be followed by an old-style time component:
            # YYYYMMDD_HHMMSS, YYYYMMDD_HHMM, or YYYYMMDD_XXX.
            end = match.end()
            time_match = re.match(r"[_-]\d{3,6}(?=[_-]|$)", name[end:])
            if time_match:
                end += time_match.end()
            base = name[: match.start()] + name[end:]
            return re.sub(r"_+", "_", base).strip("_-"), None
        next_start = match.end()
        match = COMPACT.search(name, next_start)

    return name, None


def _new_name(path: Path) -> str | None:
    stem = path.stem if path.is_file() else path.name
    base, embedded = _strip_run_timestamp(stem)
    if base == stem and embedded is None:
        return None
    timestamp = embedded or _stamp(path.stat().st_mtime)
    suffix = path.suffix if path.is_file() else ""
    return f"{timestamp}_{base}{suffix}"


def _candidates() -> list[Path]:
    candidates: list[Path] = []
    if OUTPUTS.exists():
        candidates.extend(
            path
            for path in OUTPUTS.iterdir()
            if path.is_dir() and _new_name(path) is not None
        )
    for directory in (SUITES, RESULTS):
        if directory.exists():
            candidates.extend(
                path
                for path in directory.iterdir()
                if path.is_file() and _new_name(path) is not None
            )
    return sorted(candidates)


def _doc_candidates() -> list[Path]:
    # These are the two dated research reports and the stable handoff file.
    # Their contents are retained; only their artifact names become sortable.
    names = {"STATEFUL_POLICY_REWRITE_REPORT.md", "UTILITY_BOTTLENECK_ANALYSIS_20260727.md", "report.txt"}
    return [path for path in (ROOT / name for name in names) if path.exists()]


def _plan() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in _candidates():
        target = path.with_name(_new_name(path))
        if target == path:
            continue
        if target.exists():
            raise FileExistsError(f"Timestamp migration target already exists: {target}")
        mapping[str(path.relative_to(ROOT))] = str(target.relative_to(ROOT))

    for path in _doc_candidates():
        timestamp = _stamp(path.stat().st_mtime)
        if path.name == "UTILITY_BOTTLENECK_ANALYSIS_20260727.md":
            target_name = f"{timestamp}_UTILITY_BOTTLENECK_ANALYSIS.md"
        elif path.name == "STATEFUL_POLICY_REWRITE_REPORT.md":
            target_name = f"{timestamp}_STATEFUL_POLICY_REWRITE_REPORT.md"
        else:
            target_name = f"{timestamp}_report.txt"
        target = path.with_name(target_name)
        if target.exists():
            raise FileExistsError(f"Timestamp migration target already exists: {target}")
        mapping[str(path.relative_to(ROOT))] = str(target.relative_to(ROOT))
    return mapping


def _rewrite_references(mapping: dict[str, str]) -> int:
    # Do not rewrite immutable run outputs: they are the original audit record.
    ignored = {".git", "dataset", "third_party", "outputs", "cache", "tmp", "work"}
    replacements = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored for part in path.relative_to(ROOT).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        updated = text
        # Longer names must be replaced first: for example, a run ending in
        # ``_r3`` must not be partially matched by the base run name.
        for old, new in sorted(mapping.items(), key=lambda item: len(Path(item[0]).name), reverse=True):
            old_name = Path(old).name
            new_name = Path(new).name
            if old in updated:
                updated = updated.replace(old, new)
            if old_name in updated:
                updated = updated.replace(old_name, new_name)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            replacements += 1
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the planned renames and reference updates.")
    args = parser.parse_args()
    mapping = _plan()
    print(f"planned_renames={len(mapping)}")
    for old, new in mapping.items():
        print(f"{old} -> {new}")
    if not args.apply:
        return

    # Rename deepest-independent entries only; all targets are direct children
    # of their original parent, so ordering cannot invalidate another source.
    for old, new in mapping.items():
        os.replace(ROOT / old, ROOT / new)
    rewritten = _rewrite_references(mapping)
    print(f"reference_files_rewritten={rewritten}")


if __name__ == "__main__":
    main()
