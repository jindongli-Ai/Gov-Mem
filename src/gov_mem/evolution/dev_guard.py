"""Explicit provenance gate for artifacts that may influence future runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_DEV_ATTESTATION = {
    "artifact_scope": "development",
    "allows_adaptation": True,
    "test_data_used": False,
}


def load_dev_attestation(path: str | Path) -> dict[str, Any]:
    """Accept only an explicit development-only attestation.

    A label in an output directory is not enough: updates can later affect
    runtime, so the producing workflow must declare its data scope in a
    separate immutable input artifact.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise ValueError(f"Missing development attestation: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid development attestation: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Development attestation must be a JSON object")
    for key, expected in REQUIRED_DEV_ATTESTATION.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Development attestation must declare {key}={expected!r}; got {payload.get(key)!r}"
            )
    source_runs = payload.get("source_runs")
    if not isinstance(source_runs, list) or not all(str(item).strip() for item in source_runs):
        raise ValueError("Development attestation requires non-empty source_runs")
    return {
        **payload,
        "attestation_path": str(resolved.resolve()),
        "source_runs": [str(item) for item in source_runs],
    }


def require_matching_attestation(*, artifacts: list[dict[str, Any]], attestation: dict[str, Any]) -> None:
    """Reject legacy or mismatched analysis artifacts before they become updates."""
    expected_path = str(attestation.get("attestation_path") or "")
    for artifact in artifacts:
        provenance = dict(artifact.get("provenance") or {})
        recorded = dict(provenance.get("dev_attestation") or {})
        if str(recorded.get("attestation_path") or "") != expected_path:
            raise ValueError(
                "Every analysis artifact must carry the same explicit development attestation before it can update runtime."
            )


def has_valid_embedded_dev_attestation(artifact: dict[str, Any]) -> bool:
    """Return whether a frozen runtime artifact carries dev-only provenance."""
    metadata = dict(artifact.get("metadata") or {})
    provenance = dict(artifact.get("provenance") or {})
    attestation = dict(metadata.get("dev_attestation") or provenance.get("dev_attestation") or {})
    return all(attestation.get(key) == expected for key, expected in REQUIRED_DEV_ATTESTATION.items())
