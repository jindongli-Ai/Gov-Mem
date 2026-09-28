"""Lossless prompt transport and explicit application-supplied access policy.

No dataset lookup, role classifier, or inferred policy is performed here.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json


def application_policy(config: dict, domain: str | None) -> dict | None:
    setting = config.get("access_policy") or {}
    if not setting.get("enabled", False):
        return None
    text = (setting.get("by_domain") or {}).get(domain)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"No configured access policy for domain {domain!r}")
    return {"policy_id": "application_access_policy", "text": text,
            "source": setting.get("source", "application configuration"),
            "sha256": sha256(text.encode()).hexdigest()}


def pack_prompt(payload: dict) -> tuple[dict, dict]:
    """Intern repeated long strings, preserving all fields and exact characters."""
    counts = Counter()
    def count(value):
        if isinstance(value, str) and len(value) >= 80:
            counts[value] += 1
        elif isinstance(value, dict):
            for child in value.values():
                count(child)
        elif isinstance(value, list):
            for child in value:
                count(child)
    count(payload)
    references = {value: f"text_{i}" for i, (value, n) in enumerate(counts.items())
                  if n > 1}
    def encode(value):
        if isinstance(value, str) and value in references:
            return {"$text": references[value]}
        if isinstance(value, dict):
            return {key: encode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [encode(child) for child in value]
        return value
    packed = {"text_pool": {key: value for value, key in references.items()},
              "request": encode(payload)} if references else payload
    original_chars = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    packed_chars = len(json.dumps(packed, ensure_ascii=False, separators=(",", ":")))
    if packed_chars >= original_chars:
        packed, packed_chars = payload, original_chars
    return packed, {"original_chars": original_chars, "packed_chars": packed_chars,
                    "saved_chars": original_chars - packed_chars}


def unpack_prompt(packed: dict) -> dict:
    """Offline round-trip verification; runtime validation uses original input."""
    if "text_pool" not in packed or "request" not in packed:
        return deepcopy(packed)
    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"$text"}:
                return packed["text_pool"][value["$text"]]
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value
    return decode(packed["request"])
