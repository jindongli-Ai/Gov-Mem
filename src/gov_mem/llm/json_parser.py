from __future__ import annotations

import json


def extract_json_block(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Cannot parse empty LLM response as JSON.")

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return text


def parse_json_response(text: str):
    block = extract_json_block(text)
    try:
        return json.loads(block)
    except json.JSONDecodeError as exc:
        # Providers sometimes emit literal tabs/newlines inside a quoted value.
        # strict=False preserves those exact characters; it does not fix quotes,
        # commas, missing structure, escapes, or truncated output.
        if not exc.msg.startswith("Invalid control character"):
            raise
        return json.loads(block, strict=False)

