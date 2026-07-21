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
    return json.loads(block)

