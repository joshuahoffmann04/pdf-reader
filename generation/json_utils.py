"""JSON parsing helpers.

Local LLMs sometimes wrap JSON in extra text or emit partial JSON. These
helpers extract the first top-level JSON object and parse it safely.
"""

from __future__ import annotations

import json
from typing import Any


def extract_first_json_object(text: str) -> str:
    """Return the first balanced {...} JSON object found in the text.

    If no object can be extracted, the original text is returned.
    """
    if not text:
        return ""

    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unbalanced JSON; return original for best-effort debugging.
    return text


def safe_parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from a model response.

    Strategy:
    1) Try full parse.
    2) Extract first JSON object and parse again.
    3) Return {} on failure.
    """
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    extracted = extract_first_json_object(text)
    try:
        data = json.loads(extracted)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

