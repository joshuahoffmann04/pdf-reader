"""Small, deterministic post-processing for generated answers.

This is intentionally conservative: it fixes common formatting issues without
injecting new factual content or domain-specific assumptions.
"""

from __future__ import annotations

import re


def postprocess_answer(answer: str, query: str) -> str:
    text = (answer or "").strip()
    if not text:
        return text

    # Normalize common dash variants.
    text = text.replace(chr(0x2013), "-").replace(chr(0x2014), "-")

    # Collapse excessive whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text
