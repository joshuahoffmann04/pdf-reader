"""Evaluation metrics for chunking output."""

from __future__ import annotations

import re
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]+|\d+(?:[.,]\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def token_recall(reference: str, hypothesis: str) -> float:
    ref_tokens = set(_WORD_RE.findall(reference or ""))
    hyp_tokens = set(_WORD_RE.findall(hypothesis or ""))
    if not ref_tokens:
        return 1.0
    return len(ref_tokens & hyp_tokens) / max(len(ref_tokens), 1)


def number_recall(reference: str, hypothesis: str) -> float:
    ref_numbers = set(_NUMBER_RE.findall(reference or ""))
    hyp_numbers = set(_NUMBER_RE.findall(hypothesis or ""))
    if not ref_numbers:
        return 1.0
    return len(ref_numbers & hyp_numbers) / max(len(ref_numbers), 1)


def duplication_ratio(source_tokens: int, chunk_tokens: int) -> float:
    if source_tokens <= 0:
        return 1.0
    return chunk_tokens / max(source_tokens, 1)


def sentence_boundary_ratio(chunks: Iterable[str]) -> float:
    total = 0
    ok = 0
    for text in chunks:
        if not text:
            continue
        total += 1
        if re.search(r"[.!?]$", text.strip()):
            ok += 1
    if total == 0:
        return 1.0
    return ok / total
