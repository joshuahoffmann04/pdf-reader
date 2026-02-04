"""Evaluation metrics for retrieval."""

from __future__ import annotations

from typing import Iterable


def hit_at_k(ranked_ids: list[str], expected_ids: Iterable[str]) -> float:
    expected = {eid for eid in expected_ids if eid}
    if not expected:
        return 1.0
    return 1.0 if any(rid in expected for rid in ranked_ids) else 0.0


def mrr(ranked_ids: list[str], expected_ids: Iterable[str]) -> float:
    expected = {eid for eid in expected_ids if eid}
    if not expected:
        return 1.0
    for idx, rid in enumerate(ranked_ids, start=1):
        if rid in expected:
            return 1.0 / idx
    return 0.0


def text_contains_hit(texts: list[str], expected_substrings: Iterable[str]) -> float:
    expected = [s for s in expected_substrings if s]
    if not expected:
        return 1.0
    combined = " ".join(texts).lower()
    return 1.0 if all(s.lower() in combined for s in expected) else 0.0
