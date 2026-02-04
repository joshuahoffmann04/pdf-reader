"""Evaluation metrics for generation output."""

from __future__ import annotations

from typing import Iterable


def contains_all(text: str, expected: Iterable[str]) -> float:
    expected_list = [item for item in expected if item]
    if not expected_list:
        return 1.0
    hay = (text or "").lower()
    return 1.0 if all(item.lower() in hay for item in expected_list) else 0.0


def citations_include_chunk_ids(citations: list[dict], expected_ids: Iterable[str]) -> float:
    expected = {item for item in expected_ids if item}
    if not expected:
        return 1.0
    found = {c.get("chunk_id") for c in citations if c.get("chunk_id")}
    return 1.0 if expected.issubset(found) else 0.0


def citations_include_pages(citations: list[dict], expected_pages: Iterable[int]) -> float:
    expected = {int(p) for p in expected_pages if str(p).isdigit()}
    if not expected:
        return 1.0
    pages = set()
    for c in citations:
        for p in c.get("page_numbers", []) or []:
            if isinstance(p, int) or str(p).isdigit():
                pages.add(int(p))
    return 1.0 if expected.issubset(pages) else 0.0


def missing_info_ok(missing_info: str, expected_missing: bool | None) -> float:
    if expected_missing is None:
        return 1.0
    if expected_missing:
        return 1.0 if (missing_info or "").strip() else 0.0
    return 1.0 if not (missing_info or "").strip() else 0.0
