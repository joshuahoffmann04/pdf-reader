"""Citation normalization and selection utilities.

We accept model-provided citations when they point to chunks that are actually
in the provided context. If the model fails to cite well (common), we fall back
to a deterministic selection based on lexical overlap with the answer/query.
"""

from __future__ import annotations

from typing import Any

from .context_builder import parse_page_numbers
from retrieval.bm25_index import tokenize


def _snippet_from_text(text: str, *, needle: str = "", max_chars: int = 240) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if needle:
        low_text = text.lower()
        low_needle = needle.lower()
        pos = low_text.find(low_needle)
        if pos >= 0:
            start = max(pos - 80, 0)
            end = min(pos + len(needle) + 80, len(text))
            return text[start:end].strip()[:max_chars]

    return text[:max_chars]


def _overlap_count(a: str, b: str) -> int:
    a_tokens = set(tokenize(a))
    if not a_tokens:
        return 0
    b_tokens = set(tokenize(b))
    matched: set[str] = set()
    for at in a_tokens:
        if at in b_tokens:
            matched.add(at)
            continue
        if len(at) < 5:
            continue
        for bt in b_tokens:
            if bt.startswith(at) or at.startswith(bt):
                matched.add(at)
                break
    return len(matched)


def normalize_and_select_citations(
    raw: list[dict[str, Any]] | None,
    selected_hits: list[dict[str, Any]],
    *,
    answer: str,
    query: str,
    max_citations: int = 2,
) -> list[dict[str, Any]]:
    mapping = {hit.get("chunk_id"): hit for hit in selected_hits}
    citations: list[dict[str, Any]] = []

    # 1) Keep model citations that reference an actually provided chunk.
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id")
        if not chunk_id or chunk_id not in mapping:
            continue
        hit = mapping[chunk_id]
        metadata = hit.get("metadata", {}) or {}
        pages = parse_page_numbers(metadata)
        snippet = (item.get("snippet") or "").strip()
        if not snippet:
            snippet = _snippet_from_text(hit.get("text") or "", needle=answer)
        citations.append(
            {
                "chunk_id": chunk_id,
                "page_numbers": pages,
                "snippet": snippet,
                "score": hit.get("score"),
            }
        )

    # 2) If citations are empty OR do not overlap with the answer, pick best supporting hits.
    target_text = answer.strip() or query.strip()
    if selected_hits and (not citations or _overlap_count(target_text, citations[0].get("snippet", "")) == 0):
        ranked = sorted(
            selected_hits,
            key=lambda h: (
                _overlap_count(target_text, (h.get("text") or "")),
                float(h.get("score") or 0.0),
            ),
            reverse=True,
        )

        for hit in ranked:
            if len(citations) >= max_citations:
                break
            chunk_id = hit.get("chunk_id") or ""
            if not chunk_id:
                continue
            if any(c.get("chunk_id") == chunk_id for c in citations):
                continue
            metadata = hit.get("metadata", {}) or {}
            citations.append(
                {
                    "chunk_id": chunk_id,
                    "page_numbers": parse_page_numbers(metadata),
                    "snippet": _snippet_from_text(hit.get("text") or "", needle=answer),
                    "score": hit.get("score"),
                }
            )

    return citations[:max_citations]
