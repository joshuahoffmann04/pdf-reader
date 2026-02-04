"""Lightweight, deterministic rerankers for retrieval hits.

The goal is not to replace BM25/embeddings, but to fix common failure modes
where question words (e.g. "welchen", "was") dominate BM25 or where RRF merges
pull in broadly similar but non-answer chunks.
"""

from __future__ import annotations

from typing import Iterable

from .bm25_index import tokenize
from .models import RetrievalHit


def _min_page_number(metadata: dict) -> int | None:
    raw = (metadata or {}).get("page_numbers")
    if raw is None:
        return None

    pages: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                pages.append(int(item))
            except Exception:
                continue
    elif isinstance(raw, str):
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                pages.append(int(part))
            except Exception:
                continue

    return min(pages) if pages else None


def _should_prefer_early_pages(query_tokens: set[str]) -> bool:
    # Heuristic: if a query contains module codes/numbers, page order is less helpful.
    if any(t.isdigit() for t in query_tokens):
        return False
    return True


def rerank_hits(query: str, hits: Iterable[RetrievalHit]) -> list[RetrievalHit]:
    """Rerank hits by lexical overlap with a query (plus a mild early-page bias)."""

    query_tokens = set(tokenize(query))
    hits_list = list(hits)
    if not query_tokens or not hits_list:
        return hits_list

    prefer_early = _should_prefer_early_pages(query_tokens)

    def overlap_count(q_tokens: set[str], h_tokens: set[str]) -> int:
        # Exact token overlap is often too strict for German inflection and PDF artifacts.
        # We therefore allow a conservative prefix match for longer tokens.
        matched: set[str] = set()
        for qt in q_tokens:
            if qt in h_tokens:
                matched.add(qt)
                continue
            if len(qt) < 5:
                continue
            for ht in h_tokens:
                if ht.startswith(qt) or qt.startswith(ht):
                    matched.add(qt)
                    break
        return len(matched)

    def key(hit: RetrievalHit) -> tuple[float, int, float, float]:
        hit_tokens = set(tokenize(hit.text))
        overlap = overlap_count(query_tokens, hit_tokens)
        ratio = overlap / max(len(query_tokens), 1)

        early_bonus = 0.0
        if prefer_early:
            min_page = _min_page_number(hit.metadata)
            if min_page is not None and min_page > 0:
                # Range: page=1 -> 1.0, page=10 -> 0.1
                early_bonus = 1.0 / float(min_page)
        return (ratio, overlap, early_bonus, float(hit.score))

    hits_list.sort(key=key, reverse=True)
    return hits_list
