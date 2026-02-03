import re
from typing import Any

from rank_bm25 import BM25Okapi

from .models import RetrievalHit


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
        if not self._bm25 or not self._chunks:
            return []
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        hits = []
        for idx, score in enumerate(scores):
            chunk = self._chunks[idx]
            if filters and not _match_filters(chunk.get("metadata", {}), filters):
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=chunk["chunk_id"],
                    score=float(score),
                    text=chunk["text"],
                    metadata=chunk.get("metadata", {}),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


def _match_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())
