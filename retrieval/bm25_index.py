import re
from typing import Any

from rank_bm25 import BM25Okapi

from .models import RetrievalHit


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_HYPHEN_IN_WORD_RE = re.compile(r"(?<=[^\\W\\d_])-(?=[^\\W\\d_])", re.UNICODE)
_HYPHEN_NEWLINE_RE = re.compile(r"(?<=[^\\W\\d_])-\\s*\\n\\s*(?=[^\\W\\d_])", re.UNICODE)


_UMLAUT_MAP = str.maketrans(
    {
        "\u00e4": "ae",
        "\u00f6": "oe",
        "\u00fc": "ue",
        "\u00df": "ss",
        "\u00c4": "ae",
        "\u00d6": "oe",
        "\u00dc": "ue",
    }
)

_MOJIBAKE_MAP = {
    # Common mojibake variants (UTF-8 read as Latin-1)
    "\u00c3\u00a4": "ae",
    "\u00c3\u00b6": "oe",
    "\u00c3\u00bc": "ue",
    "\u00c3\u009f": "ss",
    "\u00c3\u0084": "ae",
    "\u00c3\u0096": "oe",
    "\u00c3\u009c": "ue",
}


# NOTE: BM25 is sensitive to high-frequency function words.
# We normalize umlauts and common OCR variants to ASCII to keep stopwords
# simple and robust across noisy PDF extractions.
_STOPWORDS = {
    # de articles / determiners
    "der",
    "die",
    "das",
    "den",
    "dem",
    "des",
    "ein",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    # de common function words
    "und",
    "oder",
    "sowie",
    "als",
    "im",
    "in",
    "auf",
    "an",
    "am",
    "zu",
    "zum",
    "zur",
    "von",
    "fuer",
    "mit",
    "ohne",
    "bei",
    "dass",
    "ist",
    "sind",
    "war",
    "waren",
    "hat",
    "haben",
    "wird",
    "werden",
    "kann",
    "koennen",
    "muss",
    "muessen",
    "nicht",
    "nur",
    "auch",
    "da",
    # de question words
    "was",
    "wer",
    "wen",
    "wem",
    "wessen",
    "welche",
    "welcher",
    "welches",
    "welchen",
    "wie",
    "viel",
    "viele",
    "wieviel",
    "wieviele",
    "wievielen",
    "wo",
    "wofuer",
    "wozu",
    "warum",
    "wieso",
    "weshalb",
    "wann",
    # en
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
}


def _normalize_token(token: str) -> str:
    text = (token or "").lower()
    for bad, good in _MOJIBAKE_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
    return text.translate(_UMLAUT_MAP)


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    cleaned = text or ""
    # Fix common PDF extraction artifacts (hyphenation inside words).
    cleaned = _HYPHEN_NEWLINE_RE.sub("", cleaned)
    cleaned = _HYPHEN_IN_WORD_RE.sub("", cleaned)
    for raw in _TOKEN_PATTERN.findall(cleaned):
        token = _normalize_token(raw)
        if not token or token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def tokenize(text: str) -> list[str]:
    """Public wrapper for the BM25 tokenizer (used by rerankers/tests)."""
    return _tokenize(text)


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
            if filters and not _match_filters(chunk, filters):
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


def _match_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
    metadata = chunk.get("metadata", {})
    for key, value in filters.items():
        if key == "document_id":
            if chunk.get("document_id") != value:
                return False
            continue
        if metadata.get(key) != value:
            return False
    return True
