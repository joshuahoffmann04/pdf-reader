"""Metrics for evaluating MARley answers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable


_WORD_RE = re.compile(r"[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]+|\d+(?:[.,]\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

_UMLAUT_MAP = str.maketrans(
    {
        "\u00e4": "ae",
        "\u00f6": "oe",
        "\u00fc": "ue",
        "\u00df": "ss",
        "\u00c4": "ae",
        "\u00d6": "oe",
        "\u00dc": "ue",
        # Common mojibake variants (UTF-8 read as Latin-1)
        "\u00c3\u00a4": "ae",
        "\u00c3\u00b6": "oe",
        "\u00c3\u00bc": "ue",
        "\u00c3\u009f": "ss",
        "\u00c3\u0084": "ae",
        "\u00c3\u0096": "oe",
        "\u00c3\u009c": "ue",
    }
)

_STOPWORDS = {
    # de
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines", "einem",
    "und", "oder", "sowie", "als", "im", "in", "auf", "an", "am", "zu", "zum", "zur",
    "von", "fuer", "mit", "ohne", "bei", "dass", "ist", "sind", "wird", "werden", "kann",
    "koennen", "muss", "muessen", "nicht", "nur", "auch", "wie",
    "was", "wer", "welche", "welcher", "welches",
    # en
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def word_tokens(text: str) -> list[str]:
    return [(t.lower().translate(_UMLAUT_MAP)) for t in _WORD_RE.findall(text or "")]


def number_tokens(text: str) -> list[str]:
    return [t for t in _NUMBER_RE.findall(text or "")]


def content_tokens(text: str) -> list[str]:
    tokens = word_tokens(text)
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS and not t.isdigit()]


def token_recall(reference: str, hypothesis: str, *, content_only: bool = True) -> float:
    ref = set(content_tokens(reference) if content_only else word_tokens(reference))
    hyp = set(content_tokens(hypothesis) if content_only else word_tokens(hypothesis))
    if not ref:
        return 1.0
    return len(ref & hyp) / max(len(ref), 1)


def number_recall(reference: str, hypothesis: str) -> float:
    ref = set(number_tokens(reference))
    hyp = set(number_tokens(hypothesis))
    if not ref:
        return 1.0
    return len(ref & hyp) / max(len(ref), 1)


def contains_all_substrings(text: str, substrings: Iterable[str]) -> float:
    expected = [s for s in substrings if s]
    if not expected:
        return 1.0
    lower = (text or "").lower()
    return 1.0 if all(s.lower() in lower for s in expected) else 0.0


def citation_page_hit(citations: list[dict], reference_pages: list[int]) -> float:
    if not reference_pages:
        return 1.0
    ref = set(int(p) for p in reference_pages)
    for cite in citations or []:
        pages = cite.get("page_numbers") or []
        try:
            cited = {int(p) for p in pages}
        except Exception:
            continue
        if cited & ref:
            return 1.0
    return 0.0


@dataclass
class AnswerScore:
    token_recall: float
    number_recall: float
    semantic_similarity: float
    pass_: bool


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += float(a) * float(b)
        norm_a += float(a) * float(a)
        norm_b += float(b) * float(b)
    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 0.0:
        return 0.0
    return max(min(dot / denom, 1.0), -1.0)


def score_answer(
    expected: str,
    actual: str,
    *,
    semantic_similarity: float,
    semantic_similarity_min: float,
    number_recall_min: float,
) -> AnswerScore:
    exp = normalize_text(expected)
    act = normalize_text(actual)

    # For very short answers, be stricter.
    exp_tokens = content_tokens(exp)
    if len(exp_tokens) <= 3 and exp:
        # If expected is short, require substring match (case-insensitive).
        if exp.lower() in act.lower():
            tr = 1.0
        else:
            tr = token_recall(exp, act, content_only=True)
    else:
        tr = token_recall(exp, act, content_only=True)

    nr = number_recall(exp, act)
    return AnswerScore(
        token_recall=tr,
        number_recall=nr,
        semantic_similarity=semantic_similarity,
        pass_=(semantic_similarity >= semantic_similarity_min and nr >= number_recall_min),
    )


def quote_support_score(reference_quote: str, text: str) -> float:
    # Use content-token recall against the quote as a stable proxy.
    return token_recall(reference_quote, text, content_only=True)
