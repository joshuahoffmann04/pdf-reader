"""Evaluation metrics for PDF extraction output."""

from __future__ import annotations

from typing import Iterable, Sequence
import re


_WORD_RE = re.compile(r"[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]+|\d+(?:[.,]\d+)?")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_text(text: str) -> str:
    """Normalize whitespace for fair distance comparisons."""
    text = text or ""
    return re.sub(r"\s+", " ", text).strip()


def levenshtein_distance(seq1: Sequence, seq2: Sequence) -> int:
    """Compute Levenshtein distance between two sequences."""
    if seq1 == seq2:
        return 0
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    prev = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, start=1):
        curr = [i]
        for j, b in enumerate(seq2, start=1):
            cost = 0 if a == b else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            ))
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    dist = levenshtein_distance(list(ref), list(hyp))
    return dist / max(len(ref), 1)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    ref_words = ref.split()
    hyp_words = hyp.split()
    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / max(len(ref_words), 1)


def token_recall(reference: str, hypothesis: str) -> float:
    """Recall over unique word and number tokens."""
    ref_tokens = set(_WORD_RE.findall(reference or ""))
    hyp_tokens = set(_WORD_RE.findall(hypothesis or ""))
    if not ref_tokens:
        return 1.0
    return len(ref_tokens & hyp_tokens) / max(len(ref_tokens), 1)


def number_recall(reference: str, hypothesis: str) -> float:
    """Recall over numeric tokens."""
    ref_numbers = set(_NUMBER_RE.findall(reference or ""))
    hyp_numbers = set(_NUMBER_RE.findall(hypothesis or ""))
    if not ref_numbers:
        return 1.0
    return len(ref_numbers & hyp_numbers) / max(len(ref_numbers), 1)


def precision_recall_f1(reference_items: Iterable[str], hypothesis_items: Iterable[str]) -> dict:
    """Precision/recall/F1 for set-like collections."""
    ref = {item for item in reference_items if item}
    hyp = {item for item in hypothesis_items if item}

    if not ref and not hyp:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not hyp:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not ref:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    tp = len(ref & hyp)
    precision = tp / max(len(hyp), 1)
    recall = tp / max(len(ref), 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def diff_sets(reference_items: Iterable[str], hypothesis_items: Iterable[str]) -> dict:
    """Return missing/extra items for diagnostics."""
    ref = {item for item in reference_items if item}
    hyp = {item for item in hypothesis_items if item}
    return {
        "missing": sorted(ref - hyp),
        "extra": sorted(hyp - ref),
    }
