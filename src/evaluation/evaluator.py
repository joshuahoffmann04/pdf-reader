"""
Evaluation Module

Provides metrics for comparing extracted text against ground truth
or evaluating extraction quality.
"""

import re
from dataclasses import dataclass
from typing import Optional
from collections import Counter
import numpy as np


@dataclass
class EvaluationResult:
    """Results from text comparison evaluation."""
    cosine_similarity: float
    bleu_score: float
    word_overlap: float  # Jaccard similarity of words
    char_overlap: float  # Character-level overlap
    section_count_match: bool
    missing_sections: list[str]
    extra_sections: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "cosine_similarity": round(self.cosine_similarity, 4),
            "bleu_score": round(self.bleu_score, 4),
            "word_overlap": round(self.word_overlap, 4),
            "char_overlap": round(self.char_overlap, 4),
            "section_count_match": self.section_count_match,
            "missing_sections": self.missing_sections,
            "extra_sections": self.extra_sections
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=== Evaluation Results ===",
            f"Cosine Similarity: {self.cosine_similarity:.2%}",
            f"BLEU Score: {self.bleu_score:.2%}",
            f"Word Overlap (Jaccard): {self.word_overlap:.2%}",
            f"Character Overlap: {self.char_overlap:.2%}",
            f"Section Count Match: {'Yes' if self.section_count_match else 'No'}",
        ]

        if self.missing_sections:
            lines.append(f"Missing Sections: {', '.join(self.missing_sections)}")

        if self.extra_sections:
            lines.append(f"Extra Sections: {', '.join(self.extra_sections)}")

        return "\n".join(lines)


class Evaluator:
    """
    Evaluates the quality of PDF text extraction.

    Provides multiple metrics:
    - Cosine similarity (TF-IDF based)
    - BLEU score (n-gram precision)
    - Word overlap (Jaccard similarity)
    - Section completeness check
    """

    def __init__(self, language: str = "de"):
        """
        Initialize the evaluator.

        Args:
            language: Language for text processing ("de" or "en").
        """
        self.language = language

        # Simple stopwords for German and English
        self.stopwords = {
            "de": {"der", "die", "das", "und", "in", "zu", "den", "von", "ist",
                   "mit", "für", "auf", "dem", "des", "eine", "ein", "im", "nicht",
                   "sich", "als", "auch", "es", "an", "werden", "aus", "er", "hat",
                   "dass", "sie", "nach", "wird", "bei", "einer", "einem", "eines",
                   "noch", "zum", "war", "haben", "nur", "oder", "aber", "vor",
                   "zur", "bis", "mehr", "durch", "über", "kann", "keine", "kein"},
            "en": {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
                   "for", "of", "with", "by", "from", "is", "are", "was", "were",
                   "be", "been", "being", "have", "has", "had", "do", "does", "did",
                   "will", "would", "could", "should", "may", "might", "must",
                   "shall", "can", "this", "that", "these", "those", "it", "its"}
        }

    def evaluate(
        self,
        extracted_text: str,
        reference_text: str,
        extracted_sections: Optional[list[str]] = None,
        reference_sections: Optional[list[str]] = None
    ) -> EvaluationResult:
        """
        Evaluate extracted text against reference.

        Args:
            extracted_text: Text extracted from PDF.
            reference_text: Ground truth text.
            extracted_sections: List of section IDs in extracted text.
            reference_sections: List of section IDs in reference.

        Returns:
            EvaluationResult with all metrics.
        """
        # Normalize texts
        norm_extracted = self._normalize_text(extracted_text)
        norm_reference = self._normalize_text(reference_text)

        # Calculate metrics
        cosine_sim = self._cosine_similarity(norm_extracted, norm_reference)
        bleu = self._bleu_score(norm_extracted, norm_reference)
        word_overlap = self._jaccard_similarity(norm_extracted, norm_reference)
        char_overlap = self._character_overlap(extracted_text, reference_text)

        # Section comparison
        section_match = True
        missing = []
        extra = []

        if extracted_sections is not None and reference_sections is not None:
            extracted_set = set(extracted_sections)
            reference_set = set(reference_sections)

            missing = list(reference_set - extracted_set)
            extra = list(extracted_set - reference_set)
            section_match = len(missing) == 0 and len(extra) == 0

        return EvaluationResult(
            cosine_similarity=cosine_sim,
            bleu_score=bleu,
            word_overlap=word_overlap,
            char_overlap=char_overlap,
            section_count_match=section_match,
            missing_sections=missing,
            extra_sections=extra
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove punctuation (keep alphanumeric and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra spaces again
        text = " ".join(text.split())

        return text

    def _tokenize(self, text: str, remove_stopwords: bool = False) -> list[str]:
        """Tokenize text into words."""
        words = text.split()

        if remove_stopwords:
            stopwords = self.stopwords.get(self.language, set())
            words = [w for w in words if w not in stopwords]

        return words

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF vectors."""
        words1 = self._tokenize(text1, remove_stopwords=True)
        words2 = self._tokenize(text2, remove_stopwords=True)

        if not words1 or not words2:
            return 0.0

        # Build term frequency vectors
        tf1 = Counter(words1)
        tf2 = Counter(words2)

        # Get all unique terms
        all_terms = set(tf1.keys()) | set(tf2.keys())

        # Build vectors
        vec1 = np.array([tf1.get(term, 0) for term in all_terms])
        vec2 = np.array([tf2.get(term, 0) for term in all_terms])

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _bleu_score(
        self,
        candidate: str,
        reference: str,
        max_n: int = 4
    ) -> float:
        """
        Calculate BLEU score (simplified version).

        Uses n-gram precision with brevity penalty.
        """
        candidate_words = self._tokenize(candidate)
        reference_words = self._tokenize(reference)

        if not candidate_words or not reference_words:
            return 0.0

        # Calculate n-gram precisions
        precisions = []

        for n in range(1, max_n + 1):
            candidate_ngrams = self._get_ngrams(candidate_words, n)
            reference_ngrams = self._get_ngrams(reference_words, n)

            if not candidate_ngrams:
                precisions.append(0.0)
                continue

            # Count matching n-grams
            candidate_counts = Counter(candidate_ngrams)
            reference_counts = Counter(reference_ngrams)

            matches = 0
            total = 0

            for ngram, count in candidate_counts.items():
                matches += min(count, reference_counts.get(ngram, 0))
                total += count

            precision = matches / total if total > 0 else 0.0
            precisions.append(precision)

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            log_precision = sum(np.log(p) for p in precisions) / len(precisions)
            geo_mean = np.exp(log_precision)
        else:
            # If any precision is 0, use a smoothed version
            smoothed = [(p + 0.01) for p in precisions]
            log_precision = sum(np.log(p) for p in smoothed) / len(smoothed)
            geo_mean = np.exp(log_precision)

        # Brevity penalty
        c = len(candidate_words)
        r = len(reference_words)

        if c > r:
            brevity_penalty = 1.0
        else:
            brevity_penalty = np.exp(1 - r / c) if c > 0 else 0.0

        return float(brevity_penalty * geo_mean)

    def _get_ngrams(self, words: list[str], n: int) -> list[tuple]:
        """Generate n-grams from a list of words."""
        if len(words) < n:
            return []

        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity (word overlap)."""
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _character_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate character-level overlap using longest common subsequence ratio.

        This is a simplified version that computes character set overlap.
        """
        # Normalize whitespace but keep case for character comparison
        chars1 = set(text1.lower().replace(" ", ""))
        chars2 = set(text2.lower().replace(" ", ""))

        if not chars1 or not chars2:
            return 0.0

        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)

        return intersection / union if union > 0 else 0.0

    def compare_section_content(
        self,
        section_id: str,
        extracted_content: str,
        reference_content: str
    ) -> dict:
        """
        Compare content of a specific section.

        Returns detailed metrics for the section.
        """
        result = self.evaluate(extracted_content, reference_content)

        return {
            "section_id": section_id,
            "metrics": result.to_dict(),
            "extracted_length": len(extracted_content),
            "reference_length": len(reference_content),
            "length_ratio": len(extracted_content) / len(reference_content) if reference_content else 0
        }
