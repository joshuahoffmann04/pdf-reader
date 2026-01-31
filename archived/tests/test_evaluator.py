"""
Tests for the Evaluator module.

Tests cover:
- Text normalization
- Tokenization
- Similarity metrics (Cosine, BLEU, Jaccard)
- Section comparison
- EvaluationResult dataclass
"""

import pytest
from src.evaluation import Evaluator, EvaluationResult


class TestEvaluationResult:
    """Tests for the EvaluationResult dataclass."""

    def test_creation(self):
        """Test basic creation of EvaluationResult."""
        result = EvaluationResult(
            cosine_similarity=0.95,
            bleu_score=0.80,
            word_overlap=0.85,
            char_overlap=0.90,
            section_count_match=True,
            missing_sections=[],
            extra_sections=[]
        )
        assert result.cosine_similarity == 0.95
        assert result.bleu_score == 0.80

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = EvaluationResult(
            cosine_similarity=0.9567,
            bleu_score=0.8123,
            word_overlap=0.7891,
            char_overlap=0.9012,
            section_count_match=False,
            missing_sections=["§1", "§2"],
            extra_sections=["§99"]
        )
        d = result.to_dict()

        # Should round to 4 decimal places
        assert d["cosine_similarity"] == 0.9567
        assert d["bleu_score"] == 0.8123
        assert d["word_overlap"] == 0.7891
        assert d["char_overlap"] == 0.9012
        assert d["section_count_match"] is False
        assert d["missing_sections"] == ["§1", "§2"]
        assert d["extra_sections"] == ["§99"]

    def test_summary(self):
        """Test human-readable summary."""
        result = EvaluationResult(
            cosine_similarity=0.95,
            bleu_score=0.80,
            word_overlap=0.85,
            char_overlap=0.90,
            section_count_match=True,
            missing_sections=[],
            extra_sections=[]
        )
        summary = result.summary()

        assert "95" in summary  # 95%
        assert "80" in summary  # 80%
        assert "Cosine Similarity" in summary
        assert "BLEU Score" in summary
        assert "Word Overlap" in summary
        assert "Section Count Match: Yes" in summary

    def test_summary_with_missing_sections(self):
        """Test summary shows missing sections."""
        result = EvaluationResult(
            cosine_similarity=0.90,
            bleu_score=0.75,
            word_overlap=0.80,
            char_overlap=0.85,
            section_count_match=False,
            missing_sections=["§5", "§10"],
            extra_sections=[]
        )
        summary = result.summary()

        assert "Missing Sections: §5, §10" in summary
        assert "Section Count Match: No" in summary


class TestEvaluator:
    """Tests for the Evaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create an Evaluator instance."""
        return Evaluator(language="de")

    @pytest.fixture
    def evaluator_en(self):
        """Create an English Evaluator instance."""
        return Evaluator(language="en")

    def test_initialization(self, evaluator):
        """Test default initialization."""
        assert evaluator.language == "de"
        assert "der" in evaluator.stopwords["de"]
        assert "the" in evaluator.stopwords["en"]

    def test_normalize_text_lowercase(self, evaluator):
        """Test that normalization lowercases text."""
        text = "Hello WORLD Test"
        normalized = evaluator._normalize_text(text)
        assert normalized == "hello world test"

    def test_normalize_text_whitespace(self, evaluator):
        """Test that normalization handles whitespace."""
        text = "Hello    World\n\nTest"
        normalized = evaluator._normalize_text(text)
        assert normalized == "hello world test"

    def test_normalize_text_punctuation(self, evaluator):
        """Test that normalization removes punctuation."""
        text = "Hello, World! Test?"
        normalized = evaluator._normalize_text(text)
        assert normalized == "hello world test"

    def test_tokenize_basic(self, evaluator):
        """Test basic tokenization."""
        text = "hello world test"
        tokens = evaluator._tokenize(text)
        assert tokens == ["hello", "world", "test"]

    def test_tokenize_with_stopwords(self, evaluator):
        """Test tokenization with stopword removal."""
        text = "der hund ist groß"  # "der" and "ist" are stopwords
        tokens = evaluator._tokenize(text, remove_stopwords=True)
        assert "der" not in tokens
        assert "ist" not in tokens
        assert "hund" in tokens
        assert "groß" in tokens

    def test_tokenize_english_stopwords(self, evaluator_en):
        """Test English stopword removal."""
        text = "the dog is big"
        tokens = evaluator_en._tokenize(text, remove_stopwords=True)
        assert "the" not in tokens
        assert "is" not in tokens
        assert "dog" in tokens
        assert "big" in tokens

    # Cosine Similarity Tests

    def test_cosine_similarity_identical(self, evaluator):
        """Test cosine similarity of identical texts."""
        text = "der hund läuft im park"
        sim = evaluator._cosine_similarity(text, text)
        assert sim == pytest.approx(1.0, rel=0.01)

    def test_cosine_similarity_different(self, evaluator):
        """Test cosine similarity of different texts."""
        text1 = "der hund läuft im park"
        text2 = "die katze schläft im haus"
        sim = evaluator._cosine_similarity(text1, text2)
        # May be zero if no content words overlap after stopword removal
        assert 0.0 <= sim < 0.5

    def test_cosine_similarity_empty(self, evaluator):
        """Test cosine similarity with empty text."""
        sim = evaluator._cosine_similarity("", "hello")
        assert sim == 0.0

    def test_cosine_similarity_similar(self, evaluator):
        """Test cosine similarity of similar texts."""
        text1 = "der student studiert informatik an der universität"
        text2 = "der student lernt informatik an der hochschule"
        sim = evaluator._cosine_similarity(text1, text2)
        assert sim >= 0.5  # Should be fairly similar

    # BLEU Score Tests

    def test_bleu_score_identical(self, evaluator):
        """Test BLEU score of identical texts."""
        text = "der hund läuft schnell"
        score = evaluator._bleu_score(text, text)
        assert score > 0.9

    def test_bleu_score_different(self, evaluator):
        """Test BLEU score of different texts."""
        text1 = "der hund läuft schnell"
        text2 = "die katze schläft lange"
        score = evaluator._bleu_score(text1, text2)
        assert score < 0.3

    def test_bleu_score_empty(self, evaluator):
        """Test BLEU score with empty text."""
        score = evaluator._bleu_score("", "hello world")
        assert score == 0.0

    def test_bleu_score_partial_overlap(self, evaluator):
        """Test BLEU score with partial overlap."""
        text1 = "der hund läuft im park"
        text2 = "der hund spielt im garten"
        score = evaluator._bleu_score(text1, text2)
        # BLEU with n-gram precision tends to be low for partial overlap
        # Just verify it's between 0 and 1
        assert 0.0 <= score < 0.5

    # N-gram Tests

    def test_get_ngrams_unigrams(self, evaluator):
        """Test unigram generation."""
        words = ["a", "b", "c", "d"]
        ngrams = evaluator._get_ngrams(words, 1)
        assert ngrams == [("a",), ("b",), ("c",), ("d",)]

    def test_get_ngrams_bigrams(self, evaluator):
        """Test bigram generation."""
        words = ["a", "b", "c", "d"]
        ngrams = evaluator._get_ngrams(words, 2)
        assert ngrams == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_get_ngrams_trigrams(self, evaluator):
        """Test trigram generation."""
        words = ["a", "b", "c", "d"]
        ngrams = evaluator._get_ngrams(words, 3)
        assert ngrams == [("a", "b", "c"), ("b", "c", "d")]

    def test_get_ngrams_too_short(self, evaluator):
        """Test n-gram generation with text shorter than n."""
        words = ["a", "b"]
        ngrams = evaluator._get_ngrams(words, 5)
        assert ngrams == []

    # Jaccard Similarity Tests

    def test_jaccard_similarity_identical(self, evaluator):
        """Test Jaccard similarity of identical texts."""
        text = "der hund läuft"
        sim = evaluator._jaccard_similarity(text, text)
        assert sim == 1.0

    def test_jaccard_similarity_no_overlap(self, evaluator):
        """Test Jaccard similarity with no word overlap."""
        text1 = "hund katze maus"
        text2 = "auto bus zug"
        sim = evaluator._jaccard_similarity(text1, text2)
        assert sim == 0.0

    def test_jaccard_similarity_partial(self, evaluator):
        """Test Jaccard similarity with partial overlap."""
        text1 = "hund katze maus"
        text2 = "hund vogel fisch"
        sim = evaluator._jaccard_similarity(text1, text2)
        # 1 overlap (hund), 5 unique words
        assert sim == pytest.approx(1/5, rel=0.01)

    def test_jaccard_similarity_empty(self, evaluator):
        """Test Jaccard similarity with empty text."""
        sim = evaluator._jaccard_similarity("", "hello")
        assert sim == 0.0

    # Character Overlap Tests

    def test_char_overlap_identical(self, evaluator):
        """Test character overlap of identical texts."""
        text = "hello"
        overlap = evaluator._character_overlap(text, text)
        assert overlap == 1.0

    def test_char_overlap_different(self, evaluator):
        """Test character overlap of different texts."""
        text1 = "abc"
        text2 = "xyz"
        overlap = evaluator._character_overlap(text1, text2)
        assert overlap == 0.0

    def test_char_overlap_partial(self, evaluator):
        """Test character overlap with some common characters."""
        text1 = "abc"
        text2 = "bcd"
        overlap = evaluator._character_overlap(text1, text2)
        # {a,b,c} and {b,c,d} - intersection: {b,c}, union: {a,b,c,d}
        assert overlap == pytest.approx(2/4, rel=0.01)

    # Full Evaluation Tests

    def test_evaluate_identical_texts(self, evaluator):
        """Test full evaluation of identical texts."""
        text = "Dies ist ein Beispieltext für die Evaluation."
        result = evaluator.evaluate(text, text)

        assert result.cosine_similarity > 0.99
        assert result.word_overlap > 0.99
        assert result.char_overlap > 0.99

    def test_evaluate_similar_texts(self, evaluator):
        """Test evaluation of similar texts."""
        text1 = "Der Student studiert Informatik an der Universität Darmstadt."
        text2 = "Der Student lernt Informatik an der TU Darmstadt."

        result = evaluator.evaluate(text1, text2)

        assert result.cosine_similarity > 0.5
        assert result.word_overlap > 0.3
        assert result.char_overlap > 0.7

    def test_evaluate_with_sections_match(self, evaluator):
        """Test evaluation with matching sections."""
        result = evaluator.evaluate(
            "text 1",
            "text 2",
            extracted_sections=["§1", "§2", "§3"],
            reference_sections=["§1", "§2", "§3"]
        )

        assert result.section_count_match is True
        assert result.missing_sections == []
        assert result.extra_sections == []

    def test_evaluate_with_sections_mismatch(self, evaluator):
        """Test evaluation with section mismatch."""
        result = evaluator.evaluate(
            "text 1",
            "text 2",
            extracted_sections=["§1", "§2", "§5"],
            reference_sections=["§1", "§2", "§3", "§4"]
        )

        assert result.section_count_match is False
        assert "§3" in result.missing_sections
        assert "§4" in result.missing_sections
        assert "§5" in result.extra_sections

    # Section Content Comparison Tests

    def test_compare_section_content(self, evaluator):
        """Test section content comparison."""
        extracted = "Der Geltungsbereich dieser Ordnung umfasst den Studiengang Informatik."
        reference = "Der Geltungsbereich dieser Ordnung umfasst den Studiengang Informatik BSc."

        result = evaluator.compare_section_content("§1", extracted, reference)

        assert result["section_id"] == "§1"
        assert "metrics" in result
        assert result["extracted_length"] == len(extracted)
        assert result["reference_length"] == len(reference)
        assert "length_ratio" in result


class TestEvaluatorIntegration:
    """Integration tests for Evaluator with real data."""

    def test_high_similarity_detection(self):
        """Test that evaluator correctly identifies high similarity."""
        evaluator = Evaluator()

        # Simulate extracted vs reference with minor differences
        reference = """
        § 1 Geltungsbereich
        Diese Ordnung regelt das Studium und die Prüfungen im Bachelorstudiengang
        Informatik an der Technischen Universität Darmstadt.
        """

        extracted = """
        § 1 Geltungsbereich
        Diese Ordnung regelt das Studium und die Prüfung im Bachelorstudiengang
        Informatik an der Technischen Universität Darmstadt.
        """

        result = evaluator.evaluate(extracted, reference)

        # Should detect high similarity despite minor difference
        assert result.cosine_similarity > 0.9
        assert result.word_overlap > 0.85

    def test_low_similarity_detection(self):
        """Test that evaluator correctly identifies low similarity."""
        evaluator = Evaluator()

        reference = "Der Hund läuft schnell durch den Park."
        extracted = "Die Katze schläft lange auf dem Sofa."

        result = evaluator.evaluate(extracted, reference)

        # Should detect low similarity
        assert result.cosine_similarity < 0.5
        assert result.word_overlap < 0.3
