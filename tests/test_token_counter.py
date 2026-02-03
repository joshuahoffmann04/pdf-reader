"""Tests for chunking.token_counter."""

from chunking.token_counter import count_tokens, count_tokens_batch


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_english(self):
        tokens = count_tokens("Hello world")
        assert tokens >= 2  # At least 2 tokens

    def test_simple_german(self):
        tokens = count_tokens("Dies ist ein Beispielsatz.")
        assert tokens >= 4

    def test_german_compound_word(self):
        tokens = count_tokens("Prüfungsordnung")
        assert tokens >= 1

    def test_long_german_text(self):
        text = (
            "Die Studien- und Prüfungsordnung regelt die studienbezogenen "
            "und prüfungsbezogenen Bestimmungen für den Monobachelorstudiengang "
            "Informatik mit dem Abschluss Bachelor of Science."
        )
        tokens = count_tokens(text)
        assert 20 < tokens < 100  # Reasonable range

    def test_special_characters(self):
        tokens = count_tokens("§ 5 Abs. 2 Nr. 3")
        assert tokens >= 3

    def test_returns_int(self):
        result = count_tokens("Test")
        assert isinstance(result, int)


class TestCountTokensBatch:
    def test_empty_list(self):
        assert count_tokens_batch([]) == []

    def test_single_item(self):
        result = count_tokens_batch(["Hello"])
        assert len(result) == 1
        assert result[0] >= 1

    def test_multiple_items(self):
        texts = ["Erster Satz.", "Zweiter Satz.", "Dritter Satz."]
        result = count_tokens_batch(texts)
        assert len(result) == 3
        assert all(isinstance(c, int) for c in result)
        assert all(c > 0 for c in result)

    def test_empty_string_in_batch(self):
        result = count_tokens_batch(["Text", "", "Mehr Text"])
        assert result[1] == 0
        assert result[0] > 0
        assert result[2] > 0

    def test_consistency_with_single(self):
        texts = ["Prüfungsordnung", "Leistungspunkte", "Bachelorarbeit"]
        batch = count_tokens_batch(texts)
        singles = [count_tokens(t) for t in texts]
        assert batch == singles
