"""
Tests for prompt templates and helper functions.
"""

import pytest

from src.llm_processor.prompts import (
    CONTEXT_ANALYSIS_SYSTEM,
    CONTEXT_ANALYSIS_USER,
    PAGE_EXTRACTION_SYSTEM,
    PAGE_EXTRACTION_USER,
    build_context_string,
    get_page_extraction_system_prompt,
    get_page_extraction_user_prompt,
)


class TestPromptTemplates:
    """Tests for prompt template strings."""

    def test_context_analysis_system_prompt(self):
        """Test context analysis system prompt."""
        assert "Prüfungsordnungen" in CONTEXT_ANALYSIS_SYSTEM
        assert "Modulhandbücher" in CONTEXT_ANALYSIS_SYSTEM
        assert "KRITISCHE REGELN" in CONTEXT_ANALYSIS_SYSTEM

    def test_context_analysis_user_prompt(self):
        """Test context analysis user prompt."""
        assert "JSON" in CONTEXT_ANALYSIS_USER
        assert "document_type" in CONTEXT_ANALYSIS_USER
        assert "chapters" in CONTEXT_ANALYSIS_USER
        assert "abbreviations" in CONTEXT_ANALYSIS_USER

    def test_page_extraction_system_prompt(self):
        """Test page extraction system prompt."""
        assert "{document_context}" in PAGE_EXTRACTION_SYSTEM
        assert "TABELLEN" in PAGE_EXTRACTION_SYSTEM
        assert "PRÄZISION" in PAGE_EXTRACTION_SYSTEM
        assert "PARAGRAPHEN" in PAGE_EXTRACTION_SYSTEM

    def test_page_extraction_user_prompt(self):
        """Test page extraction user prompt."""
        assert "{page_number}" in PAGE_EXTRACTION_USER
        assert "{total_pages}" in PAGE_EXTRACTION_USER
        assert "section_numbers" in PAGE_EXTRACTION_USER
        assert "paragraph_numbers" in PAGE_EXTRACTION_USER


class TestBuildContextString:
    """Tests for build_context_string function."""

    def test_minimal_context(self):
        """Test with minimal context."""
        context = {
            "document_type": "pruefungsordnung",
            "title": "Test Title",
            "institution": "Test Uni",
        }

        result = build_context_string(context)

        assert "Dokumenttyp: pruefungsordnung" in result
        assert "Titel: Test Title" in result
        assert "Institution: Test Uni" in result

    def test_full_context(self):
        """Test with full context."""
        context = {
            "document_type": "pruefungsordnung",
            "title": "Test Title",
            "institution": "Test Uni",
            "faculty": "Test Faculty",
            "degree_program": "Test Program",
            "abbreviations": {"AB": "Test AB", "LP": "Test LP"},
            "chapters": ["Chapter 1", "Chapter 2", "Chapter 3"],
        }

        result = build_context_string(context)

        assert "Fachbereich: Test Faculty" in result
        assert "Studiengang: Test Program" in result
        assert "Abkürzungen:" in result
        assert "AB=Test AB" in result
        assert "Gliederung:" in result

    def test_missing_optional_fields(self):
        """Test with missing optional fields."""
        context = {
            "document_type": "other",
        }

        result = build_context_string(context)

        # Should use defaults
        assert "Dokumenttyp: other" in result
        assert "Titel: Unbekannt" in result


class TestGetPageExtractionSystemPrompt:
    """Tests for get_page_extraction_system_prompt function."""

    def test_basic_prompt(self):
        """Test basic prompt generation."""
        context = {
            "document_type": "pruefungsordnung",
            "title": "Test",
            "institution": "Test Uni",
        }

        prompt = get_page_extraction_system_prompt(context)

        assert "Dokumenttyp: pruefungsordnung" in prompt
        assert "KRITISCHE REGELN" in prompt
        assert "TABELLEN" in prompt

    def test_with_abbreviations(self):
        """Test prompt includes abbreviations."""
        context = {
            "document_type": "pruefungsordnung",
            "title": "Test",
            "institution": "Test Uni",
            "abbreviations": {"LP": "Leistungspunkte"},
        }

        prompt = get_page_extraction_system_prompt(context)

        assert "LP=Leistungspunkte" in prompt


class TestGetPageExtractionUserPrompt:
    """Tests for get_page_extraction_user_prompt function."""

    def test_page_numbers(self):
        """Test page numbers are inserted correctly."""
        prompt = get_page_extraction_user_prompt(5, 50)

        assert "Seite 5 von 50" in prompt

    def test_first_page(self):
        """Test first page prompt."""
        prompt = get_page_extraction_user_prompt(1, 10)

        assert "Seite 1 von 10" in prompt

    def test_last_page(self):
        """Test last page prompt."""
        prompt = get_page_extraction_user_prompt(10, 10)

        assert "Seite 10 von 10" in prompt

    def test_contains_json_format(self):
        """Test prompt contains JSON format instructions."""
        prompt = get_page_extraction_user_prompt(1, 1)

        assert "```json" in prompt
        assert "section_numbers" in prompt
        assert "paragraph_numbers" in prompt


class TestPromptQuality:
    """Tests for prompt quality and completeness."""

    def test_context_prompt_has_examples(self):
        """Test context prompt has example values."""
        assert "Teil I" in CONTEXT_ANALYSIS_USER or "Allgemeines" in CONTEXT_ANALYSIS_USER
        assert "LP" in CONTEXT_ANALYSIS_USER
        assert "Anlage 1" in CONTEXT_ANALYSIS_USER  # Should have Anlagen examples

    def test_page_prompt_has_table_guidance(self):
        """Test page prompt has table conversion guidance."""
        assert "Notentabelle" in PAGE_EXTRACTION_SYSTEM or "TABELLEN" in PAGE_EXTRACTION_SYSTEM

    def test_page_prompt_has_section_rules(self):
        """Test page prompt has clear section detection rules."""
        # Should explain when to detect sections
        assert "BEGINNEN" in PAGE_EXTRACTION_SYSTEM or "beginnt" in PAGE_EXTRACTION_SYSTEM

    def test_page_prompt_mentions_anlagen(self):
        """Test page prompt mentions Anlagen."""
        assert "Anlagen" in PAGE_EXTRACTION_SYSTEM or "Anlage" in PAGE_EXTRACTION_SYSTEM

    def test_prompts_are_german(self):
        """Test prompts are in German."""
        german_words = ["Dokument", "Seite", "Paragraph", "Abschnitt"]
        combined = CONTEXT_ANALYSIS_SYSTEM + PAGE_EXTRACTION_SYSTEM

        found = sum(1 for word in german_words if word in combined)
        assert found >= 2, "Prompts should contain German terminology"
