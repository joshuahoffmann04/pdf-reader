"""
Tests for prompt templates and helper functions.
"""

import pytest

from pdf_extractor.prompts import (
    STRUCTURE_ANALYSIS_SYSTEM,
    STRUCTURE_ANALYSIS_USER,
    SECTION_EXTRACTION_SYSTEM,
    SECTION_EXTRACTION_USER,
    CONTINUATION_HINT_START,
    CONTINUATION_HINT_MIDDLE,
    CONTINUATION_HINT_END,
    CONTINUATION_HINT_SINGLE,
    build_context_string,
    build_document_context_string,
    get_structure_analysis_prompt,
    get_section_extraction_prompts,
)
from pdf_extractor import (
    DocumentContext,
    StructureEntry,
    DocumentType,
    SectionType,
    Abbreviation,
)


class TestPromptTemplates:
    """Tests for prompt template strings."""

    def test_structure_analysis_system_prompt(self):
        """Test structure analysis system prompt."""
        assert "Inhaltsverzeichnis" in STRUCTURE_ANALYSIS_SYSTEM
        assert "PARAGRAPH" in STRUCTURE_ANALYSIS_SYSTEM
        assert "ANLAGE" in STRUCTURE_ANALYSIS_SYSTEM
        assert "OVERVIEW" in STRUCTURE_ANALYSIS_SYSTEM

    def test_structure_analysis_user_prompt(self):
        """Test structure analysis user prompt."""
        assert "{total_pages}" in STRUCTURE_ANALYSIS_USER
        assert "JSON" in STRUCTURE_ANALYSIS_USER
        assert "has_toc" in STRUCTURE_ANALYSIS_USER
        assert "structure" in STRUCTURE_ANALYSIS_USER

    def test_section_extraction_system_prompt(self):
        """Test section extraction system prompt."""
        assert "{document_context}" in SECTION_EXTRACTION_SYSTEM
        assert "TABELLEN" in SECTION_EXTRACTION_SYSTEM
        assert "PRÄZISION" in SECTION_EXTRACTION_SYSTEM
        assert "Abschnitt" in SECTION_EXTRACTION_SYSTEM

    def test_section_extraction_user_prompt(self):
        """Test section extraction user prompt."""
        assert "{section_identifier}" in SECTION_EXTRACTION_USER
        assert "{section_type}" in SECTION_EXTRACTION_USER
        assert "{visible_pages}" in SECTION_EXTRACTION_USER
        assert "{continuation_hint}" in SECTION_EXTRACTION_USER

    def test_continuation_hints(self):
        """Test continuation hint templates."""
        assert "ERSTE" in CONTINUATION_HINT_START
        assert "MITTLERER" in CONTINUATION_HINT_MIDDLE
        assert "LETZTE" in CONTINUATION_HINT_END
        assert "vollständig" in CONTINUATION_HINT_SINGLE.lower()


class TestBuildContextString:
    """Tests for build_context_string function (legacy)."""

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
            "abbreviations": [
                {"short": "AB", "long": "Test AB"},
                {"short": "LP", "long": "Test LP"},
            ],
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


class TestBuildDocumentContextString:
    """Tests for build_document_context_string function."""

    def test_minimal_context(self):
        """Test with minimal DocumentContext."""
        context = DocumentContext(
            document_type=DocumentType.PRUEFUNGSORDNUNG,
            title="Test Title",
            institution="Test Uni",
        )

        result = build_document_context_string(context)

        assert "Dokumenttyp: pruefungsordnung" in result
        assert "Titel: Test Title" in result
        assert "Institution: Test Uni" in result

    def test_full_context(self):
        """Test with full DocumentContext."""
        context = DocumentContext(
            document_type=DocumentType.PRUEFUNGSORDNUNG,
            title="Test Title",
            institution="Test Uni",
            faculty="Test Faculty",
            degree_program="Test Program",
            abbreviations=[
                Abbreviation(short="AB", long="Test AB"),
                Abbreviation(short="LP", long="Test LP"),
            ],
            chapters=["Chapter 1", "Chapter 2", "Chapter 3"],
        )

        result = build_document_context_string(context)

        assert "Fachbereich: Test Faculty" in result
        assert "Studiengang: Test Program" in result
        assert "Abkürzungen:" in result
        assert "AB=Test AB" in result
        assert "Gliederung:" in result


class TestGetStructureAnalysisPrompt:
    """Tests for get_structure_analysis_prompt function."""

    def test_returns_tuple(self):
        """Test returns tuple of prompts."""
        system, user = get_structure_analysis_prompt(50)

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_total_pages_inserted(self):
        """Test total pages is inserted into user prompt."""
        system, user = get_structure_analysis_prompt(42)

        # Should appear multiple times
        assert "42" in user

    def test_system_prompt_unchanged(self):
        """Test system prompt is the constant."""
        system, user = get_structure_analysis_prompt(50)

        assert system == STRUCTURE_ANALYSIS_SYSTEM


class TestGetSectionExtractionPrompts:
    """Tests for get_section_extraction_prompts function."""

    @pytest.fixture
    def sample_context(self):
        """Create sample DocumentContext."""
        return DocumentContext(
            document_type=DocumentType.PRUEFUNGSORDNUNG,
            title="Test PO",
            institution="Test Uni",
        )

    @pytest.fixture
    def sample_entry(self):
        """Create sample StructureEntry."""
        return StructureEntry(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 1",
            section_title="Geltungsbereich",
            start_page=3,
            end_page=5,
        )

    def test_returns_tuple(self, sample_context, sample_entry):
        """Test returns tuple of prompts."""
        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_context_in_system_prompt(self, sample_context, sample_entry):
        """Test context is included in system prompt."""
        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "Test PO" in system
        assert "Test Uni" in system

    def test_section_info_in_user_prompt(self, sample_context, sample_entry):
        """Test section info is included in user prompt."""
        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "§ 1" in user
        assert "Geltungsbereich" in user
        assert "paragraph" in user

    def test_single_section_hint(self, sample_context, sample_entry):
        """Test single section uses CONTINUATION_HINT_SINGLE."""
        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],  # Covers entire section
            is_continuation=False,
        )

        # Should contain the single hint
        assert "vollständig" in user.lower() or "alle Seiten" in user

    def test_first_part_hint(self, sample_context, sample_entry):
        """Test first part of sliding window."""
        # Entry spans pages 3-5, but we're only showing 3-4
        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4],  # Only first 2 pages
            is_continuation=False,
        )

        assert "ERSTE" in user

    def test_middle_part_hint(self, sample_context):
        """Test middle part of sliding window."""
        entry = StructureEntry(
            section_type=SectionType.ANLAGE,
            section_number="Anlage 1",
            section_title="Studienverlaufsplan",
            start_page=50,
            end_page=60,  # Long section
        )

        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=entry,
            visible_pages=[54, 55, 56, 57, 58],
            is_continuation=True,
            is_final_part=False,
            overlap_page=54,
        )

        assert "MITTLERER" in user

    def test_last_part_hint(self, sample_context):
        """Test last part of sliding window."""
        entry = StructureEntry(
            section_type=SectionType.ANLAGE,
            section_number="Anlage 1",
            section_title="Studienverlaufsplan",
            start_page=50,
            end_page=60,
        )

        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=entry,
            visible_pages=[58, 59, 60],
            is_continuation=True,
            is_final_part=True,
            overlap_page=58,
        )

        assert "LETZTE" in user

    def test_visible_pages_formatted_single(self, sample_context):
        """Test single page is formatted correctly."""
        entry = StructureEntry(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 5",
            start_page=10,
            end_page=10,
        )

        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=entry,
            visible_pages=[10],
        )

        assert "10" in user

    def test_visible_pages_formatted_range(self, sample_context, sample_entry):
        """Test page range is formatted correctly."""
        system, user = get_section_extraction_prompts(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "3 bis 5" in user


class TestPromptQuality:
    """Tests for prompt quality and completeness."""

    def test_structure_prompt_has_examples(self):
        """Test structure prompt has example values."""
        assert "§ 1" in STRUCTURE_ANALYSIS_USER or "section_number" in STRUCTURE_ANALYSIS_USER
        assert "Anlage 1" in STRUCTURE_ANALYSIS_USER

    def test_section_prompt_has_table_guidance(self):
        """Test section prompt has table conversion guidance."""
        assert "Notentabelle" in SECTION_EXTRACTION_SYSTEM or "TABELLEN" in SECTION_EXTRACTION_SYSTEM

    def test_section_prompt_mentions_anlagen(self):
        """Test section prompt handles Anlagen."""
        combined = SECTION_EXTRACTION_SYSTEM + SECTION_EXTRACTION_USER
        assert "Anlage" in combined

    def test_prompts_are_german(self):
        """Test prompts are in German."""
        german_words = ["Dokument", "Seite", "Abschnitt", "Inhalt"]
        combined = STRUCTURE_ANALYSIS_SYSTEM + SECTION_EXTRACTION_SYSTEM

        found = sum(1 for word in german_words if word in combined)
        assert found >= 2, "Prompts should contain German terminology"

    def test_json_format_instructions(self):
        """Test prompts include JSON format instructions."""
        assert "```json" in STRUCTURE_ANALYSIS_USER
        assert "```json" in SECTION_EXTRACTION_USER
