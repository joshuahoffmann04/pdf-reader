"""
Tests for prompt templates and helper functions.
"""

import pytest

from pdf_extractor.prompts import (
    STRUCTURE_SYSTEM,
    STRUCTURE_USER,
    SECTION_SYSTEM,
    SECTION_USER,
    HINT_SINGLE,
    HINT_FIRST,
    HINT_MIDDLE,
    HINT_LAST,
    get_structure_prompt,
    get_section_prompt,
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

    def test_structure_system_prompt(self):
        """Test structure analysis system prompt."""
        assert "OFFSET" in STRUCTURE_SYSTEM
        assert "PDF-Seite" in STRUCTURE_SYSTEM
        assert "PARAGRAPH" in STRUCTURE_SYSTEM
        assert "ANLAGE" in STRUCTURE_SYSTEM

    def test_structure_user_prompt(self):
        """Test structure analysis user prompt."""
        assert "{total_pages}" in STRUCTURE_USER
        assert "JSON" in STRUCTURE_USER
        assert "has_toc" in STRUCTURE_USER
        assert "page_offset" in STRUCTURE_USER

    def test_section_system_prompt(self):
        """Test section extraction system prompt."""
        assert "{context}" in SECTION_SYSTEM
        assert "TABELLEN" in SECTION_SYSTEM
        assert "PRÄZISION" in SECTION_SYSTEM

    def test_section_user_prompt(self):
        """Test section extraction user prompt."""
        assert "{section_id}" in SECTION_USER
        assert "{visible_pages}" in SECTION_USER
        assert "{hint}" in SECTION_USER

    def test_hints(self):
        """Test hint templates."""
        assert "ERSTE" in HINT_FIRST
        assert "MITTLERER" in HINT_MIDDLE
        assert "LETZTE" in HINT_LAST
        assert "vollständig" in HINT_SINGLE.lower()


class TestGetStructurePrompt:
    """Tests for get_structure_prompt function."""

    def test_returns_tuple(self):
        """Test returns tuple of prompts."""
        system, user = get_structure_prompt(50)

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_total_pages_inserted(self):
        """Test total pages is inserted into user prompt."""
        system, user = get_structure_prompt(42)

        assert "42" in user

    def test_system_prompt_unchanged(self):
        """Test system prompt is the constant."""
        system, user = get_structure_prompt(50)

        assert system == STRUCTURE_SYSTEM


class TestGetSectionPrompt:
    """Tests for get_section_prompt function."""

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
        system, user = get_section_prompt(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_context_in_system_prompt(self, sample_context, sample_entry):
        """Test context is included in system prompt."""
        system, user = get_section_prompt(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "Test PO" in system
        assert "Test Uni" in system

    def test_section_info_in_user_prompt(self, sample_context, sample_entry):
        """Test section info is included in user prompt."""
        system, user = get_section_prompt(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "§ 1" in user
        assert "Geltungsbereich" in user

    def test_single_section_hint(self, sample_context, sample_entry):
        """Test single section uses HINT_SINGLE."""
        system, user = get_section_prompt(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],  # Covers entire section
            is_continuation=False,
        )

        assert "vollständig" in user.lower()

    def test_first_part_hint(self, sample_context, sample_entry):
        """Test first part of sliding window."""
        system, user = get_section_prompt(
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
            end_page=60,
        )

        system, user = get_section_prompt(
            context=sample_context,
            entry=entry,
            visible_pages=[54, 55, 56, 57, 58],
            is_continuation=True,
            is_last=False,
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

        system, user = get_section_prompt(
            context=sample_context,
            entry=entry,
            visible_pages=[58, 59, 60],
            is_continuation=True,
            is_last=True,
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

        system, user = get_section_prompt(
            context=sample_context,
            entry=entry,
            visible_pages=[10],
        )

        assert "10" in user

    def test_visible_pages_formatted_range(self, sample_context, sample_entry):
        """Test page range is formatted correctly."""
        system, user = get_section_prompt(
            context=sample_context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "3 bis 5" in user

    def test_abbreviations_in_context(self, sample_entry):
        """Test abbreviations are included in context."""
        context = DocumentContext(
            document_type=DocumentType.PRUEFUNGSORDNUNG,
            title="Test PO",
            institution="Test Uni",
            abbreviations=[
                Abbreviation(short="LP", long="Leistungspunkte"),
                Abbreviation(short="AB", long="Allgemeine Bestimmungen"),
            ],
        )

        system, user = get_section_prompt(
            context=context,
            entry=sample_entry,
            visible_pages=[3, 4, 5],
        )

        assert "LP=Leistungspunkte" in system


class TestPromptQuality:
    """Tests for prompt quality and completeness."""

    def test_structure_prompt_mentions_offset(self):
        """Test structure prompt explains page offset."""
        assert "Offset" in STRUCTURE_SYSTEM or "offset" in STRUCTURE_SYSTEM.lower()
        assert "PDF-Seite" in STRUCTURE_SYSTEM

    def test_section_prompt_has_table_guidance(self):
        """Test section prompt has table conversion guidance."""
        assert "TABELLEN" in SECTION_SYSTEM

    def test_prompts_are_german(self):
        """Test prompts are in German."""
        german_words = ["Dokument", "Seite", "Abschnitt", "Inhalt"]
        combined = STRUCTURE_SYSTEM + SECTION_SYSTEM

        found = sum(1 for word in german_words if word in combined)
        assert found >= 2, "Prompts should contain German terminology"

    def test_json_format_instructions(self):
        """Test prompts include JSON format instructions."""
        assert "```json" in STRUCTURE_USER
        assert "```json" in SECTION_USER
