"""
Tests for PDF Extractor data models.
"""

import pytest
from datetime import datetime

from pdf_extractor import (
    # Models
    DocumentContext,
    DocumentStructure,
    SectionLocation,
    PageScanResult,
    DetectedSection,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    ExtractionConfig,
    # Enums
    DocumentType,
    SectionType,
    Language,
    # Helpers
    Abbreviation,
)


class TestDetectedSection:
    """Tests for DetectedSection model."""

    def test_create(self):
        """Test creating a detected section."""
        section = DetectedSection(
            section_type=SectionType.PARAGRAPH,
            identifier="§ 1",
            title="Geltungsbereich",
        )
        assert section.section_type == SectionType.PARAGRAPH
        assert section.identifier == "§ 1"
        assert section.title == "Geltungsbereich"

    def test_hashable(self):
        """Test that DetectedSection is hashable (for set operations)."""
        section1 = DetectedSection(section_type=SectionType.PARAGRAPH, identifier="§ 1")
        section2 = DetectedSection(section_type=SectionType.PARAGRAPH, identifier="§ 1")
        section3 = DetectedSection(section_type=SectionType.PARAGRAPH, identifier="§ 2")

        # Same identifier should be equal
        assert section1 == section2
        assert hash(section1) == hash(section2)

        # Different identifier should not be equal
        assert section1 != section3

        # Can use in sets
        sections = {section1, section2, section3}
        assert len(sections) == 2


class TestPageScanResult:
    """Tests for PageScanResult model."""

    def test_create(self, sample_page_scan_result):
        """Test creating a page scan result."""
        assert sample_page_scan_result.page_number == 3
        assert len(sample_page_scan_result.sections) == 1
        assert sample_page_scan_result.is_empty is False

    def test_empty_page(self):
        """Test creating an empty page result."""
        result = PageScanResult(
            page_number=10,
            sections=[],
            is_empty=True,
            scan_notes="Leere Seite",
        )
        assert result.is_empty is True
        assert result.scan_notes == "Leere Seite"


class TestSectionLocation:
    """Tests for SectionLocation model."""

    def test_create(self, sample_section_location):
        """Test creating a section location."""
        assert sample_section_location.section_type == SectionType.PARAGRAPH
        assert sample_section_location.identifier == "§ 1"
        assert sample_section_location.pages == [3, 4]

    def test_computed_properties(self, sample_section_location):
        """Test computed properties."""
        assert sample_section_location.start_page == 3
        assert sample_section_location.end_page == 4
        assert sample_section_location.page_count == 2
        assert sample_section_location.display_name == "§ 1 Geltungsbereich"

    def test_display_name_variations(self):
        """Test display_name for different cases."""
        # With identifier only
        loc1 = SectionLocation(
            section_type=SectionType.PARAGRAPH,
            identifier="§ 5",
            title=None,
            pages=[10],
        )
        assert loc1.display_name == "§ 5"

        # With title only (preamble)
        loc2 = SectionLocation(
            section_type=SectionType.PREAMBLE,
            identifier=None,
            title="Deckblatt",
            pages=[1],
        )
        assert loc2.display_name == "Deckblatt"

        # Neither (preamble without title)
        loc3 = SectionLocation(
            section_type=SectionType.PREAMBLE,
            identifier=None,
            title=None,
            pages=[1],
        )
        assert loc3.display_name == "Präambel"


class TestDocumentStructure:
    """Tests for DocumentStructure model."""

    def test_create(self, sample_structure):
        """Test creating a document structure."""
        assert sample_structure.total_pages == 56
        assert sample_structure.has_preamble is True
        assert len(sample_structure.sections) == 4

    def test_get_section(self, sample_structure):
        """Test finding a section by identifier."""
        section = sample_structure.get_section("§ 1")
        assert section is not None
        assert section.identifier == "§ 1"

        # Non-existent section
        assert sample_structure.get_section("§ 99") is None

    def test_get_paragraphs(self, sample_structure):
        """Test getting all paragraphs."""
        paragraphs = sample_structure.get_paragraphs()
        assert len(paragraphs) == 2
        assert all(s.section_type == SectionType.PARAGRAPH for s in paragraphs)

    def test_get_anlagen(self, sample_structure):
        """Test getting all Anlagen."""
        anlagen = sample_structure.get_anlagen()
        assert len(anlagen) == 1
        assert anlagen[0].identifier == "Anlage 1"

    def test_get_preamble(self, sample_structure):
        """Test getting the preamble."""
        preamble = sample_structure.get_preamble()
        assert preamble is not None
        assert preamble.section_type == SectionType.PREAMBLE


class TestDocumentContext:
    """Tests for DocumentContext model."""

    def test_create_minimal(self):
        """Test creating context with minimal required fields."""
        context = DocumentContext(
            document_type=DocumentType.PRUEFUNGSORDNUNG,
            title="Test Document",
            institution="Test University",
        )
        assert context.title == "Test Document"
        assert context.document_type == DocumentType.PRUEFUNGSORDNUNG
        assert context.total_pages == 0  # Default

    def test_create_full(self, sample_context):
        """Test creating context with all fields."""
        assert sample_context.faculty == "Fachbereich Mathematik und Informatik"
        assert len(sample_context.chapters) == 4
        assert len(sample_context.abbreviations) == 2

    def test_get_abbreviation_dict(self, sample_context):
        """Test abbreviation dictionary generation."""
        abbrev_dict = sample_context.get_abbreviation_dict()
        assert abbrev_dict["AB"] == "Allgemeine Bestimmungen"
        assert abbrev_dict["LP"] == "Leistungspunkte"

    def test_serialization(self, sample_context):
        """Test JSON serialization."""
        data = sample_context.model_dump()
        assert data["title"] == sample_context.title
        assert data["document_type"] == "pruefungsordnung"

        # Reconstruct from dict
        reconstructed = DocumentContext(**data)
        assert reconstructed.title == sample_context.title


class TestStructureEntry:
    """Tests for StructureEntry model (legacy)."""

    def test_create_paragraph(self):
        """Test creating a paragraph entry."""
        entry = StructureEntry(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 1",
            section_title="Geltungsbereich",
            start_page=3,
            end_page=4,
        )
        assert entry.section_type == SectionType.PARAGRAPH
        assert entry.section_number == "§ 1"
        assert entry.pages == [3, 4]

    def test_page_validation(self):
        """Test that end_page must be >= start_page."""
        with pytest.raises(ValueError):
            StructureEntry(
                section_type=SectionType.PARAGRAPH,
                section_number="§ 1",
                start_page=5,
                end_page=3,  # Invalid: before start_page
            )


class TestExtractedSection:
    """Tests for ExtractedSection model."""

    def test_create_minimal(self):
        """Test creating section with minimal fields."""
        section = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 1",
            content="Test content",
        )
        assert section.section_number == "§ 1"
        assert section.content == "Test content"
        assert section.pages == []
        assert section.has_table is False

    def test_create_full(self, sample_section):
        """Test creating section with all fields."""
        assert sample_section.section_number == "§ 1"
        assert sample_section.chapter == "I. Allgemeines"
        assert "(1)" in sample_section.subsections
        assert sample_section.has_list is True

    def test_identifier_property(self):
        """Test identifier property."""
        # With number and title
        section = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 10",
            section_title="Module und Leistungspunkte",
            content="Content",
        )
        assert section.identifier == "§ 10 Module und Leistungspunkte"

        # With number only
        section2 = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 5",
            content="Content",
        )
        assert section2.identifier == "§ 5"

        # Preamble (no number)
        section3 = ExtractedSection(
            section_type=SectionType.PREAMBLE,
            content="Preamble content",
        )
        assert section3.identifier == "Präambel"

    def test_format_source_reference(self):
        """Test source reference generation."""
        section = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 10",
            section_title="Module",
            content="Content",
            pages=[12, 13, 14],
        )
        ref = section.format_source_reference("Prüfungsordnung Informatik")
        assert "Prüfungsordnung Informatik" in ref
        assert "§ 10" in ref
        assert "12-14" in ref

    def test_token_estimate(self):
        """Test token count estimation."""
        content = "A" * 400  # 400 chars -> ~100 tokens
        section = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 1",
            content=content,
        )
        assert section.token_estimate == 100

    def test_serialization(self, sample_section):
        """Test JSON serialization."""
        data = sample_section.model_dump()
        assert data["section_number"] == "§ 1"
        assert len(data["subsections"]) == 2


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_create_full(self, sample_result):
        """Test creating result with all fields."""
        assert sample_result.source_file == "test.pdf"
        assert len(sample_result.sections) == 4
        assert sample_result.processing_time_seconds == 10.5

    def test_get_section(self, sample_result):
        """Test getting section by number."""
        section = sample_result.get_section("§ 1")
        assert section is not None
        assert section.section_number == "§ 1"

        # Non-existent section
        assert sample_result.get_section("§ 99") is None

    def test_get_preamble(self, sample_result):
        """Test getting preamble section."""
        preamble = sample_result.get_preamble()
        assert preamble is not None
        assert preamble.section_type == SectionType.PREAMBLE

    def test_get_paragraphs(self, sample_result):
        """Test getting all paragraph sections."""
        paragraphs = sample_result.get_paragraphs()
        assert len(paragraphs) == 2
        assert all(s.section_type == SectionType.PARAGRAPH for s in paragraphs)

    def test_get_anlagen(self, sample_result):
        """Test getting all Anlage sections."""
        anlagen = sample_result.get_anlagen()
        assert len(anlagen) == 1
        assert anlagen[0].section_number == "Anlage 1"

    def test_get_statistics(self, sample_result):
        """Test statistics calculation."""
        stats = sample_result.get_statistics()
        assert stats["total_sections"] == 4
        assert stats["paragraphs"] == 2
        assert stats["anlagen"] == 1
        assert stats["has_preamble"] is True
        assert stats["sections_with_tables"] == 2

    def test_get_full_text(self, sample_result):
        """Test getting full content."""
        content = sample_result.get_full_text()
        assert "Inhaltsverzeichnis" in content or "Präambel" in content
        assert "Geltungsbereich" in content

    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        data = sample_result.to_dict()
        assert data["source_file"] == "test.pdf"
        assert isinstance(data["sections"], list)
        assert isinstance(data["context"], dict)

    def test_to_json(self, sample_result):
        """Test to_json method."""
        json_str = sample_result.to_json()
        assert isinstance(json_str, str)
        assert "test.pdf" in json_str


class TestExtractionConfig:
    """Tests for ExtractionConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExtractionConfig()
        assert config.model == "gpt-4o"
        assert config.max_retries == 3
        assert config.temperature == 0.0
        assert config.max_tokens == 4096

    def test_custom_values(self):
        """Test custom configuration."""
        config = ExtractionConfig(
            model="gpt-4o-mini",
            max_retries=5,
            temperature=0.1,
            max_tokens=8192,
        )
        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 5
        assert config.max_tokens == 8192


class TestAbbreviation:
    """Tests for Abbreviation model."""

    def test_create(self):
        """Test creating abbreviation."""
        abbrev = Abbreviation(short="LP", long="Leistungspunkte")
        assert abbrev.short == "LP"
        assert abbrev.long == "Leistungspunkte"


class TestEnums:
    """Tests for enum types."""

    def test_document_types(self):
        """Test all document types are accessible."""
        assert DocumentType.PRUEFUNGSORDNUNG.value == "pruefungsordnung"
        assert DocumentType.MODULHANDBUCH.value == "modulhandbuch"
        assert DocumentType.OTHER.value == "other"

    def test_section_types(self):
        """Test all section types are accessible."""
        assert SectionType.PREAMBLE.value == "preamble"
        assert SectionType.PARAGRAPH.value == "paragraph"
        assert SectionType.ANLAGE.value == "anlage"

    def test_language(self):
        """Test language enum."""
        assert Language.DE.value == "de"
        assert Language.EN.value == "en"
