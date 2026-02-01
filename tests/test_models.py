"""
Tests for PDF Extractor data models.
"""

import pytest
from datetime import datetime

from pdf_extractor import (
    DocumentContext,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    ExtractionConfig,
    DocumentType,
    SectionType,
    Language,
    Abbreviation,
)


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

    def test_to_dict(self, sample_context):
        """Test to_dict method."""
        data = sample_context.to_dict()
        assert data["title"] == sample_context.title
        assert isinstance(data, dict)


class TestStructureEntry:
    """Tests for StructureEntry model."""

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
        assert entry.page_count == 2

    def test_create_anlage(self):
        """Test creating an anlage entry."""
        entry = StructureEntry(
            section_type=SectionType.ANLAGE,
            section_number="Anlage 1",
            section_title="Studienverlaufsplan",
            start_page=50,
            end_page=55,
        )
        assert entry.section_type == SectionType.ANLAGE
        assert entry.page_count == 6
        assert entry.identifier == "Anlage 1"

    def test_create_overview(self):
        """Test creating an overview entry."""
        entry = StructureEntry(
            section_type=SectionType.OVERVIEW,
            section_number=None,
            section_title="Inhaltsverzeichnis",
            start_page=1,
            end_page=2,
        )
        assert entry.section_type == SectionType.OVERVIEW
        assert entry.identifier == "Übersicht"

    def test_page_validation(self):
        """Test that end_page must be >= start_page."""
        with pytest.raises(ValueError):
            StructureEntry(
                section_type=SectionType.PARAGRAPH,
                section_number="§ 1",
                start_page=5,
                end_page=3,  # Invalid: before start_page
            )

    def test_pages_property(self, sample_structure_entry):
        """Test pages property."""
        assert sample_structure_entry.pages == [3, 4]


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
        assert "(1)" in sample_section.paragraphs
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

        # Overview (no number)
        section3 = ExtractedSection(
            section_type=SectionType.OVERVIEW,
            content="Overview content",
        )
        assert section3.identifier == "Übersicht"

    def test_get_source_reference(self):
        """Test source reference generation."""
        section = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 10",
            section_title="Module",
            content="Content",
            pages=[12, 13, 14],
        )
        ref = section.get_source_reference("Prüfungsordnung Informatik")
        assert "Prüfungsordnung Informatik" in ref
        assert "§ 10" in ref
        assert "12-14" in ref

    def test_token_count_calculation(self):
        """Test auto-calculation of token count."""
        content = "A" * 400  # 400 chars -> ~100 tokens
        section = ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 1",
            content=content,
        )
        assert section.token_count == 100

    def test_serialization(self, sample_section):
        """Test JSON serialization."""
        data = sample_section.model_dump()
        assert data["section_number"] == "§ 1"
        assert len(data["paragraphs"]) == 2


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_create_minimal(self, sample_context):
        """Test creating result with minimal fields."""
        result = ExtractionResult(
            source_file="test.pdf",
            context=sample_context,
        )
        assert result.source_file == "test.pdf"
        assert result.sections == []
        assert result.errors == []

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

    def test_get_overview(self, sample_result):
        """Test getting overview section."""
        overview = sample_result.get_overview()
        assert overview is not None
        assert overview.section_type == SectionType.OVERVIEW

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

    def test_get_stats(self, sample_result):
        """Test statistics calculation."""
        stats = sample_result.get_stats()
        assert stats["total_sections"] == 4
        assert stats["paragraphs"] == 2
        assert stats["anlagen"] == 1
        assert stats["has_overview"] is True
        assert stats["sections_with_tables"] == 2

    def test_get_stats_empty(self, sample_context):
        """Test stats with no sections."""
        result = ExtractionResult(
            source_file="test.pdf",
            context=sample_context,
        )
        stats = result.get_stats()
        assert stats["total_sections"] == 0
        assert stats["avg_tokens_per_section"] == 0

    def test_get_full_content(self, sample_result):
        """Test getting full content."""
        content = sample_result.get_full_content()
        assert "Inhaltsverzeichnis" in content
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
        assert config.max_images_per_request == 5

    def test_custom_values(self):
        """Test custom configuration."""
        config = ExtractionConfig(
            model="gpt-4o-mini",
            max_retries=5,
            temperature=0.1,
            max_images_per_request=10,
        )
        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 5
        assert config.max_images_per_request == 10

    def test_max_images_validation(self):
        """Test max_images_per_request validation."""
        # Valid range
        config = ExtractionConfig(max_images_per_request=1)
        assert config.max_images_per_request == 1

        config = ExtractionConfig(max_images_per_request=20)
        assert config.max_images_per_request == 20

        # Invalid (too low)
        with pytest.raises(ValueError):
            ExtractionConfig(max_images_per_request=0)

        # Invalid (too high)
        with pytest.raises(ValueError):
            ExtractionConfig(max_images_per_request=21)


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
        assert SectionType.OVERVIEW.value == "overview"
        assert SectionType.PARAGRAPH.value == "paragraph"
        assert SectionType.ANLAGE.value == "anlage"

    def test_language(self):
        """Test language enum."""
        assert Language.DE.value == "de"
        assert Language.EN.value == "en"
