"""
Tests for PDF Extractor data models.
"""

import pytest
from datetime import datetime

from pdf_extractor import (
    DocumentContext,
    ExtractedPage,
    ExtractionResult,
    SectionMarker,
    ProcessingConfig,
    DocumentType,
    ContentType,
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


class TestExtractedPage:
    """Tests for ExtractedPage model."""

    def test_create_minimal(self):
        """Test creating page with minimal fields."""
        page = ExtractedPage(
            page_number=1,
            content="Test content",
        )
        assert page.page_number == 1
        assert page.content == "Test content"
        assert page.sections == []
        assert page.has_table is False

    def test_create_full(self, sample_page):
        """Test creating page with all fields."""
        assert sample_page.page_number == 5
        assert len(sample_page.sections) == 2
        assert sample_page.sections[0].number == "§1"
        assert "(1)" in sample_page.paragraph_numbers

    def test_page_number_validation(self):
        """Test page number must be positive."""
        with pytest.raises(ValueError):
            ExtractedPage(page_number=0, content="Test")

    def test_serialization(self, sample_page):
        """Test JSON serialization."""
        data = sample_page.model_dump()
        assert data["page_number"] == 5
        assert len(data["sections"]) == 2

    def test_to_dict(self, sample_page):
        """Test to_dict method."""
        data = sample_page.to_dict()
        assert data["page_number"] == 5
        assert isinstance(data, dict)


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_create_minimal(self, sample_context):
        """Test creating result with minimal fields."""
        result = ExtractionResult(
            source_file="test.pdf",
            context=sample_context,
        )
        assert result.source_file == "test.pdf"
        assert result.pages == []
        assert result.errors == []

    def test_create_full(self, sample_result):
        """Test creating result with all fields."""
        assert sample_result.source_file == "test.pdf"
        assert len(sample_result.pages) == 3
        assert sample_result.processing_time_seconds == 10.5

    def test_get_page_stats(self, sample_result):
        """Test page statistics calculation."""
        stats = sample_result.get_page_stats()
        assert stats["total_pages"] == 3
        assert stats["pages_with_tables"] == 1
        assert stats["pages_with_lists"] == 0
        assert stats["avg_content_length"] > 0

    def test_get_page_stats_empty(self, sample_context):
        """Test stats with no pages."""
        result = ExtractionResult(
            source_file="test.pdf",
            context=sample_context,
        )
        stats = result.get_page_stats()
        assert stats["total_pages"] == 0
        assert stats["avg_content_length"] == 0

    def test_get_all_sections(self, sample_result):
        """Test getting all sections."""
        sections = sample_result.get_all_sections()
        assert len(sections) == 3
        assert sections[0].number == "§1"

    def test_get_full_content(self, sample_result):
        """Test getting full content."""
        content = sample_result.get_full_content()
        assert "Seite 1 Inhalt" in content
        assert "Seite 2 Inhalt" in content

    def test_to_dict(self, sample_result):
        """Test to_dict method."""
        data = sample_result.to_dict()
        assert data["source_file"] == "test.pdf"
        assert isinstance(data["pages"], list)
        assert isinstance(data["context"], dict)

    def test_to_json(self, sample_result):
        """Test to_json method."""
        json_str = sample_result.to_json()
        assert isinstance(json_str, str)
        assert "test.pdf" in json_str


class TestProcessingConfig:
    """Tests for ProcessingConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        assert config.model == "gpt-4o"
        assert config.max_retries == 3
        assert config.temperature == 0.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            model="gpt-4o-mini",
            max_retries=5,
            temperature=0.1,
        )
        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 5
        assert config.temperature == 0.1


class TestSectionMarker:
    """Tests for SectionMarker model."""

    def test_create_minimal(self):
        """Test creating section marker with minimal fields."""
        section = SectionMarker(number="§1")
        assert section.number == "§1"
        assert section.title is None
        assert section.level == 1

    def test_create_full(self):
        """Test creating section marker with all fields."""
        section = SectionMarker(
            number="§10",
            title="Module und Leistungspunkte",
            level=2,
            starts_on_page=True,
        )
        assert section.number == "§10"
        assert section.title == "Module und Leistungspunkte"
        assert section.level == 2


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

    def test_content_types(self):
        """Test all content types are accessible."""
        assert ContentType.SECTION.value == "section"
        assert ContentType.TABLE.value == "table"
        assert ContentType.DEFINITION.value == "definition"

    def test_language(self):
        """Test language enum."""
        assert Language.DE.value == "de"
        assert Language.EN.value == "en"
