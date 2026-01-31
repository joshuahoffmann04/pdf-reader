"""
Tests for data models.
"""

import pytest
from datetime import datetime

from src.llm_processor.models import (
    DocumentContext,
    ExtractedPage,
    SectionMarker,
    RAGChunk,
    ChunkMetadata,
    ProcessingConfig,
    ProcessingResult,
    DocumentType,
    ChunkType,
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


class TestRAGChunk:
    """Tests for RAGChunk model."""

    def test_create_chunk(self, sample_chunk):
        """Test creating a RAG chunk."""
        assert sample_chunk.id == "pruefungsordnung-10-abc123"
        assert "§10" in sample_chunk.text
        assert sample_chunk.metadata.section_number == "§10"

    def test_to_langchain_document(self, sample_chunk):
        """Test LangChain export format."""
        lc_doc = sample_chunk.to_langchain_document()
        assert "page_content" in lc_doc
        assert "metadata" in lc_doc
        assert lc_doc["metadata"]["section"] == "§10"
        assert lc_doc["metadata"]["doc_type"] == "pruefungsordnung"

    def test_to_llamaindex_node(self, sample_chunk):
        """Test LlamaIndex export format."""
        li_node = sample_chunk.to_llamaindex_node()
        assert "id_" in li_node
        assert "text" in li_node
        assert li_node["id_"] == sample_chunk.id
        assert "excluded_embed_metadata_keys" in li_node

    def test_to_haystack_document(self, sample_chunk):
        """Test Haystack export format."""
        hs_doc = sample_chunk.to_haystack_document()
        assert "id" in hs_doc
        assert "content" in hs_doc
        assert "meta" in hs_doc
        assert hs_doc["meta"]["section"] == "§10"

    def test_to_jsonl_entry(self, sample_chunk):
        """Test JSONL export format."""
        jsonl = sample_chunk.to_jsonl_entry()
        assert isinstance(jsonl, str)
        assert "pruefungsordnung-10-abc123" in jsonl
        assert "§10" in jsonl

    def test_to_embedding_input(self, sample_chunk):
        """Test embedding input generation."""
        embedding_text = sample_chunk.to_embedding_input()
        assert "§10" in embedding_text
        assert "Module und Leistungspunkte" in embedding_text

    def test_id_validation(self):
        """Test chunk ID validation."""
        metadata = ChunkMetadata(
            source_document="test",
            source_pages=[1],
            document_type=DocumentType.OTHER,
        )
        # Valid ID
        chunk = RAGChunk(id="valid-id_123.test", text="Test", metadata=metadata)
        assert chunk.id == "valid-id_123.test"

        # Invalid ID (special characters)
        with pytest.raises(ValueError):
            RAGChunk(id="invalid id!", text="Test", metadata=metadata)


class TestProcessingConfig:
    """Tests for ProcessingConfig model."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        assert config.model == "gpt-4o"
        assert config.target_chunk_size == 500
        assert config.max_chunk_size == 1000
        assert config.temperature == 0.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = ProcessingConfig(
            model="gpt-4o-mini",
            target_chunk_size=800,
            max_chunk_size=1600,
        )
        assert config.model == "gpt-4o-mini"
        assert config.target_chunk_size == 800

    def test_chunk_size_validation(self):
        """Test max_chunk_size must be >= target_chunk_size."""
        with pytest.raises(ValueError):
            ProcessingConfig(
                target_chunk_size=1000,
                max_chunk_size=500,  # Invalid: smaller than target
            )


class TestProcessingResult:
    """Tests for ProcessingResult model."""

    def test_get_chunk_stats_empty(self):
        """Test stats with no chunks."""
        result = ProcessingResult(
            source_file="test.pdf",
            context=DocumentContext(
                document_type=DocumentType.OTHER,
                title="Test",
                institution="Test",
            ),
        )
        stats = result.get_chunk_stats()
        assert stats["total_chunks"] == 0
        assert stats["avg_length"] == 0

    def test_get_chunk_stats(self, sample_chunk, sample_context):
        """Test stats calculation."""
        result = ProcessingResult(
            source_file="test.pdf",
            context=sample_context,
            chunks=[sample_chunk, sample_chunk],
        )
        stats = result.get_chunk_stats()
        assert stats["total_chunks"] == 2
        assert stats["avg_length"] > 0
        assert "section" in stats["by_type"]


class TestEnums:
    """Tests for enum types."""

    def test_document_types(self):
        """Test all document types are accessible."""
        assert DocumentType.PRUEFUNGSORDNUNG.value == "pruefungsordnung"
        assert DocumentType.MODULHANDBUCH.value == "modulhandbuch"
        assert DocumentType.OTHER.value == "other"

    def test_chunk_types(self):
        """Test all chunk types are accessible."""
        assert ChunkType.SECTION.value == "section"
        assert ChunkType.TABLE.value == "table"
        assert ChunkType.DEFINITION.value == "definition"

    def test_language(self):
        """Test language enum."""
        assert Language.DE.value == "de"
        assert Language.EN.value == "en"
