"""
Tests for ChunkGenerator.
"""

import pytest
import tempfile
import json
from pathlib import Path

from src.llm_processor.chunk_generator import ChunkGenerator, ChunkingStats
from src.llm_processor.vision_processor import VisionProcessorResult
from src.llm_processor.models import (
    ProcessingConfig,
    DocumentContext,
    ExtractedPage,
    SectionMarker,
    DocumentType,
    ChunkType,
)


class TestChunkGenerator:
    """Tests for ChunkGenerator class."""

    @pytest.fixture
    def generator(self, sample_config):
        """Create a ChunkGenerator instance."""
        return ChunkGenerator(config=sample_config)

    @pytest.fixture
    def mock_extraction_result(self, sample_context, sample_pages):
        """Create a mock extraction result."""
        return VisionProcessorResult(
            context=sample_context,
            pages=sample_pages,
            processing_time_seconds=10.0,
            total_input_tokens=1000,
            total_output_tokens=500,
            errors=[],
        )

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = ChunkGenerator()
        assert generator.config.model == "gpt-4o"

    def test_init_custom_config(self, sample_config):
        """Test initialization with custom config."""
        generator = ChunkGenerator(config=sample_config)
        assert generator.config.target_chunk_size == 500

    def test_generate_from_extraction(self, generator, mock_extraction_result):
        """Test full chunk generation pipeline."""
        result = generator.generate_from_extraction(
            mock_extraction_result,
            "test-document",
        )

        # Should have chunks
        assert len(result.chunks) > 0

        # Should have metadata chunk as first
        assert result.chunks[0].metadata.chunk_type == ChunkType.METADATA

        # Should preserve context
        assert result.context == mock_extraction_result.context

    def test_merge_page_content(self, generator, sample_pages):
        """Test page content merging."""
        merged = generator._merge_page_content(sample_pages)

        # Should merge pages 1 and 2 (continues_to_next/continues_from_previous)
        assert len(merged) >= 1

    def test_chunk_section_small(self, generator, sample_context):
        """Test chunking small content (no split needed)."""
        section = {
            "content": "Short content that fits in one chunk.",
            "pages": [1],
            "section_numbers": ["§1"],
            "section_titles": ["Test"],
            "has_table": False,
            "has_list": False,
            "content_types": [ChunkType.SECTION],
        }

        chunks = generator._chunk_section(section, "test-doc", sample_context)
        assert len(chunks) == 1

    def test_chunk_section_large(self, generator, sample_context):
        """Test chunking large content (split needed)."""
        # Create content larger than max_chunk_size
        large_content = "Test content. " * 200  # ~2600 chars

        section = {
            "content": large_content,
            "pages": [1, 2],
            "section_numbers": ["§1"],
            "section_titles": ["Test"],
            "has_table": False,
            "has_list": False,
            "content_types": [ChunkType.SECTION],
        }

        chunks = generator._chunk_section(section, "test-doc", sample_context)
        assert len(chunks) > 1

        # Each chunk should be within limits
        for chunk in chunks:
            assert len(chunk.text) <= generator.config.max_chunk_size

    def test_split_content(self, generator):
        """Test content splitting."""
        # Content with paragraph markers
        content = "(1) First paragraph. (2) Second paragraph. (3) Third paragraph."

        chunks = generator._split_content(content)
        assert len(chunks) >= 1

    def test_create_metadata_chunk(self, generator, sample_context):
        """Test metadata chunk creation."""
        chunk = generator._create_metadata_chunk(sample_context, "test-doc")

        assert chunk.metadata.chunk_type == ChunkType.METADATA
        assert sample_context.title in chunk.text
        assert sample_context.institution in chunk.text

    def test_export_jsonl(self, generator, mock_extraction_result):
        """Test JSONL export."""
        result = generator.generate_from_extraction(
            mock_extraction_result,
            "test-document",
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name

        try:
            generator.export_jsonl(result.chunks, output_path)

            # Read and verify
            with open(output_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == len(result.chunks)

            # Each line should be valid JSON
            for line in lines:
                data = json.loads(line)
                assert "id" in data
                assert "text" in data
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_json(self, generator, mock_extraction_result):
        """Test JSON export."""
        result = generator.generate_from_extraction(
            mock_extraction_result,
            "test-document",
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            generator.export_json(result.chunks, output_path)

            # Read and verify
            with open(output_path, 'r') as f:
                data = json.load(f)

            assert "total_chunks" in data
            assert data["total_chunks"] == len(result.chunks)
            assert "chunks" in data
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_get_stats(self, generator, mock_extraction_result):
        """Test statistics calculation."""
        result = generator.generate_from_extraction(
            mock_extraction_result,
            "test-document",
        )

        stats = generator.get_stats(result.chunks)

        assert isinstance(stats, ChunkingStats)
        assert stats.total_chunks == len(result.chunks)
        assert stats.avg_chunk_length > 0
        assert stats.min_chunk_length <= stats.max_chunk_length

    def test_get_stats_empty(self, generator):
        """Test statistics with empty chunks."""
        stats = generator.get_stats([])

        assert stats.total_chunks == 0
        assert stats.avg_chunk_length == 0

    def test_link_related_chunks(self, generator, mock_extraction_result):
        """Test chunk linking."""
        result = generator.generate_from_extraction(
            mock_extraction_result,
            "test-document",
        )

        # Chunks should have related_sections populated
        # (depends on content, so just check no errors)
        for chunk in result.chunks:
            assert isinstance(chunk.metadata.related_sections, list)

    def test_extract_keywords(self, generator, sample_context):
        """Test keyword extraction."""
        content = "Das Modul umfasst 6 Leistungspunkte und endet mit einer Klausur."

        keywords = generator._extract_keywords(content, sample_context)

        assert isinstance(keywords, list)
        # Should find known terms
        assert any("Modul" in kw for kw in keywords) or any("Leistungspunkte" in kw for kw in keywords)

    def test_determine_chapter(self, generator, sample_context):
        """Test chapter determination."""
        # Early section (§1-3) should be in first chapter
        chapter = generator._determine_chapter(["§2"], sample_context)
        assert chapter == sample_context.chapters[0]

        # No section number
        chapter = generator._determine_chapter([], sample_context)
        assert chapter is None
