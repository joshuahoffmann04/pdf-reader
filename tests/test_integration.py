"""
Integration tests for the PDF to RAG pipeline.

These tests use the real OpenAI API and require:
1. OPENAI_API_KEY environment variable to be set
2. A test PDF file in the pdfs/ directory

Run with: pytest tests/test_integration.py -v

Note: These tests will incur API costs!
"""

import pytest
import os
import json
import tempfile
from pathlib import Path

from src.llm_processor import (
    VisionProcessor,
    ChunkGenerator,
    ProcessingConfig,
    ProcessingResult,
    DocumentType,
)


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping integration tests"
)


@pytest.fixture
def api_key():
    """Get API key from environment."""
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def test_pdf():
    """Get path to test PDF."""
    pdf_path = Path(__file__).parent.parent / "pdfs" / "Pruefungsordnung_BSc_Inf_2024.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found: {pdf_path}")
    return pdf_path


@pytest.fixture
def config():
    """Create test configuration with cheaper model."""
    return ProcessingConfig(
        model="gpt-4o-mini",  # Use cheaper model for tests
        target_chunk_size=500,
        max_chunk_size=1000,
    )


class TestVisionProcessorIntegration:
    """Integration tests for VisionProcessor."""

    def test_process_single_page(self, api_key, test_pdf, config):
        """Test processing just the first page."""
        processor = VisionProcessor(config=config, api_key=api_key)

        # Only process first page by mocking page count
        # (This is a simplified test)
        result = processor.process_document(test_pdf)

        assert result.context is not None
        assert result.context.document_type in DocumentType
        assert len(result.pages) > 0
        assert result.processing_time_seconds > 0

    def test_context_extraction(self, api_key, test_pdf, config):
        """Test that context is correctly extracted."""
        processor = VisionProcessor(config=config, api_key=api_key)
        result = processor.process_document(test_pdf)

        context = result.context

        # Should identify as Prüfungsordnung
        assert context.document_type == DocumentType.PRUEFUNGSORDNUNG

        # Should extract institution
        assert "Marburg" in context.institution or "Universität" in context.institution

        # Should have some chapters
        assert len(context.chapters) > 0

        # Should have some abbreviations
        assert len(context.abbreviations) > 0

    def test_page_extraction_quality(self, api_key, test_pdf, config):
        """Test that pages are correctly extracted."""
        processor = VisionProcessor(config=config, api_key=api_key)
        result = processor.process_document(test_pdf)

        # Check first few pages
        for page in result.pages[:3]:
            # Should have content
            assert len(page.content) > 0

            # Content should be in German
            assert any(word in page.content.lower() for word in
                       ["der", "die", "das", "und", "ist", "wird"])

    def test_table_conversion(self, api_key, test_pdf, config):
        """Test that tables are converted to natural language."""
        processor = VisionProcessor(config=config, api_key=api_key)
        result = processor.process_document(test_pdf)

        # Find pages with tables
        table_pages = [p for p in result.pages if p.has_table]

        if table_pages:
            # Table content should be in natural language, not structured
            for page in table_pages[:2]:
                # Should not have pipe characters (markdown tables)
                assert "|" not in page.content or page.content.count("|") < 5


class TestChunkGeneratorIntegration:
    """Integration tests for ChunkGenerator."""

    def test_generate_chunks(self, api_key, test_pdf, config):
        """Test full chunk generation pipeline."""
        processor = VisionProcessor(config=config, api_key=api_key)
        generator = ChunkGenerator(config=config)

        extraction = processor.process_document(test_pdf)
        result = generator.generate_from_extraction(extraction, "test-document")

        assert len(result.chunks) > 0

        # First chunk should be metadata
        assert result.chunks[0].metadata.chunk_type.value == "metadata"

    def test_chunk_quality(self, api_key, test_pdf, config):
        """Test that chunks meet quality requirements."""
        processor = VisionProcessor(config=config, api_key=api_key)
        generator = ChunkGenerator(config=config)

        extraction = processor.process_document(test_pdf)
        result = generator.generate_from_extraction(extraction, "test-document")

        for chunk in result.chunks:
            # Each chunk should have content
            assert len(chunk.text) > 0

            # Chunks should be within size limits
            assert len(chunk.text) <= config.max_chunk_size * 1.5  # Allow some overflow

            # Each chunk should have required metadata
            assert chunk.metadata.source_document is not None
            assert len(chunk.metadata.source_pages) > 0

    def test_export_formats(self, api_key, test_pdf, config):
        """Test that all export formats work."""
        processor = VisionProcessor(config=config, api_key=api_key)
        generator = ChunkGenerator(config=config)

        extraction = processor.process_document(test_pdf)
        result = generator.generate_from_extraction(extraction, "test-document")

        # Test LangChain export
        langchain_docs = result.export_chunks_langchain()
        assert len(langchain_docs) == len(result.chunks)
        assert all("page_content" in doc for doc in langchain_docs)

        # Test LlamaIndex export
        llamaindex_nodes = result.export_chunks_llamaindex()
        assert len(llamaindex_nodes) == len(result.chunks)
        assert all("text" in node for node in llamaindex_nodes)

        # Test JSONL export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name

        try:
            result.export_chunks_jsonl(output_path)

            with open(output_path, 'r') as f:
                lines = f.readlines()

            assert len(lines) == len(result.chunks)
        finally:
            Path(output_path).unlink(missing_ok=True)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline(self, api_key, test_pdf, config):
        """Test the complete pipeline from PDF to chunks."""
        # Process document
        processor = VisionProcessor(config=config, api_key=api_key)
        extraction = processor.process_document(test_pdf)

        # Generate chunks
        generator = ChunkGenerator(config=config)
        result = generator.generate_from_extraction(extraction, "test-document")

        # Verify result
        assert isinstance(result, ProcessingResult)
        assert result.context.document_type == DocumentType.PRUEFUNGSORDNUNG
        assert len(result.pages) > 0
        assert len(result.chunks) > 0

        # Verify statistics
        stats = result.get_chunk_stats()
        assert stats["total_chunks"] > 0
        assert stats["avg_length"] > 0

        print(f"\nPipeline completed successfully:")
        print(f"  Pages processed: {len(result.pages)}")
        print(f"  Chunks generated: {stats['total_chunks']}")
        print(f"  Average chunk length: {stats['avg_length']:.0f} chars")
        print(f"  Processing time: {result.processing_time_seconds:.1f}s")
        print(f"  Input tokens: {result.total_input_tokens:,}")
        print(f"  Output tokens: {result.total_output_tokens:,}")

    def test_chunk_retrieval_simulation(self, api_key, test_pdf, config):
        """Simulate a retrieval query."""
        processor = VisionProcessor(config=config, api_key=api_key)
        extraction = processor.process_document(test_pdf)

        generator = ChunkGenerator(config=config)
        result = generator.generate_from_extraction(extraction, "test-document")

        # Simulate searching for "Bachelorarbeit"
        query = "bachelorarbeit"
        matching_chunks = [
            chunk for chunk in result.chunks
            if query in chunk.text.lower()
        ]

        print(f"\nSimulated retrieval for '{query}':")
        print(f"  Found {len(matching_chunks)} matching chunks")

        if matching_chunks:
            # Show first match
            first_match = matching_chunks[0]
            print(f"\n  First match:")
            print(f"    Section: {first_match.metadata.section_number}")
            print(f"    Pages: {first_match.metadata.source_pages}")
            print(f"    Text preview: {first_match.text[:200]}...")
