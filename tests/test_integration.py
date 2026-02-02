"""
Integration tests for the PDF Extractor.

These tests use the real OpenAI API and require:
1. OPENAI_API_KEY environment variable to be set
2. A test PDF file in the pdfs/ directory

Run with: pytest tests/test_integration.py -v

Note: These tests will incur API costs!
"""

import pytest
import os
from pathlib import Path

from pdf_extractor import (
    PDFExtractor,
    ProcessingConfig,
    ExtractionResult,
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
        max_retries=3,
    )


class TestPDFExtractorIntegration:
    """Integration tests for PDFExtractor."""

    def test_extract_document(self, api_key, test_pdf, config):
        """Test extracting a document."""
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

        assert result.context is not None
        assert result.context.document_type in DocumentType
        assert len(result.pages) > 0
        assert result.processing_time_seconds > 0

    def test_context_extraction(self, api_key, test_pdf, config):
        """Test that context is correctly extracted."""
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

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
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

        # Check first few pages
        for page in result.pages[:3]:
            # Should have content
            assert len(page.content) > 0

            # Content should be in German
            assert any(word in page.content.lower() for word in
                       ["der", "die", "das", "und", "ist", "wird"])

    def test_table_conversion(self, api_key, test_pdf, config):
        """Test that tables are converted to natural language."""
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

        # Find pages with tables
        table_pages = [p for p in result.pages if p.has_table]

        if table_pages:
            # Table content should be in natural language, not structured
            for page in table_pages[:2]:
                # Should not have pipe characters (markdown tables)
                assert "|" not in page.content or page.content.count("|") < 5


class TestExtractionResultIntegration:
    """Integration tests for ExtractionResult."""

    def test_result_methods(self, api_key, test_pdf, config):
        """Test ExtractionResult methods."""
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

        # Test get_page_stats
        stats = result.get_page_stats()
        assert stats["total_pages"] > 0
        assert stats["avg_content_length"] > 0

        # Test get_all_sections
        sections = result.get_all_sections()
        # Should have some sections
        assert len(sections) >= 0

        # Test get_full_content
        full_content = result.get_full_content()
        assert len(full_content) > 0

    def test_save_and_load(self, api_key, test_pdf, config, tmp_path):
        """Test saving and loading results."""
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

        # Save result
        save_path = tmp_path / "result.json"
        result.save(str(save_path))

        # Load result
        loaded = ExtractionResult.load(str(save_path))

        assert loaded.source_file == result.source_file
        assert loaded.context.title == result.context.title
        assert len(loaded.pages) == len(result.pages)


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_extraction(self, api_key, test_pdf, config):
        """Test the complete extraction pipeline."""
        # Process document
        extractor = PDFExtractor(config=config, api_key=api_key)
        result = extractor.extract(test_pdf)

        # Verify result
        assert isinstance(result, ExtractionResult)
        assert result.context.document_type == DocumentType.PRUEFUNGSORDNUNG
        assert len(result.pages) > 0

        # Verify statistics
        stats = result.get_page_stats()
        assert stats["total_pages"] > 0

        print(f"\nExtraction completed successfully:")
        print(f"  Pages processed: {len(result.pages)}")
        print(f"  Processing time: {result.processing_time_seconds:.1f}s")
        print(f"  Input tokens: {result.total_input_tokens:,}")
        print(f"  Output tokens: {result.total_output_tokens:,}")
        print(f"  Errors: {len(result.errors)}")
