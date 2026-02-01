"""
Tests for PDF Extractor exceptions.
"""

import pytest

from pdf_extractor import (
    ExtractionError,
    NoTableOfContentsError,
    StructureExtractionError,
    SectionExtractionError,
    PageRenderError,
    APIError,
)


class TestExtractionError:
    """Tests for base ExtractionError."""

    def test_create_basic(self):
        """Test creating basic exception."""
        error = ExtractionError("Test error message")
        assert str(error) == "Test error message"

    def test_inheritance(self):
        """Test that all exceptions inherit from ExtractionError."""
        assert issubclass(NoTableOfContentsError, ExtractionError)
        assert issubclass(StructureExtractionError, ExtractionError)
        assert issubclass(SectionExtractionError, ExtractionError)
        assert issubclass(PageRenderError, ExtractionError)
        assert issubclass(APIError, ExtractionError)


class TestNoTableOfContentsError:
    """Tests for NoTableOfContentsError."""

    def test_create_default(self):
        """Test creating with default message."""
        error = NoTableOfContentsError()
        assert "Inhaltsverzeichnis" in str(error)

    def test_create_with_path(self):
        """Test creating with document path."""
        error = NoTableOfContentsError(document_path="/path/to/doc.pdf")
        assert error.document_path == "/path/to/doc.pdf"

    def test_create_with_custom_message(self):
        """Test creating with custom message."""
        error = NoTableOfContentsError(message="Custom message")
        assert str(error) == "Custom message"

    def test_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(NoTableOfContentsError) as exc_info:
            raise NoTableOfContentsError(document_path="test.pdf")

        assert exc_info.value.document_path == "test.pdf"

    def test_can_be_caught_as_extraction_error(self):
        """Test that exception can be caught as ExtractionError."""
        with pytest.raises(ExtractionError):
            raise NoTableOfContentsError()


class TestStructureExtractionError:
    """Tests for StructureExtractionError."""

    def test_create_basic(self):
        """Test creating basic exception."""
        error = StructureExtractionError("Failed to parse structure")
        assert str(error) == "Failed to parse structure"


class TestSectionExtractionError:
    """Tests for SectionExtractionError."""

    def test_create_with_section(self):
        """Test creating with section number."""
        error = SectionExtractionError("Extraction failed", section_number="§ 5")
        assert error.section_number == "§ 5"
        assert "§ 5" in str(error)

    def test_create_without_section(self):
        """Test creating without section number."""
        error = SectionExtractionError("Extraction failed")
        assert error.section_number is None

    def test_create_with_pages(self):
        """Test creating with pages."""
        error = SectionExtractionError("Extraction failed", section_number="§ 5", pages=[10, 11])
        assert error.pages == [10, 11]


class TestPageRenderError:
    """Tests for PageRenderError."""

    def test_create_with_page(self):
        """Test creating with page number."""
        error = PageRenderError("Render failed", page_number=5)
        assert error.page_number == 5
        assert "5" in str(error)


class TestAPIError:
    """Tests for APIError."""

    def test_create_basic(self):
        """Test creating basic exception."""
        error = APIError("API call failed")
        assert str(error) == "API call failed"

    def test_create_with_original_error(self):
        """Test creating with original error."""
        original = ValueError("Original error")
        error = APIError("Rate limit exceeded", original_error=original)
        assert error.original_error is original
        assert "Original error" in str(error)
