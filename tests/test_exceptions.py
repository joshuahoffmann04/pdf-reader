"""
Tests for PDF Extractor exceptions.
"""

import pytest

from pdf_extractor import (
    # Base
    ExtractionError,
    # PDF errors
    PDFError,
    PDFNotFoundError,
    PDFCorruptedError,
    PageRenderError,
    # Scan errors
    ScanError,
    PageScanError,
    StructureAggregationError,
    # Content extraction errors
    ContentExtractionError,
    ContextExtractionError,
    SectionExtractionError,
    # API errors
    APIError,
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
    # Utilities
    is_retryable,
    format_error_chain,
)


class TestExtractionError:
    """Tests for base ExtractionError."""

    def test_create_simple(self):
        """Test creating error with message only."""
        error = ExtractionError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details is None

    def test_create_with_details(self):
        """Test creating error with details."""
        error = ExtractionError("Error occurred", details="More info here")
        assert "Error occurred" in str(error)
        assert "More info here" in str(error)
        assert error.details == "More info here"


class TestPDFErrors:
    """Tests for PDF-related errors."""

    def test_pdf_not_found(self):
        """Test PDFNotFoundError."""
        error = PDFNotFoundError("/path/to/missing.pdf")
        assert error.path == "/path/to/missing.pdf"
        assert "missing.pdf" in str(error)

    def test_pdf_corrupted(self):
        """Test PDFCorruptedError."""
        original = ValueError("Invalid PDF structure")
        error = PDFCorruptedError("/path/to/broken.pdf", original)
        assert error.path == "/path/to/broken.pdf"
        assert error.original_error == original
        assert "corrupted" in str(error).lower()

    def test_page_render_error(self):
        """Test PageRenderError."""
        error = PageRenderError(page_number=5, path="/doc.pdf")
        assert error.page_number == 5
        assert "5" in str(error)


class TestScanErrors:
    """Tests for scan-related errors."""

    def test_page_scan_error(self):
        """Test PageScanError."""
        error = PageScanError(
            page_number=10,
            message="Failed to scan page 10",
            api_response="Invalid JSON",
        )
        assert error.page_number == 10
        assert error.api_response == "Invalid JSON"
        assert "10" in str(error)

    def test_structure_aggregation_error(self):
        """Test StructureAggregationError."""
        error = StructureAggregationError(
            message="No sections found",
            details="Document appears empty",
        )
        assert "No sections found" in str(error)
        assert error.details == "Document appears empty"


class TestContentExtractionErrors:
    """Tests for content extraction errors."""

    def test_context_extraction_error(self):
        """Test ContextExtractionError."""
        error = ContextExtractionError("Failed to extract metadata")
        assert "metadata" in str(error)

    def test_section_extraction_error(self):
        """Test SectionExtractionError."""
        error = SectionExtractionError(
            section_identifier="§ 10",
            pages=[12, 13, 14],
        )
        assert error.section_identifier == "§ 10"
        assert error.pages == [12, 13, 14]
        assert "§ 10" in str(error)


class TestAPIErrors:
    """Tests for API-related errors."""

    def test_api_error(self):
        """Test base APIError."""
        original = RuntimeError("Network timeout")
        error = APIError("Request failed", original, status_code=500)
        assert error.status_code == 500
        assert error.original_error == original
        assert "500" in str(error)

    def test_api_connection_error(self):
        """Test APIConnectionError."""
        error = APIConnectionError("Cannot reach API")
        assert "Cannot reach" in str(error)

    def test_api_rate_limit_error(self):
        """Test APIRateLimitError."""
        error = APIRateLimitError(retry_after=30.0)
        assert error.retry_after == 30.0
        assert error.status_code == 429
        assert "30" in str(error)

    def test_api_response_error(self):
        """Test APIResponseError."""
        error = APIResponseError(
            message="Invalid JSON",
            response_content='{"broken": ',
        )
        assert error.response_content == '{"broken": '


class TestIsRetryable:
    """Tests for is_retryable utility."""

    def test_retryable_errors(self):
        """Test that retryable errors are identified."""
        assert is_retryable(APIConnectionError()) is True
        assert is_retryable(APIRateLimitError()) is True
        assert is_retryable(APIError("Server error", status_code=502)) is True
        assert is_retryable(PageScanError(1)) is True

    def test_non_retryable_errors(self):
        """Test that non-retryable errors are identified."""
        assert is_retryable(PDFNotFoundError("/path")) is False
        assert is_retryable(PDFCorruptedError("/path")) is False
        assert is_retryable(APIResponseError("Policy violation")) is False
        assert is_retryable(ValueError("Invalid argument")) is False


class TestFormatErrorChain:
    """Tests for format_error_chain utility."""

    def test_single_error(self):
        """Test formatting a single error."""
        error = ExtractionError("Simple error")
        formatted = format_error_chain(error)
        assert "ExtractionError" in formatted
        assert "Simple error" in formatted

    def test_error_chain(self):
        """Test formatting chained errors."""
        original = ValueError("Root cause")
        api_error = APIError("API failed", original)

        formatted = format_error_chain(api_error)
        assert "APIError" in formatted
        assert "ValueError" in formatted
        assert "Root cause" in formatted


class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    def test_hierarchy(self):
        """Test that exceptions follow correct hierarchy."""
        # All errors inherit from ExtractionError
        assert issubclass(PDFError, ExtractionError)
        assert issubclass(ScanError, ExtractionError)
        assert issubclass(ContentExtractionError, ExtractionError)
        assert issubclass(APIError, ExtractionError)

        # PDF errors
        assert issubclass(PDFNotFoundError, PDFError)
        assert issubclass(PDFCorruptedError, PDFError)
        assert issubclass(PageRenderError, PDFError)

        # Scan errors
        assert issubclass(PageScanError, ScanError)
        assert issubclass(StructureAggregationError, ScanError)

        # Content extraction errors
        assert issubclass(ContextExtractionError, ContentExtractionError)
        assert issubclass(SectionExtractionError, ContentExtractionError)

        # API errors
        assert issubclass(APIConnectionError, APIError)
        assert issubclass(APIRateLimitError, APIError)
        assert issubclass(APIResponseError, APIError)

    def test_can_catch_broadly(self):
        """Test that errors can be caught with base class."""
        def raise_pdf_error():
            raise PDFNotFoundError("/path")

        with pytest.raises(ExtractionError):
            raise_pdf_error()

        with pytest.raises(PDFError):
            raise_pdf_error()
