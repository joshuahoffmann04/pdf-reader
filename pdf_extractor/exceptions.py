"""
Custom Exceptions for PDF Section Extraction.

This module defines a hierarchy of exceptions for precise error handling
in the page-by-page scanning extraction pipeline.

Exception Hierarchy:
    ExtractionError (base)
    ├── PDFError
    │   ├── PDFNotFoundError
    │   ├── PDFCorruptedError
    │   └── PageRenderError
    ├── ScanError
    │   ├── PageScanError
    │   └── StructureAggregationError
    ├── ContentExtractionError
    │   ├── ContextExtractionError
    │   └── SectionExtractionError
    └── APIError
        ├── APIConnectionError
        ├── APIRateLimitError
        └── APIResponseError

Usage:
    from pdf_extractor.exceptions import (
        ExtractionError,
        PDFNotFoundError,
        PageScanError,
    )

    try:
        result = extractor.extract("document.pdf")
    except PDFNotFoundError as e:
        print(f"File not found: {e.path}")
    except PageScanError as e:
        print(f"Scan failed on page {e.page_number}: {e}")
    except ExtractionError as e:
        print(f"Extraction failed: {e}")
"""

from __future__ import annotations

from typing import Optional


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class ExtractionError(Exception):
    """
    Base exception for all extraction-related errors.

    All custom exceptions inherit from this class, allowing for
    broad exception catching when needed.

    Attributes:
        message: Human-readable error description
        details: Additional technical details (optional)
    """

    def __init__(
        self,
        message: str = "An extraction error occurred",
        details: Optional[str] = None,
    ):
        self.message = message
        self.details = details

        full_message = message
        if details:
            full_message = f"{message} | Details: {details}"

        super().__init__(full_message)


# =============================================================================
# PDF ERRORS
# =============================================================================


class PDFError(ExtractionError):
    """Base class for PDF-related errors."""

    def __init__(
        self,
        message: str = "PDF error",
        path: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.path = path
        if path:
            message = f"{message} [{path}]"
        super().__init__(message, details)


class PDFNotFoundError(PDFError):
    """
    Raised when the PDF file cannot be found.

    Attributes:
        path: Path to the missing file
    """

    def __init__(self, path: str):
        super().__init__(
            message=f"PDF file not found: {path}",
            path=path,
        )


class PDFCorruptedError(PDFError):
    """
    Raised when the PDF file is corrupted or cannot be opened.

    Attributes:
        path: Path to the corrupted file
        original_error: The underlying error from the PDF library
    """

    def __init__(
        self,
        path: str,
        original_error: Optional[Exception] = None,
    ):
        self.original_error = original_error
        details = str(original_error) if original_error else None
        super().__init__(
            message=f"PDF file is corrupted or unreadable: {path}",
            path=path,
            details=details,
        )


class PageRenderError(PDFError):
    """
    Raised when a PDF page cannot be rendered to an image.

    Attributes:
        page_number: The page that failed to render (1-indexed)
        path: Path to the PDF file
    """

    def __init__(
        self,
        page_number: int,
        path: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.page_number = page_number
        self.original_error = original_error
        details = str(original_error) if original_error else None
        super().__init__(
            message=f"Failed to render page {page_number}",
            path=path,
            details=details,
        )


# =============================================================================
# SCAN ERRORS
# =============================================================================


class ScanError(ExtractionError):
    """Base class for scanning-phase errors."""

    pass


class PageScanError(ScanError):
    """
    Raised when a page cannot be scanned for sections.

    This occurs during Phase 1 when the LLM fails to analyze
    a specific page for section detection.

    Attributes:
        page_number: The page that failed to scan (1-indexed)
        api_response: Raw API response if available
    """

    def __init__(
        self,
        page_number: int,
        message: Optional[str] = None,
        api_response: Optional[str] = None,
    ):
        self.page_number = page_number
        self.api_response = api_response
        msg = message or f"Failed to scan page {page_number}"
        super().__init__(msg, details=api_response)


class StructureAggregationError(ScanError):
    """
    Raised when scan results cannot be aggregated into a document structure.

    This can happen when:
    - No sections were detected in any page
    - Section identifiers are inconsistent across pages
    - The document structure is malformed
    """

    def __init__(
        self,
        message: str = "Failed to aggregate document structure from scan results",
        details: Optional[str] = None,
    ):
        super().__init__(message, details)


# =============================================================================
# CONTENT EXTRACTION ERRORS
# =============================================================================


class ContentExtractionError(ExtractionError):
    """Base class for content extraction errors."""

    pass


class ContextExtractionError(ContentExtractionError):
    """
    Raised when document context/metadata cannot be extracted.

    This occurs during Phase 3 when the LLM fails to extract
    document-level information like title, institution, etc.
    """

    def __init__(
        self,
        message: str = "Failed to extract document context",
        details: Optional[str] = None,
    ):
        super().__init__(message, details)


class SectionExtractionError(ContentExtractionError):
    """
    Raised when a specific section cannot be extracted.

    This occurs during Phase 4 when the LLM fails to extract
    the content of a specific section.

    Attributes:
        section_identifier: The section that failed (e.g., "§ 10", "Anlage 2")
        pages: The pages that were being processed
    """

    def __init__(
        self,
        section_identifier: str,
        pages: Optional[list[int]] = None,
        message: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.section_identifier = section_identifier
        self.pages = pages or []

        msg = message or f"Failed to extract section: {section_identifier}"
        if pages:
            msg = f"{msg} (pages {pages[0]}-{pages[-1]})"

        super().__init__(msg, details)


# =============================================================================
# API ERRORS
# =============================================================================


class APIError(ExtractionError):
    """
    Base class for OpenAI API errors.

    Wraps API-specific errors for consistent error handling.

    Attributes:
        original_error: The underlying API exception
        status_code: HTTP status code if available
    """

    def __init__(
        self,
        message: str = "API error",
        original_error: Optional[Exception] = None,
        status_code: Optional[int] = None,
    ):
        self.original_error = original_error
        self.status_code = status_code

        details = None
        if original_error:
            details = str(original_error)
        if status_code:
            message = f"{message} (HTTP {status_code})"

        super().__init__(message, details)


class APIConnectionError(APIError):
    """
    Raised when the API cannot be reached.

    This includes network errors, DNS failures, and timeouts.
    """

    def __init__(
        self,
        message: str = "Cannot connect to OpenAI API",
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, original_error)


class APIRateLimitError(APIError):
    """
    Raised when the API rate limit is exceeded.

    Attributes:
        retry_after: Suggested wait time in seconds (if provided by API)
    """

    def __init__(
        self,
        retry_after: Optional[float] = None,
        original_error: Optional[Exception] = None,
    ):
        self.retry_after = retry_after
        message = "OpenAI API rate limit exceeded"
        if retry_after:
            message = f"{message} (retry after {retry_after}s)"
        super().__init__(message, original_error, status_code=429)


class APIResponseError(APIError):
    """
    Raised when the API returns an unexpected or invalid response.

    This includes:
    - JSON parsing failures
    - Missing required fields in response
    - Content policy violations
    - Model refusals

    Attributes:
        response_content: Raw response content if available
    """

    def __init__(
        self,
        message: str = "Invalid API response",
        response_content: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        self.response_content = response_content
        super().__init__(message, original_error)
        if response_content:
            self.details = response_content[:500]  # Truncate long responses


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def is_retryable(error: Exception) -> bool:
    """
    Check if an error is potentially recoverable by retrying.

    Returns True for:
    - Network/connection errors
    - Rate limit errors
    - Temporary API failures

    Returns False for:
    - File not found
    - Corrupted PDFs
    - Invalid responses (content policy, etc.)
    """
    if isinstance(error, (APIConnectionError, APIRateLimitError)):
        return True
    if isinstance(error, APIError) and error.status_code in (500, 502, 503, 504):
        return True
    if isinstance(error, PageScanError):
        # Page scan failures might be transient
        return True
    return False


def format_error_chain(error: Exception) -> str:
    """
    Format an exception and its chain for logging.

    Returns a multi-line string showing the error hierarchy.
    """
    lines = []
    current = error
    depth = 0

    while current is not None:
        prefix = "  " * depth + ("└─ " if depth > 0 else "")
        lines.append(f"{prefix}{type(current).__name__}: {current}")

        if hasattr(current, "original_error") and current.original_error:
            current = current.original_error
            depth += 1
        elif current.__cause__:
            current = current.__cause__
            depth += 1
        else:
            break

    return "\n".join(lines)
