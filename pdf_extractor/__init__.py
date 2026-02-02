"""
PDF Extractor - Page-by-Page Section Extraction for German Academic Documents.

This package extracts structured content from PDF documents using OpenAI's Vision API.
It uses a four-phase pipeline:

1. PAGE SCAN: Scan each page individually to detect sections
2. STRUCTURE: Aggregate results to calculate accurate page ranges
3. CONTEXT: Extract document metadata
4. EXTRACTION: Extract full content for each section

Quick Start:
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    result = extractor.extract("pruefungsordnung.pdf")

    for section in result.sections:
        print(f"{section.identifier}: {section.content[:100]}...")

    result.save("output.json")

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)

Documentation:
    See README.md for full documentation.
"""

__version__ = "3.0.0"

# =============================================================================
# Main Extractor
# =============================================================================

from .extractor import PDFExtractor

# =============================================================================
# Configuration
# =============================================================================

from .models import ExtractionConfig

# =============================================================================
# Result Models
# =============================================================================

from .models import (
    # Final result
    ExtractionResult,
    ExtractedSection,
    # Document info
    DocumentContext,
    DocumentStructure,
    # Section location
    SectionLocation,
    # Scan results (for debugging)
    PageScanResult,
    DetectedSection,
    # Legacy compatibility
    StructureEntry,
    # Metadata
    Abbreviation,
)

# =============================================================================
# Enums
# =============================================================================

from .models import (
    SectionType,
    DocumentType,
    Language,
)

# =============================================================================
# Exceptions
# =============================================================================

from .exceptions import (
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

# =============================================================================
# PDF Utilities
# =============================================================================

from .pdf_utils import (
    PDFRenderer,
    PageImage,
    PDFInfo,
    validate_pdf,
    estimate_api_cost,
)

# =============================================================================
# API Client (for advanced usage)
# =============================================================================

from .api_client import (
    VisionAPIClient,
    APIResponse,
    TokenUsage,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Main extractor
    "PDFExtractor",
    # Configuration
    "ExtractionConfig",
    # Result models
    "ExtractionResult",
    "ExtractedSection",
    "DocumentContext",
    "DocumentStructure",
    "SectionLocation",
    "PageScanResult",
    "DetectedSection",
    "StructureEntry",  # Legacy
    "Abbreviation",
    # Enums
    "SectionType",
    "DocumentType",
    "Language",
    # Exceptions
    "ExtractionError",
    "PDFError",
    "PDFNotFoundError",
    "PDFCorruptedError",
    "PageRenderError",
    "ScanError",
    "PageScanError",
    "StructureAggregationError",
    "ContentExtractionError",
    "ContextExtractionError",
    "SectionExtractionError",
    "APIError",
    "APIConnectionError",
    "APIRateLimitError",
    "APIResponseError",
    "is_retryable",
    "format_error_chain",
    # PDF utilities
    "PDFRenderer",
    "PageImage",
    "PDFInfo",
    "validate_pdf",
    "estimate_api_cost",
    # API client
    "VisionAPIClient",
    "APIResponse",
    "TokenUsage",
]
