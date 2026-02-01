"""
PDF Extractor - Section-Based PDF Content Extraction

A production-ready module for extracting structured content from PDF documents
using OpenAI's Vision API (GPT-4o). Optimized for German academic documents
(Prüfungsordnungen, Modulhandbücher, etc.).

Features:
- Two-phase extraction: structure analysis + section-by-section extraction
- Sends ALL pages of a section in ONE API call (no page-boundary issues)
- Sliding window for large sections (> max_images_per_request)
- High-quality content transformation to natural language
- Robust error handling with retry mechanism
- Raises NoTableOfContentsError if no ToC found

Quick Start:
    from pdf_extractor import PDFExtractor

    # Initialize extractor
    extractor = PDFExtractor()

    # Extract content from PDF
    result = extractor.extract("document.pdf")

    # Access extracted data
    print(f"Title: {result.context.title}")
    print(f"Sections: {len(result.sections)}")

    for section in result.sections:
        print(f"{section.section_number}: {section.content[:100]}...")

    # Save result
    result.save("output.json")

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)

For more information, see the README.md in this directory.
"""

__version__ = "2.0.0"
__author__ = "PDF Extractor Team"

# Main extractor class
from .extractor import PDFExtractor, estimate_api_cost

# Data models
from .models import (
    # Enums
    DocumentType,
    SectionType,
    Language,
    # Core models
    DocumentContext,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    Abbreviation,
    # Configuration
    ExtractionConfig,
)

# Exceptions
from .exceptions import (
    ExtractionError,
    NoTableOfContentsError,
    StructureExtractionError,
    SectionExtractionError,
    PageRenderError,
    APIError,
)

# PDF utilities
from .pdf_to_images import (
    PDFToImages,
    PageImage,
)

# Legacy compatibility aliases
from .extractor import ProcessingConfig  # Alias for ExtractionConfig

__all__ = [
    # Version
    "__version__",
    # Main class
    "PDFExtractor",
    "estimate_api_cost",
    # Enums
    "DocumentType",
    "SectionType",
    "Language",
    # Models
    "DocumentContext",
    "StructureEntry",
    "ExtractedSection",
    "ExtractionResult",
    "Abbreviation",
    "ExtractionConfig",
    "ProcessingConfig",  # Legacy alias
    # Exceptions
    "ExtractionError",
    "NoTableOfContentsError",
    "StructureExtractionError",
    "SectionExtractionError",
    "PageRenderError",
    "APIError",
    # PDF utilities
    "PDFToImages",
    "PageImage",
]
