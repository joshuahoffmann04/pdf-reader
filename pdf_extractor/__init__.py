"""
PDF Extractor - Vision-LLM based PDF Content Extraction

A production-ready module for extracting structured content from PDF documents
using OpenAI's Vision API (GPT-4o). Optimized for German academic documents
(Prüfungsordnungen, Modulhandbücher, etc.).

Features:
- Two-phase extraction: context analysis + page-by-page extraction
- High-quality content transformation to natural language
- Robust error handling with retry mechanism
- Structured output ready for downstream processing

Quick Start:
    from pdf_extractor import PDFExtractor

    # Initialize extractor
    extractor = PDFExtractor()

    # Extract content from PDF
    result = extractor.extract("document.pdf")

    # Access extracted data
    print(f"Title: {result.context.title}")
    print(f"Pages: {len(result.pages)}")

    # Save result
    result.save("output.json")

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)

For more information, see the README.md in this directory.
"""

__version__ = "1.0.0"
__author__ = "PDF Extractor Team"

# Main extractor class
from .extractor import PDFExtractor
from .service import ExtractionService
from .config import ExtractorConfig

# Data models
from .models import (
    # Enums
    DocumentType,
    ContentType,
    Language,
    # Core models
    DocumentContext,
    ExtractedPage,
    ExtractionResult,
    SectionMarker,
    Abbreviation,
    # Configuration
    ProcessingConfig,
)

# PDF utilities
from .pdf_to_images import (
    PDFToImages,
    PageImage,
    estimate_api_cost,
)

# Convenience alias (backwards compatibility)
VisionProcessor = PDFExtractor

__all__ = [
    # Version
    "__version__",
    # Main class
    "PDFExtractor",
    "ExtractionService",
    "ExtractorConfig",
    "VisionProcessor",  # backwards compatibility
    # Enums
    "DocumentType",
    "ContentType",
    "Language",
    # Models
    "DocumentContext",
    "ExtractedPage",
    "ExtractionResult",
    "SectionMarker",
    "Abbreviation",
    "ProcessingConfig",
    # PDF utilities
    "PDFToImages",
    "PageImage",
    "estimate_api_cost",
]
