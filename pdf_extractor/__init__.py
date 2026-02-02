"""
PDF Extractor - Section-Based PDF Content Extraction

Extracts structured content from PDF documents using OpenAI's Vision API.
Optimized for German academic documents (Prüfungsordnungen, etc.).

Quick Start:
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    result = extractor.extract("document.pdf")

    for section in result.sections:
        print(f"{section.section_number}: {section.content[:100]}...")

    result.save("output.json")

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)
"""

__version__ = "2.0.0"

# Main extractor
from .extractor import PDFExtractor, estimate_api_cost

# Data models
from .models import (
    DocumentType,
    SectionType,
    Language,
    DocumentContext,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    Abbreviation,
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
from .pdf_to_images import PDFToImages, PageImage

__all__ = [
    "__version__",
    # Main
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
    # Exceptions
    "ExtractionError",
    "NoTableOfContentsError",
    "StructureExtractionError",
    "SectionExtractionError",
    "PageRenderError",
    "APIError",
    # PDF
    "PDFToImages",
    "PageImage",
]
