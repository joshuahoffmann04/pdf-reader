"""
PDF Extractor - Hybrid PDF Content Extraction

Production-ready extraction for German academic documents:
- Text-native extraction (PyMuPDF)
- Optional OCR (Tesseract)
- Optional Vision-LLM fallback
- Coverage validation to prevent information loss

Quick Start:
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    result = extractor.extract("document.pdf")
    result.save("output.json")

Environment (optional):
    OPENAI_API_KEY: Enable LLM vision extraction
    TESSERACT_CMD: Path to tesseract.exe for OCR
"""

__version__ = "1.0.0"
__author__ = "Joshua Hoffmann"

# Main extractor class
from .extractor import PDFExtractor
from .service import ExtractionService
from .config import ExtractorConfig

# Data models
from .models import (
    # Enums
    DocumentType,
    ContentType,
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
)

__all__ = [
    # Version
    "__version__",
    # Main class
    "PDFExtractor",
    "ExtractionService",
    "ExtractorConfig",
    # Enums
    "DocumentType",
    "ContentType",
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
]
