"""
PDF Reader - Extract and structure PDF content by sections.

Core components:
- PDFExtractor: Raw text extraction from PDF files
- DocumentParser: Hierarchical structure parsing (Chapters, Sections, Appendices)
- TableExtractor: Table detection and extraction
- ImageExtractor: Image extraction with deduplication
- Evaluator: Similarity metrics for quality assessment
"""

from src.extractor import PDFExtractor
from src.parser import DocumentParser, ParsedDocument, Chapter, Section, Appendix
from src.tables import TableExtractor
from src.images import ImageExtractor
from src.evaluation import Evaluator

__all__ = [
    "PDFExtractor",
    "DocumentParser",
    "ParsedDocument",
    "Chapter",
    "Section",
    "Appendix",
    "TableExtractor",
    "ImageExtractor",
    "Evaluator",
]
