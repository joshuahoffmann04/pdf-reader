"""
PDF Text Extraction Module

Uses PyMuPDF (fitz) for text extraction with layout preservation.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    blocks: list = field(default_factory=list)  # Raw text blocks with positions


@dataclass
class PDFDocument:
    """Represents an extracted PDF document."""
    path: str
    title: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get concatenated text from all pages."""
        return "\n\n".join(page.text for page in self.pages)


class PDFExtractor:
    """
    Extracts text and structure from PDF files using PyMuPDF.

    Optimized for Word-exported PDFs with complex layouts.
    """

    def __init__(self, preserve_layout: bool = True):
        """
        Initialize the PDF extractor.

        Args:
            preserve_layout: If True, attempt to preserve reading order and layout.
        """
        self.preserve_layout = preserve_layout

    def extract(self, pdf_path: str | Path) -> PDFDocument:
        """
        Extract content from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            PDFDocument containing all extracted content.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)

        try:
            # Extract metadata
            metadata = {
                "author": doc.metadata.get("author", ""),
                "title": doc.metadata.get("title", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
            }

            # Use filename as title if metadata title is empty
            title = metadata["title"] or pdf_path.stem

            # Extract pages
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_content = self._extract_page(page, page_num + 1)
                pages.append(page_content)

            return PDFDocument(
                path=str(pdf_path),
                title=title,
                total_pages=len(doc),
                pages=pages,
                metadata=metadata
            )
        finally:
            doc.close()

    def _extract_page(self, page: fitz.Page, page_number: int) -> PageContent:
        """
        Extract content from a single page.

        Args:
            page: PyMuPDF page object.
            page_number: 1-indexed page number.

        Returns:
            PageContent with extracted text and blocks.
        """
        # Extract text with layout preservation
        if self.preserve_layout:
            # Use "dict" extraction for detailed block information
            blocks = page.get_text("dict", sort=True)["blocks"]

            # Also get plain text with preserved layout
            text = page.get_text("text", sort=True)
        else:
            blocks = []
            text = page.get_text("text")

        # Clean up text
        text = self._clean_text(text)

        # Process blocks for structure information
        processed_blocks = []
        for block in blocks:
            if block.get("type") == 0:  # Text block
                processed_blocks.append({
                    "type": "text",
                    "bbox": block.get("bbox", []),
                    "lines": self._extract_lines(block),
                })
            elif block.get("type") == 1:  # Image block
                processed_blocks.append({
                    "type": "image",
                    "bbox": block.get("bbox", []),
                    "width": block.get("width", 0),
                    "height": block.get("height", 0),
                })

        return PageContent(
            page_number=page_number,
            text=text,
            blocks=processed_blocks
        )

    def _extract_lines(self, block: dict) -> list[dict]:
        """Extract line information from a text block."""
        lines = []
        for line in block.get("lines", []):
            line_text = ""
            font_sizes = []
            is_bold = False

            for span in line.get("spans", []):
                line_text += span.get("text", "")
                font_sizes.append(span.get("size", 12))
                # Check for bold font
                font_name = span.get("font", "").lower()
                if "bold" in font_name or "black" in font_name:
                    is_bold = True

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

            lines.append({
                "text": line_text,
                "bbox": line.get("bbox", []),
                "font_size": avg_font_size,
                "is_bold": is_bold,
            })

        return lines

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        - Normalize whitespace
        - Remove excessive blank lines
        - Fix common extraction artifacts
        """
        # Replace multiple spaces with single space (but preserve newlines)
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Normalize whitespace within line
            cleaned_line = ' '.join(line.split())
            cleaned_lines.append(cleaned_line)

        # Remove excessive blank lines (more than 2 consecutive)
        result_lines = []
        blank_count = 0

        for line in cleaned_lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        return '\n'.join(result_lines)

    def extract_text_only(self, pdf_path: str | Path) -> str:
        """
        Quick extraction of just the text content.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Concatenated text from all pages.
        """
        doc = self.extract(pdf_path)
        return doc.get_full_text()
