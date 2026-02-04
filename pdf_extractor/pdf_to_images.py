"""
PDF to Images Converter

Converts PDF pages to base64-encoded images for Vision LLM processing.
Uses PyMuPDF (fitz) for high-quality rendering.
"""

import base64
import fitz  # PyMuPDF
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PageImage:
    """Represents a single page rendered as an image."""
    page_number: int  # 1-indexed
    image_base64: str
    width: int
    height: int
    mime_type: str = "image/png"


class PDFToImages:
    """
    Converts PDF documents to images for Vision LLM processing.

    Features:
    - High-quality rendering with configurable DPI
    - Batch rendering for selected pages
    - Automatic image optimization
    """

    def __init__(
        self,
        dpi: int = 150,
        image_format: str = "png",
        max_dimension: int = 2000,
    ):
        """
        Initialize the converter.

        Args:
            dpi: Resolution for rendering (higher = better quality, larger files)
            image_format: Output format ("png" or "jpeg")
            max_dimension: Maximum width/height in pixels (for optimization)
        """
        self.dpi = dpi
        self.image_format = image_format
        self.max_dimension = max_dimension
        self.zoom = dpi / 72  # 72 is the default PDF DPI

    def render_page(
        self,
        pdf_path: str | Path,
        page_number: int,
    ) -> PageImage:
        """
        Render a single page to an image.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed)

        Returns:
            PageImage with base64-encoded image data
        """
        with fitz.open(pdf_path) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"Page {page_number} out of range (1-{len(doc)})")

            page = doc[page_number - 1]  # 0-indexed internally
            return self._render_page_object(page, page_number)

    def render_pages_batch(
        self,
        pdf_path: str | Path,
        page_numbers: list[int],
    ) -> list[PageImage]:
        """
        Render specific pages.

        Args:
            pdf_path: Path to the PDF file
            page_numbers: List of page numbers (1-indexed)

        Returns:
            List of PageImage objects
        """
        images = []
        with fitz.open(pdf_path) as doc:
            for page_num in page_numbers:
                if 1 <= page_num <= len(doc):
                    page = doc[page_num - 1]
                    images.append(self._render_page_object(page, page_num))
        return images

    def _render_page_object(self, page: fitz.Page, page_number: int) -> PageImage:
        """Internal method to render a fitz.Page object."""
        # Create transformation matrix
        mat = fitz.Matrix(self.zoom, self.zoom)

        # Render to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Resize if too large
        if pix.width > self.max_dimension or pix.height > self.max_dimension:
            scale = self.max_dimension / max(pix.width, pix.height)

            # Re-render with adjusted matrix
            adjusted_zoom = self.zoom * scale
            mat = fitz.Matrix(adjusted_zoom, adjusted_zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to bytes
        img_bytes = pix.tobytes(output=self.image_format)

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return PageImage(
            page_number=page_number,
            image_base64=img_base64,
            width=pix.width,
            height=pix.height,
            mime_type=f"image/{self.image_format}",
        )

    def get_document_info(self, pdf_path: str | Path) -> dict:
        """Get basic document information."""
        with fitz.open(pdf_path) as doc:
            metadata = doc.metadata
            return {
                "page_count": len(doc),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
            }
