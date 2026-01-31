"""
PDF to Images Converter

Converts PDF pages to base64-encoded images for Vision LLM processing.
Uses PyMuPDF (fitz) for high-quality rendering.
"""

import base64
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Generator
import io


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
    - Memory-efficient streaming for large documents
    - Support for page ranges
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

    def get_page_count(self, pdf_path: str | Path) -> int:
        """Get the total number of pages in a PDF."""
        with fitz.open(pdf_path) as doc:
            return len(doc)

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

    def render_all_pages(
        self,
        pdf_path: str | Path,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> Generator[PageImage, None, None]:
        """
        Render all pages as a generator (memory-efficient).

        Args:
            pdf_path: Path to the PDF file
            start_page: First page to render (1-indexed)
            end_page: Last page to render (None = all pages)

        Yields:
            PageImage objects for each page
        """
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            end = end_page or total_pages

            for page_num in range(start_page, min(end + 1, total_pages + 1)):
                page = doc[page_num - 1]
                yield self._render_page_object(page, page_num)

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

    def render_thumbnail(
        self,
        pdf_path: str | Path,
        max_size: int = 400,
    ) -> PageImage:
        """
        Render a thumbnail of the first page (for document preview).

        Args:
            pdf_path: Path to the PDF file
            max_size: Maximum dimension in pixels

        Returns:
            PageImage with thumbnail
        """
        with fitz.open(pdf_path) as doc:
            page = doc[0]

            # Calculate zoom for thumbnail
            rect = page.rect
            scale = max_size / max(rect.width, rect.height)
            mat = fitz.Matrix(scale, scale)

            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_bytes = pix.tobytes(output=self.image_format)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            return PageImage(
                page_number=1,
                image_base64=img_base64,
                width=pix.width,
                height=pix.height,
                mime_type=f"image/{self.image_format}",
            )

    def _render_page_object(self, page: fitz.Page, page_number: int) -> PageImage:
        """Internal method to render a fitz.Page object."""
        # Create transformation matrix
        mat = fitz.Matrix(self.zoom, self.zoom)

        # Render to pixmap
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Resize if too large
        if pix.width > self.max_dimension or pix.height > self.max_dimension:
            scale = self.max_dimension / max(pix.width, pix.height)
            new_width = int(pix.width * scale)
            new_height = int(pix.height * scale)

            # Re-render with adjusted matrix
            adjusted_zoom = self.zoom * scale
            mat = fitz.Matrix(adjusted_zoom, adjusted_zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to bytes
        img_bytes = pix.tobytes(output=self.image_format)

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

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


def estimate_api_cost(
    page_count: int,
    model: str = "claude-sonnet-4-20250514",
    avg_input_tokens_per_page: int = 1500,
    avg_output_tokens_per_page: int = 800,
) -> dict:
    """
    Estimate the API cost for processing a document.

    Args:
        page_count: Number of pages
        model: Model to use
        avg_input_tokens_per_page: Estimated input tokens per page (image + prompt)
        avg_output_tokens_per_page: Estimated output tokens per page

    Returns:
        Dictionary with cost estimates
    """
    # Pricing (as of 2024, may change)
    pricing = {
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # per 1M tokens
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    if model not in pricing:
        return {"error": f"Unknown model: {model}"}

    prices = pricing[model]

    # Context analysis (first pass with all pages)
    context_input = page_count * 500  # Rough estimate for thumbnails/overview
    context_output = 500

    # Page-by-page extraction
    page_input = page_count * avg_input_tokens_per_page
    page_output = page_count * avg_output_tokens_per_page

    total_input = context_input + page_input
    total_output = context_output + page_output

    cost_input = (total_input / 1_000_000) * prices["input"]
    cost_output = (total_output / 1_000_000) * prices["output"]
    total_cost = cost_input + cost_output

    return {
        "model": model,
        "page_count": page_count,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_breakdown": {
            "input": round(cost_input, 4),
            "output": round(cost_output, 4),
        }
    }
