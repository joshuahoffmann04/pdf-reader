"""
PDF Utilities for Section Extraction.

This module provides PDF handling functionality:
- PDF validation and metadata extraction
- Page rendering to base64-encoded images
- Memory-efficient batch processing

Uses PyMuPDF (fitz) for high-quality PDF rendering.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union

import fitz  # PyMuPDF

from .exceptions import PDFNotFoundError, PDFCorruptedError, PageRenderError


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class PageImage:
    """
    Represents a single rendered PDF page as a base64 image.

    Attributes:
        page_number: Page number (1-indexed, matches PDF page number)
        image_base64: Base64-encoded PNG image data
        width: Image width in pixels
        height: Image height in pixels
        mime_type: MIME type (default: "image/png")
    """

    page_number: int
    image_base64: str
    width: int
    height: int
    mime_type: str = "image/png"

    def to_data_url(self) -> str:
        """Convert to data URL for embedding in HTML/Markdown."""
        return f"data:{self.mime_type};base64,{self.image_base64}"

    def to_api_format(self) -> dict:
        """
        Convert to OpenAI Vision API format.

        Returns:
            Dictionary ready for use in API image_url field
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.mime_type};base64,{self.image_base64}",
                "detail": "high",
            },
        }


@dataclass(frozen=True)
class PDFInfo:
    """
    Basic PDF document information.

    Attributes:
        path: Path to the PDF file
        page_count: Total number of pages
        title: Document title from metadata (may be empty)
        author: Document author from metadata (may be empty)
        file_size_bytes: File size in bytes
    """

    path: str
    page_count: int
    title: str
    author: str
    file_size_bytes: int

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return self.file_size_bytes / (1024 * 1024)


# =============================================================================
# PDF RENDERER CLASS
# =============================================================================


class PDFRenderer:
    """
    High-quality PDF to image renderer.

    Converts PDF pages to base64-encoded PNG images suitable for Vision LLM APIs.
    Optimized for accuracy and memory efficiency.

    Usage:
        renderer = PDFRenderer(dpi=150)

        # Get document info
        info = renderer.get_info("document.pdf")
        print(f"Pages: {info.page_count}")

        # Render single page
        image = renderer.render_page("document.pdf", page_number=1)

        # Render all pages (generator)
        for image in renderer.render_all("document.pdf"):
            process(image)

        # Render specific pages
        images = renderer.render_batch("document.pdf", [1, 5, 10])
    """

    def __init__(
        self,
        dpi: int = 150,
        max_dimension: int = 2000,
        image_format: str = "png",
    ):
        """
        Initialize the PDF renderer.

        Args:
            dpi: Resolution for rendering (150 is good balance of quality/size)
            max_dimension: Maximum width/height in pixels (prevents memory issues)
            image_format: Output format ("png" or "jpeg")
        """
        self.dpi = dpi
        self.max_dimension = max_dimension
        self.image_format = image_format
        self.zoom = dpi / 72  # 72 is default PDF DPI

    def get_info(self, pdf_path: Union[str, Path]) -> PDFInfo:
        """
        Get basic information about a PDF document.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            PDFInfo with document metadata

        Raises:
            PDFNotFoundError: If file doesn't exist
            PDFCorruptedError: If file can't be opened
        """
        path = Path(pdf_path)

        if not path.exists():
            raise PDFNotFoundError(str(path))

        try:
            with fitz.open(path) as doc:
                metadata = doc.metadata or {}
                return PDFInfo(
                    path=str(path),
                    page_count=len(doc),
                    title=metadata.get("title", "") or "",
                    author=metadata.get("author", "") or "",
                    file_size_bytes=path.stat().st_size,
                )
        except Exception as e:
            if "cannot open" in str(e).lower() or "invalid" in str(e).lower():
                raise PDFCorruptedError(str(path), e)
            raise

    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """
        Get the total number of pages in a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages

        Raises:
            PDFNotFoundError: If file doesn't exist
            PDFCorruptedError: If file can't be opened
        """
        return self.get_info(pdf_path).page_count

    def render_page(
        self,
        pdf_path: Union[str, Path],
        page_number: int,
    ) -> PageImage:
        """
        Render a single page to an image.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (1-indexed)

        Returns:
            PageImage with base64-encoded image data

        Raises:
            PDFNotFoundError: If file doesn't exist
            PDFCorruptedError: If file can't be opened
            PageRenderError: If page can't be rendered
        """
        path = Path(pdf_path)

        if not path.exists():
            raise PDFNotFoundError(str(path))

        try:
            with fitz.open(path) as doc:
                if page_number < 1 or page_number > len(doc):
                    raise PageRenderError(
                        page_number,
                        str(path),
                        ValueError(f"Page {page_number} out of range (1-{len(doc)})"),
                    )

                page = doc[page_number - 1]  # 0-indexed internally
                return self._render_page_object(page, page_number)

        except PageRenderError:
            raise
        except Exception as e:
            raise PageRenderError(page_number, str(path), e)

    def render_all(
        self,
        pdf_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> Generator[PageImage, None, None]:
        """
        Render all pages as a generator (memory-efficient).

        Args:
            pdf_path: Path to the PDF file
            start_page: First page to render (1-indexed, default: 1)
            end_page: Last page to render (None = all pages)

        Yields:
            PageImage for each page in order

        Raises:
            PDFNotFoundError: If file doesn't exist
            PDFCorruptedError: If file can't be opened
        """
        path = Path(pdf_path)

        if not path.exists():
            raise PDFNotFoundError(str(path))

        try:
            with fitz.open(path) as doc:
                total = len(doc)
                end = min(end_page or total, total)

                for page_num in range(start_page, end + 1):
                    page = doc[page_num - 1]
                    yield self._render_page_object(page, page_num)

        except Exception as e:
            if "cannot open" in str(e).lower():
                raise PDFCorruptedError(str(path), e)
            raise

    def render_batch(
        self,
        pdf_path: Union[str, Path],
        page_numbers: list[int],
    ) -> list[PageImage]:
        """
        Render specific pages.

        Args:
            pdf_path: Path to the PDF file
            page_numbers: List of page numbers to render (1-indexed)

        Returns:
            List of PageImage objects (in requested order)

        Raises:
            PDFNotFoundError: If file doesn't exist
            PDFCorruptedError: If file can't be opened
            PageRenderError: If any page can't be rendered
        """
        path = Path(pdf_path)

        if not path.exists():
            raise PDFNotFoundError(str(path))

        images = []
        try:
            with fitz.open(path) as doc:
                total = len(doc)

                for page_num in page_numbers:
                    if page_num < 1 or page_num > total:
                        raise PageRenderError(
                            page_num,
                            str(path),
                            ValueError(f"Page {page_num} out of range (1-{total})"),
                        )

                    page = doc[page_num - 1]
                    images.append(self._render_page_object(page, page_num))

        except PageRenderError:
            raise
        except Exception as e:
            if "cannot open" in str(e).lower():
                raise PDFCorruptedError(str(path), e)
            raise

        return images

    def _render_page_object(self, page: fitz.Page, page_number: int) -> PageImage:
        """
        Internal method to render a fitz.Page object.

        Args:
            page: PyMuPDF page object
            page_number: Page number for metadata

        Returns:
            PageImage with rendered image
        """
        # Create transformation matrix for desired DPI
        mat = fitz.Matrix(self.zoom, self.zoom)

        # Render to pixmap (image buffer)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Resize if too large (prevents memory issues)
        if pix.width > self.max_dimension or pix.height > self.max_dimension:
            scale = self.max_dimension / max(pix.width, pix.height)
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


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def validate_pdf(pdf_path: Union[str, Path]) -> PDFInfo:
    """
    Validate a PDF file and return its info.

    Convenience function that creates a temporary renderer.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        PDFInfo with document metadata

    Raises:
        PDFNotFoundError: If file doesn't exist
        PDFCorruptedError: If file can't be opened
    """
    return PDFRenderer().get_info(pdf_path)


def estimate_api_cost(
    page_count: int,
    model: str = "gpt-4o",
    avg_input_tokens_per_page: int = 1500,
    avg_output_tokens_per_page: int = 300,
) -> dict:
    """
    Estimate the API cost for processing a document.

    This is a rough estimate based on typical token usage.
    Actual costs may vary based on image complexity and content.

    Args:
        page_count: Number of pages to process
        model: Model to use ("gpt-4o" or "gpt-4o-mini")
        avg_input_tokens_per_page: Estimated input tokens per page
        avg_output_tokens_per_page: Estimated output tokens per page

    Returns:
        Dictionary with cost estimates
    """
    # Pricing per 1M tokens (as of 2024)
    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    if model not in pricing:
        return {"error": f"Unknown model: {model}"}

    prices = pricing[model]

    # Phase 1: Page scanning (each page individually)
    scan_input = page_count * avg_input_tokens_per_page
    scan_output = page_count * 100  # Small JSON output per page

    # Phase 3: Context extraction (first few pages)
    context_pages = min(5, page_count)
    context_input = context_pages * avg_input_tokens_per_page
    context_output = 500

    # Phase 4: Section extraction (all pages, but grouped by section)
    # Estimate ~same as page count since pages are reused across sections
    extract_input = page_count * avg_input_tokens_per_page
    extract_output = page_count * avg_output_tokens_per_page

    # Totals
    total_input = scan_input + context_input + extract_input
    total_output = scan_output + context_output + extract_output

    cost_input = (total_input / 1_000_000) * prices["input"]
    cost_output = (total_output / 1_000_000) * prices["output"]
    total_cost = cost_input + cost_output

    return {
        "model": model,
        "page_count": page_count,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_cost_usd": round(total_cost, 4),
        "breakdown": {
            "phase1_scan": {
                "input_tokens": scan_input,
                "output_tokens": scan_output,
            },
            "phase3_context": {
                "input_tokens": context_input,
                "output_tokens": context_output,
            },
            "phase4_extract": {
                "input_tokens": extract_input,
                "output_tokens": extract_output,
            },
        },
    }
