"""
Image Extraction Module

Extracts images from PDF files using PyMuPDF.
Saves images to disk and provides metadata for reference.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import hashlib


@dataclass
class ExtractedImage:
    """Represents an extracted image from a PDF."""
    page_number: int
    index: int  # Index of image on the page
    width: int
    height: int
    bbox: tuple  # Bounding box (x0, y0, x1, y1)
    image_path: Optional[str] = None  # Path where image was saved
    image_hash: Optional[str] = None  # Hash for deduplication
    image_bytes: Optional[bytes] = None  # Raw image data
    extension: str = "png"

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "page": self.page_number,
            "index": self.index,
            "width": self.width,
            "height": self.height,
            "bbox": self.bbox,
            "path": self.image_path,
            "hash": self.image_hash,
            "extension": self.extension
        }

    def to_markdown_reference(self, alt_text: str = "") -> str:
        """Generate a Markdown image reference."""
        if not alt_text:
            alt_text = f"Image from page {self.page_number}"

        if self.image_path:
            return f"![{alt_text}]({self.image_path})"
        return f"[{alt_text}]"


class ImageExtractor:
    """
    Extracts images from PDF files.

    Features:
    - Extracts all embedded images
    - Deduplication based on image hash
    - Multiple output formats
    - Size filtering
    """

    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50,
        deduplicate: bool = True
    ):
        """
        Initialize the image extractor.

        Args:
            min_width: Minimum image width to extract (pixels).
            min_height: Minimum image height to extract (pixels).
            deduplicate: If True, skip duplicate images based on hash.
        """
        self.min_width = min_width
        self.min_height = min_height
        self.deduplicate = deduplicate
        self._seen_hashes = set()

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        output_dir: Optional[str | Path] = None,
        pages: Optional[list[int]] = None
    ) -> list[ExtractedImage]:
        """
        Extract all images from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to save extracted images. If None, images
                        are not saved but bytes are stored in ExtractedImage.
            pages: Optional list of page numbers to process (1-indexed).

        Returns:
            List of ExtractedImage objects.
        """
        pdf_path = Path(pdf_path)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Reset seen hashes for new extraction
        self._seen_hashes = set()

        images = []
        doc = fitz.open(pdf_path)

        try:
            page_range = range(len(doc))
            if pages:
                page_range = [p - 1 for p in pages if 0 < p <= len(doc)]

            for page_idx in page_range:
                page = doc[page_idx]
                page_images = self._extract_from_page(
                    doc, page, page_idx + 1, output_dir, pdf_path.stem
                )
                images.extend(page_images)

        finally:
            doc.close()

        return images

    def _extract_from_page(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_number: int,
        output_dir: Optional[Path],
        pdf_name: str
    ) -> list[ExtractedImage]:
        """Extract images from a single page."""
        images = []

        # Get list of images on the page
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            try:
                xref = img_info[0]  # Image XREF

                # Extract base image
                base_image = doc.extract_image(xref)

                if not base_image:
                    continue

                image_bytes = base_image["image"]
                width = base_image["width"]
                height = base_image["height"]
                ext = base_image["ext"]

                # Check minimum size
                if width < self.min_width or height < self.min_height:
                    continue

                # Calculate hash for deduplication
                image_hash = hashlib.md5(image_bytes).hexdigest()

                if self.deduplicate and image_hash in self._seen_hashes:
                    continue

                self._seen_hashes.add(image_hash)

                # Get image position on page
                bbox = self._get_image_bbox(page, img_info)

                # Create extracted image object
                extracted = ExtractedImage(
                    page_number=page_number,
                    index=img_idx,
                    width=width,
                    height=height,
                    bbox=bbox,
                    image_hash=image_hash,
                    extension=ext
                )

                # Save to disk if output directory specified
                if output_dir:
                    filename = f"{pdf_name}_page{page_number}_img{img_idx}.{ext}"
                    image_path = output_dir / filename

                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    extracted.image_path = str(image_path)
                else:
                    # Store bytes if not saving to disk
                    extracted.image_bytes = image_bytes

                images.append(extracted)

            except Exception as e:
                print(f"Warning: Failed to extract image {img_idx} from page {page_number}: {e}")
                continue

        return images

    def _get_image_bbox(self, page: fitz.Page, img_info: tuple) -> tuple:
        """
        Get the bounding box of an image on the page.

        Args:
            page: PyMuPDF page object.
            img_info: Image info tuple from get_images().

        Returns:
            Bounding box tuple (x0, y0, x1, y1) or empty tuple if not found.
        """
        try:
            # Get image rectangles on the page
            xref = img_info[0]

            # Try to find the image rectangle
            for img_rect in page.get_image_rects(xref):
                return (img_rect.x0, img_rect.y0, img_rect.x1, img_rect.y1)

        except Exception:
            pass

        return ()

    def get_images_in_region(
        self,
        pdf_path: str | Path,
        page_number: int,
        bbox: tuple
    ) -> list[ExtractedImage]:
        """
        Get images that fall within a specific region.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-indexed).
            bbox: Region bounding box (x0, y0, x1, y1).

        Returns:
            List of images within the region.
        """
        all_images = self.extract_from_pdf(pdf_path, pages=[page_number])

        region_images = []
        for img in all_images:
            if img.bbox and self._bbox_overlaps(img.bbox, bbox):
                region_images.append(img)

        return region_images

    def _bbox_overlaps(self, bbox1: tuple, bbox2: tuple) -> bool:
        """Check if two bounding boxes overlap."""
        if len(bbox1) != 4 or len(bbox2) != 4:
            return False

        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        # Check for no overlap
        if x1_1 < x0_2 or x1_2 < x0_1:
            return False
        if y1_1 < y0_2 or y1_2 < y0_1:
            return False

        return True

    def extract_page_as_image(
        self,
        pdf_path: str | Path,
        page_number: int,
        output_path: Optional[str | Path] = None,
        dpi: int = 150
    ) -> Optional[bytes]:
        """
        Render an entire PDF page as an image.

        Useful for pages with complex layouts or diagrams.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-indexed).
            output_path: Optional path to save the image.
            dpi: Resolution in dots per inch.

        Returns:
            Image bytes if successful, None otherwise.
        """
        doc = fitz.open(pdf_path)

        try:
            if page_number > len(doc):
                return None

            page = doc[page_number - 1]

            # Calculate zoom factor for desired DPI
            zoom = dpi / 72  # 72 is the default PDF DPI
            mat = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat)

            # Get PNG bytes
            image_bytes = pix.tobytes("png")

            if output_path:
                with open(output_path, "wb") as f:
                    f.write(image_bytes)

            return image_bytes

        finally:
            doc.close()
