"""
Tests for the ImageExtractor module.

Tests cover:
- Image detection and extraction
- Size filtering
- Deduplication
- ExtractedImage dataclass
"""

import pytest
from pathlib import Path
from src.images import ImageExtractor, ExtractedImage


class TestExtractedImage:
    """Tests for the ExtractedImage dataclass."""

    def test_basic_creation(self):
        """Test basic image creation."""
        img = ExtractedImage(
            page_number=1,
            index=0,
            width=800,
            height=600,
            bbox=(0, 0, 800, 600)
        )
        assert img.page_number == 1
        assert img.index == 0
        assert img.width == 800
        assert img.height == 600
        assert img.extension == "png"  # Default

    def test_with_all_fields(self):
        """Test image with all optional fields."""
        img = ExtractedImage(
            page_number=5,
            index=2,
            width=1024,
            height=768,
            bbox=(100, 200, 300, 400),
            image_path="images/page5_img2.jpeg",
            image_hash="abc123def456",
            extension="jpeg"
        )
        assert img.bbox == (100, 200, 300, 400)
        assert img.image_path == "images/page5_img2.jpeg"
        assert img.image_hash == "abc123def456"
        assert img.extension == "jpeg"

    def test_to_dict(self):
        """Test dictionary conversion."""
        img = ExtractedImage(
            page_number=3,
            index=1,
            width=640,
            height=480,
            bbox=(0, 0, 640, 480),
            image_path="test.png",
            image_hash="hash123",
            extension="png"
        )
        d = img.to_dict()

        assert d["page"] == 3
        assert d["index"] == 1
        assert d["width"] == 640
        assert d["height"] == 480
        assert d["bbox"] == (0, 0, 640, 480)
        assert d["path"] == "test.png"
        assert d["hash"] == "hash123"
        assert d["extension"] == "png"

    def test_default_optional_fields(self):
        """Test that optional fields default correctly."""
        img = ExtractedImage(
            page_number=1,
            index=0,
            width=100,
            height=100,
            bbox=(0, 0, 100, 100)
        )
        assert img.image_path is None
        assert img.image_hash is None
        assert img.image_bytes is None
        assert img.extension == "png"

    def test_to_markdown_reference_with_path(self):
        """Test Markdown reference generation with path."""
        img = ExtractedImage(
            page_number=2,
            index=0,
            width=400,
            height=300,
            bbox=(0, 0, 400, 300),
            image_path="images/fig1.png"
        )
        md = img.to_markdown_reference("Figure 1")
        assert md == "![Figure 1](images/fig1.png)"

    def test_to_markdown_reference_without_path(self):
        """Test Markdown reference generation without path."""
        img = ExtractedImage(
            page_number=2,
            index=0,
            width=400,
            height=300,
            bbox=(0, 0, 400, 300)
        )
        md = img.to_markdown_reference()
        assert "[Image from page 2]" in md


class TestImageExtractor:
    """Tests for the ImageExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create an ImageExtractor with default settings."""
        return ImageExtractor()

    @pytest.fixture
    def strict_extractor(self):
        """Create an ImageExtractor with strict size requirements."""
        return ImageExtractor(min_width=500, min_height=500)

    @pytest.fixture
    def lenient_extractor(self):
        """Create an ImageExtractor with lenient size requirements."""
        return ImageExtractor(min_width=10, min_height=10)

    def test_default_initialization(self, extractor):
        """Test default initialization values."""
        assert extractor.min_width == 50
        assert extractor.min_height == 50
        assert extractor.deduplicate is True

    def test_custom_initialization(self):
        """Test custom initialization values."""
        extractor = ImageExtractor(min_width=200, min_height=300, deduplicate=False)
        assert extractor.min_width == 200
        assert extractor.min_height == 300
        assert extractor.deduplicate is False

    def test_bbox_overlaps_true(self, extractor):
        """Test bbox overlap detection - overlapping."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 150, 150)
        assert extractor._bbox_overlaps(bbox1, bbox2) is True

    def test_bbox_overlaps_false(self, extractor):
        """Test bbox overlap detection - not overlapping."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (100, 100, 150, 150)
        assert extractor._bbox_overlaps(bbox1, bbox2) is False

    def test_bbox_overlaps_touching(self, extractor):
        """Test bbox overlap detection - touching edges count as overlapping."""
        bbox1 = (0, 0, 100, 100)
        bbox2 = (100, 0, 200, 100)
        # Touching edges are considered as overlapping in this implementation
        assert extractor._bbox_overlaps(bbox1, bbox2) is True

    def test_bbox_overlaps_contained(self, extractor):
        """Test bbox overlap detection - one contains other."""
        bbox1 = (0, 0, 200, 200)
        bbox2 = (50, 50, 100, 100)
        assert extractor._bbox_overlaps(bbox1, bbox2) is True

    def test_bbox_overlaps_invalid(self, extractor):
        """Test bbox overlap with invalid bboxes."""
        assert extractor._bbox_overlaps((0, 0, 100), (0, 0, 100, 100)) is False
        assert extractor._bbox_overlaps((), (0, 0, 100, 100)) is False


class TestImageExtractorIntegration:
    """Integration tests for ImageExtractor with real PDF."""

    @pytest.fixture
    def pdf_path(self):
        """Path to test PDF."""
        import os
        path = "pdfs/Pruefungsordnung_BSc_Inf_2024.pdf"
        if not os.path.exists(path):
            pytest.skip(f"Test PDF not found: {path}")
        return path

    def test_extract_images_from_pdf(self, pdf_path):
        """Test basic image extraction."""
        extractor = ImageExtractor(min_width=100, min_height=100)
        images = extractor.extract_from_pdf(pdf_path)

        # Should find at least some images
        assert len(images) >= 1

        # All images should meet minimum size
        for img in images:
            assert img.width >= 100
            assert img.height >= 100

    def test_extract_images_with_output_dir(self, pdf_path, tmp_path):
        """Test image extraction with file output."""
        output_dir = tmp_path / "images"
        extractor = ImageExtractor(min_width=100, min_height=100)
        images = extractor.extract_from_pdf(pdf_path, output_dir=output_dir)

        # Check that output directory was created
        assert output_dir.exists()

        # Check that images were saved (should have paths)
        saved_images = [img for img in images if img.image_path]
        assert len(saved_images) > 0

    def test_extract_images_deduplication(self, pdf_path):
        """Test that duplicate images are filtered."""
        extractor = ImageExtractor(min_width=50, min_height=50, deduplicate=True)
        images = extractor.extract_from_pdf(pdf_path)

        # Check that no two images have the same hash
        hashes = [img.image_hash for img in images if img.image_hash]
        assert len(hashes) == len(set(hashes)), "Duplicate images found"

    def test_extract_images_no_deduplication(self, pdf_path):
        """Test extraction without deduplication."""
        extractor = ImageExtractor(min_width=50, min_height=50, deduplicate=False)
        images = extractor.extract_from_pdf(pdf_path)

        # Should still work, may have duplicates
        assert isinstance(images, list)

    def test_extract_images_page_numbers(self, pdf_path):
        """Test that page numbers are correct."""
        extractor = ImageExtractor(min_width=100, min_height=100)
        images = extractor.extract_from_pdf(pdf_path)

        for img in images:
            # Page numbers should be positive integers
            assert img.page_number >= 1
            assert isinstance(img.page_number, int)

    def test_strict_size_filter(self, pdf_path):
        """Test strict size filtering reduces count."""
        lenient = ImageExtractor(min_width=50, min_height=50)
        strict = ImageExtractor(min_width=500, min_height=500)

        lenient_images = lenient.extract_from_pdf(pdf_path)
        strict_images = strict.extract_from_pdf(pdf_path)

        # Strict should find fewer or equal images
        assert len(strict_images) <= len(lenient_images)

    def test_extract_specific_pages(self, pdf_path):
        """Test extracting from specific pages."""
        extractor = ImageExtractor(min_width=50, min_height=50)

        # Extract from pages 1-5
        images = extractor.extract_from_pdf(pdf_path, pages=[1, 2, 3, 4, 5])

        for img in images:
            assert img.page_number in [1, 2, 3, 4, 5]

    def test_render_page_as_image(self, pdf_path, tmp_path):
        """Test rendering full page as image."""
        extractor = ImageExtractor()
        output_path = tmp_path / "page1.png"

        image_bytes = extractor.extract_page_as_image(
            pdf_path,
            page_number=1,
            output_path=output_path,
            dpi=150
        )

        assert image_bytes is not None
        assert len(image_bytes) > 0
        assert output_path.exists()

    def test_render_invalid_page(self, pdf_path):
        """Test rendering invalid page number."""
        extractor = ImageExtractor()

        result = extractor.extract_page_as_image(pdf_path, page_number=9999)
        assert result is None
