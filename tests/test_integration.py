"""
Integration tests with the real PDF document.
"""

import pytest
import sys
import os

sys.path.insert(0, '.')

from src.extractor.pdf_extractor import PDFExtractor
from src.parser.document_parser import DocumentParser
from src.tables.table_extractor import TableExtractor
from src.images.image_extractor import ImageExtractor


# Skip if PDF not available
PDF_PATH = "Pruefungsordnung_BSc_Inf_2024.pdf"
SKIP_REASON = f"PDF file not found: {PDF_PATH}"


@pytest.fixture
def pdf_text():
    """Extract text from the real PDF."""
    if not os.path.exists(PDF_PATH):
        pytest.skip(SKIP_REASON)

    extractor = PDFExtractor()
    pdf_doc = extractor.extract(PDF_PATH)
    return pdf_doc.get_full_text()


@pytest.fixture
def parsed_doc(pdf_text):
    """Parse the real PDF."""
    parser = DocumentParser()
    return parser.parse(pdf_text, title="Test Document")


class TestPDFExtraction:
    """Integration tests for PDF extraction."""

    def test_pdf_extraction_succeeds(self):
        """Test that PDF extraction completes without errors."""
        if not os.path.exists(PDF_PATH):
            pytest.skip(SKIP_REASON)

        extractor = PDFExtractor()
        pdf_doc = extractor.extract(PDF_PATH)

        assert pdf_doc is not None
        assert pdf_doc.total_pages == 56
        assert len(pdf_doc.get_full_text()) > 100000  # Should be substantial

    def test_table_extraction_succeeds(self):
        """Test that table extraction completes without errors."""
        if not os.path.exists(PDF_PATH):
            pytest.skip(SKIP_REASON)

        extractor = TableExtractor()
        tables = extractor.extract_from_pdf(PDF_PATH)

        assert len(tables) > 0
        # Should find multiple tables in the module lists

    def test_image_extraction_succeeds(self):
        """Test that image extraction completes without errors."""
        if not os.path.exists(PDF_PATH):
            pytest.skip(SKIP_REASON)

        extractor = ImageExtractor(min_width=50, min_height=50)
        images = extractor.extract_from_pdf(PDF_PATH)

        assert len(images) >= 1  # At least the diagram on page 2


class TestDocumentStructure:
    """Integration tests for document structure parsing."""

    def test_correct_number_of_chapters(self, parsed_doc):
        """Test that exactly 4 chapters are found."""
        assert len(parsed_doc.chapters) == 4

    def test_chapter_identifiers(self, parsed_doc):
        """Test that chapters have correct Roman numeral identifiers."""
        chapter_ids = [c.id for c in parsed_doc.chapters]
        assert chapter_ids == ["I", "II", "III", "IV"]

    def test_chapter_titles(self, parsed_doc):
        """Test that chapters have expected titles."""
        titles = {c.id: c.title for c in parsed_doc.chapters}

        assert "Allgemeines" in titles["I"]
        assert "Studienbezogene" in titles["II"]
        assert "Prüfungsbezogene" in titles["III"]
        assert "Schlussbestimmungen" in titles["IV"]

    def test_correct_number_of_main_sections(self, parsed_doc):
        """Test that exactly 40 main sections (§1-§40) are found."""
        all_sections = parsed_doc.get_all_main_sections()
        assert len(all_sections) == 40

    def test_section_distribution_chapter_i(self, parsed_doc):
        """Test that Chapter I has §1-§3."""
        chapter_i = parsed_doc.chapters[0]
        section_numbers = sorted([s.number for s in chapter_i.sections])
        assert section_numbers == [1, 2, 3]

    def test_section_distribution_chapter_ii(self, parsed_doc):
        """Test that Chapter II has §4-§17."""
        chapter_ii = parsed_doc.chapters[1]
        section_numbers = sorted([s.number for s in chapter_ii.sections])
        assert section_numbers == list(range(4, 18))

    def test_section_distribution_chapter_iii(self, parsed_doc):
        """Test that Chapter III has §18-§38."""
        chapter_iii = parsed_doc.chapters[2]
        section_numbers = sorted([s.number for s in chapter_iii.sections])
        assert section_numbers == list(range(18, 39))

    def test_section_distribution_chapter_iv(self, parsed_doc):
        """Test that Chapter IV has §39-§40."""
        chapter_iv = parsed_doc.chapters[3]
        section_numbers = sorted([s.number for s in chapter_iv.sections])
        assert section_numbers == [39, 40]

    def test_correct_number_of_appendices(self, parsed_doc):
        """Test that exactly 5 appendices are found."""
        assert len(parsed_doc.appendices) == 5

    def test_appendix_identifiers(self, parsed_doc):
        """Test that appendices have correct identifiers."""
        appendix_numbers = [a.number for a in parsed_doc.appendices]
        assert appendix_numbers == ["1", "2", "3", "4", "5"]

    def test_appendix_1_3_have_no_sections(self, parsed_doc):
        """Test that Anlage 1-3 have no sub-sections."""
        for appendix in parsed_doc.appendices[:3]:
            assert len(appendix.sections) == 0, f"Anlage {appendix.number} should have no sections"

    def test_appendix_4_has_4_sections(self, parsed_doc):
        """Test that Anlage 4 has exactly 4 sub-sections."""
        anlage_4 = parsed_doc.appendices[3]
        assert len(anlage_4.sections) == 4

        section_numbers = [s.number for s in anlage_4.sections]
        assert section_numbers == [1, 2, 3, 4]

    def test_appendix_5_has_10_sections(self, parsed_doc):
        """Test that Anlage 5 has exactly 10 sub-sections."""
        anlage_5 = parsed_doc.appendices[4]
        assert len(anlage_5.sections) == 10

        section_numbers = [s.number for s in anlage_5.sections]
        assert section_numbers == list(range(1, 11))

    def test_total_appendix_sections(self, parsed_doc):
        """Test that total appendix sections equals 14 (4 + 10)."""
        total = sum(len(a.sections) for a in parsed_doc.appendices)
        assert total == 14

    def test_ab_excerpts_exist(self, parsed_doc):
        """Test that AB excerpts are detected."""
        all_ab = parsed_doc.get_all_ab_excerpts()
        assert len(all_ab) > 0

    def test_sections_have_content(self, parsed_doc):
        """Test that main sections have non-empty content."""
        all_sections = parsed_doc.get_all_main_sections()

        for section in all_sections:
            assert len(section.content) > 0, f"{section.id} should have content"

    def test_section_titles_are_meaningful(self, parsed_doc):
        """Test that section titles are meaningful (not just numbers or fragments)."""
        all_sections = parsed_doc.get_all_main_sections()

        for section in all_sections:
            # Title should be at least 5 characters
            assert len(section.title) >= 5, f"{section.id} has too short title: '{section.title}'"

            # Title should not start with common inline reference prefixes
            invalid_prefixes = ["Abs", "Satz", "Nr", "Nummer"]
            for prefix in invalid_prefixes:
                assert not section.title.startswith(prefix), \
                    f"{section.id} has invalid title starting with '{prefix}': '{section.title}'"


class TestStatistics:
    """Test document statistics."""

    def test_statistics_accuracy(self, parsed_doc):
        """Test that statistics are accurate."""
        stats = parsed_doc.get_statistics()

        assert stats["chapters"] == 4
        assert stats["main_sections"] == 40
        assert stats["appendices"] == 5
        assert stats["appendix_sections"] == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
