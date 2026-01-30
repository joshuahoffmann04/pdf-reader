"""
Tests for the TableExtractor module.

Tests cover:
- Table detection strategies
- Multi-page table merging
- Header detection
- Page number filtering
- False positive filtering
"""

import pytest
from src.tables import TableExtractor, ExtractedTable


class TestExtractedTable:
    """Tests for the ExtractedTable dataclass."""

    def test_basic_table_creation(self):
        """Test creating a basic table."""
        table = ExtractedTable(
            page_number=1,
            rows=[["A", "B"], ["1", "2"]]
        )
        assert table.page_number == 1
        assert len(table.rows) == 2
        assert table.end_page == 1  # Should default to page_number

    def test_table_with_headers(self):
        """Test table with explicit headers."""
        table = ExtractedTable(
            page_number=5,
            rows=[["data1", "data2"]],
            headers=["Col A", "Col B"]
        )
        assert table.headers == ["Col A", "Col B"]
        assert table.rows == [["data1", "data2"]]

    def test_multi_page_table(self):
        """Test table spanning multiple pages."""
        table = ExtractedTable(
            page_number=10,
            rows=[["a", "b"], ["c", "d"]],
            end_page=12
        )
        assert table.page_number == 10
        assert table.end_page == 12

    def test_to_markdown_with_headers(self):
        """Test Markdown conversion with headers."""
        table = ExtractedTable(
            page_number=1,
            rows=[["Alice", "30"], ["Bob", "25"]],
            headers=["Name", "Age"]
        )
        md = table.to_markdown()

        assert "| Name | Age |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 30 |" in md
        assert "| Bob | 25 |" in md

    def test_to_markdown_without_headers(self):
        """Test Markdown uses first row as header when no explicit headers."""
        table = ExtractedTable(
            page_number=1,
            rows=[["Name", "Age"], ["Alice", "30"]]
        )
        md = table.to_markdown()

        assert "| Name | Age |" in md
        assert "| Alice | 30 |" in md

    def test_to_markdown_empty_table(self):
        """Test Markdown of empty table."""
        table = ExtractedTable(page_number=1, rows=[])
        assert table.to_markdown() == ""

    def test_to_dict(self):
        """Test dictionary conversion."""
        table = ExtractedTable(
            page_number=3,
            rows=[["x", "y"]],
            headers=["A", "B"],
            bbox=(0, 0, 100, 100),
            end_page=4
        )
        d = table.to_dict()

        assert d["page"] == 3
        assert d["end_page"] == 4
        assert d["headers"] == ["A", "B"]
        assert d["rows"] == [["x", "y"]]
        assert d["bbox"] == (0, 0, 100, 100)


class TestTableExtractor:
    """Tests for the TableExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a TableExtractor instance."""
        return TableExtractor()

    def test_initialization(self, extractor):
        """Test default initialization."""
        assert extractor.snap_tolerance == 3
        assert extractor.join_tolerance == 3

    def test_custom_tolerance(self):
        """Test custom tolerance values."""
        extractor = TableExtractor(snap_tolerance=5, join_tolerance=10)
        assert extractor.snap_tolerance == 5
        assert extractor.join_tolerance == 10

    def test_is_likely_real_table_valid(self, extractor):
        """Test that real tables are identified correctly."""
        table = ExtractedTable(
            page_number=1,
            rows=[
                ["Name", "Age", "City"],
                ["Alice", "30", "Berlin"],
                ["Bob", "25", "Munich"],
                ["Charlie", "35", "Hamburg"]
            ]
        )
        assert extractor._is_likely_real_table(table) is True

    def test_is_likely_real_table_single_row(self, extractor):
        """Test that single row is rejected."""
        table = ExtractedTable(
            page_number=1,
            rows=[["A", "B", "C"]]
        )
        assert extractor._is_likely_real_table(table) is False

    def test_is_likely_real_table_single_column(self, extractor):
        """Test that single column data is rejected."""
        table = ExtractedTable(
            page_number=1,
            rows=[
                ["Line 1", "", ""],
                ["Line 2", "", ""],
                ["Line 3", "", ""]
            ]
        )
        assert extractor._is_likely_real_table(table) is False

    def test_is_likely_real_table_long_text(self, extractor):
        """Test that tables with long text cells are rejected."""
        long_text = "This is a very long text that would typically be a paragraph rather than a table cell. " * 5
        table = ExtractedTable(
            page_number=1,
            rows=[
                [long_text, long_text],
                [long_text, long_text]
            ]
        )
        assert extractor._is_likely_real_table(table) is False

    def test_filter_page_number_rows(self, extractor):
        """Test that page number rows are filtered."""
        rows = [
            ["Header", "Data"],
            ["", "42", ""],  # Just a page number
            ["Value", "100"],
            ["", "", "15", ""]  # Another page number
        ]
        filtered = extractor._filter_page_number_rows(rows)

        # Should keep only non-page-number rows
        assert len(filtered) == 2
        assert ["Header", "Data"] in filtered
        assert ["Value", "100"] in filtered

    def test_looks_like_header_with_text(self, extractor):
        """Test header detection with text cells."""
        row = ["Name", "Age", "City", "Country"]
        assert extractor._looks_like_header(row) is True

    def test_looks_like_header_numeric(self, extractor):
        """Test header detection rejects numeric rows."""
        row = ["100", "200", "300", "400"]
        assert extractor._looks_like_header(row) is False

    def test_looks_like_header_mixed(self, extractor):
        """Test header detection with mixed content."""
        row = ["ID", "100", "Name", "Active"]
        # 2 numeric, 2 text - should be accepted as header
        assert extractor._looks_like_header(row) is True

    def test_looks_like_header_mostly_empty(self, extractor):
        """Test header detection rejects mostly empty rows."""
        row = ["", "", "Data", ""]
        assert extractor._looks_like_header(row) is False

    def test_rows_equal_identical(self, extractor):
        """Test row equality for identical rows."""
        row1 = ["A", "B", "C"]
        row2 = ["A", "B", "C"]
        assert extractor._rows_equal(row1, row2) is True

    def test_rows_equal_case_insensitive(self, extractor):
        """Test row equality is case-insensitive."""
        row1 = ["Name", "AGE", "City"]
        row2 = ["name", "age", "city"]
        assert extractor._rows_equal(row1, row2) is True

    def test_rows_equal_different_length(self, extractor):
        """Test row equality with different lengths."""
        row1 = ["A", "B"]
        row2 = ["A", "B", "C"]
        assert extractor._rows_equal(row1, row2) is False

    def test_rows_equal_whitespace(self, extractor):
        """Test row equality ignores whitespace."""
        row1 = ["  Name  ", "Age", " City "]
        row2 = ["Name", "Age", "City"]
        assert extractor._rows_equal(row1, row2) is True

    def test_should_merge_consecutive_pages(self, extractor):
        """Test merge decision for consecutive pages."""
        t1 = ExtractedTable(page_number=5, rows=[["A", "B"]], end_page=5)
        t2 = ExtractedTable(page_number=6, rows=[["C", "D"]])
        assert extractor._should_merge(t1, t2) is True

    def test_should_merge_non_consecutive(self, extractor):
        """Test merge decision for non-consecutive pages."""
        t1 = ExtractedTable(page_number=5, rows=[["A", "B"]], end_page=5)
        t2 = ExtractedTable(page_number=8, rows=[["C", "D"]])
        assert extractor._should_merge(t1, t2) is False

    def test_should_merge_different_columns(self, extractor):
        """Test merge decision with different column counts."""
        t1 = ExtractedTable(page_number=5, rows=[["A", "B", "C"]], end_page=5)
        t2 = ExtractedTable(page_number=6, rows=[["D", "E"]])
        assert extractor._should_merge(t1, t2) is False

    def test_merge_two_tables_basic(self, extractor):
        """Test basic table merging."""
        t1 = ExtractedTable(
            page_number=1,
            rows=[["A", "B"], ["1", "2"]],
            headers=["Col1", "Col2"]
        )
        t2 = ExtractedTable(
            page_number=2,
            rows=[["3", "4"], ["5", "6"]]
        )

        merged = extractor._merge_two_tables(t1, t2)

        assert merged.page_number == 1
        assert merged.end_page == 2
        assert merged.headers == ["Col1", "Col2"]
        assert len(merged.rows) == 4

    def test_merge_two_tables_remove_repeated_header(self, extractor):
        """Test that repeated headers are removed during merge."""
        t1 = ExtractedTable(
            page_number=1,
            rows=[["A", "B"]],
            headers=["Col1", "Col2"]
        )
        t2 = ExtractedTable(
            page_number=2,
            rows=[["Col1", "Col2"], ["C", "D"]]  # First row is header
        )

        merged = extractor._merge_two_tables(t1, t2)

        # Should have 2 rows (A,B and C,D), not 3
        assert len(merged.rows) == 2
        assert ["Col1", "Col2"] not in merged.rows

    def test_merge_cross_page_tables_chain(self, extractor):
        """Test merging a chain of tables."""
        tables = [
            ExtractedTable(page_number=1, rows=[["A"]], headers=["X"]),
            ExtractedTable(page_number=2, rows=[["B"]]),
            ExtractedTable(page_number=3, rows=[["C"]]),
        ]

        merged = extractor._merge_cross_page_tables(tables)

        assert len(merged) == 1
        assert merged[0].page_number == 1
        assert merged[0].end_page == 3
        assert len(merged[0].rows) == 3

    def test_clean_table_normalizes_whitespace(self, extractor):
        """Test that table cleaning normalizes whitespace."""
        table = ExtractedTable(
            page_number=1,
            rows=[
                ["  Name  ", "Age\n\n"],
                ["Alice   ", "  30"]
            ]
        )

        cleaned = extractor._clean_table(table)

        assert cleaned.rows[0][0] == "Alice"
        assert cleaned.rows[0][1] == "30"

    def test_clean_table_handles_none(self, extractor):
        """Test that table cleaning handles None cells."""
        table = ExtractedTable(
            page_number=1,
            rows=[
                ["Header1", "Header2", "Header3"],  # Clear header row
                [None, "Value", "Test"],
                ["Data", None, "More"]
            ]
        )

        cleaned = extractor._clean_table(table)

        # After cleaning, first row becomes header
        assert cleaned.headers == ["Header1", "Header2", "Header3"]
        # Remaining rows should have None converted to empty string
        assert "" in cleaned.rows[0]  # The None was converted
        assert "" in cleaned.rows[1]  # The None was converted


class TestTableExtractorIntegration:
    """Integration tests for TableExtractor with real PDF."""

    @pytest.fixture
    def pdf_path(self):
        """Path to test PDF."""
        import os
        path = "pdfs/Pruefungsordnung_BSc_Inf_2024.pdf"
        if not os.path.exists(path):
            pytest.skip(f"Test PDF not found: {path}")
        return path

    def test_extract_tables_from_pdf(self, pdf_path):
        """Test table extraction from real PDF."""
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf(pdf_path, merge_cross_page=True)

        # Should find tables
        assert len(tables) > 0

        # All tables should have rows
        for table in tables:
            assert table.rows is not None

    def test_extract_specific_pages(self, pdf_path):
        """Test extracting tables from specific pages."""
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf(pdf_path, pages=[6, 7])

        # Should only have tables from pages 6 and 7
        for table in tables:
            assert table.page_number in [6, 7]

    def test_extract_with_merge_disabled(self, pdf_path):
        """Test extraction without cross-page merging."""
        extractor = TableExtractor()
        tables_no_merge = extractor.extract_from_pdf(pdf_path, merge_cross_page=False)
        tables_merged = extractor.extract_from_pdf(pdf_path, merge_cross_page=True)

        # Without merging, should have more tables
        assert len(tables_no_merge) >= len(tables_merged)

    def test_module_table_structure(self, pdf_path):
        """Test that module list tables have correct structure."""
        extractor = TableExtractor()
        # Pages 30-40 contain the module list
        tables = extractor.extract_from_pdf(pdf_path, pages=list(range(30, 41)), merge_cross_page=True)

        assert len(tables) >= 1

        # Check that we got a merged table
        main_table = tables[0]
        assert main_table.end_page is not None
        assert main_table.end_page > main_table.page_number
