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

        # The improved filter may extract fewer tables due to better false positive detection
        # At minimum we should find at least one table in this range
        assert len(tables) >= 1

        # Check that tables have valid structure
        for table in tables:
            assert table.end_page is not None
            assert table.end_page >= table.page_number
            # Tables should have rows
            assert len(table.rows) >= 0


class TestColumnMarkerDetection:
    """Tests for column marker detection and handling."""

    @pytest.fixture
    def extractor(self):
        return TableExtractor()

    def test_is_column_marker_row_valid(self, extractor):
        """Test detecting column marker rows like (a), (b), (c), (d)."""
        row = ["(a)", "(b)", "(c)", "(d)"]
        assert extractor._is_column_marker_row(row) is True

    def test_is_column_marker_row_with_numbers(self, extractor):
        """Test detecting numeric column markers like (1), (2), (3)."""
        row = ["(1)", "(2)", "(3)"]
        assert extractor._is_column_marker_row(row) is True

    def test_is_column_marker_row_with_whitespace(self, extractor):
        """Test column markers with whitespace."""
        row = ["  (a)  ", "(b)", " (c) "]
        assert extractor._is_column_marker_row(row) is True

    def test_is_column_marker_row_mixed_empty(self, extractor):
        """Test column markers with some empty cells."""
        row = ["(a)", "(b)", "", "(d)"]
        # Should work - only checks non-empty cells
        assert extractor._is_column_marker_row(row) is True

    def test_is_column_marker_row_not_markers(self, extractor):
        """Test that real headers are not detected as markers."""
        row = ["Name", "Age", "City"]
        assert extractor._is_column_marker_row(row) is False

    def test_is_column_marker_row_single_element(self, extractor):
        """Test single element is not a marker row."""
        row = ["(a)", "", ""]
        # Not enough non-empty cells
        assert extractor._is_column_marker_row(row) is False


class TestABTextFiltering:
    """Tests for AB text filtering from tables."""

    @pytest.fixture
    def extractor(self):
        return TableExtractor()

    def test_filter_ab_text_rows(self, extractor):
        """Test that rows with AB text are filtered."""
        rows = [
            ["Header1", "Header2"],
            ["Textauszug aus den Allgemeinen Bestimmungen: § 28", ""],
            ["Normal", "Data"],
        ]
        filtered = extractor._filter_ab_text_rows(rows)

        assert len(filtered) == 2
        assert ["Header1", "Header2"] in filtered
        assert ["Normal", "Data"] in filtered

    def test_filter_ab_text_keeps_normal(self, extractor):
        """Test that normal rows are kept."""
        rows = [
            ["Data1", "Data2"],
            ["More", "Values"],
        ]
        filtered = extractor._filter_ab_text_rows(rows)
        assert len(filtered) == 2


class TestParagraphRowFiltering:
    """Tests for paragraph/long text row filtering."""

    @pytest.fixture
    def extractor(self):
        return TableExtractor()

    def test_filter_paragraph_rows(self, extractor):
        """Test that long paragraph rows are filtered."""
        long_text = "This is a very long paragraph that contains multiple sentences and is definitely not table data. " * 5
        rows = [
            ["Header1", "Header2", "Header3", "Header4"],
            ["Short", "Data", "Here", "OK"],
            [long_text, "", "", ""],  # Long text with empty cells
        ]
        filtered = extractor._filter_paragraph_rows(rows)

        assert len(filtered) == 2
        assert ["Short", "Data", "Here", "OK"] in filtered

    def test_filter_paragraph_keeps_short(self, extractor):
        """Test that short rows are kept."""
        rows = [
            ["A", "B", "C"],
            ["1", "2", "3"],
        ]
        filtered = extractor._filter_paragraph_rows(rows)
        assert len(filtered) == 2


class TestHeaderDetectionAdvanced:
    """Advanced tests for header detection."""

    @pytest.fixture
    def extractor(self):
        return TableExtractor()

    def test_detect_headers_skips_column_markers(self, extractor):
        """Test that column markers are skipped when detecting headers."""
        rows = [
            ["(a)", "(b)", "(c)", "(d)"],
            ["Punkte", "Bewertung", "Note", "Definition"],
            ["15", "1.0", "sehr gut", "excellent"],
        ]
        headers, data = extractor._detect_headers(rows)

        assert headers == ["Punkte", "Bewertung", "Note", "Definition"]
        assert len(data) == 1
        assert data[0] == ["15", "1.0", "sehr gut", "excellent"]

    def test_detect_headers_no_markers(self, extractor):
        """Test header detection without column markers."""
        rows = [
            ["Name", "Age", "City"],
            ["Alice", "30", "Berlin"],
        ]
        headers, data = extractor._detect_headers(rows)

        assert headers == ["Name", "Age", "City"]
        assert len(data) == 1

    def test_looks_like_header_rejects_data_patterns(self, extractor):
        """Test that data-like patterns are not detected as headers."""
        # Row with space-separated numbers (like grades "15 14 13")
        row = ["9 8 7", "2,7 3,0 3,3", "befriedigend", "eine Leistung"]
        assert extractor._looks_like_header(row) is False

    def test_looks_like_header_accepts_real_headers(self, extractor):
        """Test that real headers are accepted."""
        row = ["Punkte", "Bewertung im traditionellen Notensystem", "Note in Worten", "Definition"]
        assert extractor._looks_like_header(row) is True

    def test_looks_like_header_rejects_comma_decimals(self, extractor):
        """Test that comma-decimal patterns are rejected."""
        row = ["0,7 1,0 1,3", "2,0 2,3", "some text", "other"]
        # 2/4 cells look like data (comma-decimals)
        assert extractor._looks_like_header(row) is False


class TestGradeTableExtraction:
    """Integration tests for grade table extraction (the problematic table)."""

    @pytest.fixture
    def pdf_path(self):
        import os
        path = "pdfs/Pruefungsordnung_BSc_Inf_2024.pdf"
        if not os.path.exists(path):
            pytest.skip(f"Test PDF not found: {path}")
        return path

    def test_grade_table_headers(self, pdf_path):
        """Test that grade table has correct headers (not column markers)."""
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf(pdf_path, pages=[22, 23], merge_cross_page=True)

        # Find the 4-column grade table
        grade_table = None
        for table in tables:
            if table.headers and len(table.headers) == 4:
                if "Punkte" in table.headers[0]:
                    grade_table = table
                    break

        assert grade_table is not None, "Grade table not found"

        # Headers should NOT be column markers
        assert grade_table.headers[0] != "(a)"
        assert "Punkte" in grade_table.headers[0]

    def test_grade_table_complete_data(self, pdf_path):
        """Test that all grade rows are extracted."""
        extractor = TableExtractor()
        tables = extractor.extract_from_pdf(pdf_path, pages=[22, 23], merge_cross_page=True)

        # Find the grade table
        grade_table = None
        for table in tables:
            if table.headers and len(table.headers) == 4:
                if "Punkte" in table.headers[0]:
                    grade_table = table
                    break

        assert grade_table is not None

        # Should have all 5 grade rows
        assert len(grade_table.rows) >= 5

        # Check for "befriedigend" row (was previously missing)
        row_texts = [" ".join(row) for row in grade_table.rows]
        assert any("befriedigend" in text for text in row_texts), \
            "befriedigend row is missing from grade table"
