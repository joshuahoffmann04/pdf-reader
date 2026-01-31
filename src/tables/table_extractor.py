"""
Table Extraction Module

Uses pdfplumber for accurate table detection and extraction from PDFs.
Handles multi-page tables, tables without visible lines, and filters page numbers.
"""

import pdfplumber
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ExtractedTable:
    """Represents an extracted table."""
    page_number: int
    rows: list[list[str]]
    headers: list[str] = field(default_factory=list)
    bbox: tuple = None
    end_page: int = None  # For multi-page tables

    def __post_init__(self):
        if self.end_page is None:
            self.end_page = self.page_number

    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if not self.rows and not self.headers:
            return ""

        lines = []
        headers = self.headers if self.headers else (self.rows[0] if self.rows else [])
        data_rows = self.rows if self.headers else (self.rows[1:] if self.rows else [])

        if not headers:
            return ""

        # Header row
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # Data rows
        for row in data_rows:
            padded = list(row) + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(str(c) for c in padded[:len(headers)]) + " |")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert table to dictionary format."""
        return {
            "page": self.page_number,
            "end_page": self.end_page,
            "headers": self.headers,
            "rows": self.rows,
            "bbox": self.bbox
        }


class TableExtractor:
    """
    Extracts tables from PDF files using pdfplumber.

    Features:
    - Automatic table detection with multiple strategies
    - Multi-page table merging with header deduplication
    - Page number filtering
    - Column marker detection (a), (b), (c), (d)
    - Fallback to text-based extraction for borderless tables
    """

    # Pattern to detect standalone page numbers
    PAGE_NUMBER_PATTERN = re.compile(r'^\s*\d{1,3}\s*$')

    # Pattern to detect column markers like (a), (b), (1), (2)
    COLUMN_MARKER_PATTERN = re.compile(r'^\s*\([a-z0-9]\)\s*$', re.IGNORECASE)

    # Pattern to detect AB excerpt text
    AB_TEXT_PATTERN = re.compile(r'Textauszug\s+aus\s+den\s+Allgemeinen', re.IGNORECASE)

    def __init__(self, snap_tolerance: int = 3, join_tolerance: int = 3):
        """Initialize the table extractor."""
        self.snap_tolerance = snap_tolerance
        self.join_tolerance = join_tolerance

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None,
        merge_cross_page: bool = True
    ) -> list[ExtractedTable]:
        """
        Extract all tables from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            pages: Optional list of page numbers (1-indexed). None = all pages.
            merge_cross_page: If True, merge tables spanning multiple pages.

        Returns:
            List of ExtractedTable objects.
        """
        pdf_path = Path(pdf_path)
        all_tables = []

        with pdfplumber.open(pdf_path) as pdf:
            page_indices = range(len(pdf.pages))
            if pages:
                page_indices = [p - 1 for p in pages if 0 < p <= len(pdf.pages)]

            for page_idx in page_indices:
                page = pdf.pages[page_idx]
                page_tables = self._extract_from_page(page, page_idx + 1)
                all_tables.extend(page_tables)

        # Merge cross-page tables if requested
        if merge_cross_page and len(all_tables) > 1:
            all_tables = self._merge_cross_page_tables(all_tables)

        return all_tables

    def _extract_from_page(self, page: pdfplumber.page.Page, page_number: int) -> list[ExtractedTable]:
        """Extract tables from a single page, trying multiple strategies."""
        tables = []

        # Strategy 1: Line-based extraction (for tables with borders)
        line_tables = self._extract_with_strategy(page, page_number, "lines", "lines")

        # Strategy 2: Text-based extraction (for borderless tables)
        # Only use if no line-based tables found
        text_tables = []
        if not line_tables:
            text_tables = self._extract_with_strategy(page, page_number, "text", "text")
            # Filter text-based tables more strictly to avoid false positives
            text_tables = [t for t in text_tables if self._is_likely_real_table(t)]

        # Use line-based if found, otherwise text-based
        if line_tables:
            tables = line_tables
        elif text_tables:
            tables = text_tables

        # Filter and clean tables
        cleaned_tables = []
        for table in tables:
            cleaned = self._clean_table(table)
            if cleaned and cleaned.rows:
                cleaned_tables.append(cleaned)

        return cleaned_tables

    def _is_likely_real_table(self, table: ExtractedTable) -> bool:
        """Heuristic to detect if extraction result is a real table or just text blocks."""
        if not table.rows:
            return False

        # Need at least 2 rows
        if len(table.rows) < 2:
            return False

        # Count columns with actual content
        col_counts = []
        for row in table.rows:
            non_empty = sum(1 for cell in row if cell and cell.strip())
            col_counts.append(non_empty)

        # Average columns with content
        avg_cols = sum(col_counts) / len(col_counts) if col_counts else 0

        # Real tables typically have multiple columns with content
        if avg_cols < 2:
            return False

        # Check consistency: real tables have similar column usage across rows
        if col_counts:
            min_cols = min(col_counts)
            max_cols = max(col_counts)
            # If column usage varies wildly, it's probably text, not a table
            if max_cols > 0 and min_cols / max_cols < 0.3:
                return False

        # Check for long text cells (tables rarely have very long cells)
        long_cells = 0
        total_cells = 0
        for row in table.rows:
            for cell in row:
                if cell:
                    total_cells += 1
                    if len(cell) > 100:  # Long text
                        long_cells += 1

        # If more than 30% of cells are long text, it's probably not a table
        if total_cells > 0 and long_cells / total_cells > 0.3:
            return False

        return True

    def _extract_with_strategy(
        self,
        page: pdfplumber.page.Page,
        page_number: int,
        vertical: str,
        horizontal: str
    ) -> list[ExtractedTable]:
        """Extract tables using specific strategies."""
        tables = []
        settings = {
            "vertical_strategy": vertical,
            "horizontal_strategy": horizontal,
            "snap_tolerance": self.snap_tolerance,
            "join_tolerance": self.join_tolerance,
        }

        try:
            found = page.find_tables(table_settings=settings)
            for table in found:
                extracted = table.extract()
                if extracted and any(row for row in extracted if any(cell for cell in row if cell)):
                    tables.append(ExtractedTable(
                        page_number=page_number,
                        rows=extracted,
                        bbox=table.bbox
                    ))
        except Exception:
            pass

        return tables

    def _clean_table(self, table: ExtractedTable) -> Optional[ExtractedTable]:
        """Clean table: normalize cells, filter page numbers, detect headers."""
        if not table.rows:
            return None

        cleaned_rows = []
        for row in table.rows:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Normalize whitespace
                    cleaned_row.append(" ".join(str(cell).split()))
            cleaned_rows.append(cleaned_row)

        # Filter rows that are just page numbers
        cleaned_rows = self._filter_page_number_rows(cleaned_rows)

        # Filter rows with AB excerpt text
        cleaned_rows = self._filter_ab_text_rows(cleaned_rows)

        # Filter rows that are long paragraph text (not table data)
        cleaned_rows = self._filter_paragraph_rows(cleaned_rows)

        if not cleaned_rows:
            return None

        # Detect headers (with column marker handling)
        headers, cleaned_rows = self._detect_headers(cleaned_rows)

        if not cleaned_rows:
            return None

        return ExtractedTable(
            page_number=table.page_number,
            rows=cleaned_rows,
            headers=headers,
            bbox=table.bbox
        )

    def _filter_page_number_rows(self, rows: list[list[str]]) -> list[list[str]]:
        """Remove rows that appear to be just page numbers."""
        filtered = []
        for row in rows:
            # Check if row is just a page number
            non_empty = [c for c in row if c.strip()]
            if len(non_empty) == 1 and self.PAGE_NUMBER_PATTERN.match(non_empty[0]):
                continue  # Skip page number rows
            if non_empty:  # Keep non-empty rows
                filtered.append(row)
        return filtered

    def _filter_ab_text_rows(self, rows: list[list[str]]) -> list[list[str]]:
        """Remove rows that contain AB excerpt text markers."""
        filtered = []
        for row in rows:
            has_ab_text = any(
                cell and self.AB_TEXT_PATTERN.search(cell)
                for cell in row
            )
            if not has_ab_text:
                filtered.append(row)
        return filtered

    def _filter_paragraph_rows(self, rows: list[list[str]]) -> list[list[str]]:
        """Remove rows that are long paragraph text, not table data.

        Tables rarely have cells with very long text. If a row has one cell
        with very long text and other cells are mostly empty, it's likely
        paragraph text captured by mistake.
        """
        filtered = []
        for row in rows:
            non_empty = [c for c in row if c.strip()]
            if not non_empty:
                continue

            # If there's a single cell with very long text (>300 chars), skip
            max_cell_len = max(len(c) for c in row if c)
            if max_cell_len > 300:
                # Check if most other cells are empty
                empty_count = sum(1 for c in row if not c.strip())
                if empty_count >= len(row) / 2:
                    continue  # Skip this row - it's paragraph text

            filtered.append(row)
        return filtered

    def _is_column_marker_row(self, row: list[str]) -> bool:
        """Check if row contains only column markers like (a), (b), (c), (d)."""
        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < 2:
            return False

        # All non-empty cells must match the column marker pattern
        return all(self.COLUMN_MARKER_PATTERN.match(c) for c in non_empty)

    def _detect_headers(self, rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
        """Detect headers, handling column markers and multi-row headers.

        Returns:
            Tuple of (headers, remaining_rows)
        """
        if not rows:
            return [], []

        start_idx = 0

        # Check if first row is column markers like (a), (b), (c), (d)
        if self._is_column_marker_row(rows[0]):
            start_idx = 1

        if start_idx >= len(rows):
            return [], rows

        # Check if the row at start_idx looks like a header
        if self._looks_like_header(rows[start_idx]):
            headers = rows[start_idx]
            remaining = rows[start_idx + 1:]
            return headers, remaining

        # No headers detected
        return [], rows[start_idx:]

    def _looks_like_header(self, row: list[str]) -> bool:
        """Heuristic: is this row a header?

        A row is likely a header if:
        - It has mostly non-empty cells
        - Cells contain text labels (not numeric data)
        - Cells don't look like data patterns (multiple numbers, grades, etc.)
        """
        if not row:
            return False

        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < len(row) / 2:
            return False

        # Check for data-like patterns that indicate this is NOT a header
        data_like = 0
        for cell in non_empty:
            # Multiple space-separated numbers (like "15 14 13" or "9 8 7")
            if re.match(r'^[\d\s]+$', cell.strip()) and len(cell.split()) > 1:
                data_like += 1
                continue

            # Multiple comma-decimal numbers (like "0,7 1,0 1,3")
            if re.match(r'^[\d,\s\.]+$', cell.strip()) and ',' in cell:
                data_like += 1
                continue

            # Single number
            try:
                float(cell.replace(",", ".").replace(" ", ""))
                data_like += 1
                continue
            except ValueError:
                pass

        # If half or more cells look like data, this is not a header
        if data_like >= len(non_empty) / 2:
            return False

        return True

    def _merge_cross_page_tables(self, tables: list[ExtractedTable]) -> list[ExtractedTable]:
        """Merge tables that span multiple pages."""
        if not tables:
            return []

        merged = []
        current = None

        for table in tables:
            if current is None:
                current = table
                continue

            # Check if should merge
            if self._should_merge(current, table):
                current = self._merge_two_tables(current, table)
            else:
                merged.append(current)
                current = table

        if current:
            merged.append(current)

        return merged

    def _should_merge(self, t1: ExtractedTable, t2: ExtractedTable) -> bool:
        """Determine if two tables should be merged."""
        # Must be on consecutive pages
        if t2.page_number != t1.end_page + 1:
            return False

        # Compare column structure
        cols1 = len(t1.headers) or (len(t1.rows[0]) if t1.rows else 0)
        cols2 = len(t2.headers) or (len(t2.rows[0]) if t2.rows else 0)

        if cols1 == 0 or cols2 == 0:
            return False

        # Same column count = likely continuation
        return cols1 == cols2

    def _merge_two_tables(self, t1: ExtractedTable, t2: ExtractedTable) -> ExtractedTable:
        """Merge two tables, removing repeated headers."""
        headers = t1.headers or t2.headers

        # Check if t2's first row is a repeated header
        rows_to_add = t2.rows
        if headers and t2.rows and self._rows_equal(t2.rows[0], headers):
            rows_to_add = t2.rows[1:]  # Skip repeated header

        combined_rows = t1.rows + rows_to_add

        return ExtractedTable(
            page_number=t1.page_number,
            rows=combined_rows,
            headers=headers,
            bbox=t1.bbox,
            end_page=t2.page_number
        )

    def _rows_equal(self, row1: list[str], row2: list[str]) -> bool:
        """Check if two rows are equal (for header detection)."""
        if len(row1) != len(row2):
            return False
        return all(c1.strip().lower() == c2.strip().lower() for c1, c2 in zip(row1, row2))
