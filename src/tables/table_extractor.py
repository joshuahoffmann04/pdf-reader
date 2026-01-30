"""
Table Extraction Module

Uses pdfplumber for accurate table detection and extraction from PDFs.
Handles multi-page tables and complex table structures.
"""

import pdfplumber
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TableCell:
    """Represents a single cell in a table."""
    text: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


@dataclass
class ExtractedTable:
    """Represents an extracted table."""
    page_number: int
    rows: list[list[str]]  # 2D array of cell values
    headers: list[str] = field(default_factory=list)
    bbox: tuple = None  # Bounding box (x0, y0, x1, y1)

    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if not self.rows:
            return ""

        lines = []

        # Use first row as headers if headers not explicitly set
        headers = self.headers if self.headers else self.rows[0]
        data_rows = self.rows[1:] if not self.headers else self.rows

        # Header row
        header_line = "| " + " | ".join(str(h) for h in headers) + " |"
        lines.append(header_line)

        # Separator
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(separator)

        # Data rows
        for row in data_rows:
            # Pad row if needed
            while len(row) < len(headers):
                row.append("")
            row_line = "| " + " | ".join(str(cell) for cell in row[:len(headers)]) + " |"
            lines.append(row_line)

        return "\n".join(lines)

    def to_csv(self, delimiter: str = ",") -> str:
        """Convert table to CSV format."""
        lines = []

        # Include headers
        if self.headers:
            lines.append(delimiter.join(f'"{h}"' for h in self.headers))

        for row in self.rows:
            # Escape quotes in cells and wrap in quotes
            escaped_row = ['"' + str(cell).replace('"', '""') + '"' for cell in row]
            lines.append(delimiter.join(escaped_row))

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert table to dictionary format."""
        return {
            "page": self.page_number,
            "headers": self.headers,
            "rows": self.rows,
            "bbox": self.bbox
        }


class TableExtractor:
    """
    Extracts tables from PDF files using pdfplumber.

    Features:
    - Automatic table detection
    - Multi-page table merging
    - Multiple output formats (Markdown, CSV, dict)
    """

    def __init__(
        self,
        vertical_strategy: str = "lines",
        horizontal_strategy: str = "lines",
        snap_tolerance: int = 3,
        join_tolerance: int = 3
    ):
        """
        Initialize the table extractor.

        Args:
            vertical_strategy: Strategy for detecting vertical lines ("lines", "text", "explicit").
            horizontal_strategy: Strategy for detecting horizontal lines.
            snap_tolerance: Pixel tolerance for snapping to lines.
            join_tolerance: Pixel tolerance for joining broken lines.
        """
        self.table_settings = {
            "vertical_strategy": vertical_strategy,
            "horizontal_strategy": horizontal_strategy,
            "snap_tolerance": snap_tolerance,
            "join_tolerance": join_tolerance,
        }

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        pages: Optional[list[int]] = None
    ) -> list[ExtractedTable]:
        """
        Extract all tables from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            pages: Optional list of page numbers to process (1-indexed).
                   If None, processes all pages.

        Returns:
            List of ExtractedTable objects.
        """
        pdf_path = Path(pdf_path)
        tables = []

        with pdfplumber.open(pdf_path) as pdf:
            page_range = range(len(pdf.pages))
            if pages:
                page_range = [p - 1 for p in pages if 0 < p <= len(pdf.pages)]

            for page_idx in page_range:
                page = pdf.pages[page_idx]
                page_tables = self._extract_from_page(page, page_idx + 1)
                tables.extend(page_tables)

        return tables

    def _extract_from_page(
        self,
        page: pdfplumber.page.Page,
        page_number: int
    ) -> list[ExtractedTable]:
        """Extract tables from a single page."""
        tables = []

        try:
            # Find tables on the page
            found_tables = page.find_tables(table_settings=self.table_settings)

            for table in found_tables:
                # Extract table data
                extracted = table.extract()

                if not extracted or not any(extracted):
                    continue

                # Clean cell values
                cleaned_rows = []
                for row in extracted:
                    cleaned_row = []
                    for cell in row:
                        if cell is None:
                            cleaned_row.append("")
                        else:
                            # Clean whitespace and normalize
                            cleaned_cell = " ".join(str(cell).split())
                            cleaned_row.append(cleaned_cell)
                    cleaned_rows.append(cleaned_row)

                # Try to detect headers (first row if it looks like headers)
                headers = []
                if cleaned_rows and self._looks_like_header(cleaned_rows[0]):
                    headers = cleaned_rows[0]
                    cleaned_rows = cleaned_rows[1:]

                ext_table = ExtractedTable(
                    page_number=page_number,
                    rows=cleaned_rows,
                    headers=headers,
                    bbox=table.bbox
                )

                tables.append(ext_table)

        except Exception as e:
            # Log but don't fail on table extraction errors
            print(f"Warning: Table extraction failed on page {page_number}: {e}")

        return tables

    def _looks_like_header(self, row: list[str]) -> bool:
        """
        Heuristic to determine if a row looks like a table header.

        Checks for:
        - All non-empty cells
        - No numeric-only cells
        - Reasonable text length
        """
        if not row:
            return False

        non_empty = [cell for cell in row if cell.strip()]

        # At least half of cells should be non-empty
        if len(non_empty) < len(row) / 2:
            return False

        # Headers typically don't have purely numeric values
        numeric_count = 0
        for cell in non_empty:
            # Check if cell is purely numeric
            try:
                float(cell.replace(",", ".").replace(" ", ""))
                numeric_count += 1
            except ValueError:
                pass

        # If most cells are numeric, probably not a header
        if numeric_count > len(non_empty) / 2:
            return False

        return True

    def merge_tables(
        self,
        tables: list[ExtractedTable],
        similarity_threshold: float = 0.8
    ) -> list[ExtractedTable]:
        """
        Merge tables that span multiple pages.

        Tables are merged if they have similar column structures.

        Args:
            tables: List of tables to potentially merge.
            similarity_threshold: Column similarity required for merging.

        Returns:
            List of merged tables.
        """
        if not tables:
            return []

        merged = []
        current_table = None

        for table in tables:
            if current_table is None:
                current_table = table
                continue

            # Check if this table should be merged with the current one
            if self._should_merge(current_table, table, similarity_threshold):
                # Merge tables
                current_table = self._merge_two_tables(current_table, table)
            else:
                # Save current and start new
                merged.append(current_table)
                current_table = table

        if current_table:
            merged.append(current_table)

        return merged

    def _should_merge(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable,
        threshold: float
    ) -> bool:
        """Determine if two tables should be merged."""
        # Must be on consecutive pages
        if table2.page_number != table1.page_number + 1:
            return False

        # Compare column counts
        cols1 = len(table1.headers) if table1.headers else (
            len(table1.rows[0]) if table1.rows else 0
        )
        cols2 = len(table2.headers) if table2.headers else (
            len(table2.rows[0]) if table2.rows else 0
        )

        if cols1 == 0 or cols2 == 0:
            return False

        # Same number of columns suggests continuation
        if cols1 == cols2:
            return True

        # Allow slight variation
        similarity = min(cols1, cols2) / max(cols1, cols2)
        return similarity >= threshold

    def _merge_two_tables(
        self,
        table1: ExtractedTable,
        table2: ExtractedTable
    ) -> ExtractedTable:
        """Merge two tables into one."""
        # Keep headers from first table
        headers = table1.headers if table1.headers else table2.headers

        # Combine rows
        combined_rows = table1.rows + table2.rows

        return ExtractedTable(
            page_number=table1.page_number,  # Keep original page
            rows=combined_rows,
            headers=headers,
            bbox=table1.bbox  # Keep original bbox
        )

    def extract_table_at_position(
        self,
        pdf_path: str | Path,
        page_number: int,
        bbox: tuple
    ) -> Optional[ExtractedTable]:
        """
        Extract a specific table by its position.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Page number (1-indexed).
            bbox: Bounding box (x0, y0, x1, y1).

        Returns:
            ExtractedTable if found, None otherwise.
        """
        with pdfplumber.open(pdf_path) as pdf:
            if page_number > len(pdf.pages):
                return None

            page = pdf.pages[page_number - 1]

            # Crop to the specified area
            cropped = page.within_bbox(bbox)

            # Extract table from cropped area
            tables = cropped.find_tables(table_settings=self.table_settings)

            if tables:
                extracted = tables[0].extract()
                if extracted:
                    return ExtractedTable(
                        page_number=page_number,
                        rows=extracted,
                        bbox=bbox
                    )

        return None
