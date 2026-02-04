"""
Table extraction helpers.

Uses pdfplumber when available to extract structured tables from text-based PDFs.
This is optional and only used when enabled in ProcessingConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None

logger = logging.getLogger(__name__)


@dataclass
class TableExtractionResult:
    tables: list[list[list[str]]]
    text: str


class TableExtractor:
    def __init__(self, table_settings: dict[str, Any] | None = None) -> None:
        self.table_settings = table_settings or {}

    def extract_tables(self, pdf_path: str | Path, page_number: int) -> TableExtractionResult:
        if pdfplumber is None:
            return TableExtractionResult(tables=[], text="")

        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number < 1 or page_number > len(pdf.pages):
                return TableExtractionResult(tables=[], text="")
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables(table_settings=self.table_settings) or []

        text = self._tables_to_text(tables)
        return TableExtractionResult(tables=tables, text=text)

    def _tables_to_text(self, tables: list[list[list[str]]]) -> str:
        if not tables:
            return ""

        chunks: list[str] = []
        for t_idx, table in enumerate(tables, start=1):
            if not table:
                continue
            header, rows = self._split_header(table)
            row_texts = []
            for row in rows:
                row_texts.append(self._row_to_sentence(row, header))
            if row_texts:
                chunks.append(f"Tabelle {t_idx}: " + " ".join(row_texts))

        return "\n".join(chunks).strip()

    @staticmethod
    def _split_header(table: list[list[str]]) -> tuple[list[str] | None, list[list[str]]]:
        if len(table) < 2:
            return None, table
        first = [cell or "" for cell in table[0]]
        second = [cell or "" for cell in table[1]]

        def score(row: list[str]) -> int:
            return sum(1 for c in row if c.strip())

        # Heuristic: header has more non-empty text cells and fewer numeric-only cells
        numeric_only = sum(1 for c in first if c.strip().isdigit())
        if score(first) >= score(second) and numeric_only <= max(1, len(first) // 3):
            return first, table[1:]
        return None, table

    @staticmethod
    def _row_to_sentence(row: list[str], header: list[str] | None) -> str:
        cells = [(c or "").strip() for c in row]
        if header:
            pairs = []
            for h, c in zip(header, cells):
                h = (h or "").strip()
                if not h and not c:
                    continue
                if h:
                    pairs.append(f"{h}: {c}")
                else:
                    pairs.append(c)
            return "; ".join(pairs) + "."
        return " | ".join(cells) + "."
