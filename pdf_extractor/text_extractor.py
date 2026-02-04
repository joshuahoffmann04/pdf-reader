"""
Text-native PDF extraction utilities.

This module extracts selectable text from PDFs (via PyMuPDF) and performs
lightweight structural analysis for legal/academic documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
from typing import Iterable

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
SECTION_SIGN = "\u00a7"


@dataclass
class PageTextAnalysis:
    text: str
    section_numbers: list[str]
    section_titles: list[str]
    paragraph_numbers: list[str]
    has_table: bool
    has_list: bool
    internal_references: list[str]
    external_references: list[str]
    continues_from_previous: bool
    continues_to_next: bool
    quality_ok: bool


class TextExtractor:
    def __init__(
        self,
        min_chars: int = 200,
        min_alpha_ratio: float = 0.20,
        sort_blocks: bool = True,
        preserve_line_breaks: bool = True,
        layout_mode: str = "simple",
        column_gap_ratio: float = 0.25,
    ) -> None:
        self.min_chars = min_chars
        self.min_alpha_ratio = min_alpha_ratio
        self.sort_blocks = sort_blocks
        self.preserve_line_breaks = preserve_line_breaks
        self.layout_mode = layout_mode
        self.column_gap_ratio = column_gap_ratio

    def extract_page(self, pdf_path: str | Path, page_number: int) -> PageTextAnalysis:
        raw_text = self._extract_text_blocks(pdf_path, page_number)
        return self.analyze_text(raw_text)

    def analyze_text(self, text: str) -> PageTextAnalysis:
        cleaned = self._clean_text(text)
        return self._analyze_text(cleaned)

    def _extract_text_blocks(self, pdf_path: str | Path, page_number: int) -> str:
        with fitz.open(pdf_path) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"Page {page_number} out of range (1-{len(doc)})")
            page = doc[page_number - 1]
            blocks = page.get_text("blocks", sort=False)
            page_width = page.rect.width

        texts: list[str] = []
        ordered_blocks = self._order_blocks(blocks, page_width) if self.sort_blocks else blocks
        for block in ordered_blocks:
            if len(block) < 5:
                continue
            text = block[4]
            block_type = block[-1] if isinstance(block[-1], int) else 0
            if block_type != 0:
                continue  # skip non-text blocks
            if text and text.strip():
                texts.append(text)

        return "\n\n".join(texts).strip()

    def _order_blocks(self, blocks: list, page_width: float) -> list:
        if self.layout_mode != "columns":
            return sorted(blocks, key=lambda b: (b[1], b[0]))

        text_blocks = [b for b in blocks if len(b) >= 5 and (b[-1] if isinstance(b[-1], int) else 0) == 0]
        x0s = sorted(b[0] for b in text_blocks if b[4] and str(b[4]).strip())
        if len(x0s) < 6:
            return sorted(blocks, key=lambda b: (b[1], b[0]))

        gaps = [(x0s[i + 1] - x0s[i], i) for i in range(len(x0s) - 1)]
        max_gap, idx = max(gaps, key=lambda g: g[0])
        if max_gap < page_width * self.column_gap_ratio:
            return sorted(blocks, key=lambda b: (b[1], b[0]))

        split = x0s[idx]
        left = [b for b in blocks if b[0] <= split]
        right = [b for b in blocks if b[0] > split]
        left_sorted = sorted(left, key=lambda b: (b[1], b[0]))
        right_sorted = sorted(right, key=lambda b: (b[1], b[0]))
        return left_sorted + right_sorted

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        # De-hyphenate line breaks: "Stu-\n dium" -> "Studium"
        cleaned = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        if not self.preserve_line_breaks:
            cleaned = re.sub(r"\n{2,}", " \n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _analyze_text(self, text: str) -> PageTextAnalysis:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        section_numbers: list[str] = []
        section_titles: list[str] = []

        for idx, line in enumerate(lines):
            sec_match = re.match(rf"^{SECTION_SIGN}\s*(\d+[a-zA-Z]?)\b\s*(.*)$", line)
            if sec_match:
                number = f"{SECTION_SIGN} {sec_match.group(1)}"
                title = sec_match.group(2).strip()
                if not title and idx + 1 < len(lines):
                    nxt = lines[idx + 1]
                    if not self._looks_like_section_start(nxt):
                        title = nxt.strip()
                section_numbers.append(number)
                section_titles.append(title)
                continue

            ann_match = re.match(r"^(Anlage)\s*(\d+)\b\s*[:\\-]?\s*(.*)$", line, re.IGNORECASE)
            if ann_match:
                number = f"Anlage {ann_match.group(2)}"
                title = ann_match.group(3).strip()
                if not title and idx + 1 < len(lines):
                    nxt = lines[idx + 1]
                    if not self._looks_like_section_start(nxt):
                        title = nxt.strip()
                section_numbers.append(number)
                section_titles.append(title)

        paragraph_numbers = self._unique_in_order(re.findall(r"\(\d+\)", text))

        has_list = self._looks_like_list(lines)
        has_table = self._looks_like_table(lines, text)

        internal_refs = self._unique_in_order(
            re.findall(rf"{SECTION_SIGN}\s*\d+[a-zA-Z]?(?:\s*Abs\.\s*\d+)?", text)
            + re.findall(r"Anlage\s*\d+", text, flags=re.IGNORECASE)
        )

        external_refs = self._detect_external_references(text)

        continues_from_previous = self._continues_from_previous(lines, section_numbers)
        continues_to_next = self._continues_to_next(lines)

        quality_ok = self._quality_ok(text)

        return PageTextAnalysis(
            text=text,
            section_numbers=section_numbers,
            section_titles=section_titles,
            paragraph_numbers=paragraph_numbers,
            has_table=has_table,
            has_list=has_list,
            internal_references=internal_refs,
            external_references=external_refs,
            continues_from_previous=continues_from_previous,
            continues_to_next=continues_to_next,
            quality_ok=quality_ok,
        )

    def _quality_ok(self, text: str) -> bool:
        if not text:
            return False
        if len(text) < self.min_chars:
            return False
        alpha = sum(1 for c in text if c.isalpha())
        ratio = alpha / max(len(text), 1)
        return ratio >= self.min_alpha_ratio

    @staticmethod
    def _unique_in_order(items: Iterable[str]) -> list[str]:
        seen = set()
        result: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    @staticmethod
    def _looks_like_section_start(line: str) -> bool:
        return line.startswith(SECTION_SIGN) or line.lower().startswith("anlage")

    @staticmethod
    def _looks_like_list(lines: list[str]) -> bool:
        list_like = 0
        for line in lines:
            if re.match("^[-\u2022\u00b7]\\s+", line):
                list_like += 1
            elif re.match(r"^\(?[0-9]{1,2}[.)]\s+", line):
                list_like += 1
            elif re.match(r"^\(?[a-zA-Z][.)]\s+", line):
                list_like += 1
        return list_like >= 2

    @staticmethod
    def _looks_like_table(lines: list[str], text: str) -> bool:
        if "|" in text:
            return True
        spaced_lines = sum(1 for line in lines if re.search(r"\s{2,}", line))
        return spaced_lines >= 2

    @staticmethod
    def _detect_external_references(text: str) -> list[str]:
        candidates = [
            "Allgemeine Bestimmungen",
            "Hessisches Hochschulgesetz",
            "Hochschulgesetz",
            "Immatrikulationsordnung",
        ]
        found = []
        for term in candidates:
            if term in text:
                found.append(term)
        return found

    @staticmethod
    def _continues_from_previous(lines: list[str], section_numbers: list[str]) -> bool:
        if not lines:
            return False
        first = lines[0]
        if first.startswith(SECTION_SIGN) or first.lower().startswith("anlage"):
            return False
        if section_numbers:
            return False
        if first and first[0].islower():
            return True
        return False

    @staticmethod
    def _continues_to_next(lines: list[str]) -> bool:
        if not lines:
            return False
        last = lines[-1]
        if last.endswith("-"):
            return True
        if last and last[-1] not in ".!?":
            return True
        return False
