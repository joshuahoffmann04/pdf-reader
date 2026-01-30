"""
Document Parser Module

Parses extracted PDF text into a hierarchical structure:
- Chapters (I., II., III., IV.)
- Sections (§1, §2, ...)
- AB-Excerpts (Allgemeine Bestimmungen)
- Appendices (Anlage 1-5) with optional sub-sections

This is a general-purpose parser for German legal/academic documents.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Section:
    """A section (§) within a chapter or appendix."""
    id: str  # e.g., "§1", "§40"
    number: int  # Numeric part for sorting
    title: str
    content: str
    is_ab_excerpt: bool = False  # True if this is from "Allgemeine Bestimmungen"


@dataclass
class Chapter:
    """A main chapter (I., II., III., IV.)."""
    id: str  # e.g., "I", "II"
    numeral: str  # Roman numeral
    title: str
    sections: list[Section] = field(default_factory=list)
    ab_excerpts: list[Section] = field(default_factory=list)  # AB sections within this chapter


@dataclass
class Appendix:
    """An appendix (Anlage/Anhang)."""
    id: str  # e.g., "Anlage 1", "Anlage 5"
    number: str  # "1", "2", etc.
    title: str
    content: str  # Content without sub-sections
    sections: list[Section] = field(default_factory=list)  # Sub-§ if any


@dataclass
class ParsedDocument:
    """Complete parsed document structure."""
    title: str
    preamble: str  # Content before ToC
    chapters: list[Chapter] = field(default_factory=list)
    appendices: list[Appendix] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_all_main_sections(self) -> list[Section]:
        """Get all main §1-§40 sections (not AB excerpts, not from appendices)."""
        sections = []
        for chapter in self.chapters:
            sections.extend(chapter.sections)
        return sections

    def get_all_ab_excerpts(self) -> list[Section]:
        """Get all AB excerpt sections."""
        excerpts = []
        for chapter in self.chapters:
            excerpts.extend(chapter.ab_excerpts)
        return excerpts

    def get_statistics(self) -> dict:
        """Get document statistics."""
        main_sections = self.get_all_main_sections()
        ab_excerpts = self.get_all_ab_excerpts()
        appendix_sections = sum(len(a.sections) for a in self.appendices)

        return {
            "chapters": len(self.chapters),
            "main_sections": len(main_sections),
            "ab_excerpts": len(ab_excerpts),
            "appendices": len(self.appendices),
            "appendix_sections": appendix_sections,
            "section_range": f"§{min(s.number for s in main_sections)}-§{max(s.number for s in main_sections)}" if main_sections else "N/A"
        }


class DocumentParser:
    """
    Parses German legal/academic documents into hierarchical structure.

    Designed to be general-purpose, not specific to any single document.
    """

    # Roman numerals pattern for chapters
    ROMAN_NUMERALS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']

    # Pattern for chapter headers: "I.\nAllgemeines" or "I. Allgemeines"
    CHAPTER_PATTERN = re.compile(
        r'^(I{1,3}V?|IV|V?I{1,3})\.\s*\n?\s*([A-ZÄÖÜ][^\n]+?)(?:\s*\.{3,}\s*\d+)?$',
        re.MULTILINE
    )

    # Pattern for ToC entries (contains ... and page number)
    TOC_PATTERN = re.compile(r'\.{3,}\s*\d+\s*$')

    # Pattern for main sections: "§ 1\nGeltungsbereich" or "§ 1 Geltungsbereich"
    SECTION_PATTERN = re.compile(
        r'^§\s*(\d+[a-z]?)\s*\n?\s*([A-ZÄÖÜ][^\n]+?)(?:\s*\.{3,}\s*\d+)?$',
        re.MULTILINE
    )

    # Invalid title prefixes (indicate inline references, not section headers)
    INVALID_TITLE_PREFIXES = ['Abs', 'Satz', 'Nr', 'Nummer', 'Buchst', 'Ziffer', 'Halbsatz']

    # Pattern for AB excerpt marker
    AB_MARKER_PATTERN = re.compile(
        r'Textauszug\s+aus\s+den\s+Allgemeinen\s+Bestimmungen\s*:?',
        re.IGNORECASE
    )

    # Pattern for appendices
    APPENDIX_PATTERN = re.compile(
        r'^(Anlage|Anhang)\s*(\d+)\s*[:\s]*([^\n]*?)(?:\s*\.{3,}\s*\d+)?$',
        re.MULTILINE | re.IGNORECASE
    )

    # Pattern to detect end of ToC (usually starts with page number + chapter)
    TOC_END_PATTERN = re.compile(r'^\d+\s*\n\s*(I\.)', re.MULTILINE)

    def __init__(self):
        """Initialize the document parser."""
        pass

    def parse(self, text: str, title: str = "", metadata: dict = None) -> ParsedDocument:
        """
        Parse document text into hierarchical structure.

        Args:
            text: Full document text.
            title: Document title.
            metadata: Additional metadata.

        Returns:
            ParsedDocument with complete structure.
        """
        metadata = metadata or {}

        # Step 1: Find where ToC ends and main content begins
        toc_end_pos = self._find_toc_end(text)

        # Step 2: Extract preamble (before ToC or start of content)
        preamble_end = self._find_preamble_end(text)
        preamble = text[:preamble_end].strip() if preamble_end > 0 else ""

        # Step 3: Get main content (after ToC)
        main_content = text[toc_end_pos:]

        # Step 4: Find where appendices start
        appendix_start = self._find_appendix_start(main_content)

        # Step 5: Parse main content (chapters and sections)
        main_text = main_content[:appendix_start] if appendix_start else main_content
        chapters = self._parse_chapters(main_text)

        # Step 6: Parse appendices
        appendix_text = main_content[appendix_start:] if appendix_start else ""
        appendices = self._parse_appendices(appendix_text)

        return ParsedDocument(
            title=title,
            preamble=preamble,
            chapters=chapters,
            appendices=appendices,
            metadata=metadata
        )

    def _find_toc_end(self, text: str) -> int:
        """Find where the Table of Contents ends."""
        # Look for pattern: page number followed by "I." chapter start
        match = self.TOC_END_PATTERN.search(text)
        if match:
            return match.start()

        # Fallback: Find "Inhaltsverzeichnis" and skip to next chapter
        toc_match = re.search(r'Inhaltsverzeichnis', text, re.IGNORECASE)
        if toc_match:
            # Find next occurrence of "I." that's not a ToC entry
            search_start = toc_match.end()
            for match in re.finditer(r'^\d+\s*\n\s*I\.', text[search_start:], re.MULTILINE):
                return search_start + match.start()

        return 0

    def _find_preamble_end(self, text: str) -> int:
        """Find where the preamble ends (before ToC)."""
        toc_match = re.search(r'Inhaltsverzeichnis', text, re.IGNORECASE)
        if toc_match:
            return toc_match.start()
        return 0

    def _find_appendix_start(self, text: str) -> Optional[int]:
        """Find where appendices start in the main content."""
        # Strategy: Find "Anlage 1" that is:
        # 1. NOT a ToC entry (no "..." followed by page number)
        # 2. NOT an inline reference (not preceded by certain words or ")")
        # 3. Followed by actual content, not just more ToC entries

        candidates = []

        for match in self.APPENDIX_PATTERN.finditer(text):
            full_match = match.group(0)
            number = match.group(2)
            title = match.group(3).strip() if match.group(3) else ""

            # Only consider Anlage 1 as the start marker
            if number != "1":
                continue

            # Skip ToC entries (contain ... and page numbers)
            if self.TOC_PATTERN.search(full_match) or self.TOC_PATTERN.search(title):
                continue

            # Skip inline references
            context_start = max(0, match.start() - 50)
            context = text[context_start:match.start()].lower()

            # Check for inline reference patterns
            is_inline = False
            if any(word in context for word in ['gemäß', 'siehe', 'nach', 'laut', 'in der']):
                is_inline = True
            if context.rstrip().endswith(')') or '(' in context[-20:]:
                is_inline = True

            if is_inline:
                continue

            # Check what comes after - real appendix headers are followed by content
            after_text = text[match.end():match.end() + 200]

            # If followed by another "Anlage" with ToC pattern, this is still ToC area
            if re.search(r'Anlage\s*\d+.*\.{3,}', after_text):
                continue

            candidates.append(match.start())

        # Return the first valid candidate (should be the real Anlage 1)
        return candidates[0] if candidates else None

    def _parse_chapters(self, text: str) -> list[Chapter]:
        """Parse main content into chapters with sections."""
        chapters = []

        # Find all chapter markers
        chapter_markers = []
        for match in self.CHAPTER_PATTERN.finditer(text):
            # Skip ToC entries
            if self.TOC_PATTERN.search(match.group(0)):
                continue

            chapter_markers.append({
                "position": match.start(),
                "end": match.end(),
                "numeral": match.group(1),
                "title": match.group(2).strip()
            })

        # Parse each chapter
        for i, marker in enumerate(chapter_markers):
            # Determine chapter content boundaries
            start = marker["end"]
            end = chapter_markers[i + 1]["position"] if i + 1 < len(chapter_markers) else len(text)

            chapter_content = text[start:end]

            # Parse sections within this chapter
            sections, ab_excerpts = self._parse_sections_in_chapter(chapter_content)

            chapter = Chapter(
                id=marker["numeral"],
                numeral=marker["numeral"],
                title=marker["title"],
                sections=sections,
                ab_excerpts=ab_excerpts
            )
            chapters.append(chapter)

        return chapters

    def _parse_sections_in_chapter(self, text: str) -> tuple[list[Section], list[Section]]:
        """Parse sections within a chapter, separating main sections from AB excerpts.

        Strategy: Main sections follow a monotonically increasing sequence.
        When a § number "goes backward" (e.g., after §17 comes §6), it's an AB excerpt.
        Also check for "Textauszug" markers preceding a section.
        """
        sections = []
        ab_excerpts = []

        # Find all section markers
        section_markers = []
        for match in self.SECTION_PATTERN.finditer(text):
            # Skip ToC entries
            if self.TOC_PATTERN.search(match.group(0)):
                continue

            number_str = match.group(1)
            try:
                number = int(re.match(r'\d+', number_str).group())
            except (ValueError, AttributeError):
                continue

            title = match.group(2).strip()

            # Skip inline references like "§ 30 Abs. 2" or "§ 5 Satz 1"
            title_start = title.split()[0] if title.split() else ""
            if any(title_start.startswith(prefix) for prefix in self.INVALID_TITLE_PREFIXES):
                continue

            section_markers.append({
                "position": match.start(),
                "end": match.end(),
                "number": number,
                "number_str": number_str,
                "title": title,
            })

        # Find all AB marker positions
        ab_marker_positions = [m.end() for m in self.AB_MARKER_PATTERN.finditer(text)]

        # Identify main sections vs AB excerpts
        # Strategy:
        # 1. A § is an AB excerpt if there's an AB marker DIRECTLY before it (no other § in between)
        # 2. Otherwise, use monotonic sequence: if number goes backward, it's an AB excerpt
        max_main_number = 0  # Track the highest main section number seen

        for i, marker in enumerate(section_markers):
            # Determine section content boundaries
            start = marker["end"]
            end = section_markers[i + 1]["position"] if i + 1 < len(section_markers) else len(text)
            content = text[start:end].strip()

            # Determine if this is a main section or AB excerpt
            is_ab = False

            # Check if there's an AB marker directly before this section
            # "Directly" means: AB marker exists between previous § and this §
            prev_section_end = section_markers[i - 1]["end"] if i > 0 else 0
            between_text_start = prev_section_end
            between_text_end = marker["position"]

            # Check if any AB marker falls in this range
            has_direct_ab_marker = any(
                between_text_start <= ab_pos <= between_text_end
                for ab_pos in ab_marker_positions
            )

            if has_direct_ab_marker:
                is_ab = True
            # If number goes backward, it's likely an AB excerpt
            elif marker["number"] < max_main_number:
                is_ab = True

            if not is_ab:
                max_main_number = max(max_main_number, marker["number"])

            section = Section(
                id=f"§{marker['number_str']}",
                number=marker["number"],
                title=marker["title"],
                content=content,
                is_ab_excerpt=is_ab
            )

            if is_ab:
                ab_excerpts.append(section)
            else:
                sections.append(section)

        return sections, ab_excerpts

    def _find_ab_ranges(self, text: str) -> list[tuple[int, int]]:
        """Find text ranges that are AB excerpts."""
        ranges = []

        for match in self.AB_MARKER_PATTERN.finditer(text):
            start = match.end()

            # Find the end of the AB excerpt (next main section or next AB marker)
            # AB excerpts typically end before the next non-AB section
            # We'll mark the range as starting after the marker

            # Look for next section that's NOT immediately after this marker
            next_section = None
            for sec_match in self.SECTION_PATTERN.finditer(text[start:]):
                # Skip ToC entries
                if self.TOC_PATTERN.search(sec_match.group(0)):
                    continue
                next_section = start + sec_match.start()
                break

            # The AB range goes from the marker to... we need a heuristic
            # Usually AB excerpts are followed by the next main section of the document
            # For now, mark a generous range

            end = next_section + 5000 if next_section else start + 5000
            end = min(end, len(text))

            ranges.append((start, end))

        return ranges

    def _is_in_ab_range(self, position: int, ranges: list[tuple[int, int]]) -> bool:
        """Check if a position is within an AB excerpt range."""
        for start, end in ranges:
            if start <= position <= end:
                return True
        return False

    def _parse_appendices(self, text: str) -> list[Appendix]:
        """Parse appendices with their sub-sections if any."""
        appendices = []

        # Find all appendix markers
        appendix_markers = []
        for match in self.APPENDIX_PATTERN.finditer(text):
            # Skip ToC entries
            if self.TOC_PATTERN.search(match.group(0)):
                continue

            # Skip inline references
            context_start = max(0, match.start() - 30)
            context = text[context_start:match.start()].lower()
            if any(word in context for word in ['gemäß', 'siehe', 'nach', 'laut', '(', ')']):
                continue

            appendix_markers.append({
                "position": match.start(),
                "end": match.end(),
                "number": match.group(2),
                "title": match.group(3).strip()
            })

        # Deduplicate by number (keep first occurrence that's a real header)
        seen_numbers = set()
        unique_markers = []
        for marker in appendix_markers:
            if marker["number"] not in seen_numbers:
                seen_numbers.add(marker["number"])
                unique_markers.append(marker)

        # Parse each appendix
        for i, marker in enumerate(unique_markers):
            start = marker["end"]
            end = unique_markers[i + 1]["position"] if i + 1 < len(unique_markers) else len(text)

            appendix_content = text[start:end]

            # Parse sub-sections within this appendix
            sub_sections = self._parse_appendix_sections(appendix_content)

            # Content without sections (or full content if no sections)
            if sub_sections:
                # Content before first section
                first_section_pos = self._find_first_section_pos(appendix_content)
                content = appendix_content[:first_section_pos].strip() if first_section_pos else ""
            else:
                content = appendix_content.strip()

            appendix = Appendix(
                id=f"Anlage {marker['number']}",
                number=marker["number"],
                title=marker["title"],
                content=content,
                sections=sub_sections
            )
            appendices.append(appendix)

        return appendices

    def _parse_appendix_sections(self, text: str) -> list[Section]:
        """Parse sections within an appendix."""
        sections = []

        section_markers = []
        for match in self.SECTION_PATTERN.finditer(text):
            # Skip ToC entries
            if self.TOC_PATTERN.search(match.group(0)):
                continue

            number_str = match.group(1)
            try:
                number = int(re.match(r'\d+', number_str).group())
            except (ValueError, AttributeError):
                continue

            title = match.group(2).strip()

            # Skip inline references like "§ 30 Abs. 2"
            title_start = title.split()[0] if title.split() else ""
            if any(title_start.startswith(prefix) for prefix in self.INVALID_TITLE_PREFIXES):
                continue

            section_markers.append({
                "position": match.start(),
                "end": match.end(),
                "number": number,
                "number_str": number_str,
                "title": title
            })

        for i, marker in enumerate(section_markers):
            start = marker["end"]
            end = section_markers[i + 1]["position"] if i + 1 < len(section_markers) else len(text)

            content = text[start:end].strip()

            section = Section(
                id=f"§{marker['number_str']}",
                number=marker["number"],
                title=marker["title"],
                content=content,
                is_ab_excerpt=False
            )
            sections.append(section)

        return sections

    def _find_first_section_pos(self, text: str) -> Optional[int]:
        """Find position of first section in text."""
        for match in self.SECTION_PATTERN.finditer(text):
            if self.TOC_PATTERN.search(match.group(0)):
                continue

            title = match.group(2).strip()
            title_start = title.split()[0] if title.split() else ""
            if any(title_start.startswith(prefix) for prefix in self.INVALID_TITLE_PREFIXES):
                continue

            return match.start()
        return None
