"""
Section Parser Module

Parses extracted PDF text to identify and structure §-sections.
Distinguishes between main structural § markers and inline § references.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Section:
    """Represents a parsed section from the document."""
    id: str  # e.g., "§1", "§40", "Anhang 1"
    title: str  # e.g., "Geltungsbereich"
    content: str  # Full text content of the section
    subsections: list["Section"] = field(default_factory=list)
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class ParsedDocument:
    """Represents a fully parsed document with sections."""
    title: str
    metadata: dict
    sections: list[Section]
    appendices: list[Section]  # Anlagen/Anhänge
    preamble: str  # Content before first §


class SectionParser:
    """
    Parses document text into structured sections based on § markers.

    Recognizes:
    - Main sections: § followed by number and title (e.g., "§ 1 Geltungsbereich")
    - Appendices: "Anlage" or "Anhang" followed by number/letter
    - Distinguishes from inline references (e.g., "gemäß § 50 Abs. 1")
    """

    # Pattern for main § sections: § + number + title starting with capital letter
    # Must be at start of line or after newline
    MAIN_SECTION_PATTERN = re.compile(
        r'^[§]\s*(\d+[a-z]?)\s+([A-ZÄÖÜ][^\n]+)',
        re.MULTILINE
    )

    # Alternative pattern for sections with line break between number and title
    MAIN_SECTION_PATTERN_ALT = re.compile(
        r'^[§]\s*(\d+[a-z]?)\s*\n\s*([A-ZÄÖÜ][^\n]+)',
        re.MULTILINE
    )

    # Pattern to detect ToC entries (title followed by dots and page number)
    TOC_ENTRY_PATTERN = re.compile(r'\.{3,}\s*\d+\s*$')

    # Pattern for appendices (Anlage/Anhang)
    APPENDIX_PATTERN = re.compile(
        r'^(Anlage|Anhang)\s*(\d+|[A-Z]|[IVX]+)?\s*[:\-]?\s*([^\n]*)',
        re.MULTILINE | re.IGNORECASE
    )

    # Pattern for inline § references (to be preserved in text, not parsed as sections)
    REFERENCE_PATTERN = re.compile(
        r'(?:gemäß|nach|siehe|vgl\.|entsprechend|i\.?\s*S\.?\s*d\.?|im Sinne)\s+§\s*\d+',
        re.IGNORECASE
    )

    def __init__(self, language: str = "de"):
        """
        Initialize the section parser.

        Args:
            language: Document language ("de" for German, "en" for English).
        """
        self.language = language

    def parse(self, text: str, title: str = "", metadata: dict = None) -> ParsedDocument:
        """
        Parse document text into structured sections.

        Args:
            text: Full document text.
            title: Document title.
            metadata: Document metadata.

        Returns:
            ParsedDocument with parsed sections.
        """
        metadata = metadata or {}

        # Find all main section markers
        section_markers = self._find_section_markers(text)

        # Find all appendix markers
        appendix_markers = self._find_appendix_markers(text)

        # Extract preamble (content before first section)
        preamble = ""
        if section_markers:
            first_pos = section_markers[0]["position"]
            preamble = text[:first_pos].strip()

        # Parse main sections
        sections = self._extract_sections(text, section_markers, appendix_markers)

        # Parse appendices
        appendices = self._extract_appendices(text, appendix_markers)

        return ParsedDocument(
            title=title,
            metadata=metadata,
            sections=sections,
            appendices=appendices,
            preamble=preamble
        )

    def _find_section_markers(self, text: str) -> list[dict]:
        """Find all main § section markers in the text."""
        markers = []

        # Find matches with standard pattern
        for match in self.MAIN_SECTION_PATTERN.finditer(text):
            # Check if this is likely a reference (preceded by reference words)
            start_pos = match.start()
            context_start = max(0, start_pos - 50)
            context = text[context_start:start_pos]

            if self._is_reference_context(context):
                continue

            title = match.group(2).strip()

            # Skip ToC entries (contain dots followed by page numbers)
            if self.TOC_ENTRY_PATTERN.search(title):
                continue

            markers.append({
                "position": match.start(),
                "end": match.end(),
                "number": match.group(1),
                "title": title,
                "full_match": match.group(0)
            })

        # Also check alternative pattern (newline between number and title)
        for match in self.MAIN_SECTION_PATTERN_ALT.finditer(text):
            start_pos = match.start()

            # Skip if we already found this position
            if any(m["position"] == start_pos for m in markers):
                continue

            context_start = max(0, start_pos - 50)
            context = text[context_start:start_pos]

            if self._is_reference_context(context):
                continue

            title = match.group(2).strip()

            # Skip ToC entries
            if self.TOC_ENTRY_PATTERN.search(title):
                continue

            markers.append({
                "position": match.start(),
                "end": match.end(),
                "number": match.group(1),
                "title": title,
                "full_match": match.group(0)
            })

        # Sort by position
        markers.sort(key=lambda x: x["position"])

        return markers

    def _find_appendix_markers(self, text: str) -> list[dict]:
        """Find all appendix/Anlage markers in the text."""
        markers = []

        for match in self.APPENDIX_PATTERN.finditer(text):
            appendix_type = match.group(1)  # "Anlage" or "Anhang"
            number = match.group(2) or ""
            title = match.group(3).strip() if match.group(3) else ""

            # Skip ToC entries
            if self.TOC_ENTRY_PATTERN.search(title):
                continue

            markers.append({
                "position": match.start(),
                "end": match.end(),
                "type": appendix_type,
                "number": number,
                "title": title,
                "full_match": match.group(0)
            })

        # Sort by position
        markers.sort(key=lambda x: x["position"])

        return markers

    def _is_reference_context(self, context: str) -> bool:
        """Check if the context suggests this is a reference, not a section header."""
        reference_words = [
            "gemäß", "nach", "siehe", "vgl.", "entsprechend",
            "i.s.d.", "im sinne", "abs.", "absatz", "satz",
            "in", "des", "der", "den", "dem"
        ]

        context_lower = context.lower().strip()

        # Check if context ends with a reference word
        for word in reference_words:
            if context_lower.endswith(word):
                return True

        return False

    def _extract_sections(
        self,
        text: str,
        section_markers: list[dict],
        appendix_markers: list[dict]
    ) -> list[Section]:
        """Extract section content based on markers."""
        sections = []

        # Determine where appendices start (to know where last section ends)
        appendix_start = len(text)
        if appendix_markers:
            appendix_start = appendix_markers[0]["position"]

        for i, marker in enumerate(section_markers):
            # Determine end of this section
            if i + 1 < len(section_markers):
                # Next section starts
                section_end = section_markers[i + 1]["position"]
            else:
                # This is the last section - ends at appendices or end of text
                section_end = appendix_start

            # Extract content (skip the header itself)
            content_start = marker["end"]
            content = text[content_start:section_end].strip()

            section = Section(
                id=f"§{marker['number']}",
                title=marker["title"],
                content=content,
            )

            sections.append(section)

        return sections

    def _extract_appendices(
        self,
        text: str,
        appendix_markers: list[dict]
    ) -> list[Section]:
        """Extract appendix content based on markers."""
        appendices = []

        for i, marker in enumerate(appendix_markers):
            # Determine end of this appendix
            if i + 1 < len(appendix_markers):
                appendix_end = appendix_markers[i + 1]["position"]
            else:
                appendix_end = len(text)

            # Extract content
            content_start = marker["end"]
            content = text[content_start:appendix_end].strip()

            # Build ID
            number = marker["number"] or str(i + 1)
            appendix_id = f"Anhang {number}"

            section = Section(
                id=appendix_id,
                title=marker["title"],
                content=content,
            )

            appendices.append(section)

        return appendices

    def get_section_by_id(self, doc: ParsedDocument, section_id: str) -> Optional[Section]:
        """
        Get a specific section by its ID.

        Args:
            doc: Parsed document.
            section_id: Section ID (e.g., "§1", "Anhang 1").

        Returns:
            Section if found, None otherwise.
        """
        # Normalize the ID
        normalized_id = section_id.replace(" ", "").lower()

        for section in doc.sections:
            if section.id.replace(" ", "").lower() == normalized_id:
                return section

        for appendix in doc.appendices:
            if appendix.id.replace(" ", "").lower() == normalized_id:
                return appendix

        return None

    def get_toc(self, doc: ParsedDocument) -> list[dict]:
        """
        Generate a table of contents from the parsed document.

        Returns:
            List of dicts with section IDs and titles.
        """
        toc = []

        for section in doc.sections:
            toc.append({
                "id": section.id,
                "title": section.title,
                "type": "section"
            })

        for appendix in doc.appendices:
            toc.append({
                "id": appendix.id,
                "title": appendix.title,
                "type": "appendix"
            })

        return toc
