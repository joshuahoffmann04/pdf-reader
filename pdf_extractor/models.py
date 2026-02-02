"""
Data Models for PDF Section Extraction.

This module defines the core data structures for the page-by-page scanning
extraction pipeline. The pipeline works in four phases:

1. PAGE SCAN: Each page is analyzed individually to detect which sections appear on it
2. STRUCTURE: Section locations are aggregated from scan results to build page ranges
3. CONTEXT: Document metadata is extracted from representative pages
4. EXTRACTION: Full content is extracted for each section using accurate page ranges

Architecture:
    PDF → [Page Scan] → PageScanResult[]
                            ↓
        [Aggregate] → SectionLocation[]
                            ↓
        [Context]   → DocumentContext
                            ↓
        [Extract]   → ExtractedSection[]
                            ↓
                    ExtractionResult

Design Principles:
    - Pydantic v2 for validation, serialization, and JSON Schema generation
    - Immutable models where possible (frozen=True for value objects)
    - Clear separation between internal processing models and output models
    - German academic documents focus (Prüfungsordnungen, Modulhandbücher)

Usage:
    from pdf_extractor import PDFExtractor, ExtractionConfig

    extractor = PDFExtractor()
    result = extractor.extract("pruefungsordnung.pdf")

    for section in result.sections:
        print(f"{section.identifier}: {section.content[:100]}...")
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator, computed_field


# =============================================================================
# ENUMS
# =============================================================================


class SectionType(str, Enum):
    """
    Classification of structural elements in German academic documents.

    PREAMBLE: Content before the first numbered section (title page, ToC, preface)
    PARAGRAPH: Numbered sections (§ 1, § 2, ... § 40)
    ANLAGE: Appendices (Anlage 1, Anlage 2, ...)
    """

    PREAMBLE = "preamble"
    PARAGRAPH = "paragraph"
    ANLAGE = "anlage"


class DocumentType(str, Enum):
    """
    Classification of German academic document types.

    Used for retrieval filtering and context-aware processing.
    """

    PRUEFUNGSORDNUNG = "pruefungsordnung"
    MODULHANDBUCH = "modulhandbuch"
    STUDIENORDNUNG = "studienordnung"
    ALLGEMEINE_BESTIMMUNGEN = "allgemeine_bestimmungen"
    PRAKTIKUMSORDNUNG = "praktikumsordnung"
    ZULASSUNGSORDNUNG = "zulassungsordnung"
    SATZUNG = "satzung"
    OTHER = "other"


class Language(str, Enum):
    """Supported document languages."""

    DE = "de"
    EN = "en"


# =============================================================================
# PHASE 1: PAGE SCAN MODELS
# =============================================================================


class DetectedSection(BaseModel):
    """
    A section detected on a single page during the scan phase.

    This model represents what the LLM found on ONE page. Multiple pages
    may contain the same section, which is later aggregated into SectionLocation.

    Examples:
        - {"section_type": "paragraph", "identifier": "§ 5", "title": "Regelstudienzeit"}
        - {"section_type": "anlage", "identifier": "Anlage 1", "title": "Modulübersicht"}
    """

    section_type: SectionType = Field(
        ...,
        description="Type of section (paragraph, anlage, preamble)"
    )
    identifier: Optional[str] = Field(
        None,
        description="Section identifier (e.g., '§ 5', 'Anlage 1'). None for preamble."
    )
    title: Optional[str] = Field(
        None,
        description="Section title if visible on this page"
    )

    model_config = {"frozen": True}

    def __hash__(self) -> int:
        """Allow use in sets for deduplication."""
        return hash((self.section_type, self.identifier))

    def __eq__(self, other: object) -> bool:
        """Equality based on type and identifier."""
        if not isinstance(other, DetectedSection):
            return False
        return self.section_type == other.section_type and self.identifier == other.identifier


class PageScanResult(BaseModel):
    """
    Result of scanning a single page for sections.

    Each page is scanned individually to determine which sections appear on it.
    This allows accurate page range calculation even when sections share pages.

    Example:
        Page 5 might contain the end of § 4 and the beginning of § 5:
        {"page_number": 5, "sections": [
            {"section_type": "paragraph", "identifier": "§ 4", "title": null},
            {"section_type": "paragraph", "identifier": "§ 5", "title": "Regelstudienzeit"}
        ]}
    """

    page_number: int = Field(
        ...,
        ge=1,
        description="Page number (1-indexed, matches PDF page number)"
    )
    sections: list[DetectedSection] = Field(
        default_factory=list,
        description="Sections detected on this page"
    )
    is_empty: bool = Field(
        False,
        description="True if page contains no relevant content"
    )
    scan_notes: Optional[str] = Field(
        None,
        description="Optional notes about the page (e.g., 'Deckblatt', 'leere Seite')"
    )

    model_config = {"frozen": True}


# =============================================================================
# PHASE 2: STRUCTURE MODELS
# =============================================================================


class SectionLocation(BaseModel):
    """
    Aggregated location of a section across all pages.

    Built from PageScanResult data by finding all pages where a section appears.
    This gives us accurate page ranges even for sections that span partial pages.

    Example:
        § 5 appears on pages [5, 6, 7]:
        - Starts in the middle of page 5 (after § 4 ends)
        - Continues through page 6
        - Ends in the first half of page 7 (before § 6 starts)
    """

    section_type: SectionType = Field(
        ...,
        description="Type of section"
    )
    identifier: Optional[str] = Field(
        None,
        description="Section identifier (e.g., '§ 5', 'Anlage 1')"
    )
    title: Optional[str] = Field(
        None,
        description="Section title (from the page where it was most clearly visible)"
    )
    pages: list[int] = Field(
        ...,
        min_length=1,
        description="All pages where this section appears (1-indexed, sorted)"
    )

    @computed_field
    @property
    def start_page(self) -> int:
        """First page of this section."""
        return self.pages[0]

    @computed_field
    @property
    def end_page(self) -> int:
        """Last page of this section."""
        return self.pages[-1]

    @computed_field
    @property
    def page_count(self) -> int:
        """Number of pages this section spans."""
        return len(self.pages)

    @computed_field
    @property
    def display_name(self) -> str:
        """Human-readable name for this section."""
        if self.identifier and self.title:
            return f"{self.identifier} {self.title}"
        if self.identifier:
            return self.identifier
        if self.title:
            return self.title
        return "Präambel"

    def to_structure_entry(self) -> "StructureEntry":
        """Convert to StructureEntry for backward compatibility."""
        return StructureEntry(
            section_type=self.section_type,
            section_number=self.identifier,
            section_title=self.title,
            start_page=self.start_page,
            end_page=self.end_page,
        )


class DocumentStructure(BaseModel):
    """
    Complete structure of a document derived from page scanning.

    Contains all sections with their accurate page ranges, ready for extraction.
    """

    sections: list[SectionLocation] = Field(
        default_factory=list,
        description="All sections in document order"
    )
    total_pages: int = Field(
        ...,
        ge=1,
        description="Total number of pages in the document"
    )
    has_preamble: bool = Field(
        False,
        description="Whether the document has content before the first numbered section"
    )
    scan_results: list[PageScanResult] = Field(
        default_factory=list,
        description="Raw scan results for debugging/verification"
    )

    def get_section(self, identifier: str) -> Optional[SectionLocation]:
        """Find a section by its identifier."""
        for section in self.sections:
            if section.identifier == identifier:
                return section
        return None

    def get_paragraphs(self) -> list[SectionLocation]:
        """Get all § sections."""
        return [s for s in self.sections if s.section_type == SectionType.PARAGRAPH]

    def get_anlagen(self) -> list[SectionLocation]:
        """Get all Anlage sections."""
        return [s for s in self.sections if s.section_type == SectionType.ANLAGE]

    def get_preamble(self) -> Optional[SectionLocation]:
        """Get the preamble section if it exists."""
        for s in self.sections:
            if s.section_type == SectionType.PREAMBLE:
                return s
        return None


# =============================================================================
# BACKWARD COMPATIBILITY: StructureEntry
# =============================================================================


class StructureEntry(BaseModel):
    """
    Legacy structure entry for backward compatibility.

    Prefer using SectionLocation for new code.
    """

    section_type: SectionType = Field(
        ...,
        description="Type of section"
    )
    section_number: Optional[str] = Field(
        None,
        description="Section identifier"
    )
    section_title: Optional[str] = Field(
        None,
        description="Section title"
    )
    start_page: int = Field(
        ...,
        ge=1,
        description="First page (1-indexed)"
    )
    end_page: int = Field(
        ...,
        ge=1,
        description="Last page (1-indexed, inclusive)"
    )

    @model_validator(mode="after")
    def validate_page_range(self) -> "StructureEntry":
        """Ensure end_page >= start_page."""
        if self.end_page < self.start_page:
            raise ValueError(
                f"end_page ({self.end_page}) must be >= start_page ({self.start_page})"
            )
        return self

    @property
    def pages(self) -> list[int]:
        """All pages in this section."""
        return list(range(self.start_page, self.end_page + 1))

    @property
    def identifier(self) -> str:
        """Human-readable identifier."""
        if self.section_number:
            return self.section_number
        return "Präambel"


# =============================================================================
# PHASE 3: DOCUMENT CONTEXT
# =============================================================================


class Abbreviation(BaseModel):
    """An abbreviation and its full form."""

    short: str = Field(..., description="Abbreviation (e.g., 'LP', 'ECTS')")
    long: str = Field(..., description="Full form (e.g., 'Leistungspunkte')")

    model_config = {"frozen": True}


class DocumentContext(BaseModel):
    """
    Document-level metadata extracted from representative pages.

    Contains information about the document that applies to all sections
    and enriches the extraction results.
    """

    # Required identification
    document_type: DocumentType = Field(
        ...,
        description="Type of document"
    )
    title: str = Field(
        ...,
        description="Official document title",
        min_length=1
    )
    institution: str = Field(
        ...,
        description="Issuing institution (university, faculty)"
    )

    # Version information
    version_date: Optional[str] = Field(
        None,
        description="Publication or effective date"
    )
    version_info: Optional[str] = Field(
        None,
        description="Version description (e.g., 'Nichtamtliche Lesefassung')"
    )

    # Scope
    degree_program: Optional[str] = Field(
        None,
        description="Degree program (e.g., 'Bachelor of Science Informatik')"
    )
    faculty: Optional[str] = Field(
        None,
        description="Faculty or department"
    )

    # Structure info
    total_pages: int = Field(
        0,
        ge=0,
        description="Total number of pages"
    )
    chapters: list[str] = Field(
        default_factory=list,
        description="Main chapter headings (e.g., 'I. Allgemeines', 'II. Prüfungen')"
    )

    # Terminology
    abbreviations: list[Abbreviation] = Field(
        default_factory=list,
        description="Abbreviations used in the document"
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="Important domain-specific terms"
    )

    # External references
    referenced_documents: list[str] = Field(
        default_factory=list,
        description="Other documents referenced"
    )
    legal_basis: Optional[str] = Field(
        None,
        description="Legal basis for the document"
    )

    # Language
    language: Language = Field(
        Language.DE,
        description="Primary language"
    )

    def get_abbreviation_dict(self) -> dict[str, str]:
        """Get abbreviations as a lookup dictionary."""
        return {a.short: a.long for a in self.abbreviations}


# =============================================================================
# PHASE 4: EXTRACTION OUTPUT
# =============================================================================


class ExtractedSection(BaseModel):
    """
    A fully extracted section from the document.

    This is the main output model containing the complete content
    of a section ready for downstream processing (chunking, RAG indexing).
    """

    # Identification
    section_type: SectionType = Field(
        ...,
        description="Type of section"
    )
    section_number: Optional[str] = Field(
        None,
        description="Section identifier (e.g., '§ 10', 'Anlage 2')"
    )
    section_title: Optional[str] = Field(
        None,
        description="Section title"
    )

    # Content
    content: str = Field(
        ...,
        description="Complete text content in natural language"
    )

    # Position
    pages: list[int] = Field(
        default_factory=list,
        description="Page numbers where this section appears (1-indexed)"
    )
    chapter: Optional[str] = Field(
        None,
        description="Parent chapter (e.g., 'II. Studienbezogene Bestimmungen')"
    )

    # Internal structure
    subsections: list[str] = Field(
        default_factory=list,
        description="Numbered subsections found (e.g., ['(1)', '(2)', '(3)'])"
    )

    # References
    internal_references: list[str] = Field(
        default_factory=list,
        description="References to other sections (e.g., ['§ 5 Abs. 2', 'Anlage 1'])"
    )
    external_references: list[str] = Field(
        default_factory=list,
        description="References to external documents"
    )

    # Content properties
    has_table: bool = Field(
        False,
        description="Contains tabular data"
    )
    has_list: bool = Field(
        False,
        description="Contains enumerated lists"
    )

    # Metadata
    extraction_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction quality (1.0 = perfect)"
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Notes about extraction issues"
    )

    @computed_field
    @property
    def token_estimate(self) -> int:
        """Estimated token count (~4 chars per token for German)."""
        return len(self.content) // 4

    @computed_field
    @property
    def identifier(self) -> str:
        """Human-readable identifier."""
        if self.section_number and self.section_title:
            return f"{self.section_number} {self.section_title}"
        if self.section_number:
            return self.section_number
        if self.section_title:
            return self.section_title
        return "Präambel"

    def format_source_reference(self, doc_title: str = "") -> str:
        """
        Generate a source reference for RAG citations.

        Args:
            doc_title: Document title to include

        Returns:
            Formatted reference (e.g., "Prüfungsordnung, § 10, S. 12-14")
        """
        page_str = self._format_pages()
        section_str = self.section_number or "Präambel"

        if self.section_title and self.section_number:
            section_str = f"{self.section_number} {self.section_title}"

        if doc_title:
            return f"{doc_title}, {section_str}, S. {page_str}"
        return f"{section_str}, S. {page_str}"

    def _format_pages(self) -> str:
        """Format page numbers for display."""
        if not self.pages:
            return "?"
        if len(self.pages) == 1:
            return str(self.pages[0])
        return f"{self.pages[0]}-{self.pages[-1]}"


# =============================================================================
# CONFIGURATION
# =============================================================================


class ExtractionConfig(BaseModel):
    """
    Configuration for the PDF extraction pipeline.

    Controls API settings, processing behavior, and output options.
    """

    # API Configuration
    model: str = Field(
        "gpt-4o",
        description="OpenAI model to use for extraction"
    )
    max_tokens: int = Field(
        4096,
        ge=100,
        le=128000,
        description="Maximum tokens per API response"
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 = deterministic)"
    )

    # Retry configuration
    max_retries: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed API calls"
    )
    retry_delay: float = Field(
        1.0,
        ge=0.1,
        le=30.0,
        description="Initial delay between retries (seconds)"
    )

    # Processing options
    extract_preamble: bool = Field(
        True,
        description="Extract content before the first numbered section"
    )
    include_scan_results: bool = Field(
        False,
        description="Include raw page scan results in output (for debugging)"
    )

    # Output language
    language: Language = Field(
        Language.DE,
        description="Output language for extracted content"
    )


# =============================================================================
# FINAL RESULT
# =============================================================================


class ExtractionResult(BaseModel):
    """
    Complete result of PDF section extraction.

    Contains all extracted sections, document metadata, and processing statistics.
    Ready for downstream processing (chunking, embedding, RAG indexing).
    """

    # Source
    source_file: str = Field(
        ...,
        description="Path to source PDF file"
    )

    # Document information
    context: DocumentContext = Field(
        ...,
        description="Document-level metadata"
    )
    structure: DocumentStructure = Field(
        ...,
        description="Document structure with page ranges"
    )

    # Extracted content
    sections: list[ExtractedSection] = Field(
        default_factory=list,
        description="All extracted sections"
    )

    # Processing statistics
    processing_time_seconds: float = Field(
        0.0,
        description="Total processing time"
    )
    total_input_tokens: int = Field(
        0,
        description="Total input tokens consumed"
    )
    total_output_tokens: int = Field(
        0,
        description="Total output tokens generated"
    )

    # Quality information
    errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered during processing"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings generated during processing"
    )

    # Timestamp
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Extraction timestamp"
    )

    # --- Query Methods ---

    def get_section(self, identifier: str) -> Optional[ExtractedSection]:
        """Get a section by its identifier (e.g., '§ 10', 'Anlage 2')."""
        for section in self.sections:
            if section.section_number == identifier:
                return section
        return None

    def get_preamble(self) -> Optional[ExtractedSection]:
        """Get the preamble section if it exists."""
        for section in self.sections:
            if section.section_type == SectionType.PREAMBLE:
                return section
        return None

    def get_paragraphs(self) -> list[ExtractedSection]:
        """Get all § sections."""
        return [s for s in self.sections if s.section_type == SectionType.PARAGRAPH]

    def get_anlagen(self) -> list[ExtractedSection]:
        """Get all Anlage sections."""
        return [s for s in self.sections if s.section_type == SectionType.ANLAGE]

    # --- Statistics ---

    @computed_field
    @property
    def total_sections(self) -> int:
        """Total number of extracted sections."""
        return len(self.sections)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total estimated tokens in extracted content."""
        return sum(s.token_estimate for s in self.sections)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive extraction statistics."""
        return {
            "total_sections": self.total_sections,
            "paragraphs": len(self.get_paragraphs()),
            "anlagen": len(self.get_anlagen()),
            "has_preamble": self.get_preamble() is not None,
            "total_pages": self.context.total_pages,
            "total_content_tokens": self.total_tokens,
            "avg_tokens_per_section": self.total_tokens // max(1, self.total_sections),
            "sections_with_tables": sum(1 for s in self.sections if s.has_table),
            "sections_with_lists": sum(1 for s in self.sections if s.has_list),
            "low_confidence_sections": sum(
                1 for s in self.sections if s.extraction_confidence < 1.0
            ),
            "api_tokens_used": self.total_input_tokens + self.total_output_tokens,
            "processing_time_seconds": self.processing_time_seconds,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }

    # --- Export Methods ---

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for serialization."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Export as formatted JSON string."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str) -> None:
        """Save extraction result to a JSON file."""
        from pathlib import Path
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ExtractionResult":
        """Load extraction result from a JSON file."""
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def get_full_text(self, separator: str = "\n\n---\n\n") -> str:
        """Concatenate all section contents into a single string."""
        return separator.join(s.content for s in self.sections)
