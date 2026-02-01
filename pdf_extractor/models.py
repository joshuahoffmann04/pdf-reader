"""
Data Models for Section-Based PDF Extraction Pipeline.

This module defines the core data structures for:
1. Document context (metadata extracted from the document)
2. Document structure (sections, their types, and page ranges)
3. Extracted sections (complete content of each section)
4. Processing configuration

Design Principles:
- Pydantic v2 for validation, serialization, and JSON Schema generation
- Section-based extraction (§§, Anlagen) instead of page-based
- Optimized for downstream processing (chunking, RAG)
- Clear source references for each section

Usage:
    from pdf_extractor import PDFExtractor, ExtractionConfig

    config = ExtractionConfig(max_images_per_request=5)
    extractor = PDFExtractor(config=config)
    result = extractor.extract("document.pdf")

    for section in result.sections:
        print(f"{section.section_number}: {section.section_title}")
        print(f"  Pages: {section.pages}")
        print(f"  Content: {section.content[:100]}...")
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# ENUMS
# =============================================================================

class SectionType(str, Enum):
    """
    Type of extracted section.

    Used to distinguish between different structural elements in the document.
    """
    OVERVIEW = "overview"      # Präambel, Inhaltsverzeichnis, Einleitung (before first §)
    PARAGRAPH = "paragraph"    # § with number (§ 1, § 2, ... § 40)
    ANLAGE = "anlage"          # Anlage 1, Anlage 2, ...


class DocumentType(str, Enum):
    """
    Classification of document types.
    Used for retrieval filtering and context-aware generation.
    """
    PRUEFUNGSORDNUNG = "pruefungsordnung"
    MODULHANDBUCH = "modulhandbuch"
    STUDIENORDNUNG = "studienordnung"
    ALLGEMEINE_BESTIMMUNGEN = "allgemeine_bestimmungen"
    PRAKTIKUMSORDNUNG = "praktikumsordnung"
    ZULASSUNGSORDNUNG = "zulassungsordnung"
    SATZUNG = "satzung"
    WEBSITE = "website"
    FAQ = "faq"
    OTHER = "other"


class Language(str, Enum):
    """Supported languages."""
    DE = "de"
    EN = "en"


# =============================================================================
# STRUCTURE MODELS (Internal use for structure mapping)
# =============================================================================

class StructureEntry(BaseModel):
    """
    An entry in the document structure map.

    Created during Phase 1 (structure analysis) from the table of contents.
    Used to determine which pages to send for each section extraction.
    """
    section_type: SectionType = Field(
        ...,
        description="Type of section (overview, paragraph, anlage)"
    )
    section_number: Optional[str] = Field(
        None,
        description="Section identifier (e.g., '§ 10', 'Anlage 2', None for overview)"
    )
    section_title: Optional[str] = Field(
        None,
        description="Section title (e.g., 'Module und Leistungspunkte')"
    )
    start_page: int = Field(
        ...,
        ge=1,
        description="First page of the section (1-indexed)"
    )
    end_page: int = Field(
        ...,
        ge=1,
        description="Last page of the section (1-indexed, inclusive)"
    )

    @model_validator(mode='after')
    def validate_pages(self) -> 'StructureEntry':
        """Ensure end_page >= start_page."""
        if self.end_page < self.start_page:
            raise ValueError(
                f"end_page ({self.end_page}) must be >= start_page ({self.start_page})"
            )
        return self

    @property
    def pages(self) -> list[int]:
        """Get list of all pages in this section."""
        return list(range(self.start_page, self.end_page + 1))

    @property
    def page_count(self) -> int:
        """Get number of pages in this section."""
        return self.end_page - self.start_page + 1

    @property
    def identifier(self) -> str:
        """Get human-readable identifier for this section."""
        if self.section_number:
            return self.section_number
        return "Übersicht"


# =============================================================================
# EXTRACTED SECTION (Main output model)
# =============================================================================

class ExtractedSection(BaseModel):
    """
    A fully extracted section from the document.

    This is the main output model containing the complete content
    of a section (§, Anlage, or Overview) ready for downstream processing.
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
        description="Complete text content of the section in natural language"
    )

    # Position in document
    pages: list[int] = Field(
        default_factory=list,
        description="Page numbers where this section appears (1-indexed)"
    )
    chapter: Optional[str] = Field(
        None,
        description="Parent chapter (e.g., 'II. Studienbezogene Bestimmungen')"
    )

    # Internal structure
    paragraphs: list[str] = Field(
        default_factory=list,
        description="Paragraph numbers found in section (e.g., ['(1)', '(2)', '(3)'])"
    )

    # References
    internal_references: list[str] = Field(
        default_factory=list,
        description="References to other sections (e.g., ['§ 5 Abs. 2', 'Anlage 1'])"
    )
    external_references: list[str] = Field(
        default_factory=list,
        description="References to external documents (e.g., ['Allgemeine Bestimmungen'])"
    )

    # Content properties
    has_table: bool = Field(
        False,
        description="Section contains tabular data (converted to text)"
    )
    has_list: bool = Field(
        False,
        description="Section contains lists (converted to text)"
    )

    # Metadata
    token_count: int = Field(
        0,
        description="Estimated token count (len(content) // 4)"
    )
    extraction_confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction quality"
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Notes about extraction issues"
    )

    @model_validator(mode='after')
    def calculate_token_count(self) -> 'ExtractedSection':
        """Auto-calculate token count if not set."""
        if self.token_count == 0 and self.content:
            # Rough estimate: ~4 characters per token for German text
            self.token_count = len(self.content) // 4
        return self

    @property
    def identifier(self) -> str:
        """Get human-readable identifier."""
        if self.section_number:
            if self.section_title:
                return f"{self.section_number} {self.section_title}"
            return self.section_number
        return "Übersicht"

    def get_source_reference(self, doc_title: str = "") -> str:
        """
        Generate a source reference string for RAG citations.

        Args:
            doc_title: Document title to include in reference

        Returns:
            Formatted source reference string

        Example:
            >>> section.get_source_reference("Prüfungsordnung Informatik")
            "Prüfungsordnung Informatik, § 10 Module und Leistungspunkte, S. 12-14"
        """
        page_str = self._format_pages()

        if self.section_type == SectionType.OVERVIEW:
            if doc_title:
                return f"{doc_title}, Übersicht, S. {page_str}"
            return f"Übersicht, S. {page_str}"

        section_str = self.section_number or ""
        if self.section_title:
            section_str = f"{section_str} {self.section_title}".strip()

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

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return self.model_dump()


# =============================================================================
# DOCUMENT CONTEXT
# =============================================================================

class Abbreviation(BaseModel):
    """An abbreviation and its expansion."""
    short: str = Field(..., description="The abbreviation (e.g., 'LP', 'ECTS')")
    long: str = Field(..., description="The full form (e.g., 'Leistungspunkte')")


class DocumentContext(BaseModel):
    """
    Document-level context extracted during structure analysis.

    Contains metadata about the document that enriches all extracted sections.
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
        description="Issuing institution"
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
        description="Degree program if applicable"
    )
    faculty: Optional[str] = Field(
        None,
        description="Faculty or department"
    )

    # Structure
    total_pages: int = Field(
        0,
        ge=0,
        description="Total number of pages"
    )
    chapters: list[str] = Field(
        default_factory=list,
        description="Main chapters (e.g., ['I. Allgemeines', 'II. Prüfungen'])"
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

    # Metadata
    language: Language = Field(
        Language.DE,
        description="Primary language"
    )

    def get_abbreviation_dict(self) -> dict[str, str]:
        """Get abbreviations as a simple dictionary."""
        return {a.short: a.long for a in self.abbreviations}

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary."""
        return self.model_dump()


# =============================================================================
# EXTRACTION CONFIGURATION
# =============================================================================

class ExtractionConfig(BaseModel):
    """
    Configuration for the section-based PDF extraction pipeline.

    Controls API settings, image handling, and retry behavior.
    """

    # API Configuration
    model: str = Field(
        "gpt-4o",
        description="OpenAI model to use"
    )
    max_tokens_per_request: int = Field(
        4096,
        ge=100,
        le=100000,
        description="Maximum tokens per API request"
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0 for deterministic)"
    )

    # Image handling
    max_images_per_request: int = Field(
        5,
        ge=1,
        le=20,
        description="Maximum images per API request (for sliding window)"
    )

    # Retry configuration
    max_retries: int = Field(
        3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed extractions"
    )

    # Language
    language: Language = Field(
        Language.DE,
        description="Output language"
    )


# =============================================================================
# EXTRACTION RESULT
# =============================================================================

class ExtractionResult(BaseModel):
    """
    Complete result of section-based PDF extraction.

    Contains all extracted sections and processing metadata.
    Ready for downstream processing (chunking, RAG indexing).
    """

    # Document information
    source_file: str = Field(
        ...,
        description="Path to source PDF"
    )
    context: DocumentContext = Field(
        ...,
        description="Document-level context"
    )

    # Extracted content (SECTIONS instead of pages)
    sections: list[ExtractedSection] = Field(
        default_factory=list,
        description="Extracted sections (§§, Anlagen, Overview)"
    )

    # Processing statistics
    processing_time_seconds: float = Field(
        0.0,
        description="Total processing time"
    )
    total_input_tokens: int = Field(
        0,
        description="Total input tokens used"
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

    # Timestamps
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When processing completed"
    )

    # --- Query methods ---

    def get_section(self, number: str) -> Optional[ExtractedSection]:
        """
        Get a section by its number.

        Args:
            number: Section identifier (e.g., "§ 10", "Anlage 2")

        Returns:
            ExtractedSection if found, None otherwise
        """
        for section in self.sections:
            if section.section_number == number:
                return section
        return None

    def get_overview(self) -> Optional[ExtractedSection]:
        """Get the overview section (Präambel/Inhaltsverzeichnis)."""
        for section in self.sections:
            if section.section_type == SectionType.OVERVIEW:
                return section
        return None

    def get_paragraphs(self) -> list[ExtractedSection]:
        """Get all § sections."""
        return [s for s in self.sections if s.section_type == SectionType.PARAGRAPH]

    def get_anlagen(self) -> list[ExtractedSection]:
        """Get all Anlage sections."""
        return [s for s in self.sections if s.section_type == SectionType.ANLAGE]

    # --- Statistics ---

    def get_stats(self) -> dict[str, Any]:
        """Get extraction statistics."""
        paragraphs = self.get_paragraphs()
        anlagen = self.get_anlagen()

        total_tokens = sum(s.token_count for s in self.sections)
        avg_tokens = total_tokens // len(self.sections) if self.sections else 0

        return {
            "total_sections": len(self.sections),
            "paragraphs": len(paragraphs),
            "anlagen": len(anlagen),
            "has_overview": self.get_overview() is not None,
            "total_tokens": total_tokens,
            "avg_tokens_per_section": avg_tokens,
            "sections_with_tables": sum(1 for s in self.sections if s.has_table),
            "sections_with_lists": sum(1 for s in self.sections if s.has_list),
            "failed_sections": sum(1 for s in self.sections if s.extraction_confidence < 1.0),
        }

    def get_full_content(self, separator: str = "\n\n") -> str:
        """Get full document content as a single string."""
        return separator.join(s.content for s in self.sections)

    # --- Export methods ---

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for serialization."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
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
