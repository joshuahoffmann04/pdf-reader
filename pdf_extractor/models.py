"""
Data Models for PDF Extraction Pipeline

This module defines the core data structures for:
1. Document context (metadata extracted from the document)
2. Page-level extraction (content from each PDF page)
3. Processing configuration

Design Principles:
- Pydantic v2 for validation, serialization, and JSON Schema generation
- Clear separation between document context and page content
- Optimized for downstream processing (chunking, RAG, etc.)

Usage:
    # Context analysis phase
    context = DocumentContext(...)

    # Page extraction phase
    page = ExtractedPage(...)

    # Export for downstream processing
    result.to_dict()
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class DocumentType(str, Enum):
    """
    Classification of document types.
    Used for retrieval filtering and context-aware generation.
    """
    PRUEFUNGSORDNUNG = "pruefungsordnung"      # Examination regulations
    MODULHANDBUCH = "modulhandbuch"            # Module handbook
    STUDIENORDNUNG = "studienordnung"          # Study regulations
    ALLGEMEINE_BESTIMMUNGEN = "allgemeine_bestimmungen"  # General provisions
    PRAKTIKUMSORDNUNG = "praktikumsordnung"   # Internship regulations
    ZULASSUNGSORDNUNG = "zulassungsordnung"   # Admission regulations
    SATZUNG = "satzung"                        # Statutes
    WEBSITE = "website"                        # Web content
    FAQ = "faq"                                # FAQ document
    OTHER = "other"                            # Other document types


class ContentType(str, Enum):
    """
    Classification of content types found on pages.
    Used for downstream processing decisions.
    """
    # Structural types
    SECTION = "section"              # Regular paragraph (§)
    SUBSECTION = "subsection"        # Subsection (Absatz)
    ARTICLE = "article"              # Article in statutes

    # Content types
    DEFINITION = "definition"        # Term definition
    REGULATION = "regulation"        # Specific rule/requirement
    PROCEDURE = "procedure"          # Process description
    DEADLINE = "deadline"            # Deadline/timeline information
    REQUIREMENT = "requirement"      # Admission/graduation requirements

    # Structured content (converted to natural language)
    TABLE = "table"                  # Table content
    LIST = "list"                    # Enumerated/bulleted list
    GRADE_SCALE = "grade_scale"      # Grading information

    # Meta types
    METADATA = "metadata"            # Document metadata
    REFERENCE = "reference"          # Cross-reference information
    OVERVIEW = "overview"            # Summary/overview content


class Language(str, Enum):
    """Supported languages."""
    DE = "de"  # German
    EN = "en"  # English


# =============================================================================
# EXTRACTION MODELS
# =============================================================================

class Abbreviation(BaseModel):
    """An abbreviation and its expansion."""
    short: str = Field(..., description="The abbreviation (e.g., 'LP', 'ECTS')")
    long: str = Field(..., description="The full form (e.g., 'Leistungspunkte')")

    class Config:
        json_schema_extra = {
            "example": {"short": "LP", "long": "Leistungspunkte"}
        }


class SectionMarker(BaseModel):
    """
    A section or paragraph marker found in the document.

    German legal documents use specific numbering:
    - § for sections (Paragraphen)
    - (1), (2) for subsections (Absätze)
    - 1., 2. for items within subsections
    """
    number: str = Field(
        ...,
        description="Section identifier (e.g., '§10', '§10 Abs. 2')"
    )
    title: Optional[str] = Field(
        None,
        description="Section title if present"
    )
    level: int = Field(
        1,
        description="Nesting level (1=§, 2=Absatz, 3=item)",
        ge=1,
        le=5
    )
    starts_on_page: bool = Field(
        True,
        description="Whether this section starts on this page"
    )


class DocumentContext(BaseModel):
    """
    Document-level context extracted by analyzing sample pages.

    This information guides page-by-page extraction and enriches metadata.
    The Vision LLM fills this during the context analysis phase.
    """
    # Required identification
    document_type: DocumentType = Field(
        ...,
        description="Type of document for retrieval filtering"
    )
    title: str = Field(
        ...,
        description="Official document title",
        min_length=1
    )
    institution: str = Field(
        ...,
        description="Issuing institution (e.g., 'Philipps-Universität Marburg')"
    )

    # Version information
    version_date: Optional[str] = Field(
        None,
        description="Publication or effective date (ISO format preferred)"
    )
    version_info: Optional[str] = Field(
        None,
        description="Version description (e.g., 'Nichtamtliche Lesefassung')"
    )

    # Scope
    degree_program: Optional[str] = Field(
        None,
        description="Degree program if applicable (e.g., 'Mathematik B.Sc.')"
    )
    faculty: Optional[str] = Field(
        None,
        description="Faculty or department"
    )

    # Structure
    total_pages: int = Field(
        0,
        description="Total number of pages",
        ge=0
    )
    chapters: list[str] = Field(
        default_factory=list,
        description="Main chapters/sections (e.g., ['I. Allgemeines', 'II. Prüfungen'])"
    )
    main_topics: list[str] = Field(
        default_factory=list,
        description="Key topics covered in the document"
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
        description="Other documents referenced (e.g., 'Allgemeine Bestimmungen')"
    )
    legal_basis: Optional[str] = Field(
        None,
        description="Legal basis for the document (e.g., 'HHG §44')"
    )

    # Metadata
    language: Language = Field(
        Language.DE,
        description="Primary language of the document"
    )

    def get_abbreviation_dict(self) -> dict[str, str]:
        """Get abbreviations as a simple dictionary."""
        return {a.short: a.long for a in self.abbreviations}

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for downstream processing."""
        return self.model_dump()

    class Config:
        json_schema_extra = {
            "example": {
                "document_type": "pruefungsordnung",
                "title": "Prüfungsordnung für den Studiengang Mathematik B.Sc.",
                "institution": "Philipps-Universität Marburg",
                "version_date": "2023-10-15",
                "degree_program": "Mathematik B.Sc.",
                "total_pages": 42,
                "chapters": ["I. Allgemeines", "II. Studienbezogene Bestimmungen"],
                "abbreviations": [{"short": "LP", "long": "Leistungspunkte"}],
                "key_terms": ["Modul", "Regelstudienzeit", "Prüfungsleistung"]
            }
        }


class ExtractedPage(BaseModel):
    """
    Content extracted from a single PDF page by the Vision LLM.

    The LLM converts all content to natural language, including tables and lists.
    Structural metadata is preserved for downstream processing.
    """
    page_number: int = Field(
        ...,
        description="Page number (1-indexed)",
        ge=1
    )

    # Main content (natural language)
    content: str = Field(
        ...,
        description="Page content converted to natural, flowing text"
    )

    # Structural markers found on this page
    sections: list[SectionMarker] = Field(
        default_factory=list,
        description="Sections that start or continue on this page"
    )

    # Paragraph numbers found on this page
    paragraph_numbers: list[str] = Field(
        default_factory=list,
        description="Paragraph numbers like (1), (2), (3) found on this page"
    )

    # Content classification
    content_types: list[ContentType] = Field(
        default_factory=list,
        description="Types of content found on this page"
    )
    has_table: bool = Field(
        False,
        description="Page contains tabular data (converted to text)"
    )
    has_list: bool = Field(
        False,
        description="Page contains lists (converted to text)"
    )
    has_figure: bool = Field(
        False,
        description="Page contains figures/images"
    )

    # Cross-references
    internal_references: list[str] = Field(
        default_factory=list,
        description="References to other sections (e.g., '§5 Abs. 2')"
    )
    external_references: list[str] = Field(
        default_factory=list,
        description="References to external documents"
    )

    # Continuation markers
    continues_from_previous: bool = Field(
        False,
        description="Content continues from previous page (mid-sentence/paragraph)"
    )
    continues_to_next: bool = Field(
        False,
        description="Content continues to next page"
    )

    # Quality indicators
    extraction_confidence: float = Field(
        1.0,
        description="LLM confidence in extraction quality (0-1)",
        ge=0.0,
        le=1.0
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Notes about extraction issues or ambiguities"
    )

    def to_dict(self) -> dict[str, Any]:
        """Export as dictionary for downstream processing."""
        return self.model_dump()


# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

class ProcessingConfig(BaseModel):
    """
    Configuration for the PDF extraction pipeline.

    Controls:
    - Model selection (OpenAI GPT-4o variants)
    - Processing behavior
    - Output options
    """
    # API Configuration (OpenAI only)
    model: str = Field(
        "gpt-4o",
        description="OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)"
    )
    max_tokens_per_request: int = Field(
        4096,
        description="Maximum tokens per API request",
        ge=100,
        le=100000
    )
    temperature: float = Field(
        0.0,
        description="Sampling temperature (0 for deterministic)",
        ge=0.0,
        le=2.0
    )

    # Processing Options
    expand_abbreviations: bool = Field(
        True,
        description="Expand abbreviations in output text"
    )
    include_page_context: bool = Field(
        True,
        description="Include page number context in output"
    )
    merge_cross_page_content: bool = Field(
        True,
        description="Merge content that spans page boundaries"
    )

    # Retry Configuration
    max_retries: int = Field(
        3,
        description="Maximum retry attempts for failed pages",
        ge=1,
        le=10
    )

    # Output Configuration
    output_format: str = Field(
        "json",
        description="Output format: 'json' or 'jsonl'"
    )
    language: Language = Field(
        Language.DE,
        description="Output language"
    )


# =============================================================================
# EXTRACTION RESULT
# =============================================================================

class ExtractionResult(BaseModel):
    """
    Complete result of extracting content from a PDF document.

    Contains all extracted content and processing statistics.
    Ready for downstream processing (chunking, indexing, etc.).
    """
    # Document information
    source_file: str = Field(..., description="Path to source PDF")
    context: DocumentContext = Field(..., description="Document-level context")

    # Extracted content
    pages: list[ExtractedPage] = Field(
        default_factory=list,
        description="Extracted page contents"
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
    estimated_cost_usd: float = Field(
        0.0,
        description="Estimated API cost"
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

    def get_page_stats(self) -> dict[str, Any]:
        """Get statistics about extracted pages."""
        if not self.pages:
            return {
                "total_pages": 0,
                "pages_with_tables": 0,
                "pages_with_lists": 0,
                "avg_content_length": 0,
                "failed_pages": 0,
            }

        lengths = [len(p.content) for p in self.pages]
        failed = [p for p in self.pages if p.extraction_confidence < 1.0]

        return {
            "total_pages": len(self.pages),
            "pages_with_tables": sum(1 for p in self.pages if p.has_table),
            "pages_with_lists": sum(1 for p in self.pages if p.has_list),
            "avg_content_length": sum(lengths) / len(lengths),
            "min_content_length": min(lengths),
            "max_content_length": max(lengths),
            "failed_pages": len(failed),
        }

    def get_all_sections(self) -> list[SectionMarker]:
        """Get all section markers from all pages."""
        sections = []
        for page in self.pages:
            sections.extend(page.sections)
        return sections

    def get_full_content(self, separator: str = "\n\n") -> str:
        """Get full document content as a single string."""
        return separator.join(p.content for p in self.pages)

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


class ExtractRequest(BaseModel):
    pdf_path: str = Field(..., min_length=1)


class ExtractResponse(BaseModel):
    document_id: str
    output_path: str
    pages: int


# Update forward references for nested models
ExtractedPage.model_rebuild()
