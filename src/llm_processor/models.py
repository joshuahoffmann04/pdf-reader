"""
Data Models for LLM-based PDF Processing Pipeline

This module defines the core data structures for:
1. LLM extraction output (what the Vision LLM produces)
2. RAG-optimized chunks (what gets stored in vector databases)
3. Pipeline configuration

Design Principles:
- Pydantic v2 for validation, serialization, and JSON Schema generation
- Compatible with major RAG frameworks (LangChain, LlamaIndex, Haystack)
- Clear separation between LLM output and system-generated metadata
- Optimized for both dense (embedding) and sparse (BM25) retrieval

Usage:
    # LLM fills these during extraction
    context = DocumentContext(...)
    page = ExtractedPage(...)

    # System creates these from extracted content
    chunk = RAGChunk(...)

    # Export for different frameworks
    chunk.to_langchain_document()
    chunk.to_llamaindex_node()
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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


class ChunkType(str, Enum):
    """
    Classification of chunk content types.
    Enables type-specific retrieval and generation strategies.
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
# LLM EXTRACTION MODELS (What the Vision LLM fills)
# =============================================================================

class Abbreviation(BaseModel):
    """An abbreviation and its expansion."""
    short: str = Field(..., description="The abbreviation (e.g., 'LP', 'ECTS')")
    long: str = Field(..., description="The full form (e.g., 'Leistungspunkte')")

    class Config:
        json_schema_extra = {
            "example": {"short": "LP", "long": "Leistungspunkte"}
        }


class DocumentContext(BaseModel):
    """
    Document-level context extracted by analyzing sample pages.

    This information guides page-by-page extraction and enriches chunk metadata.
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
    Structural metadata is preserved for chunking and retrieval.
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
    sections: list["SectionMarker"] = Field(
        default_factory=list,
        description="Sections that start or continue on this page"
    )

    # Content classification
    content_types: list[ChunkType] = Field(
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


# =============================================================================
# RAG CHUNK MODEL (System creates from extracted content)
# =============================================================================

class ChunkMetadata(BaseModel):
    """
    Metadata for a RAG chunk, optimized for filtering and context.

    This metadata enables:
    - Filtered retrieval (by document type, section, etc.)
    - Source attribution in generated answers
    - Context enrichment for generation
    """
    # Source attribution
    source_document: str = Field(
        ...,
        description="Source document identifier"
    )
    source_pages: list[int] = Field(
        ...,
        description="Page numbers where content was found"
    )
    document_type: DocumentType = Field(
        ...,
        description="Type of source document"
    )

    # Structural location
    section_number: Optional[str] = Field(
        None,
        description="Section number (e.g., '§10')"
    )
    section_title: Optional[str] = Field(
        None,
        description="Section title"
    )
    chapter: Optional[str] = Field(
        None,
        description="Chapter this section belongs to"
    )
    paragraph: Optional[str] = Field(
        None,
        description="Paragraph number within section (e.g., '(2)')"
    )

    # Semantic classification
    chunk_type: ChunkType = Field(
        ChunkType.SECTION,
        description="Type of content in this chunk"
    )
    topics: list[str] = Field(
        default_factory=list,
        description="Topics covered in this chunk"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords for sparse retrieval"
    )

    # Relationships
    related_sections: list[str] = Field(
        default_factory=list,
        description="Related section numbers"
    )
    parent_chunk_id: Optional[str] = Field(
        None,
        description="ID of parent chunk if this is a sub-chunk"
    )

    # Context for generation
    institution: Optional[str] = Field(
        None,
        description="Institution for attribution"
    )
    degree_program: Optional[str] = Field(
        None,
        description="Degree program if applicable"
    )
    version_date: Optional[str] = Field(
        None,
        description="Document version date"
    )

    # Quality indicators
    is_complete: bool = Field(
        True,
        description="Whether chunk contains complete information"
    )
    requires_context: bool = Field(
        False,
        description="Whether chunk needs other chunks for full understanding"
    )
    confidence: float = Field(
        1.0,
        description="Extraction confidence",
        ge=0.0,
        le=1.0
    )


class RAGChunk(BaseModel):
    """
    A single chunk optimized for RAG retrieval and generation.

    Design principles:
    - Self-contained: Can be understood without other chunks
    - Natural language: Optimized for embedding models
    - Rich metadata: Enables filtered retrieval
    - Framework agnostic: Can export to any RAG framework

    The text field contains natural language that:
    - Tables are converted to sentences
    - Lists are converted to prose
    - Context is included where needed
    - Abbreviations are expanded
    """
    # Unique identifier
    id: str = Field(
        ...,
        description="Unique chunk identifier",
        min_length=1
    )

    # Content for embedding and retrieval
    text: str = Field(
        ...,
        description="Natural language content for embedding",
        min_length=1
    )

    # Rich metadata
    metadata: ChunkMetadata = Field(
        ...,
        description="Structured metadata for filtering and context"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this chunk was created"
    )

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID contains only safe characters."""
        import re
        if not re.match(r'^[\w\-\.]+$', v):
            raise ValueError('ID must contain only alphanumeric, dash, dot, or underscore')
        return v

    # =========================================================================
    # Export methods for different RAG frameworks
    # =========================================================================

    def to_langchain_document(self) -> dict[str, Any]:
        """
        Export as LangChain Document format.

        Compatible with:
        - langchain.schema.Document
        - langchain_core.documents.Document

        Usage:
            from langchain.schema import Document
            doc = Document(**chunk.to_langchain_document())
        """
        return {
            "page_content": self.text,
            "metadata": {
                "id": self.id,
                "source": self.metadata.source_document,
                "page": self.metadata.source_pages[0] if self.metadata.source_pages else None,
                "pages": self.metadata.source_pages,
                "doc_type": self.metadata.document_type.value,
                "chunk_type": self.metadata.chunk_type.value,
                "section": self.metadata.section_number,
                "section_title": self.metadata.section_title,
                "chapter": self.metadata.chapter,
                "topics": self.metadata.topics,
                "keywords": self.metadata.keywords,
                "institution": self.metadata.institution,
                "degree_program": self.metadata.degree_program,
                "version_date": self.metadata.version_date,
                "confidence": self.metadata.confidence,
            }
        }

    def to_llamaindex_node(self) -> dict[str, Any]:
        """
        Export as LlamaIndex TextNode format.

        Compatible with:
        - llama_index.schema.TextNode
        - llama_index.core.schema.TextNode

        Usage:
            from llama_index.schema import TextNode
            node = TextNode(**chunk.to_llamaindex_node())
        """
        return {
            "id_": self.id,
            "text": self.text,
            "metadata": {
                "source_document": self.metadata.source_document,
                "source_pages": self.metadata.source_pages,
                "document_type": self.metadata.document_type.value,
                "chunk_type": self.metadata.chunk_type.value,
                "section_number": self.metadata.section_number,
                "section_title": self.metadata.section_title,
                "chapter": self.metadata.chapter,
                "paragraph": self.metadata.paragraph,
                "topics": self.metadata.topics,
                "keywords": self.metadata.keywords,
                "related_sections": self.metadata.related_sections,
                "institution": self.metadata.institution,
                "degree_program": self.metadata.degree_program,
            },
            "excluded_embed_metadata_keys": ["source_pages", "keywords"],
            "excluded_llm_metadata_keys": ["keywords", "related_sections"],
        }

    def to_haystack_document(self) -> dict[str, Any]:
        """
        Export as Haystack Document format.

        Compatible with:
        - haystack.schema.Document
        - haystack.dataclasses.Document
        """
        return {
            "id": self.id,
            "content": self.text,
            "meta": {
                "source": self.metadata.source_document,
                "pages": self.metadata.source_pages,
                "doc_type": self.metadata.document_type.value,
                "chunk_type": self.metadata.chunk_type.value,
                "section": self.metadata.section_number,
                "title": self.metadata.section_title,
                "chapter": self.metadata.chapter,
                "topics": self.metadata.topics,
                "keywords": self.metadata.keywords,
                "institution": self.metadata.institution,
            }
        }

    def to_dict(self) -> dict[str, Any]:
        """Export as generic dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.model_dump(),
            "created_at": self.created_at.isoformat(),
        }

    def to_jsonl_entry(self) -> str:
        """Export as JSONL line for file storage."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    def to_embedding_input(self) -> str:
        """
        Get text optimized for embedding models.

        Prepends key metadata for better semantic matching.
        """
        prefix_parts = []
        if self.metadata.section_number:
            prefix_parts.append(self.metadata.section_number)
        if self.metadata.section_title:
            prefix_parts.append(self.metadata.section_title)

        if prefix_parts:
            return f"{' - '.join(prefix_parts)}: {self.text}"
        return self.text


# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

class ProcessingConfig(BaseModel):
    """
    Configuration for the PDF processing pipeline.

    Controls:
    - API provider and model selection
    - Chunking strategy
    - Output format
    """
    # API Configuration
    api_provider: str = Field(
        "anthropic",
        description="API provider: 'anthropic' or 'openai'"
    )
    model: str = Field(
        "claude-sonnet-4-20250514",
        description="Model identifier"
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

    # Chunking Configuration
    target_chunk_size: int = Field(
        500,
        description="Target characters per chunk",
        ge=100,
        le=10000
    )
    max_chunk_size: int = Field(
        1000,
        description="Maximum characters per chunk",
        ge=200,
        le=20000
    )
    chunk_overlap: int = Field(
        50,
        description="Character overlap between chunks",
        ge=0,
        le=500
    )

    # Processing Options
    expand_abbreviations: bool = Field(
        True,
        description="Expand abbreviations in output text"
    )
    include_page_context: bool = Field(
        True,
        description="Include page number context in chunks"
    )
    merge_cross_page_content: bool = Field(
        True,
        description="Merge content that spans page boundaries"
    )

    # Output Configuration
    output_format: str = Field(
        "jsonl",
        description="Output format: 'jsonl', 'json', or 'parquet'"
    )
    language: Language = Field(
        Language.DE,
        description="Output language"
    )

    @model_validator(mode='after')
    def validate_chunk_sizes(self) -> 'ProcessingConfig':
        """Ensure max_chunk_size >= target_chunk_size."""
        if self.max_chunk_size < self.target_chunk_size:
            raise ValueError('max_chunk_size must be >= target_chunk_size')
        return self


# =============================================================================
# PROCESSING RESULT
# =============================================================================

class ProcessingResult(BaseModel):
    """
    Complete result of processing a PDF document.

    Contains all extracted content, generated chunks, and processing statistics.
    """
    # Document information
    source_file: str = Field(..., description="Path to source PDF")
    context: DocumentContext = Field(..., description="Document-level context")

    # Extracted content
    pages: list[ExtractedPage] = Field(
        default_factory=list,
        description="Extracted page contents"
    )

    # Generated chunks
    chunks: list[RAGChunk] = Field(
        default_factory=list,
        description="RAG-optimized chunks"
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

    def get_chunk_stats(self) -> dict[str, Any]:
        """Get statistics about generated chunks."""
        if not self.chunks:
            return {
                "total_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "by_type": {},
            }

        lengths = [len(c.text) for c in self.chunks]
        type_counts: dict[str, int] = {}
        for chunk in self.chunks:
            t = chunk.metadata.chunk_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_chunks": len(self.chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "by_type": type_counts,
        }

    def export_chunks_jsonl(self, path: str) -> None:
        """Export chunks to JSONL file."""
        from pathlib import Path
        with open(Path(path), 'w', encoding='utf-8') as f:
            for chunk in self.chunks:
                f.write(chunk.to_jsonl_entry() + '\n')

    def export_chunks_langchain(self) -> list[dict[str, Any]]:
        """Export chunks as LangChain documents."""
        return [c.to_langchain_document() for c in self.chunks]

    def export_chunks_llamaindex(self) -> list[dict[str, Any]]:
        """Export chunks as LlamaIndex nodes."""
        return [c.to_llamaindex_node() for c in self.chunks]


# Update forward references for nested models
ExtractedPage.model_rebuild()
