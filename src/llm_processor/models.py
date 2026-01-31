"""
Data models for the LLM-based PDF processor.

Uses Pydantic for validation and serialization.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DocumentType(Enum):
    """Types of documents the processor can handle."""
    PRUEFUNGSORDNUNG = "pruefungsordnung"
    MODULHANDBUCH = "modulhandbuch"
    STUDIENORDNUNG = "studienordnung"
    WEBSITE = "website"
    OTHER = "other"


class ChunkType(Enum):
    """Types of content chunks."""
    SECTION = "section"           # Regular paragraph/section
    TABLE = "table"               # Table converted to text
    LIST = "list"                 # Enumerated list
    DEFINITION = "definition"     # Definition or explanation
    REFERENCE = "reference"       # Cross-reference to other document
    METADATA = "metadata"         # Document metadata


@dataclass
class DocumentContext:
    """
    Context information extracted from analyzing the entire document.
    This is used to guide page-by-page extraction.
    """
    document_type: DocumentType
    title: str
    institution: str
    version_date: Optional[str] = None

    # Structure information
    total_pages: int = 0
    chapters: list[str] = field(default_factory=list)
    main_topics: list[str] = field(default_factory=list)

    # Document-specific info
    degree_program: Optional[str] = None  # e.g., "Mathematik B.Sc."

    # Key terms and abbreviations found
    abbreviations: dict[str, str] = field(default_factory=dict)
    key_terms: list[str] = field(default_factory=list)

    # References to other documents
    referenced_documents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_type": self.document_type.value,
            "title": self.title,
            "institution": self.institution,
            "version_date": self.version_date,
            "total_pages": self.total_pages,
            "chapters": self.chapters,
            "main_topics": self.main_topics,
            "degree_program": self.degree_program,
            "abbreviations": self.abbreviations,
            "key_terms": self.key_terms,
            "referenced_documents": self.referenced_documents,
        }


@dataclass
class PageContent:
    """Content extracted from a single page."""
    page_number: int

    # Main content in natural language
    content: str

    # Structural information
    section_numbers: list[str] = field(default_factory=list)  # e.g., ["§10", "§11"]
    section_titles: list[str] = field(default_factory=list)

    # Content classification
    has_table: bool = False
    has_list: bool = False
    has_image: bool = False

    # Cross-references found on this page
    internal_references: list[str] = field(default_factory=list)  # e.g., ["§5 Abs. 2"]
    external_references: list[str] = field(default_factory=list)  # e.g., ["Allgemeine Bestimmungen"]

    # Continuation info
    continues_from_previous: bool = False
    continues_to_next: bool = False

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "content": self.content,
            "section_numbers": self.section_numbers,
            "section_titles": self.section_titles,
            "has_table": self.has_table,
            "has_list": self.has_list,
            "has_image": self.has_image,
            "internal_references": self.internal_references,
            "external_references": self.external_references,
            "continues_from_previous": self.continues_from_previous,
            "continues_to_next": self.continues_to_next,
        }


@dataclass
class RAGChunk:
    """
    A single chunk optimized for RAG retrieval.

    Design principles:
    - Self-contained: Can be understood without other chunks
    - Natural language: Optimized for embedding models
    - Rich metadata: Enables filtered retrieval
    """
    id: str

    # The actual text content (natural language)
    text: str

    # Chunk classification
    chunk_type: ChunkType

    # Source information
    source_document: str
    source_pages: list[int]

    # Structural metadata
    section_number: Optional[str] = None  # e.g., "§10"
    section_title: Optional[str] = None
    chapter: Optional[str] = None
    paragraph: Optional[str] = None  # e.g., "(3)"

    # Semantic metadata
    topics: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # Relationships
    related_sections: list[str] = field(default_factory=list)
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: list[str] = field(default_factory=list)

    # For answer generation
    is_definitive: bool = True  # False if content is ambiguous/requires context
    confidence_note: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "chunk_type": self.chunk_type.value,
            "metadata": {
                "source_document": self.source_document,
                "source_pages": self.source_pages,
                "section_number": self.section_number,
                "section_title": self.section_title,
                "chapter": self.chapter,
                "paragraph": self.paragraph,
                "topics": self.topics,
                "keywords": self.keywords,
                "related_sections": self.related_sections,
                "parent_chunk_id": self.parent_chunk_id,
                "child_chunk_ids": self.child_chunk_ids,
                "is_definitive": self.is_definitive,
                "confidence_note": self.confidence_note,
            }
        }

    def to_jsonl_entry(self) -> str:
        """Convert to JSONL format for RAG ingestion."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ProcessingConfig:
    """Configuration for the PDF processing pipeline."""

    # API settings
    api_provider: str = "anthropic"  # "anthropic" or "openai"
    model: str = "claude-sonnet-4-20250514"  # or "gpt-4o" for OpenAI

    # Processing settings
    max_tokens_per_request: int = 4096
    temperature: float = 0.0  # Deterministic output

    # Chunking settings
    target_chunk_size: int = 500  # Target tokens per chunk
    max_chunk_size: int = 1000    # Maximum tokens per chunk
    chunk_overlap: int = 50       # Overlap between chunks

    # Output settings
    output_format: str = "jsonl"  # "jsonl" or "json"
    include_raw_content: bool = False  # Include original extracted text

    # Language
    output_language: str = "de"  # Output language for natural text
