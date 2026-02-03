"""
Data Models for the Chunking Pipeline

Defines:
1. ChunkingConfig - Configuration for chunk size, overlap, etc.
2. ChunkMetadata - Rich metadata linking chunks to their source
3. Chunk - A single text chunk with metadata
4. ChunkingResult - Complete chunking output with statistics

Design Principles:
- Pydantic v2 for validation and serialization (consistent with pdf_extractor)
- Rich metadata for downstream retrieval filtering
- Prev/next pointers for neighbor-chunk navigation
- Save/load pattern matching ExtractionResult

Usage:
    config = ChunkingConfig(max_chunk_tokens=512)
    result = chunker.chunk(extraction_result)
    result.save("chunks.json")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """
    Configuration for the chunking pipeline.

    Controls chunk size limits, overlap behavior, and token counting.
    Defaults are optimized for a local LLM with 2048 token context window.
    """
    max_chunk_tokens: int = Field(
        512,
        description="Maximum tokens per chunk",
        ge=50,
        le=4096,
    )
    overlap_tokens: int = Field(
        100,
        description="Target overlap in tokens between consecutive chunks (sliding window)",
        ge=0,
    )
    min_chunk_tokens: int = Field(
        50,
        description="Minimum tokens for a chunk (avoids micro-chunks)",
        ge=1,
    )

    def model_post_init(self, __context: Any) -> None:
        if self.overlap_tokens >= self.max_chunk_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than "
                f"max_chunk_tokens ({self.max_chunk_tokens})"
            )


class ChunkMetadata(BaseModel):
    """
    Rich metadata attached to each chunk for retrieval filtering
    and context reconstruction.
    """
    # Document-level (from DocumentContext)
    document_id: str = Field(
        ...,
        description="Unique identifier for the source document",
    )
    document_title: str = Field(
        ...,
        description="Title of the source document",
    )
    document_type: str = Field(
        ...,
        description="Type of document (e.g., 'pruefungsordnung')",
    )
    institution: str = Field(
        "",
        description="Issuing institution",
    )
    degree_program: str = Field(
        "",
        description="Degree program if applicable",
    )

    # Position within document
    page_numbers: list[int] = Field(
        default_factory=list,
        description="Source page numbers this chunk originates from",
    )
    chunk_index: int = Field(
        ...,
        description="Position of this chunk within the document (0-indexed)",
    )
    total_chunks: int = Field(
        ...,
        description="Total number of chunks in the document",
    )

    # Neighbor pointers for context navigation
    prev_chunk_id: Optional[str] = Field(
        None,
        description="ID of the previous chunk (for context expansion)",
    )
    next_chunk_id: Optional[str] = Field(
        None,
        description="ID of the next chunk (for context expansion)",
    )


class Chunk(BaseModel):
    """
    A single text chunk with metadata, ready for embedding and storage.
    """
    chunk_id: str = Field(
        ...,
        description="Unique identifier (format: {document_id}_chunk_{index:04d})",
    )
    text: str = Field(
        ...,
        description="The chunk text content",
        min_length=1,
    )
    token_count: int = Field(
        ...,
        description="Number of tokens in this chunk",
        ge=1,
    )
    metadata: ChunkMetadata = Field(
        ...,
        description="Rich metadata for retrieval and navigation",
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ChunkingStats(BaseModel):
    """Statistics about the chunking process."""
    total_chunks: int = 0
    total_tokens: int = 0
    avg_chunk_tokens: float = 0.0
    min_chunk_tokens: int = 0
    max_chunk_tokens: int = 0
    total_sentences: int = 0
    total_pages_processed: int = 0


class ChunkingResult(BaseModel):
    """
    Complete result of chunking a document.

    Contains all chunks with metadata and processing statistics.
    Ready for downstream embedding and vector store ingestion.
    """
    source_file: str = Field(
        ...,
        description="Path to the source extraction JSON",
    )
    document_id: str = Field(
        ...,
        description="Unique document identifier",
    )
    config: ChunkingConfig = Field(
        ...,
        description="Configuration used for chunking",
    )
    chunks: list[Chunk] = Field(
        default_factory=list,
        description="All chunks with metadata",
    )
    stats: ChunkingStats = Field(
        default_factory=ChunkingStats,
        description="Chunking statistics",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When chunking was performed",
    )

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Find a chunk by its ID."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    def get_neighbors(self, chunk_id: str) -> tuple[Optional[Chunk], Optional[Chunk]]:
        """Get the previous and next chunks for context expansion."""
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return None, None
        prev_chunk = (
            self.get_chunk_by_id(chunk.metadata.prev_chunk_id)
            if chunk.metadata.prev_chunk_id
            else None
        )
        next_chunk = (
            self.get_chunk_by_id(chunk.metadata.next_chunk_id)
            if chunk.metadata.next_chunk_id
            else None
        )
        return prev_chunk, next_chunk

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str) -> None:
        """Save chunking result to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "ChunkingResult":
        """Load chunking result from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
