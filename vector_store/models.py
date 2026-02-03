"""
Data Models for the Vector Store Pipeline

Defines:
1. StoreConfig - Configuration for ChromaDB, Ollama, and embedding settings
2. SearchResult - A single search hit with distance/similarity
3. RetrievalResult - Complete retrieval output with context expansion
4. IngestStats - Statistics from document ingestion

Design Principles:
- Pydantic v2 for validation (consistent with pdf_extractor and chunking)
- Clean separation between search (raw hits) and retrieval (expanded context)
- Token-budget-aware context building for LLM consumption
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from chunking.models import Chunk, ChunkMetadata


class StoreConfig(BaseModel):
    """Configuration for the vector store."""
    collection_name: str = Field(
        "documents",
        description="ChromaDB collection name",
    )
    persist_directory: str = Field(
        "./chroma_db",
        description="Directory for ChromaDB persistent storage",
    )
    embedding_model: str = Field(
        "nomic-embed-text",
        description="Ollama embedding model name",
    )
    ollama_base_url: str = Field(
        "http://localhost:11434",
        description="Ollama API base URL",
    )
    distance_metric: str = Field(
        "cosine",
        description="Distance metric for ChromaDB (cosine, l2, ip)",
    )


class SearchResult(BaseModel):
    """A single search result from the vector store."""
    chunk_id: str = Field(
        ...,
        description="ID of the matching chunk",
    )
    text: str = Field(
        ...,
        description="Text content of the matching chunk",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata from ChromaDB",
    )
    distance: float = Field(
        ...,
        description="Distance score (0 = identical, higher = less similar)",
    )
    similarity: float = Field(
        ...,
        description="Similarity score (1 = identical, lower = less similar)",
    )


class RetrievalResult(BaseModel):
    """
    Complete retrieval result with neighbor-expanded context.

    This is what the chatbot receives: a ready-to-use context string
    built from the top search results plus their neighbor chunks.
    """
    query: str = Field(
        ...,
        description="The original search query",
    )
    results: list[SearchResult] = Field(
        default_factory=list,
        description="Raw search results ranked by similarity",
    )
    context_chunks: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Expanded and deduplicated chunks in document order",
    )
    context_text: str = Field(
        "",
        description="Final context string ready for LLM prompt",
    )
    token_count: int = Field(
        0,
        description="Token count of the context_text",
    )
    total_results: int = Field(
        0,
        description="Number of raw search results before expansion",
    )


class IngestStats(BaseModel):
    """Statistics from a document ingestion operation."""
    document_id: str = Field(
        ...,
        description="ID of the ingested document",
    )
    chunks_stored: int = Field(
        0,
        description="Number of chunks successfully stored",
    )
    chunks_skipped: int = Field(
        0,
        description="Number of chunks skipped (already existed)",
    )
    embedding_time_seconds: float = Field(
        0.0,
        description="Time spent generating embeddings",
    )
    total_time_seconds: float = Field(
        0.0,
        description="Total ingestion time",
    )
