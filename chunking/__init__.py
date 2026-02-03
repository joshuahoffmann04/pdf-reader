"""
Chunking Module - Sentence-aligned sliding window chunking for RAG

Splits extracted PDF content into overlapping chunks with rich metadata,
optimized for retrieval with a local LLM (2048 token context window).

Quick Start:
    from pdf_extractor import ExtractionResult
    from chunking import DocumentChunker, ChunkingConfig

    result = ExtractionResult.load("output.json")
    chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
    chunks = chunker.chunk(result)
    chunks.save("chunks.json")
"""

__version__ = "1.0.0"

from .chunker import DocumentChunker
from .service import ChunkingService
from .config import ChunkingServiceConfig
from .models import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStats,
    ChunkMetadata,
)
from .sentence_splitter import split_sentences
from .token_counter import count_tokens

__all__ = [
    "__version__",
    "DocumentChunker",
    "ChunkingService",
    "ChunkingServiceConfig",
    "Chunk",
    "ChunkingConfig",
    "ChunkingResult",
    "ChunkingStats",
    "ChunkMetadata",
    "split_sentences",
    "count_tokens",
]
