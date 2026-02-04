"""
Retrieval component for RAG pipelines.

Provides BM25, vector, and hybrid retrieval with a unified API.
"""

__version__ = "1.0.0"

from .config import RetrievalConfig
from .service import RetrievalService
from .models import (
    ChunkInput,
    IngestRequest,
    QueryRequest,
    RetrievalHit,
    RetrievalResponse,
    DocumentSummary,
)
from .bm25_index import BM25Index
from .vector_index import VectorIndex
from .embedder import OllamaEmbedder
from .hybrid import rrf_merge

__all__ = [
    "__version__",
    "RetrievalConfig",
    "RetrievalService",
    "ChunkInput",
    "IngestRequest",
    "QueryRequest",
    "RetrievalHit",
    "RetrievalResponse",
    "DocumentSummary",
    "BM25Index",
    "VectorIndex",
    "OllamaEmbedder",
    "rrf_merge",
]
