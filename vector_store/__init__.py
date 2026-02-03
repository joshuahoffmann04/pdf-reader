"""
Vector Store Module - ChromaDB + Ollama local embedding pipeline

Stores document chunks as vector embeddings in a local ChromaDB database,
enabling semantic similarity search for the RAG chatbot.

Quick Start:
    from vector_store import DocumentStore, ChunkRetriever

    # Ingest chunks
    store = DocumentStore()
    stats = store.ingest_from_file("chunks.json")

    # Search
    results = store.search("Bachelorarbeit LP", n_results=3)

    # Retrieve with neighbor expansion (for LLM context)
    retriever = ChunkRetriever(store)
    result = retriever.retrieve("Wie viele LP hat die Bachelorarbeit?")
    print(result.context_text)
"""

__version__ = "1.0.0"

from .embedder import OllamaEmbedder
from .models import IngestStats, RetrievalResult, SearchResult, StoreConfig
from .retriever import ChunkRetriever
from .store import DocumentStore

__all__ = [
    "__version__",
    "DocumentStore",
    "ChunkRetriever",
    "OllamaEmbedder",
    "StoreConfig",
    "SearchResult",
    "RetrievalResult",
    "IngestStats",
]
