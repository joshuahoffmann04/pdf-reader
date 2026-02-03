"""
Document Store - ChromaDB-backed vector storage for chunks

Manages the lifecycle of document chunks in ChromaDB:
- Ingest: Embed chunks and store with metadata
- Search: Semantic similarity search with metadata filtering
- Manage: List, count, and delete documents

Design:
- Uses ChromaDB PersistentClient for on-disk storage
- Cosine distance for text similarity (configurable)
- Upsert semantics: re-ingesting a document updates existing chunks
- Metadata stored as flat key-value pairs (ChromaDB limitation)

Usage:
    from vector_store import DocumentStore

    store = DocumentStore()
    stats = store.ingest(chunking_result)
    results = store.search("Bachelorarbeit LP", n_results=3)
"""

import logging
import time
from typing import Optional

import chromadb

from chunking.models import ChunkingResult

from .embedder import OllamaEmbedder
from .models import IngestStats, SearchResult, StoreConfig

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Vector store backed by ChromaDB with Ollama embeddings.

    Stores document chunks as embeddings with rich metadata,
    enabling semantic search and metadata-filtered retrieval.
    """

    def __init__(
        self,
        config: Optional[StoreConfig] = None,
        chroma_client: Optional[chromadb.ClientAPI] = None,
    ):
        """
        Initialize the document store.

        Args:
            config: Store configuration. Uses defaults if not provided.
            chroma_client: Optional pre-created ChromaDB client (for testing).
                           If not provided, a PersistentClient is created.
        """
        self.config = config or StoreConfig()
        self._embedder = OllamaEmbedder(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )
        if chroma_client is not None:
            self._client = chroma_client
        else:
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory,
            )
        self._collection = self._client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": self.config.distance_metric},
        )

    @property
    def embedder(self) -> OllamaEmbedder:
        """Access the underlying embedder."""
        return self._embedder

    def ingest(
        self,
        chunking_result: ChunkingResult,
        progress_callback: Optional[callable] = None,
    ) -> IngestStats:
        """
        Embed and store all chunks from a ChunkingResult.

        Uses upsert semantics: if chunks with the same IDs already exist,
        they will be updated. This allows safe re-ingestion.

        Args:
            chunking_result: The chunking output to ingest.
            progress_callback: Optional callback(current, total, status).

        Returns:
            IngestStats with timing and count information.
        """
        total_start = time.time()
        chunks = chunking_result.chunks

        if not chunks:
            return IngestStats(
                document_id=chunking_result.document_id,
                total_time_seconds=time.time() - total_start,
            )

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(self._flatten_metadata(chunk))

        # Generate embeddings
        if progress_callback:
            progress_callback(0, len(chunks), "Generating embeddings...")

        embed_start = time.time()
        embeddings = self._embedder.embed_batch(documents)
        embed_time = time.time() - embed_start

        if progress_callback:
            progress_callback(len(chunks), len(chunks), "Embeddings generated")

        # Upsert into ChromaDB
        if progress_callback:
            progress_callback(0, len(chunks), "Storing in ChromaDB...")

        # ChromaDB has a batch size limit, process in batches of 500
        batch_size = 500
        stored = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            stored += end - i

            if progress_callback:
                progress_callback(stored, len(chunks), "Storing...")

        total_time = time.time() - total_start

        if progress_callback:
            progress_callback(stored, len(chunks), "Done")

        return IngestStats(
            document_id=chunking_result.document_id,
            chunks_stored=stored,
            embedding_time_seconds=round(embed_time, 2),
            total_time_seconds=round(total_time, 2),
        )

    def ingest_from_file(
        self,
        chunks_json: str,
        progress_callback: Optional[callable] = None,
    ) -> IngestStats:
        """
        Load a ChunkingResult from JSON and ingest it.

        Args:
            chunks_json: Path to the chunking result JSON file.
            progress_callback: Optional callback(current, total, status).

        Returns:
            IngestStats with timing and count information.
        """
        chunking_result = ChunkingResult.load(chunks_json)
        return self.ingest(chunking_result, progress_callback=progress_callback)

    def search(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Perform semantic similarity search.

        Args:
            query: The search query text.
            n_results: Number of results to return.
            where: Optional metadata filter dict for ChromaDB.
                   Example: {"document_type": "pruefungsordnung"}

        Returns:
            List of SearchResult objects ranked by similarity (best first).
        """
        # Embed the query
        query_embedding = self._embedder.embed(query)

        # Query ChromaDB
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        raw = self._collection.query(**query_params)

        # Parse results
        results: list[SearchResult] = []
        if not raw["ids"] or not raw["ids"][0]:
            return results

        for i, chunk_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i]
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=raw["documents"][0][i],
                metadata=raw["metadatas"][0][i],
                distance=round(distance, 6),
                similarity=round(1 - distance, 6),
            ))

        return results

    def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: The chunk ID to look up.

        Returns:
            SearchResult if found, None otherwise.
        """
        try:
            result = self._collection.get(
                ids=[chunk_id],
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                return None

            return SearchResult(
                chunk_id=result["ids"][0],
                text=result["documents"][0],
                metadata=result["metadatas"][0],
                distance=0.0,
                similarity=1.0,
            )
        except Exception:
            return None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[SearchResult]:
        """
        Retrieve multiple chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to look up.

        Returns:
            List of SearchResult objects (only found chunks).
        """
        if not chunk_ids:
            return []

        try:
            result = self._collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"],
            )

            results = []
            for i, chunk_id in enumerate(result["ids"]):
                results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=result["documents"][i],
                    metadata=result["metadatas"][i],
                    distance=0.0,
                    similarity=1.0,
                ))
            return results
        except Exception:
            return []

    def get_document_ids(self) -> list[str]:
        """
        List all unique document IDs in the store.

        Returns:
            Sorted list of document IDs.
        """
        all_metadata = self._collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in all_metadata["metadatas"]:
            if "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return sorted(doc_ids)

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: The document ID whose chunks should be removed.

        Returns:
            Number of chunks deleted.
        """
        # Get all chunk IDs for this document
        results = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        chunk_ids = results["ids"]

        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

        return len(chunk_ids)

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        return self._collection.count()

    def health_check(self) -> dict:
        """
        Check the health of the store and its dependencies.

        Returns:
            Dict with health status of ChromaDB and Ollama.
        """
        embedder_health = self._embedder.health_check()
        return {
            "chromadb_ok": True,  # If we got here, PersistentClient works
            "collection": self.config.collection_name,
            "chunks_stored": self.count(),
            "documents_stored": len(self.get_document_ids()),
            "persist_directory": self.config.persist_directory,
            **embedder_health,
        }

    def _flatten_metadata(self, chunk) -> dict:
        """
        Flatten chunk metadata for ChromaDB storage.

        ChromaDB only supports flat key-value metadata (no nested dicts
        or lists). Lists are converted to comma-separated strings.
        """
        meta = chunk.metadata
        flat = {
            "document_id": meta.document_id,
            "document_title": meta.document_title,
            "document_type": meta.document_type,
            "institution": meta.institution,
            "degree_program": meta.degree_program,
            "chunk_index": meta.chunk_index,
            "total_chunks": meta.total_chunks,
            "token_count": chunk.token_count,
            # Lists → comma-separated strings
            "page_numbers": ",".join(str(p) for p in meta.page_numbers),
        }

        # Optional neighbor pointers (ChromaDB doesn't accept None values)
        if meta.prev_chunk_id:
            flat["prev_chunk_id"] = meta.prev_chunk_id
        if meta.next_chunk_id:
            flat["next_chunk_id"] = meta.next_chunk_id

        return flat
