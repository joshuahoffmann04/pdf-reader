"""Tests for vector_store.store — DocumentStore."""

import uuid

import pytest
from unittest.mock import patch

import chromadb

from chunking.models import Chunk, ChunkingConfig, ChunkingResult, ChunkingStats, ChunkMetadata
from vector_store.store import DocumentStore
from vector_store.models import StoreConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 768


def _make_chunk(index: int, total: int = 5, doc_id: str = "test_doc") -> Chunk:
    prev_id = f"{doc_id}_chunk_{index - 1:04d}" if index > 0 else None
    next_id = f"{doc_id}_chunk_{index + 1:04d}" if index < total - 1 else None

    return Chunk(
        chunk_id=f"{doc_id}_chunk_{index:04d}",
        text=f"Dies ist der Inhalt von Chunk Nummer {index}. Er enthält wichtige Informationen.",
        token_count=20,
        metadata=ChunkMetadata(
            document_id=doc_id,
            document_title="Test Dokument",
            document_type="pruefungsordnung",
            institution="Test Universität",
            degree_program="Informatik B.Sc.",
            page_numbers=[index + 1],
            chunk_index=index,
            total_chunks=total,
            prev_chunk_id=prev_id,
            next_chunk_id=next_id,
        ),
    )


def _make_chunking_result(n_chunks: int = 5, doc_id: str = "test_doc") -> ChunkingResult:
    chunks = [_make_chunk(i, n_chunks, doc_id) for i in range(n_chunks)]
    return ChunkingResult(
        source_file="test.pdf",
        document_id=doc_id,
        config=ChunkingConfig(),
        chunks=chunks,
        stats=ChunkingStats(total_chunks=n_chunks),
    )


@pytest.fixture
def mock_embedder():
    """Patch OllamaEmbedder to avoid needing Ollama running."""
    with patch("vector_store.store.OllamaEmbedder") as MockEmbedder:
        instance = MockEmbedder.return_value
        instance.embed.return_value = FAKE_EMBEDDING
        instance.embed_batch.side_effect = lambda texts: [
            FAKE_EMBEDDING for _ in texts
        ]
        instance.health_check.return_value = {
            "healthy": True,
            "ollama_running": True,
            "model_available": True,
            "model": "nomic-embed-text",
            "error": "",
        }
        yield instance


@pytest.fixture
def store(mock_embedder):
    """Create a DocumentStore with in-memory ChromaDB and mocked embedder."""
    client = chromadb.EphemeralClient()
    config = StoreConfig(collection_name=f"test_{uuid.uuid4().hex[:8]}")
    return DocumentStore(config=config, chroma_client=client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngest:
    def test_ingest_chunks(self, store, mock_embedder):
        result = _make_chunking_result(5)
        stats = store.ingest(result)

        assert stats.chunks_stored == 5
        assert stats.document_id == "test_doc"
        assert stats.embedding_time_seconds >= 0
        assert store.count() == 5

    def test_ingest_empty(self, store):
        result = _make_chunking_result(0)
        stats = store.ingest(result)

        assert stats.chunks_stored == 0
        assert store.count() == 0

    def test_ingest_upsert(self, store, mock_embedder):
        """Re-ingesting the same document should update, not duplicate."""
        result = _make_chunking_result(3)
        store.ingest(result)
        store.ingest(result)

        assert store.count() == 3  # Not 6

    def test_ingest_multiple_documents(self, store, mock_embedder):
        result1 = _make_chunking_result(3, doc_id="doc_a")
        result2 = _make_chunking_result(4, doc_id="doc_b")
        store.ingest(result1)
        store.ingest(result2)

        assert store.count() == 7
        assert set(store.get_document_ids()) == {"doc_a", "doc_b"}

    def test_ingest_with_callback(self, store, mock_embedder):
        calls = []
        def callback(current, total, status):
            calls.append((current, total, status))

        result = _make_chunking_result(3)
        store.ingest(result, progress_callback=callback)

        assert len(calls) > 0
        assert calls[-1][2] == "Done"


class TestSearch:
    def test_search_returns_results(self, store, mock_embedder):
        result = _make_chunking_result(5)
        store.ingest(result)

        results = store.search("Testsuche", n_results=3)
        assert len(results) == 3

    def test_search_has_similarity(self, store, mock_embedder):
        result = _make_chunking_result(3)
        store.ingest(result)

        results = store.search("Test", n_results=1)
        assert len(results) == 1
        assert results[0].similarity == 1 - results[0].distance

    def test_search_with_metadata_filter(self, store, mock_embedder):
        result1 = _make_chunking_result(3, doc_id="doc_a")
        result2 = _make_chunking_result(3, doc_id="doc_b")
        store.ingest(result1)
        store.ingest(result2)

        results = store.search(
            "Test",
            n_results=10,
            where={"document_id": "doc_a"},
        )
        assert all(r.metadata["document_id"] == "doc_a" for r in results)

    def test_search_empty_collection(self, store, mock_embedder):
        results = store.search("Test", n_results=3)
        assert results == []

    def test_search_result_has_metadata(self, store, mock_embedder):
        result = _make_chunking_result(3)
        store.ingest(result)

        results = store.search("Test", n_results=1)
        meta = results[0].metadata
        assert "document_id" in meta
        assert "document_type" in meta
        assert "chunk_index" in meta
        assert "page_numbers" in meta


class TestGetChunk:
    def test_get_by_id(self, store, mock_embedder):
        result = _make_chunking_result(3)
        store.ingest(result)

        chunk = store.get_chunk_by_id("test_doc_chunk_0001")
        assert chunk is not None
        assert chunk.chunk_id == "test_doc_chunk_0001"

    def test_get_by_id_not_found(self, store, mock_embedder):
        result = _make_chunking_result(3)
        store.ingest(result)

        chunk = store.get_chunk_by_id("nonexistent_chunk")
        assert chunk is None

    def test_get_multiple_by_ids(self, store, mock_embedder):
        result = _make_chunking_result(5)
        store.ingest(result)

        chunks = store.get_chunks_by_ids([
            "test_doc_chunk_0001",
            "test_doc_chunk_0003",
        ])
        assert len(chunks) == 2


class TestDocumentManagement:
    def test_get_document_ids(self, store, mock_embedder):
        store.ingest(_make_chunking_result(3, doc_id="doc_a"))
        store.ingest(_make_chunking_result(3, doc_id="doc_b"))

        ids = store.get_document_ids()
        assert ids == ["doc_a", "doc_b"]

    def test_delete_document(self, store, mock_embedder):
        store.ingest(_make_chunking_result(3, doc_id="doc_a"))
        store.ingest(_make_chunking_result(3, doc_id="doc_b"))

        deleted = store.delete_document("doc_a")
        assert deleted == 3
        assert store.count() == 3
        assert store.get_document_ids() == ["doc_b"]

    def test_delete_nonexistent(self, store):
        deleted = store.delete_document("nonexistent")
        assert deleted == 0

    def test_count(self, store, mock_embedder):
        assert store.count() == 0
        store.ingest(_make_chunking_result(5))
        assert store.count() == 5


class TestMetadataFlattening:
    def test_page_numbers_as_string(self, store, mock_embedder):
        result = _make_chunking_result(1)
        store.ingest(result)

        chunk = store.get_chunk_by_id("test_doc_chunk_0000")
        assert isinstance(chunk.metadata["page_numbers"], str)

    def test_neighbor_pointers_stored(self, store, mock_embedder):
        result = _make_chunking_result(3)
        store.ingest(result)

        chunk = store.get_chunk_by_id("test_doc_chunk_0001")
        assert "prev_chunk_id" in chunk.metadata
        assert "next_chunk_id" in chunk.metadata

    def test_first_chunk_no_prev(self, store, mock_embedder):
        result = _make_chunking_result(3)
        store.ingest(result)

        chunk = store.get_chunk_by_id("test_doc_chunk_0000")
        assert "prev_chunk_id" not in chunk.metadata


class TestHealthCheck:
    def test_health_check(self, store, mock_embedder):
        result = store.health_check()
        assert result["chromadb_ok"] is True
        assert result["healthy"] is True
        assert "chunks_stored" in result
        assert "documents_stored" in result
