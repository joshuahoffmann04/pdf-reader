"""Tests for vector_store.retriever — ChunkRetriever."""

import uuid

import pytest
from unittest.mock import patch

import chromadb

from chunking.models import Chunk, ChunkingConfig, ChunkingResult, ChunkingStats, ChunkMetadata
from vector_store.store import DocumentStore
from vector_store.retriever import ChunkRetriever
from vector_store.models import StoreConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 768


def _make_chunk(index: int, total: int = 10, doc_id: str = "test_doc") -> Chunk:
    prev_id = f"{doc_id}_chunk_{index - 1:04d}" if index > 0 else None
    next_id = f"{doc_id}_chunk_{index + 1:04d}" if index < total - 1 else None

    return Chunk(
        chunk_id=f"{doc_id}_chunk_{index:04d}",
        text=f"Inhalt von Chunk {index}. Dieses Segment enthält Informationen über Thema {index}.",
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


def _make_chunking_result(n: int = 10, doc_id: str = "test_doc") -> ChunkingResult:
    return ChunkingResult(
        source_file="test.pdf",
        document_id=doc_id,
        config=ChunkingConfig(),
        chunks=[_make_chunk(i, n, doc_id) for i in range(n)],
        stats=ChunkingStats(total_chunks=n),
    )


@pytest.fixture
def mock_embedder():
    with patch("vector_store.store.OllamaEmbedder") as MockEmbedder:
        instance = MockEmbedder.return_value
        instance.embed.return_value = FAKE_EMBEDDING
        instance.embed_batch.side_effect = lambda texts: [
            FAKE_EMBEDDING for _ in texts
        ]
        yield instance


@pytest.fixture
def store(mock_embedder):
    client = chromadb.EphemeralClient()
    config = StoreConfig(collection_name=f"test_{uuid.uuid4().hex[:8]}")
    s = DocumentStore(config=config, chroma_client=client)
    s.ingest(_make_chunking_result(10))
    return s


@pytest.fixture
def retriever(store):
    return ChunkRetriever(store)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_basic_retrieval(self, retriever):
        result = retriever.retrieve("Testsuche", n_results=3)
        assert result.total_results == 3
        assert len(result.results) == 3
        assert result.context_text != ""
        assert result.token_count > 0

    def test_empty_store(self, mock_embedder):
        client = chromadb.EphemeralClient()
        config = StoreConfig(collection_name=f"empty_{uuid.uuid4().hex[:8]}")
        empty_store = DocumentStore(config=config, chroma_client=client)
        retriever = ChunkRetriever(empty_store)

        result = retriever.retrieve("Test")
        assert result.total_results == 0
        assert result.context_text == ""

    def test_context_chunks_sorted(self, retriever):
        result = retriever.retrieve("Test", n_results=3)

        # Context chunks should be in document order
        indices = [c["metadata"]["chunk_index"] for c in result.context_chunks]
        assert indices == sorted(indices)

    def test_retrieve_context_string(self, retriever):
        text = retriever.retrieve_context_string("Test", n_results=2)
        assert isinstance(text, str)
        assert len(text) > 0


class TestNeighborExpansion:
    def test_neighbors_included(self, retriever):
        result = retriever.retrieve("Test", n_results=1, expand_neighbors=True)

        # With 1 result and expansion, we should get 2-3 chunks
        # (the result + its prev/next neighbors)
        assert len(result.context_chunks) >= 2

    def test_no_expansion(self, retriever):
        result = retriever.retrieve("Test", n_results=1, expand_neighbors=False)

        # Without expansion, exactly 1 chunk
        assert len(result.context_chunks) == 1

    def test_deduplication(self, retriever):
        """If two search results are neighbors, they shouldn't be duplicated."""
        result = retriever.retrieve("Test", n_results=3, expand_neighbors=True)

        chunk_ids = [c["chunk_id"] for c in result.context_chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunks found"


class TestTokenBudget:
    def test_respects_max_tokens(self, retriever):
        result = retriever.retrieve(
            "Test",
            n_results=10,
            max_context_tokens=50,
        )
        # With a tiny budget, we should get fewer chunks
        assert result.token_count <= 60  # Small margin for join overhead

    def test_at_least_one_chunk(self, retriever):
        """Even with a tiny budget, at least one chunk should be returned."""
        result = retriever.retrieve(
            "Test",
            n_results=3,
            max_context_tokens=5,  # Impossibly small
        )
        assert len(result.context_chunks) >= 1

    def test_large_budget_gets_more(self, retriever):
        small = retriever.retrieve("Test", n_results=3, max_context_tokens=50)
        large = retriever.retrieve("Test", n_results=3, max_context_tokens=10000)

        assert len(large.context_chunks) >= len(small.context_chunks)


class TestContextFormatting:
    def test_separator_between_non_consecutive(self, retriever):
        """Non-consecutive chunks should have a separator."""
        result = retriever.retrieve(
            "Test",
            n_results=2,
            max_context_tokens=10000,
            expand_neighbors=False,
        )

        # If the two results are not consecutive, there should be "---"
        if len(result.context_chunks) >= 2:
            indices = [c["metadata"]["chunk_index"] for c in result.context_chunks]
            if any(indices[i+1] - indices[i] > 1 for i in range(len(indices)-1)):
                assert "---" in result.context_text

    def test_no_separator_between_consecutive(self, retriever):
        """Consecutive chunks should NOT have a separator."""
        result = retriever.retrieve(
            "Test",
            n_results=1,
            max_context_tokens=10000,
            expand_neighbors=True,
        )

        # Neighbors are consecutive, so no separator
        context_parts = result.context_text.split("\n\n")
        assert "---" not in [p.strip() for p in context_parts if len(p.strip()) < 5]


class TestMetadataFilter:
    def test_filter_by_document(self, store, mock_embedder):
        # Add a second document
        store.ingest(_make_chunking_result(5, doc_id="other_doc"))

        retriever = ChunkRetriever(store)
        result = retriever.retrieve(
            "Test",
            n_results=5,
            where={"document_id": "test_doc"},
        )

        for chunk in result.context_chunks:
            assert chunk["metadata"]["document_id"] == "test_doc"
