"""Tests for vector_store.models — Data models."""

import pytest

from vector_store.models import (
    IngestStats,
    RetrievalResult,
    SearchResult,
    StoreConfig,
)


class TestStoreConfig:
    def test_defaults(self):
        config = StoreConfig()
        assert config.collection_name == "documents"
        assert config.persist_directory == "./chroma_db"
        assert config.embedding_model == "nomic-embed-text"
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.distance_metric == "cosine"

    def test_custom_values(self):
        config = StoreConfig(
            collection_name="my_docs",
            persist_directory="/data/chroma",
            embedding_model="mxbai-embed-large",
            ollama_base_url="http://gpu-server:11434",
            distance_metric="l2",
        )
        assert config.collection_name == "my_docs"
        assert config.embedding_model == "mxbai-embed-large"


class TestSearchResult:
    def test_creation(self):
        result = SearchResult(
            chunk_id="doc_chunk_0001",
            text="Some text",
            metadata={"document_id": "doc"},
            distance=0.2,
            similarity=0.8,
        )
        assert result.chunk_id == "doc_chunk_0001"
        assert result.similarity == 0.8

    def test_similarity_calculation(self):
        result = SearchResult(
            chunk_id="test",
            text="text",
            distance=0.3,
            similarity=0.7,
        )
        assert abs(result.distance + result.similarity - 1.0) < 0.001


class TestRetrievalResult:
    def test_empty(self):
        result = RetrievalResult(query="test")
        assert result.query == "test"
        assert result.results == []
        assert result.context_text == ""
        assert result.token_count == 0

    def test_with_data(self):
        result = RetrievalResult(
            query="Bachelorarbeit",
            results=[
                SearchResult(
                    chunk_id="c1",
                    text="text 1",
                    distance=0.1,
                    similarity=0.9,
                ),
            ],
            context_text="text 1",
            token_count=5,
            total_results=1,
        )
        assert result.total_results == 1
        assert result.context_text == "text 1"


class TestIngestStats:
    def test_creation(self):
        stats = IngestStats(
            document_id="test_doc",
            chunks_stored=57,
            embedding_time_seconds=3.5,
            total_time_seconds=4.2,
        )
        assert stats.chunks_stored == 57
        assert stats.embedding_time_seconds == 3.5
