"""Tests for chunking.models."""

import json
import tempfile
from pathlib import Path

import pytest

from chunking.models import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStats,
    ChunkMetadata,
)


class TestChunkingConfig:
    def test_defaults(self):
        config = ChunkingConfig()
        assert config.max_chunk_tokens == 512
        assert config.overlap_tokens == 100
        assert config.min_chunk_tokens == 50

    def test_custom_values(self):
        config = ChunkingConfig(
            max_chunk_tokens=256,
            overlap_tokens=50,
            min_chunk_tokens=20,
        )
        assert config.max_chunk_tokens == 256
        assert config.overlap_tokens == 50

    def test_overlap_must_be_less_than_max(self):
        with pytest.raises(ValueError, match="overlap_tokens"):
            ChunkingConfig(max_chunk_tokens=100, overlap_tokens=100)

    def test_overlap_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="overlap_tokens"):
            ChunkingConfig(max_chunk_tokens=100, overlap_tokens=150)

    def test_min_chunk_tokens_validation(self):
        with pytest.raises(Exception):
            ChunkingConfig(min_chunk_tokens=0)

    def test_max_chunk_tokens_too_small(self):
        with pytest.raises(Exception):
            ChunkingConfig(max_chunk_tokens=10)


class TestChunkMetadata:
    def test_creation(self):
        meta = ChunkMetadata(
            document_id="doc001",
            document_title="Test Document",
            document_type="pruefungsordnung",
            chunk_index=0,
            total_chunks=10,
        )
        assert meta.document_id == "doc001"
        assert meta.prev_chunk_id is None
        assert meta.next_chunk_id is None

    def test_with_neighbors(self):
        meta = ChunkMetadata(
            document_id="doc001",
            document_title="Test",
            document_type="pruefungsordnung",
            chunk_index=5,
            total_chunks=10,
            prev_chunk_id="doc001_chunk_0004",
            next_chunk_id="doc001_chunk_0006",
        )
        assert meta.prev_chunk_id == "doc001_chunk_0004"
        assert meta.next_chunk_id == "doc001_chunk_0006"

    def test_page_numbers(self):
        meta = ChunkMetadata(
            document_id="doc001",
            document_title="Test",
            document_type="pruefungsordnung",
            page_numbers=[3, 4, 5],
            chunk_index=0,
            total_chunks=1,
        )
        assert meta.page_numbers == [3, 4, 5]


class TestChunk:
    def test_creation(self):
        chunk = Chunk(
            chunk_id="doc001_chunk_0000",
            text="Dies ist der Chunk-Text.",
            token_count=7,
            metadata=ChunkMetadata(
                document_id="doc001",
                document_title="Test",
                document_type="other",
                chunk_index=0,
                total_chunks=1,
            ),
        )
        assert chunk.chunk_id == "doc001_chunk_0000"
        assert chunk.token_count == 7

    def test_to_dict(self):
        chunk = Chunk(
            chunk_id="doc001_chunk_0000",
            text="Test text.",
            token_count=3,
            metadata=ChunkMetadata(
                document_id="doc001",
                document_title="Test",
                document_type="other",
                chunk_index=0,
                total_chunks=1,
            ),
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "doc001_chunk_0000"
        assert d["text"] == "Test text."
        assert d["metadata"]["document_id"] == "doc001"

    def test_text_min_length(self):
        with pytest.raises(Exception):
            Chunk(
                chunk_id="doc001_chunk_0000",
                text="",
                token_count=0,
                metadata=ChunkMetadata(
                    document_id="doc001",
                    document_title="Test",
                    document_type="other",
                    chunk_index=0,
                    total_chunks=1,
                ),
            )


class TestChunkingResult:
    def _make_result(self) -> ChunkingResult:
        chunks = []
        for i in range(3):
            chunks.append(Chunk(
                chunk_id=f"doc001_chunk_{i:04d}",
                text=f"Chunk {i} text content.",
                token_count=5,
                metadata=ChunkMetadata(
                    document_id="doc001",
                    document_title="Test Document",
                    document_type="pruefungsordnung",
                    chunk_index=i,
                    total_chunks=3,
                    prev_chunk_id=f"doc001_chunk_{i - 1:04d}" if i > 0 else None,
                    next_chunk_id=f"doc001_chunk_{i + 1:04d}" if i < 2 else None,
                ),
            ))
        return ChunkingResult(
            source_file="test.json",
            document_id="doc001",
            config=ChunkingConfig(),
            chunks=chunks,
            stats=ChunkingStats(total_chunks=3, total_tokens=15),
        )

    def test_total_chunks_property(self):
        result = self._make_result()
        assert result.total_chunks == 3

    def test_get_chunk_by_id(self):
        result = self._make_result()
        chunk = result.get_chunk_by_id("doc001_chunk_0001")
        assert chunk is not None
        assert chunk.chunk_id == "doc001_chunk_0001"

    def test_get_chunk_by_id_not_found(self):
        result = self._make_result()
        assert result.get_chunk_by_id("nonexistent") is None

    def test_get_neighbors(self):
        result = self._make_result()
        prev_chunk, next_chunk = result.get_neighbors("doc001_chunk_0001")
        assert prev_chunk is not None
        assert prev_chunk.chunk_id == "doc001_chunk_0000"
        assert next_chunk is not None
        assert next_chunk.chunk_id == "doc001_chunk_0002"

    def test_get_neighbors_first_chunk(self):
        result = self._make_result()
        prev_chunk, next_chunk = result.get_neighbors("doc001_chunk_0000")
        assert prev_chunk is None
        assert next_chunk is not None

    def test_get_neighbors_last_chunk(self):
        result = self._make_result()
        prev_chunk, next_chunk = result.get_neighbors("doc001_chunk_0002")
        assert prev_chunk is not None
        assert next_chunk is None

    def test_save_and_load(self):
        result = self._make_result()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.save(path)
            loaded = ChunkingResult.load(path)
            assert loaded.document_id == "doc001"
            assert loaded.total_chunks == 3
            assert loaded.chunks[0].chunk_id == "doc001_chunk_0000"
            assert loaded.chunks[1].metadata.prev_chunk_id == "doc001_chunk_0000"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_to_json(self):
        result = self._make_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["document_id"] == "doc001"
        assert len(parsed["chunks"]) == 3
