"""Tests for chunking.chunker — Integration tests."""

import pytest

from pdf_extractor.models import (
    DocumentContext,
    DocumentType,
    ExtractedPage,
    ExtractionResult,
    SectionMarker,
)
from chunking import DocumentChunker, ChunkingConfig
from chunking.token_counter import count_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> DocumentContext:
    defaults = dict(
        document_type=DocumentType.PRUEFUNGSORDNUNG,
        title="Prüfungsordnung Informatik B.Sc.",
        institution="Philipps-Universität Marburg",
        degree_program="Informatik B.Sc.",
        total_pages=10,
    )
    defaults.update(overrides)
    return DocumentContext(**defaults)


def _make_page(number: int, content: str, **overrides) -> ExtractedPage:
    defaults = dict(page_number=number, content=content)
    defaults.update(overrides)
    return ExtractedPage(**defaults)


def _make_result(pages: list[ExtractedPage], **overrides) -> ExtractionResult:
    defaults = dict(
        source_file="test_document.pdf",
        context=_make_context(),
        pages=pages,
    )
    defaults.update(overrides)
    return ExtractionResult(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicChunking:
    def test_single_short_page(self):
        """A single short page should produce exactly one chunk."""
        pages = [_make_page(1, "Dies ist ein kurzer Text.")]
        result = _make_result(pages)
        chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
        chunks = chunker.chunk(result)

        assert chunks.total_chunks == 1
        assert "Dies ist ein kurzer Text." in chunks.chunks[0].text

    def test_empty_pages(self):
        """Empty pages should produce no chunks."""
        pages = [_make_page(1, ""), _make_page(2, "  ")]
        result = _make_result(pages)
        chunker = DocumentChunker()
        chunks = chunker.chunk(result)

        assert chunks.total_chunks == 0

    def test_no_pages(self):
        """No pages at all should produce no chunks."""
        result = _make_result([])
        chunker = DocumentChunker()
        chunks = chunker.chunk(result)

        assert chunks.total_chunks == 0

    def test_multiple_pages_merged(self):
        """Content from multiple pages should be merged."""
        pages = [
            _make_page(1, "Erster Absatz auf Seite eins."),
            _make_page(2, "Zweiter Absatz auf Seite zwei."),
        ]
        result = _make_result(pages)
        chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
        chunks = chunker.chunk(result)

        # With enough token budget, both pages fit in one chunk
        assert chunks.total_chunks >= 1
        full_text = " ".join(c.text for c in chunks.chunks)
        assert "Erster Absatz" in full_text
        assert "Zweiter Absatz" in full_text


class TestTokenLimits:
    def test_no_chunk_exceeds_max_tokens(self):
        """No chunk should exceed max_chunk_tokens (except single long sentences)."""
        # Generate text with many sentences
        sentences = [f"Dies ist Satz Nummer {i} im Testdokument." for i in range(50)]
        text = " ".join(sentences)
        pages = [_make_page(1, text)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=100, overlap_tokens=20, min_chunk_tokens=10)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        assert chunks.total_chunks > 1
        for chunk in chunks.chunks:
            assert chunk.token_count <= config.max_chunk_tokens + 10  # small margin for join

    def test_long_single_sentence_allowed(self):
        """A single sentence longer than max_tokens should still become a chunk."""
        long_sentence = "Die " + "sehr " * 200 + "lange Prüfungsordnung."
        pages = [_make_page(1, long_sentence)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=100, overlap_tokens=20)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        # Should still produce at least one chunk
        assert chunks.total_chunks >= 1


class TestSlidingWindowOverlap:
    def test_overlap_exists(self):
        """Consecutive chunks should share overlapping text."""
        sentences = [f"Testtext Nummer {i} enthält Information." for i in range(30)]
        text = " ".join(sentences)
        pages = [_make_page(1, text)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=100, overlap_tokens=30, min_chunk_tokens=10)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        assert chunks.total_chunks >= 3

        # Check that consecutive chunks share some text
        for i in range(len(chunks.chunks) - 1):
            words_current = set(chunks.chunks[i].text.split())
            words_next = set(chunks.chunks[i + 1].text.split())
            overlap = words_current & words_next
            assert len(overlap) > 0, f"No overlap between chunk {i} and {i+1}"

    def test_progress_guaranteed(self):
        """Chunker must always make forward progress (no infinite loops)."""
        sentences = [f"Satz {i}." for i in range(100)]
        text = " ".join(sentences)
        pages = [_make_page(1, text)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=50, overlap_tokens=30, min_chunk_tokens=5)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        # Should complete and produce chunks (not hang)
        assert chunks.total_chunks > 0
        assert chunks.total_chunks < 200  # Reasonable upper bound


class TestMetadata:
    def test_document_metadata(self):
        """Chunks should carry document-level metadata."""
        pages = [_make_page(1, "Ein Testtext für Metadaten.")]
        result = _make_result(pages)
        chunker = DocumentChunker()
        chunks = chunker.chunk(result)

        meta = chunks.chunks[0].metadata
        assert meta.document_title == "Prüfungsordnung Informatik B.Sc."
        assert meta.document_type == "pruefungsordnung"
        assert meta.institution == "Philipps-Universität Marburg"
        assert meta.degree_program == "Informatik B.Sc."

    def test_page_numbers_tracked(self):
        """Chunks should know which pages they come from."""
        pages = [
            _make_page(1, "Inhalt auf Seite eins."),
            _make_page(2, "Inhalt auf Seite zwei."),
        ]
        result = _make_result(pages)
        chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
        chunks = chunker.chunk(result)

        # With enough budget, all content fits in one chunk spanning both pages
        all_pages = set()
        for c in chunks.chunks:
            all_pages.update(c.metadata.page_numbers)
        assert 1 in all_pages
        assert 2 in all_pages

    def test_neighbor_pointers(self):
        """Chunks should have correct prev/next pointers."""
        sentences = [f"Satz {i} im Dokument." for i in range(30)]
        text = " ".join(sentences)
        pages = [_make_page(1, text)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=80, overlap_tokens=20, min_chunk_tokens=10)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        assert chunks.total_chunks >= 3

        # First chunk has no prev
        assert chunks.chunks[0].metadata.prev_chunk_id is None
        assert chunks.chunks[0].metadata.next_chunk_id is not None

        # Last chunk has no next
        assert chunks.chunks[-1].metadata.next_chunk_id is None
        assert chunks.chunks[-1].metadata.prev_chunk_id is not None

        # Middle chunks have both
        for i in range(1, len(chunks.chunks) - 1):
            assert chunks.chunks[i].metadata.prev_chunk_id is not None
            assert chunks.chunks[i].metadata.next_chunk_id is not None

    def test_chunk_index_and_total(self):
        """Each chunk should know its position and the total count."""
        sentences = [f"Satz {i} im Dokument." for i in range(30)]
        text = " ".join(sentences)
        pages = [_make_page(1, text)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=80, overlap_tokens=20, min_chunk_tokens=10)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        total = chunks.total_chunks
        for i, chunk in enumerate(chunks.chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.total_chunks == total

    def test_chunk_id_format(self):
        """Chunk IDs should follow the format doc_id_chunk_XXXX."""
        pages = [_make_page(1, "Testtext für die ID.")]
        result = _make_result(pages)
        chunker = DocumentChunker()
        chunks = chunker.chunk(result)

        assert chunks.chunks[0].chunk_id == "test_document_chunk_0000"


class TestDocumentId:
    def test_document_id_from_filename(self):
        """Document ID should be derived from the source filename."""
        pages = [_make_page(1, "Test.")]
        result = _make_result(pages, source_file="path/to/my_document.pdf")
        chunker = DocumentChunker()
        chunks = chunker.chunk(result)

        assert chunks.document_id == "my_document"


class TestStats:
    def test_stats_populated(self):
        """ChunkingResult should contain valid statistics."""
        sentences = [f"Satz {i} im Dokument." for i in range(20)]
        text = " ".join(sentences)
        pages = [_make_page(1, text)]
        result = _make_result(pages)

        config = ChunkingConfig(max_chunk_tokens=100, overlap_tokens=20, min_chunk_tokens=10)
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(result)

        assert chunks.stats.total_chunks == chunks.total_chunks
        assert chunks.stats.total_tokens > 0
        assert chunks.stats.avg_chunk_tokens > 0
        assert chunks.stats.min_chunk_tokens > 0
        assert chunks.stats.max_chunk_tokens > 0
        assert chunks.stats.total_sentences == 20
        assert chunks.stats.total_pages_processed == 1


class TestChunkFromFile:
    def test_chunk_from_file(self, tmp_path):
        """chunk_from_file should load JSON and chunk it."""
        pages = [_make_page(1, "Ein Test zum Laden aus Datei.")]
        result = _make_result(pages)
        json_path = tmp_path / "test.json"
        result.save(str(json_path))

        chunker = DocumentChunker()
        chunks = chunker.chunk_from_file(str(json_path))

        assert chunks.total_chunks == 1
        assert "Test zum Laden" in chunks.chunks[0].text
