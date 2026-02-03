"""
Document Chunker - Core chunking logic for the RAG pipeline

Takes an ExtractionResult (from pdf_extractor) and produces a ChunkingResult
with sentence-aligned, sliding-window chunks enriched with metadata.

Algorithm:
1. Merge all page contents into a continuous text stream, tracking page origins.
2. Split the merged text into sentences (regex-based, German-aware).
3. Accumulate sentences into chunks up to max_chunk_tokens.
4. Apply sliding window: the next chunk starts N sentences back to create overlap.
5. Attach metadata: document info, page numbers, neighbor pointers.

Usage:
    from pdf_extractor import ExtractionResult
    from chunking import DocumentChunker, ChunkingConfig

    result = ExtractionResult.load("output.json")
    chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
    chunks = chunker.chunk(result)
    chunks.save("chunks.json")
"""

from pathlib import Path
from typing import Optional

from pdf_extractor.models import ExtractionResult

from .models import (
    Chunk,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStats,
    ChunkMetadata,
)
from .sentence_splitter import split_sentences
from .token_counter import count_tokens


class DocumentChunker:
    """
    Splits extracted PDF content into overlapping, sentence-aligned chunks
    with rich metadata for RAG retrieval.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, extraction_result: ExtractionResult) -> ChunkingResult:
        """
        Chunk an ExtractionResult into retrieval-ready chunks.

        Args:
            extraction_result: Output from the PDF extraction pipeline.

        Returns:
            ChunkingResult with all chunks, metadata, and statistics.
        """
        document_id = self._make_document_id(extraction_result.source_file)

        # Step 1: Merge pages into continuous text with page tracking
        merged_text, char_to_page = self._merge_pages(extraction_result)

        if not merged_text.strip():
            return self._empty_result(extraction_result.source_file, document_id)

        # Step 2: Split into sentences
        sentences = split_sentences(merged_text)

        if not sentences:
            return self._empty_result(extraction_result.source_file, document_id)

        # Step 3: Map each sentence to its source page numbers
        sentence_pages = self._map_sentences_to_pages(
            sentences, merged_text, char_to_page
        )

        # Step 4: Build chunks with sliding window
        raw_chunks = self._build_chunks(sentences, sentence_pages)

        # Step 5: Create Chunk objects with metadata
        chunks = self._create_chunk_objects(
            raw_chunks, document_id, extraction_result
        )

        # Step 6: Compute statistics
        stats = self._compute_stats(chunks, sentences, extraction_result)

        return ChunkingResult(
            source_file=extraction_result.source_file,
            document_id=document_id,
            config=self.config,
            chunks=chunks,
            stats=stats,
        )

    def chunk_from_file(self, json_path: str) -> ChunkingResult:
        """
        Load an ExtractionResult from JSON and chunk it.

        Args:
            json_path: Path to the extraction result JSON file.

        Returns:
            ChunkingResult with all chunks.
        """
        extraction_result = ExtractionResult.load(json_path)
        return self.chunk(extraction_result)

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _make_document_id(self, source_file: str) -> str:
        """Generate a document ID from the source file path."""
        # Normalize Windows backslashes for cross-platform compatibility
        normalized = source_file.replace("\\", "/")
        return Path(normalized).stem

    def _merge_pages(
        self, extraction_result: ExtractionResult
    ) -> tuple[str, dict[int, int]]:
        """
        Merge all page contents into a single text string.

        Returns:
            Tuple of (merged_text, char_to_page) where char_to_page maps
            character indices to page numbers.
        """
        char_to_page: dict[int, int] = {}
        parts: list[str] = []
        current_pos = 0

        for page in extraction_result.pages:
            content = page.content.strip()
            if not content:
                continue

            # Track character positions for this page
            for i in range(len(content)):
                char_to_page[current_pos + i] = page.page_number

            parts.append(content)
            current_pos += len(content)

            # Add separator between pages (2 chars: \n\n)
            char_to_page[current_pos] = page.page_number
            char_to_page[current_pos + 1] = page.page_number
            current_pos += 2

        merged = "\n\n".join(parts)
        return merged, char_to_page

    def _map_sentences_to_pages(
        self,
        sentences: list[str],
        merged_text: str,
        char_to_page: dict[int, int],
    ) -> list[list[int]]:
        """
        Map each sentence to the page numbers it originates from.

        Returns:
            List of page number lists, one per sentence.
        """
        sentence_pages: list[list[int]] = []
        search_start = 0

        for sentence in sentences:
            idx = merged_text.find(sentence, search_start)
            if idx == -1:
                # Fallback: if exact match fails, use the previous search position
                idx = search_start

            # Collect all page numbers that this sentence spans
            pages_in_sentence: set[int] = set()
            for char_pos in range(idx, min(idx + len(sentence), len(merged_text))):
                if char_pos in char_to_page:
                    pages_in_sentence.add(char_to_page[char_pos])

            sentence_pages.append(sorted(pages_in_sentence) if pages_in_sentence else [])
            search_start = idx + len(sentence)

        return sentence_pages

    def _build_chunks(
        self,
        sentences: list[str],
        sentence_pages: list[list[int]],
    ) -> list[dict]:
        """
        Build chunks using sliding window over sentences.

        Each chunk accumulates sentences up to max_chunk_tokens.
        The next chunk starts overlap_tokens back from the end of the
        previous chunk to create sliding window overlap.

        Returns:
            List of dicts with keys: text, token_count, page_numbers, sentence_indices
        """
        chunks: list[dict] = []
        total_sentences = len(sentences)
        start_idx = 0

        while start_idx < total_sentences:
            # Accumulate sentences until we hit the token limit
            chunk_sentences: list[int] = []
            chunk_tokens = 0

            for i in range(start_idx, total_sentences):
                sentence_tokens = count_tokens(sentences[i])

                # Special case: single sentence exceeds max tokens
                if not chunk_sentences and sentence_tokens > self.config.max_chunk_tokens:
                    chunk_sentences.append(i)
                    chunk_tokens = sentence_tokens
                    break

                # Check if adding this sentence would exceed the limit
                if chunk_tokens + sentence_tokens > self.config.max_chunk_tokens:
                    break

                chunk_sentences.append(i)
                chunk_tokens += sentence_tokens

            if not chunk_sentences:
                break

            # Build chunk text and collect page numbers
            chunk_text = " ".join(sentences[idx] for idx in chunk_sentences)
            chunk_pages: set[int] = set()
            for idx in chunk_sentences:
                chunk_pages.update(sentence_pages[idx])

            chunks.append({
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
                "page_numbers": sorted(chunk_pages),
                "sentence_indices": chunk_sentences,
            })

            # Find the start index for the next chunk (sliding window).
            # Walk backwards from the end of the current chunk until
            # we've accumulated ~overlap_tokens worth of sentences.
            last_idx = chunk_sentences[-1]
            next_start = last_idx + 1  # default: no overlap

            if next_start >= total_sentences:
                break  # No more sentences to process

            overlap_accumulated = 0
            overlap_start = last_idx + 1  # will be walked backward

            for j in range(last_idx, chunk_sentences[0] - 1, -1):
                overlap_accumulated += count_tokens(sentences[j])
                if overlap_accumulated >= self.config.overlap_tokens:
                    overlap_start = j
                    break

            # The next chunk starts at overlap_start, but it must advance
            # at least one sentence beyond the current chunk's start to
            # guarantee progress.
            next_start = max(overlap_start, start_idx + 1)
            start_idx = next_start

        # Filter out micro-chunks (below min_chunk_tokens), but keep
        # the last chunk even if small (it contains the document's end).
        if len(chunks) > 1:
            filtered = []
            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                if chunk["token_count"] >= self.config.min_chunk_tokens or is_last:
                    filtered.append(chunk)
            chunks = filtered

        return chunks

    def _create_chunk_objects(
        self,
        raw_chunks: list[dict],
        document_id: str,
        extraction_result: ExtractionResult,
    ) -> list[Chunk]:
        """Create Chunk objects with full metadata and neighbor pointers."""
        total = len(raw_chunks)
        ctx = extraction_result.context

        chunks: list[Chunk] = []
        for i, raw in enumerate(raw_chunks):
            chunk_id = f"{document_id}_chunk_{i:04d}"
            prev_id = f"{document_id}_chunk_{i - 1:04d}" if i > 0 else None
            next_id = f"{document_id}_chunk_{i + 1:04d}" if i < total - 1 else None

            metadata = ChunkMetadata(
                document_id=document_id,
                document_title=ctx.title,
                document_type=ctx.document_type.value,
                institution=ctx.institution,
                degree_program=ctx.degree_program or "",
                page_numbers=raw["page_numbers"],
                chunk_index=i,
                total_chunks=total,
                prev_chunk_id=prev_id,
                next_chunk_id=next_id,
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=raw["text"],
                token_count=raw["token_count"],
                metadata=metadata,
            ))

        return chunks

    def _compute_stats(
        self,
        chunks: list[Chunk],
        sentences: list[str],
        extraction_result: ExtractionResult,
    ) -> ChunkingStats:
        """Compute statistics about the chunking result."""
        if not chunks:
            return ChunkingStats()

        token_counts = [c.token_count for c in chunks]
        return ChunkingStats(
            total_chunks=len(chunks),
            total_tokens=sum(token_counts),
            avg_chunk_tokens=sum(token_counts) / len(token_counts),
            min_chunk_tokens=min(token_counts),
            max_chunk_tokens=max(token_counts),
            total_sentences=len(sentences),
            total_pages_processed=len(extraction_result.pages),
        )

    def _empty_result(self, source_file: str, document_id: str) -> ChunkingResult:
        """Return an empty ChunkingResult for edge cases."""
        return ChunkingResult(
            source_file=source_file,
            document_id=document_id,
            config=self.config,
            chunks=[],
            stats=ChunkingStats(),
        )
