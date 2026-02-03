"""
Chunk Retriever - Intelligent retrieval with neighbor expansion

Takes raw search results from the DocumentStore and enriches them by:
1. Loading neighbor chunks (prev/next) for context continuity
2. Deduplicating overlapping chunks
3. Sorting by document order
4. Building a token-budget-aware context string for the LLM

This is the bridge between the vector store and the chatbot:
the chatbot calls retrieve_with_context() and gets a ready-to-use
context string that fits within the LLM's token budget.

Usage:
    from vector_store import DocumentStore, ChunkRetriever

    store = DocumentStore()
    retriever = ChunkRetriever(store)

    # Get context for an LLM prompt
    result = retriever.retrieve(
        query="Wie viele LP hat die Bachelorarbeit?",
        n_results=3,
        max_context_tokens=1024,
    )
    print(result.context_text)  # Ready for LLM
"""

import logging
from typing import Optional

from chunking.token_counter import count_tokens

from .models import RetrievalResult, SearchResult
from .store import DocumentStore

logger = logging.getLogger(__name__)


class ChunkRetriever:
    """
    Retrieves chunks with neighbor expansion and token-budget management.

    Given a query, the retriever:
    1. Searches the vector store for the top-N most relevant chunks
    2. Expands each result by loading its prev/next neighbor chunks
    3. Deduplicates (overlap chunks may appear in multiple results)
    4. Sorts all chunks by document order (document_id, chunk_index)
    5. Builds a context string that fits within the token budget
    """

    def __init__(self, store: DocumentStore):
        """
        Initialize the retriever.

        Args:
            store: The DocumentStore to retrieve from.
        """
        self.store = store

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        max_context_tokens: int = 1024,
        expand_neighbors: bool = True,
        where: Optional[dict] = None,
    ) -> RetrievalResult:
        """
        Search and build an expanded, token-budget-aware context.

        Args:
            query: The search query.
            n_results: Number of top results to retrieve.
            max_context_tokens: Maximum tokens for the context string.
            expand_neighbors: Whether to load prev/next neighbor chunks.
            where: Optional metadata filter for the search.

        Returns:
            RetrievalResult with the context string and metadata.
        """
        # Step 1: Search
        search_results = self.store.search(
            query=query,
            n_results=n_results,
            where=where,
        )

        if not search_results:
            return RetrievalResult(query=query)

        # Step 2: Expand with neighbor chunks
        if expand_neighbors:
            all_chunks = self._expand_neighbors(search_results)
        else:
            all_chunks = {r.chunk_id: r for r in search_results}

        # Step 3: Sort by document order
        sorted_chunks = self._sort_by_document_order(list(all_chunks.values()))

        # Step 4: Build context within token budget
        context_chunks, context_text, token_count = self._build_context(
            sorted_chunks, max_context_tokens
        )

        return RetrievalResult(
            query=query,
            results=search_results,
            context_chunks=[
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
                for c in context_chunks
            ],
            context_text=context_text,
            token_count=token_count,
            total_results=len(search_results),
        )

    def retrieve_context_string(
        self,
        query: str,
        n_results: int = 3,
        max_context_tokens: int = 1024,
        where: Optional[dict] = None,
    ) -> str:
        """
        Convenience method: retrieve and return just the context string.

        Args:
            query: The search query.
            n_results: Number of top results to retrieve.
            max_context_tokens: Maximum tokens for the context string.
            where: Optional metadata filter.

        Returns:
            Context string ready for LLM prompt, or empty string if no results.
        """
        result = self.retrieve(
            query=query,
            n_results=n_results,
            max_context_tokens=max_context_tokens,
            where=where,
        )
        return result.context_text

    def _expand_neighbors(
        self, search_results: list[SearchResult]
    ) -> dict[str, SearchResult]:
        """
        Expand search results by loading prev/next neighbor chunks.

        Collects all neighbor IDs, batch-fetches them from the store,
        and merges them with the original results (deduplicating by chunk_id).

        Returns:
            Dict mapping chunk_id → SearchResult (deduplicated).
        """
        # Start with the search results themselves
        all_chunks: dict[str, SearchResult] = {}
        for result in search_results:
            all_chunks[result.chunk_id] = result

        # Collect neighbor IDs to fetch
        neighbor_ids: set[str] = set()
        for result in search_results:
            prev_id = result.metadata.get("prev_chunk_id")
            next_id = result.metadata.get("next_chunk_id")
            if prev_id and prev_id not in all_chunks:
                neighbor_ids.add(prev_id)
            if next_id and next_id not in all_chunks:
                neighbor_ids.add(next_id)

        # Batch-fetch neighbors
        if neighbor_ids:
            neighbors = self.store.get_chunks_by_ids(list(neighbor_ids))
            for neighbor in neighbors:
                if neighbor.chunk_id not in all_chunks:
                    all_chunks[neighbor.chunk_id] = neighbor

        return all_chunks

    def _sort_by_document_order(
        self, chunks: list[SearchResult]
    ) -> list[SearchResult]:
        """
        Sort chunks by document_id and chunk_index for coherent reading order.
        """
        def sort_key(chunk: SearchResult) -> tuple:
            doc_id = chunk.metadata.get("document_id", "")
            chunk_index = chunk.metadata.get("chunk_index", 0)
            return (doc_id, chunk_index)

        return sorted(chunks, key=sort_key)

    def _build_context(
        self,
        sorted_chunks: list[SearchResult],
        max_tokens: int,
    ) -> tuple[list[SearchResult], str, int]:
        """
        Build a context string from sorted chunks within a token budget.

        Chunks are added in order until the token budget is exhausted.
        A separator line is inserted between chunks from different
        document regions (non-consecutive chunk indices).

        Returns:
            Tuple of (included_chunks, context_text, token_count).
        """
        if not sorted_chunks:
            return [], "", 0

        included: list[SearchResult] = []
        parts: list[str] = []
        total_tokens = 0
        prev_index: Optional[int] = None
        prev_doc: Optional[str] = None

        for chunk in sorted_chunks:
            chunk_tokens = count_tokens(chunk.text)

            # Check if adding this chunk would exceed the budget
            # Account for separator tokens (~3 tokens for "---\n")
            separator_cost = 3 if parts else 0
            if total_tokens + chunk_tokens + separator_cost > max_tokens:
                # If we haven't added any chunks yet, add at least one
                if not included:
                    included.append(chunk)
                    parts.append(chunk.text)
                    total_tokens = chunk_tokens
                break

            # Add separator between non-consecutive chunks
            current_doc = chunk.metadata.get("document_id", "")
            current_index = chunk.metadata.get("chunk_index", 0)

            if parts:
                if current_doc != prev_doc or (
                    prev_index is not None and current_index > prev_index + 1
                ):
                    parts.append("---")
                    total_tokens += 3

            parts.append(chunk.text)
            included.append(chunk)
            total_tokens += chunk_tokens
            prev_index = current_index
            prev_doc = current_doc

        context_text = "\n\n".join(parts)
        # Recount to be precise (joining adds characters)
        actual_tokens = count_tokens(context_text)

        return included, context_text, actual_tokens
