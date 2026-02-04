from chunking.token_counter import count_tokens

from .bm25_index import BM25Index
from .config import RetrievalConfig
from .hybrid import rrf_merge
from .models import IngestRequest, IngestResponse, RetrievalHit, RetrievalResponse
from .storage import ChunkStore
from .vector_index import VectorIndex


class RetrievalService:
    def __init__(self, config: RetrievalConfig, vector_index: VectorIndex):
        self.config = config
        self.store = ChunkStore(config.data_dir)
        self.bm25 = BM25Index(k1=config.bm25_k1, b=config.bm25_b)
        self.vector = vector_index
        self._load_indexes()

    def _load_indexes(self) -> None:
        chunks = self.store.load_chunks()
        if chunks:
            self.bm25.build(chunks)

    def ingest(self, request: IngestRequest) -> IngestResponse:
        chunks = [
            {
                "document_id": request.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in request.chunks
        ]
        self.store.save_chunks(request.document_id, request.chunks)
        all_chunks = self.store.load_chunks()
        self.bm25.build(all_chunks)
        self.vector.ingest(chunks)
        return IngestResponse(document_id=request.document_id, chunks_ingested=len(chunks))

    def retrieve_bm25(self, query: str, top_k: int, filters: dict | None) -> RetrievalResponse:
        hits = self.bm25.search(query, top_k=top_k, filters=filters)
        context = build_context(hits, self.config.max_context_tokens)
        return RetrievalResponse(query=query, mode="bm25", results=hits, context_text=context)

    def retrieve_vector(self, query: str, top_k: int, filters: dict | None) -> RetrievalResponse:
        hits = self.vector.search(query, top_k=top_k, filters=filters)
        context = build_context(hits, self.config.max_context_tokens)
        return RetrievalResponse(query=query, mode="vector", results=hits, context_text=context)

    def retrieve_hybrid(self, query: str, top_k: int, filters: dict | None) -> RetrievalResponse:
        bm25_hits = self.bm25.search(query, top_k=top_k, filters=filters)
        vector_hits = self.vector.search(query, top_k=top_k, filters=filters)
        merged_hits = rrf_merge(bm25_hits, vector_hits, top_k, self.config.rrf_k)
        context = build_context(merged_hits, self.config.max_context_tokens)
        return RetrievalResponse(query=query, mode="hybrid", results=merged_hits, context_text=context)


def build_context(hits: list[RetrievalHit], max_tokens: int) -> str:
    if not hits:
        return ""
    separator_tokens = count_tokens("---")
    parts: list[str] = []
    token_count = 0
    for hit in hits:
        chunk_tokens = count_tokens(hit.text)
        separator_cost = separator_tokens if parts else 0
        if token_count + chunk_tokens + separator_cost > max_tokens:
            if not parts:
                parts.append(hit.text)
            break
        if parts:
            parts.append("---")
            token_count += separator_tokens
        parts.append(hit.text)
        token_count += chunk_tokens
    return "\n\n".join(parts)
