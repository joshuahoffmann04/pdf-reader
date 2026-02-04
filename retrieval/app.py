from pathlib import Path
import os

from fastapi import FastAPI

import chromadb

from .config import RetrievalConfig
from .models import IngestRequest, QueryRequest, RetrievalResponse
from .service import RetrievalService
from .vector_index import VectorIndex


def create_app(config: RetrievalConfig | None = None) -> FastAPI:
    cfg = config or RetrievalConfig()
    vector_dir = Path(cfg.data_dir) / "chroma"
    use_in_memory = os.environ.get("RETRIEVAL_CHROMA_IN_MEMORY") == "1"
    chroma_client = chromadb.Client() if use_in_memory else None
    vector_index = VectorIndex(
        persist_directory=str(vector_dir),
        collection_name=cfg.collection_name,
        chroma_client=chroma_client,
    )
    service = RetrievalService(cfg, vector_index)

    app = FastAPI(
        title="Retrieval Service",
        version="1.0.0",
        description="BM25, vector, and hybrid retrieval API.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/documents")
    def documents() -> list:
        return [doc.model_dump() for doc in service.store.list_documents()]

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict:
        return service.ingest(request)

    @app.post("/retrieve/bm25", response_model=RetrievalResponse)
    def retrieve_bm25(request: QueryRequest) -> RetrievalResponse:
        return service.retrieve_bm25(request.query, request.top_k, request.filters)

    @app.post("/retrieve/vector", response_model=RetrievalResponse)
    def retrieve_vector(request: QueryRequest) -> RetrievalResponse:
        return service.retrieve_vector(request.query, request.top_k, request.filters)

    @app.post("/retrieve/hybrid", response_model=RetrievalResponse)
    def retrieve_hybrid(request: QueryRequest) -> RetrievalResponse:
        return service.retrieve_hybrid(request.query, request.top_k, request.filters)

    return app


app = create_app()
