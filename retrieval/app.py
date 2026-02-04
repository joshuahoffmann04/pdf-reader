from pathlib import Path
import os

from fastapi import FastAPI, HTTPException

import chromadb

from .config import RetrievalConfig
from .models import (
    DocumentSummary,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    RetrievalResponse,
)
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

    @app.get("/documents", response_model=list[DocumentSummary])
    def documents() -> list[DocumentSummary]:
        return service.store.list_documents()

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        try:
            return service.ingest(request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/retrieve/bm25", response_model=RetrievalResponse)
    def retrieve_bm25(request: QueryRequest) -> RetrievalResponse:
        try:
            return service.retrieve_bm25(request.query, request.top_k, request.filters)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/retrieve/vector", response_model=RetrievalResponse)
    def retrieve_vector(request: QueryRequest) -> RetrievalResponse:
        try:
            return service.retrieve_vector(request.query, request.top_k, request.filters)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/retrieve/hybrid", response_model=RetrievalResponse)
    def retrieve_hybrid(request: QueryRequest) -> RetrievalResponse:
        try:
            return service.retrieve_hybrid(request.query, request.top_k, request.filters)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
