from pathlib import Path

import chromadb

from retrieval.config import RetrievalConfig
from retrieval.models import ChunkInput, IngestRequest
from retrieval.service import RetrievalService
from retrieval.vector_index import DummyEmbedder, VectorIndex


def build_service(tmp_path: Path) -> RetrievalService:
    config = RetrievalConfig(data_dir=str(tmp_path))
    vector_index = VectorIndex(
        persist_directory=str(tmp_path / "chroma"),
        collection_name=config.collection_name,
        embedder=DummyEmbedder(),
        chroma_client=chromadb.Client(),
    )
    return RetrievalService(config, vector_index)


def test_bm25_retrieval_returns_expected_chunk(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    request = IngestRequest(
        document_id="doc1",
        chunks=[
            ChunkInput(chunk_id="c1", text="Die Bachelorarbeit umfasst 12 LP.", metadata={"page": 1}),
            ChunkInput(chunk_id="c2", text="Ein anderes Kapitel über Module.", metadata={"page": 2}),
        ],
    )
    service.ingest(request)
    response = service.retrieve_bm25("Bachelorarbeit LP", top_k=1, filters=None)
    assert response.results
    assert response.results[0].chunk_id == "c1"
    assert "Bachelorarbeit" in response.context_text


def test_hybrid_retrieval_merges_sources(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    request = IngestRequest(
        document_id="doc1",
        chunks=[
            ChunkInput(chunk_id="c1", text="Bachelorarbeit mit 12 LP.", metadata={"page": 1}),
            ChunkInput(chunk_id="c2", text="Prüfungsleistung im Modul.", metadata={"page": 2}),
        ],
    )
    service.ingest(request)
    response = service.retrieve_hybrid("Bachelorarbeit", top_k=2, filters=None)
    assert response.results
    assert response.mode == "hybrid"
