"""Evaluation runner for the retrieval pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import tempfile

from chunking import ChunkingResult
from retrieval.config import RetrievalConfig
from retrieval.service import RetrievalService
from retrieval.models import ChunkInput, IngestRequest
import chromadb

from retrieval.vector_index import VectorIndex

from .metrics import hit_at_k, mrr, text_contains_hit
from .report import Thresholds, build_report, write_report


def _load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _ingest_chunks(
    chunks_path: Path,
    config: RetrievalConfig,
    vector_dir: Path,
) -> RetrievalService:
    chunk_result = ChunkingResult.load(str(chunks_path))
    chroma_client = chromadb.Client()
    vector = VectorIndex(
        persist_directory=str(vector_dir),
        collection_name=config.collection_name,
        chroma_client=chroma_client,
    )
    service = RetrievalService(config, vector)

    ingest = IngestRequest(
        document_id=chunk_result.document_id,
        chunks=[
            ChunkInput(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata.model_dump(),
            )
            for chunk in chunk_result.chunks
        ],
    )
    service.ingest(ingest)
    return service


def _run_mode(
    service: RetrievalService,
    mode: str,
    queries: list[dict],
    default_top_k: int,
    default_filters: dict | None,
    thresholds: Thresholds,
    allow_skip: bool,
) -> dict[str, Any]:
    per_query: list[dict] = []
    skipped = False
    error: str | None = None

    for q in queries:
        query = q.get("query", "").strip()
        if not query:
            continue
        top_k = int(q.get("top_k") or default_top_k)
        filters = q.get("filters", default_filters)
        expected_ids = q.get("expected_chunk_ids", [])
        expected_text = q.get("expected_text_contains", [])

        try:
            if mode == "bm25":
                response = service.retrieve_bm25(query, top_k, filters)
            elif mode == "vector":
                response = service.retrieve_vector(query, top_k, filters)
            elif mode == "hybrid":
                response = service.retrieve_hybrid(query, top_k, filters)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            ranked_ids = [hit.chunk_id for hit in response.results]
            hit = hit_at_k(ranked_ids, expected_ids)
            rr = mrr(ranked_ids, expected_ids)
            text_hit = text_contains_hit([res.text for res in response.results], expected_text)

            per_query.append({
                "id": q.get("id", ""),
                "query": query,
                "top_k": top_k,
                "filters": filters or {},
                "expected_chunk_ids": expected_ids,
                "expected_text_contains": expected_text,
                "hit_at_k": hit,
                "mrr": rr,
                "text_contains_hit": text_hit,
                "results": [
                    {
                        "chunk_id": res.chunk_id,
                        "score": res.score,
                        "text": res.text,
                        "metadata": res.metadata,
                    }
                    for res in response.results
                ],
            })
        except Exception as exc:
            if allow_skip:
                skipped = True
                error = str(exc)
                break
            raise

    report = build_report(
        chunk_path="",
        eval_path="",
        mode=mode,
        per_query=per_query,
        thresholds=thresholds,
        skipped=skipped,
        error=error,
    )
    return report


def run_evaluation(
    chunks_path: Path,
    eval_path: Path,
    output_dir: Path,
    modes: list[str],
    thresholds: Thresholds | None = None,
    allow_skip: bool = True,
) -> dict[str, Any]:
    thresholds = thresholds or Thresholds()
    eval_data = _load_eval(eval_path)

    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=output_dir))
    data_dir = run_dir / "data"
    chroma_dir = run_dir / "chroma"
    data_dir.mkdir(parents=True, exist_ok=True)

    config = RetrievalConfig(data_dir=str(data_dir))
    service = _ingest_chunks(
        chunks_path=chunks_path,
        config=config,
        vector_dir=chroma_dir,
    )

    queries = eval_data.get("queries", [])
    default_top_k = int(eval_data.get("default_top_k", 5))
    default_filters = eval_data.get("default_filters")

    results: dict[str, Any] = {}
    for mode in modes:
        report = _run_mode(
            service=service,
            mode=mode,
            queries=queries,
            default_top_k=default_top_k,
            default_filters=default_filters,
            thresholds=thresholds,
            allow_skip=allow_skip,
        )
        report["chunk_path"] = str(chunks_path)
        report["eval_path"] = str(eval_path)
        write_report(report, output_dir, mode)
        results[mode] = report

    return results
