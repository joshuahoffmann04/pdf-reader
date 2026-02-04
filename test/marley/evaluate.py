"""Evaluation runner for the MARley chatbot against a curated question set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import os
import time

import chromadb

from chunking import ChunkingResult
from generation.citations import normalize_and_select_citations
from generation.context_builder import build_context
from generation.json_utils import safe_parse_json
from generation.ollama_client import chat
from generation.postprocess import postprocess_answer
from generation.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from retrieval.bm25_index import BM25Index
from retrieval.embedder import OllamaEmbedder
from retrieval.hybrid import rrf_merge
from retrieval.models import RetrievalHit
from retrieval.rerank import rerank_hits
from retrieval.vector_index import VectorIndex

from .metrics import citation_page_hit, cosine_similarity, quote_support_score, score_answer
from .report import Thresholds, build_report, write_report


@dataclass
class MarleyEvalConfig:
    mode: str = "hybrid"
    top_k: int = 8
    max_context_tokens: int = 2048
    output_tokens: int = 512
    temperature: float = 0.2
    rrf_k: int = 60

    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
    embed_model: str = os.environ.get("MARLEY_EMBED_MODEL", "nomic-embed-text")


def _load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, dict)]


def _find_chunk_file_for_document(document_path: str) -> Path | None:
    doc_id = Path(document_path).stem

    # Prefer test output (commonly used by MARley).
    test_path = Path("test/chunking/output/chunks.json")
    if test_path.exists():
        try:
            chunk_result = ChunkingResult.load(str(test_path))
            if chunk_result.document_id == doc_id:
                return test_path
        except Exception:
            pass

    # Otherwise search under data/chunking/<doc>/chunks.
    candidate_dir = Path("data/chunking") / doc_id / "chunks"
    if candidate_dir.exists():
        files = sorted(candidate_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return files[0]

    return None


def _build_indexes(chunks_path: Path, config: MarleyEvalConfig) -> tuple[str, dict[str, Any], BM25Index, VectorIndex]:
    chunk_result = ChunkingResult.load(str(chunks_path))
    document_id = chunk_result.document_id

    chunks: list[dict[str, Any]] = []
    by_id: dict[str, Any] = {}
    for chunk in chunk_result.chunks:
        payload = {
            "document_id": document_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata.model_dump(),
        }
        chunks.append(payload)
        by_id[chunk.chunk_id] = payload

    bm25 = BM25Index()
    bm25.build(chunks)

    embedder = OllamaEmbedder(model=config.embed_model, base_url=config.ollama_base_url)
    chroma_client = chromadb.Client()
    vector = VectorIndex(
        persist_directory=":memory:",
        collection_name="marley_eval_chunks",
        embedder=embedder,
        chroma_client=chroma_client,
    )
    vector.ingest(chunks)
    return document_id, by_id, bm25, vector


def _retrieve(
    query: str,
    *,
    mode: str,
    top_k: int,
    document_id: str,
    bm25: BM25Index,
    vector: VectorIndex,
    rrf_k: int,
) -> list[RetrievalHit]:
    filters = {"document_id": document_id}
    if mode == "bm25":
        return bm25.search(query, top_k=top_k, filters=filters)
    if mode == "vector":
        return vector.search(query, top_k=top_k, filters=filters)
    if mode == "hybrid":
        candidate_k = min(max(top_k * 6, 50), 120)
        bm25_hits = bm25.search(query, top_k=candidate_k, filters=filters)
        vector_hits = vector.search(query, top_k=candidate_k, filters=filters)
        merge_k = min(candidate_k * 2, len(bm25_hits) + len(vector_hits))
        merged = rrf_merge(bm25_hits, vector_hits, merge_k, rrf_k)
        return rerank_hits(query, merged)[:top_k]
    raise ValueError(f"Unsupported mode: {mode}")


def run_evaluation(
    questions_path: Path,
    output_dir: Path,
    chunks_path: Path | None = None,
    config: MarleyEvalConfig | None = None,
    thresholds: Thresholds | None = None,
    allow_skip: bool = True,
) -> dict[str, Any]:
    thresholds = thresholds or Thresholds()
    config = config or MarleyEvalConfig()

    questions = _load_questions(questions_path)
    if not questions:
        report = build_report(
            questions_path=str(questions_path),
            chunk_path="",
            config=asdict(config),
            thresholds=thresholds,
            per_question=[],
            skipped=True,
            error="No questions loaded.",
        )
        write_report(report, output_dir)
        return report

    document_path = str(questions[0].get("document") or "")
    resolved_chunk_path = chunks_path
    if resolved_chunk_path is None:
        resolved_chunk_path = _find_chunk_file_for_document(document_path)
    if resolved_chunk_path is None or not resolved_chunk_path.exists():
        report = build_report(
            questions_path=str(questions_path),
            chunk_path=str(resolved_chunk_path or ""),
            config=asdict(config),
            thresholds=thresholds,
            per_question=[],
            skipped=True,
            error="Chunk file not found. Provide chunks_path or generate chunks first.",
        )
        write_report(report, output_dir)
        return report

    document_id, chunk_by_id, bm25, vector = _build_indexes(resolved_chunk_path, config)

    per_question: list[dict[str, Any]] = []
    skipped = False
    error: str | None = None

    for q in questions:
        qid = str(q.get("id") or "")
        query = str(q.get("question") or "").strip()
        expected_answer = str(q.get("expected_answer") or "").strip()
        reference_pages = q.get("reference_page_numbers") or []
        reference_quote = str(q.get("reference_quote") or "").strip()

        if not query:
            continue

        started = time.time()
        mode = (q.get("mode") or config.mode or "hybrid").strip().lower()
        top_k = int(q.get("top_k") or config.top_k)

        try:
            hits = _retrieve(
                query,
                mode=mode,
                top_k=top_k,
                document_id=document_id,
                bm25=bm25,
                vector=vector,
                rrf_k=config.rrf_k,
            )
            results = [hit.model_dump() for hit in hits]

            # Quote support checks for diagnosis.
            retrieval_quote_support = 0.0
            if reference_quote:
                retrieval_quote_support = max(
                    (quote_support_score(reference_quote, hit.text) for hit in hits),
                    default=0.0,
                )

            ctx = build_context(
                query=query,
                results=results,
                max_context_tokens=config.max_context_tokens,
                system_prompt=SYSTEM_PROMPT,
                user_template=USER_PROMPT_TEMPLATE,
            )

            selected_quote_support = 0.0
            if reference_quote:
                selected_quote_support = max(
                    (quote_support_score(reference_quote, (hit.get("text") or "")) for hit in ctx.selected_chunks),
                    default=0.0,
                )

            user_prompt = USER_PROMPT_TEMPLATE.format(query=query, context=ctx.context_text)
            raw = chat(
                base_url=config.ollama_base_url,
                model=config.ollama_model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_schema=RESPONSE_SCHEMA,
                temperature=config.temperature,
                output_tokens=config.output_tokens,
            )

            payload = safe_parse_json(raw)
            answer = postprocess_answer(str(payload.get("answer") or "").strip(), query)
            missing_info = str(payload.get("missing_info") or "").strip()
            citations = normalize_and_select_citations(
                payload.get("citations", []),
                ctx.selected_chunks,
                answer=answer,
                query=query,
            )
            if answer:
                missing_info = ""
            elif not missing_info:
                missing_info = "Information nicht im Dokument enthalten."

            # Evaluate answer/citations.
            semantic_similarity_expected = 0.0
            semantic_similarity_quote = 0.0
            semantic_similarity = 0.0
            try:
                embeddings = vector.embedder.embed_batch([expected_answer, reference_quote or "", answer])
                exp_vec, quote_vec, ans_vec = embeddings[0], embeddings[1], embeddings[2]
                if exp_vec and ans_vec:
                    semantic_similarity_expected = cosine_similarity(exp_vec, ans_vec)
                if reference_quote and quote_vec and ans_vec:
                    semantic_similarity_quote = cosine_similarity(quote_vec, ans_vec)
                semantic_similarity = max(semantic_similarity_expected, semantic_similarity_quote)
            except Exception:
                # If embeddings are unavailable, semantic similarity is not computed.
                semantic_similarity = 0.0

            answer_score = score_answer(
                expected_answer,
                answer,
                semantic_similarity=semantic_similarity,
                semantic_similarity_min=thresholds.answer_semantic_similarity_min,
                number_recall_min=thresholds.answer_number_recall_min,
            )

            cite_page = citation_page_hit(citations, list(reference_pages))

            citation_quote_support = 0.0
            if reference_quote and citations:
                citation_quote_support = max(
                    (
                        quote_support_score(
                            reference_quote,
                            (chunk_by_id.get(cite.get("chunk_id"), {}).get("text") or ""),
                        )
                        for cite in citations
                    ),
                    default=0.0,
                )

            pass_answer = answer_score.pass_
            pass_citations = cite_page >= thresholds.citation_page_hit_min and (
                not reference_quote or citation_quote_support >= thresholds.quote_support_min
            )
            passes = pass_answer and pass_citations

            per_question.append(
                {
                    "id": qid,
                    "question": query,
                    "expected_answer": expected_answer,
                    "reference_page_numbers": reference_pages,
                    "reference_quote": reference_quote,
                    "mode": mode,
                    "top_k": top_k,
                    "answer": answer,
                    "missing_info": missing_info,
                    "citations": citations,
                    "answer_token_recall": answer_score.token_recall,
                    "answer_number_recall": answer_score.number_recall,
                    "answer_semantic_similarity": answer_score.semantic_similarity,
                    "answer_semantic_similarity_expected": semantic_similarity_expected,
                    "answer_semantic_similarity_quote": semantic_similarity_quote,
                    "citation_page_hit": cite_page,
                    "retrieval_quote_support": retrieval_quote_support,
                    "selected_quote_support": selected_quote_support,
                    "citation_quote_support": citation_quote_support,
                    "latency_s": round(time.time() - started, 3),
                    "pass_answer": pass_answer,
                    "pass_citations": pass_citations,
                    "pass": passes,
                    "debug": {
                        "selected_chunks": len(ctx.selected_chunks),
                        "context_used_tokens": ctx.used_tokens,
                        "context_available_tokens": ctx.available_tokens,
                        "retrieval_top_ids": [hit.chunk_id for hit in hits],
                    },
                }
            )
        except Exception as exc:
            if allow_skip:
                skipped = True
                error = f"{qid}: {exc}"
                break
            raise

    report = build_report(
        questions_path=str(questions_path),
        chunk_path=str(resolved_chunk_path),
        config=asdict(config),
        thresholds=thresholds,
        per_question=per_question,
        skipped=skipped,
        error=error,
    )
    write_report(report, output_dir)
    return report
