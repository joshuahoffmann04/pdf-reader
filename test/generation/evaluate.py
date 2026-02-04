"""Evaluation runner for generation output."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from generation import GenerationConfig, GenerationService
from generation.models import GenerateRequest

from .metrics import (
    contains_all,
    citations_include_chunk_ids,
    citations_include_pages,
    missing_info_ok,
)
from .report import Thresholds, build_report, write_report


def _load_eval(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_evaluation(
    eval_path: Path,
    output_dir: Path,
    thresholds: Thresholds | None = None,
    allow_skip: bool = True,
    config: GenerationConfig | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or Thresholds()
    eval_data = _load_eval(eval_path)
    queries = eval_data.get("queries", [])
    default_mode = eval_data.get("default_mode", "hybrid")

    service = GenerationService(config=config)

    per_query: list[dict] = []
    skipped = False
    error: str | None = None

    for q in queries:
        query = (q.get("query") or "").strip()
        if not query:
            continue
        mode = q.get("mode") or default_mode
        expected_answer = q.get("expected_answer_contains", [])
        expected_missing = q.get("expected_missing_info")
        expected_chunk_ids = q.get("expected_citation_chunk_ids", [])
        expected_pages = q.get("expected_page_numbers", [])
        min_citations = int(q.get("min_citations") or 0)

        try:
            request = GenerateRequest(query=query, mode=mode)
            response = service.generate(request)
        except Exception as exc:
            if allow_skip:
                skipped = True
                error = str(exc)
                break
            raise

        citations = [c.model_dump() for c in response.citations]
        answer_hit = contains_all(response.answer, expected_answer)
        missing_hit = missing_info_ok(response.missing_info, expected_missing)
        chunk_id_hit = citations_include_chunk_ids(citations, expected_chunk_ids)
        page_hit = citations_include_pages(citations, expected_pages)
        citation_hit = 1.0 if (chunk_id_hit == 1.0 and page_hit == 1.0) else 0.0
        if min_citations:
            citation_hit = 1.0 if (citation_hit == 1.0 and len(citations) >= min_citations) else 0.0

        per_query.append({
            "id": q.get("id", ""),
            "query": query,
            "mode": mode,
            "expected_answer_contains": expected_answer,
            "expected_missing_info": expected_missing,
            "expected_citation_chunk_ids": expected_chunk_ids,
            "expected_page_numbers": expected_pages,
            "min_citations": min_citations,
            "answer": response.answer,
            "missing_info": response.missing_info,
            "citations": citations,
            "answer_contains_hit": answer_hit,
            "citation_hit": citation_hit,
            "missing_info_hit": missing_hit,
        })

    report = build_report(
        eval_path=str(eval_path),
        per_query=per_query,
        thresholds=thresholds,
        skipped=skipped,
        error=error,
    )
    write_report(report, output_dir)
    return report
