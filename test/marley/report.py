"""Report building for MARley evaluation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


@dataclass
class Thresholds:
    answer_semantic_similarity_min: float = 0.80
    answer_number_recall_min: float = 1.0
    citation_page_hit_min: float = 1.0
    quote_support_min: float = 0.80


def build_summary(per_question: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_question)
    passed_strict = sum(1 for q in per_question if q.get("pass"))
    passed_answer = sum(1 for q in per_question if q.get("pass_answer"))
    passed_citations = sum(1 for q in per_question if q.get("pass_citations"))
    if total == 0:
        return {"questions": 0, "pass": True}

    def _avg(key: str) -> float:
        values = [float(q.get(key) or 0.0) for q in per_question]
        return sum(values) / max(len(values), 1)

    return {
        "questions": total,
        "pass": passed_strict == total,
        "passed_strict": passed_strict,
        "failed_strict": total - passed_strict,
        "passed_answer": passed_answer,
        "failed_answer": total - passed_answer,
        "passed_citations": passed_citations,
        "failed_citations": total - passed_citations,
        "answer_semantic_similarity_avg": _avg("answer_semantic_similarity"),
        "answer_token_recall_avg": _avg("answer_token_recall"),
        "answer_number_recall_avg": _avg("answer_number_recall"),
        "citation_page_hit_rate": _avg("citation_page_hit"),
        "retrieval_quote_support_avg": _avg("retrieval_quote_support"),
        "selected_quote_support_avg": _avg("selected_quote_support"),
        "citation_quote_support_avg": _avg("citation_quote_support"),
    }


def build_report(
    questions_path: str,
    chunk_path: str,
    config: dict[str, Any],
    thresholds: Thresholds,
    per_question: list[dict[str, Any]],
    skipped: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "questions_path": questions_path,
        "chunk_path": chunk_path,
        "config": config,
        "thresholds": asdict(thresholds),
        "skipped": skipped,
        "error": error or "",
        "summary": build_summary(per_question),
        "questions": per_question,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
