"""Report helpers for chunking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class Thresholds:
    token_recall_min: float = 0.99
    number_recall_min: float = 1.0
    duplication_ratio_max: float = 1.8
    sentence_boundary_ratio_min: float = 0.80
    max_chunk_tokens: int = 512
    min_chunk_tokens: int = 50


def summarize(metrics: dict) -> dict:
    summary = {
        "pass": metrics.get("pass", False),
        "token_recall": metrics.get("token_recall", 0.0),
        "number_recall": metrics.get("number_recall", 0.0),
        "duplication_ratio": metrics.get("duplication_ratio", 0.0),
        "sentence_boundary_ratio": metrics.get("sentence_boundary_ratio", 0.0),
        "chunks": metrics.get("chunks", 0),
        "pages_in_source": metrics.get("pages_in_source", 0),
        "pages_covered": metrics.get("pages_covered", 0),
        "too_small_chunks": metrics.get("too_small_chunks", 0),
        "too_large_chunks": metrics.get("too_large_chunks", 0),
        "metadata_errors": metrics.get("metadata_errors", 0),
    }
    return summary


def build_report(
    extraction_path: str,
    metrics: dict,
    thresholds: Thresholds,
    chunk_output_path: str | None = None,
) -> dict[str, Any]:
    return {
        "extraction_path": extraction_path,
        "chunk_output_path": chunk_output_path,
        "thresholds": thresholds.__dict__,
        "summary": summarize(metrics),
        "details": metrics,
    }


def write_report(report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.json"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"report": report_path, "summary": summary_path}
