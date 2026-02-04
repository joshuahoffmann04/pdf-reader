"""Report helpers for generation evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class Thresholds:
    answer_hit_rate_min: float = 0.80
    citation_hit_rate_min: float = 0.70
    missing_info_hit_rate_min: float = 0.80


def summarize(per_query: list[dict]) -> dict:
    total = len(per_query)
    if total == 0:
        return {
            "queries": 0,
            "answer_hit_rate": 0.0,
            "citation_hit_rate": 0.0,
            "missing_info_hit_rate": 0.0,
            "pass": False,
        }

    answer_hit_rate = sum(q["answer_contains_hit"] for q in per_query) / total
    citation_hit_rate = sum(q["citation_hit"] for q in per_query) / total
    missing_info_hit_rate = sum(q["missing_info_hit"] for q in per_query) / total

    return {
        "queries": total,
        "answer_hit_rate": answer_hit_rate,
        "citation_hit_rate": citation_hit_rate,
        "missing_info_hit_rate": missing_info_hit_rate,
    }


def build_report(
    eval_path: str,
    per_query: list[dict],
    thresholds: Thresholds,
    skipped: bool,
    error: str | None,
) -> dict[str, Any]:
    summary = summarize(per_query)
    summary["pass"] = (
        not skipped
        and summary["answer_hit_rate"] >= thresholds.answer_hit_rate_min
        and summary["citation_hit_rate"] >= thresholds.citation_hit_rate_min
        and summary["missing_info_hit_rate"] >= thresholds.missing_info_hit_rate_min
    )

    return {
        "eval_path": eval_path,
        "thresholds": thresholds.__dict__,
        "skipped": skipped,
        "error": error or "",
        "summary": summary,
        "queries": per_query,
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
