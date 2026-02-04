"""Report helpers for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class Thresholds:
    hit_rate_min: float = 0.80
    mrr_min: float = 0.50
    text_hit_rate_min: float = 0.80


def summarize(per_query: list[dict]) -> dict:
    total = len(per_query)
    if total == 0:
        return {
            "queries": 0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "text_hit_rate": 0.0,
            "pass": False,
        }

    hit_rate = sum(q["hit_at_k"] for q in per_query) / total
    mrr = sum(q["mrr"] for q in per_query) / total
    text_hit_rate = sum(q["text_contains_hit"] for q in per_query) / total

    return {
        "queries": total,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "text_hit_rate": text_hit_rate,
    }


def build_report(
    chunk_path: str,
    eval_path: str,
    mode: str,
    per_query: list[dict],
    thresholds: Thresholds,
    skipped: bool,
    error: str | None,
) -> dict[str, Any]:
    summary = summarize(per_query)
    summary["pass"] = (
        not skipped
        and summary["hit_rate"] >= thresholds.hit_rate_min
        and summary["mrr"] >= thresholds.mrr_min
        and summary["text_hit_rate"] >= thresholds.text_hit_rate_min
    )

    return {
        "chunk_path": chunk_path,
        "eval_path": eval_path,
        "mode": mode,
        "thresholds": thresholds.__dict__,
        "skipped": skipped,
        "error": error or "",
        "summary": summary,
        "queries": per_query,
    }


def write_report(report: dict[str, Any], output_dir: Path, mode: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{mode}.json"
    summary_path = output_dir / f"summary_{mode}.json"

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"report": report_path, "summary": summary_path}
