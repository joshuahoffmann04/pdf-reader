"""Report helpers for the pdf_extractor evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class Thresholds:
    cer_max: float = 0.02
    wer_max: float = 0.05
    token_recall_min: float = 0.99
    number_recall_min: float = 1.0
    section_precision_min: float = 0.90
    section_recall_min: float = 0.90
    paragraph_precision_min: float = 0.90
    paragraph_recall_min: float = 0.90
    internal_ref_precision_min: float = 0.85
    internal_ref_recall_min: float = 0.85


def page_passes(page_metrics: dict, thresholds: Thresholds) -> bool:
    return (
        page_metrics["cer"] <= thresholds.cer_max
        and page_metrics["wer"] <= thresholds.wer_max
        and page_metrics["token_recall"] >= thresholds.token_recall_min
        and page_metrics["number_recall"] >= thresholds.number_recall_min
        and page_metrics["sections"]["precision"] >= thresholds.section_precision_min
        and page_metrics["sections"]["recall"] >= thresholds.section_recall_min
        and page_metrics["paragraphs"]["precision"] >= thresholds.paragraph_precision_min
        and page_metrics["paragraphs"]["recall"] >= thresholds.paragraph_recall_min
        and page_metrics["internal_references"]["precision"] >= thresholds.internal_ref_precision_min
        and page_metrics["internal_references"]["recall"] >= thresholds.internal_ref_recall_min
    )


def summarize(pages: list[dict]) -> dict:
    if not pages:
        return {
            "pages": 0,
            "pass": True,
            "cer_avg": 0.0,
            "wer_avg": 0.0,
            "token_recall_avg": 1.0,
            "number_recall_avg": 1.0,
        }

    def avg(key: str) -> float:
        return sum(p[key] for p in pages) / max(len(pages), 1)

    summary = {
        "pages": len(pages),
        "pass": all(p["pass"] for p in pages),
        "passed_pages": sum(1 for p in pages if p["pass"]),
        "failed_pages": sum(1 for p in pages if not p["pass"]),
        "cer_avg": avg("cer"),
        "wer_avg": avg("wer"),
        "token_recall_avg": avg("token_recall"),
        "number_recall_avg": avg("number_recall"),
    }

    return summary


def build_report(
    pdf_path: str,
    reference_path: str,
    page_results: list[dict],
    thresholds: Thresholds,
) -> dict[str, Any]:
    return {
        "pdf_path": pdf_path,
        "reference_path": reference_path,
        "thresholds": thresholds.__dict__,
        "summary": summarize(page_results),
        "pages": page_results,
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
