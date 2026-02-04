"""Evaluation runner for the pdf_extractor pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from pdf_extractor import PDFExtractor, ProcessingConfig

from .metrics import cer, wer, token_recall, number_recall, precision_recall_f1, diff_sets
from .report import Thresholds, build_report, write_report, page_passes


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_page_index(pages: list[dict]) -> dict[int, dict]:
    index: dict[int, dict] = {}
    for page in pages:
        page_number = page.get("page_number")
        if isinstance(page_number, int):
            index[page_number] = page
    return index


def _get_list(page: dict, key: str) -> list[str]:
    value = page.get(key, [])
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def _get_text(page: dict) -> str:
    return str(page.get("content") or "")


def _evaluate_page(reference: dict, extracted: dict) -> dict:
    ref_text = _get_text(reference)
    ext_text = _get_text(extracted)

    sections = precision_recall_f1(
        _get_list(reference, "section_numbers"),
        _get_list(extracted, "section_numbers"),
    )
    paragraphs = precision_recall_f1(
        _get_list(reference, "paragraph_numbers"),
        _get_list(extracted, "paragraph_numbers"),
    )
    internal_refs = precision_recall_f1(
        _get_list(reference, "internal_references"),
        _get_list(extracted, "internal_references"),
    )

    return {
        "page_number": reference.get("page_number"),
        "cer": cer(ref_text, ext_text),
        "wer": wer(ref_text, ext_text),
        "token_recall": token_recall(ref_text, ext_text),
        "number_recall": number_recall(ref_text, ext_text),
        "sections": sections,
        "paragraphs": paragraphs,
        "internal_references": internal_refs,
        "diff": {
            "sections": diff_sets(
                _get_list(reference, "section_numbers"),
                _get_list(extracted, "section_numbers"),
            ),
            "paragraphs": diff_sets(
                _get_list(reference, "paragraph_numbers"),
                _get_list(extracted, "paragraph_numbers"),
            ),
            "internal_references": diff_sets(
                _get_list(reference, "internal_references"),
                _get_list(extracted, "internal_references"),
            ),
        },
        "reference_length": len(ref_text),
        "extracted_length": len(ext_text),
    }


def run_evaluation(
    pdf_path: Path,
    reference_path: Path,
    output_dir: Path,
    config: ProcessingConfig | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    config = config or ProcessingConfig()
    thresholds = thresholds or Thresholds()

    extractor = PDFExtractor(config=config)
    extraction = extractor.extract(str(pdf_path))
    extracted = extraction.to_dict()

    reference = _load_json(reference_path)
    ref_pages = _build_page_index(reference.get("pages", []))
    ext_pages = _build_page_index(extracted.get("pages", []))

    page_results: list[dict] = []
    for page_number, ref_page in sorted(ref_pages.items()):
        ext_page = ext_pages.get(page_number, {})
        page_metrics = _evaluate_page(ref_page, ext_page)
        page_metrics["pass"] = page_passes(page_metrics, thresholds)
        page_results.append(page_metrics)

    report = build_report(
        pdf_path=str(pdf_path),
        reference_path=str(reference_path),
        page_results=page_results,
        thresholds=thresholds,
    )
    report["meta"] = {
        "reference_pages": len(ref_pages),
        "extracted_pages": len(ext_pages),
        "evaluated_pages": len(page_results),
    }
    write_report(report, output_dir)
    return report
