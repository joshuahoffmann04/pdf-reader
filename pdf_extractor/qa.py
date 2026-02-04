"""
Quality assurance helpers for the extraction pipeline.

Provides:
- One question per page (page_questions)
- Extraction check against expected snippets
- Token/number coverage validation
- Scanned page detection
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chunking.sentence_splitter import split_sentences
from .text_extractor import TextExtractor
from .table_extractor import TableExtractor


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def first_sentence(text: str) -> str:
    sentences = split_sentences(text or "")
    if sentences:
        return sentences[0].strip()
    return normalize_whitespace((text or "")[:200])


def build_question(page_number: int, analysis, table_text: str) -> tuple[str, str, str]:
    text = analysis.text or ""

    for number, title in zip(analysis.section_numbers, analysis.section_titles):
        if title:
            return (
                f"Wie lautet der Titel von {number}?",
                title.strip(),
                "section_title",
            )

    if "Regelstudienzeit" in text:
        match = re.search(
            r"Regelstudienzeit.*?(\d+\s+Semester)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return (
                "Wie viele Semester betraegt die Regelstudienzeit?",
                match.group(1).strip(),
                "regelstudienzeit",
            )

    match = re.search(r"(\d+\s+Leistungspunkte)", text)
    if match:
        return (
            f"Welche Leistungspunkte werden auf Seite {page_number} genannt?",
            match.group(1).strip(),
            "leistungspunkte",
        )

    if table_text:
        return (
            f"Nenne eine Tabellenzeile von Seite {page_number}.",
            first_sentence(table_text),
            "table_row",
        )

    return (
        f"Nenne einen Satz von Seite {page_number}.",
        first_sentence(text),
        "first_sentence",
    )


def generate_page_questions(pdf_path: str | Path, max_pages: int = 0) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_extractor = TextExtractor()
    table_extractor = TableExtractor()

    import fitz
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)

    limit = max_pages or total_pages
    questions = []
    for page_number in range(1, min(limit, total_pages) + 1):
        analysis = text_extractor.extract_page(pdf_path, page_number)
        table_text = table_extractor.extract_tables(pdf_path, page_number).text
        question, expected, source = build_question(page_number, analysis, table_text)
        questions.append(
            {
                "page_number": page_number,
                "question": question,
                "expected_snippet": expected,
                "source": source,
            }
        )

    return {
        "pdf_path": str(pdf_path),
        "page_count": total_pages,
        "questions": questions,
    }


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"(?<=\w)-\s+(?=\w)", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def check_page_questions(
    questions: dict[str, Any],
    extraction: dict[str, Any],
    use_raw: bool = False,
) -> dict[str, Any]:
    pages = {p["page_number"]: p for p in extraction.get("pages", [])}
    total = len(questions.get("questions", []))
    passed = 0
    failures: list[dict[str, Any]] = []

    for item in questions.get("questions", []):
        page_number = item["page_number"]
        expected = item["expected_snippet"]
        page = pages.get(page_number)
        if not page:
            failures.append({"page_number": page_number, "reason": "missing_page"})
            continue
        content = page.get("raw_content") if use_raw else page.get("content", "")
        ok = _normalize_for_match(expected) in _normalize_for_match(content)
        if ok:
            passed += 1
        else:
            failures.append(
                {
                    "page_number": page_number,
                    "expected_snippet": expected,
                }
            )

    accuracy = passed / total if total else 1.0
    return {
        "passed": passed,
        "total": total,
        "accuracy": round(accuracy, 4),
        "use_raw": use_raw,
        "failures": failures,
    }


def _normalize_for_coverage(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+|\d+(?:[.,]\d+)?", text, flags=re.UNICODE)


def _recall(required: set[str], observed: set[str]) -> float:
    if not required:
        return 1.0
    return len(required & observed) / max(len(required), 1)


def validate_coverage(
    pdf_path: str | Path,
    extraction: dict[str, Any],
    min_token_recall: float = 0.99,
    min_number_recall: float = 1.0,
) -> dict[str, Any]:
    pdf_path = Path(pdf_path)
    pages = {p["page_number"]: p for p in extraction.get("pages", [])}
    extractor = TextExtractor()

    import fitz
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)

    report = {
        "pdf_path": str(pdf_path),
        "pages": [],
    }

    total = 0
    passed = 0

    for page_number in range(1, total_pages + 1):
        total += 1
        page = pages.get(page_number, {})
        extracted = page.get("content", "")
        raw = page.get("raw_content", "")
        if not raw:
            raw = extractor.extract_page(pdf_path, page_number).text

        norm_raw = _normalize_for_coverage(raw)
        norm_extracted = _normalize_for_coverage(extracted)

        raw_tokens = set(_tokenize(norm_raw))
        extracted_tokens = set(_tokenize(norm_extracted))
        token_recall = _recall(raw_tokens, extracted_tokens)

        raw_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", norm_raw))
        extracted_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", norm_extracted))
        number_recall = _recall(raw_numbers, extracted_numbers)

        ok = token_recall >= min_token_recall and number_recall >= min_number_recall
        if ok:
            passed += 1

        missing = sorted(raw_tokens - extracted_tokens)
        report["pages"].append(
            {
                "page_number": page_number,
                "token_recall": round(token_recall, 4),
                "number_recall": round(number_recall, 4),
                "ok": ok,
                "missing_sample": missing[:15],
            }
        )

    accuracy = passed / max(total, 1)
    report["summary"] = {
        "total_pages": total,
        "passed_pages": passed,
        "accuracy": round(accuracy, 4),
        "min_token_recall": min_token_recall,
        "min_number_recall": min_number_recall,
    }

    return report


def detect_scanned_pages(
    pdf_path: str | Path,
    min_chars: int = 200,
    min_alpha: float = 0.20,
) -> list[int]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    extractor = TextExtractor(min_chars=min_chars, min_alpha_ratio=min_alpha)
    table_extractor = TableExtractor()

    import fitz
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)

    scanned: list[int] = []
    for page_number in range(1, total_pages + 1):
        analysis = extractor.extract_page(pdf_path, page_number)
        table_text = table_extractor.extract_tables(pdf_path, page_number).text
        if not analysis.quality_ok and not table_text:
            scanned.append(page_number)

    return scanned
