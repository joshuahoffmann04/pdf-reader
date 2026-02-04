"""
Validation helpers to ensure LLM output does not drop critical facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    ok: bool
    numeric_recall: float
    section_recall: float
    char_recall: float
    reason: str = ""


SECTION_SIGN = "\u00a7"


def _tokenize_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)?", text)


def _tokenize_sections(text: str) -> list[str]:
    return re.findall(rf"{SECTION_SIGN}\s*\d+[a-zA-Z]?", text)


def validate_llm_output(
    raw_text: str,
    llm_text: str,
    min_numeric_recall: float = 0.98,
    min_section_recall: float = 0.90,
    min_char_recall: float = 0.90,
) -> ValidationResult:
    if not raw_text:
        return ValidationResult(ok=True, numeric_recall=1.0, section_recall=1.0, char_recall=1.0)

    if not llm_text:
        return ValidationResult(ok=False, numeric_recall=0.0, section_recall=0.0, char_recall=0.0, reason="empty_llm")

    raw_numbers = set(_tokenize_numbers(raw_text))
    llm_numbers = set(_tokenize_numbers(llm_text))
    numeric_recall = _recall(raw_numbers, llm_numbers)

    raw_sections = set(_tokenize_sections(raw_text))
    llm_sections = set(_tokenize_sections(llm_text))
    section_recall = _recall(raw_sections, llm_sections)

    char_recall = _char_recall(raw_text, llm_text)

    ok = (
        numeric_recall >= min_numeric_recall
        and section_recall >= min_section_recall
        and char_recall >= min_char_recall
    )

    reason = ""
    if not ok:
        reason = (
            f"numeric_recall={numeric_recall:.2f}, "
            f"section_recall={section_recall:.2f}, "
            f"char_recall={char_recall:.2f}"
        )

    return ValidationResult(
        ok=ok,
        numeric_recall=numeric_recall,
        section_recall=section_recall,
        char_recall=char_recall,
        reason=reason,
    )


def validate_text_coverage(
    raw_text: str,
    extracted_text: str,
    min_token_recall: float = 0.99,
    min_number_recall: float = 1.0,
) -> ValidationResult:
    if not raw_text:
        return ValidationResult(ok=True, numeric_recall=1.0, section_recall=1.0, char_recall=1.0)

    if not extracted_text:
        return ValidationResult(ok=False, numeric_recall=0.0, section_recall=0.0, char_recall=0.0, reason="empty_extracted")

    token_recall = _token_recall(raw_text, extracted_text)
    number_recall = _number_recall(raw_text, extracted_text)
    char_recall = _char_recall(raw_text, extracted_text)

    ok = token_recall >= min_token_recall and number_recall >= min_number_recall
    reason = ""
    if not ok:
        reason = f"token_recall={token_recall:.2f}, number_recall={number_recall:.2f}"

    return ValidationResult(
        ok=ok,
        numeric_recall=number_recall,
        section_recall=token_recall,
        char_recall=char_recall,
        reason=reason,
    )


def _recall(required: set[str], observed: set[str]) -> float:
    if not required:
        return 1.0
    return len(required & observed) / max(len(required), 1)


def _char_recall(raw_text: str, llm_text: str) -> float:
    raw_chars = {c for c in raw_text if c.isalnum()}
    llm_chars = {c for c in llm_text if c.isalnum()}
    if not raw_chars:
        return 1.0
    return len(raw_chars & llm_chars) / max(len(raw_chars), 1)


def _token_recall(raw_text: str, extracted_text: str) -> float:
    pattern = r"[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]+|\d+(?:[.,]\d+)?"
    tokens = set(re.findall(pattern, raw_text))
    extracted = set(re.findall(pattern, extracted_text))
    if not tokens:
        return 1.0
    return len(tokens & extracted) / max(len(tokens), 1)

def _number_recall(raw_text: str, extracted_text: str) -> float:
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", raw_text))
    extracted = set(re.findall(r"\d+(?:[.,]\d+)?", extracted_text))
    if not numbers:
        return 1.0
    return len(numbers & extracted) / max(len(numbers), 1)
