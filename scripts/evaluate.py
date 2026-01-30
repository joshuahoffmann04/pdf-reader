#!/usr/bin/env python3
"""
Evaluates PDF extraction quality by comparing extracted text with original.

Usage: python -m scripts.evaluate
"""

import json
from pathlib import Path

from src.extractor import PDFExtractor
from src.evaluation import Evaluator


def load_extracted_json(json_path: str) -> dict:
    """Load extracted content from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_text_from_json(data: dict) -> str:
    """Reconstruct full text from extracted JSON structure."""
    parts = []

    if data.get("preamble"):
        parts.append(data["preamble"])

    for chapter in data.get("chapters", []):
        parts.append(f"\n{chapter['numeral']}. {chapter['title']}\n")
        for section in chapter.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])

    for appendix in data.get("appendices", []):
        parts.append(f"\n{appendix['id']}: {appendix['title']}\n")
        if appendix.get("content"):
            parts.append(appendix["content"])
        for section in appendix.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])

    return "\n".join(parts)


def main():
    """Run extraction quality evaluation."""
    print("=" * 60)
    print("PDF EXTRACTION QUALITY EVALUATION")
    print("=" * 60)

    json_path = Path("output/Pruefungsordnung_BSc_Inf_2024_extracted.json")
    pdf_path = Path("Pruefungsordnung_BSc_Inf_2024.pdf")

    # Load extracted data
    print(f"\n1. Loading: {json_path}")
    try:
        extracted_data = load_extracted_json(json_path)
    except FileNotFoundError:
        print("   ERROR: Run main.py first to generate extraction.")
        return

    # Extract original text
    print(f"2. Extracting original from: {pdf_path}")
    extractor = PDFExtractor()
    original_text = extractor.extract(pdf_path).get_full_text()

    # Reconstruct and compare
    print("3. Calculating similarity...")
    reconstructed = reconstruct_text_from_json(extracted_data)
    evaluator = Evaluator(language="de")
    result = evaluator.evaluate(reconstructed, original_text)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(result.summary())

    stats = extracted_data["statistics"]
    all_correct = (
        stats["chapters"] == 4
        and stats["main_sections"] == 40
        and stats["appendices"] == 5
        and stats["appendix_sections"] == 14
    )

    print(f"\nStructure: {'VALID' if all_correct else 'INVALID'}")
    print(f"Cosine Similarity: {result.cosine_similarity:.2%}")


if __name__ == "__main__":
    main()
