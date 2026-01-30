#!/usr/bin/env python3
"""
Evaluation script for PDF extraction quality.

Compares extracted text with original PDF text using various similarity metrics.
"""

import json
import sys
sys.path.insert(0, '.')

from src.extractor.pdf_extractor import PDFExtractor
from src.parser.document_parser import DocumentParser
from src.evaluation.evaluator import Evaluator


def load_extracted_json(json_path: str) -> dict:
    """Load extracted content from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconstruct_text_from_json(data: dict) -> str:
    """Reconstruct full text from extracted JSON structure."""
    parts = []

    # Preamble
    if data.get("preamble"):
        parts.append(data["preamble"])

    # Chapters and sections
    for chapter in data.get("chapters", []):
        parts.append(f"\n{chapter['numeral']}. {chapter['title']}\n")

        for section in chapter.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])

    # Appendices
    for appendix in data.get("appendices", []):
        parts.append(f"\n{appendix['id']}: {appendix['title']}\n")
        if appendix.get("content"):
            parts.append(appendix["content"])

        for section in appendix.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])

    return "\n".join(parts)


def evaluate_section_by_section(data: dict, original_text: str, evaluator: Evaluator) -> dict:
    """Evaluate extraction quality section by section."""
    results = {
        "chapter_results": [],
        "appendix_results": [],
        "overall": {}
    }

    # Evaluate each chapter
    for chapter in data.get("chapters", []):
        chapter_result = {
            "id": chapter["id"],
            "title": chapter["title"],
            "sections": []
        }

        for section in chapter.get("sections", []):
            # Try to find this section's content in original
            section_marker = f"§ {section['number']}"

            # Simple heuristic: section content should appear in original
            content_in_original = section["content"][:100] in original_text if section["content"] else False

            chapter_result["sections"].append({
                "id": section["id"],
                "title": section["title"],
                "content_length": len(section["content"]),
                "found_in_original": content_in_original
            })

        results["chapter_results"].append(chapter_result)

    # Evaluate appendices
    for appendix in data.get("appendices", []):
        appendix_result = {
            "id": appendix["id"],
            "section_count": len(appendix.get("sections", [])),
            "has_content": bool(appendix.get("content"))
        }
        results["appendix_results"].append(appendix_result)

    return results


def main():
    """Run the evaluation."""
    print("="*60)
    print("PDF EXTRACTION QUALITY EVALUATION")
    print("="*60)

    # Step 1: Load extracted JSON
    json_path = "output/Pruefungsordnung_BSc_Inf_2024_extracted.json"
    print(f"\n1. Loading extracted data from: {json_path}")

    try:
        extracted_data = load_extracted_json(json_path)
        print(f"   Loaded successfully.")
    except FileNotFoundError:
        print(f"   ERROR: File not found. Run main.py first.")
        return

    # Step 2: Extract original text from PDF
    pdf_path = "Pruefungsordnung_BSc_Inf_2024.pdf"
    print(f"\n2. Extracting original text from: {pdf_path}")

    extractor = PDFExtractor()
    pdf_doc = extractor.extract(pdf_path)
    original_text = pdf_doc.get_full_text()
    print(f"   Original text length: {len(original_text):,} characters")

    # Step 3: Reconstruct text from extracted structure
    print("\n3. Reconstructing text from extracted structure...")
    reconstructed_text = reconstruct_text_from_json(extracted_data)
    print(f"   Reconstructed text length: {len(reconstructed_text):,} characters")

    # Step 4: Calculate similarity metrics
    print("\n4. Calculating similarity metrics...")
    evaluator = Evaluator(language="de")
    result = evaluator.evaluate(reconstructed_text, original_text)

    print("\n" + "="*60)
    print("SIMILARITY RESULTS")
    print("="*60)
    print(result.summary())

    # Step 5: Section-by-section analysis
    print("\n" + "="*60)
    print("SECTION-BY-SECTION ANALYSIS")
    print("="*60)

    section_results = evaluate_section_by_section(extracted_data, original_text, evaluator)

    for chapter_result in section_results["chapter_results"]:
        total = len(chapter_result["sections"])
        found = sum(1 for s in chapter_result["sections"] if s["found_in_original"])
        print(f"\n{chapter_result['id']}. {chapter_result['title']}")
        print(f"   Sections: {total}, Content found in original: {found}/{total}")

        # Show any missing sections
        missing = [s for s in chapter_result["sections"] if not s["found_in_original"]]
        if missing:
            print(f"   Missing: {[s['id'] for s in missing]}")

    print("\n" + "="*60)
    print("STRUCTURE VALIDATION")
    print("="*60)

    stats = extracted_data["statistics"]
    print(f"Chapters: {stats['chapters']} (expected: 4)")
    print(f"Main Sections: {stats['main_sections']} (expected: 40)")
    print(f"Appendices: {stats['appendices']} (expected: 5)")
    print(f"Appendix Sections: {stats['appendix_sections']} (expected: 14)")

    # Validation
    all_correct = (
        stats['chapters'] == 4 and
        stats['main_sections'] == 40 and
        stats['appendices'] == 5 and
        stats['appendix_sections'] == 14
    )

    if all_correct:
        print("\n✓ All structure validations PASSED!")
    else:
        print("\n✗ Some structure validations FAILED!")

    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)

    if result.cosine_similarity >= 0.8:
        quality = "EXCELLENT"
    elif result.cosine_similarity >= 0.6:
        quality = "GOOD"
    elif result.cosine_similarity >= 0.4:
        quality = "FAIR"
    else:
        quality = "POOR"

    print(f"Extraction Quality: {quality}")
    print(f"Cosine Similarity: {result.cosine_similarity:.2%}")
    print(f"Word Overlap: {result.word_overlap:.2%}")
    print(f"Structure Accuracy: {'100%' if all_correct else 'Needs review'}")


if __name__ == "__main__":
    main()
