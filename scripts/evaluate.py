#!/usr/bin/env python3
"""
Evaluates PDF extraction quality by comparing extracted text with original.

Provides comprehensive metrics:
- Cosine similarity (content accuracy)
- Structure validation (chapters, sections, appendices)
- Feature completeness (images, tables, AB references)
- Overall quality score

Usage: python -m scripts.evaluate [--pdf PDF_PATH] [--json JSON_PATH]
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from src.extractor import PDFExtractor
from src.evaluation import Evaluator


@dataclass
class QualityReport:
    """Complete quality assessment report."""
    cosine_similarity: float
    bleu_score: float
    word_overlap: float
    structure_valid: bool
    chapters_correct: bool
    sections_correct: bool
    appendices_correct: bool
    images_found: int
    tables_found: int
    ab_excerpts_found: int
    ab_links_complete: bool
    overall_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "content_metrics": {
                "cosine_similarity": round(self.cosine_similarity, 4),
                "bleu_score": round(self.bleu_score, 4),
                "word_overlap": round(self.word_overlap, 4)
            },
            "structure_metrics": {
                "structure_valid": self.structure_valid,
                "chapters_correct": self.chapters_correct,
                "sections_correct": self.sections_correct,
                "appendices_correct": self.appendices_correct
            },
            "feature_metrics": {
                "images_found": self.images_found,
                "tables_found": self.tables_found,
                "ab_excerpts_found": self.ab_excerpts_found,
                "ab_links_complete": self.ab_links_complete
            },
            "overall_score": round(self.overall_score, 4)
        }


def load_extracted_json(json_path: Path) -> dict:
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
        for ab in chapter.get("ab_excerpts", []):
            parts.append(f"\n{ab['id']} {ab['title']} (AB)\n")
            parts.append(ab["content"])

    for appendix in data.get("appendices", []):
        parts.append(f"\n{appendix['id']}: {appendix['title']}\n")
        if appendix.get("content"):
            parts.append(appendix["content"])
        for section in appendix.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])

    return "\n".join(parts)


def validate_structure(data: dict, expected: dict) -> tuple[bool, bool, bool, bool]:
    """
    Validate document structure against expected values.

    Returns: (structure_valid, chapters_ok, sections_ok, appendices_ok)
    """
    stats = data.get("statistics", {})

    chapters_ok = stats.get("chapters", 0) == expected.get("chapters", 4)
    sections_ok = stats.get("main_sections", 0) == expected.get("main_sections", 40)
    appendices_ok = stats.get("appendices", 0) == expected.get("appendices", 5)
    appendix_sections_ok = stats.get("appendix_sections", 0) == expected.get("appendix_sections", 14)

    structure_valid = chapters_ok and sections_ok and appendices_ok and appendix_sections_ok

    return structure_valid, chapters_ok, sections_ok, appendices_ok


def check_ab_linking(data: dict) -> bool:
    """Check if all AB excerpts are properly linked."""
    for chapter in data.get("chapters", []):
        for ab in chapter.get("ab_excerpts", []):
            if not ab.get("follows_section"):
                return False
    return True


def calculate_overall_score(
    cosine_sim: float,
    structure_valid: bool,
    ab_links_complete: bool,
    images_found: int,
    tables_found: int
) -> float:
    """
    Calculate overall quality score (0-100%).

    Weights:
    - Content similarity: 50%
    - Structure validity: 25%
    - Feature completeness: 25%
    """
    # Content score (50%)
    content_score = cosine_sim * 50

    # Structure score (25%)
    structure_score = 25 if structure_valid else 0

    # Feature score (25%)
    feature_score = 0
    if ab_links_complete:
        feature_score += 10
    if images_found > 0:
        feature_score += 7.5
    if tables_found > 0:
        feature_score += 7.5

    return content_score + structure_score + feature_score


def evaluate_extraction(
    json_path: Path,
    pdf_path: Path,
    expected: dict = None
) -> QualityReport:
    """
    Run comprehensive evaluation of PDF extraction.

    Args:
        json_path: Path to extracted JSON file.
        pdf_path: Path to original PDF file.
        expected: Expected structure values (optional).

    Returns:
        QualityReport with all metrics.
    """
    if expected is None:
        expected = {
            "chapters": 4,
            "main_sections": 40,
            "appendices": 5,
            "appendix_sections": 14
        }

    # Load extracted data
    extracted_data = load_extracted_json(json_path)

    # Extract original text
    extractor = PDFExtractor()
    original_text = extractor.extract(pdf_path).get_full_text()

    # Reconstruct extracted text
    reconstructed = reconstruct_text_from_json(extracted_data)

    # Calculate similarity metrics
    evaluator = Evaluator(language="de")
    eval_result = evaluator.evaluate(reconstructed, original_text)

    # Validate structure
    structure_valid, chapters_ok, sections_ok, appendices_ok = validate_structure(
        extracted_data, expected
    )

    # Check AB linking
    ab_links_complete = check_ab_linking(extracted_data)

    # Get feature counts
    stats = extracted_data.get("statistics", {})
    images_found = stats.get("images", 0)
    tables_found = stats.get("tables", 0)
    ab_excerpts_found = stats.get("ab_excerpts", 0)

    # Calculate overall score
    overall_score = calculate_overall_score(
        eval_result.cosine_similarity,
        structure_valid,
        ab_links_complete,
        images_found,
        tables_found
    )

    return QualityReport(
        cosine_similarity=eval_result.cosine_similarity,
        bleu_score=eval_result.bleu_score,
        word_overlap=eval_result.word_overlap,
        structure_valid=structure_valid,
        chapters_correct=chapters_ok,
        sections_correct=sections_ok,
        appendices_correct=appendices_ok,
        images_found=images_found,
        tables_found=tables_found,
        ab_excerpts_found=ab_excerpts_found,
        ab_links_complete=ab_links_complete,
        overall_score=overall_score
    )


def print_report(report: QualityReport) -> None:
    """Print formatted quality report."""
    print("\n" + "=" * 60)
    print("PDF EXTRACTION QUALITY REPORT")
    print("=" * 60)

    print("\n📊 CONTENT METRICS")
    print("-" * 40)
    print(f"  Cosine Similarity: {report.cosine_similarity:>8.2%}")
    print(f"  BLEU Score:        {report.bleu_score:>8.2%}")
    print(f"  Word Overlap:      {report.word_overlap:>8.2%}")

    print("\n📋 STRUCTURE VALIDATION")
    print("-" * 40)
    print(f"  Structure Valid:   {'✓' if report.structure_valid else '✗'}")
    print(f"  Chapters Correct:  {'✓' if report.chapters_correct else '✗'}")
    print(f"  Sections Correct:  {'✓' if report.sections_correct else '✗'}")
    print(f"  Appendices Correct:{'✓' if report.appendices_correct else '✗'}")

    print("\n🔗 FEATURE COMPLETENESS")
    print("-" * 40)
    print(f"  Images Found:      {report.images_found:>8}")
    print(f"  Tables Found:      {report.tables_found:>8}")
    print(f"  AB Excerpts:       {report.ab_excerpts_found:>8}")
    print(f"  AB Links Complete: {'✓' if report.ab_links_complete else '✗'}")

    print("\n" + "=" * 60)
    print(f"OVERALL QUALITY SCORE: {report.overall_score:.1f}%")
    print("=" * 60)

    if report.overall_score >= 95:
        print("✓ EXCELLENT - Extraction meets quality standards")
    elif report.overall_score >= 85:
        print("○ GOOD - Minor improvements possible")
    elif report.overall_score >= 70:
        print("△ ACCEPTABLE - Some issues detected")
    else:
        print("✗ POOR - Significant improvements needed")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate PDF extraction quality"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("pdfs/Pruefungsordnung_BSc_Inf_2024.pdf"),
        help="Path to original PDF"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("output_final/Pruefungsordnung_BSc_Inf_2024_extracted.json"),
        help="Path to extracted JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save report as JSON"
    )

    args = parser.parse_args()

    # Check files exist
    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}")
        return 1

    if not args.json.exists():
        print(f"ERROR: JSON not found: {args.json}")
        print("       Run main.py first to generate extraction.")
        return 1

    print(f"Evaluating: {args.json.name}")
    print(f"Reference:  {args.pdf.name}")

    # Run evaluation
    try:
        report = evaluate_extraction(args.json, args.pdf)
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return 1

    # Print report
    print_report(report)

    # Save JSON report if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {args.output}")

    # Return exit code based on quality
    return 0 if report.overall_score >= 95 else 1


if __name__ == "__main__":
    sys.exit(main())
