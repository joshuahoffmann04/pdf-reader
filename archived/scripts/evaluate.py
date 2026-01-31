#!/usr/bin/env python3
"""
Evaluates PDF extraction quality by comparing extracted text with original.

Provides comprehensive metrics:
- Cosine similarity (content accuracy)
- BLEU score (n-gram precision)
- Word overlap (Jaccard similarity)
- Structure validation

Usage:
    python -m scripts.evaluate --pdf document.pdf --json output/document_extracted.json
    python -m scripts.evaluate --pdf document.pdf --json output/document_extracted.json --output report.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.extractor import PDFExtractor
from src.evaluation import Evaluator


@dataclass
class QualityReport:
    """Complete quality assessment report."""
    # Content metrics
    cosine_similarity: float
    bleu_score: float
    word_overlap: float

    # Structure metrics
    structure_detected: bool
    chapters_count: int
    sections_count: int
    appendices_count: int
    ab_excerpts_count: int

    # Feature metrics
    images_count: int
    tables_count: int
    ab_links_complete: bool

    # Overall
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
                "structure_detected": self.structure_detected,
                "chapters": self.chapters_count,
                "sections": self.sections_count,
                "appendices": self.appendices_count,
                "ab_excerpts": self.ab_excerpts_count
            },
            "feature_metrics": {
                "images": self.images_count,
                "tables": self.tables_count,
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

    # Preamble
    if data.get("preamble"):
        parts.append(data["preamble"])

    # Chapters and sections
    for chapter in data.get("chapters", []):
        parts.append(f"\n{chapter['numeral']}. {chapter['title']}\n")
        for section in chapter.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])
        for ab in chapter.get("ab_excerpts", []):
            parts.append(f"\n{ab['id']} {ab['title']} (AB)\n")
            parts.append(ab["content"])

    # Appendices
    for appendix in data.get("appendices", []):
        parts.append(f"\n{appendix['id']}: {appendix['title']}\n")
        if appendix.get("content"):
            parts.append(appendix["content"])
        for section in appendix.get("sections", []):
            parts.append(f"\n{section['id']} {section['title']}\n")
            parts.append(section["content"])

    return "\n".join(parts)


def check_ab_linking(data: dict) -> bool:
    """Check if all AB excerpts are properly linked."""
    for chapter in data.get("chapters", []):
        for ab in chapter.get("ab_excerpts", []):
            if not ab.get("follows_section"):
                return False
    return True


def calculate_overall_score(
    cosine_sim: float,
    structure_detected: bool,
    ab_links_complete: bool,
    images_count: int,
    tables_count: int
) -> float:
    """
    Calculate overall quality score (0-100%).

    Weights:
    - Content similarity: 60%
    - Structure detection: 20%
    - Feature extraction: 20%
    """
    # Content score (60%)
    content_score = cosine_sim * 60

    # Structure score (20%)
    structure_score = 20 if structure_detected else 0

    # Feature score (20%)
    feature_score = 0
    if ab_links_complete:
        feature_score += 8
    if images_count > 0:
        feature_score += 6
    if tables_count > 0:
        feature_score += 6

    return content_score + structure_score + feature_score


def evaluate_extraction(
    json_path: Path,
    pdf_path: Path
) -> QualityReport:
    """
    Run comprehensive evaluation of PDF extraction.

    Args:
        json_path: Path to extracted JSON file.
        pdf_path: Path to original PDF file.

    Returns:
        QualityReport with all metrics.
    """
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

    # Get statistics from extracted data
    stats = extracted_data.get("statistics", {})
    chapters_count = stats.get("chapters", 0)
    sections_count = stats.get("main_sections", 0)
    appendices_count = stats.get("appendices", 0)
    ab_excerpts_count = stats.get("ab_excerpts", 0)
    images_count = stats.get("images", 0)
    tables_count = stats.get("tables", 0)

    # Check structure detection
    structure_detected = chapters_count > 0 or sections_count > 0

    # Check AB linking
    ab_links_complete = check_ab_linking(extracted_data)

    # Calculate overall score
    overall_score = calculate_overall_score(
        eval_result.cosine_similarity,
        structure_detected,
        ab_links_complete,
        images_count,
        tables_count
    )

    return QualityReport(
        cosine_similarity=eval_result.cosine_similarity,
        bleu_score=eval_result.bleu_score,
        word_overlap=eval_result.word_overlap,
        structure_detected=structure_detected,
        chapters_count=chapters_count,
        sections_count=sections_count,
        appendices_count=appendices_count,
        ab_excerpts_count=ab_excerpts_count,
        images_count=images_count,
        tables_count=tables_count,
        ab_links_complete=ab_links_complete,
        overall_score=overall_score
    )


def print_report(report: QualityReport) -> None:
    """Print formatted quality report."""
    print("\n" + "=" * 60)
    print("PDF EXTRACTION QUALITY REPORT")
    print("=" * 60)

    print("\nCONTENT METRICS")
    print("-" * 40)
    print(f"  Cosine Similarity: {report.cosine_similarity:>8.2%}")
    print(f"  BLEU Score:        {report.bleu_score:>8.2%}")
    print(f"  Word Overlap:      {report.word_overlap:>8.2%}")

    print("\nSTRUCTURE DETECTION")
    print("-" * 40)
    print(f"  Structure Found:   {'Yes' if report.structure_detected else 'No'}")
    print(f"  Chapters:          {report.chapters_count:>8}")
    print(f"  Main Sections:     {report.sections_count:>8}")
    print(f"  AB Excerpts:       {report.ab_excerpts_count:>8}")
    print(f"  Appendices:        {report.appendices_count:>8}")

    print("\nFEATURE EXTRACTION")
    print("-" * 40)
    print(f"  Images Found:      {report.images_count:>8}")
    print(f"  Tables Found:      {report.tables_count:>8}")
    print(f"  AB Links Complete: {'Yes' if report.ab_links_complete else 'No'}")

    print("\n" + "=" * 60)
    print(f"OVERALL QUALITY SCORE: {report.overall_score:.1f}%")
    print("=" * 60)

    if report.overall_score >= 95:
        print("EXCELLENT - Extraction meets high quality standards")
    elif report.overall_score >= 85:
        print("GOOD - Minor improvements possible")
    elif report.overall_score >= 70:
        print("ACCEPTABLE - Some issues detected")
    else:
        print("NEEDS IMPROVEMENT - Review extraction settings")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate PDF extraction quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pdf document.pdf --json output/document_extracted.json
  %(prog)s --pdf document.pdf --json output/document_extracted.json --output report.json
        """
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to original PDF"
    )
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to extracted JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save report as JSON file"
    )

    args = parser.parse_args()

    # Check files exist
    if not args.pdf.exists():
        print(f"Error: PDF not found: {args.pdf}")
        return 1

    if not args.json.exists():
        print(f"Error: JSON not found: {args.json}")
        print("       Run main.py first to generate extraction.")
        return 1

    print(f"Evaluating: {args.json.name}")
    print(f"Reference:  {args.pdf.name}")

    # Run evaluation
    try:
        report = evaluate_extraction(args.json, args.pdf)
    except Exception as e:
        print(f"Error: Evaluation failed: {e}")
        return 1

    # Print report
    print_report(report)

    # Save JSON report if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {args.output}")

    # Return exit code based on quality
    return 0 if report.overall_score >= 70 else 1


if __name__ == "__main__":
    sys.exit(main())
