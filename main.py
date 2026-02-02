#!/usr/bin/env python3
"""
PDF Section Extractor - Command Line Interface.

Extract structured content from German academic PDF documents using GPT-4o Vision.

Usage:
    python main.py <pdf_file> [options]

Examples:
    python main.py document.pdf
    python main.py document.pdf -o output.json
    python main.py document.pdf --model gpt-4o-mini --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from pdf_extractor import (
    PDFExtractor,
    ExtractionConfig,
    ExtractionResult,
    PDFNotFoundError,
    StructureAggregationError,
    APIError,
    validate_pdf,
    estimate_api_cost,
)


# =============================================================================
# CLI FUNCTIONS
# =============================================================================


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging based on verbosity level."""
    if debug:
        level = logging.DEBUG
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    elif verbose:
        level = logging.INFO
        fmt = "%(levelname)s: %(message)s"
    else:
        level = logging.WARNING
        fmt = "%(levelname)s: %(message)s"

    logging.basicConfig(level=level, format=fmt)


def print_progress(current: int, total: int, message: str) -> None:
    """Print extraction progress."""
    bar_width = 30
    filled = int(bar_width * current / max(total, 1))
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] {current}/{total} {message}", end="", flush=True)
    if current == total:
        print()  # Newline at completion


def print_result_summary(result: ExtractionResult) -> None:
    """Print a summary of extraction results."""
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    # Document info
    print(f"\nDocument: {result.context.title}")
    print(f"Institution: {result.context.institution}")
    print(f"Type: {result.context.document_type.value}")

    if result.context.degree_program:
        print(f"Program: {result.context.degree_program}")

    # Statistics
    stats = result.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total sections: {stats['total_sections']}")
    print(f"  - Paragraphs (§): {stats['paragraphs']}")
    print(f"  - Anlagen: {stats['anlagen']}")
    print(f"  - Has preamble: {stats['has_preamble']}")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Content tokens: ~{stats['total_content_tokens']:,}")

    # Processing info
    print(f"\nProcessing:")
    print(f"  Time: {result.processing_time_seconds:.1f}s")
    print(f"  API tokens: {result.total_input_tokens + result.total_output_tokens:,}")

    # Show first few sections
    print(f"\nSections (first 5):")
    for section in result.sections[:5]:
        pages_str = f"{section.pages[0]}-{section.pages[-1]}" if len(section.pages) > 1 else str(section.pages[0])
        content_preview = section.content[:60].replace("\n", " ")
        print(f"  [{pages_str}] {section.identifier}: {content_preview}...")

    if len(result.sections) > 5:
        print(f"  ... and {len(result.sections) - 5} more")

    # Errors
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more")

    # Warnings
    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings[:3]:
            print(f"  - {warning}")


def estimate_cost(pdf_path: str) -> None:
    """Estimate API cost for a PDF."""
    try:
        info = validate_pdf(pdf_path)
        estimate = estimate_api_cost(info.page_count)

        print(f"\nCost Estimate for: {pdf_path}")
        print(f"  Pages: {info.page_count}")
        print(f"  Estimated tokens: ~{estimate['estimated_input_tokens'] + estimate['estimated_output_tokens']:,}")
        print(f"  Estimated cost: ~${estimate['estimated_cost_usd']:.4f}")
        print(f"\n  Note: Actual cost may vary based on content complexity.")

    except PDFNotFoundError:
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract structured content from German academic PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    Extract and save to document_extracted.json
  %(prog)s document.pdf -o output.json     Extract and save to output.json
  %(prog)s document.pdf --estimate         Estimate API cost without extraction
  %(prog)s document.pdf --verbose          Show detailed progress
        """,
    )

    parser.add_argument(
        "pdf_file",
        help="Path to the PDF file to extract",
    )

    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: <input>_extracted.json)",
    )

    parser.add_argument(
        "--model",
        choices=["gpt-4o", "gpt-4o-mini"],
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )

    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate API cost without performing extraction",
    )

    parser.add_argument(
        "--include-scan-results",
        action="store_true",
        help="Include raw page scan results in output (for debugging)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug information",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.0.0",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Setup logging
    setup_logging(verbose=args.verbose, debug=args.debug)

    # Check if file exists
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return 1

    # Cost estimation only
    if args.estimate:
        estimate_cost(str(pdf_path))
        return 0

    # Determine output path
    output_path = args.output or str(pdf_path.stem) + "_extracted.json"

    # Create config
    config = ExtractionConfig(
        model=args.model,
        include_scan_results=args.include_scan_results,
    )

    # Create extractor
    try:
        extractor = PDFExtractor(config=config)
    except ValueError as e:
        print(f"Error: {e}")
        print("Set OPENAI_API_KEY environment variable or create a .env file.")
        return 1

    # Run extraction
    print(f"Extracting: {pdf_path}")
    print(f"Model: {args.model}")

    try:
        result = extractor.extract(
            pdf_path,
            progress_callback=print_progress if args.verbose else None,
        )

        # Print summary
        print_result_summary(result)

        # Save result
        result.save(output_path)
        print(f"\nSaved: {output_path}")

        return 0

    except PDFNotFoundError as e:
        print(f"Error: PDF not found: {e}")
        return 1

    except StructureAggregationError as e:
        print(f"Error: Could not detect document structure: {e}")
        return 1

    except APIError as e:
        print(f"Error: API error: {e}")
        return 1

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
