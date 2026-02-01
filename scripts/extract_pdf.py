#!/usr/bin/env python3
"""
PDF Extraction Script

Extracts content from PDF documents using OpenAI's Vision API (GPT-4o).
Uses section-based extraction (§§, Anlagen) for optimal downstream processing.

Usage:
    python scripts/extract_pdf.py input.pdf -o output/
    python scripts/extract_pdf.py input.pdf --estimate-cost
    python scripts/extract_pdf.py input.pdf --model gpt-4o-mini

Environment:
    OPENAI_API_KEY: Your OpenAI API key (required)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

from pdf_extractor import (
    PDFExtractor,
    PDFToImages,
    ExtractionConfig,
    estimate_api_cost,
    NoTableOfContentsError,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def progress_callback(current: int, total: int, status: str):
    """Print progress updates."""
    percent = (current / total * 100) if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = '=' * filled + '-' * (bar_len - filled)
    print(f'\r[{bar}] {percent:5.1f}% - {status}', end='', flush=True)
    if current == total:
        print()  # Newline when done


def main():
    parser = argparse.ArgumentParser(
        description='Extract content from PDF documents using OpenAI Vision API (Section-Based)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract content from a PDF with default settings (gpt-4o)
  python scripts/extract_pdf.py document.pdf -o output/

  # Estimate cost before processing
  python scripts/extract_pdf.py document.pdf --estimate-cost

  # Use a cheaper/faster model
  python scripts/extract_pdf.py document.pdf --model gpt-4o-mini

  # Verbose output for debugging
  python scripts/extract_pdf.py document.pdf -o output/ -v

  # More images per request (faster for small sections)
  python scripts/extract_pdf.py document.pdf --max-images 10
        """
    )

    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--api-key',
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model', default='gpt-4o',
                        help='OpenAI model to use (default: gpt-4o)')
    parser.add_argument('--estimate-cost', action='store_true',
                        help='Only estimate cost, do not process')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum retries for failed sections (default: 3)')
    parser.add_argument('--max-images', type=int, default=5,
                        help='Maximum images per API request (default: 5, max: 20)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)

    # Get page count
    converter = PDFToImages()
    try:
        page_count = converter.get_page_count(pdf_path)
        doc_info = converter.get_document_info(pdf_path)
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PDF Extractor v2.0 - Section-Based Extraction")
    print(f"{'='*60}")
    print(f"Document: {pdf_path.name}")
    print(f"Pages: {page_count}")
    if doc_info.get('title'):
        print(f"Title: {doc_info['title']}")
    print(f"Model: {args.model}")
    print(f"Max images/request: {args.max_images}")
    print(f"{'='*60}\n")

    # Estimate cost
    cost_estimate = estimate_api_cost(page_count, args.model)
    if 'error' not in cost_estimate:
        print(f"Cost Estimate:")
        print(f"  Input tokens:  ~{cost_estimate['estimated_input_tokens']:,}")
        print(f"  Output tokens: ~{cost_estimate['estimated_output_tokens']:,}")
        print(f"  Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")
        print()

    if args.estimate_cost:
        print("Use without --estimate-cost to process the document.")
        sys.exit(0)

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        logger.error("No API key provided. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure processing
    config = ExtractionConfig(
        model=args.model,
        max_retries=args.max_retries,
        max_images_per_request=min(args.max_images, 20),
    )

    # Initialize extractor
    print("Initializing PDF Extractor...")
    extractor = PDFExtractor(config=config, api_key=api_key)

    # Process document
    print("\nExtracting sections...")
    print("This may take a while depending on document size.\n")

    try:
        result = extractor.extract(
            pdf_path,
            progress_callback=progress_callback,
        )
    except NoTableOfContentsError as e:
        logger.error(f"No table of contents found: {e}")
        print("\nERROR: This document has no table of contents.")
        print("Section-based extraction requires a ToC to determine document structure.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

    print(f"\nExtraction complete in {result.processing_time_seconds:.1f}s")
    print(f"Tokens used: {result.total_input_tokens:,} input, {result.total_output_tokens:,} output")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for error in result.errors[:5]:
            print(f"  - {error}")
        if len(result.errors) > 5:
            print(f"  ... and {len(result.errors) - 5} more")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for warning in result.warnings[:5]:
            print(f"  - {warning}")

    # Get statistics
    stats = result.get_stats()
    print(f"\nExtraction Statistics:")
    print(f"  Total sections: {stats['total_sections']}")
    print(f"  - Paragraphs (§§): {stats['paragraphs']}")
    print(f"  - Anlagen: {stats['anlagen']}")
    print(f"  - Has Overview: {'Yes' if stats['has_overview'] else 'No'}")
    print(f"  Sections with tables: {stats['sections_with_tables']}")
    print(f"  Sections with lists: {stats['sections_with_lists']}")
    print(f"  Failed sections: {stats['failed_sections']}")
    print(f"  Avg tokens/section: {stats['avg_tokens_per_section']:,}")

    # Save result
    print("\nSaving result...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_name = pdf_path.stem

    result_path = output_dir / f"{source_name}_{timestamp}_result.json"
    result.save(str(result_path))
    print(f"  Saved to: {result_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"\nOutput file ready for downstream processing (chunking, RAG, etc.)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
