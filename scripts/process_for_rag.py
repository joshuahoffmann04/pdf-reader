#!/usr/bin/env python3
"""
PDF to RAG Pipeline

Processes PDF documents using Vision LLM for high-quality natural language extraction.

Usage:
    python scripts/process_for_rag.py input.pdf -o output_dir/
    python scripts/process_for_rag.py input.pdf --estimate-cost
    python scripts/process_for_rag.py input.pdf --model claude-3-haiku-20240307

Environment:
    ANTHROPIC_API_KEY: Your Anthropic API key (or use --api-key)
    OPENAI_API_KEY: Your OpenAI API key (if using OpenAI)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_processor.models import ProcessingConfig
from src.llm_processor.pdf_to_images import PDFToImages, estimate_api_cost
from src.llm_processor.vision_processor import VisionProcessor
from src.llm_processor.chunk_generator import ChunkGenerator


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
        description='Process PDF documents for RAG using Vision LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a PDF with default settings
  python scripts/process_for_rag.py document.pdf -o output/

  # Estimate cost before processing
  python scripts/process_for_rag.py document.pdf --estimate-cost

  # Use a cheaper model
  python scripts/process_for_rag.py document.pdf --model claude-3-haiku-20240307

  # Use OpenAI instead of Anthropic
  python scripts/process_for_rag.py document.pdf --provider openai --model gpt-4o
        """
    )

    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', default='output_rag',
                        help='Output directory (default: output_rag)')
    parser.add_argument('--api-key', help='API key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--provider', choices=['anthropic', 'openai'], default='anthropic',
                        help='API provider (default: anthropic)')
    parser.add_argument('--model', default='claude-sonnet-4-20250514',
                        help='Model to use (default: claude-sonnet-4-20250514)')
    parser.add_argument('--estimate-cost', action='store_true',
                        help='Only estimate cost, do not process')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Target chunk size in characters (default: 500)')
    parser.add_argument('--format', choices=['jsonl', 'json', 'both'], default='both',
                        help='Output format (default: both)')
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
    print(f"PDF to RAG Processor")
    print(f"{'='*60}")
    print(f"Document: {pdf_path.name}")
    print(f"Pages: {page_count}")
    if doc_info.get('title'):
        print(f"Title: {doc_info['title']}")
    print(f"{'='*60}\n")

    # Estimate cost
    cost_estimate = estimate_api_cost(page_count, args.model)
    if 'error' not in cost_estimate:
        print(f"Cost Estimate ({args.model}):")
        print(f"  Input tokens:  ~{cost_estimate['estimated_input_tokens']:,}")
        print(f"  Output tokens: ~{cost_estimate['estimated_output_tokens']:,}")
        print(f"  Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")
        print()

    if args.estimate_cost:
        print("Use without --estimate-cost to process the document.")
        sys.exit(0)

    # Get API key
    api_key = args.api_key
    if not api_key:
        env_var = 'ANTHROPIC_API_KEY' if args.provider == 'anthropic' else 'OPENAI_API_KEY'
        api_key = os.environ.get(env_var)
        if not api_key:
            logger.error(f"No API key provided. Set {env_var} or use --api-key")
            sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure processing
    config = ProcessingConfig(
        api_provider=args.provider,
        model=args.model,
        target_chunk_size=args.chunk_size,
        max_chunk_size=args.chunk_size * 2,
    )

    # Initialize processor
    print("Initializing Vision Processor...")
    processor = VisionProcessor(config=config, api_key=api_key)
    chunk_generator = ChunkGenerator(config=config)

    # Process document
    print("\nProcessing document...")
    print("This may take a while depending on document size.\n")

    try:
        extraction_result = processor.process_document(
            pdf_path,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

    print(f"\nExtraction complete in {extraction_result.processing_time_seconds:.1f}s")
    print(f"Tokens used: {extraction_result.total_input_tokens:,} input, {extraction_result.total_output_tokens:,} output")

    if extraction_result.errors:
        print(f"\nWarnings/Errors ({len(extraction_result.errors)}):")
        for error in extraction_result.errors[:5]:
            print(f"  - {error}")
        if len(extraction_result.errors) > 5:
            print(f"  ... and {len(extraction_result.errors) - 5} more")

    # Generate chunks
    print("\nGenerating RAG chunks...")
    source_name = pdf_path.stem
    result = chunk_generator.generate_from_extraction(extraction_result, source_name)

    stats = result.get_chunk_stats()
    print(f"Generated {stats['total_chunks']} chunks")
    print(f"  Average length: {stats['avg_length']:.0f} chars")
    print(f"  Range: {stats['min_length']} - {stats['max_length']} chars")
    print(f"  Types: {stats['by_type']}")

    # Export
    print("\nExporting...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.format in ['jsonl', 'both']:
        jsonl_path = output_dir / f"{source_name}_{timestamp}_chunks.jsonl"
        result.export_chunks_jsonl(str(jsonl_path))
        print(f"  JSONL: {jsonl_path}")

    if args.format in ['json', 'both']:
        json_path = output_dir / f"{source_name}_{timestamp}_chunks.json"
        chunk_generator.export_json(result.chunks, json_path)
        print(f"  JSON:  {json_path}")

    # Save full processing result
    result_path = output_dir / f"{source_name}_{timestamp}_result.json"
    result_data = {
        "document": str(pdf_path),
        "processed_at": datetime.now().isoformat(),
        "context": result.context.model_dump(),
        "pages": [p.model_dump() for p in result.pages],
        "stats": {
            "processing_time_seconds": result.processing_time_seconds,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
            "chunk_count": stats['total_chunks'],
        },
        "errors": result.errors,
    }
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Full result: {result_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
