"""Run extraction + chunking in one step.

Usage:
  python run_extract_chunk.py --pdf path\\to\\file.pdf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from pdf_extractor import ProcessingConfig
from pdf_extractor.config import ExtractorConfig
from pdf_extractor.service import ExtractionService
from chunking import DocumentChunker, ChunkingConfig
from chunking.storage import ChunkingStorage


def _progress(page: int, total: int, status: str) -> None:
    print(f"[{page}/{total}] {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extraction and chunking.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    if load_dotenv:
        load_dotenv(Path(".env"))

    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))

    processing = ProcessingConfig(
        extraction_mode="hybrid",
        use_llm=has_api_key,
        llm_postprocess=False,
        context_mode="llm_text" if has_api_key else "heuristic",
        table_extraction=True,
        layout_mode="columns",
        enforce_text_coverage=True,
        ocr_enabled=True,
        ocr_before_vision=True,
    )

    extractor_config = ExtractorConfig(processing=processing)
    service = ExtractionService(config=extractor_config)

    print("Starting extraction ...")
    result, document_id, extraction_path = service.extract_and_save(
        str(pdf_path),
        progress_callback=_progress,
    )

    print("Starting chunking ...")
    chunker = DocumentChunker(
        ChunkingConfig(
            max_chunk_tokens=512,
            min_chunk_tokens=50,
        )
    )
    chunk_result = chunker.chunk(result)
    chunk_storage = ChunkingStorage("data/chunking")
    chunk_paths = chunk_storage.save(chunk_result)

    print("Done.")
    print(f"document_id: {document_id}")
    print(f"extraction_path: {extraction_path}")
    print(f"chunk_path: {chunk_paths.chunk_file}")


if __name__ == "__main__":
    main()
