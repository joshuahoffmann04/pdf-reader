"""Evaluation runner for the chunking pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pdf_extractor import ExtractionResult
from chunking import DocumentChunker, ChunkingConfig
from chunking.token_counter import count_tokens

from .metrics import token_recall, number_recall, duplication_ratio, sentence_boundary_ratio
from .report import Thresholds, build_report, write_report


def _collect_source_text(extraction: ExtractionResult) -> str:
    return extraction.get_full_content(separator="\n\n")


def _validate_metadata(chunks: list[dict], document_id: str) -> list[str]:
    errors: list[str] = []
    for idx, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        chunk_id = chunk.get("chunk_id", "")

        if not chunk_id.endswith(f"{idx:04d}"):
            errors.append(f"chunk_id_mismatch:{chunk_id}")

        if meta.get("chunk_index") != idx:
            errors.append(f"chunk_index_mismatch:{chunk_id}")

        if meta.get("document_id") != document_id:
            errors.append(f"document_id_mismatch:{chunk_id}")

        if meta.get("prev_chunk_id") and idx == 0:
            errors.append(f"prev_chunk_present_on_first:{chunk_id}")

        if meta.get("next_chunk_id") and idx == len(chunks) - 1:
            errors.append(f"next_chunk_present_on_last:{chunk_id}")

        pages = meta.get("page_numbers", [])
        if pages != sorted(pages):
            errors.append(f"page_numbers_unsorted:{chunk_id}")

    return errors


def _chunk_size_violations(chunks: list[dict], thresholds: Thresholds) -> tuple[int, int]:
    too_small = 0
    too_large = 0
    for i, chunk in enumerate(chunks):
        token_count = chunk.get("token_count", 0)
        if token_count > thresholds.max_chunk_tokens:
            too_large += 1
        if token_count < thresholds.min_chunk_tokens and i < len(chunks) - 1:
            too_small += 1
    return too_small, too_large


def run_evaluation(
    extraction_path: Path,
    output_dir: Path,
    config: ChunkingConfig | None = None,
    thresholds: Thresholds | None = None,
    save_chunks: bool = True,
) -> dict[str, Any]:
    thresholds = thresholds or Thresholds()
    config = config or ChunkingConfig(
        max_chunk_tokens=thresholds.max_chunk_tokens,
        min_chunk_tokens=thresholds.min_chunk_tokens,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    extraction = ExtractionResult.load(str(extraction_path))
    chunker = DocumentChunker(config=config)
    chunking_result = chunker.chunk(extraction)
    chunk_output_path = None

    if save_chunks:
        chunk_output_path = str(output_dir / "chunks.json")
        chunking_result.save(chunk_output_path)

    source_text = _collect_source_text(extraction)
    chunk_texts = [chunk.text for chunk in chunking_result.chunks]
    merged_chunks = "\n\n".join(chunk_texts)

    source_tokens = count_tokens(source_text)
    chunk_tokens = sum(chunk.token_count for chunk in chunking_result.chunks)

    covered_pages = set()
    for chunk in chunking_result.chunks:
        covered_pages.update(chunk.metadata.page_numbers)

    metadata_errors = _validate_metadata(
        [chunk.to_dict() for chunk in chunking_result.chunks],
        chunking_result.document_id,
    )

    too_small, too_large = _chunk_size_violations(
        [chunk.to_dict() for chunk in chunking_result.chunks],
        thresholds,
    )

    metrics = {
        "chunks": len(chunking_result.chunks),
        "pages_in_source": len(extraction.pages),
        "pages_covered": len(covered_pages),
        "token_recall": token_recall(source_text, merged_chunks),
        "number_recall": number_recall(source_text, merged_chunks),
        "duplication_ratio": duplication_ratio(source_tokens, chunk_tokens),
        "sentence_boundary_ratio": sentence_boundary_ratio(chunk_texts),
        "too_small_chunks": too_small,
        "too_large_chunks": too_large,
        "metadata_errors": len(metadata_errors),
        "metadata_error_list": metadata_errors,
    }

    metrics["pass"] = (
        metrics["token_recall"] >= thresholds.token_recall_min
        and metrics["number_recall"] >= thresholds.number_recall_min
        and metrics["duplication_ratio"] <= thresholds.duplication_ratio_max
        and metrics["sentence_boundary_ratio"] >= thresholds.sentence_boundary_ratio_min
        and metrics["too_small_chunks"] == 0
        and metrics["too_large_chunks"] == 0
        and metrics["metadata_errors"] == 0
    )

    report = build_report(
        extraction_path=str(extraction_path),
        metrics=metrics,
        thresholds=thresholds,
        chunk_output_path=chunk_output_path,
    )
    write_report(report, output_dir)
    return report
