"""Entry point for the chunking evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunking import ChunkingConfig  # noqa: E402
from test.chunking.evaluate import run_evaluation  # noqa: E402
from test.chunking.report import Thresholds  # noqa: E402


EXTRACTION_PATH = Path("output.json")
OUTPUT_DIR = Path("test/chunking/output")


def main() -> None:
    thresholds = Thresholds(
        token_recall_min=0.99,
        number_recall_min=1.0,
        duplication_ratio_max=1.8,
        sentence_boundary_ratio_min=0.80,
        max_chunk_tokens=512,
        min_chunk_tokens=50,
    )

    config = ChunkingConfig(
        max_chunk_tokens=thresholds.max_chunk_tokens,
        overlap_tokens=100,
        min_chunk_tokens=thresholds.min_chunk_tokens,
    )

    report = run_evaluation(
        extraction_path=EXTRACTION_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
        thresholds=thresholds,
    )

    summary = report.get("summary", {})
    print(f"Pass: {summary.get('pass')}")
    print(f"Chunks: {summary.get('chunks')}")
    print(f"Token recall: {summary.get('token_recall'):.4f}")
    print(f"Number recall: {summary.get('number_recall'):.4f}")
    print(f"Duplication ratio: {summary.get('duplication_ratio'):.4f}")


if __name__ == "__main__":
    main()
