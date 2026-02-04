"""Entry point for the retrieval evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.retrieval.evaluate import run_evaluation  # noqa: E402
from test.retrieval.report import Thresholds  # noqa: E402


CHUNKS_PATH = Path("test\\chunking\\output\\chunks.json")
EVAL_PATH = Path("test/retrieval/queries.json")
OUTPUT_DIR = Path("test/retrieval/output")
MODES = ["bm25", "vector", "hybrid"]


def _load_eval(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _find_chunks_path(expected_document_id: str | None) -> Path | None:
    # Prefer the standard test output.
    if CHUNKS_PATH.exists():
        return CHUNKS_PATH

    # Prefer chunks for the expected document id.
    if expected_document_id:
        candidate_dir = Path("data/chunking") / expected_document_id / "chunks"
        if candidate_dir.exists():
            files = sorted(
                candidate_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if files:
                return files[0]

    # Fallback: newest chunk file in data/chunking.
    data_root = Path("data/chunking")
    if data_root.exists():
        candidates = list(data_root.rglob("chunks/*.json"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def main() -> None:
    eval_data = _load_eval(EVAL_PATH)
    default_filters = eval_data.get("default_filters") or {}
    expected_document_id = default_filters.get("document_id")

    chunks_path = _find_chunks_path(expected_document_id)
    if not chunks_path:
        raise SystemExit(
            "No chunk file found. Run chunking first (e.g. via run_extract_chunk.py) "
            "or set CHUNKS_PATH in test/retrieval/main.py."
        )
    print(f"Using chunks: {chunks_path}")

    thresholds = Thresholds(
        hit_rate_min=0.80,
        mrr_min=0.50,
        text_hit_rate_min=0.80,
    )

    results = run_evaluation(
        chunks_path=chunks_path,
        eval_path=EVAL_PATH,
        output_dir=OUTPUT_DIR,
        modes=MODES,
        thresholds=thresholds,
        allow_skip=True,
    )

    for mode, report in results.items():
        summary = report.get("summary", {})
        print(f"{mode}: pass={summary.get('pass')}, hit_rate={summary.get('hit_rate'):.3f}")


if __name__ == "__main__":
    main()
