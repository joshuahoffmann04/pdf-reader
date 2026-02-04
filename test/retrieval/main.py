"""Entry point for the retrieval evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
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


def main() -> None:
    thresholds = Thresholds(
        hit_rate_min=0.80,
        mrr_min=0.50,
        text_hit_rate_min=0.80,
    )

    results = run_evaluation(
        chunks_path=CHUNKS_PATH,
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
