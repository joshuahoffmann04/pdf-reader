"""Entry point for generation evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.generation.evaluate import run_evaluation  # noqa: E402
from test.generation.report import Thresholds  # noqa: E402


EVAL_PATH = Path("test/generation/queries.json")
OUTPUT_DIR = Path("test/generation/output")


def main() -> None:
    thresholds = Thresholds(
        answer_hit_rate_min=0.80,
        citation_hit_rate_min=0.70,
        missing_info_hit_rate_min=0.80,
    )

    report = run_evaluation(
        eval_path=EVAL_PATH,
        output_dir=OUTPUT_DIR,
        thresholds=thresholds,
        allow_skip=True,
    )

    summary = report.get("summary", {})
    print(f"Pass: {summary.get('pass')}")
    print(f"Answer hit rate: {summary.get('answer_hit_rate'):.3f}")
    print(f"Citation hit rate: {summary.get('citation_hit_rate'):.3f}")


if __name__ == "__main__":
    main()
