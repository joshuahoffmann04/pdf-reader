"""Entry point for MARley evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(ROOT / ".env")

from test.marley.evaluate import MarleyEvalConfig, run_evaluation  # noqa: E402
from test.marley.report import Thresholds  # noqa: E402


QUESTIONS_PATH = Path("test/marley/msc_computer_science_questions.json")
OUTPUT_DIR = Path("test/marley/output")


def main() -> None:
    config = MarleyEvalConfig(
        mode="hybrid",
        top_k=8,
        max_context_tokens=2048,
        output_tokens=512,
    )

    thresholds = Thresholds(
        answer_semantic_similarity_min=0.80,
        answer_number_recall_min=1.0,
        citation_page_hit_min=1.0,
        quote_support_min=0.80,
    )

    questions_path = QUESTIONS_PATH
    if not questions_path.exists():
        fallback = Path("msc_computer_science_questions.json")
        if fallback.exists():
            questions_path = fallback

    report = run_evaluation(
        questions_path=questions_path,
        output_dir=OUTPUT_DIR,
        config=config,
        thresholds=thresholds,
        allow_skip=True,
    )

    summary = report.get("summary", {})
    print(f"Questions: {summary.get('questions')}")
    print(f"Pass: {summary.get('pass')}")
    print(f"Passed (strict): {summary.get('passed_strict')}, Failed (strict): {summary.get('failed_strict')}")
    print(f"Passed (answer): {summary.get('passed_answer')}, Failed (answer): {summary.get('failed_answer')}")
    print(f"Passed (citations): {summary.get('passed_citations')}, Failed (citations): {summary.get('failed_citations')}")
    print(f"Answer semantic sim avg: {summary.get('answer_semantic_similarity_avg'):.3f}")
    print(f"Answer number recall avg: {summary.get('answer_number_recall_avg'):.3f}")
    print(f"Citation page hit rate: {summary.get('citation_page_hit_rate'):.3f}")


if __name__ == "__main__":
    main()
