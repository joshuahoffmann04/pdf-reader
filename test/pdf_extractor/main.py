"""Entry point for the pdf_extractor evaluation pipeline."""

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

from pdf_extractor import ProcessingConfig  # noqa: E402
from test.pdf_extractor.evaluate import run_evaluation  # noqa: E402
from test.pdf_extractor.report import Thresholds  # noqa: E402


PDF_PATH = Path("pdfs\\2-aend-19-02-25_msc-computer-science_lese.pdf")
REFERENCE_PATH = Path("pdf_extractor_reference.json")
OUTPUT_DIR = Path("test/pdf_extractor/output")


def main() -> None:
    config = ProcessingConfig(
        extraction_mode="hybrid",
        use_llm=False,
        llm_postprocess=False,
        context_mode="heuristic",
        table_extraction=True,
        layout_mode="columns",
        enforce_text_coverage=True,
        ocr_enabled=True,
        ocr_before_vision=True,
    )

    thresholds = Thresholds(
        cer_max=0.02,
        wer_max=0.05,
        token_recall_min=0.99,
        number_recall_min=1.0,
    )

    report = run_evaluation(
        pdf_path=PDF_PATH,
        reference_path=REFERENCE_PATH,
        output_dir=OUTPUT_DIR,
        config=config,
        thresholds=thresholds,
    )

    summary = report.get("summary", {})
    print(f"Pages: {summary.get('pages')}")
    print(f"Pass: {summary.get('pass')}")
    print(f"CER avg: {summary.get('cer_avg'):.4f}")
    print(f"WER avg: {summary.get('wer_avg'):.4f}")


if __name__ == "__main__":
    main()
