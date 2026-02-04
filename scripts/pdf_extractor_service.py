import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv(ROOT / ".env")

from pdf_extractor.app import create_app
from pdf_extractor.config import ExtractorConfig
from pdf_extractor.models import ProcessingConfig
from pdf_extractor.service import ExtractionService
import uvicorn


def build_processing_config() -> ProcessingConfig:
    return ProcessingConfig(
        extraction_mode="hybrid",
        use_llm=True,
        llm_postprocess=False,
        context_mode="llm_text",
        table_extraction=True,
        layout_mode="columns",
        enforce_text_coverage=True,
        ocr_enabled=True,
        ocr_before_vision=True,
    )


def run_extract(pdf_path: str, output_path: str | None = None) -> None:
    config = ExtractorConfig(processing=build_processing_config())
    service = ExtractionService(config)
    result, document_id, stored_path = service.extract_and_save(pdf_path)
    print(f"document_id: {document_id}")
    print(f"output_path: {stored_path}")
    print(f"pages: {len(result.pages)}")
    if output_path:
        result.save(output_path)
        print(f"saved_copy: {output_path}")


def run_server(host: str, port: int) -> None:
    config = ExtractorConfig(processing=build_processing_config())
    app = create_app(config)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF Extractor runner (CLI extraction or API server)."
    )
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8001, help="Server port")
    parser.add_argument("--pdf", help="Path to a PDF to extract")
    parser.add_argument("--output", help="Optional output path for extraction JSON")
    args = parser.parse_args()

    if args.serve:
        run_server(args.host, args.port)
        return

    if not args.pdf:
        parser.error("Provide --pdf or use --serve to run the API.")
    run_extract(args.pdf, args.output)


if __name__ == "__main__":
    main()
