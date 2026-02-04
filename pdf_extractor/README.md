# PDF Extractor Service

Extrahiert strukturierte Inhalte aus PDF-Dokumenten via OpenAI Vision API. Das Modul liefert eine JSON-Ausgabe, die fuer Chunking und Retrieval vorbereitet ist.

## Struktur

```
pdf_extractor/
|-- app.py              # FastAPI OpenAPI Service
|-- config.py           # Service-Konfiguration
|-- service.py          # Extraktions-Orchestrierung
|-- storage.py          # Persistenz (data/pdf_extractor/<document_id>/extraction/...)
|-- extractor.py        # Extractor (Core)
|-- text_extractor.py   # Text-native Extraktion (PyMuPDF)
|-- validation.py       # LLM-Ausgabe-Validierung
|-- qa.py               # QA-Utilities (Questions, Coverage, Scanned Pages)
|-- models.py           # Datenmodelle + Request/Response
|-- prompts.py          # Prompt-Vorlagen
|-- pdf_to_images.py    # PDF -> Image Pipeline
`-- README.md
```

## Quick Start (Python)

```python
from pdf_extractor import PDFExtractor

extractor = PDFExtractor()
result = extractor.extract("pdfs/example.pdf")
result.save("output.json")
```

## Service starten

```bash
python scripts/pdf_extractor_service.py --serve
```

### API

- `GET /health` -> `{ "status": "ok" }`
- `POST /extract` -> startet eine Extraktion

**Request**
```json
{ "pdf_path": "pdfs/example.pdf" }
```

**Response**
```json
{
  "document_id": "example",
  "output_path": "data/pdf_extractor/example/extraction/example_20240101_120000.json",
  "pages": 42
}
```

## Persistenz

- Standard-Ordner: `data/pdf_extractor`
- Ablagepfad: `data/pdf_extractor/<document_id>/extraction/<document_id>_<timestamp>.json`

Die Pfade werden von `ExtractionStorage` erzeugt und sind deterministisch pro Dokument und Zustand.

## Konfiguration

```python
from pdf_extractor.config import ExtractorConfig
from pdf_extractor.models import ProcessingConfig

config = ExtractorConfig(
    data_dir="data/pdf_extractor",
    processing=ProcessingConfig(model="gpt-4o-mini"),
)
```

### Extraction Mode (empfohlen)

```python
from pdf_extractor.models import ProcessingConfig

processing = ProcessingConfig(
    extraction_mode="hybrid",    # text|vision|hybrid
    use_llm=True,                # optional
    llm_postprocess=False,       # sicherer Default (keine Zusammenfassung)
    context_mode="llm_text",     # llm_text|llm_vision|heuristic
    table_extraction=False,      # optional: pdfplumber Tabellen
    layout_mode="columns",       # bessere Reihenfolge bei Mehrspalten
    enforce_text_coverage=True,  # hartes Coverage-Gate
    ocr_enabled=False,           # OCR-Fallback fuer gescannte Seiten
    ocr_before_vision=True,      # OCR vor Vision (hybrid)
    ocr_dpi=300,                 # OCR-Render-DPI
)
```

## CLI: Einfache Extraktion

```bash
python scripts/pdf_extractor_service.py --pdf pdfs/example.pdf
```

Optional mit Output-Datei:

```bash
python scripts/pdf_extractor_service.py --pdf pdfs/example.pdf --output data/output.json
```

### OCR-Check (gescannte Seiten erkennen)

```bash
python scripts/detect_scanned_pages.py pdfs/example.pdf
```

Hinweis: Text-native Extraktion ist i.d.R. genauer als reine Vision-Extraktion. Bei fehlendem API-Key kann der Extractor automatisch auf Text-only fallen.

### Tesseract Pfad (falls nicht im PATH)

Falls `tesseract` installiert ist, aber nicht gefunden wird, setze den Pfad:

```
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
```

## Tests

```bash
pytest tests/test_extractor.py tests/test_extraction_models.py -v
```

## Page-Level QA (one question per page)

Generate one question per page for manual/automatic verification:

```bash
python scripts/generate_page_questions.py pdfs/example.pdf --output data/eval/page_questions_example.json
```

Check extraction output against the questions:

```bash
python scripts/check_page_questions.py data/eval/page_questions_example.json data/pdf_extractor/example/extraction/example_*.json
```

Validate token/number coverage (strict):

```bash
python scripts/validate_extraction_coverage.py pdfs/example.pdf data/pdf_extractor/example/extraction/example_*.json --output data/eval/coverage_report_example.json
```

## Extractor Pipeline Test

```bash
pytest tests/test_extractor_pipeline.py -v
```

## Batch QA Suite

Run extraction + QA (one question per page + coverage) for all PDFs:

```bash
python scripts/batch_test_suite.py --table --layout columns
```

Reuse the latest extraction output (skip re-extraction):

```bash
python scripts/batch_test_suite.py --reuse-latest --table --layout columns
```
