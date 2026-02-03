# PDF Extractor Service

Extrahiert strukturierte Inhalte aus PDF-Dokumenten via OpenAI Vision API. Das Modul liefert eine JSON-Ausgabe, die für Chunking und Retrieval vorbereitet ist.

## Struktur

```
pdf_extractor/
├── app.py              # FastAPI OpenAPI Service
├── config.py           # Service-Konfiguration
├── service.py          # Extraktions-Orchestrierung
├── storage.py          # Persistenz (data/pdf_extractor/<document_id>/extraction/...)
├── extractor.py        # Vision-Extraktor (Core)
├── models.py           # Datenmodelle + Request/Response
├── prompts.py          # Prompt-Vorlagen
├── pdf_to_images.py    # PDF → Image Pipeline
└── README.md
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
python scripts/pdf_extractor_service.py
```

### API

- `GET /health` → `{ "status": "ok" }`
- `POST /extract` → startet eine Extraktion

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

## Tests

```bash
pytest tests/test_extractor.py tests/test_extraction_models.py -v
```
