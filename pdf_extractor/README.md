# PDF Extractor

Der PDF Extractor erzeugt eine strukturierte, seitenweise Extraktion aus PDFs.
Er ist die erste Stufe der Pipeline und liefert die Grundlage fuer Chunking, Retrieval und Generation.

## Schnellstart (Python)

```python
from pdf_extractor import PDFExtractor, ProcessingConfig

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

extractor = PDFExtractor(config=config)
result = extractor.extract("pdfs/example.pdf")
result.save("output.json")
```

## Schnellstart (API)

```powershell
uvicorn pdf_extractor.app:app --host 127.0.0.1 --port 8001
```

Endpoints:
- `GET /health`
- `POST /extract`

Request:

```json
{ "pdf_path": "pdfs/example.pdf" }
```

Response:

```json
{
  "document_id": "example",
  "output_path": "data/pdf_extractor/example/extraction/example_20240101_120000.json",
  "pages": 42
}
```

## Was passiert (Kurzfassung)

1. **Text-Extraktion (PyMuPDF)**: selektierbarer Text wird blockweise gelesen.
2. **Tabellen-Extraktion (optional)**: Tabellen werden in Fliesstext ueberfuehrt.
3. **OCR-Fallback (optional)**: gescannte Seiten werden mit Tesseract erkannt.
4. **Vision-Fallback (optional)**: wenn Text/OCR nicht reicht, kann ein LLM genutzt werden.
5. **Validierung**: Inhalte werden gegen Rohtext geprueft, um Informationsverlust zu vermeiden.

## Konfiguration

Wichtige Optionen in `ProcessingConfig`:
- `extraction_mode`: `text | vision | hybrid`
- `use_llm`: LLM fuer Vision-Extraction aktivieren
- `context_mode`: `llm_text | llm_vision | heuristic`
- `table_extraction`: Tabellen zusaetzlich extrahieren
- `ocr_enabled`: OCR-Fallback aktivieren
- `ocr_before_vision`: OCR vor Vision versuchen
- `layout_mode`: `simple | columns`
- `enforce_text_coverage`: harte Coverage-Pruefung
- `min_token_recall`, `min_number_recall`: Coverage-Schwellen

Service-Konfiguration: `pdf_extractor.config.ExtractorConfig` steuert u.a. den Output-Ordner.

## Output-Format (ExtractionResult)

Top-Level:
- `source_file`
- `context` (Dokument-Metadaten)
- `pages` (Liste aller Seiten)
- `errors`, `warnings`
- `processing_time_seconds`

Pro Seite (`ExtractedPage`):
- `page_number`
- `content` (finaler Text)
- `raw_content` (Rohtext zur Nachvollziehbarkeit)
- `content_source` (`text | ocr | vision | llm_text`)
- `sections`, `paragraph_numbers`
- `has_table`, `has_list`
- `internal_references`, `external_references`
- `continues_from_previous`, `continues_to_next`

## Abhaengigkeiten (optional)

- `OPENAI_API_KEY` fuer Vision-LLM
- `TESSERACT_CMD` fuer OCR (Tesseract-Installation)
- `pdfplumber` fuer Tabellen