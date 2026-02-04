# PDF Extractor

Komponente zur praezisen Extraktion von Inhalten aus PDF-Dokumenten. Sie erzeugt eine strukturierte JSON-Ausgabe pro Seite und ist die erste Stufe der Pipeline (vor Chunking und Retrieval).

## Zweck

- Vollstaendige, seitenweise Extraktion mit nachvollziehbarer Herkunft
- Prioritaet: text-native Extraktion (verlustarm), mit optionalem OCR und LLM-Fallback
- Strukturierte Ausgabe mit Metadaten fuer spaetere Verarbeitung

## Architektur und Ablauf

1. **Text-Extraktion (PyMuPDF)**  
   Selektierbarer Text wird blockweise gelesen und in natuerlicher Reihenfolge zusammengefuehrt.
2. **Tabellen-Extraktion (optional)**  
   Bei Bedarf werden Tabellen zusaetzlich als Text eingefuegt (pdfplumber).
3. **OCR-Fallback (optional)**  
   Fuer gescannte Seiten wird Tesseract verwendet.
4. **Vision-Fallback (optional, LLM)**  
   Falls Text/OCR nicht ausreichend sind, wird die Seite via Vision-LLM extrahiert.
5. **Validierungs-Gates**  
   Inhalte werden gegen Raw-Text geprueft, um Informationsverlust zu verhindern.

## Struktur

```
pdf_extractor/
|-- app.py              # FastAPI Service
|-- config.py           # Service-Konfiguration
|-- service.py          # Extraktions-Orchestrierung
|-- storage.py          # Persistenz (data/pdf_extractor/<doc>/extraction/...)
|-- extractor.py        # Pipeline-Logik
|-- text_extractor.py   # Text-native Extraktion (PyMuPDF)
|-- table_extractor.py  # Tabellen (pdfplumber)
|-- validation.py       # Coverage- und LLM-Validierung
|-- models.py           # Datenmodelle
|-- prompts.py          # LLM-Prompts
|-- pdf_to_images.py    # PDF -> Image
```

## Schnellstart (Python)

```python
from pdf_extractor import PDFExtractor, ProcessingConfig

config = ProcessingConfig(
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

extractor = PDFExtractor(config=config)
result = extractor.extract("pdfs/example.pdf")
result.save("output.json")
```

## Service (API)

Die Komponente stellt eine FastAPI-App bereit: `pdf_extractor.app:create_app` bzw. `pdf_extractor.app:app`.
Integration erfolgt durch Einbindung in die uebergeordnete Service-Schicht.

Endpoints:
- `GET /health`
- `POST /extract` mit `{ "pdf_path": "pdfs/..." }`

## Konfiguration

Die wichtigsten Optionen in `ProcessingConfig`:

- `extraction_mode`: `text | vision | hybrid`
- `use_llm`: LLM fuer Vision-Extraction aktivieren
- `context_mode`: `llm_text | llm_vision | heuristic`
- `table_extraction`: Tabellen zusaetzlich extrahieren
- `ocr_enabled`: OCR-Fallback aktivieren
- `ocr_before_vision`: OCR vor Vision versuchen
- `layout_mode`: `simple | columns`
- `enforce_text_coverage`: harte Coverage-Pruefung
- `min_token_recall`, `min_number_recall`: Coverage-Schwellen

## Ausgabeformat (ExtractionResult)

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
