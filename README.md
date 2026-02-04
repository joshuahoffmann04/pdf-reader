# PDF Reader Pipeline

Dieses Repository enthaelt vier Komponenten:
- `pdf_extractor/` fuer PDF-Extraktion (Text, Tabellen, OCR, optional LLM)
- `chunking/` fuer satzbasiertes Chunking
- `retrieval/` fuer BM25, Vektor und Hybrid Retrieval
- `generation/` fuer Chat-Generierung mit Ollama (RAG)

Ziel: eine robuste, nachvollziehbare Pipeline von PDF -> Extraktion -> Chunking -> Retrieval -> Chat.

## Schnellstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Umgebungsvariablen

- `OPENAI_API_KEY` (optional, aber empfohlen fuer volle Extractor-Pipeline)
- `TESSERACT_CMD` (optional, falls tesseract nicht im PATH ist)
- OCR (optional): Tesseract installieren und `pytesseract` via requirements aktivieren
- Retrieval/Generation (optional): `ollama` installieren, `ollama serve` starten, Modell pullen

## Scripts (genau 3)

- `scripts/pdf_extractor_service.py` -> Extractor (CLI oder API)
- `scripts/chunking_service.py` -> Chunking API
- `scripts/retrieval_service.py` -> Retrieval API

### Extractor CLI (einfachste Nutzung)

```bash
python scripts/pdf_extractor_service.py --pdf pdfs/2-aend-19-02-25_msc-computer-science_lese.pdf
```

Optional mit Output-Datei:

```bash
python scripts/pdf_extractor_service.py --pdf pdfs/2-aend-19-02-25_msc-computer-science_lese.pdf --output data/output.json
```

### Extractor API

```bash
python scripts/pdf_extractor_service.py --serve
```

API:
- `GET /health`
- `POST /extract` mit `{ "pdf_path": "pdfs/..." }`

### Chunking API

```bash
python scripts/chunking_service.py
```

### Retrieval API

```bash
python scripts/retrieval_service.py
```

### Generation (Chat) API

```bash
uvicorn generation.app:app --port 8003
```

### Generation (Chat) CLI

```bash
python -m generation.cli
```

## Tests

### Extractor Pipeline Test (M.Sc. Informatik)

```bash
pytest tests/test_extractor_pipeline.py -v
```

Hinweise:
- Dieser Test nutzt standardmaessig die volle Extractor-Pipeline.
- Fuer LLM-Context brauchst du `OPENAI_API_KEY`.
- OCR ist optional, wird aber genutzt wenn installiert.

### Weitere Tests

```bash
pytest -v
```

## Troubleshooting

- Kein LLM-Output: `OPENAI_API_KEY` fehlt -> Extractor faellt automatisch auf Text-only zurueck.
- OCR funktioniert nicht: `pytesseract` installieren und Tesseract-Binary verfuegbar machen.
- Retrieval Embeddings: `ollama serve` starten und `ollama pull nomic-embed-text`.
- Generation Chat: `ollama serve` starten und `ollama pull llama3.1:latest`.
