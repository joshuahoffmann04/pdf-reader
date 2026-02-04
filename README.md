# PDF Reader Pipeline

Dieses Repository enthaelt vier Komponenten:
- `pdf_extractor/` fuer PDF-Extraktion (Text, Tabellen, OCR, optional LLM)
- `chunking/` fuer satzbasiertes Chunking
- `retrieval/` fuer BM25, Vektor und Hybrid Retrieval
- `generation/` fuer Chat-Generierung mit Ollama (RAG)

Ziel: eine robuste, nachvollziehbare Pipeline von PDF -> Extraktion -> Chunking -> Retrieval -> Chat.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Umgebungsvariablen (optional)

- `OPENAI_API_KEY` (LLM-vision im Extractor)
- `TESSERACT_CMD` (OCR via Tesseract)
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL` (Generation + Retrieval Embeddings)

## Evaluation

Die Evaluations-Harnesses liegen unter `test/`:

- `test/pdf_extractor/main.py` (PDF vs. Referenz-JSON)
- `test/chunking/main.py` (Coverage/Overlap/Metadaten)
- `test/retrieval/main.py` (Queries gegen BM25/Vector/Hybrid)
- `test/generation/main.py` (Antwort/Citations/Missing-Info)

Jede Harness hat eine `schema.md` mit dem erwarteten Input-Format.
