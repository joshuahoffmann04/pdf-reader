# PDF Reader Pipeline

## Schnellstart

### 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional: `.env` anlegen (siehe `.env.example`).

### 2) Ollama starten

```powershell
ollama serve
```

### 3) PDF extrahieren + chunken

```powershell
python run_extract_chunk.py --pdf pdfs\beispiel.pdf
```

Das erzeugt:
- Extraktion unter `data/pdf_extractor/<document_id>/extraction/...json`
- Chunks unter `data/chunking/<document_id>/chunks/...json`

### 4) MARley starten

```powershell
python frontend\marley_app.py
```

MARley laedt automatisch die aktuellste Chunk-Datei (oder `MARLEY_CHUNKS_PATH`).

---

Dieses Repository implementiert eine komplette Pipeline fuer ein RAG-System:

1. `pdf_extractor/` - PDF -> strukturierte Seiten-Extraktion (Text, optional Tabellen/OCR/LLM)
2. `chunking/` - Extraktion -> satzbasierte, ueberlappende Chunks
3. `retrieval/` - BM25, Vektor und Hybrid Retrieval ueber Chunks
4. `generation/` - RAG-Antwort (JSON) mit Zitaten ueber Ollama
5. `frontend/` - MARley: Browser-Chatbot (lokal)

Hinweis: Ersetze `pdfs\beispiel.pdf` durch eine Datei aus `pdfs/` oder eine eigene PDF.
Die PDFs unter `pdfs/` dienen als Testdaten und bleiben bewusst im Repo.

## Services (optional)

Jede Komponente kann auch als FastAPI-Service gestartet werden:

```powershell
uvicorn pdf_extractor.app:app --host 127.0.0.1 --port 8001
uvicorn chunking.app:app --host 127.0.0.1 --port 8002
uvicorn retrieval.app:app --host 127.0.0.1 --port 8000
uvicorn generation.app:app --host 127.0.0.1 --port 8003
```

Hinweis:
- `generation/` ruft standardmaessig `retrieval/` ueber `RETRIEVAL_BASE_URL` auf.
- `frontend/` nutzt die Chunk-Datei direkt (kein Retrieval-/Generation-Service notwendig).

## Evaluation / Tests

Unter `test/` liegen reproduzierbare Testpipelines je Komponente.
Jede Pipeline hat ein `schema.md`, das Input/Output und die wichtigsten Metriken dokumentiert.

- `test/pdf_extractor/main.py` - PDF vs Referenz-Extraktion
- `test/chunking/main.py` - Chunking-Qualitaet (Coverage/Duplizierung/Metadaten)
- `test/retrieval/main.py` - Retrieval-Qualitaet (Hit@K/MRR)
- `test/generation/main.py` - Generation-Qualitaet (Answer/Citations/Missing-Info)
- `test/marley/main.py` - End-to-End MARley (lokale Indizes aus Chunk-Datei)

## Ordnerstruktur

- `pdfs/`: Test-PDFs (im Repo)
- `data/`: generierte Artefakte (nicht versioniert)
- `reference/`: manuelle Referenzen fuer Evaluation (versioniert)
- `test/**/output/`: Reports/Artefakte aus Evaluationslaeufen (nicht versioniert)

## Komponenten-README

Details pro Komponente:
- `pdf_extractor/README.md`
- `chunking/README.md`
- `retrieval/README.md`
- `generation/README.md`
- `frontend/README.md`
