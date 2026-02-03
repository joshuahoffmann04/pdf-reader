# Chunking Service

Satzbasiertes Sliding-Window-Chunking für die RAG-Pipeline. Das Modul nimmt die JSON-Ausgabe des PDF-Extractors und erzeugt überlappende Chunks mit Metadaten, optimiert für LLM-Kontextfenster.

## Struktur

```
chunking/
├── app.py              # FastAPI OpenAPI Service
├── config.py           # Service-Konfiguration
├── service.py          # Chunking-Orchestrierung
├── storage.py          # Persistenz (data/chunking/<document_id>/chunks/...)
├── models.py           # Datenmodelle + Request/Response
├── chunker.py          # Satzbasierter Chunker
├── sentence_splitter.py
├── token_counter.py
└── README.md
```

## Quick Start (Python)

```python
from chunking import DocumentChunker, ChunkingConfig
from pdf_extractor import ExtractionResult

result = ExtractionResult.load("data/pdf_extractor/example/extraction/example_20240101_120000.json")
chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
chunks = chunker.chunk(result)
chunks.save("chunks.json")
```

## Service starten

```bash
python scripts/chunking_service.py
```

### API

- `GET /health` → `{ "status": "ok" }`
- `POST /chunk` → chunked eine Extraktionsdatei

**Request**
```json
{ "extraction_path": "data/pdf_extractor/<document_id>/extraction/<file>.json" }
```

**Response**
```json
{
  "document_id": "<document_id>",
  "output_path": "data/chunking/<document_id>/chunks/<file>.json",
  "total_chunks": 57
}
```

## Persistenz

- Standard-Ordner: `data/chunking`
- Ablagepfad: `data/chunking/<document_id>/chunks/<document_id>_<timestamp>.json`

Die Pfade werden von `ChunkingStorage` erzeugt und sind deterministisch pro Dokument und Zustand.

## Konfiguration

```python
from chunking.config import ChunkingServiceConfig

config = ChunkingServiceConfig(
    data_dir="data/chunking",
    chunking=ChunkingConfig(max_chunk_tokens=512, overlap_tokens=100),
)
```

## Tests

```bash
pytest tests/test_chunker.py tests/test_chunking_models.py -v
```
