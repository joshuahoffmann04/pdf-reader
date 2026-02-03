# Retrieval Service

Hybrid-Retrieval für RAG: BM25 (klassisch), Vektor-Retrieval (Ollama + ChromaDB) und Hybrid-Ranking (RRF). Das Modul bietet drei dedizierte Endpunkte und erzeugt LLM-taugliche Kontextstrings.

## Struktur

```
retrieval/
├── app.py           # FastAPI OpenAPI Service
├── config.py        # Service-Konfiguration
├── service.py       # Ingest + Retrieval
├── storage.py       # Chunk-Store (JSONL)
├── bm25_index.py    # BM25 Index
├── vector_index.py  # ChromaDB + Embedding
├── embedder.py      # Ollama Embedder
├── hybrid.py        # RRF Fusion
├── models.py        # Datenmodelle + Request/Response
└── README.md
```

## Service starten

```bash
python scripts/bm25_service.py
```

Die drei dedizierten Services laufen auf dem gleichen App-Code; du kannst für unterschiedliche Ports die jeweiligen Scripts nutzen:

```bash
python scripts/bm25_service.py
python scripts/vector_service.py
python scripts/hybrid_service.py
```

## API

- `GET /health` → `{ "status": "ok" }`
- `POST /ingest` → speichert Chunks + baut Indizes
- `POST /retrieve/bm25` → BM25
- `POST /retrieve/vector` → Vektor
- `POST /retrieve/hybrid` → RRF-Hybrid
- `GET /documents` → Übersicht über ingestete Dokumente

### Ingest Request

```json
{
  "document_id": "example",
  "chunks": [
    { "chunk_id": "example_chunk_0001", "text": "...", "metadata": {"page_numbers": [1]} }
  ]
}
```

### Retrieval Request

```json
{ "query": "Was ist die Regelstudienzeit?", "top_k": 5, "filters": {"document_id": "example"} }
```

### Retrieval Response (vereinheitlicht)

```json
{
  "query": "...",
  "mode": "bm25",
  "results": [
    { "chunk_id": "...", "score": 0.42, "text": "...", "metadata": {"page_numbers": [1]} }
  ],
  "context_text": "..."
}
```

## Persistenz

- Standard-Ordner: `data/retrieval`
- Chunks: `data/retrieval/chunks.jsonl`
- ChromaDB Persistenz: `data/retrieval` (Standardpfad von ChromaDB)

## Konfiguration

```python
from retrieval.config import RetrievalConfig

config = RetrievalConfig(
    data_dir="data/retrieval",
    max_context_tokens=1024,
    bm25_k1=1.5,
    bm25_b=0.75,
    rrf_k=60,
)
```

## Tests

```bash
pytest tests/test_retrieval_service.py tests/test_storage_paths.py -v
```
