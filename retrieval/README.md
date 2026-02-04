# Retrieval

## Schnellstart (Service)

```powershell
uvicorn retrieval.app:app --host 127.0.0.1 --port 8000
```

Endpoints:
- `GET /health`
- `GET /documents`
- `POST /ingest`
- `POST /retrieve/bm25`
- `POST /retrieve/vector`
- `POST /retrieve/hybrid`

### Ingest

Request:

```json
{
  "document_id": "example",
  "chunks": [
    {"chunk_id": "example_chunk_0001", "text": "...", "metadata": {"page_numbers": [1]}}
  ]
}
```

Response:

```json
{ "document_id": "example", "chunks_ingested": 1 }
```

### Retrieve

Request:

```json
{ "query": "Welche Voraussetzungen gelten?", "top_k": 5, "filters": {"document_id": "example"} }
```

Response (verkuerzt):

```json
{
  "query": "...",
  "mode": "bm25",
  "results": [
    {"chunk_id": "...", "score": 0.42, "text": "...", "metadata": {"page_numbers": [1]}}
  ],
  "context_text": "..."
}
```

## Schnellstart (Python)

```python
from retrieval import RetrievalConfig
from retrieval.service import RetrievalService
from retrieval.vector_index import VectorIndex

vector = VectorIndex(persist_directory=":memory:", collection_name="demo")
service = RetrievalService(RetrievalConfig(data_dir="data/retrieval"), vector)
```

Hinweis: In der Praxis wird der Service meist ueber FastAPI genutzt.

## Was passiert (Kurzfassung)

- **BM25**: robuste lexikalische Suche (z. B. Abschnittsverweise, Kennungen)
- **Vector**: semantische Suche ueber Embeddings (Ollama)
- **Hybrid (RRF)**: kombiniert BM25 + Vector per Reciprocal-Rank Fusion
- **Kontext**: `context_text` wird aus Treffern innerhalb eines Token-Budgets aufgebaut

## Persistenz

Standardpfade (konfigurierbar):
- Chunk-Store: `data/retrieval/chunks.jsonl`
- ChromaDB (Vektorindex): `data/retrieval/chroma`

## Konfiguration

`retrieval/config.py`:
- `data_dir`: Basisverzeichnis
- `collection_name`: Chroma-Collection
- `bm25_k1`, `bm25_b`: BM25-Parameter
- `max_context_tokens`: Token-Budget fuer `context_text`
- `rrf_k`: Parameter fuer RRF
