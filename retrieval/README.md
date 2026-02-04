# Retrieval

Hybrid retrieval for the RAG pipeline. This component supports BM25, vector
search (Ollama embeddings + ChromaDB), and a hybrid RRF fusion. It exposes
a unified API and produces LLM-ready context strings.

## Purpose

- Reliable lexical retrieval (BM25)
- Semantic retrieval via embeddings
- Hybrid ranking with reciprocal-rank fusion (RRF)
- Consistent context assembly for generation

## Structure

```
retrieval/
|-- app.py           # FastAPI service
|-- config.py        # Service configuration
|-- service.py       # Ingest + retrieval orchestration
|-- storage.py       # Chunk store (JSONL)
|-- bm25_index.py    # BM25 index
|-- vector_index.py  # ChromaDB + embeddings
|-- embedder.py      # Ollama embedder
|-- hybrid.py        # RRF fusion
|-- models.py        # Data models + request/response
|-- README.md
```

## API

The component exposes a FastAPI app: `retrieval.app:create_app` (or `retrieval.app:app`).

Endpoints:
- `GET /health`
- `GET /documents`
- `POST /ingest`
- `POST /retrieve/bm25`
- `POST /retrieve/vector`
- `POST /retrieve/hybrid`

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

### Retrieval Response

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

## Persistence

- Chunk store: `data/retrieval/chunks.jsonl`
- ChromaDB persistence: `data/retrieval/chroma`

## Configuration

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

## Dependencies (optional)

- Ollama for embeddings
- ChromaDB for vector index persistence
