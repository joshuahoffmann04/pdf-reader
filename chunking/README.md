# Chunking

Sentence-aligned, sliding-window chunking for the RAG pipeline. The component
consumes the JSON output from `pdf_extractor` and produces overlapping chunks
with metadata optimized for retrieval.

## Purpose

- Stable, sentence-aligned chunks for downstream retrieval
- Configurable token window and overlap
- Rich metadata for traceability and navigation

## Structure

```
chunking/
|-- app.py              # FastAPI service
|-- config.py           # Service configuration
|-- service.py          # Chunking orchestration
|-- storage.py          # Persistence (data/chunking/<document_id>/chunks/...)
|-- models.py           # Data models + request/response
|-- chunker.py          # Chunking logic
|-- sentence_splitter.py
|-- token_counter.py
|-- README.md
```

## Usage (Python)

```python
from chunking import DocumentChunker, ChunkingConfig
from pdf_extractor import ExtractionResult

result = ExtractionResult.load(
    "data/pdf_extractor/example/extraction/example_20240101_120000.json"
)
chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
chunks = chunker.chunk(result)
chunks.save("chunks.json")
```

## Service (API)

The component exposes a FastAPI app: `chunking.app:create_app` (or `chunking.app:app`).

Endpoints:
- `GET /health`
- `POST /chunk` with `{ "extraction_path": "data/pdf_extractor/<document_id>/extraction/<file>.json" }`

Response:
```json
{
  "document_id": "<document_id>",
  "output_path": "data/chunking/<document_id>/chunks/<file>.json",
  "total_chunks": 57
}
```

## Output Format

Top-level:
- `source_file`
- `document_id`
- `config`
- `chunks` (list of Chunk)
- `stats`
- `created_at`

Chunk:
- `chunk_id`
- `text`
- `token_count`
- `metadata` (document_id, page_numbers, neighbor ids, etc.)

## Configuration

```python
from chunking.config import ChunkingServiceConfig
from chunking.models import ChunkingConfig

config = ChunkingServiceConfig(
    data_dir="data/chunking",
    chunking=ChunkingConfig(max_chunk_tokens=512, overlap_tokens=100),
)
```

## Token Counting

Token counting uses `tiktoken` with the `cl100k_base` encoding as a conservative
approximation. This keeps chunk sizes safely within limits for most local LLMs.
