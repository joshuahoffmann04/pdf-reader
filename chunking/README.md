# Chunking

## Schnellstart (Python)

```python
from pdf_extractor import ExtractionResult
from chunking import DocumentChunker, ChunkingConfig

extraction = ExtractionResult.load(
    "data/pdf_extractor/<document_id>/extraction/<file>.json"
)

chunker = DocumentChunker(
    ChunkingConfig(
        max_chunk_tokens=512,
        overlap_tokens=100,
        min_chunk_tokens=50,
    )
)

chunking_result = chunker.chunk(extraction)
chunking_result.save("chunks.json")
```

## Schnellstart (API)

```powershell
uvicorn chunking.app:app --host 127.0.0.1 --port 8002
```

Endpoints:
- `GET /health`
- `POST /chunk`

Request:

```json
{ "extraction_path": "data/pdf_extractor/<document_id>/extraction/<file>.json" }
```

Response:

```json
{
  "document_id": "<document_id>",
  "output_path": "data/chunking/<document_id>/chunks/<file>.json",
  "total_chunks": 57
}
```

## Was passiert (Kurzfassung)

Die Komponente erstellt satzbasierte, ueberlappende Chunks mit Metadaten.
Ziel ist ein stabiler, gut durchsuchbarer Kontext fuer Retrieval und Generation.

## Output-Format

Top-Level (`ChunkingResult`):
- `source_file`
- `document_id`
- `config`
- `chunks` (Liste von Chunks)
- `stats`
- `created_at`

Pro Chunk:
- `chunk_id`
- `text`
- `token_count`
- `metadata` (u.a. `document_id`, `page_numbers`, `chunk_index`, `prev_chunk_id`, `next_chunk_id`)

## Konfiguration

Wichtige Parameter in `ChunkingConfig`:
- `max_chunk_tokens`: Ziel-Fenster (Tokens)
- `overlap_tokens`: Token-Ueberlappung zwischen Chunks
- `min_chunk_tokens`: Mindestgroesse

Service-Konfiguration (`ChunkingServiceConfig`) steuert u.a. den Output-Ordner:

```python
from chunking.config import ChunkingServiceConfig
from chunking.models import ChunkingConfig

cfg = ChunkingServiceConfig(
    data_dir="data/chunking",
    chunking=ChunkingConfig(max_chunk_tokens=512, overlap_tokens=100),
)
```

## Token Counting

Das Token-Counting verwendet `tiktoken` mit `cl100k_base` als konservative Approximation.
Das stellt sicher, dass Chunk-Groessen auch bei unterschiedlichen Modellen stabil bleiben.