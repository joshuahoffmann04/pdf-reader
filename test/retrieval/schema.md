# Retrieval Evaluation Schema

The evaluation uses:
- A chunking output JSON (ChunkingResult)
- A query set JSON describing expected matches

## Query Set (JSON)

Top-level fields:
- `queries`: list of query objects
- `default_top_k` (optional, default: 5)
- `default_filters` (optional)

Each query object:
```
{
  "id": "q1",
  "query": "Was ist die Regelstudienzeit?",
  "expected_chunk_ids": ["doc_chunk_0003", "doc_chunk_0004"],
  "expected_text_contains": ["Regelstudienzeit"],
  "top_k": 5,
  "filters": {"document_id": "doc"}
}
```

Notes:
- `expected_chunk_ids` and `expected_text_contains` are optional; include at least one.
- `filters` are optional; used by BM25/vector/hybrid.
