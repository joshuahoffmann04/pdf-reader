# Retrieval – Evaluations-Schema

Diese Test-Pipeline evaluiert die **Retrieval-Komponente** (BM25, Vektor, Hybrid) auf einem Chunk-Export.

Ziel: sicherstellen, dass relevante Chunks fuer typische Queries in den Top‑K Ergebnissen landen.

## Input

### Chunks (Pfad)

Die Pipeline erwartet eine JSON-Datei im Format von `chunking.ChunkingResult`, z. B.:

- `test/chunking/output/chunks.json`
- `data/chunking/<document_id>/chunks/<timestamp>.json`

In `test/retrieval/main.py` ist der Pfad als `CHUNKS_PATH` hinterlegt.

### Queries (JSON)

Datei: `test/retrieval/queries.json`

Schema:

```json
{
  "default_top_k": 5,
  "default_filters": { "document_id": "<document_id>" },
  "queries": [
    {
      "id": "q1",
      "query": "…",
      "top_k": 5,
      "filters": { "document_id": "<document_id>" },
      "expected_chunk_ids": ["..."],
      "expected_text_contains": ["..."]
    }
  ]
}
```

Bedeutung:
- `expected_chunk_ids`: mindestens einer dieser Chunk-IDs soll im Ranking auftauchen
- `expected_text_contains`: optionaler Plausibilitaetscheck (Substring-Hit im Result-Text)

## Output

Im Output-Ordner (Standard: `test/retrieval/output/`) werden pro Modus geschrieben:

- `report_bm25.json`, `summary_bm25.json`
- `report_vector.json`, `summary_vector.json`
- `report_hybrid.json`, `summary_hybrid.json`

## Report (Kurzuebersicht)

Wichtige Felder in `summary_<mode>.json`:

- `pass`
- `hit_rate` (Hit@K)
- `mrr` (Mean Reciprocal Rank)
- `text_hit_rate` (Substring-Hit-Rate)

