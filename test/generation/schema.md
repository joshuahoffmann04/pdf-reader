# Generation – Evaluations-Schema

Diese Test-Pipeline evaluiert die **Generation-Komponente** (RAG‑Antwort + Zitate) anhand eines kleinen Query-Sets.

Wichtig: Die Generation-Komponente ruft standardmaessig den **Retrieval-Service** ueber HTTP auf.

## Voraussetzungen

- Ollama laeuft (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`)
- Retrieval-Service laeuft und ist mit Chunks ingested (`RETRIEVAL_BASE_URL`)

## Input

### Queries (JSON)

Datei: `test/generation/queries.json`

Schema:

```json
{
  "default_mode": "hybrid",
  "queries": [
    {
      "id": "g1",
      "query": "…",
      "mode": "hybrid",
      "expected_answer_contains": ["..."],
      "expected_missing_info": false,
      "expected_citation_chunk_ids": ["..."],
      "expected_page_numbers": [1, 2],
      "min_citations": 1
    }
  ]
}
```

Bedeutung:
- `expected_answer_contains`: Substrings, die in der Antwort vorkommen sollen
- `expected_missing_info`: ob `missing_info` leer sein soll (false) oder gesetzt (true)
- `expected_citation_chunk_ids` / `expected_page_numbers`: optionale Checks fuer Zitate
- `min_citations`: Mindestanzahl an Zitaten

## Output

Im Output-Ordner (Standard: `test/generation/output/`) werden geschrieben:

- `report.json`
- `summary.json`

## Report (Kurzuebersicht)

Wichtige Felder in `summary.json`:

- `pass`
- `answer_hit_rate`
- `citation_hit_rate`
- `missing_info_hit_rate`

