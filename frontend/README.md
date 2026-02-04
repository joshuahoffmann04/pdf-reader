# Frontend (MARley)

## Schnellstart

Voraussetzung: Ollama laeuft (`ollama serve`).

```powershell
python frontend\marley_app.py
```

Der Browser oeffnet sich automatisch.

## Konfiguration (.env, optional)

- `MARLEY_CHUNKS_PATH` - Pfad zur Chunk-Datei (`*.json`)
  Default: `test/chunking/output/chunks.json` oder die neueste Datei in `data/chunking/**/chunks/*.json`
- `MARLEY_HOST`, `MARLEY_PORT`
- `MARLEY_RETRIEVAL_MODE` (`bm25|vector|hybrid`)
- `MARLEY_TOP_K`
- `MARLEY_MAX_CONTEXT_TOKENS`
- `MARLEY_OUTPUT_TOKENS`
- `MARLEY_EMBED_MODEL` (Default: `nomic-embed-text`)
- `MARLEY_RRF_K`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`

## Was passiert (Kurzfassung)

1. **Chunks laden**: MARley liest `chunking.ChunkingResult` aus der Chunk-Datei.
2. **Indizes bauen**:
   - BM25 wird lokal im Prozess aufgebaut.
   - Vektorindex wird lokal (Chroma in-memory) aufgebaut; Embeddings kommen von Ollama.
3. **Retrieval**: je nach Modus BM25/Vektor/Hybrid (RRF + deterministisches Reranking).
4. **Kontextaufbau**: es werden so viele Chunks selektiert, wie in das Token-Budget passen.
5. **LLM**: MARley ruft Ollama fuer die Antwortgenerierung auf und gibt JSON + Citations aus.

Hinweis: MARley nutzt keine separaten Retrieval-/Generation-Services.