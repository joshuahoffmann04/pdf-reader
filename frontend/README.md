# MARley Frontend

Ein schlankes Web‑Frontend für den Chatbot. Es startet eine lokale Web‑App, lädt automatisch den aktuellsten Chunk‑Export und verbindet sich direkt mit Ollama.

## Start

```powershell
python frontend/marley_app.py
```

Der Browser öffnet sich automatisch. Ollama muss separat laufen (`ollama serve`).

## Konfiguration (optional, via .env)

- `MARLEY_CHUNKS_PATH` – Pfad zur Chunk‑Datei (`*.json`)  
  Default: `test/chunking/output/chunks.json` oder letztes in `data/chunking/**/chunks/*.json`
- `MARLEY_HOST` / `MARLEY_PORT`
- `MARLEY_TOP_K`
- `MARLEY_MAX_CONTEXT_TOKENS`
- `MARLEY_OUTPUT_TOKENS`
- `MARLEY_EMBED_MODEL` (Default: `nomic-embed-text`)
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
