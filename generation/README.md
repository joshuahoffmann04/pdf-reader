# Generation Service

Chat generation with retrieval-augmented context (RAG) using Ollama.

## Voraussetzungen
- Retrieval-Service laeuft auf `http://localhost:8000`
- Ollama laeuft auf `http://localhost:11434`
- Modell: `llama3.1:latest`

## API starten

```bash
uvicorn generation.app:app --port 8003
```

Health:
```bash
curl.exe http://localhost:8003/health
```

Generate:
```bash
curl.exe -X POST http://localhost:8003/generate `
  -H "Content-Type: application/json" `
  -d "{""query"": ""Regelstudienzeit Informatik"", ""mode"": ""hybrid""}"
```

## CLI Chat (Terminal)

```bash
python -m generation.cli
```

Die CLI gibt JSON aus (answer, citations, missing_info).

## Konfiguration (.env)

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
RETRIEVAL_BASE_URL=http://localhost:8000
GENERATION_RETRIEVAL_MODE=hybrid
GENERATION_MAX_CONTEXT_TOKENS=2048
GENERATION_OUTPUT_TOKENS=512
GENERATION_CANDIDATE_TOP_K=30
GENERATION_TEMPERATURE=0.2
```

## Dynamisches Top-K

Die Komponente holt viele Treffer (candidate_top_k) und packt so viele in den Kontext,
wie in das Token-Budget passen (Systemprompt + Userfrage + Kontext <= max_context_tokens).
