# Generation

## Schnellstart (Service)

Voraussetzungen:
- Ollama laeuft (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`)
- Retrieval-Service laeuft und ist ingested (`RETRIEVAL_BASE_URL`)

```powershell
uvicorn generation.app:app --host 127.0.0.1 --port 8003
```

Endpoints:
- `GET /health`
- `POST /generate`

Request:

```json
{ "query": "Welche Voraussetzungen gelten?", "mode": "hybrid" }
```

Response:

```json
{
  "answer": "...",
  "citations": [
    {"chunk_id": "...", "page_numbers": [12], "snippet": "...", "score": 0.12}
  ],
  "missing_info": "",
  "metadata": {"mode": "hybrid", "selected_chunks": 5}
}
```

## Schnellstart (Python)

```python
from generation import GenerationConfig, GenerationService
from generation.models import GenerateRequest

service = GenerationService(GenerationConfig.from_env())
resp = service.generate(GenerateRequest(query="Welche Voraussetzungen gelten?", mode="hybrid"))
print(resp.model_dump())
```

## Was passiert (Kurzfassung)

1. **Retrieval**: `candidate_top_k` Treffer vom Retrieval-Service holen.
2. **Kontextaufbau**: Chunks so selektieren, dass das Token-Budget passt.
3. **LLM**: Ollama erzeugt eine JSON-Antwort mit Zitaten.
4. **Citations**: Zitate werden gegen die im Kontext enthaltenen Chunks normalisiert.

## Konfiguration (.env)

Wichtige Variablen:

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

## Hinweis zu Token-Budgets

`GENERATION_MAX_CONTEXT_TOKENS` ist das Budget fuer Systemprompt + Userprompt + Kontext.
Die Komponente selektiert Chunks so, dass das Budget nicht ueberschritten wird.
