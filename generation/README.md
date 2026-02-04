# Generation

Chat generation with retrieval-augmented context using Ollama. The component
fetches retrieval results, builds a context window within a token budget, and
returns a JSON answer with citations.

## Purpose

- Produce grounded answers from retrieved chunks
- Emit JSON with citations and missing-info markers
- Respect a fixed token budget with dynamic top-k selection

## Structure

```
generation/
|-- app.py           # FastAPI service
|-- config.py        # Configuration (env driven)
|-- service.py       # Orchestration
|-- context_builder.py
|-- ollama_client.py
|-- http_client.py
|-- prompts.py
|-- models.py
|-- cli.py
|-- README.md
```

## Usage (Python)

```python
from generation import GenerationConfig, GenerationService
from generation.models import GenerateRequest

config = GenerationConfig.from_env()
service = GenerationService(config)
response = service.generate(GenerateRequest(query="Regelstudienzeit Informatik"))
print(response.model_dump())
```

## API

The component exposes a FastAPI app: `generation.app:create_app` (or `generation.app:app`).

Endpoints:
- `GET /health`
- `POST /generate`

Request:
```json
{ "query": "Regelstudienzeit Informatik", "mode": "hybrid" }
```

Response:
```json
{
  "answer": "...",
  "citations": [
    {"chunk_id": "...", "page_numbers": [12], "snippet": "..."}
  ],
  "missing_info": "",
  "metadata": {"mode": "hybrid", "selected_chunks": 5}
}
```

## CLI

```bash
python -m generation.cli
```

The CLI prints JSON per question (answer, citations, missing_info).

## Configuration (.env)

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

## Dynamic Top-K

The component first retrieves `candidate_top_k` chunks and then selects as many
as fit into the token budget (system + user + context <= max_context_tokens).
