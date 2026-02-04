from fastapi import FastAPI, HTTPException

from .config import GenerationConfig
from .models import GenerateRequest, GenerateResponse
from .service import GenerationService


def create_app(config: GenerationConfig | None = None) -> FastAPI:
    service = GenerationService(config=config)
    app = FastAPI(
        title="Generation Service",
        version="1.0.0",
        description="Chat generation with retrieval-augmented context via Ollama.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        try:
            return service.generate(request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
