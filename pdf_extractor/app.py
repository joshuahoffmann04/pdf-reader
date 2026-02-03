from fastapi import FastAPI, HTTPException

from config import ExtractorConfig
from models import ExtractRequest, ExtractResponse
from service import ExtractionService


def create_app(config: ExtractorConfig | None = None) -> FastAPI:
    service = ExtractionService(config=config)
    app = FastAPI(
        title="PDF Extractor Service",
        version="1.0.0",
        description="PDF extraction via OpenAI Vision API.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(request: ExtractRequest) -> ExtractResponse:
        try:
            result, document_id, output_path = service.extract_and_save(request.pdf_path)
            return ExtractResponse(
                document_id=document_id,
                output_path=output_path,
                pages=len(result.pages),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
