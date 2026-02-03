from fastapi import FastAPI, HTTPException

from .models import ChunkRequest, ChunkResponse
from .service import ChunkingService


def create_app() -> FastAPI:
    service = ChunkingService()
    app = FastAPI(
        title="Chunking Service",
        version="1.0.0",
        description="Sentence-based chunking service.",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chunk", response_model=ChunkResponse)
    def chunk(request: ChunkRequest) -> ChunkResponse:
        try:
            result, output_path = service.chunk_and_save(request.extraction_path)
            return ChunkResponse(
                document_id=result.document_id,
                output_path=output_path,
                total_chunks=result.total_chunks,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
