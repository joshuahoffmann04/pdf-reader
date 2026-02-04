from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkInput(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    chunks: list[ChunkInput] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    chunks_ingested: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    filters: Optional[dict[str, Any]] = None


class RetrievalHit(BaseModel):
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class RetrievalResponse(BaseModel):
    query: str
    mode: str
    results: list[RetrievalHit] = Field(default_factory=list)
    context_text: str


class DocumentSummary(BaseModel):
    document_id: str
    chunk_count: int
