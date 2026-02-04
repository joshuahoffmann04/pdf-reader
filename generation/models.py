from typing import Any, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: Optional[str] = None
    max_context_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class Citation(BaseModel):
    chunk_id: str
    page_numbers: list[int] = Field(default_factory=list)
    snippet: str = ""
    score: Optional[float] = None


class GenerateResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    missing_info: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
