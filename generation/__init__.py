"""
Generation component for RAG pipelines.

Uses retrieval context and Ollama to produce JSON answers with citations.
"""

__version__ = "1.0.0"

from .config import GenerationConfig
from .models import GenerateRequest, GenerateResponse, Citation
from .service import GenerationService

__all__ = [
    "__version__",
    "GenerationConfig",
    "GenerateRequest",
    "GenerateResponse",
    "Citation",
    "GenerationService",
]
