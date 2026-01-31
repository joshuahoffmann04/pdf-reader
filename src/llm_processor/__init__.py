"""
LLM-based PDF Processor for RAG Applications

This module provides a Vision-LLM pipeline for converting PDF documents
into RAG-optimized natural language chunks with rich metadata.

Architecture:
1. Context Phase: Analyze entire PDF for document understanding
2. Page-by-Page Extraction: Process each page with full context
3. Chunk Generation: Create RAG-ready chunks with metadata
"""

from .models import (
    DocumentContext,
    PageContent,
    RAGChunk,
    ProcessingConfig,
)
from .vision_processor import VisionProcessor
from .chunk_generator import ChunkGenerator

__all__ = [
    "DocumentContext",
    "PageContent",
    "RAGChunk",
    "ProcessingConfig",
    "VisionProcessor",
    "ChunkGenerator",
]
