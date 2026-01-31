"""
LLM-based PDF Processor for RAG Applications

This module provides a Vision-LLM pipeline for converting PDF documents
into RAG-optimized natural language chunks with rich metadata.

Architecture:
1. Context Phase: Analyze entire PDF for document understanding
2. Page-by-Page Extraction: Process each page with full context
3. Chunk Generation: Create RAG-ready chunks with metadata

Usage:
    from src.llm_processor import VisionProcessor, ProcessingConfig

    config = ProcessingConfig(model="claude-sonnet-4-20250514")
    processor = VisionProcessor(config=config)
    result = processor.process_document("document.pdf")

    # Export for different frameworks
    langchain_docs = result.export_chunks_langchain()
    llamaindex_nodes = result.export_chunks_llamaindex()
"""

from .models import (
    # Enums
    DocumentType,
    ChunkType,
    Language,
    # LLM extraction models
    Abbreviation,
    DocumentContext,
    ExtractedPage,
    SectionMarker,
    # RAG chunk models
    ChunkMetadata,
    RAGChunk,
    # Configuration
    ProcessingConfig,
    ProcessingResult,
)
from .vision_processor import VisionProcessor
from .chunk_generator import ChunkGenerator
from .pdf_to_images import PDFToImages, PageImage, estimate_api_cost

__all__ = [
    # Enums
    "DocumentType",
    "ChunkType",
    "Language",
    # LLM extraction models
    "Abbreviation",
    "DocumentContext",
    "ExtractedPage",
    "SectionMarker",
    # RAG chunk models
    "ChunkMetadata",
    "RAGChunk",
    # Configuration
    "ProcessingConfig",
    "ProcessingResult",
    # Processors
    "VisionProcessor",
    "ChunkGenerator",
    # PDF utilities
    "PDFToImages",
    "PageImage",
    "estimate_api_cost",
]

__version__ = "2.0.0"
