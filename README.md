# PDF to RAG Pipeline

A Vision-LLM powered pipeline for converting PDF documents into RAG-optimized natural language chunks.

## Overview

This tool uses Vision-capable LLMs (Claude, GPT-4V) to extract content from PDF documents and convert it into chunks optimized for Retrieval-Augmented Generation (RAG) systems.

**Key Features:**
- Converts tables, lists, and structured content to natural language
- Preserves document structure (§ sections, paragraphs)
- Generates rich metadata for filtered retrieval
- Compatible with LangChain, LlamaIndex, and Haystack
- Supports both Anthropic (Claude) and OpenAI (GPT-4V) APIs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Processing Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Context Analysis                                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐           │
│  │   PDF    │───▶│  Sample  │───▶│   Vision LLM    │           │
│  │          │    │  Pages   │    │ (Context Prompt) │           │
│  └──────────┘    └──────────┘    └────────┬─────────┘           │
│                                           │                      │
│                                           ▼                      │
│                                  DocumentContext                 │
│                                  - document_type                 │
│                                  - title, institution            │
│                                  - abbreviations                 │
│                                  - key_terms                     │
│                                                                  │
│  Phase 2: Page-by-Page Extraction                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐           │
│  │  Page N  │───▶│  Image   │───▶│   Vision LLM    │           │
│  │          │    │          │    │ (+ Context)      │           │
│  └──────────┘    └──────────┘    └────────┬─────────┘           │
│                                           │                      │
│                                           ▼                      │
│                                     ExtractedPage                │
│                                  - natural language text         │
│                                  - section info                  │
│                                  - references                    │
│                                                                  │
│  Phase 3: Chunk Generation                                       │
│  ┌──────────────┐    ┌──────────────────┐                       │
│  │ All Pages    │───▶│  Smart Chunking  │                       │
│  │              │    │  + Metadata      │                       │
│  └──────────────┘    └────────┬─────────┘                       │
│                               │                                  │
│                               ▼                                  │
│                          RAGChunk (JSONL)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/joshuahoffmann04/pdf-reader.git
cd pdf-reader

# Install dependencies
pip install -r requirements.txt

# For Anthropic Claude
pip install anthropic

# For OpenAI GPT-4V
pip install openai
```

## Quick Start

### Command Line

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Process a PDF
python scripts/process_for_rag.py document.pdf -o output/

# Estimate cost first
python scripts/process_for_rag.py document.pdf --estimate-cost

# Use a cheaper model
python scripts/process_for_rag.py document.pdf --model claude-3-haiku-20240307
```

### Python API

```python
from src.llm_processor import (
    VisionProcessor,
    ChunkGenerator,
    ProcessingConfig,
)

# Configure
config = ProcessingConfig(
    model="claude-sonnet-4-20250514",
    target_chunk_size=500,
)

# Process document
processor = VisionProcessor(config=config)
extraction = processor.process_document("document.pdf")

# Generate chunks
generator = ChunkGenerator(config=config)
result = generator.generate_from_extraction(extraction, "document-name")

# Export for different frameworks
langchain_docs = result.export_chunks_langchain()
llamaindex_nodes = result.export_chunks_llamaindex()

# Or export to file
result.export_chunks_jsonl("output.jsonl")
```

## Output Format

### RAGChunk Structure

Each chunk is optimized for RAG retrieval with rich metadata:

```python
RAGChunk(
    id="doc-name-§10-abc123",
    text="§10 Module und Leistungspunkte: Ein Modul ist eine...",
    metadata=ChunkMetadata(
        source_document="Pruefungsordnung_2024",
        source_pages=[9, 10],
        document_type="pruefungsordnung",
        section_number="§10",
        section_title="Module und Leistungspunkte",
        chapter="II. Studienbezogene Bestimmungen",
        chunk_type="section",
        topics=["Module", "Leistungspunkte"],
        keywords=["Modul", "LP", "ECTS"],
        related_sections=["§7", "§11"],
        institution="Philipps-Universität Marburg",
        degree_program="Mathematik B.Sc.",
    )
)
```

### Export Formats

**LangChain:**
```python
chunk.to_langchain_document()
# Returns: {"page_content": "...", "metadata": {...}}
```

**LlamaIndex:**
```python
chunk.to_llamaindex_node()
# Returns: {"id_": "...", "text": "...", "metadata": {...}}
```

**Haystack:**
```python
chunk.to_haystack_document()
# Returns: {"id": "...", "content": "...", "meta": {...}}
```

**JSONL (for file storage):**
```python
chunk.to_jsonl_entry()
# Returns JSON string with all fields
```

## Data Models

### DocumentContext
Document-level metadata extracted during context analysis:
- `document_type`: pruefungsordnung, modulhandbuch, etc.
- `title`: Official document title
- `institution`: University/organization
- `abbreviations`: List of abbreviations and expansions
- `key_terms`: Important domain terminology
- `chapters`: Document structure

### ExtractedPage
Content extracted from each page:
- `content`: Natural language text
- `sections`: Section markers (§10, §11, etc.)
- `has_table`, `has_list`: Content classification
- `continues_from_previous`, `continues_to_next`: Pagination info

### RAGChunk
Final chunk for RAG ingestion:
- `id`: Unique identifier
- `text`: Natural language content
- `metadata`: Rich metadata for filtering and context

## Cost Estimation

| Model | ~Cost per 50 pages |
|-------|-------------------|
| claude-sonnet-4-20250514 | $0.15 - $0.25 |
| claude-3-haiku | $0.02 - $0.04 |
| gpt-4o | $0.12 - $0.20 |
| gpt-4o-mini | $0.01 - $0.02 |

## Configuration Options

```python
ProcessingConfig(
    # API settings
    api_provider="anthropic",  # or "openai"
    model="claude-sonnet-4-20250514",
    temperature=0.0,  # Deterministic output

    # Chunking settings
    target_chunk_size=500,   # Target chars per chunk
    max_chunk_size=1000,     # Maximum chars
    chunk_overlap=50,        # Overlap between chunks

    # Processing options
    expand_abbreviations=True,
    merge_cross_page_content=True,

    # Output settings
    output_format="jsonl",   # or "json"
)
```

## Project Structure

```
pdf-reader/
├── src/
│   └── llm_processor/
│       ├── __init__.py
│       ├── models.py          # Pydantic data models
│       ├── vision_processor.py # Main LLM processor
│       ├── chunk_generator.py  # Chunking logic
│       ├── pdf_to_images.py   # PDF rendering
│       ├── prompts.py         # LLM prompts
│       └── README.md
├── scripts/
│   └── process_for_rag.py     # CLI tool
├── archived/                   # Old approach (kept for reference)
├── pdfs/                       # Test PDFs
├── requirements.txt
└── README.md
```

## License

MIT License
