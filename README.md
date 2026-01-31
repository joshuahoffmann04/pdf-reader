# PDF to RAG Pipeline

A production-ready pipeline for converting PDF documents into RAG-optimized natural language chunks using OpenAI's GPT-4o Vision API.

## Features

- **Vision-LLM Processing**: Uses GPT-4o to extract and understand PDF content
- **Natural Language Output**: Converts tables, lists, and structured content to flowing text
- **Rich Metadata**: Section numbers, chapters, topics, keywords for filtered retrieval
- **Multi-Framework Export**: Compatible with LangChain, LlamaIndex, and Haystack
- **German Academic Documents**: Optimized for Prüfungsordnungen, Modulhandbücher, etc.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pdf-reader.git
cd pdf-reader

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"
```

### Usage

```bash
# Process a PDF (default: gpt-4o)
python scripts/process_for_rag.py document.pdf -o output/

# Estimate cost before processing
python scripts/process_for_rag.py document.pdf --estimate-cost

# Use cheaper model
python scripts/process_for_rag.py document.pdf --model gpt-4o-mini
```

### Python API

```python
from src.llm_processor import VisionProcessor, ChunkGenerator, ProcessingConfig

# Configure
config = ProcessingConfig(
    model="gpt-4o",
    target_chunk_size=500,
)

# Process document
processor = VisionProcessor(config=config)
extraction = processor.process_document("document.pdf")

# Generate chunks
generator = ChunkGenerator(config=config)
result = generator.generate_from_extraction(extraction, "my-document")

# Export for different frameworks
langchain_docs = result.export_chunks_langchain()
llamaindex_nodes = result.export_chunks_llamaindex()

# Or export to file
result.export_chunks_jsonl("output.jsonl")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF to RAG Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Context Analysis                                   │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   PDF    │───▶│ Sample Pages │───▶│   GPT-4o Vision │    │
│  │          │    │ (1, mid, end)│    │                 │    │
│  └──────────┘    └──────────────┘    └────────┬────────┘    │
│                                               │              │
│                                               ▼              │
│                                      DocumentContext         │
│                                      ├─ document_type        │
│                                      ├─ chapters             │
│                                      ├─ abbreviations        │
│                                      └─ key_terms            │
│                                                              │
│  Phase 2: Page-by-Page Extraction                            │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │ Page N   │───▶│ Page Image   │───▶│   GPT-4o Vision │    │
│  │          │    │              │    │   + Context     │    │
│  └──────────┘    └──────────────┘    └────────┬────────┘    │
│                                               │              │
│                                               ▼              │
│                                         ExtractedPage        │
│                                      ├─ natural language     │
│                                      ├─ section markers      │
│                                      └─ references           │
│                                                              │
│  Phase 3: Chunk Generation                                   │
│  ┌──────────────┐    ┌──────────────────┐                   │
│  │  All Pages   │───▶│  Smart Chunking  │                   │
│  │              │    │  + Metadata      │                   │
│  └──────────────┘    └────────┬─────────┘                   │
│                               │                              │
│                               ▼                              │
│                          RAGChunk[]                          │
│                      ├─ id, text                             │
│                      └─ metadata (section, topics, etc.)     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Output Format

### RAGChunk Structure

```python
RAGChunk(
    id="doc-name-§10-abc123",
    text="§10 Module und Leistungspunkte: Ein Modul umfasst...",
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
        institution="Philipps-Universität Marburg",
    )
)
```

### Export Formats

| Framework | Method | Format |
|-----------|--------|--------|
| LangChain | `chunk.to_langchain_document()` | `{"page_content": ..., "metadata": ...}` |
| LlamaIndex | `chunk.to_llamaindex_node()` | `{"id_": ..., "text": ..., "metadata": ...}` |
| Haystack | `chunk.to_haystack_document()` | `{"id": ..., "content": ..., "meta": ...}` |
| JSONL | `chunk.to_jsonl_entry()` | JSON string per line |

## Configuration

```python
ProcessingConfig(
    # Model (OpenAI GPT-4o variants)
    model="gpt-4o",              # Best quality
    # model="gpt-4o-mini",       # Faster, cheaper

    # Chunking
    target_chunk_size=500,       # Target chars per chunk
    max_chunk_size=1000,         # Maximum chars per chunk
    chunk_overlap=50,            # Overlap between chunks

    # Processing
    temperature=0.0,             # Deterministic output
    max_tokens_per_request=4096,
)
```

## Cost Estimation

| Model | ~Cost per 50 pages |
|-------|-------------------|
| gpt-4o | $0.30 - $0.50 |
| gpt-4o-mini | $0.02 - $0.04 |

Use `--estimate-cost` to see estimated costs before processing:

```bash
python scripts/process_for_rag.py document.pdf --estimate-cost
```

## Project Structure

```
pdf-reader/
├── src/
│   └── llm_processor/
│       ├── __init__.py          # Public API exports
│       ├── models.py            # Pydantic data models
│       ├── vision_processor.py  # OpenAI Vision API processor
│       ├── chunk_generator.py   # Chunking logic
│       ├── pdf_to_images.py     # PDF rendering
│       └── prompts.py           # LLM prompts
├── scripts/
│   └── process_for_rag.py       # CLI tool
├── tests/
│   ├── test_models.py           # Model tests
│   ├── test_chunk_generator.py  # Chunking tests
│   ├── test_prompts.py          # Prompt tests
│   ├── test_vision_processor.py # Processor tests (mocked)
│   └── test_integration.py      # Real API tests
├── pdfs/                        # Test PDFs
├── requirements.txt
├── .env.example
└── README.md
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run unit tests (no API calls)
pytest tests/ -v --ignore=tests/test_integration.py

# Run integration tests (requires API key, costs money!)
export OPENAI_API_KEY="sk-your-key"
pytest tests/test_integration.py -v
```

## API Reference

### VisionProcessor

```python
from src.llm_processor import VisionProcessor

processor = VisionProcessor(
    config=ProcessingConfig(),  # Optional
    api_key="sk-...",           # Or set OPENAI_API_KEY env var
)

result = processor.process_document(
    pdf_path="document.pdf",
    progress_callback=lambda cur, total, status: print(f"{cur}/{total}: {status}"),
)

# Result contains:
# - result.context: DocumentContext
# - result.pages: list[ExtractedPage]
# - result.processing_time_seconds: float
# - result.total_input_tokens: int
# - result.total_output_tokens: int
# - result.errors: list[str]
```

### ChunkGenerator

```python
from src.llm_processor import ChunkGenerator

generator = ChunkGenerator(config=ProcessingConfig())

result = generator.generate_from_extraction(
    extraction_result,
    source_document="my-document",
)

# Result is ProcessingResult with:
# - result.chunks: list[RAGChunk]
# - result.export_chunks_langchain()
# - result.export_chunks_llamaindex()
# - result.export_chunks_jsonl(path)
```

## License

MIT License
