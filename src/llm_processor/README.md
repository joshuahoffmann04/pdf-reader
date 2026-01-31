# LLM-based PDF Processor for RAG

A Vision-LLM pipeline for converting PDF documents into RAG-optimized natural language chunks.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Processing Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Context Analysis                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐          │
│  │   PDF    │───▶│  Sample  │───▶│   Vision LLM    │          │
│  │          │    │  Pages   │    │ (Context Prompt) │          │
│  └──────────┘    └──────────┘    └────────┬─────────┘          │
│                                           │                     │
│                                           ▼                     │
│                                  DocumentContext                │
│                                  - document_type                │
│                                  - title, institution           │
│                                  - abbreviations                │
│                                  - key_terms                    │
│                                                                 │
│  Phase 2: Page-by-Page Extraction                               │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐          │
│  │  Page N  │───▶│  Image   │───▶│   Vision LLM    │          │
│  │          │    │          │    │ (+ Context)      │          │
│  └──────────┘    └──────────┘    └────────┬─────────┘          │
│                                           │                     │
│                                           ▼                     │
│                                     PageContent                 │
│                                  - natural language text        │
│                                  - section info                 │
│                                  - references                   │
│                                                                 │
│  Phase 3: Chunk Generation                                      │
│  ┌──────────────┐    ┌──────────────────┐                      │
│  │ All Pages    │───▶│  Smart Chunking  │                      │
│  │              │    │  + Metadata      │                      │
│  └──────────────┘    └────────┬─────────┘                      │
│                               │                                 │
│                               ▼                                 │
│                          RAG Chunks (JSONL)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install with LLM support
pip install -e ".[llm]"

# Or install anthropic/openai separately
pip install anthropic  # for Claude
pip install openai     # for GPT-4V
```

### Basic Usage

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
from src.llm_processor import VisionProcessor, ChunkGenerator, ProcessingConfig

# Configure
config = ProcessingConfig(
    model="claude-sonnet-4-20250514",
    target_chunk_size=500,
)

# Process document
processor = VisionProcessor(config=config)
result = processor.process_document("document.pdf")

# Generate chunks
generator = ChunkGenerator(config=config)
chunks = generator.generate_chunks(result, "document-name")

# Export
generator.export_jsonl(chunks, "output.jsonl")
```

## Output Format

### JSONL (for RAG ingestion)

Each line is a self-contained chunk:

```json
{
  "id": "pruefungsordnung-§10-abc123",
  "text": "§10 Module und Leistungspunkte: Ein Modul ist eine inhaltlich und zeitlich abgeschlossene Lehr- und Lerneinheit...",
  "chunk_type": "section",
  "metadata": {
    "source_document": "Pruefungsordnung_BSc_Mathematik_2024",
    "source_pages": [9, 10],
    "section_number": "§10",
    "section_title": "Module und Leistungspunkte",
    "chapter": "II. Studienbezogene Bestimmungen",
    "topics": ["Module", "Leistungspunkte", "ECTS"],
    "keywords": ["Modul", "LP", "Regelstudienzeit"],
    "related_sections": ["§7", "§11"]
  }
}
```

## Cost Estimation

| Model | ~Cost per 50 pages |
|-------|-------------------|
| claude-sonnet-4-20250514 | $0.15 - $0.25 |
| claude-3-haiku | $0.02 - $0.04 |
| gpt-4o | $0.12 - $0.20 |
| gpt-4o-mini | $0.01 - $0.02 |

## Key Features

### 1. Context-Aware Extraction

The first phase analyzes the entire document to understand:
- Document type (Prüfungsordnung, Modulhandbuch, etc.)
- Institution and degree program
- Abbreviations (LP, AB, ECTS, etc.)
- Key terminology

This context is then used to guide page-by-page extraction.

### 2. Natural Language Output

Tables, lists, and structured content are converted to natural language:

**Before (raw table):**
```
| Punkte | Note |
| 15-13  | sehr gut |
```

**After (natural language):**
```
Gemäß der Bewertungstabelle entsprechen 15 bis 13 Punkte
der Note "sehr gut" (Dezimalnote 0,7 bis 1,3).
```

### 3. Smart Chunking

- Respects document structure (§, paragraphs)
- Merges content that spans multiple pages
- Maintains context within each chunk
- Links related chunks via metadata

### 4. Rich Metadata

Each chunk includes:
- Source information (document, pages)
- Structural info (section, paragraph, chapter)
- Semantic info (topics, keywords)
- Relationships (related sections, references)

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

    # Output settings
    output_format="jsonl",   # or "json"
)
```

## Error Handling

The processor is designed to be robust:
- Failed pages are logged but don't stop processing
- Placeholder content is created for failed extractions
- All errors are collected in the result object

## Limitations

1. **API Costs**: Processing large documents can be expensive
2. **Rate Limits**: Built-in delays to avoid rate limiting
3. **Hallucination Risk**: LLMs may occasionally add information - use temperature=0
4. **Language**: Prompts are optimized for German academic documents
