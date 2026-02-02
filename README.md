# PDF Section Extractor v3.0

A production-ready PDF extraction pipeline for German academic documents using OpenAI's GPT-4o Vision API.

## Key Features

- **Page-by-Page Scanning**: Scans every page to detect which sections appear on it
- **Accurate Page Ranges**: Handles sections that span partial pages correctly
- **Section Extraction**: Extracts complete content for each § and Anlage
- **Rich Metadata**: Document context, chapters, abbreviations, references
- **German Academic Documents**: Optimized for Prüfungsordnungen, Modulhandbücher, etc.

## How It Works

Unlike ToC-based approaches, this extractor scans **every page** individually to detect which sections appear on it. This solves the problem of sections that share pages (e.g., § 4 ends on page 6, § 5 starts on page 6).

```
Phase 1: PAGE SCAN
    For each page → "Which §§/Anlagen are on this page?"

Phase 2: STRUCTURE AGGREGATION
    § 5 appears on pages [5, 6, 7] → pages: [5, 6, 7]

Phase 3: CONTEXT EXTRACTION
    Extract document metadata from first pages

Phase 4: SECTION EXTRACTION
    Extract full content for each section
```

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

### Command Line Usage

```bash
# Extract a PDF
python main.py document.pdf

# With options
python main.py document.pdf -o output.json --verbose

# Estimate cost before extraction
python main.py document.pdf --estimate

# Use cheaper model
python main.py document.pdf --model gpt-4o-mini
```

### Python API

```python
from pdf_extractor import PDFExtractor, ExtractionConfig

# Basic usage
extractor = PDFExtractor()
result = extractor.extract("pruefungsordnung.pdf")

# Access results
print(f"Document: {result.context.title}")
print(f"Sections: {len(result.sections)}")

for section in result.sections:
    print(f"  {section.identifier}: {len(section.content)} chars")

# Save to JSON
result.save("output.json")

# With progress callback
def progress(current, total, message):
    print(f"[{current}/{total}] {message}")

result = extractor.extract("document.pdf", progress_callback=progress)

# With custom configuration
config = ExtractionConfig(
    model="gpt-4o-mini",   # Cheaper model
    max_retries=5,         # More retries
    include_scan_results=True,  # Include debug info
)
extractor = PDFExtractor(config=config)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   PDF Section Extractor v3.0                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PHASE 1: Page Scanning                                           │
│  ┌─────────┐     ┌──────────────┐     ┌─────────────────┐        │
│  │  PDF    │────▶│  Page Image  │────▶│   GPT-4o Vision │        │
│  │ Page N  │     │              │     │   "What §§?"    │        │
│  └─────────┘     └──────────────┘     └────────┬────────┘        │
│                                                 │                  │
│                                                 ▼                  │
│                                        PageScanResult              │
│                                        ├─ page_number: 5           │
│                                        └─ sections: [§4, §5]       │
│                                                                    │
│  PHASE 2: Structure Aggregation                                    │
│  ┌──────────────────────┐     ┌──────────────────────────┐        │
│  │  All PageScanResults │────▶│  Aggregate by Section    │        │
│  │                      │     │                          │        │
│  └──────────────────────┘     └────────────┬─────────────┘        │
│                                            │                       │
│                                            ▼                       │
│                                   DocumentStructure                │
│                                   └─ sections:                     │
│                                       ├─ § 4: pages [4, 5, 6]      │
│                                       ├─ § 5: pages [6, 7, 8]      │
│                                       └─ ...                       │
│                                                                    │
│  PHASE 3: Context Extraction                                       │
│  ┌────────────────┐     ┌─────────────────────────┐               │
│  │  First Pages   │────▶│  Extract Metadata       │               │
│  │  (1-5)         │     │  (title, institution)   │               │
│  └────────────────┘     └────────────┬────────────┘               │
│                                      │                             │
│                                      ▼                             │
│                              DocumentContext                       │
│                              ├─ title                              │
│                              ├─ institution                        │
│                              ├─ chapters                           │
│                              └─ abbreviations                      │
│                                                                    │
│  PHASE 4: Section Extraction                                       │
│  ┌────────────────────┐     ┌─────────────────────────┐           │
│  │  Section Pages     │────▶│  Extract Full Content   │           │
│  │  (e.g., 6, 7, 8)   │     │  for § 5                │           │
│  └────────────────────┘     └────────────┬────────────┘           │
│                                          │                         │
│                                          ▼                         │
│                                  ExtractedSection                  │
│                                  ├─ identifier: "§ 5"              │
│                                  ├─ content: "..."                 │
│                                  └─ pages: [6, 7, 8]               │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

## Output Format

### ExtractionResult Structure

```python
ExtractionResult(
    source_file="pruefungsordnung.pdf",
    context=DocumentContext(
        document_type="pruefungsordnung",
        title="Prüfungsordnung für den Studiengang Informatik B.Sc.",
        institution="Philipps-Universität Marburg",
        total_pages=56,
        chapters=["I. Allgemeines", "II. Prüfungen", ...],
        abbreviations=[Abbreviation(short="LP", long="Leistungspunkte")],
    ),
    structure=DocumentStructure(
        sections=[
            SectionLocation(identifier="§ 1", pages=[3]),
            SectionLocation(identifier="§ 2", pages=[3, 4]),
            # ...
        ],
        total_pages=56,
    ),
    sections=[
        ExtractedSection(
            section_number="§ 1",
            section_title="Geltungsbereich",
            content="(1) Diese Ordnung regelt das Studium...",
            pages=[3],
            chapter="I. Allgemeines",
            subsections=["(1)", "(2)"],
            internal_references=["§ 5 Abs. 2"],
        ),
        # ...
    ],
    processing_time_seconds=45.2,
    total_input_tokens=50000,
    total_output_tokens=15000,
)
```

## Configuration

```python
ExtractionConfig(
    # Model
    model="gpt-4o",           # Best quality
    # model="gpt-4o-mini",    # Faster, cheaper

    # Tokens
    max_tokens=4096,          # Max response tokens
    temperature=0.0,          # Deterministic output

    # Retry
    max_retries=3,            # API retry attempts
    retry_delay=1.0,          # Initial retry delay

    # Options
    extract_preamble=True,    # Extract content before first §
    include_scan_results=False,  # Include debug info
)
```

## Cost Estimation

| Model | ~Cost per 50 pages |
|-------|-------------------|
| gpt-4o | $0.80 - $1.50 |
| gpt-4o-mini | $0.05 - $0.10 |

Note: The page-by-page scanning approach uses more API calls than ToC-based extraction, but provides more accurate page ranges.

```bash
# Estimate cost before extraction
python main.py document.pdf --estimate
```

## Project Structure

```
pdf-reader/
├── pdf_extractor/
│   ├── __init__.py          # Public API exports
│   ├── models.py            # Pydantic data models
│   ├── exceptions.py        # Exception hierarchy
│   ├── prompts.py           # LLM prompts
│   ├── pdf_utils.py         # PDF rendering
│   ├── api_client.py        # OpenAI Vision API client
│   └── extractor.py         # Main extraction pipeline
├── tests/
│   ├── conftest.py          # Test fixtures
│   ├── test_models.py       # Model tests
│   ├── test_exceptions.py   # Exception tests
│   └── test_integration.py  # Integration tests
├── main.py                  # CLI tool
├── requirements.txt
└── README.md
```

## Testing

```bash
# Install test dependencies
pip install pytest pytest-mock

# Run unit tests (no API calls)
pytest tests/ -v --ignore=tests/test_integration.py

# Run integration tests (requires API key)
export OPENAI_API_KEY="sk-your-key"
pytest tests/test_integration.py -v
```

## Exception Handling

```python
from pdf_extractor import (
    PDFExtractor,
    PDFNotFoundError,
    StructureAggregationError,
    APIError,
)

try:
    result = extractor.extract("document.pdf")
except PDFNotFoundError as e:
    print(f"File not found: {e.path}")
except StructureAggregationError as e:
    print(f"No sections detected: {e}")
except APIError as e:
    print(f"API error: {e}")
```

## Why Page-by-Page Scanning?

Previous approaches used the Table of Contents to determine page ranges:

```
ToC says: § 5 starts on page 5
         § 6 starts on page 6
Therefore: § 5 is on page 5

Problem: § 5 actually ends in the MIDDLE of page 6!
         Both § 5 and § 6 share page 6.
```

Our approach scans each page:

```
Page 5: [§ 5]
Page 6: [§ 5, § 6]  ← Both sections appear!
Page 7: [§ 6]

Result: § 5 is on pages [5, 6]
        § 6 is on pages [6, 7]
```

This gives accurate page ranges for every section.

## License

MIT License
