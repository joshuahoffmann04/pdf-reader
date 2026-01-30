# PDF Reader

A high-accuracy PDF extraction tool for German legal and academic documents. Extracts and structures content with **99%+ similarity** to the original text.

## Features

- **Text Extraction**: Layout-preserving extraction using PyMuPDF
- **Structure Recognition**: Automatic detection of chapters (I., II., ...), sections (§1, §2, ...), and appendices (Anlage 1, 2, ...)
- **Table Extraction**: Multi-page table support with intelligent merging
- **Image Extraction**: With deduplication and size filtering
- **AB Linking**: Automatic linking of "Allgemeine Bestimmungen" excerpts to main sections
- **Page Tracking**: Every section includes page number references
- **Quality Evaluation**: Built-in similarity metrics for extraction validation
- **Dual Export**: JSON and Markdown output formats

## Supported Document Types

This tool is optimized for German legal and academic documents that follow common structural patterns:

| Pattern | Example | Description |
|---------|---------|-------------|
| Chapters | `I. Allgemeines` | Roman numeral headings |
| Sections | `§ 1 Geltungsbereich` | Paragraph sections |
| Appendices | `Anlage 1` / `Anhang 1` | Numbered attachments |
| AB Excerpts | `Textauszug aus den Allgemeinen Bestimmungen` | Referenced general provisions |

**Example documents**: Prüfungsordnungen (examination regulations), Satzungen (statutes), Ordnungen (regulations)

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using pip with development dependencies

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- PyMuPDF (fitz)
- pdfplumber
- Pillow
- numpy
- scikit-learn
- nltk

## Quick Start

### Command Line

```bash
# Basic extraction
python main.py document.pdf

# Custom output directory and format
python main.py document.pdf -o results/ -f json

# Verbose output
python main.py document.pdf -v

# Skip image/table extraction
python main.py document.pdf --no-images --no-tables
```

### Python API

```python
from pathlib import Path
from main import PDFReaderPipeline, ExtractionConfig

# Configure extraction
config = ExtractionConfig(
    extract_images=True,
    extract_tables=True,
    merge_cross_page_tables=True,
    output_format="both"  # "json", "markdown", or "both"
)

# Run extraction
pipeline = PDFReaderPipeline(config)
result = pipeline.extract(Path("document.pdf"), Path("output/"))

# Access results
print(f"Found {result.statistics['chapters']} chapters")
print(f"Found {result.statistics['main_sections']} sections")

for chapter in result.chapters:
    print(f"{chapter['numeral']}. {chapter['title']}")
    for section in chapter['sections']:
        print(f"  {section['id']}: {section['title']}")
```

### Low-Level API

```python
from src import PDFExtractor, DocumentParser, TableExtractor, ImageExtractor

# Extract raw text
extractor = PDFExtractor()
pdf_doc = extractor.extract("document.pdf")

# Parse structure (with page markers for tracking)
parser = DocumentParser()
doc = parser.parse(pdf_doc.get_full_text(include_page_markers=True))

# Access parsed structure
for chapter in doc.chapters:
    print(f"{chapter.numeral}. {chapter.title}")
    for section in chapter.sections:
        print(f"  {section.id} {section.title} (Pages: {section.pages})")

# Extract tables
table_extractor = TableExtractor()
tables = table_extractor.extract_from_pdf("document.pdf")

# Extract images
image_extractor = ImageExtractor(min_width=100, min_height=100)
images = image_extractor.extract_from_pdf("document.pdf", output_dir="images/")
```

## Output Structure

### JSON Schema

```json
{
  "version": "1.0",
  "extracted_at": "2024-01-15T10:30:00",
  "metadata": {
    "source": "document.pdf",
    "title": "Document Title",
    "total_pages": 50
  },
  "preamble": "Introductory text...",
  "chapters": [
    {
      "id": "I",
      "numeral": "I",
      "title": "Allgemeines",
      "sections": [
        {
          "id": "§1",
          "number": 1,
          "title": "Geltungsbereich",
          "content": "Section content...",
          "pages": [3, 4],
          "ab_references": ["§6"]
        }
      ],
      "ab_excerpts": [
        {
          "id": "§6",
          "title": "Prüfungsausschuss",
          "content": "AB excerpt content...",
          "follows_section": "§5"
        }
      ]
    }
  ],
  "appendices": [
    {
      "id": "Anlage 1",
      "number": "1",
      "title": "Modulhandbuch",
      "content": "...",
      "sections": []
    }
  ],
  "tables": [...],
  "images": [...],
  "statistics": {
    "chapters": 4,
    "main_sections": 40,
    "ab_excerpts": 8,
    "appendices": 5,
    "tables": 12,
    "images": 3
  }
}
```

## Scripts

### Analyze PDF Structure

Analyze a PDF to understand its structure before extraction:

```bash
python -m scripts.analyze document.pdf
python -m scripts.analyze document.pdf --verbose
```

### Validate Extraction

Validate the parsed structure:

```bash
python -m scripts.validate document.pdf
python -m scripts.validate document.pdf --expected-chapters 4
```

### Evaluate Quality

Compare extracted content against the original PDF:

```bash
python -m scripts.evaluate --pdf document.pdf --json output/document_extracted.json
python -m scripts.evaluate --pdf document.pdf --json output/document_extracted.json --output report.json
```

## Quality Metrics

The evaluation system provides comprehensive quality metrics:

| Metric | Description |
|--------|-------------|
| Cosine Similarity | Content accuracy based on TF-IDF vectors |
| BLEU Score | N-gram precision (1-4 grams) |
| Word Overlap | Jaccard similarity of word sets |
| Overall Score | Weighted combination (0-100%) |

**Typical results for German legal documents:**
- Cosine Similarity: 99%+
- BLEU Score: 98%+
- Overall Score: 95%+

## Project Structure

```
pdf-reader/
├── main.py                 # CLI and main pipeline
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── src/
│   ├── __init__.py         # Public API exports
│   ├── logging_config.py   # Logging configuration
│   ├── extractor/          # PDF text extraction
│   │   ├── __init__.py
│   │   └── pdf_extractor.py
│   ├── parser/             # Document structure parsing
│   │   ├── __init__.py
│   │   └── document_parser.py
│   ├── tables/             # Table extraction
│   │   ├── __init__.py
│   │   └── table_extractor.py
│   ├── images/             # Image extraction
│   │   ├── __init__.py
│   │   └── image_extractor.py
│   └── evaluation/         # Quality metrics
│       ├── __init__.py
│       └── evaluator.py
├── scripts/                # Utility scripts
│   ├── analyze.py          # PDF structure analysis
│   ├── validate.py         # Structure validation
│   └── evaluate.py         # Quality evaluation
└── tests/                  # Test suite
    ├── test_document_parser.py
    ├── test_table_extractor.py
    ├── test_image_extractor.py
    ├── test_evaluator.py
    └── test_integration.py
```

## Architecture

```
PDF File
    │
    ▼
┌─────────────────┐
│  PDFExtractor   │  ──▶  Raw text with page markers
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ DocumentParser  │  ──▶  Hierarchical structure
└─────────────────┘       (Chapters, Sections, Appendices)
    │
    ├──────────────────┐
    ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│ TableExtractor  │  │ ImageExtractor  │
└─────────────────┘  └─────────────────┘
    │                  │
    ▼                  ▼
┌─────────────────────────────────────┐
│         ExtractionResult            │
│  (JSON / Markdown Export)           │
└─────────────────────────────────────┘
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_document_parser.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

### ExtractionConfig Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extract_images` | bool | True | Extract images from PDF |
| `extract_tables` | bool | True | Extract tables from PDF |
| `merge_cross_page_tables` | bool | True | Merge tables spanning multiple pages |
| `min_image_width` | int | 100 | Minimum image width in pixels |
| `min_image_height` | int | 100 | Minimum image height in pixels |
| `output_format` | str | "both" | Output format: "json", "markdown", or "both" |

## Extending for New Document Types

To adapt the parser for different document structures:

1. **Analyze the structure** using `scripts/analyze.py`
2. **Modify patterns** in `src/parser/document_parser.py`:
   - `CHAPTER_PATTERN` for chapter headers
   - `SECTION_PATTERN` for section headers
   - `APPENDIX_PATTERN` for appendices
   - `AB_MARKER_PATTERN` for special excerpts

3. **Validate** using `scripts/validate.py`
4. **Evaluate quality** using `scripts/evaluate.py`

## License

MIT

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request
