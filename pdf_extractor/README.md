# PDF Extractor

Ein produktionsreifes Modul zur Extraktion strukturierter Inhalte aus PDF-Dokumenten mittels OpenAI's Vision API (GPT-4o). Optimiert für deutsche akademische Dokumente (Prüfungsordnungen, Modulhandbücher, etc.).

## Features

- **Zwei-Phasen-Extraktion**: Kontextanalyse + seitenweise Extraktion
- **Hochwertige Inhaltstransformation**: Tabellen und Listen werden in natürliche Sprache umgewandelt
- **Robuste Fehlerbehandlung**: Retry-Mechanismus mit exponentiellem Backoff
- **Strukturierte Ausgabe**: Bereit für Downstream-Verarbeitung (Chunking, RAG, etc.)

## Architektur

```
pdf_extractor/
├── __init__.py          # Öffentliche API
├── extractor.py         # Hauptextraktor (PDFExtractor)
├── models.py            # Pydantic-Datenmodelle
├── prompts.py           # LLM-Prompts für Extraktion
├── pdf_to_images.py     # PDF zu Bild Konvertierung
└── README.md            # Diese Dokumentation
```

## Installation

### Voraussetzungen

```bash
pip install openai pydantic pymupdf python-dotenv
```

### Umgebungsvariable

```bash
export OPENAI_API_KEY="your-api-key"
```

Oder erstelle eine `.env` Datei:
```
OPENAI_API_KEY=your-api-key
```

## Verwendung

### Schnellstart

```python
from pdf_extractor import PDFExtractor

# Extractor initialisieren
extractor = PDFExtractor()

# PDF verarbeiten
result = extractor.extract("dokument.pdf")

# Ergebnisse nutzen
print(f"Titel: {result.context.title}")
print(f"Seiten: {len(result.pages)}")

# In Datei speichern
result.save("output.json")
```

### Mit Fortschrittsanzeige

```python
def progress(current, total, status):
    print(f"Seite {current}/{total}: {status}")

result = extractor.extract("dokument.pdf", progress_callback=progress)
```

### Mit Konfiguration

```python
from pdf_extractor import PDFExtractor, ProcessingConfig

config = ProcessingConfig(
    model="gpt-4o-mini",    # Günstigeres Modell
    max_retries=5,          # Mehr Wiederholungsversuche
    temperature=0.0,        # Deterministisch
)

extractor = PDFExtractor(config=config)
result = extractor.extract("dokument.pdf")
```

### Kommandozeile

```bash
# Standard-Extraktion
python scripts/extract_pdf.py dokument.pdf -o output/

# Kosten schätzen
python scripts/extract_pdf.py dokument.pdf --estimate-cost

# Anderes Modell verwenden
python scripts/extract_pdf.py dokument.pdf --model gpt-4o-mini
```

## API-Referenz

### PDFExtractor

Die Hauptklasse für die PDF-Extraktion.

```python
class PDFExtractor:
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            config: Verarbeitungskonfiguration
            api_key: OpenAI API-Key (oder OPENAI_API_KEY Umgebungsvariable)
        """

    def extract(
        self,
        pdf_path: str | Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExtractionResult:
        """
        Extrahiert Inhalte aus einem PDF-Dokument.

        Args:
            pdf_path: Pfad zur PDF-Datei
            progress_callback: Optionaler Callback für Fortschritt

        Returns:
            ExtractionResult mit Kontext und allen Seiteninhalten
        """
```

### ExtractionResult

Enthält die vollständigen Extraktionsergebnisse.

```python
class ExtractionResult:
    source_file: str              # Pfad zur Quelldatei
    context: DocumentContext      # Dokumentenkontext
    pages: list[ExtractedPage]    # Extrahierte Seiten
    processing_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    errors: list[str]
    warnings: list[str]

    # Methoden
    def get_page_stats(self) -> dict         # Statistiken
    def get_all_sections(self) -> list       # Alle Abschnitte
    def get_full_content(self) -> str        # Gesamter Inhalt
    def to_dict(self) -> dict                # Als Dictionary
    def to_json(self) -> str                 # Als JSON
    def save(self, path: str) -> None        # In Datei speichern
    @classmethod
    def load(cls, path: str) -> ExtractionResult  # Aus Datei laden
```

### DocumentContext

Dokument-Metadaten aus der Kontextanalyse.

```python
class DocumentContext:
    document_type: DocumentType   # Art des Dokuments
    title: str                    # Offizieller Titel
    institution: str              # Universität/Hochschule
    version_date: Optional[str]   # Datum der Fassung
    version_info: Optional[str]   # Versionsinfo
    degree_program: Optional[str] # Studiengang
    faculty: Optional[str]        # Fachbereich
    total_pages: int              # Seitenzahl
    chapters: list[str]           # Kapitel/Gliederung
    main_topics: list[str]        # Hauptthemen
    abbreviations: list[Abbreviation]  # Abkürzungen
    key_terms: list[str]          # Fachbegriffe
    referenced_documents: list[str]
    legal_basis: Optional[str]
    language: Language
```

### ExtractedPage

Inhalt einer einzelnen Seite.

```python
class ExtractedPage:
    page_number: int              # Seitennummer (1-indiziert)
    content: str                  # Inhalt in natürlicher Sprache
    sections: list[SectionMarker] # Abschnitte (§§, Anlagen)
    paragraph_numbers: list[str]  # Absatznummern (1), (2), etc.
    content_types: list[ContentType]
    has_table: bool
    has_list: bool
    has_figure: bool
    internal_references: list[str]   # z.B. "§ 5 Abs. 2"
    external_references: list[str]   # z.B. "Allgemeine Bestimmungen"
    continues_from_previous: bool
    continues_to_next: bool
    extraction_confidence: float     # 0.0 - 1.0
    extraction_notes: Optional[str]
```

### ProcessingConfig

Konfiguration für die Extraktion.

```python
class ProcessingConfig:
    model: str = "gpt-4o"                # OpenAI-Modell
    max_tokens_per_request: int = 4096   # Max Tokens pro Request
    temperature: float = 0.0             # Sampling-Temperatur
    max_retries: int = 3                 # Wiederholungsversuche
    expand_abbreviations: bool = True
    include_page_context: bool = True
    merge_cross_page_content: bool = True
    output_format: str = "json"
    language: Language = Language.DE
```

## Datenmodelle

### DocumentType (Enum)

```python
class DocumentType(str, Enum):
    PRUEFUNGSORDNUNG = "pruefungsordnung"
    MODULHANDBUCH = "modulhandbuch"
    STUDIENORDNUNG = "studienordnung"
    ALLGEMEINE_BESTIMMUNGEN = "allgemeine_bestimmungen"
    PRAKTIKUMSORDNUNG = "praktikumsordnung"
    ZULASSUNGSORDNUNG = "zulassungsordnung"
    SATZUNG = "satzung"
    WEBSITE = "website"
    FAQ = "faq"
    OTHER = "other"
```

### ContentType (Enum)

```python
class ContentType(str, Enum):
    SECTION = "section"           # Paragraph (§)
    SUBSECTION = "subsection"     # Absatz
    ARTICLE = "article"
    DEFINITION = "definition"
    REGULATION = "regulation"
    PROCEDURE = "procedure"
    DEADLINE = "deadline"
    REQUIREMENT = "requirement"
    TABLE = "table"
    LIST = "list"
    GRADE_SCALE = "grade_scale"
    METADATA = "metadata"
    REFERENCE = "reference"
    OVERVIEW = "overview"
```

## Extraktionsprozess

### Phase 1: Kontextanalyse

1. **Stichprobenseiten**: Seiten 1-3 (Titelseite, Inhaltsverzeichnis), Mitte und Ende
2. **Analyse**: Dokumenttyp, Titel, Institution, Kapitelstruktur, Abkürzungen
3. **Output**: `DocumentContext` mit allen Metadaten

### Phase 2: Seitenweise Extraktion

1. **Für jede Seite**:
   - Seite als Bild rendern (150 DPI)
   - Mit Kontextinformationen an GPT-4o senden
   - Antwort parsen und validieren
2. **Inhaltstransformation**:
   - Tabellen → Natürliche Sprache
   - Listen → Fließtext
   - Strukturmarker erhalten (§§, Anlagen)
3. **Fehlerbehandlung**:
   - Retry bei API-Verweigerung
   - Exponentielles Backoff (2s, 4s, 8s)

## Output-Format

### JSON-Struktur

```json
{
  "source_file": "dokument.pdf",
  "context": {
    "document_type": "pruefungsordnung",
    "title": "Prüfungsordnung für...",
    "institution": "Philipps-Universität Marburg",
    "faculty": "Fachbereich Mathematik und Informatik",
    "degree_program": "Informatik B.Sc.",
    "chapters": ["Teil I - Allgemeines", "Teil II - ...", "Anlage 1: ..."],
    "abbreviations": [{"short": "LP", "long": "Leistungspunkte"}],
    "...": "..."
  },
  "pages": [
    {
      "page_number": 1,
      "content": "Die Prüfungsordnung regelt...",
      "sections": [{"number": "§ 1", "title": "Geltungsbereich"}],
      "has_table": false,
      "continues_to_next": true,
      "...": "..."
    }
  ],
  "processing_time_seconds": 120.5,
  "total_input_tokens": 50000,
  "total_output_tokens": 25000,
  "errors": [],
  "warnings": []
}
```

## Kosten

### Geschätzte Kosten (GPT-4o)

| Seitenzahl | Input Tokens | Output Tokens | Kosten (USD) |
|------------|--------------|---------------|--------------|
| 10         | ~17.500      | ~8.500        | ~0.13        |
| 50         | ~77.500      | ~40.500       | ~0.60        |
| 100        | ~152.500     | ~80.500       | ~1.19        |

### Kosten reduzieren

```python
# GPT-4o-mini ist ~15x günstiger
config = ProcessingConfig(model="gpt-4o-mini")
```

## Fehlerbehandlung

### Retry-Mechanismus

Bei API-Verweigerungen (z.B. "I'm sorry, I can't assist..."):
1. Erster Retry nach 2 Sekunden
2. Zweiter Retry nach 4 Sekunden
3. Dritter Retry nach 8 Sekunden
4. Bei Fehlschlag: Seite mit `extraction_confidence=0.0` markiert

### Erkannte Verweigerungsphrasen

- "I'm sorry, I can't assist"
- "I cannot assist"
- "I'm not able to"
- "I cannot help"
- "I'm unable to"
- etc.