# PDF Extractor v2.0

Ein produktionsreifes Modul zur Extraktion strukturierter Inhalte aus PDF-Dokumenten mittels OpenAI's Vision API (GPT-4o). Optimiert für deutsche akademische Dokumente (Prüfungsordnungen, Modulhandbücher, etc.).

## Features

- **Sektionsbasierte Extraktion**: Extraktion nach §§ und Anlagen statt Seiten
- **Zwei-Phasen-Extraktion**: Strukturanalyse (ToC) + sektionsweise Extraktion
- **Sliding Window**: Lange Sektionen werden mit überlappenden Fenstern verarbeitet
- **Hochwertige Inhaltstransformation**: Tabellen und Listen werden in natürliche Sprache umgewandelt
- **Robuste Fehlerbehandlung**: Retry-Mechanismus mit exponentiellem Backoff
- **Strukturierte Ausgabe**: Bereit für Downstream-Verarbeitung (Chunking, RAG, etc.)

## Architektur

```
pdf_extractor/
├── __init__.py          # Öffentliche API
├── extractor.py         # Hauptextraktor (PDFExtractor)
├── models.py            # Pydantic v2 Datenmodelle
├── prompts.py           # LLM-Prompts für Extraktion
├── pdf_to_images.py     # PDF zu Bild Konvertierung
├── exceptions.py        # Custom Exceptions
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
from pdf_extractor import PDFExtractor, NoTableOfContentsError
from dotenv import load_dotenv

# .env laden
load_dotenv()

# Extractor initialisieren
extractor = PDFExtractor()

try:
    # PDF verarbeiten
    result = extractor.extract("dokument.pdf")

    # Ergebnisse nutzen (v2.0 API: sections statt pages)
    print(f"Titel: {result.context.title}")
    print(f"Sektionen: {len(result.sections)}")

    # Einzelne Sektionen durchgehen
    for section in result.sections:
        print(f"{section.identifier}: {section.content[:100]}...")

    # In Datei speichern
    result.save("output.json")

except NoTableOfContentsError:
    print("FEHLER: Kein Inhaltsverzeichnis gefunden!")
```

### Mit Fortschrittsanzeige

```python
def progress(current, total, status):
    print(f"Sektion {current}/{total}: {status}")

result = extractor.extract("dokument.pdf", progress_callback=progress)
```

### Mit Konfiguration

```python
from pdf_extractor import PDFExtractor, ExtractionConfig

config = ExtractionConfig(
    model="gpt-4o-mini",        # Günstigeres Modell
    max_retries=5,              # Mehr Wiederholungsversuche
    max_images_per_request=10,  # Mehr Seiten pro Request (schneller)
    temperature=0.0,            # Deterministisch
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

# Mehr Bilder pro Request
python scripts/extract_pdf.py dokument.pdf --max-images 10
```

## API-Referenz

### PDFExtractor

Die Hauptklasse für die PDF-Extraktion.

```python
class PDFExtractor:
    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Args:
            config: Extraktionskonfiguration
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
            progress_callback: Optionaler Callback (current, total, status)

        Returns:
            ExtractionResult mit Kontext und allen Sektionen

        Raises:
            NoTableOfContentsError: Wenn kein Inhaltsverzeichnis gefunden wird
            ValueError: Wenn kein API-Key gesetzt ist
            FileNotFoundError: Wenn die PDF-Datei nicht existiert
        """
```

### ExtractionResult

Enthält die vollständigen Extraktionsergebnisse.

```python
class ExtractionResult:
    source_file: str                    # Pfad zur Quelldatei
    context: DocumentContext            # Dokumentenkontext
    sections: list[ExtractedSection]    # Extrahierte Sektionen (§§, Anlagen)
    processing_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    errors: list[str]
    warnings: list[str]
    processed_at: datetime

    # Query-Methoden
    def get_section(number: str) -> Optional[ExtractedSection]  # z.B. "§ 10"
    def get_overview() -> Optional[ExtractedSection]            # Übersicht/Präambel
    def get_paragraphs() -> list[ExtractedSection]              # Alle §§
    def get_anlagen() -> list[ExtractedSection]                 # Alle Anlagen

    # Statistiken und Export
    def get_stats() -> dict                              # Statistiken
    def get_full_content(separator: str) -> str          # Gesamter Inhalt
    def to_dict() -> dict                                # Als Dictionary
    def to_json(indent: int) -> str                      # Als JSON
    def save(path: str) -> None                          # In Datei speichern
    @classmethod
    def load(path: str) -> ExtractionResult              # Aus Datei laden
```

### ExtractedSection

Inhalt einer extrahierten Sektion (§, Anlage, oder Übersicht).

```python
class ExtractedSection:
    # Identifikation
    section_type: SectionType             # overview, paragraph, anlage
    section_number: Optional[str]         # z.B. "§ 10", "Anlage 2"
    section_title: Optional[str]          # z.B. "Module und Leistungspunkte"

    # Inhalt
    content: str                          # Vollständiger Inhalt in natürlicher Sprache
    pages: list[int]                      # Seitennummern (1-indiziert)
    chapter: Optional[str]                # Übergeordnetes Kapitel
    paragraphs: list[str]                 # Absatznummern ["(1)", "(2)", ...]

    # Referenzen
    internal_references: list[str]        # z.B. ["§ 5 Abs. 2", "Anlage 1"]
    external_references: list[str]        # z.B. ["Allgemeine Bestimmungen"]

    # Content-Eigenschaften
    has_table: bool                       # Enthält Tabellen
    has_list: bool                        # Enthält Listen

    # Metadaten
    token_count: int                      # Geschätzte Token-Anzahl
    extraction_confidence: float          # 0.0 - 1.0
    extraction_notes: Optional[str]       # Hinweise bei Problemen

    # Eigenschaften
    @property
    def identifier(self) -> str           # "§ 10 Titel" oder "Übersicht"

    # Methoden
    def get_source_reference(doc_title: str) -> str  # Quellenangabe für RAG
```

### DocumentContext

Dokument-Metadaten aus der Strukturanalyse.

```python
class DocumentContext:
    document_type: DocumentType     # Art des Dokuments
    title: str                      # Offizieller Titel
    institution: str                # Universität/Hochschule
    version_date: Optional[str]     # Datum der Fassung
    version_info: Optional[str]     # Versionsinfo
    degree_program: Optional[str]   # Studiengang
    faculty: Optional[str]          # Fachbereich
    total_pages: int                # Seitenzahl
    chapters: list[str]             # Kapitel/Gliederung
    abbreviations: list[Abbreviation]  # Abkürzungen
    key_terms: list[str]            # Fachbegriffe
    referenced_documents: list[str]
    legal_basis: Optional[str]
    language: Language

    def get_abbreviation_dict() -> dict[str, str]  # Abkürzungen als Dict
```

### ExtractionConfig

Konfiguration für die Extraktion.

```python
class ExtractionConfig:
    model: str = "gpt-4o"                 # OpenAI-Modell
    max_tokens_per_request: int = 4096    # Max Tokens pro Request
    temperature: float = 0.0              # Sampling-Temperatur
    max_images_per_request: int = 5       # Bilder pro Request (Sliding Window)
    max_retries: int = 3                  # Wiederholungsversuche
    language: Language = Language.DE
```

## Datenmodelle

### SectionType (Enum)

```python
class SectionType(str, Enum):
    OVERVIEW = "overview"      # Präambel, Inhaltsverzeichnis, Einleitung
    PARAGRAPH = "paragraph"    # § mit Nummer (§ 1, § 2, ... § 40)
    ANLAGE = "anlage"          # Anlage 1, Anlage 2, ...
```

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

### Exceptions

```python
class NoTableOfContentsError(Exception):
    """Raised when no table of contents is found in the document."""
    pass
```

## Extraktionsprozess

### Phase 1: Strukturanalyse

1. **Stichprobenseiten**: Seiten 1-3 (Titelseite, Inhaltsverzeichnis), Mitte und Ende
2. **Analyse**: Dokumenttyp, Titel, Institution, Inhaltsverzeichnis
3. **Strukturextraktion**: Mapping von Sektionen (§§, Anlagen) zu Seitenbereichen
4. **Output**: `DocumentContext` + Liste von `StructureEntry`

**Wichtig**: Falls kein Inhaltsverzeichnis gefunden wird, wirft der Extractor `NoTableOfContentsError`.

### Phase 2: Sektionsweise Extraktion

1. **Für jede Sektion** (§, Anlage, Übersicht):
   - Relevante Seiten als Bilder rendern (150 DPI)
   - Bei langen Sektionen: Sliding Window mit 1-Seite Überlappung
   - Mit Kontextinformationen an GPT-4o senden
   - Antwort parsen und validieren
2. **Inhaltstransformation**:
   - Tabellen → Natürliche Sprache
   - Listen → Fließtext
   - Strukturmarker erhalten (§§, Absätze)
3. **Fehlerbehandlung**:
   - Retry bei API-Verweigerung oder leerer Antwort
   - Exponentielles Backoff (2s, 4s, 8s)

### Sliding Window

Bei Sektionen, die mehr Seiten haben als `max_images_per_request`:

```
Sektion umfasst Seiten 1-12, max_images=5:
  Window 1: [1, 2, 3, 4, 5]
  Window 2: [5, 6, 7, 8, 9]    # Überlappung bei Seite 5
  Window 3: [9, 10, 11, 12]   # Überlappung bei Seite 9
```

Die Inhalte werden intelligent zusammengeführt.

## Output-Format

### JSON-Struktur (v2.0)

```json
{
  "source_file": "dokument.pdf",
  "context": {
    "document_type": "pruefungsordnung",
    "title": "Prüfungsordnung für...",
    "institution": "Philipps-Universität Marburg",
    "faculty": "Fachbereich Mathematik und Informatik",
    "degree_program": "Informatik B.Sc.",
    "total_pages": 56,
    "chapters": ["I. Allgemeines", "II. Studienbezogene Bestimmungen", "..."],
    "abbreviations": [{"short": "LP", "long": "Leistungspunkte"}],
    "...": "..."
  },
  "sections": [
    {
      "section_type": "overview",
      "section_number": null,
      "section_title": null,
      "content": "Diese Prüfungsordnung regelt...",
      "pages": [1, 2],
      "has_table": false,
      "extraction_confidence": 1.0
    },
    {
      "section_type": "paragraph",
      "section_number": "§ 1",
      "section_title": "Geltungsbereich",
      "content": "§ 1 Geltungsbereich. (1) Diese Prüfungsordnung regelt...",
      "pages": [3],
      "paragraphs": ["(1)", "(2)"],
      "internal_references": ["§ 5 Abs. 2"],
      "has_table": false,
      "extraction_confidence": 1.0
    },
    {
      "section_type": "anlage",
      "section_number": "Anlage 1",
      "section_title": "Modulliste",
      "content": "Die folgenden Module sind Teil des Studiengangs...",
      "pages": [45, 46, 47],
      "has_table": true,
      "extraction_confidence": 1.0
    }
  ],
  "processing_time_seconds": 120.5,
  "total_input_tokens": 50000,
  "total_output_tokens": 25000,
  "errors": [],
  "warnings": [],
  "processed_at": "2024-01-15T10:30:00Z"
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
config = ExtractionConfig(model="gpt-4o-mini")
```

## Fehlerbehandlung

### Retry-Mechanismus

Bei API-Verweigerungen, leeren Antworten, oder ungültigem JSON:
1. Erster Retry nach 2 Sekunden
2. Zweiter Retry nach 4 Sekunden
3. Dritter Retry nach 8 Sekunden
4. Bei Fehlschlag: Sektion mit `extraction_confidence=0.0` markiert

### Erkannte Probleme

- **Verweigerungsphrasen**: "I'm sorry, I can't assist", "I cannot assist", etc.
- **Leere Antworten**: Leere Strings oder nur Whitespace
- **Ungültiges JSON**: Antworten ohne `{` werden als invalid erkannt

### NoTableOfContentsError

Wenn kein Inhaltsverzeichnis gefunden wird, kann keine sektionsbasierte Extraktion durchgeführt werden:

```python
from pdf_extractor import PDFExtractor, NoTableOfContentsError

try:
    result = extractor.extract("dokument.pdf")
except NoTableOfContentsError as e:
    print(f"Kein Inhaltsverzeichnis gefunden: {e}")
    # Fallback: Manuelles Processing oder andere Strategie
```

## Best Practices

1. **Vorher Kosten schätzen**: `--estimate-cost` verwenden
2. **Ergebnis prüfen**: `result.errors` und `result.warnings` checken
3. **Fehlgeschlagene Sektionen**: `extraction_confidence < 1.0` filtern
4. **Caching**: Ergebnisse speichern, nicht mehrfach extrahieren
5. **Logging aktivieren**: Bei Problemen `logging.basicConfig(level=logging.WARNING)` setzen

## Migration von v1.0 zu v2.0

### Hauptänderungen

| v1.0 (page-based)          | v2.0 (section-based)              |
|---------------------------|-----------------------------------|
| `result.pages`            | `result.sections`                 |
| `ProcessingConfig`        | `ExtractionConfig`                |
| `ExtractedPage`           | `ExtractedSection`                |
| `get_page_stats()`        | `get_stats()`                     |
| Keine ToC-Abhängigkeit    | Erfordert Inhaltsverzeichnis      |

### Code-Migration

```python
# v1.0
for page in result.pages:
    print(f"Seite {page.page_number}: {page.content[:50]}")

# v2.0
for section in result.sections:
    print(f"{section.identifier}: {section.content[:50]}")
```
