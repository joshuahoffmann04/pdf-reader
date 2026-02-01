# PDF Extractor

Ein produktionsreifes Modul zur **sektionsbasierten** Extraktion strukturierter Inhalte aus PDF-Dokumenten mittels OpenAI's Vision API (GPT-4o). Optimiert für deutsche akademische Dokumente (Prüfungsordnungen, Modulhandbücher, etc.).

## Features

- **Zwei-Phasen-Extraktion**: Strukturanalyse (ToC) + sektionsweise Extraktion
- **Sektionsbasiert**: Extrahiert nach §§ und Anlagen statt nach Seiten
- **Alle Seiten einer Sektion in EINEM API-Call**: Keine Seitenübergangs-Probleme
- **Sliding Window**: Für Sektionen > max_images_per_request Seiten
- **Hochwertige Inhaltstransformation**: Tabellen und Listen werden in natürliche Sprache umgewandelt
- **Robuste Fehlerbehandlung**: Retry-Mechanismus mit exponentiellem Backoff
- **NoTableOfContentsError**: Wirft Fehler wenn kein Inhaltsverzeichnis gefunden

## Architektur

```
pdf_extractor/
├── __init__.py          # Öffentliche API
├── extractor.py         # Hauptextraktor (PDFExtractor)
├── models.py            # Pydantic-Datenmodelle
├── prompts.py           # LLM-Prompts für Extraktion
├── exceptions.py        # Custom Exceptions
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
print(f"Sektionen: {len(result.sections)}")

for section in result.sections:
    print(f"{section.section_number}: {section.content[:100]}...")

# In Datei speichern
result.save("output.json")
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
    model="gpt-4o-mini",       # Günstigeres Modell
    max_retries=5,             # Mehr Wiederholungsversuche
    max_images_per_request=10, # Mehr Bilder pro Request
    temperature=0.0,           # Deterministisch
)

extractor = PDFExtractor(config=config)
result = extractor.extract("dokument.pdf")
```

### Fehlerbehandlung

```python
from pdf_extractor import PDFExtractor, NoTableOfContentsError

extractor = PDFExtractor()

try:
    result = extractor.extract("dokument.pdf")
except NoTableOfContentsError as e:
    print(f"Dokument hat kein Inhaltsverzeichnis: {e}")
except FileNotFoundError as e:
    print(f"PDF nicht gefunden: {e}")
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
            progress_callback: Optionaler Callback für Fortschritt

        Returns:
            ExtractionResult mit Kontext und allen Sektionen

        Raises:
            NoTableOfContentsError: Wenn kein Inhaltsverzeichnis gefunden
            FileNotFoundError: Wenn PDF-Datei nicht existiert
        """
```

### ExtractionResult

Enthält die vollständigen Extraktionsergebnisse.

```python
class ExtractionResult:
    source_file: str                      # Pfad zur Quelldatei
    context: DocumentContext              # Dokumentenkontext
    sections: list[ExtractedSection]      # Extrahierte Sektionen
    processing_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    errors: list[str]
    warnings: list[str]

    # Methoden
    def get_section(self, number: str) -> Optional[ExtractedSection]
    def get_overview(self) -> Optional[ExtractedSection]
    def get_paragraphs(self) -> list[ExtractedSection]
    def get_anlagen(self) -> list[ExtractedSection]
    def get_stats(self) -> dict
    def get_full_content(self) -> str
    def to_dict(self) -> dict
    def to_json(self) -> str
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> ExtractionResult
```

### ExtractedSection

Eine extrahierte Sektion (§, Anlage oder Übersicht).

```python
class ExtractedSection:
    section_type: SectionType             # overview | paragraph | anlage
    section_number: Optional[str]         # z.B. "§ 10", "Anlage 2"
    section_title: Optional[str]          # z.B. "Module und Leistungspunkte"
    content: str                          # Vollständiger Inhalt in natürlicher Sprache
    pages: list[int]                      # Seitenzahlen (1-indiziert)
    chapter: Optional[str]                # Übergeordnetes Kapitel
    paragraphs: list[str]                 # Absatznummern (1), (2) etc.
    internal_references: list[str]        # z.B. ["§ 5 Abs. 2", "Anlage 1"]
    external_references: list[str]        # z.B. ["Allgemeine Bestimmungen"]
    has_table: bool
    has_list: bool
    token_count: int                      # Geschätzte Tokens
    extraction_confidence: float          # 0.0 - 1.0
    extraction_notes: Optional[str]

    # Properties
    @property
    def identifier(self) -> str           # z.B. "§ 10 Module und Leistungspunkte"

    # Methoden
    def get_source_reference(self, doc_title: str = "") -> str
```

### DocumentContext

Dokument-Metadaten aus der Strukturanalyse.

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
    abbreviations: list[Abbreviation]
    key_terms: list[str]
    referenced_documents: list[str]
    legal_basis: Optional[str]
    language: Language
```

### ExtractionConfig

Konfiguration für die Extraktion.

```python
class ExtractionConfig:
    model: str = "gpt-4o"                # OpenAI-Modell
    max_tokens_per_request: int = 4096   # Max Tokens pro Request
    temperature: float = 0.0             # Sampling-Temperatur
    max_images_per_request: int = 5      # Max Bilder pro Request (1-20)
    max_retries: int = 3                 # Wiederholungsversuche
    language: Language = Language.DE
```

## Datenmodelle

### SectionType (Enum)

```python
class SectionType(str, Enum):
    OVERVIEW = "overview"      # Übersicht (vor erstem §)
    PARAGRAPH = "paragraph"    # § mit Nummer
    ANLAGE = "anlage"          # Anlage 1, 2, etc.
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

## Extraktionsprozess

### Phase 1: Strukturanalyse

1. **Erste 5 Seiten**: Titelseite und Inhaltsverzeichnis
2. **Analyse**: Dokumenttyp, Titel, Institution, Abkürzungen
3. **Strukturkarte**: Alle §§ und Anlagen mit Seitenzahlen
4. **Output**: `DocumentContext` + Liste von `StructureEntry`

### Phase 2: Sektionsweise Extraktion

Für JEDE Sektion (§, Anlage, Übersicht):

1. **Alle Seiten der Sektion rendern**
2. **Wenn <= max_images_per_request**: Ein API-Call
3. **Wenn > max_images_per_request**: Sliding Window mit 1-Seiten-Overlap
4. **Inhaltstransformation**:
   - Tabellen → Natürliche Sprache
   - Listen → Fließtext
   - Verweise erhalten
5. **Output**: `ExtractedSection` mit vollständigem Inhalt

### Sliding Window

Für lange Sektionen (z.B. Anlage mit 15 Seiten bei max_images=5):
- Window 1: Seiten 1-5
- Window 2: Seiten 5-9 (Overlap: Seite 5)
- Window 3: Seiten 9-13 (Overlap: Seite 9)
- Window 4: Seiten 13-15 (Overlap: Seite 13)

Die LLM-Antworten werden zusammengeführt, Duplikate entfernt.

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
    "chapters": ["I. Allgemeines", "II. Prüfungen", "Anlage 1: ..."],
    "abbreviations": [{"short": "LP", "long": "Leistungspunkte"}]
  },
  "sections": [
    {
      "section_type": "overview",
      "section_number": null,
      "section_title": "Übersicht",
      "content": "Inhaltsverzeichnis und Präambel...",
      "pages": [1, 2],
      "has_table": false
    },
    {
      "section_type": "paragraph",
      "section_number": "§ 1",
      "section_title": "Geltungsbereich",
      "content": "Diese Prüfungsordnung regelt...",
      "pages": [3],
      "chapter": "I. Allgemeines",
      "paragraphs": ["(1)", "(2)"],
      "internal_references": ["§ 5 Abs. 2"],
      "has_table": false
    },
    {
      "section_type": "anlage",
      "section_number": "Anlage 1",
      "section_title": "Studienverlaufsplan",
      "content": "Im ersten Semester sind die Module...",
      "pages": [25, 26, 27, 28],
      "has_table": true
    }
  ],
  "processing_time_seconds": 120.5,
  "total_input_tokens": 50000,
  "total_output_tokens": 25000,
  "errors": [],
  "warnings": []
}
```

## Weiterverarbeitung

Das Output ist für nachgelagerte Verarbeitung optimiert:

### Für RAG-Systeme

```python
result = ExtractionResult.load("output.json")

for section in result.sections:
    # Jede Sektion ist ein eigenständiges Chunk
    chunk = {
        "content": section.content,
        "metadata": {
            "section_number": section.section_number,
            "section_title": section.section_title,
            "pages": section.pages,
            "source_ref": section.get_source_reference(result.context.title)
        }
    }
    # In Vector DB speichern...
```

### Zugriff auf spezifische Sektionen

```python
# § 10 abrufen
section_10 = result.get_section("§ 10")

# Alle Anlagen
anlagen = result.get_anlagen()

# Statistiken
stats = result.get_stats()
print(f"Paragraphen: {stats['paragraphs']}")
print(f"Anlagen: {stats['anlagen']}")
```

## Exceptions

```python
from pdf_extractor import (
    ExtractionError,           # Basis-Exception
    NoTableOfContentsError,    # Kein Inhaltsverzeichnis
    StructureExtractionError,  # Strukturanalyse fehlgeschlagen
    SectionExtractionError,    # Sektion konnte nicht extrahiert werden
    PageRenderError,           # Seite konnte nicht gerendert werden
    APIError,                  # API-Fehler
)
```

## Kosten

### Geschätzte Kosten (GPT-4o)

| Seitenzahl | Sektionen | Kosten (USD) |
|------------|-----------|--------------|
| 10         | ~5        | ~0.15        |
| 50         | ~25       | ~0.70        |
| 100        | ~50       | ~1.40        |

### Kosten reduzieren

```python
# GPT-4o-mini ist ~15x günstiger
config = ExtractionConfig(model="gpt-4o-mini")
```

## Fehlerbehandlung

### NoTableOfContentsError

Wird geworfen wenn kein Inhaltsverzeichnis gefunden wird. Es gibt KEIN Fallback - das Dokument muss ein Inhaltsverzeichnis haben.

### Retry-Mechanismus

Bei API-Verweigerungen (z.B. "I'm sorry, I can't assist..."):
1. Erster Retry nach 2 Sekunden
2. Zweiter Retry nach 4 Sekunden
3. Dritter Retry nach 8 Sekunden
4. Bei Fehlschlag: Sektion mit `extraction_confidence=0.0` markiert

## Changelog

### v2.0.0

- **BREAKING**: Sektionsbasierte Extraktion statt seitenbasiert
- Neue Modelle: `ExtractedSection`, `StructureEntry`, `SectionType`
- Entfernt: `ExtractedPage`, `SectionMarker`, `ContentType`
- Neue Exception: `NoTableOfContentsError`
- Neuer Parameter: `max_images_per_request`
- Sliding Window für lange Sektionen

### v1.0.0

- Initiale Version mit seitenbasierter Extraktion
