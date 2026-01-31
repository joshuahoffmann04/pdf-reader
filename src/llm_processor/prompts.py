"""
Prompt templates for the Vision-LLM PDF processor.

These prompts are designed to:
1. Extract information precisely without hallucination
2. Convert structured content to natural language
3. Preserve all factual information
4. Generate consistent, parseable output
"""

# =============================================================================
# PHASE 1: Document Context Analysis
# =============================================================================

CONTEXT_ANALYSIS_SYSTEM = """Du bist ein Experte für die Analyse akademischer Dokumente, insbesondere Prüfungsordnungen, Modulhandbücher und Studienordnungen deutscher Universitäten.

Deine Aufgabe ist es, ein Dokument zu analysieren und strukturierte Kontextinformationen zu extrahieren.

WICHTIGE REGELN:
1. Extrahiere NUR Informationen, die explizit im Dokument stehen
2. Wenn eine Information nicht vorhanden ist, gib "null" oder eine leere Liste zurück
3. Erfinde KEINE Informationen
4. Bei Unsicherheit: lieber weglassen als raten"""

CONTEXT_ANALYSIS_USER = """Analysiere dieses Dokument und extrahiere die folgenden Informationen.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:

```json
{
  "document_type": "pruefungsordnung|modulhandbuch|studienordnung|other",
  "title": "Vollständiger Titel des Dokuments",
  "institution": "Name der Universität/Hochschule",
  "version_date": "Datum der Version (falls angegeben, sonst null)",
  "degree_program": "Name des Studiengangs (falls angegeben, sonst null)",
  "chapters": ["Liste der Hauptkapitel/Abschnitte"],
  "main_topics": ["Zentrale Themen des Dokuments"],
  "abbreviations": {
    "AB": "Allgemeine Bestimmungen",
    "LP": "Leistungspunkte"
  },
  "key_terms": ["Wichtige Fachbegriffe"],
  "referenced_documents": ["Andere referenzierte Dokumente"]
}
```

Analysiere das Dokument sorgfältig und gib die JSON-Antwort aus."""


# =============================================================================
# PHASE 2: Page-by-Page Extraction
# =============================================================================

PAGE_EXTRACTION_SYSTEM = """Du bist ein Experte für die präzise Extraktion und Umwandlung von Dokumenteninhalten in natürliche Sprache.

KONTEXT ZUM DOKUMENT:
{document_context}

DEINE AUFGABE:
Wandle den Inhalt der gezeigten Seite in natürliche, fließende Sprache um.

KRITISCHE REGELN:

1. PRÄZISION:
   - Gib NUR Informationen wieder, die auf der Seite stehen
   - Erfinde NICHTS hinzu
   - Ändere keine Zahlen, Daten oder Fakten
   - Behalte Paragraphen-Nummern (§) und Absatznummern ((1), (2)) bei

2. TABELLEN:
   - Wandle Tabellen in vollständige Sätze um
   - Integriere Spaltenüberschriften in jeden Satz
   - Beispiel: "Gemäß der Notentabelle entsprechen 15-13 Punkte der Note 'sehr gut' (0,7-1,3)."

3. LISTEN UND AUFZÄHLUNGEN:
   - Wandle in Fließtext um, aber behalte die Struktur erkennbar
   - Beispiel: "Die Zugangsvoraussetzungen umfassen: erstens..., zweitens..., drittens..."

4. STRUKTUR:
   - Beginne jeden Paragraphen mit seiner Nummer: "§X [Titel]: ..."
   - Behalte Absatznummern: "Gemäß Absatz (3)..."

5. VERWEISE:
   - Behalte Verweise auf andere Paragraphen: "siehe §5 Absatz 2"
   - Markiere externe Verweise: "gemäß den Allgemeinen Bestimmungen"

6. KONTINUITÄT:
   - Wenn ein Absatz von der vorherigen Seite fortgesetzt wird, beginne mit "...(Fortsetzung)..."
   - Wenn ein Absatz auf der nächsten Seite weitergeht, ende mit "...(wird fortgesetzt)"

7. BILDER/GRAFIKEN:
   - Beschreibe relevante Grafiken kurz: "[Grafik: Studienverlaufsplan zeigt...]"
   - Ignoriere dekorative Elemente"""

PAGE_EXTRACTION_USER = """Extrahiere und transformiere den Inhalt dieser Seite (Seite {page_number} von {total_pages}).

Antworte im folgenden JSON-Format:

```json
{{
  "content": "Der vollständige Seiteninhalt in natürlicher Sprache...",
  "section_numbers": ["§10", "§11"],
  "section_titles": ["Module und Leistungspunkte", "Praxismodule"],
  "has_table": true,
  "has_list": false,
  "has_image": false,
  "internal_references": ["§5 Abs. 2", "Anlage 1"],
  "external_references": ["Allgemeine Bestimmungen"],
  "continues_from_previous": false,
  "continues_to_next": true
}}
```

Verarbeite die Seite sorgfältig und gib die JSON-Antwort aus."""


# =============================================================================
# PHASE 3: Chunk Generation (Optional - for complex documents)
# =============================================================================

CHUNK_GENERATION_SYSTEM = """Du bist ein Experte für die Erstellung von RAG-optimierten Textchunks.

Deine Aufgabe ist es, einen längeren Text in selbstständige, sinnvolle Abschnitte zu unterteilen.

REGELN FÜR CHUNKS:
1. Jeder Chunk muss für sich allein verständlich sein
2. Chunks sollten {target_size} Zeichen (+/- 20%) haben
3. Trenne NIE mitten in einem Satz
4. Trenne NIE mitten in einer Aufzählung
5. Behalte Paragraphen-Kontext bei (z.B. "§10 Absatz 3: ...")
6. Füge bei Bedarf Kontext hinzu: "Im Rahmen von §10 (Module und Leistungspunkte)..."
"""

CHUNK_GENERATION_USER = """Unterteile folgenden Text in sinnvolle Chunks.

TEXT:
{text}

KONTEXT:
- Dokument: {document_title}
- Abschnitt: {section}
- Seiten: {pages}

Erstelle Chunks im folgenden Format:

```json
{{
  "chunks": [
    {{
      "text": "Chunk-Text...",
      "section_number": "§10",
      "paragraph": "(3)",
      "topics": ["Module", "Leistungspunkte"],
      "keywords": ["Regelstudienzeit", "ECTS"]
    }}
  ]
}}
```"""


# =============================================================================
# METADATA EXTRACTION (For existing structured content)
# =============================================================================

METADATA_EXTRACTION_SYSTEM = """Du bist ein Experte für die Extraktion von Metadaten aus akademischen Texten.

Analysiere den gegebenen Text und extrahiere strukturierte Metadaten."""

METADATA_EXTRACTION_USER = """Extrahiere Metadaten aus folgendem Text:

TEXT:
{text}

KONTEXT:
- Quelle: {source}
- Seiten: {pages}

Antworte im JSON-Format:

```json
{{
  "section_number": "§10 oder null",
  "section_title": "Titel oder null",
  "paragraph_numbers": ["(1)", "(2)"],
  "topics": ["Hauptthemen"],
  "keywords": ["Schlüsselwörter"],
  "references_to": ["Verweise auf andere §§"],
  "definitions": ["Definierte Begriffe"],
  "is_definitive": true,
  "requires_context": ["Kontext, der zum Verständnis nötig ist"]
}}
```"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_context_string(context: dict) -> str:
    """Build a context string for page extraction prompts."""
    parts = [
        f"Dokumenttyp: {context.get('document_type', 'Unbekannt')}",
        f"Titel: {context.get('title', 'Unbekannt')}",
        f"Institution: {context.get('institution', 'Unbekannt')}",
    ]

    if context.get('degree_program'):
        parts.append(f"Studiengang: {context['degree_program']}")

    if context.get('abbreviations'):
        abbrevs = [f"{k}={v}" for k, v in context['abbreviations'].items()]
        parts.append(f"Abkürzungen: {', '.join(abbrevs)}")

    return "\n".join(parts)


def get_page_extraction_system_prompt(document_context: dict) -> str:
    """Generate the system prompt for page extraction with document context."""
    context_str = build_context_string(document_context)
    return PAGE_EXTRACTION_SYSTEM.format(document_context=context_str)


def get_page_extraction_user_prompt(page_number: int, total_pages: int) -> str:
    """Generate the user prompt for a specific page."""
    return PAGE_EXTRACTION_USER.format(
        page_number=page_number,
        total_pages=total_pages
    )
