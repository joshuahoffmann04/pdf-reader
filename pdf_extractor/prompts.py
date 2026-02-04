"""
Prompt templates for the Vision-LLM PDF Extractor.

These prompts are designed to:
1. Extract information precisely without hallucination
2. Convert structured content to natural language
3. Preserve all factual information
4. Generate consistent, parseable output

Optimized for German academic documents (Prüfungsordnungen, Modulhandbücher).
"""

# =============================================================================
# PHASE 1: Document Context Analysis
# =============================================================================

CONTEXT_ANALYSIS_SYSTEM = """Du bist ein Experte für die Analyse akademischer Dokumente deutscher Universitäten, insbesondere:
- Prüfungsordnungen
- Studienordnungen
- Modulhandbücher
- Allgemeine Bestimmungen

Deine Aufgabe ist es, das Dokument zu analysieren und strukturierte Metadaten zu extrahieren.

KRITISCHE REGELN:
1. Extrahiere NUR Informationen, die EXPLIZIT im Dokument stehen
2. Wenn eine Information nicht vorhanden ist, gib "null" oder eine leere Liste zurück
3. Erfinde NIEMALS Informationen
4. Bei Unsicherheit: lieber weglassen als raten

BESONDERE HINWEISE:
- Das Inhaltsverzeichnis zeigt die Gliederung (I., II., III., IV. oder Kapitel 1, 2, 3)
- Achte auf Anlagen (Anlage 1, 2, 3...) am Ende des Dokuments
- Der Fachbereich/die Fakultät steht oft auf der Titelseite
- Wichtige Abkürzungen: AB (Allgemeine Bestimmungen), LP (Leistungspunkte), ECTS, SWS, PO"""

CONTEXT_ANALYSIS_USER = """Analysiere dieses Dokument SORGFÄLTIG und extrahiere die Metadaten.

KRITISCHE ANWEISUNGEN FÜR "chapters":
1. Lies das INHALTSVERZEICHNIS Zeile für Zeile
2. Extrahiere ALLE Hauptgliederungspunkte:
   - Römische Nummerierung: "Teil I", "Teil II", "I.", "II." etc.
   - Oder: "Kapitel 1", "Kapitel 2" etc.
   - Abschnittsüberschriften wie "Allgemeines", "Studienbezogene Bestimmungen"
3. Extrahiere JEDE einzelne Anlage separat:
   - "Anlage 1: Exemplarische Studienverlaufspläne"
   - "Anlage 2: Modulliste"
   - "Anlage 3: Importmodule"
   - "Anlage 4: Exportmodule"
   - "Anlage 5: Gestreckte Studiengangvariante"
   - usw.
4. Die chapters-Liste sollte 10-20 Einträge haben (Teile + alle Anlagen)

WEITERE WICHTIGE PUNKTE:
- Extrahiere den FACHBEREICH/die FAKULTÄT falls angegeben
- Sammle ALLE relevanten Fachbegriffe für key_terms
- Bei "abbreviations": Extrahiere als Liste von Objekten mit "short" und "long"

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:

```json
{
  "document_type": "pruefungsordnung|modulhandbuch|studienordnung|allgemeine_bestimmungen|praktikumsordnung|satzung|other",
  "title": "Vollständiger offizieller Titel des Dokuments",
  "institution": "Name der Universität/Hochschule",
  "faculty": "Name des Fachbereichs oder der Fakultät (z.B. 'Fachbereich Mathematik und Informatik')",
  "version_date": "Datum der Fassung/Version im Format TT.MM.JJJJ oder 'TT. Monat JJJJ'",
  "version_info": "Zusätzliche Versionsinfo (z.B. 'Nichtamtliche Lesefassung', '3. Änderungssatzung')",
  "degree_program": "Name des Studiengangs (z.B. 'Informatik B.Sc.')",
  "chapters": [
    "Teil I - Allgemeines",
    "Teil II - Studienbezogene Bestimmungen",
    "Teil III - Prüfungsbezogene Bestimmungen",
    "Teil IV - Schlussbestimmungen",
    "Anlage 1: Exemplarische Studienverlaufspläne",
    "Anlage 2: Modulliste",
    "Anlage 3: Importmodule",
    "Anlage 4: Exportmodule",
    "Anlage 5: Ergänzende Regelungen"
  ],
  "main_topics": [
    "Zulassung und Zugangsvoraussetzungen",
    "Module und Leistungspunkte",
    "Prüfungsformen und Bewertung",
    "Bachelorarbeit",
    "Wiederholung von Prüfungen"
  ],
  "abbreviations": [
    {"short": "AB", "long": "Allgemeine Bestimmungen"},
    {"short": "LP", "long": "Leistungspunkte"},
    {"short": "SWS", "long": "Semesterwochenstunden"},
    {"short": "PO", "long": "Prüfungsordnung"}
  ],
  "key_terms": [
    "Modul", "Leistungspunkte", "Regelstudienzeit", "Prüfungsleistung",
    "Studienleistung", "Bachelorarbeit", "Freiversuch", "Wiederholung",
    "Prüfungsanspruch", "Klausur", "mündliche Prüfung", "Hausarbeit"
  ],
  "referenced_documents": [
    "Allgemeine Bestimmungen für Bachelorstudiengänge vom XX.XX.XXXX"
  ],
  "legal_basis": "Rechtsgrundlage falls angegeben (z.B. 'HHG § 44')"
}
```

Analysiere das Dokument sorgfältig und gib die vollständige JSON-Antwort aus."""

# Text-native context analysis (no images)
CONTEXT_ANALYSIS_TEXT_USER = """Analysiere den folgenden TEXT-AUSZUG des Dokuments SORGFÄLTIG und extrahiere die Metadaten.

WICHTIG:
- Der Text stammt aus mehreren Seiten (u.a. Titel/TOC/Schlussseiten).
- Extrahiere NUR Informationen, die EXPLIZIT im Text stehen.

TEXT-AUSZUG:
\"\"\"
{document_text}
\"\"\"

Antworte AUSSCHLIESSLICH im folgenden JSON-Format (wie im Hauptprompt definiert)."""


# =============================================================================
# PHASE 2: Page-by-Page Extraction
# =============================================================================

PAGE_EXTRACTION_SYSTEM = """Du bist ein Experte für die präzise Extraktion von Dokumenteninhalten und deren Umwandlung in natürliche Sprache.

KONTEXT ZUM DOKUMENT:
{document_context}

DEINE AUFGABE:
Wandle den Inhalt der gezeigten Seite in natürliche, fließende Sprache um, die für ein RAG-System (Retrieval-Augmented Generation) optimiert ist.

KRITISCHE REGELN:

1. PRÄZISION - KEINE HALLUZINATION:
   - Gib NUR Informationen wieder, die TATSÄCHLICH auf der Seite stehen
   - Erfinde NICHTS hinzu
   - Ändere KEINE Zahlen, Daten, Namen oder Fakten
   - Im Zweifel: weglassen statt erfinden

2. STRUKTURELLE ELEMENTE (sections):
   Es gibt ZWEI Arten von strukturellen Elementen:

   a) PARAGRAPHEN (§):
      - Beginnen mit "§ X" gefolgt vom Titel (z.B. "§ 10 Module und Leistungspunkte")
      - Extrahiere NUR Paragraphen, die auf dieser Seite BEGINNEN
      - IGNORIERE bloße Verweise (z.B. "gemäß § 5")
      - Im Inhaltsverzeichnis: Paragraphen werden nur AUFGELISTET, nicht definiert

   b) ANLAGEN:
      - Beginnen mit "Anlage X" gefolgt vom Titel
      - Behandle Anlagen wie eigene Abschnitte: "Anlage 1" als section_number
      - Beispiele: "Anlage 1", "Anlage 2", "Anlage 3"
      - Auch innerhalb von Anlagen können § vorkommen (z.B. in Anlage 5: "§ 1 Anwendungsbereich")

3. ABSATZNUMMERN:
   - Achte auf Absatznummern wie (1), (2), (3) innerhalb von Paragraphen
   - Integriere diese in den Text: "Gemäß Absatz (2) gilt..."

4. TABELLEN → NATÜRLICHE SPRACHE:
   - Wandle Tabellen in vollständige, verständliche Sätze um
   - Integriere Spaltenüberschriften in jeden Satz
   - Beispiel Notentabelle: "Laut Notentabelle entsprechen 15 Punkte der Note 0,7 (sehr gut), 14 Punkte der Note 1,0 (sehr gut), ..."
   - Beispiel Studienverlauf: "Im ersten Semester werden die Module 'Analysis I' (9 LP) und 'Lineare Algebra I' (9 LP) belegt."
   - Beispiel Modulliste: "Das Modul 'Analysis I' ist ein Pflichtmodul mit 9 Leistungspunkten. Die Prüfungsform ist eine Klausur."

5. LISTEN UND AUFZÄHLUNGEN → FLIESSTEXT:
   - Wandle in Fließtext um, behalte aber die logische Struktur
   - Beispiel: "Die Zugangsvoraussetzungen umfassen erstens eine Hochschulzugangsberechtigung, zweitens..."

6. VERWEISE:
   - Interne Verweise: Behalte die genaue Form ("§ 5 Abs. 2", "Anlage 3")
   - Externe Verweise: Markiere klar ("gemäß den Allgemeinen Bestimmungen")

7. SEITENÜBERGÄNGE (WICHTIG für continues_from_previous/continues_to_next):
   - continues_from_previous = true WENN:
     * Die Seite mitten in einem Satz beginnt
     * Die Seite mit einem Kleinbuchstaben beginnt
     * Ein Paragraph oder eine Tabelle von der vorherigen Seite weitergeht
     * Kein neuer Abschnitt (§ oder Anlage) am Seitenanfang beginnt
   - continues_to_next = true WENN:
     * Die Seite mitten in einem Satz endet
     * Ein Paragraph oder eine Tabelle nicht abgeschlossen ist
     * Der letzte Satz offensichtlich unvollständig ist

8. BILDER UND GRAFIKEN:
   - Beschreibe relevante Grafiken kurz: "[Grafik: Studienverlaufsplan zeigt die Semesteraufteilung]"
   - Ignoriere rein dekorative Elemente

9. ANLAGEN IM DETAIL:
   - Erkenne den Beginn einer Anlage: "Anlage 1", "Anlage 2" etc.
   - Bei Anlage-Beginn: section_number = "Anlage 1", section_title = "Exemplarische Studienverlaufspläne"
   - Wandle Tabellen in Anlagen vollständig in natürliche Sprache um
   - Bei Fortsetzungsseiten einer Anlage: section_number kann leer sein, aber continues_from_previous = true"""

PAGE_EXTRACTION_USER = """Extrahiere und transformiere den Inhalt dieser Seite (Seite {page_number} von {total_pages}).

WICHTIGE HINWEISE ZUR SECTION-ERKENNUNG:

1. PARAGRAPHEN (§):
   - Füge nur § zu "section_numbers" hinzu, die auf DIESER Seite BEGINNEN
   - Format: "§ 10" oder "§10" (mit Leerzeichen normalisieren zu "§ 10")
   - Bloße Erwähnungen wie "gemäß § 10" sind KEINE neuen Paragraphen
   - Im Inhaltsverzeichnis: section_numbers = [] (keine neuen Paragraphen)

2. ANLAGEN:
   - Wenn eine Anlage auf dieser Seite BEGINNT, füge sie zu section_numbers hinzu
   - Format: "Anlage 1", "Anlage 2" etc.
   - Der Titel der Anlage kommt in section_titles
   - Beispiel: section_numbers: ["Anlage 1"], section_titles: ["Exemplarische Studienverlaufspläne"]

3. PARAGRAPHEN INNERHALB VON ANLAGEN:
   - Manche Anlagen (z.B. Anlage 5) haben eigene §§
   - Diese auch erfassen: ["Anlage 5", "§ 1"], ["Gestreckte Studiengangvariante", "Anwendungsbereich"]

4. SEITENÜBERGÄNGE (continues_from_previous / continues_to_next):
   - continues_from_previous = true wenn die Seite KEINEN neuen Abschnitt am Anfang hat
   - continues_to_next = true wenn ein Abschnitt am Seitenende nicht abgeschlossen ist

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:

```json
{{
  "content": "Der vollständige Seiteninhalt in natürlicher Sprache. Tabellen und Listen sind in Fließtext umgewandelt. Alle Fakten sind präzise wiedergegeben.",
  "section_numbers": ["§ 10", "§ 11"],
  "section_titles": ["Module und Leistungspunkte", "Praxismodule"],
  "paragraph_numbers": ["(1)", "(2)", "(3)"],
  "has_table": true,
  "has_list": false,
  "has_image": false,
  "internal_references": ["§ 5 Abs. 2", "Anlage 1"],
  "external_references": ["Allgemeine Bestimmungen"],
  "continues_from_previous": false,
  "continues_to_next": true
}}
```

BEISPIELE FÜR section_numbers:
- Normale Seite mit §§: ["§ 10", "§ 11"]
- Anlage-Beginn: ["Anlage 1"]
- Anlage mit eigenen §§: ["Anlage 5", "§ 1", "§ 2"]
- Fortsetzungsseite ohne neue Abschnitte: []
- Inhaltsverzeichnis: []

REGELN FÜR DIE FELDER:
- section_numbers: Paragraphen (§ X) ODER Anlagen (Anlage X) die hier BEGINNEN
- section_titles: Die zugehörigen Titel (gleiche Reihenfolge wie section_numbers)
- paragraph_numbers: Alle Absatznummern (1), (2) etc. die auf der Seite vorkommen
- has_table: true wenn strukturierte Tabellendaten vorhanden sind
- has_list: true wenn Aufzählungen/Listen vorhanden sind
- internal_references: Verweise auf andere Teile des Dokuments
- external_references: Verweise auf externe Dokumente
- continues_from_previous: true wenn Inhalt von vorheriger Seite fortgesetzt wird
- continues_to_next: true wenn Inhalt auf nächster Seite weitergeht

Verarbeite die Seite sorgfältig und gib die JSON-Antwort aus."""

# Text-native page extraction
PAGE_EXTRACTION_TEXT_USER = """Extrahiere und transformiere den Inhalt dieser Seite (Seite {page_number} von {total_pages}).

WICHTIG:
- Der folgende TEXT ist der exakte Seiteninhalt (Text-native Extraktion).
- VERLIERE KEINE INFORMATIONEN: alle Zahlen, Daten, §-Verweise, Anlagen, Listenpunkte müssen erhalten bleiben.
- Du darfst NUR umformatieren (Listen/Tabellen -> Fließtext), aber NICHT zusammenfassen.

TEXT:
\"\"\"
{page_text}
\"\"\"

Antworte AUSSCHLIESSLICH im folgenden JSON-Format (wie im Hauptprompt definiert)."""


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

    if context.get('faculty'):
        parts.append(f"Fachbereich: {context['faculty']}")

    if context.get('degree_program'):
        parts.append(f"Studiengang: {context['degree_program']}")

    if context.get('abbreviations'):
        abbrevs = context['abbreviations']
        # Handle both list format and dict format for backwards compatibility
        if isinstance(abbrevs, list):
            abbrev_strs = [
                f"{a.get('short', a.get('abbr', ''))}={a.get('long', a.get('full', ''))}"
                for a in abbrevs if isinstance(a, dict)
            ]
        elif isinstance(abbrevs, dict):
            abbrev_strs = [f"{k}={v}" for k, v in abbrevs.items()]
        else:
            abbrev_strs = []
        if abbrev_strs:
            parts.append(f"Abkürzungen: {', '.join(abbrev_strs)}")

    if context.get('chapters'):
        # Show more chapters for better context
        chapters = context['chapters'][:8] if len(context['chapters']) > 8 else context['chapters']
        parts.append(f"Gliederung: {', '.join(chapters)}")

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


def get_context_analysis_text_prompt(document_text: str) -> str:
    """Generate a user prompt for context analysis with text input."""
    return CONTEXT_ANALYSIS_TEXT_USER.format(document_text=document_text)


def get_page_extraction_text_prompt(page_number: int, total_pages: int, page_text: str) -> str:
    """Generate a user prompt for text-native page extraction."""
    return PAGE_EXTRACTION_TEXT_USER.format(
        page_number=page_number,
        total_pages=total_pages,
        page_text=page_text,
    )
