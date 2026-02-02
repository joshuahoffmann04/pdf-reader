"""
Prompts for Section-Based PDF Extraction.

Two-phase extraction:
1. Structure Analysis: Extract document structure from table of contents
2. Section Extraction: Extract content of each section

Key principle: All page numbers must be PDF page numbers (1 = first page of PDF),
NOT the printed page numbers in the document.
"""

from .models import DocumentContext, StructureEntry


# =============================================================================
# PHASE 1: Structure Analysis
# =============================================================================

STRUCTURE_SYSTEM = """Du bist ein Experte für die Analyse akademischer Dokumente.

DEINE AUFGABE: Erstelle eine Strukturkarte des Dokuments basierend auf dem Inhaltsverzeichnis.

KRITISCH - SEITENZAHLEN:
Die Bilder sind nummeriert als "PDF-Seite 1", "PDF-Seite 2", etc.
Das Inhaltsverzeichnis zeigt GEDRUCKTE Seitenzahlen (z.B. "§ 5 ... Seite 10").
Diese sind NICHT identisch!

DU MUSST:
1. Den OFFSET berechnen: Wenn auf PDF-Seite 2 die gedruckte "Seite 1" steht, ist der Offset = 1
2. Alle Seitenzahlen in deiner Antwort als PDF-SEITENZAHLEN angeben
3. Formel: PDF-Seitenzahl = Gedruckte Seitenzahl + Offset

Beispiel:
- Du siehst auf PDF-Seite 2 die gedruckte "Seite 1" → Offset = 1
- Im ToC steht "§ 22 ... Seite 15"
- → § 22 beginnt auf PDF-Seite 16 (= 15 + 1)

STRUKTURELEMENTE:
- OVERVIEW: Titelseite, Inhaltsverzeichnis, Präambel (vor dem ersten §)
- PARAGRAPH: Nummerierte Paragraphen (§ 1, § 2, ...)
- ANLAGE: Anlagen am Ende (Anlage 1, Anlage 2, ...)"""


STRUCTURE_USER = """Analysiere die folgenden PDF-Seiten und erstelle eine Strukturkarte.

Das Dokument hat insgesamt {total_pages} PDF-Seiten.

SCHRITT 1: Bestimme den Seitenzahl-Offset
- Finde die erste Seite mit einer gedruckten Seitenzahl
- Berechne: Offset = PDF-Seitennummer - Gedruckte Seitenzahl
- Beispiel: PDF-Seite 3 zeigt "Seite 1" → Offset = 3 - 1 = 2

SCHRITT 2: Lies das Inhaltsverzeichnis
- Finde jeden § und jede Anlage mit ihrer gedruckten Seitenzahl
- Berechne die PDF-Seitenzahl: PDF-Seite = Gedruckte Seite + Offset

SCHRITT 3: Erstelle die Struktur
Für jeden Eintrag: NUR die start_page als PDF-SEITENZAHL angeben.
(Die end_page wird automatisch berechnet)

Antworte NUR mit diesem JSON:

```json
{{
  "has_toc": true,
  "page_offset": 2,
  "context": {{
    "document_type": "pruefungsordnung",
    "title": "Vollständiger Titel",
    "institution": "Name der Universität",
    "faculty": "Fachbereich",
    "degree_program": "Studiengang",
    "version_date": "Datum",
    "abbreviations": [{{"short": "LP", "long": "Leistungspunkte"}}],
    "chapters": ["I. Allgemeines", "II. Prüfungen"]
  }},
  "structure": [
    {{"section_type": "overview", "section_number": null, "section_title": "Übersicht", "start_page": 1}},
    {{"section_type": "paragraph", "section_number": "§ 1", "section_title": "Geltungsbereich", "start_page": 3}},
    {{"section_type": "paragraph", "section_number": "§ 2", "section_title": "Ziele", "start_page": 4}},
    {{"section_type": "anlage", "section_number": "Anlage 1", "section_title": "Modulliste", "start_page": 45}}
  ]
}}
```

REGELN:
- Alle start_page Werte sind PDF-SEITENZAHLEN (1 = erstes Bild)
- Sortiere nach start_page
- KEINE end_page angeben (wird automatisch berechnet)

Falls KEIN Inhaltsverzeichnis vorhanden:
```json
{{"has_toc": false, "page_offset": null, "context": null, "structure": null}}
```"""


# =============================================================================
# PHASE 2: Section Extraction
# =============================================================================

SECTION_SYSTEM = """Du bist ein Experte für präzise Dokumentenextraktion.

DOKUMENT-KONTEXT:
{context}

DEINE AUFGABE: Extrahiere den VOLLSTÄNDIGEN Inhalt des angegebenen Abschnitts.

REGELN:

1. PRÄZISION:
   - Nur Informationen wiedergeben, die auf den Seiten stehen
   - Keine Zahlen, Daten oder Fakten ändern

2. ABSCHNITT FINDEN:
   - Suche nach dem § oder der Anlage mit der angegebenen Nummer
   - Extrahiere nur diesen einen Abschnitt
   - Stoppe beim nächsten § oder bei der nächsten Anlage

3. TABELLEN → TEXT:
   - Wandle Tabellen in vollständige Sätze um
   - "Das Modul Analysis I ist ein Pflichtmodul mit 9 LP."
   - "Im 1. Semester sind Analysis I (9 LP) und Lineare Algebra I (9 LP) vorgesehen."

4. STRUKTUR BEIBEHALTEN:
   - Absatznummern (1), (2), (3) im Text behalten
   - Verweise exakt übernehmen: "§ 5 Abs. 2", "Anlage 1"

5. VOLLSTÄNDIGKEIT:
   - Gesamten Inhalt extrahieren, nichts kürzen
   - Alle Tabellenzeilen umwandeln"""


SECTION_USER = """Extrahiere: {section_id}

ABSCHNITT-INFO:
- Nummer: {section_number}
- Titel: {section_title}
- PDF-Seiten: {start_page} bis {end_page}

Du siehst PDF-Seiten {visible_pages}.
{hint}

AUSGABE als JSON:

```json
{{
  "content": "Vollständiger Inhalt als Fließtext. Tabellen in Sätze umgewandelt.",
  "paragraphs": ["(1)", "(2)"],
  "chapter": "II. Studienbezogene Bestimmungen",
  "has_table": true,
  "has_list": false,
  "internal_references": ["§ 5 Abs. 2", "Anlage 1"],
  "external_references": ["Allgemeine Bestimmungen"],
  "extraction_confidence": 1.0,
  "extraction_notes": null
}}
```"""


# Hints for sliding window extraction
HINT_SINGLE = "Extrahiere den vollständigen Abschnitt."
HINT_FIRST = "Dies ist der ERSTE Teil. Extrahiere ab dem Abschnittsbeginn."
HINT_MIDDLE = "Dies ist ein MITTLERER Teil. Seite {overlap} wurde bereits extrahiert. Beginne ab Seite {continue_from}."
HINT_LAST = "Dies ist der LETZTE Teil. Seite {overlap} wurde bereits extrahiert. Extrahiere ab Seite {continue_from} bis zum Ende."


# =============================================================================
# Helper Functions
# =============================================================================

def get_structure_prompt(total_pages: int) -> tuple[str, str]:
    """Get prompts for structure analysis."""
    return (
        STRUCTURE_SYSTEM,
        STRUCTURE_USER.format(total_pages=total_pages)
    )


def get_section_prompt(
    context: DocumentContext,
    entry: StructureEntry,
    visible_pages: list[int],
    is_continuation: bool = False,
    is_last: bool = False,
    overlap_page: int = None,
) -> tuple[str, str]:
    """Get prompts for section extraction."""

    # Build context string
    ctx_parts = [
        f"Typ: {context.document_type.value}",
        f"Titel: {context.title}",
        f"Institution: {context.institution}",
    ]
    if context.faculty:
        ctx_parts.append(f"Fachbereich: {context.faculty}")
    if context.degree_program:
        ctx_parts.append(f"Studiengang: {context.degree_program}")
    if context.abbreviations:
        abbrevs = ", ".join(f"{a.short}={a.long}" for a in context.abbreviations[:5])
        ctx_parts.append(f"Abkürzungen: {abbrevs}")

    context_str = "\n".join(ctx_parts)
    system = SECTION_SYSTEM.format(context=context_str)

    # Section identifier
    section_id = entry.section_number or "Übersicht"
    if entry.section_title:
        section_id = f"{section_id} {entry.section_title}"

    # Visible pages string
    if len(visible_pages) == 1:
        visible_str = str(visible_pages[0])
    else:
        visible_str = f"{visible_pages[0]} bis {visible_pages[-1]}"

    # Hint for sliding window
    if not is_continuation:
        hint = HINT_SINGLE if len(visible_pages) == entry.page_count else HINT_FIRST
    elif is_last:
        hint = HINT_LAST.format(overlap=overlap_page, continue_from=overlap_page + 1)
    else:
        hint = HINT_MIDDLE.format(overlap=overlap_page, continue_from=overlap_page + 1)

    user = SECTION_USER.format(
        section_id=section_id,
        section_number=entry.section_number or "Übersicht",
        section_title=entry.section_title or "Einführung",
        start_page=entry.start_page,
        end_page=entry.end_page,
        visible_pages=visible_str,
        hint=hint,
    )

    return system, user
