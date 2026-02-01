"""
Prompt templates for Section-Based PDF Extraction.

These prompts are designed for a two-phase extraction:
1. Structure Analysis: Extract document structure from table of contents
2. Section Extraction: Extract complete section content from page images

Optimized for German academic documents (Prüfungsordnungen, Modulhandbücher).
"""

from typing import Optional
from .models import DocumentContext, StructureEntry, SectionType

# =============================================================================
# PHASE 1: Structure Analysis (Table of Contents)
# =============================================================================

STRUCTURE_ANALYSIS_SYSTEM = """Du bist ein Experte für die Analyse akademischer Dokumente deutscher Universitäten.

DEINE AUFGABE:
Analysiere das Inhaltsverzeichnis und die Dokumentstruktur, um eine vollständige Strukturkarte zu erstellen.

KRITISCHE REGELN:
1. Du MUSST das Inhaltsverzeichnis finden und analysieren
2. Extrahiere JEDEN Paragraphen (§) mit Titel und Seitenzahl
3. Extrahiere JEDE Anlage mit Titel und Seitenzahl
4. Wenn KEIN Inhaltsverzeichnis vorhanden ist, setze "has_toc" auf false

STRUKTURELEMENTE:
1. OVERVIEW (Übersicht): Alles VOR dem ersten § (Titelseite, Inhaltsverzeichnis, Präambel)
2. PARAGRAPH (§): Nummerierte Paragraphen (§ 1, § 2, ..., § 40)
3. ANLAGE: Anlagen am Ende (Anlage 1, Anlage 2, ...)

WICHTIG ZUR SEITENZÄHLUNG:
- Seitenzahlen müssen 1-indiziert sein (erste Seite = 1)
- Achte auf die im Dokument gedruckten Seitenzahlen vs. PDF-Seitenzahlen
- Falls die Nummerierung bei "Seite 1" beginnt, aber das PDF mit Seite 1 startet, nutze die PDF-Seitenzahlen
- Die letzte Seite eines Abschnitts ist die Seite VOR dem nächsten Abschnitt"""

STRUCTURE_ANALYSIS_USER = """Analysiere dieses Dokument und erstelle eine vollständige Strukturkarte.

SCHRITT 1: Finde das Inhaltsverzeichnis
- Suche nach "Inhalt", "Inhaltsverzeichnis", "Gliederung"
- Lies JEDE Zeile des Inhaltsverzeichnisses

SCHRITT 2: Extrahiere Dokumentmetadaten
- Dokumenttyp (pruefungsordnung, modulhandbuch, studienordnung, etc.)
- Titel, Institution, Fachbereich
- Studiengang, Version/Datum
- Abkürzungen (oft in einer Legende oder am Anfang definiert)

SCHRITT 3: Erstelle die Strukturkarte
Für JEDEN Eintrag im Inhaltsverzeichnis:
- section_type: "overview" | "paragraph" | "anlage"
- section_number: "§ 1" | "Anlage 1" | null (für overview)
- section_title: Der Titel des Abschnitts
- start_page: Erste Seite (aus dem Inhaltsverzeichnis)
- end_page: Wird berechnet (Seite vor dem nächsten Abschnitt)

WICHTIG: Die LETZTE Seite des Dokuments ist {total_pages}.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:

```json
{{
  "has_toc": true,
  "context": {{
    "document_type": "pruefungsordnung",
    "title": "Prüfungsordnung für den Studiengang Informatik B.Sc.",
    "institution": "Philipps-Universität Marburg",
    "faculty": "Fachbereich Mathematik und Informatik",
    "degree_program": "Informatik B.Sc.",
    "version_date": "01.10.2023",
    "version_info": "Nichtamtliche Lesefassung",
    "total_pages": {total_pages},
    "abbreviations": [
      {{"short": "LP", "long": "Leistungspunkte"}},
      {{"short": "AB", "long": "Allgemeine Bestimmungen"}},
      {{"short": "SWS", "long": "Semesterwochenstunden"}}
    ],
    "key_terms": ["Modul", "Leistungspunkte", "Prüfungsleistung"],
    "chapters": ["I. Allgemeines", "II. Studienbezogene Bestimmungen"],
    "referenced_documents": ["Allgemeine Bestimmungen vom 01.01.2020"],
    "legal_basis": "HHG § 44 Abs. 1"
  }},
  "structure": [
    {{
      "section_type": "overview",
      "section_number": null,
      "section_title": "Übersicht und Inhaltsverzeichnis",
      "start_page": 1,
      "end_page": 2
    }},
    {{
      "section_type": "paragraph",
      "section_number": "§ 1",
      "section_title": "Geltungsbereich",
      "start_page": 3,
      "end_page": 3
    }},
    {{
      "section_type": "paragraph",
      "section_number": "§ 2",
      "section_title": "Ziele des Studiums",
      "start_page": 3,
      "end_page": 4
    }},
    {{
      "section_type": "anlage",
      "section_number": "Anlage 1",
      "section_title": "Exemplarische Studienverlaufspläne",
      "start_page": 25,
      "end_page": 28
    }}
  ]
}}
```

REGELN FÜR structure[]:
1. ERSTER Eintrag: section_type="overview" für Seiten vor dem ersten §
2. Sortiere nach start_page aufsteigend
3. end_page = start_page des nächsten Abschnitts - 1
4. Letzter Abschnitt: end_page = {total_pages}
5. Wenn mehrere §§ auf einer Seite beginnen: Beide haben dieselbe start_page

WENN KEIN INHALTSVERZEICHNIS:
```json
{{
  "has_toc": false,
  "context": null,
  "structure": null
}}
```

Analysiere das Dokument sorgfältig und gib die vollständige JSON-Antwort aus."""


# =============================================================================
# PHASE 2: Section Extraction
# =============================================================================

SECTION_EXTRACTION_SYSTEM = """Du bist ein Experte für die präzise Extraktion von Dokumenteninhalten.

KONTEXT ZUM DOKUMENT:
{document_context}

DEINE AUFGABE:
Extrahiere den VOLLSTÄNDIGEN Inhalt des angegebenen Abschnitts und wandle ihn in natürliche Sprache um.

KRITISCHE REGELN:

1. PRÄZISION - KEINE HALLUZINATION:
   - Gib NUR Informationen wieder, die TATSÄCHLICH auf den Seiten stehen
   - Erfinde NICHTS hinzu
   - Ändere KEINE Zahlen, Daten, Namen oder Fakten

2. FOKUS AUF DEN RICHTIGEN ABSCHNITT:
   - Du erhältst Bilder mehrerer Seiten
   - Extrahiere NUR den Inhalt des angegebenen Abschnitts
   - Ignoriere Inhalte anderer §§ oder Anlagen auf denselben Seiten
   - Der Abschnitt kann mitten auf einer Seite beginnen oder enden

3. TABELLEN → NATÜRLICHE SPRACHE:
   - Wandle ALLE Tabellen in vollständige, verständliche Sätze um
   - Integriere Spaltenüberschriften in jeden Satz
   - Beispiel Notentabelle: "Laut Notentabelle entsprechen 15 Punkte der Note 0,7 (sehr gut)."
   - Beispiel Modulliste: "Das Modul 'Analysis I' ist ein Pflichtmodul mit 9 Leistungspunkten."
   - Beispiel Studienverlauf: "Im ersten Semester sind die Module Analysis I (9 LP) und Lineare Algebra I (9 LP) vorgesehen."

4. LISTEN → FLIESSTEXT:
   - Wandle Aufzählungen in Fließtext um
   - Behalte die logische Struktur bei

5. ABSATZNUMMERN:
   - Behalte Absatznummern wie (1), (2), (3) im Text
   - Integriere sie natürlich: "Absatz (2) regelt, dass..."

6. VERWEISE:
   - Behalte interne Verweise exakt: "§ 5 Abs. 2", "Anlage 1"
   - Markiere externe Verweise: "gemäß den Allgemeinen Bestimmungen"

7. VOLLSTÄNDIGKEIT:
   - Extrahiere den GESAMTEN Inhalt des Abschnitts
   - Kürze NICHTS, lasse NICHTS weg
   - Bei langen Tabellen: ALLE Zeilen umwandeln"""

SECTION_EXTRACTION_USER = """Extrahiere den Inhalt von: {section_identifier}

DIESER ABSCHNITT:
- Typ: {section_type}
- Nummer: {section_number}
- Titel: {section_title}
- Seiten: {start_page} bis {end_page}

Du siehst die Seiten {visible_pages} des Dokuments.

{continuation_hint}

ANWEISUNGEN:
1. Finde den Beginn des Abschnitts (suche nach "{section_identifier}")
2. Extrahiere ALLEN Inhalt bis zum nächsten Abschnitt oder bis Seite {end_page} endet
3. Wandle Tabellen und Listen in natürliche Sprache um
4. Behalte alle Fakten, Zahlen und Verweise exakt bei

Antworte AUSSCHLIESSLICH im folgenden JSON-Format:

```json
{{
  "content": "Der vollständige Inhalt des Abschnitts in natürlicher Sprache. Alle Tabellen sind in Fließtext umgewandelt. Alle Fakten sind präzise wiedergegeben. Absatznummern wie (1), (2) sind enthalten.",
  "paragraphs": ["(1)", "(2)", "(3)"],
  "chapter": "II. Studienbezogene Bestimmungen",
  "has_table": true,
  "has_list": false,
  "internal_references": ["§ 5 Abs. 2", "Anlage 1", "§ 10"],
  "external_references": ["Allgemeine Bestimmungen", "HHG"],
  "extraction_confidence": 1.0,
  "extraction_notes": null
}}
```

FELDREGELN:
- content: Der vollständige Abschnittsinhalt in natürlicher Sprache
- paragraphs: Alle Absatznummern (1), (2) etc. die im Abschnitt vorkommen
- chapter: Das übergeordnete Kapitel (z.B. "II. Studienbezogene Bestimmungen")
- has_table: true wenn Tabellen vorhanden waren (jetzt in Text umgewandelt)
- has_list: true wenn Listen vorhanden waren
- internal_references: Verweise auf andere Teile des Dokuments
- external_references: Verweise auf externe Dokumente
- extraction_confidence: 1.0 wenn vollständig, 0.8 wenn teilweise, 0.5 wenn problematisch
- extraction_notes: Notizen bei Problemen, sonst null

Extrahiere den Abschnitt sorgfältig und gib die JSON-Antwort aus."""


# =============================================================================
# Continuation Hints for Sliding Window
# =============================================================================

CONTINUATION_HINT_START = """WICHTIG: Dies ist der ERSTE Teil eines langen Abschnitts.
- Beginne mit der Extraktion am Anfang des Abschnitts
- Extrahiere so viel wie möglich von den sichtbaren Seiten
- Der Abschnitt wird auf weiteren Seiten fortgesetzt"""

CONTINUATION_HINT_MIDDLE = """WICHTIG: Dies ist ein MITTLERER Teil eines langen Abschnitts.
- Du siehst Seiten {start_visible} bis {end_visible}
- Die ERSTE Seite (Seite {start_visible}) wurde bereits im vorherigen Teil extrahiert
- Beginne die Extraktion AB Seite {continue_from}
- Ignoriere bereits extrahierte Inhalte von Seite {start_visible}
- Der Abschnitt wird möglicherweise noch weiter fortgesetzt"""

CONTINUATION_HINT_END = """WICHTIG: Dies ist der LETZTE Teil eines langen Abschnitts.
- Du siehst Seiten {start_visible} bis {end_visible}
- Die ERSTE Seite (Seite {start_visible}) wurde bereits im vorherigen Teil extrahiert
- Beginne die Extraktion AB Seite {continue_from}
- Extrahiere bis zum Ende des Abschnitts"""

CONTINUATION_HINT_SINGLE = """Du siehst alle Seiten dieses Abschnitts. Extrahiere den vollständigen Inhalt."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_document_context_string(context: DocumentContext) -> str:
    """Build a context string for section extraction prompts."""
    parts = [
        f"Dokumenttyp: {context.document_type.value}",
        f"Titel: {context.title}",
        f"Institution: {context.institution}",
    ]

    if context.faculty:
        parts.append(f"Fachbereich: {context.faculty}")

    if context.degree_program:
        parts.append(f"Studiengang: {context.degree_program}")

    if context.abbreviations:
        abbrev_strs = [f"{a.short}={a.long}" for a in context.abbreviations]
        if abbrev_strs:
            parts.append(f"Abkürzungen: {', '.join(abbrev_strs[:10])}")

    if context.chapters:
        chapters = context.chapters[:6]
        parts.append(f"Gliederung: {', '.join(chapters)}")

    return "\n".join(parts)


def get_structure_analysis_prompt(total_pages: int) -> tuple[str, str]:
    """
    Get prompts for structure analysis phase.

    Args:
        total_pages: Total number of pages in the document

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return (
        STRUCTURE_ANALYSIS_SYSTEM,
        STRUCTURE_ANALYSIS_USER.format(total_pages=total_pages)
    )


def get_section_extraction_prompts(
    context: DocumentContext,
    entry: StructureEntry,
    visible_pages: list[int],
    is_continuation: bool = False,
    is_final_part: bool = False,
    overlap_page: Optional[int] = None,
) -> tuple[str, str]:
    """
    Get prompts for section extraction phase.

    Args:
        context: Document context
        entry: Structure entry for the section to extract
        visible_pages: List of page numbers being sent (1-indexed)
        is_continuation: True if this is a continuation (sliding window)
        is_final_part: True if this is the final part of a sliding window
        overlap_page: The page number that overlaps with previous window

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Build context string
    context_str = build_document_context_string(context)
    system_prompt = SECTION_EXTRACTION_SYSTEM.format(document_context=context_str)

    # Determine section identifier
    section_identifier = entry.section_number or "Übersicht"
    if entry.section_title:
        section_identifier = f"{section_identifier} {entry.section_title}".strip()

    # Determine continuation hint
    if len(visible_pages) == entry.page_count and not is_continuation:
        # Single window covers entire section
        continuation_hint = CONTINUATION_HINT_SINGLE
    elif not is_continuation:
        # First part of sliding window
        continuation_hint = CONTINUATION_HINT_START
    elif is_final_part:
        # Last part of sliding window
        continuation_hint = CONTINUATION_HINT_END.format(
            start_visible=visible_pages[0],
            end_visible=visible_pages[-1],
            continue_from=overlap_page + 1 if overlap_page else visible_pages[1],
        )
    else:
        # Middle part of sliding window
        continuation_hint = CONTINUATION_HINT_MIDDLE.format(
            start_visible=visible_pages[0],
            end_visible=visible_pages[-1],
            continue_from=overlap_page + 1 if overlap_page else visible_pages[1],
        )

    # Format visible pages string
    if len(visible_pages) == 1:
        visible_pages_str = str(visible_pages[0])
    else:
        visible_pages_str = f"{visible_pages[0]} bis {visible_pages[-1]}"

    # Build user prompt
    user_prompt = SECTION_EXTRACTION_USER.format(
        section_identifier=section_identifier,
        section_type=entry.section_type.value,
        section_number=entry.section_number or "Übersicht",
        section_title=entry.section_title or "Einführung und Inhaltsverzeichnis",
        start_page=entry.start_page,
        end_page=entry.end_page,
        visible_pages=visible_pages_str,
        continuation_hint=continuation_hint,
    )

    return system_prompt, user_prompt


def build_context_string(context: dict) -> str:
    """
    Build a context string from a dictionary (legacy compatibility).

    Args:
        context: Dictionary with document context

    Returns:
        Formatted context string
    """
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
        if isinstance(abbrevs, list):
            abbrev_strs = [
                f"{a.get('short', '')}={a.get('long', '')}"
                for a in abbrevs if isinstance(a, dict)
            ]
        elif isinstance(abbrevs, dict):
            abbrev_strs = [f"{k}={v}" for k, v in abbrevs.items()]
        else:
            abbrev_strs = []
        if abbrev_strs:
            parts.append(f"Abkürzungen: {', '.join(abbrev_strs[:10])}")

    if context.get('chapters'):
        chapters = context['chapters'][:6]
        parts.append(f"Gliederung: {', '.join(chapters)}")

    return "\n".join(parts)
