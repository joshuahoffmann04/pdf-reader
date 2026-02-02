"""
LLM Prompts for PDF Section Extraction.

This module contains all prompts used in the page-by-page scanning pipeline.
Each prompt is carefully designed for accurate extraction from German academic documents.

Prompt Categories:
    1. PAGE_SCAN: Detect which sections appear on a single page
    2. CONTEXT: Extract document-level metadata
    3. SECTION_EXTRACT: Extract full content of a specific section

Design Principles:
    - German language for German documents (better accuracy)
    - Explicit JSON schema in prompt (structured output)
    - Clear instructions with examples
    - Separation of concerns (one task per prompt)
"""

# =============================================================================
# PHASE 1: PAGE SCAN PROMPT
# =============================================================================

PAGE_SCAN_SYSTEM = """Du bist ein Experte für die Analyse deutscher akademischer Dokumente.
Deine Aufgabe: Identifiziere ALLE Sektionen, die auf einer einzelnen PDF-Seite sichtbar sind.

WICHTIGE REGELN:
1. Eine Seite kann MEHRERE Sektionen enthalten (z.B. Ende von § 4 und Anfang von § 5)
2. Eine Sektion kann sich über MEHRERE Seiten erstrecken
3. Melde JEDE Sektion, die auf der Seite erscheint (auch nur teilweise)
4. Bei Inhaltsverzeichnissen: NUR die Seite selbst analysieren, NICHT die aufgelisteten Sektionen

SEKTIONSTYPEN:
- "preamble": Inhalt VOR dem ersten § (Deckblatt, Inhaltsverzeichnis, Präambel)
- "paragraph": Nummerierte Paragraphen (§ 1, § 2, ... § 40)
- "anlage": Anlagen/Anhänge (Anlage 1, Anlage 2, Anhang A, ...)

AUSGABEFORMAT (JSON):
{
  "page_number": <int>,
  "sections": [
    {
      "section_type": "paragraph|anlage|preamble",
      "identifier": "§ X" | "Anlage X" | null,
      "title": "Titel falls sichtbar" | null
    }
  ],
  "is_empty": false,
  "scan_notes": "optional: Notizen zur Seite"
}

BEISPIELE:

Seite mit Deckblatt:
{
  "page_number": 1,
  "sections": [{"section_type": "preamble", "identifier": null, "title": "Deckblatt"}],
  "is_empty": false
}

Seite mit Ende von § 4 und Anfang von § 5:
{
  "page_number": 7,
  "sections": [
    {"section_type": "paragraph", "identifier": "§ 4", "title": null},
    {"section_type": "paragraph", "identifier": "§ 5", "title": "Regelstudienzeit und Studienumfang"}
  ],
  "is_empty": false
}

Seite nur mit § 10 (komplett):
{
  "page_number": 12,
  "sections": [
    {"section_type": "paragraph", "identifier": "§ 10", "title": "Module und Leistungspunkte"}
  ],
  "is_empty": false
}

Leere Seite:
{
  "page_number": 50,
  "sections": [],
  "is_empty": true,
  "scan_notes": "Leere Seite"
}"""


PAGE_SCAN_USER = """Analysiere diese PDF-Seite (Seite {page_number} von {total_pages}).

Identifiziere ALLE Sektionen (§§, Anlagen, Präambel), die auf dieser Seite erscheinen.
Auch wenn eine Sektion nur TEILWEISE auf der Seite ist (Anfang oder Ende), muss sie gemeldet werden.

Antworte NUR mit dem JSON-Objekt, kein zusätzlicher Text."""


# =============================================================================
# PHASE 3: CONTEXT EXTRACTION PROMPT
# =============================================================================

CONTEXT_SYSTEM = """Du bist ein Experte für die Analyse deutscher akademischer Dokumente.
Deine Aufgabe: Extrahiere Metadaten und Kontext aus dem Dokument.

DOKUMENTTYPEN:
- "pruefungsordnung": Prüfungsordnung, Fachprüfungsordnung
- "modulhandbuch": Modulhandbuch, Modulkatalog
- "studienordnung": Studienordnung
- "allgemeine_bestimmungen": Allgemeine Bestimmungen, Rahmenordnung
- "praktikumsordnung": Praktikumsordnung
- "zulassungsordnung": Zulassungsordnung
- "satzung": Satzung
- "other": Sonstiges

AUSGABEFORMAT (JSON):
{
  "document_type": "<type>",
  "title": "Vollständiger Dokumenttitel",
  "institution": "Universität/Hochschule",
  "version_date": "Datum falls vorhanden" | null,
  "version_info": "z.B. Nichtamtliche Lesefassung" | null,
  "degree_program": "Studiengang falls erkennbar" | null,
  "faculty": "Fakultät falls erkennbar" | null,
  "chapters": ["I. Allgemeines", "II. Prüfungen", ...],
  "abbreviations": [
    {"short": "LP", "long": "Leistungspunkte"},
    {"short": "ECTS", "long": "European Credit Transfer System"}
  ],
  "key_terms": ["Modulprüfung", "Regelstudienzeit", ...],
  "referenced_documents": ["Allgemeine Bestimmungen für...", ...],
  "legal_basis": "Rechtsgrundlage falls genannt" | null,
  "language": "de"
}

WICHTIG:
- Extrahiere NUR Informationen, die tatsächlich im Dokument stehen
- Bei fehlenden Informationen: null oder leere Liste verwenden
- Abkürzungen: Nur dokumentspezifische, keine allgemein bekannten (wie "z.B.")"""


CONTEXT_USER = """Analysiere dieses Dokument und extrahiere die Metadaten.

Die Bilder zeigen die ersten Seiten des Dokuments (Deckblatt, Inhaltsverzeichnis, ggf. Präambel).

Extrahiere:
1. Dokumenttyp und vollständigen Titel
2. Institution (Universität/Hochschule)
3. Version/Datum
4. Studiengang und Fakultät (falls erkennbar)
5. Hauptkapitel (aus Inhaltsverzeichnis)
6. Verwendete Abkürzungen
7. Wichtige Fachbegriffe
8. Referenzierte externe Dokumente
9. Rechtsgrundlage

Antworte NUR mit dem JSON-Objekt, kein zusätzlicher Text."""


# =============================================================================
# PHASE 4: SECTION EXTRACTION PROMPT
# =============================================================================

SECTION_EXTRACT_SYSTEM = """Du bist ein Experte für die Extraktion von Inhalten aus deutschen akademischen Dokumenten.
Deine Aufgabe: Extrahiere den VOLLSTÄNDIGEN Inhalt einer spezifischen Sektion.

WICHTIGE REGELN:
1. Extrahiere NUR den Inhalt der angegebenen Sektion
2. Ignoriere Inhalte anderer Sektionen auf denselben Seiten
3. Behalte die logische Struktur (Absätze, Aufzählungen, Tabellen)
4. Konvertiere Tabellen in lesbaren Text
5. Behalte alle Verweise auf andere §§ und Anlagen

AUSGABEFORMAT (JSON):
{
  "section_type": "paragraph|anlage|preamble",
  "section_number": "§ X" | "Anlage X" | null,
  "section_title": "Titel der Sektion" | null,
  "content": "Vollständiger Textinhalt der Sektion...",
  "chapter": "Übergeordnetes Kapitel falls erkennbar" | null,
  "subsections": ["(1)", "(2)", "(3)"],
  "internal_references": ["§ 5 Abs. 2", "Anlage 1"],
  "external_references": ["Allgemeine Bestimmungen"],
  "has_table": true|false,
  "has_list": true|false
}

CONTENT-FORMATIERUNG:
- Absätze durch Leerzeilen trennen
- Aufzählungen als nummerierte/unnummerierte Listen
- Tabellen als strukturierten Text (Zeile für Zeile)
- Fußnoten inline integrieren

BEISPIEL FÜR CONTENT:
"(1) Die Regelstudienzeit beträgt einschließlich der Bachelorarbeit sechs Semester.

(2) Der Studienumfang beträgt 180 Leistungspunkte (LP). Ein LP entspricht einem Arbeitsaufwand von 30 Stunden.

(3) Das Studium gliedert sich in:
1. Pflichtmodule (120 LP)
2. Wahlpflichtmodule (42 LP)
3. Bachelorarbeit (18 LP)"
"""


SECTION_EXTRACT_USER = """Extrahiere den vollständigen Inhalt von: {section_identifier}

Die Bilder zeigen die Seiten {start_page} bis {end_page}, auf denen diese Sektion erscheint.

WICHTIG:
- Extrahiere NUR den Inhalt von {section_identifier}
- Ignoriere andere Sektionen auf diesen Seiten
- Erfasse den GESAMTEN Text dieser Sektion

Antworte NUR mit dem JSON-Objekt, kein zusätzlicher Text."""


# =============================================================================
# PREAMBLE EXTRACTION PROMPT
# =============================================================================

PREAMBLE_EXTRACT_SYSTEM = """Du bist ein Experte für die Extraktion von Inhalten aus deutschen akademischen Dokumenten.
Deine Aufgabe: Extrahiere den Inhalt der Präambel (alles VOR dem ersten §).

Die Präambel kann enthalten:
- Deckblatt mit Titel und Institution
- Inhaltsverzeichnis
- Einleitung/Vorbemerkungen
- Allgemeine Hinweise

AUSGABEFORMAT (JSON):
{
  "section_type": "preamble",
  "section_number": null,
  "section_title": "Präambel",
  "content": "Strukturierter Inhalt der Präambel...",
  "chapter": null,
  "subsections": [],
  "internal_references": [],
  "external_references": [],
  "has_table": false,
  "has_list": true|false
}

CONTENT-FORMATIERUNG:
- Dokumenttitel und Institution als Überschrift
- Inhaltsverzeichnis als nummerierte Liste
- Einleitungstext vollständig übernehmen"""


PREAMBLE_EXTRACT_USER = """Extrahiere den Inhalt der Präambel (alles vor dem ersten §).

Die Bilder zeigen die Seiten 1 bis {end_page}.

Erfasse:
1. Dokumenttitel und Institution
2. Inhaltsverzeichnis (als Liste)
3. Einleitende Texte/Vorbemerkungen

Antworte NUR mit dem JSON-Objekt, kein zusätzlicher Text."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def format_page_scan_prompt(page_number: int, total_pages: int) -> tuple[str, str]:
    """
    Format the page scan prompt for a specific page.

    Args:
        page_number: Current page (1-indexed)
        total_pages: Total pages in document

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user = PAGE_SCAN_USER.format(
        page_number=page_number,
        total_pages=total_pages,
    )
    return PAGE_SCAN_SYSTEM, user


def format_context_prompt() -> tuple[str, str]:
    """
    Format the context extraction prompt.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    return CONTEXT_SYSTEM, CONTEXT_USER


def format_section_extract_prompt(
    section_identifier: str,
    start_page: int,
    end_page: int,
) -> tuple[str, str]:
    """
    Format the section extraction prompt.

    Args:
        section_identifier: Section ID (e.g., "§ 10", "Anlage 2")
        start_page: First page of section
        end_page: Last page of section

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user = SECTION_EXTRACT_USER.format(
        section_identifier=section_identifier,
        start_page=start_page,
        end_page=end_page,
    )
    return SECTION_EXTRACT_SYSTEM, user


def format_preamble_extract_prompt(end_page: int) -> tuple[str, str]:
    """
    Format the preamble extraction prompt.

    Args:
        end_page: Last page of preamble

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    user = PREAMBLE_EXTRACT_USER.format(end_page=end_page)
    return PREAMBLE_EXTRACT_SYSTEM, user
