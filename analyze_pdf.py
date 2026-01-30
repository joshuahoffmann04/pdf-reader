#!/usr/bin/env python3
"""
Analyse-Skript zur Untersuchung der PDF-Struktur.
Hilft beim Verstehen von:
1. Abschnitt-Formatierung (I., II., III., IV.)
2. Textfarben (für AB-Auszüge)
3. Anlagen-Struktur
"""

import fitz  # PyMuPDF
import re
from collections import defaultdict


def analyze_chapters(text: str):
    """Finde Abschnitt-Marker (I., II., III., IV.)"""
    print("\n" + "="*60)
    print("ANALYSE: Abschnitte (Chapters)")
    print("="*60)

    # Verschiedene Patterns für römische Nummerierung
    patterns = [
        r'^(I{1,3}V?|IV|V?I{0,3})\.\s+([A-ZÄÖÜ][^\n]+)',  # I. Titel
        r'^(I{1,3}V?|IV|V?I{0,3})\s+([A-ZÄÖÜ][^\n]+)',    # I Titel (ohne Punkt)
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            print(f"\nPattern: {pattern}")
            for m in matches[:10]:  # Erste 10
                print(f"  Position {m.start()}: '{m.group(0)[:60]}...'")


def analyze_colors(pdf_path: str):
    """Analysiere Textfarben im PDF."""
    print("\n" + "="*60)
    print("ANALYSE: Textfarben")
    print("="*60)

    doc = fitz.open(pdf_path)
    color_samples = defaultdict(list)

    # Analysiere erste 10 Seiten
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        color = span.get("color", 0)
                        text = span.get("text", "").strip()
                        if text and len(text) > 3:
                            # Farbe als RGB
                            r = (color >> 16) & 0xFF
                            g = (color >> 8) & 0xFF
                            b = color & 0xFF
                            color_key = f"RGB({r},{g},{b})"

                            if len(color_samples[color_key]) < 3:
                                color_samples[color_key].append({
                                    "page": page_num + 1,
                                    "text": text[:50]
                                })

    doc.close()

    print("\nGefundene Farben:")
    for color, samples in sorted(color_samples.items()):
        print(f"\n  {color}:")
        for s in samples:
            print(f"    Seite {s['page']}: '{s['text']}...'")


def analyze_appendices(text: str):
    """Analysiere Anlagen-Struktur."""
    print("\n" + "="*60)
    print("ANALYSE: Anlagen")
    print("="*60)

    # Finde alle Anlage/Anhang Marker
    pattern = r'^(Anlage|Anhang)\s*(\d+|[A-Z])?[:\s]*([^\n]*)'
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))

    print(f"\nGefundene Anlagen-Marker: {len(matches)}")
    for m in matches:
        pos = m.start()
        # Kontext: 100 Zeichen nach dem Match
        context_end = min(pos + 200, len(text))
        context = text[pos:context_end].replace('\n', ' ')[:100]
        print(f"\n  Position {pos}: '{m.group(0)[:50]}'")
        print(f"    Kontext: '{context}...'")


def analyze_section_distribution(text: str):
    """Zeige Verteilung der § im Text."""
    print("\n" + "="*60)
    print("ANALYSE: §-Verteilung")
    print("="*60)

    # Finde alle §-Marker
    pattern = r'^§\s*(\d+[a-z]?)\s+([A-ZÄÖÜ][^\n]+)'
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    # Gruppiere nach §-Nummer
    by_number = defaultdict(list)
    for m in matches:
        num = m.group(1)
        title = m.group(2).strip()[:40]
        by_number[num].append({
            "position": m.start(),
            "title": title
        })

    print(f"\nGefundene §-Nummern: {len(by_number)}")
    print("\nVerteilung (§-Nummer -> Anzahl Vorkommen):")

    for num in sorted(by_number.keys(), key=lambda x: int(re.match(r'\d+', x).group())):
        occurrences = by_number[num]
        print(f"\n  §{num}: {len(occurrences)} Vorkommen")
        for occ in occurrences:
            print(f"    Pos {occ['position']}: {occ['title']}...")


def find_ab_markers(text: str):
    """Suche nach Hinweisen auf Allgemeine Bestimmungen."""
    print("\n" + "="*60)
    print("ANALYSE: Allgemeine Bestimmungen Marker")
    print("="*60)

    patterns = [
        r'Allgemeine[n]?\s+Bestimmungen',
        r'\(AB\)',
        r'Auszug',
        r'Lesefassung',
    ]

    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        print(f"\nPattern '{pattern}': {len(matches)} Treffer")
        for m in matches[:5]:
            context_start = max(0, m.start() - 30)
            context_end = min(len(text), m.end() + 30)
            context = text[context_start:context_end].replace('\n', ' ')
            print(f"  Pos {m.start()}: '...{context}...'")


def main():
    pdf_path = "Pruefungsordnung_BSc_Inf_2024.pdf"

    # Extrahiere Text
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    doc.close()

    print(f"PDF geladen: {pdf_path}")
    print(f"Textlänge: {len(full_text)} Zeichen")

    # Führe Analysen durch
    analyze_chapters(full_text)
    analyze_colors(pdf_path)
    analyze_appendices(full_text)
    analyze_section_distribution(full_text)
    find_ab_markers(full_text)

    # Speichere den extrahierten Text zur manuellen Analyse
    with open("output/extracted_text_raw.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print("\n\nRoher Text gespeichert in: output/extracted_text_raw.txt")


if __name__ == "__main__":
    main()
