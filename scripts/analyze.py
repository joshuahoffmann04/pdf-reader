#!/usr/bin/env python3
"""
PDF structure analysis tool for debugging and development.

Analyzes: chapter markers, text colors, appendix structure, section distribution.
Usage: python -m scripts.analyze [pdf_path]
"""

import sys
import re
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF


def analyze_structure(pdf_path: str):
    """Analyze and print PDF structure information."""
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text("text") + "\n" for page in doc)

    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(doc)}, Characters: {len(full_text):,}")

    # Chapter markers (I., II., III., IV.)
    print("\n--- Chapters ---")
    for m in re.finditer(r"^(I{1,3}V?|IV)\.\s*\n?\s*([A-ZÄÖÜ][^\n]+)", full_text, re.MULTILINE):
        print(f"  {m.group(1)}. {m.group(2)[:50]}")

    # Section distribution
    print("\n--- Sections ---")
    sections = defaultdict(int)
    for m in re.finditer(r"^§\s*(\d+)", full_text, re.MULTILINE):
        sections[int(m.group(1))] += 1

    # Group by frequency
    unique = [n for n, c in sections.items() if c == 1]
    duplicates = [(n, c) for n, c in sections.items() if c > 1]

    print(f"  Unique (main): §{min(unique)}-§{max(unique)} ({len(unique)} sections)")
    if duplicates:
        print(f"  Duplicates (likely AB): {sorted(duplicates)}")

    # Appendices
    print("\n--- Appendices ---")
    for m in re.finditer(r"^(Anlage|Anhang)\s*(\d+)", full_text, re.MULTILINE | re.IGNORECASE):
        print(f"  {m.group(1)} {m.group(2)} at position {m.start()}")

    # AB markers
    print("\n--- AB Markers ---")
    for m in re.finditer(r"Textauszug\s+aus\s+den\s+Allgemeinen", full_text, re.IGNORECASE):
        context = full_text[m.start():m.start() + 80].replace("\n", " ")
        print(f"  Position {m.start()}: {context}...")

    doc.close()


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "Pruefungsordnung_BSc_Inf_2024.pdf"

    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    analyze_structure(pdf_path)


if __name__ == "__main__":
    main()
