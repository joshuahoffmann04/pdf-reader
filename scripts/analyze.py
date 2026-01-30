#!/usr/bin/env python3
"""
PDF structure analysis tool for debugging and development.

Analyzes PDF structure to help understand document patterns:
- Chapter markers (Roman numerals)
- Section distribution (§ paragraphs)
- Appendix structure
- AB excerpt markers

Useful for adapting the parser to new document types.

Usage:
    python -m scripts.analyze document.pdf
    python -m scripts.analyze document.pdf --verbose
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF


def analyze_structure(pdf_path: Path, verbose: bool = False) -> dict:
    """
    Analyze and print PDF structure information.

    Args:
        pdf_path: Path to PDF file.
        verbose: Show detailed output.

    Returns:
        Dictionary with analysis results.
    """
    doc = fitz.open(str(pdf_path))
    full_text = "".join(page.get_text("text") + "\n" for page in doc)

    results = {
        "file": pdf_path.name,
        "pages": len(doc),
        "characters": len(full_text),
        "chapters": [],
        "sections": {},
        "appendices": [],
        "ab_markers": []
    }

    print(f"PDF: {pdf_path.name}")
    print(f"Pages: {len(doc)}, Characters: {len(full_text):,}")

    # Chapter markers (I., II., III., IV., etc.)
    print("\n--- CHAPTERS ---")
    chapter_pattern = re.compile(
        r"^(I{1,3}V?|IV|V?I{1,3})\.\s*\n?\s*([A-ZÄÖÜ][^\n]+)",
        re.MULTILINE
    )
    for m in chapter_pattern.finditer(full_text):
        numeral = m.group(1)
        title = m.group(2)[:60].strip()
        results["chapters"].append({"numeral": numeral, "title": title})
        print(f"  {numeral}. {title}")

    if not results["chapters"]:
        print("  (No Roman numeral chapters found)")

    # Section distribution
    print("\n--- SECTIONS (§) ---")
    section_counts = defaultdict(int)
    for m in re.finditer(r"^§\s*(\d+)", full_text, re.MULTILINE):
        section_counts[int(m.group(1))] += 1

    results["sections"] = dict(section_counts)

    if section_counts:
        unique = [n for n, c in section_counts.items() if c == 1]
        duplicates = [(n, c) for n, c in section_counts.items() if c > 1]

        if unique:
            print(f"  Unique (main): §{min(unique)}-§{max(unique)} ({len(unique)} sections)")
        if duplicates:
            print(f"  Duplicates (likely AB excerpts): {sorted(duplicates)}")

        if verbose:
            print("\n  All sections:")
            for num in sorted(section_counts.keys()):
                count = section_counts[num]
                marker = " (x{})".format(count) if count > 1 else ""
                print(f"    §{num}{marker}")
    else:
        print("  (No § sections found)")

    # Appendices
    print("\n--- APPENDICES ---")
    appendix_pattern = re.compile(
        r"^(Anlage|Anhang)\s*(\d+)",
        re.MULTILINE | re.IGNORECASE
    )
    seen_appendices = set()
    for m in appendix_pattern.finditer(full_text):
        num = m.group(2)
        if num not in seen_appendices:
            seen_appendices.add(num)
            results["appendices"].append({
                "type": m.group(1),
                "number": num,
                "position": m.start()
            })
            print(f"  {m.group(1)} {num}")

    if not results["appendices"]:
        print("  (No appendices found)")

    # AB markers
    print("\n--- AB MARKERS ---")
    ab_pattern = re.compile(
        r"Textauszug\s+aus\s+den\s+Allgemeinen\s+Bestimmungen",
        re.IGNORECASE
    )
    for m in ab_pattern.finditer(full_text):
        results["ab_markers"].append(m.start())
        if verbose:
            context = full_text[m.start():m.start() + 80].replace("\n", " ")
            print(f"  Position {m.start()}: {context}...")

    print(f"  Found: {len(results['ab_markers'])} AB marker(s)")

    # Summary
    print("\n--- SUMMARY ---")
    print(f"  Chapters: {len(results['chapters'])}")
    print(f"  Total § sections: {len(section_counts)}")
    print(f"  Appendices: {len(results['appendices'])}")
    print(f"  AB markers: {len(results['ab_markers'])}")

    doc.close()
    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze PDF structure for debugging and development",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool helps understand the structure of German legal/academic PDFs
to verify parser compatibility or adapt patterns for new document types.

Examples:
  %(prog)s document.pdf
  %(prog)s document.pdf --verbose
        """
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file to analyze"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: File not found: {args.pdf_path}")
        return 1

    if args.pdf_path.suffix.lower() != ".pdf":
        print(f"Error: Not a PDF file: {args.pdf_path}")
        return 1

    try:
        analyze_structure(args.pdf_path, verbose=args.verbose)
        return 0
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
