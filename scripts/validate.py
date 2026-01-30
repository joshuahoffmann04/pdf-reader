#!/usr/bin/env python3
"""
Validates parsed document structure against expected values.

Usage: python -m scripts.validate
"""

import sys
from pathlib import Path

from src.extractor import PDFExtractor
from src.parser import DocumentParser


# Expected structure for validation
EXPECTED = {
    "chapters": {
        "I": list(range(1, 4)),      # §1-§3
        "II": list(range(4, 18)),    # §4-§17
        "III": list(range(18, 39)),  # §18-§38
        "IV": [39, 40],              # §39-§40
    },
    "appendices": {
        "1": [],
        "2": [],
        "3": [],
        "4": [1, 2, 3, 4],
        "5": list(range(1, 11)),     # §1-§10
    },
}


def validate(pdf_path: str = "Pruefungsordnung_BSc_Inf_2024.pdf") -> bool:
    """Validate parser output against expected structure."""
    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        return False

    # Parse document
    extractor = PDFExtractor()
    text = extractor.extract(pdf_path).get_full_text()
    doc = DocumentParser().parse(text)

    errors = []

    # Validate chapters
    for chapter in doc.chapters:
        actual = [s.number for s in chapter.sections]
        expected = EXPECTED["chapters"].get(chapter.id, [])

        if set(actual) != set(expected):
            errors.append(f"Chapter {chapter.id}: got {actual}, expected {expected}")

    # Validate appendices
    for appendix in doc.appendices:
        actual = [s.number for s in appendix.sections]
        expected = EXPECTED["appendices"].get(appendix.number, [])

        if set(actual) != set(expected):
            errors.append(f"Anlage {appendix.number}: got {actual}, expected {expected}")

    # Summary
    stats = doc.get_statistics()
    print(f"Chapters: {stats['chapters']} | Sections: {stats['main_sections']} | Appendices: {stats['appendices']}")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("\nAll validations passed.")
    return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
