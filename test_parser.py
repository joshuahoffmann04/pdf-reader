#!/usr/bin/env python3
"""
Test script for the document parser.
Tests against the expected structure.
"""

import sys
sys.path.insert(0, '.')

from src.extractor.pdf_extractor import PDFExtractor
from src.parser.document_parser import DocumentParser


def test_parser():
    """Test the document parser against expected structure."""

    # Expected structure
    EXPECTED = {
        "chapters": {
            "I": {"title_contains": "Allgemeines", "sections": [1, 2, 3]},
            "II": {"title_contains": "Studienbezogene", "sections": list(range(4, 18))},  # 4-17
            "III": {"title_contains": "Prüfungsbezogene", "sections": list(range(18, 39))},  # 18-38
            "IV": {"title_contains": "Schlussbestimmungen", "sections": [39, 40]},
        },
        "appendices": {
            "1": {"has_sections": False},
            "2": {"has_sections": False},
            "3": {"has_sections": False},
            "4": {"has_sections": True, "sections": [1, 2, 3, 4]},
            "5": {"has_sections": True, "sections": list(range(1, 11))},  # 1-10
        }
    }

    # Extract text from PDF
    print("Extracting text from PDF...")
    extractor = PDFExtractor()
    pdf_doc = extractor.extract("Pruefungsordnung_BSc_Inf_2024.pdf")
    text = pdf_doc.get_full_text()

    # Parse document
    print("Parsing document...")
    parser = DocumentParser()
    doc = parser.parse(text, title=pdf_doc.title)

    # Validate structure
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)

    errors = []
    warnings = []

    # Check chapters
    print(f"\nChapters found: {len(doc.chapters)}")
    for chapter in doc.chapters:
        section_numbers = [s.number for s in chapter.sections]
        ab_numbers = [s.number for s in chapter.ab_excerpts]
        print(f"  {chapter.id}. {chapter.title}")
        print(f"      Main sections: {section_numbers}")
        print(f"      AB excerpts: {ab_numbers}")

        if chapter.id in EXPECTED["chapters"]:
            expected = EXPECTED["chapters"][chapter.id]

            # Check title
            if expected["title_contains"].lower() not in chapter.title.lower():
                errors.append(f"Chapter {chapter.id}: Expected title containing '{expected['title_contains']}', got '{chapter.title}'")

            # Check sections
            expected_sections = set(expected["sections"])
            actual_sections = set(section_numbers)

            missing = expected_sections - actual_sections
            extra = actual_sections - expected_sections

            if missing:
                errors.append(f"Chapter {chapter.id}: Missing sections: {sorted(missing)}")
            if extra:
                warnings.append(f"Chapter {chapter.id}: Extra sections (might be AB excerpts?): {sorted(extra)}")

    # Check for expected chapters
    found_chapters = {c.id for c in doc.chapters}
    expected_chapters = set(EXPECTED["chapters"].keys())
    missing_chapters = expected_chapters - found_chapters
    if missing_chapters:
        errors.append(f"Missing chapters: {missing_chapters}")

    # Check appendices
    print(f"\nAppendices found: {len(doc.appendices)}")
    for appendix in doc.appendices:
        section_numbers = [s.number for s in appendix.sections]
        print(f"  {appendix.id}: {appendix.title[:50]}...")
        print(f"      Sections: {section_numbers if section_numbers else 'None'}")

        if appendix.number in EXPECTED["appendices"]:
            expected = EXPECTED["appendices"][appendix.number]

            if expected["has_sections"]:
                expected_sections = set(expected["sections"])
                actual_sections = set(section_numbers)

                missing = expected_sections - actual_sections
                extra = actual_sections - expected_sections

                if missing:
                    errors.append(f"Anlage {appendix.number}: Missing sections: {sorted(missing)}")
                if extra:
                    errors.append(f"Anlage {appendix.number}: Extra sections: {sorted(extra)}")
            else:
                if section_numbers:
                    errors.append(f"Anlage {appendix.number}: Should have no sections, but found: {section_numbers}")

    # Check for expected appendices
    found_appendices = {a.number for a in doc.appendices}
    expected_appendices = set(EXPECTED["appendices"].keys())
    missing_appendices = expected_appendices - found_appendices
    if missing_appendices:
        errors.append(f"Missing appendices: {missing_appendices}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    stats = doc.get_statistics()
    print(f"Chapters: {stats['chapters']}")
    print(f"Main sections: {stats['main_sections']} (expected: 40)")
    print(f"AB excerpts: {stats['ab_excerpts']}")
    print(f"Appendices: {stats['appendices']} (expected: 5)")
    print(f"Appendix sections: {stats['appendix_sections']} (expected: 14)")

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n✓ No errors!")

    if warnings:
        print(f"\n⚠ WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    return len(errors) == 0


if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)
