#!/usr/bin/env python3
"""
Validates parsed document structure.

This script validates the parsing results by checking:
- Correct chapter detection (Roman numerals I-X)
- Monotonic section numbering within chapters
- Appendix detection and sub-sections
- AB excerpt linking

Usage:
    python -m scripts.validate document.pdf
    python -m scripts.validate document.pdf --expected-chapters 4
    python -m scripts.validate document.pdf --verbose
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.extractor import PDFExtractor
from src.parser import DocumentParser, ParsedDocument


@dataclass
class ValidationResult:
    """Result of document validation."""
    valid: bool
    chapters_valid: bool
    sections_valid: bool
    appendices_valid: bool
    ab_linking_valid: bool
    errors: list[str]
    warnings: list[str]
    statistics: dict

    def print_report(self, verbose: bool = False) -> None:
        """Print validation report."""
        print("\n" + "=" * 60)
        print("DOCUMENT VALIDATION REPORT")
        print("=" * 60)

        # Statistics
        stats = self.statistics
        print(f"\nStructure Summary:")
        print(f"  Chapters:        {stats['chapters']}")
        print(f"  Main Sections:   {stats['main_sections']}")
        print(f"  AB Excerpts:     {stats['ab_excerpts']}")
        print(f"  Appendices:      {stats['appendices']}")
        print(f"  Appendix Sections: {stats['appendix_sections']}")

        # Validation status
        print(f"\nValidation:")
        print(f"  Chapters:    {'✓' if self.chapters_valid else '✗'}")
        print(f"  Sections:    {'✓' if self.sections_valid else '✗'}")
        print(f"  Appendices:  {'✓' if self.appendices_valid else '✗'}")
        print(f"  AB Linking:  {'✓' if self.ab_linking_valid else '✗'}")

        # Errors
        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  ✗ {error}")

        # Warnings
        if self.warnings and verbose:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  △ {warning}")

        # Final verdict
        print("\n" + "=" * 60)
        if self.valid:
            print("✓ VALID - Document structure is correct")
        else:
            print("✗ INVALID - Please review errors above")
        print("=" * 60)


def validate_document(
    doc: ParsedDocument,
    expected_chapters: Optional[int] = None,
    expected_sections: Optional[int] = None,
    expected_appendices: Optional[int] = None
) -> ValidationResult:
    """
    Validate parsed document structure.

    Args:
        doc: Parsed document to validate.
        expected_chapters: Expected number of chapters (optional).
        expected_sections: Expected number of main sections (optional).
        expected_appendices: Expected number of appendices (optional).

    Returns:
        ValidationResult with all validation details.
    """
    errors = []
    warnings = []
    stats = doc.get_statistics()

    # 1. Validate chapters
    chapters_valid = True

    if not doc.chapters:
        errors.append("No chapters found")
        chapters_valid = False

    if expected_chapters and stats["chapters"] != expected_chapters:
        errors.append(f"Expected {expected_chapters} chapters, found {stats['chapters']}")
        chapters_valid = False

    # Check chapter order (should be I, II, III, IV, ...)
    expected_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    for i, chapter in enumerate(doc.chapters):
        if i < len(expected_numerals) and chapter.numeral != expected_numerals[i]:
            warnings.append(f"Chapter {i+1} has numeral '{chapter.numeral}', expected '{expected_numerals[i]}'")

    # 2. Validate sections within chapters
    sections_valid = True

    for chapter in doc.chapters:
        if not chapter.sections:
            warnings.append(f"Chapter {chapter.numeral} has no sections")
            continue

        # Check monotonic section numbering
        section_numbers = [s.number for s in chapter.sections]
        for i in range(1, len(section_numbers)):
            if section_numbers[i] <= section_numbers[i-1]:
                errors.append(
                    f"Chapter {chapter.numeral}: Non-monotonic sections "
                    f"§{section_numbers[i-1]} -> §{section_numbers[i]}"
                )
                sections_valid = False

    # Check total sections
    if expected_sections and stats["main_sections"] != expected_sections:
        errors.append(f"Expected {expected_sections} main sections, found {stats['main_sections']}")
        sections_valid = False

    # 3. Validate appendices
    appendices_valid = True

    if expected_appendices and stats["appendices"] != expected_appendices:
        errors.append(f"Expected {expected_appendices} appendices, found {stats['appendices']}")
        appendices_valid = False

    # Check appendix numbering
    for i, appendix in enumerate(doc.appendices):
        expected_num = str(i + 1)
        if appendix.number != expected_num:
            warnings.append(f"Appendix {i+1} has number '{appendix.number}', expected '{expected_num}'")

    # 4. Validate AB linking
    ab_linking_valid = True
    all_ab_excerpts = doc.get_all_ab_excerpts()

    for ab in all_ab_excerpts:
        if not ab.follows_section:
            errors.append(f"AB excerpt {ab.id} is not linked to a main section")
            ab_linking_valid = False

    # Check bidirectional links
    for chapter in doc.chapters:
        for section in chapter.sections:
            for ab_ref in section.ab_references:
                matching_ab = [ab for ab in chapter.ab_excerpts if ab.id == ab_ref]
                if not matching_ab:
                    warnings.append(f"Section {section.id} references {ab_ref} but it's not in chapter")

    # Overall validity
    valid = chapters_valid and sections_valid and appendices_valid and ab_linking_valid

    return ValidationResult(
        valid=valid,
        chapters_valid=chapters_valid,
        sections_valid=sections_valid,
        appendices_valid=appendices_valid,
        ab_linking_valid=ab_linking_valid,
        errors=errors,
        warnings=warnings,
        statistics=stats
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate parsed document structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s document.pdf --expected-chapters 4
  %(prog)s document.pdf -v
        """
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file to validate"
    )
    parser.add_argument(
        "--expected-chapters",
        type=int,
        help="Expected number of chapters"
    )
    parser.add_argument(
        "--expected-sections",
        type=int,
        help="Expected number of main sections"
    )
    parser.add_argument(
        "--expected-appendices",
        type=int,
        help="Expected number of appendices"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show warnings in addition to errors"
    )

    args = parser.parse_args()

    # Check file exists
    if not args.pdf_path.exists():
        print(f"Error: File not found: {args.pdf_path}")
        return 1

    # Parse document
    print(f"Validating: {args.pdf_path.name}")

    try:
        extractor = PDFExtractor()
        pdf_doc = extractor.extract(args.pdf_path)
        doc = DocumentParser().parse(
            pdf_doc.get_full_text(include_page_markers=True),
            title=pdf_doc.title
        )
    except Exception as e:
        print(f"Error parsing document: {e}")
        return 1

    # Validate
    result = validate_document(
        doc,
        expected_chapters=args.expected_chapters,
        expected_sections=args.expected_sections,
        expected_appendices=args.expected_appendices
    )

    # Print report
    result.print_report(verbose=args.verbose)

    return 0 if result.valid else 1


if __name__ == "__main__":
    sys.exit(main())
