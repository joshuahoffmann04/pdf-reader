"""
Pytest fixtures for PDF Extractor tests.
"""

import pytest
from pathlib import Path

from pdf_extractor import (
    # Models
    DocumentContext,
    DocumentStructure,
    SectionLocation,
    PageScanResult,
    DetectedSection,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    ExtractionConfig,
    # Enums
    DocumentType,
    SectionType,
    Language,
    # Helpers
    Abbreviation,
)


@pytest.fixture
def sample_config():
    """Create a sample ExtractionConfig."""
    return ExtractionConfig(
        model="gpt-4o",
        max_retries=3,
        max_tokens=4096,
    )


@pytest.fixture
def sample_context():
    """Create a sample DocumentContext."""
    return DocumentContext(
        document_type=DocumentType.PRUEFUNGSORDNUNG,
        title="Prüfungsordnung für den Studiengang Mathematik B.Sc.",
        institution="Philipps-Universität Marburg",
        faculty="Fachbereich Mathematik und Informatik",
        version_date="25. Januar 2023",
        degree_program="Mathematik B.Sc.",
        total_pages=56,
        chapters=[
            "I. Allgemeines",
            "II. Studienbezogene Bestimmungen",
            "III. Prüfungsbezogene Bestimmungen",
            "IV. Schlussbestimmungen",
        ],
        abbreviations=[
            Abbreviation(short="AB", long="Allgemeine Bestimmungen"),
            Abbreviation(short="LP", long="Leistungspunkte"),
        ],
        key_terms=["Modul", "Leistungspunkte", "Klausur", "Bachelorarbeit"],
    )


@pytest.fixture
def sample_structure_entry():
    """Create a sample StructureEntry (legacy)."""
    return StructureEntry(
        section_type=SectionType.PARAGRAPH,
        section_number="§ 1",
        section_title="Geltungsbereich",
        start_page=3,
        end_page=4,
    )


@pytest.fixture
def sample_section_location():
    """Create a sample SectionLocation."""
    return SectionLocation(
        section_type=SectionType.PARAGRAPH,
        identifier="§ 1",
        title="Geltungsbereich",
        pages=[3, 4],
    )


@pytest.fixture
def sample_detected_section():
    """Create a sample DetectedSection."""
    return DetectedSection(
        section_type=SectionType.PARAGRAPH,
        identifier="§ 1",
        title="Geltungsbereich",
    )


@pytest.fixture
def sample_page_scan_result(sample_detected_section):
    """Create a sample PageScanResult."""
    return PageScanResult(
        page_number=3,
        sections=[sample_detected_section],
        is_empty=False,
    )


@pytest.fixture
def sample_section():
    """Create a sample ExtractedSection."""
    return ExtractedSection(
        section_type=SectionType.PARAGRAPH,
        section_number="§ 1",
        section_title="Geltungsbereich",
        content="§ 1 Geltungsbereich: Diese Studien- und Prüfungsordnung regelt das Studium...",
        pages=[3, 4],
        chapter="I. Allgemeines",
        subsections=["(1)", "(2)"],
        internal_references=["§ 5 Abs. 2"],
        external_references=["Allgemeine Bestimmungen"],
        has_table=False,
        has_list=True,
    )


@pytest.fixture
def sample_sections():
    """Create a list of sample ExtractedSections."""
    return [
        ExtractedSection(
            section_type=SectionType.PREAMBLE,
            section_number=None,
            section_title="Präambel",
            content="Inhaltsverzeichnis und Präambel...",
            pages=[1, 2],
        ),
        ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 1",
            section_title="Geltungsbereich",
            content="§ 1 Geltungsbereich: Diese Ordnung regelt...",
            pages=[3],
            chapter="I. Allgemeines",
            subsections=["(1)", "(2)"],
        ),
        ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 2",
            section_title="Ziele des Studiums",
            content="§ 2 Ziele des Studiums: Das Studium vermittelt...",
            pages=[3, 4],
            chapter="I. Allgemeines",
            subsections=["(1)", "(2)", "(3)"],
            has_table=True,
        ),
        ExtractedSection(
            section_type=SectionType.ANLAGE,
            section_number="Anlage 1",
            section_title="Studienverlaufsplan",
            content="Anlage 1 zeigt den exemplarischen Studienverlaufsplan...",
            pages=[50, 51, 52],
            has_table=True,
        ),
    ]


@pytest.fixture
def sample_section_locations():
    """Create a list of sample SectionLocations."""
    return [
        SectionLocation(
            section_type=SectionType.PREAMBLE,
            identifier=None,
            title="Präambel",
            pages=[1, 2],
        ),
        SectionLocation(
            section_type=SectionType.PARAGRAPH,
            identifier="§ 1",
            title="Geltungsbereich",
            pages=[3],
        ),
        SectionLocation(
            section_type=SectionType.PARAGRAPH,
            identifier="§ 2",
            title="Ziele des Studiums",
            pages=[3, 4],
        ),
        SectionLocation(
            section_type=SectionType.ANLAGE,
            identifier="Anlage 1",
            title="Studienverlaufsplan",
            pages=[50, 51, 52],
        ),
    ]


@pytest.fixture
def sample_structure(sample_section_locations):
    """Create a sample DocumentStructure."""
    return DocumentStructure(
        sections=sample_section_locations,
        total_pages=56,
        has_preamble=True,
    )


@pytest.fixture
def sample_result(sample_context, sample_structure, sample_sections):
    """Create a sample ExtractionResult."""
    return ExtractionResult(
        source_file="test.pdf",
        context=sample_context,
        structure=sample_structure,
        sections=sample_sections,
        processing_time_seconds=10.5,
        total_input_tokens=5000,
        total_output_tokens=2500,
    )


@pytest.fixture
def mock_page_scan_response():
    """Create a mock page scan response."""
    return {
        "page_number": 3,
        "sections": [
            {"section_type": "paragraph", "identifier": "§ 1", "title": "Geltungsbereich"}
        ],
        "is_empty": False,
        "scan_notes": None,
    }


@pytest.fixture
def mock_context_response():
    """Create a mock context extraction response."""
    return {
        "document_type": "pruefungsordnung",
        "title": "Prüfungsordnung Mathematik B.Sc.",
        "institution": "Philipps-Universität Marburg",
        "faculty": "Fachbereich Mathematik und Informatik",
        "version_date": "25.01.2023",
        "degree_program": "Mathematik B.Sc.",
        "chapters": ["I. Allgemeines", "II. Studienbezogene Bestimmungen"],
        "abbreviations": [
            {"short": "AB", "long": "Allgemeine Bestimmungen"},
            {"short": "LP", "long": "Leistungspunkte"},
        ],
        "key_terms": ["Modul", "Klausur"],
        "referenced_documents": ["Allgemeine Bestimmungen"],
        "language": "de",
    }


@pytest.fixture
def mock_section_response():
    """Create a mock section extraction response."""
    return {
        "section_type": "paragraph",
        "section_number": "§ 1",
        "section_title": "Geltungsbereich",
        "content": "§ 1 Geltungsbereich: Diese Ordnung regelt das Studium im Bachelorstudiengang Mathematik.",
        "subsections": ["(1)", "(2)"],
        "chapter": "I. Allgemeines",
        "has_table": False,
        "has_list": False,
        "internal_references": ["§ 5"],
        "external_references": ["Allgemeine Bestimmungen"],
        "extraction_confidence": 1.0,
        "extraction_notes": None,
    }


@pytest.fixture
def pdf_path():
    """Return path to test PDF if available."""
    test_pdf = Path(__file__).parent.parent / "pdfs" / "stpo_bsc-informatik_25-01-23_lese.pdf"
    if test_pdf.exists():
        return test_pdf
    return None
