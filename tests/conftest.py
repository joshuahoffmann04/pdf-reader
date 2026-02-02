"""
Pytest fixtures for PDF Extractor tests.
"""

import pytest
from pathlib import Path

from pdf_extractor import (
    DocumentContext,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    ExtractionConfig,
    DocumentType,
    SectionType,
    Abbreviation,
)


@pytest.fixture
def sample_config():
    """Create a sample ExtractionConfig."""
    return ExtractionConfig(
        model="gpt-4o",
        max_retries=3,
        max_images_per_request=5,
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
    """Create a sample StructureEntry."""
    return StructureEntry(
        section_type=SectionType.PARAGRAPH,
        section_number="§ 1",
        section_title="Geltungsbereich",
        start_page=3,
        end_page=4,
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
        paragraphs=["(1)", "(2)"],
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
            section_type=SectionType.OVERVIEW,
            section_number=None,
            section_title="Übersicht",
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
            paragraphs=["(1)", "(2)"],
        ),
        ExtractedSection(
            section_type=SectionType.PARAGRAPH,
            section_number="§ 2",
            section_title="Ziele des Studiums",
            content="§ 2 Ziele des Studiums: Das Studium vermittelt...",
            pages=[3, 4],
            chapter="I. Allgemeines",
            paragraphs=["(1)", "(2)", "(3)"],
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
def sample_result(sample_context, sample_sections):
    """Create a sample ExtractionResult."""
    return ExtractionResult(
        source_file="test.pdf",
        context=sample_context,
        sections=sample_sections,
        processing_time_seconds=10.5,
        total_input_tokens=5000,
        total_output_tokens=2500,
    )


@pytest.fixture
def mock_structure_response():
    """Create a mock structure analysis response."""
    return {
        "has_toc": True,
        "page_offset": 0,
        "context": {
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
        },
        "structure": [
            {"section_type": "overview", "section_number": None, "section_title": "Übersicht", "start_page": 1},
            {"section_type": "paragraph", "section_number": "§ 1", "section_title": "Geltungsbereich", "start_page": 3},
            {"section_type": "paragraph", "section_number": "§ 2", "section_title": "Ziele des Studiums", "start_page": 4},
        ],
    }


@pytest.fixture
def mock_section_response():
    """Create a mock section extraction response."""
    return {
        "content": "§ 1 Geltungsbereich: Diese Ordnung regelt das Studium im Bachelorstudiengang Mathematik.",
        "paragraphs": ["(1)", "(2)"],
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
    test_pdf = Path(__file__).parent.parent / "pdfs" / "Pruefungsordnung_BSc_Inf_2024.pdf"
    if test_pdf.exists():
        return test_pdf
    return None
