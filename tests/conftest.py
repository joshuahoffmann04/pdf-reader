"""
Pytest fixtures for LLM PDF Processor tests.
"""

import pytest
from pathlib import Path

from src.llm_processor.models import (
    DocumentContext,
    ExtractedPage,
    SectionMarker,
    RAGChunk,
    ChunkMetadata,
    ProcessingConfig,
    DocumentType,
    ChunkType,
    Abbreviation,
)


@pytest.fixture
def sample_config():
    """Create a sample ProcessingConfig."""
    return ProcessingConfig(
        model="gpt-4o",
        target_chunk_size=500,
        max_chunk_size=1000,
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
        main_topics=["Module", "Prüfungen", "Bachelorarbeit"],
        abbreviations=[
            Abbreviation(short="AB", long="Allgemeine Bestimmungen"),
            Abbreviation(short="LP", long="Leistungspunkte"),
        ],
        key_terms=["Modul", "Leistungspunkte", "Klausur", "Bachelorarbeit"],
    )


@pytest.fixture
def sample_page():
    """Create a sample ExtractedPage."""
    return ExtractedPage(
        page_number=5,
        content="§1 Geltungsbereich: Diese Studien- und Prüfungsordnung regelt...",
        sections=[
            SectionMarker(number="§1", title="Geltungsbereich", level=1),
            SectionMarker(number="§2", title="Ziele des Studiums", level=1),
        ],
        paragraph_numbers=["(1)", "(2)"],
        has_table=False,
        has_list=True,
        internal_references=["§5 Abs. 2"],
        external_references=["Allgemeine Bestimmungen"],
        continues_from_previous=False,
        continues_to_next=True,
    )


@pytest.fixture
def sample_chunk():
    """Create a sample RAGChunk."""
    metadata = ChunkMetadata(
        source_document="Pruefungsordnung_2024",
        source_pages=[5, 6],
        document_type=DocumentType.PRUEFUNGSORDNUNG,
        section_number="§10",
        section_title="Module und Leistungspunkte",
        chapter="II. Studienbezogene Bestimmungen",
        chunk_type=ChunkType.SECTION,
        topics=["Module", "Leistungspunkte"],
        keywords=["Modul", "LP", "ECTS"],
        related_sections=["§7", "§11"],
        institution="Philipps-Universität Marburg",
        degree_program="Mathematik B.Sc.",
    )
    return RAGChunk(
        id="pruefungsordnung-10-abc123",
        text="§10 Module und Leistungspunkte: Ein Modul ist eine inhaltlich...",
        metadata=metadata,
    )


@pytest.fixture
def sample_pages():
    """Create a list of sample ExtractedPages."""
    return [
        ExtractedPage(
            page_number=1,
            content="Seite 1 Inhalt...",
            sections=[SectionMarker(number="§1", title="Geltungsbereich", level=1)],
            continues_to_next=True,
        ),
        ExtractedPage(
            page_number=2,
            content="Seite 2 Inhalt...",
            sections=[SectionMarker(number="§2", title="Ziele", level=1)],
            continues_from_previous=True,
            continues_to_next=False,
        ),
        ExtractedPage(
            page_number=3,
            content="§3 Bachelorgrad: (1) Die Bachelorprüfung ist bestanden...",
            sections=[SectionMarker(number="§3", title="Bachelorgrad", level=1)],
            paragraph_numbers=["(1)", "(2)"],
            has_table=True,
        ),
    ]


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
        "content": "§1 Geltungsbereich: Diese Ordnung regelt...",
        "section_numbers": ["§1"],
        "section_titles": ["Geltungsbereich"],
        "paragraph_numbers": ["(1)", "(2)"],
        "has_table": False,
        "has_list": False,
        "has_image": False,
        "internal_references": ["§5"],
        "external_references": ["Allgemeine Bestimmungen"],
        "continues_from_previous": False,
        "continues_to_next": True,
    }


@pytest.fixture
def mock_context_response():
    """Create a mock context analysis response."""
    return {
        "document_type": "pruefungsordnung",
        "title": "Prüfungsordnung Mathematik B.Sc.",
        "institution": "Philipps-Universität Marburg",
        "faculty": "Fachbereich Mathematik und Informatik",
        "version_date": "25.01.2023",
        "degree_program": "Mathematik B.Sc.",
        "chapters": ["I. Allgemeines", "II. Studienbezogene Bestimmungen"],
        "main_topics": ["Module", "Prüfungen"],
        "abbreviations": {"AB": "Allgemeine Bestimmungen", "LP": "Leistungspunkte"},
        "key_terms": ["Modul", "Klausur"],
        "referenced_documents": ["Allgemeine Bestimmungen"],
    }


@pytest.fixture
def pdf_path():
    """Return path to test PDF if available."""
    test_pdf = Path(__file__).parent.parent / "pdfs" / "Pruefungsordnung_BSc_Inf_2024.pdf"
    if test_pdf.exists():
        return test_pdf
    return None
