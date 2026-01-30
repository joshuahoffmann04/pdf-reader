"""
Unit tests for the DocumentParser module.
"""

import pytest

from src.parser.document_parser import DocumentParser, Section, Chapter, Appendix


class TestDocumentParser:
    """Tests for DocumentParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return DocumentParser()

    def test_chapter_pattern_recognition(self, parser):
        """Test that chapter patterns are correctly recognized."""
        text = """
4
I.
Allgemeines
§ 1
Geltungsbereich
Some content here.

II.
Studienbezogene Bestimmungen
§ 4
Zugangsvoraussetzungen
More content.
"""
        doc = parser.parse(text)

        assert len(doc.chapters) == 2
        assert doc.chapters[0].id == "I"
        assert doc.chapters[0].title == "Allgemeines"
        assert doc.chapters[1].id == "II"
        assert "Studienbezogene" in doc.chapters[1].title

    def test_section_parsing_within_chapter(self, parser):
        """Test that sections are correctly parsed within chapters."""
        text = """
4
I.
Allgemeines
§ 1
Geltungsbereich
Content of section 1.

§ 2
Ziele des Studiums
Content of section 2.

§ 3
Bachelorgrad
Content of section 3.
"""
        doc = parser.parse(text)

        assert len(doc.chapters) == 1
        chapter = doc.chapters[0]
        assert len(chapter.sections) == 3

        assert chapter.sections[0].id == "§1"
        assert chapter.sections[0].title == "Geltungsbereich"
        assert chapter.sections[1].id == "§2"
        assert chapter.sections[2].id == "§3"

    def test_ab_excerpt_detection(self, parser):
        """Test that AB excerpts are correctly detected."""
        text = """
4
I.
Allgemeines
§ 6
Strukturvariante des Studiengangs
Der Studiengang ist ein Monobachelorstudiengang.

Textauszug aus den Allgemeinen Bestimmungen:
§ 6
Strukturvarianten von Studiengängen
AB content here.

§ 7
Studium
Main content continues.
"""
        doc = parser.parse(text)

        chapter = doc.chapters[0]

        # §6 and §7 should be main sections
        main_ids = [s.id for s in chapter.sections]
        assert "§6" in main_ids
        assert "§7" in main_ids

        # There should be an AB excerpt for §6
        ab_ids = [s.id for s in chapter.ab_excerpts]
        assert "§6" in ab_ids

    def test_toc_entries_filtered(self, parser):
        """Test that ToC entries are not parsed as sections."""
        text = """
Inhaltsverzeichnis
§ 1 Geltungsbereich ...................................................................... 4
§ 2 Ziele des Studiums ................................................................. 4

4
I.
Allgemeines
§ 1
Geltungsbereich
Real content here.
"""
        doc = parser.parse(text)

        # Should only have one §1 (from the main content, not ToC)
        all_sections = doc.get_all_main_sections()
        section_1_count = sum(1 for s in all_sections if s.id == "§1")
        assert section_1_count == 1

    def test_invalid_title_prefix_filtered(self, parser):
        """Test that inline references like '§ 30 Abs. 2' are filtered."""
        text = """
4
I.
Allgemeines
§ 1
Geltungsbereich
Content with reference to § 30 Abs. 2 Allgemeine Bestimmungen.

§ 2
Ziele
More content.
"""
        doc = parser.parse(text)

        all_sections = doc.get_all_main_sections()
        section_ids = [s.id for s in all_sections]

        assert "§1" in section_ids
        assert "§2" in section_ids
        assert "§30" not in section_ids  # Should be filtered

    def test_appendix_parsing(self, parser):
        """Test that appendices are correctly parsed."""
        text = """
4
I.
Allgemeines
§ 1
Geltungsbereich
Content.

Anlage 1: Studienverlaufspläne
Table content here.

Anlage 2: Modulliste
More tables.
"""
        doc = parser.parse(text)

        assert len(doc.appendices) == 2
        assert doc.appendices[0].id == "Anlage 1"
        assert doc.appendices[0].number == "1"
        assert doc.appendices[1].id == "Anlage 2"

    def test_appendix_with_sections(self, parser):
        """Test that appendices with sub-sections are correctly parsed."""
        text = """
4
I.
Allgemeines
§ 1
Geltungsbereich
Content.

Anlage 1: Studienverlaufspläne
Some content.

Anlage 2: Modulliste
More content.

Anlage 3: Importmodulliste
Import content.

Anlage 4: Exportmodulliste
Introduction to export modules.

§ 1
Export curricularer Module
Content of export section 1.

§ 2
Export in Studienbereiche
Content of export section 2.

Anlage 5: Gestreckte Variante
Another appendix.
"""
        doc = parser.parse(text)

        # Find Anlage 4
        anlage_4 = next((a for a in doc.appendices if a.number == "4"), None)
        assert anlage_4 is not None
        assert len(anlage_4.sections) == 2
        assert anlage_4.sections[0].id == "§1"
        assert anlage_4.sections[1].id == "§2"

    def test_monotonic_sequence_detection(self, parser):
        """Test that backward-going section numbers are detected as AB excerpts."""
        text = """
4
I.
Test
§ 1
First
Content 1.

§ 2
Second
Content 2.

§ 3
Third
Content 3.

§ 1
Duplicate One
This should be detected as backward/duplicate.

§ 4
Fourth
Content 4.
"""
        doc = parser.parse(text)
        chapter = doc.chapters[0]

        # Main sections should be 1, 2, 3, 4
        main_numbers = [s.number for s in chapter.sections]
        assert main_numbers == [1, 2, 3, 4]

        # The duplicate §1 should be in ab_excerpts
        ab_numbers = [s.number for s in chapter.ab_excerpts]
        assert 1 in ab_numbers

    def test_statistics(self, parser):
        """Test document statistics."""
        text = """
4
I.
Chapter One
§ 1
Section One
Content.

Anlage 1: First Appendix
Content.
"""
        doc = parser.parse(text)
        stats = doc.get_statistics()

        assert stats["chapters"] == 1
        assert stats["main_sections"] == 1
        assert stats["appendices"] == 1


class TestSection:
    """Tests for Section dataclass."""

    def test_section_creation(self):
        """Test Section creation."""
        section = Section(
            id="§1",
            number=1,
            title="Geltungsbereich",
            content="Test content",
            is_ab_excerpt=False
        )

        assert section.id == "§1"
        assert section.number == 1
        assert section.title == "Geltungsbereich"
        assert not section.is_ab_excerpt


class TestChapter:
    """Tests for Chapter dataclass."""

    def test_chapter_creation(self):
        """Test Chapter creation."""
        chapter = Chapter(
            id="I",
            numeral="I",
            title="Allgemeines",
            sections=[],
            ab_excerpts=[]
        )

        assert chapter.id == "I"
        assert chapter.numeral == "I"
        assert chapter.title == "Allgemeines"


class TestAppendix:
    """Tests for Appendix dataclass."""

    def test_appendix_creation(self):
        """Test Appendix creation."""
        appendix = Appendix(
            id="Anlage 1",
            number="1",
            title="Studienverlaufspläne",
            content="Content here",
            sections=[]
        )

        assert appendix.id == "Anlage 1"
        assert appendix.number == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
