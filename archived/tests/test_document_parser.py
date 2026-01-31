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


class TestPageTracking:
    """Tests for page tracking functionality."""

    @pytest.fixture
    def parser(self):
        return DocumentParser()

    def test_page_markers_extracted(self, parser):
        """Test that page markers are correctly processed."""
        text = """<<<PAGE:1>>>
4
I.
Allgemeines
<<<PAGE:2>>>
§ 1
Geltungsbereich
Content on page 2.
<<<PAGE:3>>>
More content on page 3.

§ 2
Another Section
Content.
"""
        doc = parser.parse(text)
        sections = doc.get_all_main_sections()

        # §1 should be on pages 2 and 3
        s1 = next(s for s in sections if s.id == "§1")
        assert 2 in s1.pages
        assert 3 in s1.pages

    def test_ab_references_linked(self, parser):
        """Test that AB excerpts are linked to preceding main sections."""
        text = """<<<PAGE:1>>>
4
I.
Test Chapter
§ 6
Main Section
Content.

Textauszug aus den Allgemeinen Bestimmungen:
§ 6
AB Section
AB content.

§ 7
Next Main Section
More content.
"""
        doc = parser.parse(text)
        chapter = doc.chapters[0]

        # Main §6 should have ab_references
        main_6 = next(s for s in chapter.sections if s.id == "§6")
        assert "§6" in main_6.ab_references

        # AB §6 should have follows_section
        ab_6 = next(s for s in chapter.ab_excerpts if s.id == "§6")
        assert ab_6.follows_section == "§6"

    def test_page_markers_cleaned_from_content(self, parser):
        """Test that page markers are removed from section content."""
        text = """<<<PAGE:1>>>
4
I.
Test
§ 1
Section
<<<PAGE:2>>>
Content here.
<<<PAGE:3>>>
More content.
"""
        doc = parser.parse(text)
        sections = doc.get_all_main_sections()

        s1 = sections[0]
        assert "<<<PAGE" not in s1.content
        assert "Content here" in s1.content

    def test_ab_marker_not_in_main_section_content(self, parser):
        """Test that AB marker text is not included in main section content."""
        text = """<<<PAGE:1>>>
4
I.
Test Chapter
§ 10
Module und Leistungspunkte
Es gelten die Regelungen des § 10 Allgemeine Bestimmungen.

Textauszug aus den Allgemeinen Bestimmungen:

§ 10
Module und Leistungspunkte
(1) Das Lehrangebot wird in modularer Form angeboten.
(2) Module werden als Pflichtmodule bezeichnet.

§ 11
Next Section
More content.
"""
        doc = parser.parse(text)
        chapter = doc.chapters[0]

        # Find main §10
        main_10 = next(s for s in chapter.sections if s.id == "§10")

        # Main section content should NOT contain the AB marker
        assert "Textauszug aus den Allgemeinen Bestimmungen" not in main_10.content

        # Main section content should only contain the reference text
        assert "Es gelten die Regelungen" in main_10.content

        # AB excerpt should exist and contain the actual AB content
        ab_10 = next(s for s in chapter.ab_excerpts if s.id == "§10")
        assert "(1) Das Lehrangebot" in ab_10.content
        assert ab_10.follows_section == "§10"

    def test_same_section_number_for_main_and_ab(self, parser):
        """Test that same § number works for both main section and AB excerpt."""
        text = """<<<PAGE:1>>>
4
I.
Test
§ 5
Main Five
Content of main §5.

Textauszug aus den Allgemeinen Bestimmungen:

§ 5
AB Five
Content of AB §5.

§ 6
Main Six
Content of main §6.
"""
        doc = parser.parse(text)
        chapter = doc.chapters[0]

        # Main sections should be §5 and §6
        main_ids = [s.id for s in chapter.sections]
        assert "§5" in main_ids
        assert "§6" in main_ids

        # AB excerpt should also be §5
        ab_ids = [s.id for s in chapter.ab_excerpts]
        assert "§5" in ab_ids

        # Check linking
        main_5 = next(s for s in chapter.sections if s.id == "§5")
        ab_5 = next(s for s in chapter.ab_excerpts if s.id == "§5")

        assert "§5" in main_5.ab_references
        assert ab_5.follows_section == "§5"


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

    def test_section_with_pages_and_ab_refs(self):
        """Test Section with pages and AB references."""
        section = Section(
            id="§6",
            number=6,
            title="Strukturvariante",
            content="Content",
            pages=[5, 6],
            ab_references=["§6"]
        )

        assert section.pages == [5, 6]
        assert section.ab_references == ["§6"]

    def test_ab_section_with_follows(self):
        """Test AB excerpt section with follows_section."""
        section = Section(
            id="§6",
            number=6,
            title="AB Section",
            content="AB content",
            is_ab_excerpt=True,
            follows_section="§6"
        )

        assert section.is_ab_excerpt
        assert section.follows_section == "§6"


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
