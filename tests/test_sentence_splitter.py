"""Tests for chunking.sentence_splitter."""

from chunking.sentence_splitter import split_sentences


class TestBasicSplitting:
    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   ") == []

    def test_none_input(self):
        assert split_sentences(None) == []

    def test_single_sentence(self):
        result = split_sentences("Dies ist ein Satz.")
        assert result == ["Dies ist ein Satz."]

    def test_two_sentences(self):
        result = split_sentences("Erster Satz. Zweiter Satz.")
        assert result == ["Erster Satz.", "Zweiter Satz."]

    def test_three_sentences(self):
        result = split_sentences(
            "Satz eins. Satz zwei. Satz drei."
        )
        assert len(result) == 3

    def test_question_mark(self):
        result = split_sentences("Was ist das? Es ist ein Test.")
        assert result == ["Was ist das?", "Es ist ein Test."]

    def test_exclamation_mark(self):
        result = split_sentences("Achtung! Dies ist wichtig.")
        assert result == ["Achtung!", "Dies ist wichtig."]

    def test_no_split_lowercase(self):
        """Should not split when next char is lowercase (not a new sentence)."""
        result = split_sentences("Es kostet ca. drei Euro.")
        assert len(result) == 1

    def test_preserves_whitespace_trimming(self):
        result = split_sentences("  Erster Satz.   Zweiter Satz.  ")
        assert result == ["Erster Satz.", "Zweiter Satz."]


class TestGermanAbbreviations:
    def test_abs_no_split(self):
        text = "Gemäß § 5 Abs. 2 ist die Regelung gültig."
        result = split_sentences(text)
        assert len(result) == 1

    def test_nr_no_split(self):
        text = "Dies regelt Nr. 3 der Verordnung."
        result = split_sentences(text)
        assert len(result) == 1

    def test_bzw_no_split(self):
        text = "Sie müssen Module bzw. Kurse belegen."
        result = split_sentences(text)
        assert len(result) == 1

    def test_vgl_no_split(self):
        text = "Die Regelung gilt vgl. Anlage 1."
        result = split_sentences(text)
        assert len(result) == 1

    def test_prof_dr_no_split(self):
        text = "Prof. Dr. Müller leitet den Kurs."
        result = split_sentences(text)
        assert len(result) == 1

    def test_multiple_abbreviations(self):
        text = "Gem. § 5 Abs. 2 Nr. 3 gilt die Regelung. Der Antrag ist zu stellen."
        result = split_sentences(text)
        assert len(result) == 2


class TestMultiPartAbbreviations:
    def test_z_b_no_split(self):
        text = "Es gibt verschiedene Möglichkeiten, z.B. Klausuren oder Hausarbeiten."
        result = split_sentences(text)
        assert len(result) == 1

    def test_d_h_no_split(self):
        text = "Das Modul umfasst 6 LP, d.h. 180 Stunden Arbeitsaufwand."
        result = split_sentences(text)
        assert len(result) == 1

    def test_u_a_no_split(self):
        text = "Kompetenzen umfassen u.a. analytisches Denken."
        result = split_sentences(text)
        assert len(result) == 1

    def test_i_d_r_no_split(self):
        text = "Die Prüfung findet i.d.R. am Ende statt."
        result = split_sentences(text)
        assert len(result) == 1


class TestParagraphReferences:
    def test_paragraph_abs_no_split(self):
        text = "Gemäß § 12 Abs. 3 ist die Prüfung abzulegen."
        result = split_sentences(text)
        assert len(result) == 1

    def test_paragraph_nr_no_split(self):
        text = "Nach § 8 Nr. 2 wird der Antrag gestellt."
        result = split_sentences(text)
        assert len(result) == 1

    def test_paragraph_satz_no_split(self):
        text = "Dies regelt § 3 Satz. 1 der Ordnung."
        result = split_sentences(text)
        assert len(result) == 1


class TestOrdinals:
    def test_ordinal_no_split(self):
        """Ordinal numbers like '1. ' should not cause splits."""
        text = "Im 1. Semester werden Grundlagen gelehrt."
        result = split_sentences(text)
        assert len(result) == 1

    def test_ordinal_in_enumeration(self):
        text = "Es gibt 3. Möglichkeiten zur Wiederholung."
        result = split_sentences(text)
        assert len(result) == 1


class TestEdgeCases:
    def test_sentence_without_final_period(self):
        text = "Erster Satz. Zweiter Satz ohne Punkt"
        result = split_sentences(text)
        assert len(result) == 2
        assert result[1] == "Zweiter Satz ohne Punkt"

    def test_sentence_starting_with_paragraph(self):
        """§ at sentence start should trigger a split."""
        text = "Die Regelung gilt. § 5 beschreibt die Details."
        result = split_sentences(text)
        assert len(result) == 2

    def test_sentence_starting_with_number(self):
        text = "Die Prüfung ist bestanden. 180 Leistungspunkte sind erforderlich."
        result = split_sentences(text)
        assert len(result) == 2

    def test_sentence_with_quotes(self):
        text = 'Der Studiengang heißt "Informatik". Er wird angeboten.'
        result = split_sentences(text)
        assert len(result) == 2

    def test_real_academic_text(self):
        text = (
            "Diese Studien- und Prüfungsordnung ergänzt die Allgemeinen Bestimmungen "
            "für Bachelorstudiengänge an der Philipps-Universität Marburg. "
            "Sie regelt Ziele, Inhalte, Aufbau und Gliederung des Studiums "
            "sowie die Anforderungen und Verfahren der Prüfungsleistungen im "
            "Monobachelorstudiengang Informatik mit dem Abschluss Bachelor of "
            "Science (B.Sc.). Absolventinnen und Absolventen des Bachelorstudiums "
            "verfügen über fundierte Kenntnisse."
        )
        result = split_sentences(text)
        assert len(result) == 3

    def test_multiple_newlines(self):
        text = "Erster Absatz.\n\nZweiter Absatz."
        result = split_sentences(text)
        assert len(result) == 2
