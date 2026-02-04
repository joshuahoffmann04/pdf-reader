"""
German Sentence Splitter for the Chunking Pipeline

Regex-based sentence boundary detection optimized for German academic
and legal texts. Handles common abbreviations (Abs., Nr., z.B., etc.)
without requiring external NLP libraries.

Design:
- Split at sentence-ending punctuation (.!?) followed by whitespace + uppercase
- Protect known German abbreviations from triggering false splits
- Protect common patterns: paragraph references (Section 5 Abs. 2), ordinals (1.)
- No external dependencies (no spaCy, no NLTK)

Usage:
    from chunking.sentence_splitter import split_sentences

    sentences = split_sentences("Dies ist Satz eins. Dies ist Satz zwei.")
    # ["Dies ist Satz eins.", "Dies ist Satz zwei."]
"""

import re

# Placeholder character used to protect dots from sentence splitting.
_DOT_PLACEHOLDER = "\x00"

# German abbreviations that should NOT trigger sentence splits.
_ABBREVIATIONS = {
    # Legal/academic
    "abs", "nr", "art", "lit", "bst",
    # Common German
    "bzw", "usw", "etc", "vgl", "ggf", "ca", "inkl", "exkl",
    "evtl", "bzgl", "betr", "gem", "sog", "insb", "zzgl",
    # Titles
    "dr", "prof", "dipl", "ing", "med", "phil", "rer", "nat",
    # Months (abbreviated)
    "jan", "feb", "m\u00e4r", "apr", "jun", "jul", "aug", "sep",
    "sept", "okt", "nov", "dez",
    # Units / misc
    "std", "tel", "fax", "str", "max", "min", "orig",
}

# Build a single regex that matches any known abbreviation followed by a dot.
# Uses word boundary and lookahead for whitespace to avoid false matches.
_ABBREV_PATTERN = re.compile(
    r"(?<!\w)(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.(?=\s)",
    re.IGNORECASE,
)

# Multi-part abbreviations: z.B., d.h., u.a., o.ae., i.d.R., s.o., u.U.
_MULTI_ABBREV_PATTERN = re.compile(
    r"\b[a-z\u00e4\u00f6\u00fcA-Z\u00c4\u00d6\u00dc]\.(?:[a-z\u00e4\u00f6\u00fcA-Z\u00c4\u00d6\u00dc]\.)+",
    re.UNICODE,
)

# Paragraph references: "Section 5 Abs. 2", "Section 5 Abs. 3", "Section 12 Abs. 1 Nr. 3"
_PARAGRAPH_REF_PATTERN = re.compile(
    r"\u00a7\s*\d+\s+(?:Abs|Nr|Satz)\.\s*\d+",
    re.IGNORECASE,
)

# Ordinal numbers before spaces: "1. ", "23. " -- but only when preceded by
# context that suggests an ordinal (after comma, semicolon, start, or space+number)
_ORDINAL_PATTERN = re.compile(r"(?<=\s)\d{1,3}\.(?=\s)")


def _protect_dots(text: str) -> str:
    """Replace dots in abbreviations and special patterns with placeholders."""
    # Order matters: protect multi-part abbreviations first (z.B. before "B.")
    text = _MULTI_ABBREV_PATTERN.sub(
        lambda m: m.group().replace(".", _DOT_PLACEHOLDER), text
    )
    text = _PARAGRAPH_REF_PATTERN.sub(
        lambda m: m.group().replace(".", _DOT_PLACEHOLDER), text
    )
    text = _ABBREV_PATTERN.sub(
        lambda m: m.group().replace(".", _DOT_PLACEHOLDER), text
    )
    text = _ORDINAL_PATTERN.sub(
        lambda m: m.group().replace(".", _DOT_PLACEHOLDER), text
    )
    return text


def _restore_dots(text: str) -> str:
    """Restore placeholder characters back to dots."""
    return text.replace(_DOT_PLACEHOLDER, ".")


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences at proper sentence boundaries.

    Handles German abbreviations, ordinal numbers, paragraph references,
    and multi-part abbreviations (z.B., d.h., etc.) without false splits.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentence strings. Empty/whitespace input returns empty list.
        Each sentence is stripped of leading/trailing whitespace.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Step 1: Protect dots that are NOT sentence boundaries
    protected = _protect_dots(text)

    # Step 2: Split at real sentence boundaries
    # Pattern: sentence-ending punctuation (.!?) followed by whitespace
    # and an uppercase letter, digit, opening quote/bracket.
    parts = re.split(
        r'(?<=[.!?])\s+(?=[A-Z\u00c4\u00d6\u00dc0-9\u201e"\(\u00a7])',
        protected,
    )

    # Step 3: Restore dots and clean up
    sentences = []
    for part in parts:
        restored = _restore_dots(part).strip()
        if restored:
            sentences.append(restored)

    return sentences
