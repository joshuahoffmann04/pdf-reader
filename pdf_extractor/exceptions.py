"""
Custom Exceptions for PDF Extractor.

This module defines specific exceptions for error handling
in the section-based PDF extraction pipeline.
"""


class ExtractionError(Exception):
    """
    Base exception for all extraction-related errors.

    All custom exceptions in the PDF extractor inherit from this class,
    allowing for broad exception catching when needed.
    """
    pass


class NoTableOfContentsError(ExtractionError):
    """
    Raised when no table of contents is found in the document.

    The section-based extraction requires a table of contents to determine
    the document structure (which sections exist and on which pages).
    Without a ToC, the extraction cannot proceed.

    Example:
        >>> extractor.extract("document_without_toc.pdf")
        NoTableOfContentsError: Kein Inhaltsverzeichnis im Dokument gefunden...
    """

    def __init__(self, message: str = None, document_path: str = None):
        self.document_path = document_path
        self.message = message or (
            "Kein Inhaltsverzeichnis im Dokument gefunden. "
            "Die sektionsbasierte Extraktion benötigt ein Inhaltsverzeichnis, "
            "um die Dokumentstruktur (§§, Anlagen) und deren Seitenzahlen zu erkennen. "
            "Bitte stellen Sie sicher, dass das Dokument ein Inhaltsverzeichnis enthält."
        )
        if document_path:
            self.message = f"{self.message} (Dokument: {document_path})"
        super().__init__(self.message)


class StructureExtractionError(ExtractionError):
    """
    Raised when the document structure cannot be extracted.

    This can happen when:
    - The table of contents is present but malformed
    - The LLM fails to parse the structure correctly
    - Page numbers in the ToC don't match the actual document

    Example:
        >>> extractor.extract("document_with_corrupt_toc.pdf")
        StructureExtractionError: Dokumentstruktur konnte nicht extrahiert werden...
    """

    def __init__(self, message: str = None, details: str = None):
        self.details = details
        self.message = message or (
            "Die Dokumentstruktur konnte nicht extrahiert werden. "
            "Das Inhaltsverzeichnis ist möglicherweise fehlerhaft oder unvollständig."
        )
        if details:
            self.message = f"{self.message} Details: {details}"
        super().__init__(self.message)


class SectionExtractionError(ExtractionError):
    """
    Raised when a specific section cannot be extracted.

    This can happen when:
    - The LLM refuses to process certain pages
    - The section content is corrupted or unreadable
    - API errors occur during extraction

    Attributes:
        section_number: The section that failed (e.g., "§ 10", "Anlage 2")
        pages: The pages that were being processed
    """

    def __init__(
        self,
        message: str = None,
        section_number: str = None,
        pages: list[int] = None
    ):
        self.section_number = section_number
        self.pages = pages or []

        self.message = message or f"Sektion konnte nicht extrahiert werden."
        if section_number:
            self.message = f"{self.section_number}: {self.message}"
        if pages:
            self.message = f"{self.message} (Seiten: {pages})"

        super().__init__(self.message)


class PageRenderError(ExtractionError):
    """
    Raised when a PDF page cannot be rendered to an image.

    This can happen when:
    - The PDF is corrupted
    - The page number is out of range
    - Memory issues during rendering
    """

    def __init__(self, message: str = None, page_number: int = None):
        self.page_number = page_number
        self.message = message or "PDF-Seite konnte nicht gerendert werden."
        if page_number:
            self.message = f"Seite {page_number}: {self.message}"
        super().__init__(self.message)


class APIError(ExtractionError):
    """
    Raised when the OpenAI API returns an error.

    This wraps API-specific errors to provide consistent error handling.
    """

    def __init__(self, message: str = None, original_error: Exception = None):
        self.original_error = original_error
        self.message = message or "API-Fehler bei der Extraktion."
        if original_error:
            self.message = f"{self.message} Original: {str(original_error)}"
        super().__init__(self.message)
