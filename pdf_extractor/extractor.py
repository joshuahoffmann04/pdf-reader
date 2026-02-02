"""
PDF Section Extractor with Page-by-Page Scanning.

This module implements a four-phase extraction pipeline:

1. PAGE SCAN: Scan each page individually to detect which sections appear on it
2. STRUCTURE: Aggregate scan results to calculate accurate page ranges
3. CONTEXT: Extract document metadata from representative pages
4. EXTRACTION: Extract full content for each section using correct page ranges

Key Innovation:
    Unlike ToC-based approaches, this scanner detects sections on EVERY page.
    This handles cases where sections share pages (e.g., § 4 ends and § 5 starts
    on the same page) - both sections will include that page in their range.

Usage:
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    result = extractor.extract("pruefungsordnung.pdf")

    for section in result.sections:
        print(f"{section.identifier}: {section.content[:100]}...")
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Union

from .api_client import VisionAPIClient
from .exceptions import (
    PDFNotFoundError,
    PageScanError,
    StructureAggregationError,
    ContextExtractionError,
    SectionExtractionError,
)
from .models import (
    # Phase 1
    DetectedSection,
    PageScanResult,
    # Phase 2
    SectionLocation,
    DocumentStructure,
    # Phase 3
    DocumentContext,
    DocumentType,
    Abbreviation,
    Language,
    # Phase 4
    ExtractedSection,
    # Config & Result
    ExtractionConfig,
    ExtractionResult,
    SectionType,
)
from .pdf_utils import PDFRenderer, PageImage
from .prompts import (
    format_page_scan_prompt,
    format_context_prompt,
    format_section_extract_prompt,
    format_preamble_extract_prompt,
)


logger = logging.getLogger(__name__)


# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================


class PDFExtractor:
    """
    PDF Section Extractor using page-by-page scanning.

    This extractor scans every page of a PDF to detect which sections appear
    on each page, then aggregates the results to build accurate page ranges.
    This approach handles cases where sections share pages correctly.

    Usage:
        # Basic usage
        extractor = PDFExtractor()
        result = extractor.extract("document.pdf")

        # With progress callback
        def progress(current, total, message):
            print(f"[{current}/{total}] {message}")

        result = extractor.extract("document.pdf", progress_callback=progress)

        # With custom config
        config = ExtractionConfig(model="gpt-4o-mini", max_retries=5)
        extractor = PDFExtractor(config=config)

    Attributes:
        config: Extraction configuration
        api_client: OpenAI Vision API client
        renderer: PDF page renderer
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the PDF extractor.

        Args:
            config: Extraction configuration (uses defaults if not provided)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.config = config or ExtractionConfig()

        self.api_client = VisionAPIClient(
            api_key=api_key,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
        )

        self.renderer = PDFRenderer(dpi=150)

    def extract(
        self,
        pdf_path: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExtractionResult:
        """
        Extract all sections from a PDF document.

        This is the main entry point. It runs the full four-phase pipeline:
        1. Scan all pages to detect sections
        2. Aggregate results into document structure
        3. Extract document context/metadata
        4. Extract content for each section

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback(current, total, message)
                Called to report progress during extraction.

        Returns:
            ExtractionResult containing all extracted sections and metadata

        Raises:
            PDFNotFoundError: If the PDF file doesn't exist
            StructureAggregationError: If no sections could be detected
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        # Validate PDF
        if not pdf_path.exists():
            raise PDFNotFoundError(str(pdf_path))

        info = self.renderer.get_info(pdf_path)
        total_pages = info.page_count
        logger.info(f"Processing {pdf_path.name}: {total_pages} pages")

        # =====================================================================
        # PHASE 1: Page Scanning
        # =====================================================================
        if progress_callback:
            progress_callback(0, total_pages, "Scanning pages...")

        scan_results = self._phase1_scan_pages(
            pdf_path,
            total_pages,
            progress_callback,
        )

        # =====================================================================
        # PHASE 2: Structure Aggregation
        # =====================================================================
        if progress_callback:
            progress_callback(total_pages, total_pages, "Building structure...")

        structure = self._phase2_aggregate_structure(scan_results, total_pages)

        logger.info(f"Found {len(structure.sections)} sections")
        for section in structure.sections:
            logger.info(f"  {section.display_name}: pages {section.pages}")

        # =====================================================================
        # PHASE 3: Context Extraction
        # =====================================================================
        if progress_callback:
            progress_callback(0, 1, "Extracting document context...")

        try:
            context = self._phase3_extract_context(pdf_path, structure)
        except Exception as e:
            logger.warning(f"Context extraction failed: {e}")
            errors.append(f"Context extraction: {e}")
            context = self._create_fallback_context(pdf_path, total_pages)

        # =====================================================================
        # PHASE 4: Section Extraction
        # =====================================================================
        sections: list[ExtractedSection] = []
        total_sections = len(structure.sections)

        for idx, location in enumerate(structure.sections):
            section_name = location.display_name
            if progress_callback:
                progress_callback(idx + 1, total_sections, f"Extracting {section_name}...")

            try:
                section = self._phase4_extract_section(pdf_path, location)
                sections.append(section)
                logger.info(f"Extracted {section_name}: {len(section.content)} chars")

            except Exception as e:
                logger.error(f"Failed to extract {section_name}: {e}")
                errors.append(f"{section_name}: {e}")
                sections.append(self._create_failed_section(location, str(e)))

            # Rate limiting
            time.sleep(0.3)

        # =====================================================================
        # Build Result
        # =====================================================================
        processing_time = time.time() - start_time
        usage = self.api_client.get_usage_summary()

        # Optionally include scan results for debugging
        if not self.config.include_scan_results:
            structure = DocumentStructure(
                sections=structure.sections,
                total_pages=structure.total_pages,
                has_preamble=structure.has_preamble,
                scan_results=[],  # Clear for smaller output
            )

        result = ExtractionResult(
            source_file=str(pdf_path),
            context=context,
            structure=structure,
            sections=sections,
            processing_time_seconds=processing_time,
            total_input_tokens=usage["input_tokens"],
            total_output_tokens=usage["output_tokens"],
            errors=errors,
            warnings=warnings,
        )

        logger.info(
            f"Extraction complete: {len(sections)} sections, "
            f"{processing_time:.1f}s, "
            f"{usage['total_tokens']} tokens (~${usage['estimated_cost_usd']:.4f})"
        )

        return result

    # =========================================================================
    # PHASE 1: Page Scanning
    # =========================================================================

    def _phase1_scan_pages(
        self,
        pdf_path: Path,
        total_pages: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> list[PageScanResult]:
        """
        Scan each page individually to detect sections.

        Args:
            pdf_path: Path to PDF
            total_pages: Total number of pages
            progress_callback: Optional progress callback

        Returns:
            List of PageScanResult, one per page
        """
        results: list[PageScanResult] = []

        for page_num in range(1, total_pages + 1):
            if progress_callback:
                progress_callback(page_num, total_pages, f"Scanning page {page_num}...")

            try:
                result = self._scan_single_page(pdf_path, page_num, total_pages)
                results.append(result)

                # Log what we found
                if result.sections:
                    section_ids = [s.identifier or "preamble" for s in result.sections]
                    logger.debug(f"Page {page_num}: {section_ids}")
                else:
                    logger.debug(f"Page {page_num}: empty")

            except Exception as e:
                logger.warning(f"Scan failed for page {page_num}: {e}")
                # Create empty result for failed page
                results.append(PageScanResult(
                    page_number=page_num,
                    sections=[],
                    is_empty=True,
                    scan_notes=f"Scan failed: {e}",
                ))

            # Small delay between pages
            time.sleep(0.1)

        return results

    def _scan_single_page(
        self,
        pdf_path: Path,
        page_number: int,
        total_pages: int,
    ) -> PageScanResult:
        """
        Scan a single page to detect which sections appear on it.

        Args:
            pdf_path: Path to PDF
            page_number: Page to scan (1-indexed)
            total_pages: Total pages in document

        Returns:
            PageScanResult with detected sections
        """
        # Render the page
        image = self.renderer.render_page(pdf_path, page_number)

        # Get prompts
        system_prompt, user_prompt = format_page_scan_prompt(page_number, total_pages)

        # Call API
        response = self.api_client.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[image],
            max_tokens=500,  # Small response expected
        )

        # Parse response
        data = response.content
        if not isinstance(data, dict):
            raise PageScanError(page_number, "Invalid response format")

        # Check if this is a ToC page - if so, only return preamble
        page_type = data.get("page_type", "content").lower()
        is_toc_page = page_type == "toc"

        # Parse sections
        sections: list[DetectedSection] = []

        if is_toc_page:
            # ToC pages should only report preamble, not individual sections
            # This is a safety check in case the LLM still lists sections
            sections.append(DetectedSection(
                section_type=SectionType.PREAMBLE,
                identifier=None,
                title="Inhaltsverzeichnis",
            ))
            logger.debug(f"Page {page_number}: ToC page detected, treating as preamble only")
        else:
            for s in data.get("sections", []):
                section_type_str = s.get("section_type", "").lower()

                if section_type_str == "paragraph":
                    section_type = SectionType.PARAGRAPH
                elif section_type_str == "anlage":
                    section_type = SectionType.ANLAGE
                elif section_type_str == "preamble":
                    section_type = SectionType.PREAMBLE
                else:
                    continue  # Skip unknown types

                sections.append(DetectedSection(
                    section_type=section_type,
                    identifier=s.get("identifier"),
                    title=s.get("title"),
                ))

        # Sanity check: if too many sections on one page, something might be wrong
        paragraph_count = sum(1 for s in sections if s.section_type == SectionType.PARAGRAPH)
        if paragraph_count > 5:
            logger.warning(
                f"Page {page_number}: {paragraph_count} paragraphs detected - "
                "possible ToC misinterpretation?"
            )

        return PageScanResult(
            page_number=data.get("page_number", page_number),
            sections=sections,
            is_empty=data.get("is_empty", False),
            scan_notes=data.get("scan_notes"),
        )

    # =========================================================================
    # PHASE 2: Structure Aggregation
    # =========================================================================

    def _phase2_aggregate_structure(
        self,
        scan_results: list[PageScanResult],
        total_pages: int,
    ) -> DocumentStructure:
        """
        Aggregate page scan results into document structure.

        This builds accurate page ranges by finding all pages where
        each section appears.

        Args:
            scan_results: Results from phase 1
            total_pages: Total pages in document

        Returns:
            DocumentStructure with section locations
        """
        # Build map: section key -> list of pages where it appears
        section_pages: dict[tuple, list[int]] = defaultdict(list)
        section_titles: dict[tuple, str] = {}

        has_preamble = False

        for result in scan_results:
            for section in result.sections:
                key = (section.section_type, section.identifier)
                section_pages[key].append(result.page_number)

                # Keep track of titles (prefer non-None)
                if section.title and key not in section_titles:
                    section_titles[key] = section.title

                if section.section_type == SectionType.PREAMBLE:
                    has_preamble = True

        if not section_pages:
            raise StructureAggregationError(
                "No sections detected in any page",
                "The document may not contain recognizable § or Anlage sections.",
            )

        # Build SectionLocation objects
        locations: list[SectionLocation] = []

        # Validation threshold: sections spanning > 40% of pages are suspicious
        max_reasonable_pages = max(int(total_pages * 0.4), 10)

        for (section_type, identifier), pages in section_pages.items():
            # Sort and deduplicate pages
            pages = sorted(set(pages))

            # VALIDATION: Check for unreasonably large page ranges
            if section_type == SectionType.PARAGRAPH and len(pages) > max_reasonable_pages:
                logger.warning(
                    f"Section {identifier} spans {len(pages)} pages "
                    f"(max reasonable: {max_reasonable_pages}) - "
                    "possible ToC misinterpretation, trimming to likely range"
                )
                # Try to find a reasonable contiguous range
                # Typically the ToC appears in first few pages, so filter those out
                non_toc_pages = [p for p in pages if p > 5]
                if non_toc_pages:
                    # Find the largest contiguous block
                    pages = self._find_contiguous_range(non_toc_pages)
                    logger.info(f"Section {identifier} trimmed to pages {pages}")

            locations.append(SectionLocation(
                section_type=section_type,
                identifier=identifier,
                title=section_titles.get((section_type, identifier)),
                pages=pages,
            ))

        # Sort by first page, then by identifier
        def sort_key(loc: SectionLocation) -> tuple:
            # Preamble first, then by start page, then by identifier
            type_order = {
                SectionType.PREAMBLE: 0,
                SectionType.PARAGRAPH: 1,
                SectionType.ANLAGE: 2,
            }
            return (
                type_order.get(loc.section_type, 1),
                loc.start_page,
                loc.identifier or "",
            )

        locations.sort(key=sort_key)

        # Post-validation: Log suspicious patterns
        self._validate_structure(locations, total_pages)

        return DocumentStructure(
            sections=locations,
            total_pages=total_pages,
            has_preamble=has_preamble,
            scan_results=scan_results,
        )

    def _find_contiguous_range(self, pages: list[int]) -> list[int]:
        """
        Find the largest contiguous range of pages.

        Used to recover from ToC misinterpretation where a section
        appears on many non-contiguous pages.

        Args:
            pages: Sorted list of page numbers

        Returns:
            Largest contiguous subset
        """
        if not pages:
            return pages

        # Find all contiguous ranges
        ranges: list[list[int]] = []
        current_range = [pages[0]]

        for i in range(1, len(pages)):
            if pages[i] == pages[i - 1] + 1:
                # Contiguous
                current_range.append(pages[i])
            else:
                # Gap - start new range
                ranges.append(current_range)
                current_range = [pages[i]]

        ranges.append(current_range)

        # Return the largest range
        return max(ranges, key=len)

    def _validate_structure(
        self,
        locations: list[SectionLocation],
        total_pages: int,
    ) -> None:
        """
        Validate the aggregated structure for suspicious patterns.

        Logs warnings if the structure looks incorrect.

        Args:
            locations: Section locations
            total_pages: Total pages in document
        """
        # Check for overlapping sections that might indicate problems
        paragraph_locations = [
            loc for loc in locations
            if loc.section_type == SectionType.PARAGRAPH
        ]

        if not paragraph_locations:
            logger.warning("No paragraph sections found - document may be unrecognized format")
            return

        # Check average pages per section (usually 1-3 for German academic docs)
        avg_pages = sum(loc.page_count for loc in paragraph_locations) / len(paragraph_locations)
        if avg_pages > 5:
            logger.warning(
                f"Average pages per section: {avg_pages:.1f} - "
                "this is higher than typical (1-3), possible scan issues"
            )

    # =========================================================================
    # PHASE 3: Context Extraction
    # =========================================================================

    def _phase3_extract_context(
        self,
        pdf_path: Path,
        structure: DocumentStructure,
    ) -> DocumentContext:
        """
        Extract document metadata from representative pages.

        Uses the first few pages (typically cover, ToC, preamble).

        Args:
            pdf_path: Path to PDF
            structure: Document structure from phase 2

        Returns:
            DocumentContext with metadata
        """
        # Determine which pages to use for context
        # Usually first 3-5 pages contain title, ToC, preamble
        context_pages = list(range(1, min(5, structure.total_pages) + 1))

        # Render pages
        images = self.renderer.render_batch(pdf_path, context_pages)

        # Get prompts
        system_prompt, user_prompt = format_context_prompt()

        # Call API
        response = self.api_client.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            max_tokens=1500,
        )

        # Parse response
        data = response.content
        if not isinstance(data, dict):
            raise ContextExtractionError("Invalid response format")

        return self._parse_context_data(data, structure.total_pages)

    def _parse_context_data(
        self,
        data: dict,
        total_pages: int,
    ) -> DocumentContext:
        """Parse context data into DocumentContext model."""
        # Map document type
        doc_type_map = {
            "pruefungsordnung": DocumentType.PRUEFUNGSORDNUNG,
            "modulhandbuch": DocumentType.MODULHANDBUCH,
            "studienordnung": DocumentType.STUDIENORDNUNG,
            "allgemeine_bestimmungen": DocumentType.ALLGEMEINE_BESTIMMUNGEN,
            "praktikumsordnung": DocumentType.PRAKTIKUMSORDNUNG,
            "zulassungsordnung": DocumentType.ZULASSUNGSORDNUNG,
            "satzung": DocumentType.SATZUNG,
        }
        doc_type = doc_type_map.get(
            str(data.get("document_type", "")).lower(),
            DocumentType.OTHER,
        )

        # Parse abbreviations
        abbreviations = []
        for item in data.get("abbreviations", []):
            if isinstance(item, dict) and item.get("short") and item.get("long"):
                abbreviations.append(Abbreviation(
                    short=item["short"],
                    long=item["long"],
                ))

        # Parse language
        lang_str = str(data.get("language", "de")).lower()
        language = Language.EN if lang_str == "en" else Language.DE

        return DocumentContext(
            document_type=doc_type,
            title=data.get("title", "Unbekanntes Dokument"),
            institution=data.get("institution", "Unbekannt"),
            version_date=data.get("version_date"),
            version_info=data.get("version_info"),
            degree_program=data.get("degree_program"),
            faculty=data.get("faculty"),
            total_pages=total_pages,
            chapters=data.get("chapters", []),
            abbreviations=abbreviations,
            key_terms=data.get("key_terms", []),
            referenced_documents=data.get("referenced_documents", []),
            legal_basis=data.get("legal_basis"),
            language=language,
        )

    def _create_fallback_context(
        self,
        pdf_path: Path,
        total_pages: int,
    ) -> DocumentContext:
        """Create minimal context when extraction fails."""
        return DocumentContext(
            document_type=DocumentType.OTHER,
            title=pdf_path.stem,
            institution="Unbekannt",
            total_pages=total_pages,
            language=Language.DE,
        )

    # =========================================================================
    # PHASE 4: Section Extraction
    # =========================================================================

    def _phase4_extract_section(
        self,
        pdf_path: Path,
        location: SectionLocation,
    ) -> ExtractedSection:
        """
        Extract the full content of a single section.

        Args:
            pdf_path: Path to PDF
            location: Section location from structure

        Returns:
            ExtractedSection with full content
        """
        # Handle preamble specially
        if location.section_type == SectionType.PREAMBLE:
            return self._extract_preamble(pdf_path, location)

        # Render all pages for this section
        images = self.renderer.render_batch(pdf_path, location.pages)

        # Build identifier string
        identifier = location.identifier or "Unbekannt"
        if location.title:
            identifier = f"{identifier} {location.title}"

        # Get prompts
        system_prompt, user_prompt = format_section_extract_prompt(
            section_identifier=identifier,
            start_page=location.start_page,
            end_page=location.end_page,
        )

        # Call API
        response = self.api_client.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            max_tokens=self.config.max_tokens,
        )

        # Parse response
        return self._parse_section_data(response.content, location)

    def _extract_preamble(
        self,
        pdf_path: Path,
        location: SectionLocation,
    ) -> ExtractedSection:
        """Extract preamble section (before first §)."""
        images = self.renderer.render_batch(pdf_path, location.pages)

        system_prompt, user_prompt = format_preamble_extract_prompt(location.end_page)

        response = self.api_client.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
            max_tokens=self.config.max_tokens,
        )

        return self._parse_section_data(response.content, location)

    def _parse_section_data(
        self,
        data: Union[dict, list],
        location: SectionLocation,
    ) -> ExtractedSection:
        """Parse section extraction response into ExtractedSection."""
        if not isinstance(data, dict):
            # If we got a list or something else, create minimal section
            return ExtractedSection(
                section_type=location.section_type,
                section_number=location.identifier,
                section_title=location.title,
                content=str(data),
                pages=location.pages,
                extraction_confidence=0.5,
                extraction_notes="Unexpected response format",
            )

        return ExtractedSection(
            section_type=location.section_type,
            section_number=location.identifier,
            section_title=data.get("section_title") or location.title,
            content=data.get("content", ""),
            pages=location.pages,
            chapter=data.get("chapter"),
            subsections=data.get("subsections", []),
            internal_references=data.get("internal_references", []),
            external_references=data.get("external_references", []),
            has_table=data.get("has_table", False),
            has_list=data.get("has_list", False),
            extraction_confidence=data.get("extraction_confidence", 1.0),
            extraction_notes=data.get("extraction_notes"),
        )

    def _create_failed_section(
        self,
        location: SectionLocation,
        error: str,
    ) -> ExtractedSection:
        """Create placeholder for failed extraction."""
        return ExtractedSection(
            section_type=location.section_type,
            section_number=location.identifier,
            section_title=location.title,
            content=f"[Extraction failed: {error}]",
            pages=location.pages,
            extraction_confidence=0.0,
            extraction_notes=error,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_usage(self) -> dict:
        """Get API usage statistics."""
        return self.api_client.get_usage_summary()

    def reset_usage(self) -> None:
        """Reset API usage counters."""
        self.api_client.reset_usage()
