"""
Section-Based PDF Extractor

Processes PDF documents using OpenAI's Vision API (GPT-4o) for
high-quality, section-based content extraction.

Two-phase approach:
1. Structure Analysis: Read ToC and create structure map (§§, Anlagen)
2. Section Extraction: Extract each section with all relevant page images

Usage:
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    result = extractor.extract("document.pdf")

    for section in result.sections:
        print(f"{section.section_number}: {section.content[:100]}...")

Environment:
    OPENAI_API_KEY: Your OpenAI API key
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Callable

from openai import OpenAI

from .pdf_to_images import PDFToImages
from .models import (
    DocumentContext,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    ExtractionConfig,
    SectionType,
    DocumentType,
    Abbreviation,
)
from .prompts import (
    get_structure_analysis_prompt,
    get_section_extraction_prompts,
)
from .exceptions import (
    NoTableOfContentsError,
    StructureExtractionError,
    SectionExtractionError,
    APIError,
)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Section-based PDF content extractor using OpenAI GPT-4o Vision API.

    Two-phase approach:
    1. Structure Analysis: Extract document structure from ToC
    2. Section Extraction: Extract content section by section (§§, Anlagen)

    Key features:
    - Sends ALL pages of a section in ONE API call
    - Sliding window for sections > max_images_per_request
    - Raises NoTableOfContentsError if no ToC found

    Usage:
        extractor = PDFExtractor()
        result = extractor.extract("document.pdf")

        for section in result.sections:
            print(f"{section.section_number}: {section.content[:100]}...")

    Environment:
        OPENAI_API_KEY: Your OpenAI API key
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the PDF Extractor.

        Args:
            config: Extraction configuration
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.config = config or ExtractionConfig()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.pdf_converter = PDFToImages(dpi=150)

        # Statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def extract(
        self,
        pdf_path: str | Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExtractionResult:
        """
        Extract content from a PDF document using section-based extraction.

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback(current, total, status)

        Returns:
            ExtractionResult with context and all sections

        Raises:
            NoTableOfContentsError: If no table of contents found
            FileNotFoundError: If PDF file not found
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(f"Starting section-based extraction: {pdf_path}")

        # Get document info
        doc_info = self.pdf_converter.get_document_info(pdf_path)
        total_pages = doc_info["page_count"]
        logger.info(f"Document has {total_pages} pages")

        # ===== PHASE 1: Structure Analysis =====
        if progress_callback:
            progress_callback(0, total_pages, "Analyzing document structure...")

        context, structure = self._analyze_structure(pdf_path, total_pages)
        logger.info(f"Structure analyzed: {len(structure)} sections found")

        total_sections = len(structure)

        # ===== PHASE 2: Section Extraction =====
        sections: list[ExtractedSection] = []

        for idx, entry in enumerate(structure):
            section_id = entry.identifier
            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_sections,
                    f"Extracting {section_id}..."
                )

            try:
                section = self._extract_section(
                    pdf_path, entry, context, total_pages
                )
                sections.append(section)
                logger.info(f"Extracted: {section_id} ({len(section.content)} chars)")
            except SectionExtractionError as e:
                logger.error(f"Failed to extract {section_id}: {e}")
                errors.append(f"{section_id}: {str(e)}")
                # Create placeholder section
                sections.append(self._create_failed_section(entry, str(e)))
            except Exception as e:
                logger.error(f"Unexpected error extracting {section_id}: {e}")
                errors.append(f"{section_id}: Unexpected error - {str(e)}")
                sections.append(self._create_failed_section(entry, str(e)))

            # Rate limiting
            time.sleep(0.5)

        processing_time = time.time() - start_time
        logger.info(f"Extraction complete in {processing_time:.1f}s")

        return ExtractionResult(
            source_file=str(pdf_path),
            context=context,
            sections=sections,
            processing_time_seconds=processing_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            errors=errors,
            warnings=warnings,
        )

    # =========================================================================
    # PHASE 1: Structure Analysis
    # =========================================================================

    def _analyze_structure(
        self,
        pdf_path: Path,
        total_pages: int,
    ) -> tuple[DocumentContext, list[StructureEntry]]:
        """
        Analyze document structure from table of contents.

        Args:
            pdf_path: Path to the PDF file
            total_pages: Total number of pages

        Returns:
            Tuple of (DocumentContext, list of StructureEntry)

        Raises:
            NoTableOfContentsError: If no ToC found
            StructureExtractionError: If structure analysis fails
        """
        # Select pages for structure analysis (first 5 pages for ToC)
        sample_pages = list(range(1, min(6, total_pages + 1)))
        logger.info(f"Structure analysis using pages: {sample_pages}")

        # Render sample pages
        images = self.pdf_converter.render_pages_batch(pdf_path, sample_pages)

        # Build message content
        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.mime_type};base64,{img.image_base64}"
                }
            })

        # Get prompts
        system_prompt, user_prompt = get_structure_analysis_prompt(total_pages)
        content.append({"type": "text", "text": user_prompt})

        # Call API
        try:
            response = self._call_api(system_prompt, content)
        except Exception as e:
            raise StructureExtractionError(
                f"API call failed during structure analysis: {e}"
            )

        # Parse response
        return self._parse_structure_response(response, total_pages, str(pdf_path))

    def _parse_structure_response(
        self,
        response: str,
        total_pages: int,
        document_path: str,
    ) -> tuple[DocumentContext, list[StructureEntry]]:
        """Parse the structure analysis response."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # Check if ToC was found
            if not data.get("has_toc", False):
                raise NoTableOfContentsError(document_path=document_path)

            # Parse context
            ctx_data = data.get("context", {})
            if not ctx_data:
                raise NoTableOfContentsError(document_path=document_path)

            context = self._parse_context(ctx_data, total_pages)

            # Parse structure
            structure_data = data.get("structure", [])
            if not structure_data:
                raise NoTableOfContentsError(
                    message="Inhaltsverzeichnis gefunden, aber keine Struktur extrahiert.",
                    document_path=document_path
                )

            structure = self._parse_structure_entries(structure_data, total_pages)

            return context, structure

        except (json.JSONDecodeError, KeyError) as e:
            raise StructureExtractionError(
                f"Failed to parse structure response: {e}"
            )

    def _parse_context(self, data: dict, total_pages: int) -> DocumentContext:
        """Parse context data into DocumentContext model."""
        # Map document type
        doc_type_str = data.get("document_type", "other").lower()
        doc_type_map = {
            "pruefungsordnung": DocumentType.PRUEFUNGSORDNUNG,
            "modulhandbuch": DocumentType.MODULHANDBUCH,
            "studienordnung": DocumentType.STUDIENORDNUNG,
            "allgemeine_bestimmungen": DocumentType.ALLGEMEINE_BESTIMMUNGEN,
            "praktikumsordnung": DocumentType.PRAKTIKUMSORDNUNG,
            "zulassungsordnung": DocumentType.ZULASSUNGSORDNUNG,
            "satzung": DocumentType.SATZUNG,
            "website": DocumentType.WEBSITE,
            "faq": DocumentType.FAQ,
        }
        doc_type = doc_type_map.get(doc_type_str, DocumentType.OTHER)

        # Parse abbreviations
        abbrevs_raw = data.get("abbreviations", [])
        abbreviations = []
        if isinstance(abbrevs_raw, list):
            for item in abbrevs_raw:
                if isinstance(item, dict):
                    short = item.get("short", "")
                    long = item.get("long", "")
                    if short and long:
                        abbreviations.append(Abbreviation(short=short, long=long))

        return DocumentContext(
            document_type=doc_type,
            title=data.get("title", "Unknown"),
            institution=data.get("institution", "Unknown"),
            version_date=data.get("version_date"),
            version_info=data.get("version_info"),
            faculty=data.get("faculty"),
            degree_program=data.get("degree_program"),
            total_pages=total_pages,
            chapters=data.get("chapters", []),
            abbreviations=abbreviations,
            key_terms=data.get("key_terms", []),
            referenced_documents=data.get("referenced_documents", []),
            legal_basis=data.get("legal_basis"),
        )

    def _parse_structure_entries(
        self,
        structure_data: list,
        total_pages: int,
    ) -> list[StructureEntry]:
        """Parse structure data into StructureEntry models."""
        entries = []

        for item in structure_data:
            # Map section type
            type_str = item.get("section_type", "paragraph").lower()
            type_map = {
                "overview": SectionType.OVERVIEW,
                "paragraph": SectionType.PARAGRAPH,
                "anlage": SectionType.ANLAGE,
            }
            section_type = type_map.get(type_str, SectionType.PARAGRAPH)

            # Get page range
            start_page = item.get("start_page", 1)
            end_page = item.get("end_page", start_page)

            # Validate and clamp pages
            start_page = max(1, min(start_page, total_pages))
            end_page = max(start_page, min(end_page, total_pages))

            entries.append(StructureEntry(
                section_type=section_type,
                section_number=item.get("section_number"),
                section_title=item.get("section_title"),
                start_page=start_page,
                end_page=end_page,
            ))

        # Sort by start page
        entries.sort(key=lambda e: (e.start_page, e.section_number or ""))

        return entries

    # =========================================================================
    # PHASE 2: Section Extraction
    # =========================================================================

    def _extract_section(
        self,
        pdf_path: Path,
        entry: StructureEntry,
        context: DocumentContext,
        total_pages: int,
    ) -> ExtractedSection:
        """
        Extract content from a single section.

        Uses sliding window if section spans more pages than max_images_per_request.
        """
        pages_needed = entry.pages
        max_images = self.config.max_images_per_request

        if len(pages_needed) <= max_images:
            # Single API call can handle entire section
            return self._extract_section_single(pdf_path, entry, context, pages_needed)
        else:
            # Need sliding window
            return self._extract_section_sliding_window(
                pdf_path, entry, context, pages_needed, max_images
            )

    def _extract_section_single(
        self,
        pdf_path: Path,
        entry: StructureEntry,
        context: DocumentContext,
        pages: list[int],
    ) -> ExtractedSection:
        """Extract a section that fits in a single API call."""
        # Render pages
        images = self.pdf_converter.render_pages_batch(pdf_path, pages)

        # Build content
        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.mime_type};base64,{img.image_base64}"
                }
            })

        # Get prompts
        system_prompt, user_prompt = get_section_extraction_prompts(
            context=context,
            entry=entry,
            visible_pages=pages,
            is_continuation=False,
            is_final_part=False,
        )
        content.append({"type": "text", "text": user_prompt})

        # Call API with retry
        response = self._call_api_with_retry(system_prompt, content)

        # Parse response
        return self._parse_section_response(response, entry, pages)

    def _extract_section_sliding_window(
        self,
        pdf_path: Path,
        entry: StructureEntry,
        context: DocumentContext,
        all_pages: list[int],
        max_images: int,
    ) -> ExtractedSection:
        """
        Extract a section using sliding window for large sections.

        Windows overlap by 1 page to maintain context.
        """
        windows = self._create_sliding_windows(all_pages, max_images)
        content_parts: list[str] = []
        combined_data = {
            "paragraphs": [],
            "has_table": False,
            "has_list": False,
            "internal_references": [],
            "external_references": [],
            "chapter": None,
            "extraction_confidence": 1.0,
            "extraction_notes": [],
        }

        logger.info(
            f"Using sliding window for {entry.identifier}: "
            f"{len(all_pages)} pages in {len(windows)} windows"
        )

        for i, window_pages in enumerate(windows):
            is_first = (i == 0)
            is_last = (i == len(windows) - 1)
            overlap_page = window_pages[0] if not is_first else None

            logger.debug(
                f"Window {i+1}/{len(windows)}: pages {window_pages[0]}-{window_pages[-1]}"
            )

            # Render pages
            images = self.pdf_converter.render_pages_batch(pdf_path, window_pages)

            # Build content
            msg_content = []
            for img in images:
                msg_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.mime_type};base64,{img.image_base64}"
                    }
                })

            # Get prompts with continuation hints
            system_prompt, user_prompt = get_section_extraction_prompts(
                context=context,
                entry=entry,
                visible_pages=window_pages,
                is_continuation=not is_first,
                is_final_part=is_last,
                overlap_page=overlap_page,
            )
            msg_content.append({"type": "text", "text": user_prompt})

            # Call API
            response = self._call_api_with_retry(system_prompt, msg_content)

            # Parse and accumulate
            try:
                json_str = self._extract_json(response)
                data = json.loads(json_str)

                content_parts.append(data.get("content", ""))

                # Accumulate metadata
                combined_data["paragraphs"].extend(data.get("paragraphs", []))
                combined_data["has_table"] = combined_data["has_table"] or data.get("has_table", False)
                combined_data["has_list"] = combined_data["has_list"] or data.get("has_list", False)
                combined_data["internal_references"].extend(data.get("internal_references", []))
                combined_data["external_references"].extend(data.get("external_references", []))

                if data.get("chapter") and not combined_data["chapter"]:
                    combined_data["chapter"] = data["chapter"]

                if data.get("extraction_confidence", 1.0) < combined_data["extraction_confidence"]:
                    combined_data["extraction_confidence"] = data["extraction_confidence"]

                if data.get("extraction_notes"):
                    combined_data["extraction_notes"].append(data["extraction_notes"])

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse window {i+1} response: {e}")
                content_parts.append(response)
                combined_data["extraction_confidence"] = min(
                    combined_data["extraction_confidence"], 0.5
                )

            # Rate limiting between windows
            if not is_last:
                time.sleep(0.5)

        # Combine content from all windows
        combined_content = "\n\n".join(filter(None, content_parts))

        # Deduplicate references
        combined_data["paragraphs"] = list(dict.fromkeys(combined_data["paragraphs"]))
        combined_data["internal_references"] = list(dict.fromkeys(combined_data["internal_references"]))
        combined_data["external_references"] = list(dict.fromkeys(combined_data["external_references"]))

        # Build extraction notes
        notes = None
        if combined_data["extraction_notes"]:
            notes = "; ".join(combined_data["extraction_notes"])

        return ExtractedSection(
            section_type=entry.section_type,
            section_number=entry.section_number,
            section_title=entry.section_title,
            content=combined_content,
            pages=all_pages,
            chapter=combined_data["chapter"],
            paragraphs=combined_data["paragraphs"],
            internal_references=combined_data["internal_references"],
            external_references=combined_data["external_references"],
            has_table=combined_data["has_table"],
            has_list=combined_data["has_list"],
            extraction_confidence=combined_data["extraction_confidence"],
            extraction_notes=notes,
        )

    def _create_sliding_windows(
        self,
        pages: list[int],
        max_images: int,
    ) -> list[list[int]]:
        """
        Create sliding windows with 1-page overlap.

        Args:
            pages: All pages to process
            max_images: Maximum images per window

        Returns:
            List of page lists for each window
        """
        if len(pages) <= max_images:
            return [pages]

        windows = []
        start = 0

        while start < len(pages):
            end = min(start + max_images, len(pages))
            windows.append(pages[start:end])

            # Move start forward, leaving 1-page overlap
            start = end - 1

            # Break if we've covered all pages
            if end >= len(pages):
                break

        return windows

    def _parse_section_response(
        self,
        response: str,
        entry: StructureEntry,
        pages: list[int],
    ) -> ExtractedSection:
        """Parse section extraction response into ExtractedSection."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            return ExtractedSection(
                section_type=entry.section_type,
                section_number=entry.section_number,
                section_title=entry.section_title,
                content=data.get("content", ""),
                pages=pages,
                chapter=data.get("chapter"),
                paragraphs=data.get("paragraphs", []),
                internal_references=data.get("internal_references", []),
                external_references=data.get("external_references", []),
                has_table=data.get("has_table", False),
                has_list=data.get("has_list", False),
                extraction_confidence=data.get("extraction_confidence", 1.0),
                extraction_notes=data.get("extraction_notes"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse section response: {e}")
            return ExtractedSection(
                section_type=entry.section_type,
                section_number=entry.section_number,
                section_title=entry.section_title,
                content=response,
                pages=pages,
                extraction_confidence=0.5,
                extraction_notes=f"JSON parsing failed: {e}",
            )

    def _create_failed_section(
        self,
        entry: StructureEntry,
        error_message: str,
    ) -> ExtractedSection:
        """Create a placeholder section for failed extractions."""
        return ExtractedSection(
            section_type=entry.section_type,
            section_number=entry.section_number,
            section_title=entry.section_title,
            content=f"[Extraction failed: {error_message}]",
            pages=entry.pages,
            extraction_confidence=0.0,
            extraction_notes=f"Extraction failed: {error_message}",
        )

    # =========================================================================
    # API Helpers
    # =========================================================================

    def _call_api(
        self,
        system_prompt: str,
        content: list,
    ) -> str:
        """Make an API call to OpenAI's Vision API."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens_per_request,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )

        # Track tokens
        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content

    def _call_api_with_retry(
        self,
        system_prompt: str,
        content: list,
    ) -> str:
        """Call API with retry logic for failures and refusals."""
        last_error = None
        last_response = None

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(system_prompt, content)

                # Check for refusal
                if self._is_refusal(response):
                    logger.warning(f"API refusal on attempt {attempt + 1}")
                    last_response = response

                    if attempt < self.config.max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                return response

            except Exception as e:
                logger.error(f"API error on attempt {attempt + 1}: {e}")
                last_error = e

                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries failed
        if last_response:
            return last_response

        raise APIError(f"API call failed after {self.config.max_retries} attempts: {last_error}")

    def _is_refusal(self, content: str) -> bool:
        """Check if the API response is a refusal."""
        refusal_phrases = [
            "I'm sorry, I can't assist",
            "I cannot assist",
            "I'm not able to",
            "I am not able to",
            "I cannot help",
            "I'm unable to",
            "I am unable to",
            "cannot process this",
            "can't process this",
        ]
        content_lower = content.lower()
        return any(phrase.lower() in content_lower for phrase in refusal_phrases)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from a response that may contain markdown code blocks."""
        # Try to find JSON in code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Try to find raw JSON
        if "{" in text:
            start = text.find("{")
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]

        return text


# =============================================================================
# Legacy Compatibility
# =============================================================================

# Alias for backwards compatibility
ProcessingConfig = ExtractionConfig


def estimate_api_cost(page_count: int, model: str = "gpt-4o") -> dict:
    """
    Estimate API cost for extracting a document.

    Note: This is now section-based, so actual cost depends on document structure.
    This provides a rough estimate based on page count.

    Args:
        page_count: Number of pages in the document
        model: Model to use

    Returns:
        Dictionary with cost estimate
    """
    # Rough estimates for section-based extraction
    # Structure analysis: ~5 images
    # Section extraction: ~1.5x page count (due to some overlap)

    if model == "gpt-4o":
        input_cost_per_1k = 0.005
        output_cost_per_1k = 0.015
        tokens_per_image = 1500
        output_per_section = 800
    elif model == "gpt-4o-mini":
        input_cost_per_1k = 0.00015
        output_cost_per_1k = 0.0006
        tokens_per_image = 1500
        output_per_section = 800
    else:
        return {"error": f"Unknown model: {model}"}

    # Estimate
    structure_images = 5
    section_images = int(page_count * 1.5)
    total_images = structure_images + section_images

    estimated_sections = max(10, page_count // 2)  # Rough estimate

    input_tokens = total_images * tokens_per_image
    output_tokens = estimated_sections * output_per_section

    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(total_cost, 4),
        "model": model,
        "page_count": page_count,
    }
