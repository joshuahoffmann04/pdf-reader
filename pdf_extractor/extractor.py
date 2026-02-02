"""
Section-Based PDF Extractor

Extracts content from PDF documents using OpenAI's Vision API (GPT-4o).
Two-phase approach:
1. Structure Analysis: Read ToC and create section map with PDF page numbers
2. Section Extraction: Extract each section's content

Key design: The LLM calculates the offset between printed page numbers
(in the ToC) and PDF page numbers during structure analysis.
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
from .prompts import get_structure_prompt, get_section_prompt
from .exceptions import (
    NoTableOfContentsError,
    StructureExtractionError,
    APIError,
)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Section-based PDF content extractor using OpenAI GPT-4o Vision API.

    Usage:
        extractor = PDFExtractor()
        result = extractor.extract("document.pdf")

        for section in result.sections:
            print(f"{section.section_number}: {section.content[:100]}...")
    """

    def __init__(
        self,
        config: Optional[ExtractionConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or ExtractionConfig()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter."
            )

        self.client = OpenAI(api_key=self.api_key)
        self.pdf_converter = PDFToImages(dpi=150)
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def extract(
        self,
        pdf_path: str | Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExtractionResult:
        """
        Extract content from a PDF document.

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
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        start_time = time.time()
        errors: list[str] = []

        # Get document info
        doc_info = self.pdf_converter.get_document_info(pdf_path)
        total_pages = doc_info["page_count"]
        logger.info(f"Processing {pdf_path}: {total_pages} pages")

        # Phase 1: Structure Analysis
        if progress_callback:
            progress_callback(0, total_pages, "Analyzing structure...")

        context, structure = self._analyze_structure(pdf_path, total_pages)
        logger.info(f"Found {len(structure)} sections")

        # Phase 2: Section Extraction
        sections: list[ExtractedSection] = []

        for idx, entry in enumerate(structure):
            if progress_callback:
                progress_callback(idx + 1, len(structure), f"Extracting {entry.identifier}...")

            try:
                section = self._extract_section(pdf_path, entry, context)
                sections.append(section)
                logger.info(f"Extracted {entry.identifier}: {len(section.content)} chars")
            except Exception as e:
                logger.error(f"Failed {entry.identifier}: {e}")
                errors.append(f"{entry.identifier}: {e}")
                sections.append(self._create_failed_section(entry, str(e)))

            time.sleep(0.5)  # Rate limiting

        return ExtractionResult(
            source_file=str(pdf_path),
            context=context,
            sections=sections,
            processing_time_seconds=time.time() - start_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            errors=errors,
        )

    # =========================================================================
    # Phase 1: Structure Analysis
    # =========================================================================

    def _analyze_structure(
        self,
        pdf_path: Path,
        total_pages: int,
    ) -> tuple[DocumentContext, list[StructureEntry]]:
        """Analyze document structure from table of contents."""

        # Render first 5 pages (usually contains ToC)
        sample_pages = list(range(1, min(6, total_pages + 1)))
        images = self.pdf_converter.render_pages_batch(pdf_path, sample_pages)

        # Build content with labeled PDF page numbers
        content = []
        for i, img in enumerate(images):
            page_num = sample_pages[i]
            # Label each image with its PDF page number
            content.append({
                "type": "text",
                "text": f"--- PDF-Seite {page_num} ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.mime_type};base64,{img.image_base64}"}
            })

        # Get prompts and add user prompt
        system_prompt, user_prompt = get_structure_prompt(total_pages)
        content.append({"type": "text", "text": user_prompt})

        # Call API
        response = self._call_api(system_prompt, content)

        # Parse response
        return self._parse_structure_response(response, total_pages, str(pdf_path))

    def _parse_structure_response(
        self,
        response: str,
        total_pages: int,
        document_path: str,
    ) -> tuple[DocumentContext, list[StructureEntry]]:
        """Parse structure analysis response."""
        try:
            data = json.loads(self._extract_json(response))

            if not data.get("has_toc", False):
                raise NoTableOfContentsError(document_path=document_path)

            ctx_data = data.get("context", {})
            if not ctx_data:
                raise NoTableOfContentsError(document_path=document_path)

            # Log the offset for debugging
            page_offset = data.get("page_offset", 0)
            logger.info(f"Page offset: {page_offset}")

            # Parse context
            context = self._parse_context(ctx_data, total_pages)

            # Parse structure entries
            structure_data = data.get("structure", [])
            if not structure_data:
                raise NoTableOfContentsError(
                    message="ToC found but no structure extracted.",
                    document_path=document_path
                )

            structure = self._parse_structure_entries(structure_data, total_pages)

            # Log extracted structure for debugging (WARNING level so always visible)
            logger.warning("Extracted structure from ToC:")
            for entry in structure:
                logger.warning(f"  {entry.identifier}: pages {entry.start_page}-{entry.end_page}")

            return context, structure

        except json.JSONDecodeError as e:
            raise StructureExtractionError(f"JSON parse error: {e}")

    def _parse_context(self, data: dict, total_pages: int) -> DocumentContext:
        """Parse context data into DocumentContext."""
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
            data.get("document_type", "").lower(),
            DocumentType.OTHER
        )

        # Parse abbreviations
        abbreviations = []
        for item in data.get("abbreviations", []):
            if isinstance(item, dict) and item.get("short") and item.get("long"):
                abbreviations.append(Abbreviation(short=item["short"], long=item["long"]))

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
        """Parse structure data into StructureEntry list."""
        type_map = {
            "overview": SectionType.OVERVIEW,
            "paragraph": SectionType.PARAGRAPH,
            "anlage": SectionType.ANLAGE,
        }

        entries = []
        for item in structure_data:
            section_type = type_map.get(
                item.get("section_type", "").lower(),
                SectionType.PARAGRAPH
            )

            start_page = max(1, min(item.get("start_page", 1), total_pages))
            end_page = max(start_page, min(item.get("end_page", start_page), total_pages))

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
    # Phase 2: Section Extraction
    # =========================================================================

    def _extract_section(
        self,
        pdf_path: Path,
        entry: StructureEntry,
        context: DocumentContext,
    ) -> ExtractedSection:
        """Extract content from a single section."""
        pages = entry.pages
        max_images = self.config.max_images_per_request

        # Log which pages we're sending (WARNING level so it's always visible)
        logger.warning(f"Extracting {entry.identifier}: sending PDF pages {pages}")

        if len(pages) <= max_images:
            return self._extract_section_single(pdf_path, entry, context, pages)
        else:
            return self._extract_section_windowed(pdf_path, entry, context, pages, max_images)

    def _extract_section_single(
        self,
        pdf_path: Path,
        entry: StructureEntry,
        context: DocumentContext,
        pages: list[int],
    ) -> ExtractedSection:
        """Extract section that fits in one API call."""
        images = self.pdf_converter.render_pages_batch(pdf_path, pages)

        # Build content with labeled pages
        content = []
        for i, img in enumerate(images):
            content.append({
                "type": "text",
                "text": f"--- PDF-Seite {pages[i]} ---"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.mime_type};base64,{img.image_base64}"}
            })

        system, user = get_section_prompt(context, entry, pages)
        content.append({"type": "text", "text": user})

        response = self._call_api_with_retry(system, content)
        return self._parse_section_response(response, entry, pages)

    def _extract_section_windowed(
        self,
        pdf_path: Path,
        entry: StructureEntry,
        context: DocumentContext,
        all_pages: list[int],
        max_images: int,
    ) -> ExtractedSection:
        """Extract large section using sliding window."""
        windows = self._create_windows(all_pages, max_images)
        content_parts = []
        combined = {
            "paragraphs": [],
            "has_table": False,
            "has_list": False,
            "internal_references": [],
            "external_references": [],
            "chapter": None,
            "confidence": 1.0,
        }

        logger.info(f"Using {len(windows)} windows for {entry.identifier}")

        for i, window in enumerate(windows):
            is_first = (i == 0)
            is_last = (i == len(windows) - 1)
            overlap = window[0] if not is_first else None

            images = self.pdf_converter.render_pages_batch(pdf_path, window)

            content = []
            for j, img in enumerate(images):
                content.append({"type": "text", "text": f"--- PDF-Seite {window[j]} ---"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{img.image_base64}"}
                })

            system, user = get_section_prompt(
                context, entry, window,
                is_continuation=not is_first,
                is_last=is_last,
                overlap_page=overlap,
            )
            content.append({"type": "text", "text": user})

            response = self._call_api_with_retry(system, content)

            try:
                data = json.loads(self._extract_json(response))
                content_parts.append(data.get("content", ""))

                # Accumulate metadata
                combined["paragraphs"].extend(data.get("paragraphs", []))
                combined["has_table"] = combined["has_table"] or data.get("has_table", False)
                combined["has_list"] = combined["has_list"] or data.get("has_list", False)
                combined["internal_references"].extend(data.get("internal_references", []))
                combined["external_references"].extend(data.get("external_references", []))
                if data.get("chapter") and not combined["chapter"]:
                    combined["chapter"] = data["chapter"]
                if data.get("extraction_confidence", 1.0) < combined["confidence"]:
                    combined["confidence"] = data["extraction_confidence"]

            except json.JSONDecodeError:
                content_parts.append(response)
                combined["confidence"] = min(combined["confidence"], 0.5)

            if not is_last:
                time.sleep(0.5)

        return ExtractedSection(
            section_type=entry.section_type,
            section_number=entry.section_number,
            section_title=entry.section_title,
            content="\n\n".join(filter(None, content_parts)),
            pages=all_pages,
            chapter=combined["chapter"],
            paragraphs=list(dict.fromkeys(combined["paragraphs"])),
            internal_references=list(dict.fromkeys(combined["internal_references"])),
            external_references=list(dict.fromkeys(combined["external_references"])),
            has_table=combined["has_table"],
            has_list=combined["has_list"],
            extraction_confidence=combined["confidence"],
        )

    def _create_windows(self, pages: list[int], max_size: int) -> list[list[int]]:
        """Create sliding windows with 1-page overlap."""
        if len(pages) <= max_size:
            return [pages]

        windows = []
        start = 0
        while start < len(pages):
            end = min(start + max_size, len(pages))
            windows.append(pages[start:end])
            if end >= len(pages):
                break
            start = end - 1  # 1-page overlap

        return windows

    def _parse_section_response(
        self,
        response: str,
        entry: StructureEntry,
        pages: list[int],
    ) -> ExtractedSection:
        """Parse section extraction response."""
        try:
            data = json.loads(self._extract_json(response))
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
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {entry.identifier}: {e}")
            return ExtractedSection(
                section_type=entry.section_type,
                section_number=entry.section_number,
                section_title=entry.section_title,
                content=response,
                pages=pages,
                extraction_confidence=0.5,
                extraction_notes=f"JSON parse failed: {e}",
            )

    def _create_failed_section(self, entry: StructureEntry, error: str) -> ExtractedSection:
        """Create placeholder for failed extraction."""
        return ExtractedSection(
            section_type=entry.section_type,
            section_number=entry.section_number,
            section_title=entry.section_title,
            content=f"[Extraction failed: {error}]",
            pages=entry.pages,
            extraction_confidence=0.0,
            extraction_notes=error,
        )

    # =========================================================================
    # API Helpers
    # =========================================================================

    def _call_api(self, system_prompt: str, content: list) -> str:
        """Make API call to OpenAI."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens_per_request,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )

        if response.usage:
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content or ""

    def _call_api_with_retry(self, system_prompt: str, content: list) -> str:
        """Call API with retry logic."""
        last_response = None

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(system_prompt, content)

                # Check for problems
                if not response.strip():
                    logger.warning(f"Empty response (attempt {attempt + 1})")
                    last_response = response
                elif self._is_refusal(response):
                    logger.warning(f"API refusal (attempt {attempt + 1}): {response[:100]}")
                    last_response = response
                elif "{" not in response:
                    logger.warning(f"No JSON in response (attempt {attempt + 1})")
                    last_response = response
                else:
                    return response

                # Wait before retry
                if attempt < self.config.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)

            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    raise APIError(str(e))

        return last_response or ""

    def _is_refusal(self, text: str) -> bool:
        """Check if response is a refusal."""
        if not text:
            return False
        refusals = [
            "i'm sorry", "i cannot", "i can't", "i am unable",
            "i'm unable", "i'm not able", "i am not able",
            "cannot assist", "can't assist",
        ]
        text_lower = text.lower()
        return any(r in text_lower for r in refusals)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from response (handles markdown code blocks)."""
        # Try code block
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

        # Try raw JSON
        if "{" in text:
            start = text.find("{")
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]

        return text


def estimate_api_cost(page_count: int, model: str = "gpt-4o") -> dict:
    """Estimate API cost for document extraction."""
    costs = {
        "gpt-4o": (0.005, 0.015, 1500, 800),
        "gpt-4o-mini": (0.00015, 0.0006, 1500, 800),
    }

    if model not in costs:
        return {"error": f"Unknown model: {model}"}

    input_cost, output_cost, tokens_per_img, output_per_section = costs[model]

    # Estimate: structure (5 images) + sections (~1.5x pages due to overlap)
    total_images = 5 + int(page_count * 1.5)
    sections = max(10, page_count // 2)

    input_tokens = total_images * tokens_per_img
    output_tokens = sections * output_per_section

    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_cost_usd": round(
            (input_tokens / 1000) * input_cost + (output_tokens / 1000) * output_cost, 4
        ),
        "model": model,
        "page_count": page_count,
    }
