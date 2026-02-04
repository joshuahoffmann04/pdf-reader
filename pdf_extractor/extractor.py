"""
PDF Extractor - Main Processing Engine

Hybrid pipeline for PDF content extraction:
1. Text-native extraction (PyMuPDF)
2. Optional OCR fallback (Tesseract)
3. Optional Vision fallback (LLM)
4. Coverage validation to prevent information loss

Usage:
    from pdf_extractor import PDFExtractor

    extractor = PDFExtractor()
    result = extractor.extract("document.pdf")
    result.save("output.json")

Environment (optional):
    OPENAI_API_KEY: Enable LLM-based vision extraction
    TESSERACT_CMD: Optional path to tesseract.exe for OCR
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Callable

from openai import OpenAI

from .pdf_to_images import PDFToImages
from .text_extractor import TextExtractor
from .table_extractor import TableExtractor
from .validation import validate_llm_output, validate_text_coverage
from .models import (
    DocumentContext,
    ExtractedPage,
    ExtractionResult,
    SectionMarker,
    ProcessingConfig,
    DocumentType,
    ContentType,
    Abbreviation,
)
from .prompts import (
    CONTEXT_ANALYSIS_SYSTEM,
    CONTEXT_ANALYSIS_USER,
    get_context_analysis_text_prompt,
    get_page_extraction_system_prompt,
    get_page_extraction_text_prompt,
    get_page_extraction_user_prompt,
)

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Main extractor for Vision-LLM based PDF content extraction using OpenAI GPT-4o.

    Two-phase approach:
    1. Context Analysis: Understand the entire document structure
    2. Page Extraction: Extract content page by page with full context

    Usage:
        extractor = PDFExtractor()
        result = extractor.extract("document.pdf")

        # Access results
        print(result.context.title)
        for page in result.pages:
            print(f"Page {page.page_number}: {page.content[:100]}...")

        # Save to file
        result.save("output.json")

    Environment:
        OPENAI_API_KEY: Your OpenAI API key
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the PDF Extractor.

        Args:
            config: Processing configuration
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.config = config or ProcessingConfig()
        self.llm_enabled = bool(self.config.use_llm)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None

        if self.llm_enabled:
            if not self.api_key:
                if self.config.extraction_mode in {"text", "hybrid"}:
                    logger.warning(
                        "OPENAI_API_KEY missing; falling back to text-only extraction."
                    )
                    self.llm_enabled = False
                    self.config.context_mode = "heuristic"
                    self.config.llm_postprocess = False
                else:
                    raise ValueError(
                        "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                        "or pass api_key parameter."
                    )
            else:
                self.client = OpenAI(api_key=self.api_key)

        self.pdf_converter = PDFToImages(dpi=150)
        self.text_extractor = TextExtractor(
            min_chars=self.config.text_min_chars,
            min_alpha_ratio=self.config.text_min_alpha_ratio,
            preserve_line_breaks=self.config.preserve_line_breaks,
            layout_mode=self.config.layout_mode,
            column_gap_ratio=self.config.column_gap_ratio,
        )
        self.table_extractor = TableExtractor(table_settings=self.config.table_settings)

        # Statistics
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
            progress_callback: Optional callback(current_page, total_pages, status)

        Returns:
            ExtractionResult with context and all page contents
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()
        errors: list[str] = []
        warnings: list[str] = []

        logger.info(f"Starting document extraction: {pdf_path}")

        # Get document info
        doc_info = self.pdf_converter.get_document_info(pdf_path)
        total_pages = doc_info["page_count"]
        logger.info(f"Document has {total_pages} pages")

        # Phase 1: Context Analysis
        if progress_callback:
            progress_callback(0, total_pages, "Analyzing document context...")

        try:
            context = self._analyze_context(pdf_path, total_pages)
            logger.info(f"Context analyzed: {context.document_type.value}")
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            errors.append(f"Context analysis error: {str(e)}")
            # Create minimal context
            context = DocumentContext(
                document_type=DocumentType.OTHER,
                title=pdf_path.stem,
                institution="Unknown",
                total_pages=total_pages,
            )

        # Phase 2: Page-by-Page Extraction
        pages: list[ExtractedPage] = []
        failed_pages: list[int] = []

        for page_num in range(1, total_pages + 1):
            if progress_callback:
                progress_callback(page_num, total_pages, f"Processing page {page_num}...")

            page_content = self._extract_page_with_retry(
                pdf_path,
                page_num,
                total_pages,
                context,
                max_retries=self.config.max_retries,
            )

            # Check if extraction failed (refusal or error)
            if page_content.extraction_confidence < 1.0 or self._is_refusal(page_content.content):
                failed_pages.append(page_num)
                errors.append(f"Page {page_num}: extraction failed or refused")

            pages.append(page_content)

            # Rate limiting (avoid hitting API limits)
            time.sleep(0.5)

        # Log summary of failed pages
        if failed_pages:
            logger.warning(f"Failed pages: {failed_pages}")
            warnings.append(f"Failed to extract pages: {failed_pages}")

        processing_time = time.time() - start_time
        logger.info(f"Extraction complete in {processing_time:.1f}s")

        return ExtractionResult(
            source_file=str(pdf_path),
            context=context,
            pages=pages,
            processing_time_seconds=processing_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            errors=errors,
            warnings=warnings,
        )

    def _analyze_context(
        self,
        pdf_path: Path,
        total_pages: int,
    ) -> DocumentContext:
        """
        Phase 1: Analyze document context using sample pages.

        Uses pages 1-3 (for title page and table of contents),
        optionally a middle page, and the last page.
        """
        # Select sample pages for context (include TOC)
        sample_pages = [1]
        if total_pages >= 2:
            sample_pages.append(2)
        if total_pages >= 3:
            sample_pages.append(3)
        if total_pages > 10:
            middle = total_pages // 2
            if middle not in sample_pages:
                sample_pages.append(middle)
        if total_pages > 3 and total_pages not in sample_pages:
            sample_pages.append(total_pages)

        logger.info(f"Context analysis using pages: {sample_pages}")

        if not self.llm_enabled or self.config.context_mode == "heuristic":
            return self._build_min_context(pdf_path, total_pages)

        if self.config.context_mode == "llm_text":
            text_blob = self._collect_text_samples(pdf_path, sample_pages)
            if text_blob.strip():
                content = [
                    {
                        "type": "text",
                        "text": get_context_analysis_text_prompt(text_blob),
                    }
                ]
                response = self._call_api(
                    system_prompt=CONTEXT_ANALYSIS_SYSTEM,
                    content=content,
                )
                return self._parse_context_response(response, total_pages)
            # If no text is available, fall back to vision context.
            return self._analyze_context_vision(pdf_path, sample_pages, total_pages)

        if self.config.context_mode == "llm_vision":
            return self._analyze_context_vision(pdf_path, sample_pages, total_pages)

        # Unknown mode -> minimal context
        return self._build_min_context(pdf_path, total_pages)

    def _collect_text_samples(self, pdf_path: Path, sample_pages: list[int]) -> str:
        parts: list[str] = []
        for page_number in sample_pages:
            analysis = self.text_extractor.extract_page(pdf_path, page_number)
            if analysis.text:
                parts.append(f"[Seite {page_number}]\n{analysis.text}")
        return "\n\n".join(parts)

    def _analyze_context_vision(
        self,
        pdf_path: Path,
        sample_pages: list[int],
        total_pages: int,
    ) -> DocumentContext:
        images = self.pdf_converter.render_pages_batch(pdf_path, sample_pages)

        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img.mime_type};base64,{img.image_base64}"
                }
            })

        content.append({
            "type": "text",
            "text": CONTEXT_ANALYSIS_USER,
        })

        response = self._call_api(
            system_prompt=CONTEXT_ANALYSIS_SYSTEM,
            content=content,
        )

        return self._parse_context_response(response, total_pages)

    def _build_min_context(self, pdf_path: Path, total_pages: int) -> DocumentContext:
        info = self.pdf_converter.get_document_info(pdf_path)
        title = info.get("title") or Path(pdf_path).stem
        return DocumentContext(
            document_type=DocumentType.OTHER,
            title=title or "Unknown",
            institution="Unknown",
            total_pages=total_pages,
        )

    def _extract_page(
        self,
        pdf_path: Path,
        page_number: int,
        total_pages: int,
        context: DocumentContext,
    ) -> ExtractedPage:
        """
        Phase 2: Extract content from a single page.
        """
        # 1) Text-native extraction (preferred)
        if self.config.extraction_mode in {"text", "hybrid"}:
            ocr_attempted = False
            analysis = self.text_extractor.extract_page(pdf_path, page_number)
            if analysis.quality_ok or (analysis.has_table and self.config.table_extraction):
                return self._build_page_from_text(
                    analysis, pdf_path, page_number, total_pages, context, source="text"
                )

            if self.config.extraction_mode == "hybrid":
                if self.config.ocr_enabled and self.config.ocr_before_vision:
                    ocr_attempted = True
                    ocr_text = self._extract_page_ocr(pdf_path, page_number)
                    if ocr_text:
                        ocr_analysis = self.text_extractor.analyze_text(ocr_text)
                        if ocr_analysis.text:
                            return self._build_page_from_text(
                                ocr_analysis, pdf_path, page_number, total_pages, context, source="ocr"
                            )

                if self.llm_enabled:
                    return self._extract_page_vision(
                        pdf_path, page_number, total_pages, context, raw_text=analysis.text
                    )

            if self.config.ocr_enabled and not ocr_attempted:
                ocr_text = self._extract_page_ocr(pdf_path, page_number)
                if ocr_text:
                    ocr_analysis = self.text_extractor.analyze_text(ocr_text)
                    if ocr_analysis.text:
                        return self._build_page_from_text(
                            ocr_analysis, pdf_path, page_number, total_pages, context, source="ocr"
                        )

            if self.config.extraction_mode == "text" or not self.llm_enabled:
                return self._build_page_from_text(
                    analysis, pdf_path, page_number, total_pages, context, source="text"
                )

        # 2) Vision-based extraction fallback
        return self._extract_page_vision(pdf_path, page_number, total_pages, context)

    def _extract_page_vision(
        self,
        pdf_path: Path,
        page_number: int,
        total_pages: int,
        context: DocumentContext,
        raw_text: str | None = None,
    ) -> ExtractedPage:
        page_image = self.pdf_converter.render_page(pdf_path, page_number)

        context_dict = {
            "document_type": context.document_type.value,
            "title": context.title,
            "institution": context.institution,
            "faculty": context.faculty,
            "degree_program": context.degree_program,
            "abbreviations": context.get_abbreviation_dict(),
            "chapters": context.chapters,
        }

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{page_image.mime_type};base64,{page_image.image_base64}"
                }
            },
            {
                "type": "text",
                "text": get_page_extraction_user_prompt(page_number, total_pages),
            }
        ]

        system_prompt = get_page_extraction_system_prompt(context_dict)
        response = self._call_api(
            system_prompt=system_prompt,
            content=content,
        )

        page = self._parse_page_response(response, page_number)
        page.content_source = "vision"
        if raw_text:
            page.raw_content = raw_text
            if self.config.enforce_text_coverage:
                coverage = validate_text_coverage(
                    raw_text,
                    page.content,
                    min_token_recall=self.config.min_token_recall,
                    min_number_recall=self.config.min_number_recall,
                )
                if not coverage.ok:
                    page.extraction_notes = (
                        (page.extraction_notes + " | " if page.extraction_notes else "")
                        + f"Vision coverage fallback: {coverage.reason}"
                    )
                    page.content = raw_text + "\n\n" + page.content
                    page.content_source = "vision+raw"
        return page

    def _build_page_from_text(
        self,
        analysis,
        pdf_path: Path,
        page_number: int,
        total_pages: int,
        context: DocumentContext,
        source: str,
    ) -> ExtractedPage:
        content_text = analysis.text
        raw_text = analysis.text
        extraction_notes = None
        content_source = source

        if self.llm_enabled and self.config.llm_postprocess:
            llm_text = self._postprocess_text_with_llm(
                analysis.text, page_number, total_pages, context
            )
            if llm_text:
                if self.config.validate_llm_output:
                    result = validate_llm_output(
                        raw_text,
                        llm_text,
                        min_numeric_recall=self.config.min_numeric_recall,
                        min_section_recall=self.config.min_section_recall,
                        min_char_recall=self.config.min_char_recall,
                    )
                    if result.ok:
                        content_text = llm_text
                        content_source = "llm_text"
                    else:
                        extraction_notes = f"LLM failed validation: {result.reason}"
                else:
                    content_text = llm_text
                    content_source = "llm_text"

        table_text = ""
        if self.config.table_extraction:
            table_result = self.table_extractor.extract_tables(pdf_path, page_number)
            if table_result.text:
                table_text = table_result.text
                content_text = content_text + "\n\n" + table_result.text
                if content_source:
                    content_source = f"{content_source}+table"

        sections = [
            SectionMarker(number=number, title=title or None, level=1, starts_on_page=True)
            for number, title in zip(analysis.section_numbers, analysis.section_titles)
        ]

        content_types = []
        has_table = analysis.has_table or bool(table_text)
        if has_table:
            content_types.append(ContentType.TABLE)
        if analysis.has_list:
            content_types.append(ContentType.LIST)
        if not content_types:
            content_types.append(ContentType.SECTION)

        confidence = 1.0 if source in {"text", "ocr"} else 0.85

        if self.config.enforce_text_coverage:
            coverage = validate_text_coverage(
                raw_text,
                content_text,
                min_token_recall=self.config.min_token_recall,
                min_number_recall=self.config.min_number_recall,
            )
            if not coverage.ok:
                extraction_notes = (
                    (extraction_notes + " | " if extraction_notes else "")
                    + f"Coverage fallback: {coverage.reason}"
                )
                content_text = raw_text
                content_source = "text+raw"

        return ExtractedPage(
            page_number=page_number,
            content=content_text,
            sections=sections,
            paragraph_numbers=analysis.paragraph_numbers,
            content_types=content_types,
            has_table=has_table,
            has_list=analysis.has_list,
            has_figure=False,
            internal_references=analysis.internal_references,
            external_references=analysis.external_references,
            continues_from_previous=analysis.continues_from_previous,
            continues_to_next=analysis.continues_to_next,
            extraction_confidence=confidence,
            extraction_notes=extraction_notes,
            raw_content=raw_text,
            content_source=content_source,
        )

    def _postprocess_text_with_llm(
        self,
        page_text: str,
        page_number: int,
        total_pages: int,
        context: DocumentContext,
    ) -> str | None:
        if not page_text:
            return None

        context_dict = {
            "document_type": context.document_type.value,
            "title": context.title,
            "institution": context.institution,
            "faculty": context.faculty,
            "degree_program": context.degree_program,
            "abbreviations": context.get_abbreviation_dict(),
            "chapters": context.chapters,
        }

        content = [
            {
                "type": "text",
                "text": get_page_extraction_text_prompt(page_number, total_pages, page_text),
            }
        ]

        system_prompt = get_page_extraction_system_prompt(context_dict)
        response = self._call_api(system_prompt=system_prompt, content=content)
        parsed = self._parse_page_response(response, page_number)
        return parsed.content

    def _extract_page_ocr(self, pdf_path: Path, page_number: int) -> str:
        if not self.config.ocr_enabled:
            return ""
        try:
            import base64
            import io
            import shutil
            from PIL import Image
            import pytesseract
            from pytesseract import TesseractNotFoundError
        except ImportError:
            logger.warning("OCR requested but pytesseract or Pillow is not installed.")
            return ""

        tesseract_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            logger.warning("Tesseract not found; skipping OCR for this page.")
            return ""

        ocr_renderer = PDFToImages(dpi=self.config.ocr_dpi)
        image = ocr_renderer.render_page(pdf_path, page_number)
        img_bytes = base64.b64decode(image.image_base64)
        pil_image = Image.open(io.BytesIO(img_bytes))
        tesseract_config = f"--oem {self.config.ocr_oem} --psm {self.config.ocr_psm}"
        try:
            return pytesseract.image_to_string(
                pil_image,
                lang=self.config.ocr_language,
                config=tesseract_config,
            )
        except TesseractNotFoundError:
            logger.warning("Tesseract not found at runtime; skipping OCR for this page.")
            return ""
        except RuntimeError as exc:
            logger.warning(f"OCR failed on page {page_number}: {exc}")
            return ""

    def _extract_page_with_retry(
        self,
        pdf_path: Path,
        page_number: int,
        total_pages: int,
        context: DocumentContext,
        max_retries: int = 3,
    ) -> ExtractedPage:
        """
        Extract page content with retry logic for refusals and errors.

        Uses exponential backoff between retries.
        """
        last_error = None
        last_result = None

        for attempt in range(max_retries):
            try:
                result = self._extract_page(
                    pdf_path,
                    page_number,
                    total_pages,
                    context,
                )

                # Check for refusal
                if self._is_refusal(result.content):
                    logger.warning(
                        f"Page {page_number}: API refusal on attempt {attempt + 1}/{max_retries}"
                    )
                    last_result = result

                    if attempt < max_retries - 1:
                        # Exponential backoff: 2s, 4s, 8s
                        wait_time = 2 ** (attempt + 1)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                # Check for low confidence (JSON parsing failed)
                if result.extraction_confidence < 1.0:
                    logger.warning(
                        f"Page {page_number}: Low confidence ({result.extraction_confidence}) "
                        f"on attempt {attempt + 1}/{max_retries}"
                    )
                    last_result = result

                    if attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)
                        logger.info(f"Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                # Success!
                if attempt > 0:
                    logger.info(f"Page {page_number}: Succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                logger.error(f"Page {page_number}: Error on attempt {attempt + 1}/{max_retries}: {e}")
                last_error = e

                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries failed - return best result or error placeholder
        if last_result is not None:
            return last_result

        # Create error placeholder
        return ExtractedPage(
            page_number=page_number,
            content=f"[Extraction failed after {max_retries} attempts: {last_error}]",
            extraction_confidence=0.0,
            extraction_notes=f"Failed after {max_retries} retries: {last_error}",
        )

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

    def _call_api(
        self,
        system_prompt: str,
        content: list,
    ) -> str:
        """Make an API call to OpenAI's Vision API."""
        if not self.client:
            raise RuntimeError("LLM client not initialized. Set OPENAI_API_KEY or disable LLM usage.")
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

    def _parse_context_response(self, response: str, total_pages: int) -> DocumentContext:
        """Parse the context analysis response into a DocumentContext."""
        try:
            # Extract JSON from response
            json_str = self._extract_json(response)
            data = json.loads(json_str)

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

            # Parse abbreviations (handles both list and dict format)
            abbrevs_raw = data.get("abbreviations", [])
            abbreviations = []

            if isinstance(abbrevs_raw, list):
                # New format: [{"short": "LP", "long": "Leistungspunkte"}, ...]
                for item in abbrevs_raw:
                    if isinstance(item, dict):
                        short = item.get("short", item.get("abbr", ""))
                        long = item.get("long", item.get("full", ""))
                        if short and long:
                            abbreviations.append(Abbreviation(short=short, long=long))
            elif isinstance(abbrevs_raw, dict):
                # Legacy format: {"LP": "Leistungspunkte", ...}
                abbreviations = [
                    Abbreviation(short=k, long=v)
                    for k, v in abbrevs_raw.items()
                    if k and v
                ]

            return DocumentContext(
                document_type=doc_type,
                title=data.get("title", "Unknown"),
                institution=data.get("institution", "Unknown"),
                version_date=data.get("version_date"),
                version_info=data.get("version_info"),
                faculty=data.get("faculty"),
                total_pages=total_pages,
                chapters=data.get("chapters", []),
                main_topics=data.get("main_topics", []),
                degree_program=data.get("degree_program"),
                abbreviations=abbreviations,
                key_terms=data.get("key_terms", []),
                referenced_documents=data.get("referenced_documents", []),
                legal_basis=data.get("legal_basis"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse context response: {e}")
            return DocumentContext(
                document_type=DocumentType.OTHER,
                title="Unknown",
                institution="Unknown",
                total_pages=total_pages,
            )

    def _parse_page_response(self, response: str, page_number: int) -> ExtractedPage:
        """Parse the page extraction response into an ExtractedPage."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            # Parse section markers
            section_numbers = data.get("section_numbers", [])
            section_titles = data.get("section_titles", [])
            sections = []
            for i, num in enumerate(section_numbers):
                title = section_titles[i] if i < len(section_titles) else None
                sections.append(SectionMarker(
                    number=num,
                    title=title,
                    level=1,
                    starts_on_page=True,
                ))

            # Parse paragraph numbers
            paragraph_numbers = data.get("paragraph_numbers", [])

            # Determine content types
            content_types = []
            if data.get("has_table"):
                content_types.append(ContentType.TABLE)
            if data.get("has_list"):
                content_types.append(ContentType.LIST)
            if not content_types:
                content_types.append(ContentType.SECTION)

            return ExtractedPage(
                page_number=page_number,
                content=data.get("content", ""),
                sections=sections,
                paragraph_numbers=paragraph_numbers,
                content_types=content_types,
                has_table=data.get("has_table", False),
                has_list=data.get("has_list", False),
                has_figure=data.get("has_image", False),
                internal_references=data.get("internal_references", []),
                external_references=data.get("external_references", []),
                continues_from_previous=data.get("continues_from_previous", False),
                continues_to_next=data.get("continues_to_next", False),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse page {page_number} response: {e}")
            # Return raw content as fallback
            return ExtractedPage(
                page_number=page_number,
                content=response,
                extraction_confidence=0.5,
                extraction_notes=f"JSON parsing failed: {e}",
            )

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
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]

        return text


