"""
Vision LLM Processor

Handles communication with Vision-capable LLMs (Claude, GPT-4V) for
PDF content extraction and transformation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from .pdf_to_images import PDFToImages, PageImage
from .models import (
    DocumentContext,
    PageContent,
    ProcessingConfig,
    DocumentType,
)
from .prompts import (
    CONTEXT_ANALYSIS_SYSTEM,
    CONTEXT_ANALYSIS_USER,
    get_page_extraction_system_prompt,
    get_page_extraction_user_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a PDF document."""
    context: DocumentContext
    pages: list[PageContent]
    processing_time_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    errors: list[str]


class VisionProcessor:
    """
    Main processor for Vision-LLM based PDF extraction.

    Two-phase approach:
    1. Context Analysis: Understand the entire document structure
    2. Page Extraction: Extract content page by page with full context
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Vision Processor.

        Args:
            config: Processing configuration
            api_key: API key (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)
        """
        self.config = config or ProcessingConfig()
        self.api_key = api_key
        self.pdf_converter = PDFToImages(dpi=150)

        # Initialize API client
        self._init_client()

        # Statistics
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.config.api_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
        elif self.config.api_provider == "openai":
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        else:
            raise ValueError(f"Unknown API provider: {self.config.api_provider}")

    def process_document(
        self,
        pdf_path: str | Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ProcessingResult:
        """
        Process an entire PDF document.

        Args:
            pdf_path: Path to the PDF file
            progress_callback: Optional callback(current_page, total_pages, status)

        Returns:
            ProcessingResult with context and all page contents
        """
        pdf_path = Path(pdf_path)
        start_time = time.time()
        errors = []

        logger.info(f"Starting document processing: {pdf_path}")

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
        pages = []
        for page_num in range(1, total_pages + 1):
            if progress_callback:
                progress_callback(page_num, total_pages, f"Processing page {page_num}...")

            try:
                page_content = self._extract_page(
                    pdf_path,
                    page_num,
                    total_pages,
                    context.to_dict(),
                )
                pages.append(page_content)
                logger.debug(f"Page {page_num} extracted successfully")
            except Exception as e:
                logger.error(f"Page {page_num} extraction failed: {e}")
                errors.append(f"Page {page_num} error: {str(e)}")
                # Create placeholder
                pages.append(PageContent(
                    page_number=page_num,
                    content=f"[Extraction failed: {str(e)}]",
                ))

            # Rate limiting
            time.sleep(0.5)

        processing_time = time.time() - start_time
        logger.info(f"Processing complete in {processing_time:.1f}s")

        return ProcessingResult(
            context=context,
            pages=pages,
            processing_time_seconds=processing_time,
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            errors=errors,
        )

    def _analyze_context(
        self,
        pdf_path: Path,
        total_pages: int,
    ) -> DocumentContext:
        """
        Phase 1: Analyze document context using sample pages.

        Uses first page, a middle page, and last page for context.
        """
        # Select sample pages for context
        sample_pages = [1]
        if total_pages > 2:
            sample_pages.append(total_pages // 2)
        if total_pages > 1:
            sample_pages.append(total_pages)

        # Render sample pages
        images = self.pdf_converter.render_pages_batch(pdf_path, sample_pages)

        # Build message content with images
        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.mime_type,
                    "data": img.image_base64,
                }
            })

        content.append({
            "type": "text",
            "text": CONTEXT_ANALYSIS_USER,
        })

        # Call API
        response = self._call_api(
            system_prompt=CONTEXT_ANALYSIS_SYSTEM,
            content=content,
        )

        # Parse response
        return self._parse_context_response(response, total_pages)

    def _extract_page(
        self,
        pdf_path: Path,
        page_number: int,
        total_pages: int,
        context_dict: dict,
    ) -> PageContent:
        """
        Phase 2: Extract content from a single page.
        """
        # Render page
        page_image = self.pdf_converter.render_page(pdf_path, page_number)

        # Build message content
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": page_image.mime_type,
                    "data": page_image.image_base64,
                }
            },
            {
                "type": "text",
                "text": get_page_extraction_user_prompt(page_number, total_pages),
            }
        ]

        # Call API with context
        system_prompt = get_page_extraction_system_prompt(context_dict)
        response = self._call_api(
            system_prompt=system_prompt,
            content=content,
        )

        # Parse response
        return self._parse_page_response(response, page_number)

    def _call_api(
        self,
        system_prompt: str,
        content: list,
    ) -> str:
        """Make an API call to the Vision LLM."""
        if self.config.api_provider == "anthropic":
            return self._call_anthropic(system_prompt, content)
        elif self.config.api_provider == "openai":
            return self._call_openai(system_prompt, content)
        else:
            raise ValueError(f"Unknown provider: {self.config.api_provider}")

    def _call_anthropic(self, system_prompt: str, content: list) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens_per_request,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": content}
            ],
        )

        # Track tokens
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return response.content[0].text

    def _call_openai(self, system_prompt: str, content: list) -> str:
        """Call OpenAI GPT-4V API."""
        # Convert content format for OpenAI
        openai_content = []
        for item in content:
            if item["type"] == "image":
                openai_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{item['source']['media_type']};base64,{item['source']['data']}"
                    }
                })
            else:
                openai_content.append(item)

        response = self.client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens_per_request,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": openai_content},
            ],
        )

        # Track tokens
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens

        return response.choices[0].message.content

    def _parse_context_response(self, response: str, total_pages: int) -> DocumentContext:
        """Parse the context analysis response."""
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
            }
            doc_type = doc_type_map.get(doc_type_str, DocumentType.OTHER)

            return DocumentContext(
                document_type=doc_type,
                title=data.get("title", "Unknown"),
                institution=data.get("institution", "Unknown"),
                version_date=data.get("version_date"),
                total_pages=total_pages,
                chapters=data.get("chapters", []),
                main_topics=data.get("main_topics", []),
                degree_program=data.get("degree_program"),
                abbreviations=data.get("abbreviations", {}),
                key_terms=data.get("key_terms", []),
                referenced_documents=data.get("referenced_documents", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse context response: {e}")
            return DocumentContext(
                document_type=DocumentType.OTHER,
                title="Unknown",
                institution="Unknown",
                total_pages=total_pages,
            )

    def _parse_page_response(self, response: str, page_number: int) -> PageContent:
        """Parse the page extraction response."""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            return PageContent(
                page_number=page_number,
                content=data.get("content", ""),
                section_numbers=data.get("section_numbers", []),
                section_titles=data.get("section_titles", []),
                has_table=data.get("has_table", False),
                has_list=data.get("has_list", False),
                has_image=data.get("has_image", False),
                internal_references=data.get("internal_references", []),
                external_references=data.get("external_references", []),
                continues_from_previous=data.get("continues_from_previous", False),
                continues_to_next=data.get("continues_to_next", False),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse page {page_number} response: {e}")
            # Return raw content as fallback
            return PageContent(
                page_number=page_number,
                content=response,
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
