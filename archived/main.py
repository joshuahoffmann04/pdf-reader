#!/usr/bin/env python3
"""
PDF Reader - Extract and structure PDF content by § sections.

Main entry point for the PDF extraction tool.
Extracts text, tables, and images from PDFs and organizes content
into a hierarchical structure.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.extractor import PDFExtractor, PDFDocument
from src.parser import DocumentParser, ParsedDocument, Section
from src.tables import TableExtractor, ExtractedTable
from src.images import ImageExtractor, ExtractedImage
from src.logging_config import setup_logging, get_logger

# Module logger
logger = get_logger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction."""
    extract_images: bool = True
    extract_tables: bool = True
    merge_cross_page_tables: bool = True
    min_image_width: int = 100
    min_image_height: int = 100
    output_format: str = "both"  # "json", "markdown", or "both"


@dataclass
class ExtractionResult:
    """Complete result of PDF extraction."""
    metadata: dict
    preamble: str
    chapters: list[dict]
    appendices: list[dict]
    tables: list[dict]
    images: list[dict]
    statistics: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": "1.0",
            "extracted_at": datetime.now().isoformat(),
            "metadata": self.metadata,
            "preamble": self.preamble,
            "chapters": self.chapters,
            "appendices": self.appendices,
            "tables": self.tables,
            "images": self.images,
            "statistics": self.statistics
        }


class PDFReaderPipeline:
    """
    Main pipeline for PDF extraction and structuring.

    Coordinates all extraction components and produces
    structured output in JSON and/or Markdown format.
    """

    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Extraction configuration. Uses defaults if not provided.
        """
        self.config = config or ExtractionConfig()
        self.pdf_extractor = PDFExtractor(preserve_layout=True)
        self.doc_parser = DocumentParser()
        self.table_extractor = TableExtractor()
        self.image_extractor = ImageExtractor(
            min_width=self.config.min_image_width,
            min_height=self.config.min_image_height
        )

    def extract(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None
    ) -> ExtractionResult:
        """
        Extract and structure content from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Optional directory for output files.

        Returns:
            ExtractionResult with all extracted content.

        Raises:
            FileNotFoundError: If PDF file doesn't exist.
            ValueError: If file is not a valid PDF.
        """
        # Validate input
        self._validate_pdf_path(pdf_path)

        logger.info(f"Processing: {pdf_path.name}")

        # Step 1: Extract raw text
        logger.info("Step 1/4: Extracting text...")
        pdf_doc = self.pdf_extractor.extract(pdf_path)
        logger.debug(f"Extracted {pdf_doc.total_pages} pages")

        # Step 2: Parse document structure
        logger.info("Step 2/4: Parsing document structure...")
        parsed_doc = self.doc_parser.parse(
            pdf_doc.get_full_text(include_page_markers=True),
            title=pdf_doc.title,
            metadata=pdf_doc.metadata
        )

        # Step 3: Extract tables
        tables = []
        if self.config.extract_tables:
            logger.info("Step 3/4: Extracting tables...")
            tables = self.table_extractor.extract_from_pdf(
                pdf_path,
                merge_cross_page=self.config.merge_cross_page_tables
            )
            logger.debug(f"Extracted {len(tables)} tables")

        # Step 4: Extract images
        images = []
        if self.config.extract_images:
            logger.info("Step 4/4: Extracting images...")
            image_dir = output_dir / "images" if output_dir else None
            images = self.image_extractor.extract_from_pdf(
                pdf_path,
                output_dir=image_dir
            )
            logger.debug(f"Extracted {len(images)} images")

        # Build result
        result = self._build_result(pdf_path, pdf_doc, parsed_doc, tables, images, output_dir)

        # Save output
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._save_output(result, tables, pdf_path.stem, output_dir)

        return result

    def _validate_pdf_path(self, pdf_path: Path) -> None:
        """Validate the PDF path."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {pdf_path}")

        # Check file size (warn if very large)
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            logger.warning(f"Large PDF file ({size_mb:.1f} MB) - extraction may take a while")

    def _build_result(
        self,
        pdf_path: Path,
        pdf_doc: PDFDocument,
        parsed_doc: ParsedDocument,
        tables: list[ExtractedTable],
        images: list[ExtractedImage],
        output_dir: Optional[Path]
    ) -> ExtractionResult:
        """Build the extraction result structure."""
        stats = parsed_doc.get_statistics()
        all_ab_excerpts = parsed_doc.get_all_ab_excerpts()

        # Build chapters with enriched content
        chapters = []
        for chapter in parsed_doc.chapters:
            chapter_data = {
                "id": chapter.id,
                "numeral": chapter.numeral,
                "title": chapter.title,
                "sections": [
                    self._build_section_dict(s, images, all_ab_excerpts, output_dir)
                    for s in chapter.sections
                ],
                "ab_excerpts": [
                    {
                        "id": s.id,
                        "number": s.number,
                        "title": s.title,
                        "content": s.content,
                        "pages": s.pages,
                        "follows_section": s.follows_section
                    }
                    for s in chapter.ab_excerpts
                ]
            }
            chapters.append(chapter_data)

        # Build appendices with enriched content
        appendices = []
        for appendix in parsed_doc.appendices:
            appendix_images = self._get_images_for_pages(images, appendix.pages)
            appendix_content = appendix.content
            if appendix_images:
                appendix_content += "\n\n" + "\n".join(
                    self._build_image_placeholder(img) for img in appendix_images
                )

            appendix_data = {
                "id": appendix.id,
                "number": appendix.number,
                "title": appendix.title,
                "content": appendix_content,
                "pages": appendix.pages,
                "sections": [
                    self._build_section_dict(s, images, [], output_dir)
                    for s in appendix.sections
                ]
            }
            appendices.append(appendix_data)

        return ExtractionResult(
            metadata={
                "source": str(pdf_path),
                "title": parsed_doc.title,
                "total_pages": pdf_doc.total_pages,
                "pdf_metadata": pdf_doc.metadata
            },
            preamble=parsed_doc.preamble,
            chapters=chapters,
            appendices=appendices,
            tables=[t.to_dict() for t in tables],
            images=[img.to_dict() for img in images],
            statistics={
                "chapters": stats["chapters"],
                "main_sections": stats["main_sections"],
                "ab_excerpts": stats["ab_excerpts"],
                "appendices": stats["appendices"],
                "appendix_sections": stats["appendix_sections"],
                "tables": len(tables),
                "images": len(images)
            }
        )

    def _build_section_dict(
        self,
        section: Section,
        images: list[ExtractedImage],
        ab_excerpts: list[Section],
        output_dir: Optional[Path]
    ) -> dict:
        """Build a section dictionary with enriched content."""
        return {
            "id": section.id,
            "number": section.number,
            "title": section.title,
            "content": self._enrich_section_content(section, images, ab_excerpts),
            "pages": section.pages,
            "ab_references": section.ab_references
        }

    def _enrich_section_content(
        self,
        section: Section,
        images: list[ExtractedImage],
        ab_excerpts: list[Section]
    ) -> str:
        """Enrich section content with image placeholders and AB references."""
        content_parts = [section.content]

        # Add image placeholders
        section_images = self._get_images_for_pages(images, section.pages)
        if section_images:
            content_parts.append("")
            for img in section_images:
                content_parts.append(self._build_image_placeholder(img))

        # Add AB references for main sections
        if section.ab_references and not section.is_ab_excerpt:
            for ab_id in section.ab_references:
                matching_ab = [ab for ab in ab_excerpts if ab.id == ab_id]
                if matching_ab:
                    content_parts.append("")
                    content_parts.append(self._build_ab_reference(matching_ab[0]))

        return "\n".join(content_parts)

    @staticmethod
    def _get_images_for_pages(images: list[ExtractedImage], pages: list[int]) -> list[ExtractedImage]:
        """Get images that appear on the given pages."""
        return [img for img in images if img.page_number in pages]

    @staticmethod
    def _build_image_placeholder(img: ExtractedImage) -> str:
        """Build a placeholder string for an image."""
        if img.image_path:
            return f"[Bild: {img.image_path}]"
        return f"[Bild: Seite {img.page_number}, Index {img.index}]"

    @staticmethod
    def _build_ab_reference(ab_section: Section) -> str:
        """Build a reference string for an AB excerpt."""
        return f"[AB-Verweis: {ab_section.id} {ab_section.title}]"

    def _save_output(
        self,
        result: ExtractionResult,
        tables: list[ExtractedTable],
        stem: str,
        output_dir: Path
    ) -> None:
        """Save extraction results to files."""
        fmt = self.config.output_format

        if fmt in ("json", "both"):
            json_path = output_dir / f"{stem}_extracted.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"Saved JSON: {json_path}")

        if fmt in ("markdown", "both"):
            md_path = output_dir / f"{stem}_extracted.md"
            markdown_content = self._generate_markdown(result, tables)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Saved Markdown: {md_path}")

    def _generate_markdown(self, result: ExtractionResult, tables: list[ExtractedTable]) -> str:
        """Generate Markdown representation of extracted content."""
        lines = []

        # Title
        lines.append(f"# {result.metadata['title']}")
        lines.append("")

        # Metadata
        lines.append("## Metadata")
        lines.append(f"- **Quelle**: {result.metadata['source']}")
        lines.append(f"- **Seiten**: {result.metadata['total_pages']}")
        lines.append(f"- **Kapitel**: {result.statistics['chapters']}")
        lines.append(f"- **Hauptsektionen**: {result.statistics['main_sections']}")
        lines.append(f"- **AB-Auszüge**: {result.statistics['ab_excerpts']}")
        lines.append(f"- **Anlagen**: {result.statistics['appendices']}")
        lines.append(f"- **Tabellen**: {result.statistics['tables']}")
        lines.append(f"- **Bilder**: {result.statistics['images']}")
        lines.append("")

        # Table of Contents
        lines.append("## Inhaltsverzeichnis")
        lines.append("")
        for chapter in result.chapters:
            lines.append(f"### {chapter['numeral']}. {chapter['title']}")
            for section in chapter["sections"]:
                lines.append(f"- [{section['id']} {section['title']}](#section-{section['number']})")
        lines.append("")
        lines.append("### Anlagen")
        for appendix in result.appendices:
            lines.append(f"- [{appendix['id']}: {appendix['title']}](#anlage-{appendix['number']})")
        lines.append("")

        # Preamble
        if result.preamble:
            lines.append("---")
            lines.append("")
            lines.append("## Präambel")
            lines.append("")
            lines.append(result.preamble)
            lines.append("")

        # Main chapters and sections
        for chapter in result.chapters:
            lines.append("---")
            lines.append("")
            lines.append(f"# {chapter['numeral']}. {chapter['title']}")
            lines.append("")

            for section in chapter["sections"]:
                lines.append(f"## {section['id']} {section['title']} {{#section-{section['number']}}}")
                lines.append("")
                lines.append(section["content"])
                lines.append("")

            # AB excerpts
            if chapter["ab_excerpts"]:
                lines.append("### Textauszüge aus den Allgemeinen Bestimmungen")
                lines.append("")
                for ab in chapter["ab_excerpts"]:
                    lines.append(f"#### {ab['id']} {ab['title']} (AB)")
                    if ab.get("follows_section"):
                        lines.append(f"*Folgt auf: {ab['follows_section']}*")
                    lines.append("")
                    lines.append(ab["content"])
                    lines.append("")

        # Appendices
        if result.appendices:
            lines.append("---")
            lines.append("")
            lines.append("# Anlagen")
            lines.append("")

            for appendix in result.appendices:
                lines.append(f"## {appendix['id']}: {appendix['title']} {{#anlage-{appendix['number']}}}")
                lines.append("")

                if appendix["content"]:
                    lines.append(appendix["content"])
                    lines.append("")

                for section in appendix.get("sections", []):
                    lines.append(f"### {section['id']} {section['title']}")
                    lines.append("")
                    lines.append(section["content"])
                    lines.append("")

        # Tables section
        if tables:
            lines.append("---")
            lines.append("")
            lines.append("# Tabellen")
            lines.append("")

            for i, table in enumerate(tables):
                page_info = f"Seite {table.page_number}"
                if table.end_page and table.end_page != table.page_number:
                    page_info = f"Seite {table.page_number}-{table.end_page}"
                lines.append(f"### Tabelle {i + 1} ({page_info})")
                lines.append("")
                lines.append(table.to_markdown())
                lines.append("")

        return "\n".join(lines)


def print_summary(result: ExtractionResult) -> None:
    """Print a summary of the extraction results."""
    stats = result.statistics

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Title: {result.metadata['title']}")
    print(f"Pages: {result.metadata['total_pages']}")
    print(f"Chapters: {stats['chapters']}")
    print(f"Main Sections: {stats['main_sections']}")
    print(f"AB Excerpts: {stats['ab_excerpts']}")
    print(f"Appendices: {stats['appendices']} (with {stats['appendix_sections']} sub-sections)")
    print(f"Tables: {stats['tables']}")
    print(f"Images: {stats['images']}")

    print("\nStructure:")
    for chapter in result.chapters:
        section_ids = [s["id"] for s in chapter["sections"]]
        section_range = f"{section_ids[0]}-{section_ids[-1]}" if section_ids else "none"
        print(f"  {chapter['numeral']}. {chapter['title']}: {section_range}")

    if result.appendices:
        print("\nAppendices:")
        for appendix in result.appendices:
            section_count = len(appendix.get("sections", []))
            title = appendix['title'][:40] + "..." if len(appendix['title']) > 40 else appendix['title']
            sections_info = f" ({section_count} §)" if section_count > 0 else ""
            print(f"  {appendix['id']}: {title}{sections_info}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and structure PDF content by § sections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf
  %(prog)s document.pdf -o output/ -f json
  %(prog)s document.pdf --no-images --verbose
        """
    )
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image extraction"
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table extraction"
    )
    parser.add_argument(
        "--no-merge-tables",
        action="store_true",
        help="Don't merge multi-page tables"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    setup_logging(level=log_level)

    # Validate PDF exists
    if not args.pdf_path.exists():
        logger.error(f"PDF file not found: {args.pdf_path}")
        return 1

    # Create configuration
    config = ExtractionConfig(
        extract_images=not args.no_images,
        extract_tables=not args.no_tables,
        merge_cross_page_tables=not args.no_merge_tables,
        output_format=args.format
    )

    # Run extraction
    try:
        pipeline = PDFReaderPipeline(config)
        result = pipeline.extract(args.pdf_path, args.output)

        # Print summary
        if not args.quiet:
            print_summary(result)
            print("\nDone!")

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


# Backwards compatibility function
def extract_pdf(
    pdf_path: str,
    output_dir: str = None,
    extract_images: bool = True,
    extract_tables: bool = True,
    merge_tables: bool = True,
    output_format: str = "json"
) -> dict:
    """
    Extract and structure content from a PDF file.

    This function provides backwards compatibility with the old API.
    For new code, use PDFReaderPipeline directly.
    """
    config = ExtractionConfig(
        extract_images=extract_images,
        extract_tables=extract_tables,
        merge_cross_page_tables=merge_tables,
        output_format=output_format
    )

    pipeline = PDFReaderPipeline(config)
    result = pipeline.extract(
        Path(pdf_path),
        Path(output_dir) if output_dir else None
    )

    return result.to_dict()


if __name__ == "__main__":
    sys.exit(main())
