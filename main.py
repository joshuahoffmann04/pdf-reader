#!/usr/bin/env python3
"""
PDF Reader - Extract and structure PDF content by § sections

Main entry point for the PDF extraction tool.
Extracts text, tables, and images from PDFs and organizes content
into a hierarchical structure:
- Chapters (I., II., III., IV.)
- Sections (§1, §2, ...)
- AB-Excerpts (Allgemeine Bestimmungen)
- Appendices (Anlage 1-5) with optional sub-sections
"""

import argparse
import json
import sys
from pathlib import Path

from src.extractor import PDFExtractor
from src.parser import DocumentParser
from src.tables import TableExtractor
from src.images import ImageExtractor


def _get_images_for_pages(images: list, pages: list[int]) -> list:
    """Get images that appear on the given pages."""
    return [img for img in images if img.page_number in pages]


def _build_image_placeholder(img, output_dir: Path = None) -> str:
    """Build a placeholder string for an image."""
    if img.image_path:
        # Use relative path from output directory
        return f"[Bild: {img.image_path}]"
    return f"[Bild: Seite {img.page_number}, Index {img.index}]"


def _build_ab_reference(ab_section) -> str:
    """Build a reference string for an AB excerpt."""
    return f"[AB-Verweis: {ab_section.id} {ab_section.title}]"


def _enrich_section_content(section, images: list, ab_excerpts: list, output_dir: Path = None) -> str:
    """Enrich section content with image placeholders and AB references."""
    content_parts = [section.content]

    # Add image placeholders for images on this section's pages
    section_images = _get_images_for_pages(images, section.pages)
    if section_images:
        content_parts.append("")
        for img in section_images:
            content_parts.append(_build_image_placeholder(img, output_dir))

    # Add AB references for main sections
    if section.ab_references and not section.is_ab_excerpt:
        # Find the actual AB excerpt objects
        for ab_id in section.ab_references:
            matching_ab = [ab for ab in ab_excerpts if ab.id == ab_id]
            if matching_ab:
                content_parts.append("")
                content_parts.append(_build_ab_reference(matching_ab[0]))

    return "\n".join(content_parts)


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

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save output files.
        extract_images: Whether to extract images.
        extract_tables: Whether to extract tables.
        merge_tables: Whether to merge multi-page tables.
        output_format: Output format ("json", "markdown", or "both").

    Returns:
        Dictionary with extracted and structured content.
    """
    pdf_path = Path(pdf_path)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {pdf_path.name}")

    # Step 1: Extract raw text from PDF
    print("  [1/4] Extracting text...")
    extractor = PDFExtractor(preserve_layout=True)
    pdf_doc = extractor.extract(pdf_path)

    # Step 2: Parse document structure (with page markers for tracking)
    print("  [2/4] Parsing document structure...")
    parser = DocumentParser()
    parsed_doc = parser.parse(
        pdf_doc.get_full_text(include_page_markers=True),
        title=pdf_doc.title,
        metadata=pdf_doc.metadata
    )

    # Step 3: Extract tables
    tables = []
    if extract_tables:
        print("  [3/4] Extracting tables...")
        table_extractor = TableExtractor()
        tables = table_extractor.extract_from_pdf(pdf_path)

        if merge_tables:
            tables = table_extractor.merge_tables(tables)

    # Step 4: Extract images
    images = []
    if extract_images:
        print("  [4/4] Extracting images...")
        image_dir = output_dir / "images" if output_dir else None
        image_extractor = ImageExtractor(min_width=100, min_height=100)
        images = image_extractor.extract_from_pdf(pdf_path, output_dir=image_dir)

    # Build hierarchical output structure
    stats = parsed_doc.get_statistics()

    result = {
        "metadata": {
            "source": str(pdf_path),
            "title": parsed_doc.title,
            "total_pages": pdf_doc.total_pages,
            "pdf_metadata": pdf_doc.metadata
        },
        "preamble": parsed_doc.preamble,
        "chapters": [],
        "appendices": [],
        "tables": [t.to_dict() for t in tables],
        "images": [img.to_dict() for img in images],
        "statistics": {
            "chapters": stats["chapters"],
            "main_sections": stats["main_sections"],
            "ab_excerpts": stats["ab_excerpts"],
            "appendices": stats["appendices"],
            "appendix_sections": stats["appendix_sections"],
            "table_count": len(tables),
            "image_count": len(images)
        }
    }

    # Collect all AB excerpts for reference lookup
    all_ab_excerpts = parsed_doc.get_all_ab_excerpts()

    # Add chapters with their sections (enriched with images and AB references)
    for chapter in parsed_doc.chapters:
        chapter_data = {
            "id": chapter.id,
            "numeral": chapter.numeral,
            "title": chapter.title,
            "sections": [
                {
                    "id": s.id,
                    "number": s.number,
                    "title": s.title,
                    "content": _enrich_section_content(s, images, all_ab_excerpts, output_dir),
                    "pages": s.pages,
                    "ab_references": s.ab_references
                }
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
        result["chapters"].append(chapter_data)

    # Add appendices with their sub-sections (enriched with images)
    for appendix in parsed_doc.appendices:
        # Enrich appendix content with images
        appendix_images = _get_images_for_pages(images, appendix.pages)
        appendix_content = appendix.content
        if appendix_images:
            appendix_content += "\n\n" + "\n".join(
                _build_image_placeholder(img, output_dir) for img in appendix_images
            )

        appendix_data = {
            "id": appendix.id,
            "number": appendix.number,
            "title": appendix.title,
            "content": appendix_content,
            "pages": appendix.pages,
            "sections": [
                {
                    "id": s.id,
                    "number": s.number,
                    "title": s.title,
                    "content": _enrich_section_content(s, images, [], output_dir),
                    "pages": s.pages
                }
                for s in appendix.sections
            ]
        }
        result["appendices"].append(appendix_data)

    # Save output
    if output_dir:
        if output_format in ("json", "both"):
            json_path = output_dir / f"{pdf_path.stem}_extracted.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  Saved JSON: {json_path}")

        if output_format in ("markdown", "both"):
            md_path = output_dir / f"{pdf_path.stem}_extracted.md"
            markdown_content = generate_markdown(result, tables)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"  Saved Markdown: {md_path}")

    return result


def generate_markdown(result: dict, tables: list) -> str:
    """Generate Markdown representation of extracted content."""
    lines = []

    # Title
    lines.append(f"# {result['metadata']['title']}")
    lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append(f"- **Source**: {result['metadata']['source']}")
    lines.append(f"- **Pages**: {result['metadata']['total_pages']}")
    lines.append(f"- **Chapters**: {result['statistics']['chapters']}")
    lines.append(f"- **Main Sections**: {result['statistics']['main_sections']}")
    lines.append(f"- **AB Excerpts**: {result['statistics']['ab_excerpts']}")
    lines.append(f"- **Appendices**: {result['statistics']['appendices']}")
    lines.append(f"- **Tables**: {result['statistics']['table_count']}")
    lines.append(f"- **Images**: {result['statistics']['image_count']}")
    lines.append("")

    # Table of Contents
    lines.append("## Inhaltsverzeichnis")
    lines.append("")
    for chapter in result["chapters"]:
        lines.append(f"### {chapter['numeral']}. {chapter['title']}")
        for section in chapter["sections"]:
            lines.append(f"- [{section['id']} {section['title']}](#section-{section['number']})")
    lines.append("")
    lines.append("### Anlagen")
    for appendix in result["appendices"]:
        lines.append(f"- [{appendix['id']}: {appendix['title']}](#anlage-{appendix['number']})")
    lines.append("")

    # Preamble
    if result.get("preamble"):
        lines.append("---")
        lines.append("")
        lines.append("## Präambel")
        lines.append("")
        lines.append(result["preamble"])
        lines.append("")

    # Main chapters and sections
    for chapter in result["chapters"]:
        lines.append("---")
        lines.append("")
        lines.append(f"# {chapter['numeral']}. {chapter['title']}")
        lines.append("")

        for section in chapter["sections"]:
            lines.append(f"## {section['id']} {section['title']} {{#section-{section['number']}}}")
            lines.append("")
            lines.append(section["content"])
            lines.append("")

        # AB excerpts (if any)
        if chapter["ab_excerpts"]:
            lines.append("### Textauszüge aus den Allgemeinen Bestimmungen")
            lines.append("")
            for ab in chapter["ab_excerpts"]:
                lines.append(f"#### {ab['id']} {ab['title']} (AB)")
                lines.append("")
                lines.append(ab["content"])
                lines.append("")

    # Appendices
    if result["appendices"]:
        lines.append("---")
        lines.append("")
        lines.append("# Anlagen")
        lines.append("")

        for appendix in result["appendices"]:
            lines.append(f"## {appendix['id']}: {appendix['title']} {{#anlage-{appendix['number']}}}")
            lines.append("")

            if appendix["content"]:
                lines.append(appendix["content"])
                lines.append("")

            # Appendix sub-sections
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
            lines.append(f"### Tabelle {i + 1} (Seite {table.page_number})")
            lines.append("")
            lines.append(table.to_markdown())
            lines.append("")

    return "\n".join(lines)


def print_summary(result: dict):
    """Print a summary of the extraction results."""
    stats = result["statistics"]

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Title: {result['metadata']['title']}")
    print(f"Pages: {result['metadata']['total_pages']}")
    print(f"Chapters: {stats['chapters']}")
    print(f"Main Sections: {stats['main_sections']}")
    print(f"AB Excerpts: {stats['ab_excerpts']}")
    print(f"Appendices: {stats['appendices']} (with {stats['appendix_sections']} sub-sections)")
    print(f"Tables: {stats['table_count']}")
    print(f"Images: {stats['image_count']}")

    print("\nStructure:")
    for chapter in result["chapters"]:
        section_ids = [s["id"] for s in chapter["sections"]]
        section_range = f"{section_ids[0]}-{section_ids[-1]}" if section_ids else "none"
        print(f"  {chapter['numeral']}. {chapter['title']}: {section_range}")

    if result["appendices"]:
        print("\nAppendices:")
        for appendix in result["appendices"]:
            section_count = len(appendix.get("sections", []))
            sections_info = f" ({section_count} §)" if section_count > 0 else ""
            print(f"  {appendix['id']}: {appendix['title'][:40]}...{sections_info}")


def main():
    """Main entry point."""
    arg_parser = argparse.ArgumentParser(
        description="Extract and structure PDF content by § sections"
    )
    arg_parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )
    arg_parser.add_argument(
        "-o", "--output",
        default="output",
        help="Output directory (default: output)"
    )
    arg_parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)"
    )
    arg_parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image extraction"
    )
    arg_parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table extraction"
    )
    arg_parser.add_argument(
        "--no-merge-tables",
        action="store_true",
        help="Don't merge multi-page tables"
    )
    arg_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = arg_parser.parse_args()

    # Check if PDF exists
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    # Extract PDF
    result = extract_pdf(
        pdf_path=args.pdf_path,
        output_dir=args.output,
        extract_images=not args.no_images,
        extract_tables=not args.no_tables,
        merge_tables=not args.no_merge_tables,
        output_format=args.format
    )

    # Print summary
    print_summary(result)

    print("\nDone!")


if __name__ == "__main__":
    main()
