#!/usr/bin/env python3
"""
PDF Reader - Extract and structure PDF content by § sections

Main entry point for the PDF extraction tool.
Extracts text, tables, and images from PDFs and organizes content
by § (paragraph) sections for AI model consumption.
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractor import PDFExtractor
from src.parser import SectionParser
from src.tables import TableExtractor
from src.images import ImageExtractor
from src.evaluation import Evaluator


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

    # Step 2: Parse sections
    print("  [2/4] Parsing sections...")
    parser = SectionParser(language="de")
    parsed_doc = parser.parse(
        pdf_doc.get_full_text(),
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

    # Build output structure
    result = {
        "metadata": {
            "source": str(pdf_path),
            "title": parsed_doc.title,
            "total_pages": pdf_doc.total_pages,
            "pdf_metadata": pdf_doc.metadata
        },
        "preamble": parsed_doc.preamble,
        "sections": [],
        "appendices": [],
        "tables": [t.to_dict() for t in tables],
        "images": [img.to_dict() for img in images],
        "statistics": {
            "section_count": len(parsed_doc.sections),
            "appendix_count": len(parsed_doc.appendices),
            "table_count": len(tables),
            "image_count": len(images)
        }
    }

    # Add sections
    for section in parsed_doc.sections:
        section_data = {
            "id": section.id,
            "title": section.title,
            "content": section.content
        }
        result["sections"].append(section_data)

    # Add appendices
    for appendix in parsed_doc.appendices:
        appendix_data = {
            "id": appendix.id,
            "title": appendix.title,
            "content": appendix.content
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
    lines.append(f"- **Sections**: {result['statistics']['section_count']}")
    lines.append(f"- **Appendices**: {result['statistics']['appendix_count']}")
    lines.append(f"- **Tables**: {result['statistics']['table_count']}")
    lines.append(f"- **Images**: {result['statistics']['image_count']}")
    lines.append("")

    # Table of Contents
    lines.append("## Inhaltsverzeichnis")
    lines.append("")
    for section in result["sections"]:
        lines.append(f"- [{section['id']} {section['title']}](#{section['id'].replace('§', 'section-')})")
    for appendix in result["appendices"]:
        lines.append(f"- [{appendix['id']}](#{appendix['id'].replace(' ', '-').lower()})")
    lines.append("")

    # Preamble
    if result.get("preamble"):
        lines.append("## Präambel")
        lines.append("")
        lines.append(result["preamble"])
        lines.append("")

    # Main sections
    lines.append("---")
    lines.append("")

    for section in result["sections"]:
        anchor = section["id"].replace("§", "section-")
        lines.append(f"## {section['id']} {section['title']} {{#{anchor}}}")
        lines.append("")
        lines.append(section["content"])
        lines.append("")
        lines.append("---")
        lines.append("")

    # Appendices
    if result["appendices"]:
        lines.append("# Anlagen")
        lines.append("")

        for appendix in result["appendices"]:
            anchor = appendix["id"].replace(" ", "-").lower()
            title = appendix["title"] if appendix["title"] else ""
            lines.append(f"## {appendix['id']} {title} {{#{anchor}}}")
            lines.append("")
            lines.append(appendix["content"])
            lines.append("")
            lines.append("---")
            lines.append("")

    # Tables section
    if tables:
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

    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Title: {result['metadata']['title']}")
    print(f"Pages: {result['metadata']['total_pages']}")
    print(f"Sections found: {stats['section_count']}")
    print(f"Appendices found: {stats['appendix_count']}")
    print(f"Tables extracted: {stats['table_count']}")
    print(f"Images extracted: {stats['image_count']}")

    print("\nSections:")
    for section in result["sections"]:
        content_preview = section["content"][:50].replace("\n", " ") + "..."
        print(f"  {section['id']} {section['title']}")

    if result["appendices"]:
        print("\nAppendices:")
        for appendix in result["appendices"]:
            print(f"  {appendix['id']} {appendix['title']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract and structure PDF content by § sections"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "-o", "--output",
        default="output",
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

    args = parser.parse_args()

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
