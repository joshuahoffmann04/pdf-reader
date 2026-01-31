"""
RAG Chunk Generator

Converts processed page contents into optimized chunks for RAG retrieval.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .models import (
    DocumentContext,
    PageContent,
    RAGChunk,
    ChunkType,
    ProcessingConfig,
)
from .vision_processor import ProcessingResult


@dataclass
class ChunkingStats:
    """Statistics about the chunking process."""
    total_chunks: int
    avg_chunk_length: float
    min_chunk_length: int
    max_chunk_length: int
    chunks_by_type: dict[str, int]


class ChunkGenerator:
    """
    Generates RAG-optimized chunks from processed PDF content.

    Strategies:
    1. Section-based: Split by §/paragraph boundaries
    2. Semantic: Split by topic/content changes
    3. Fixed-size: Split by token count with overlap
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()

        # Regex patterns for structure detection
        self.section_pattern = re.compile(r'§\s*(\d+)\s*([^:§]*?)(?::|(?=\s*\())')
        self.paragraph_pattern = re.compile(r'\((\d+)\)')
        self.reference_pattern = re.compile(r'§\s*\d+(?:\s*Abs(?:atz|\.)\s*\d+)?')

    def generate_chunks(
        self,
        result: ProcessingResult,
        source_document: str,
    ) -> list[RAGChunk]:
        """
        Generate RAG chunks from processing result.

        Args:
            result: ProcessingResult from VisionProcessor
            source_document: Name/identifier of the source document

        Returns:
            List of RAGChunk objects
        """
        chunks = []

        # First, merge continuous content across pages
        merged_content = self._merge_page_content(result.pages)

        # Then split into chunks
        for section in merged_content:
            section_chunks = self._chunk_section(
                section,
                source_document,
                result.context,
            )
            chunks.extend(section_chunks)

        # Add document-level metadata chunk
        meta_chunk = self._create_metadata_chunk(result.context, source_document)
        chunks.insert(0, meta_chunk)

        # Link related chunks
        self._link_related_chunks(chunks)

        return chunks

    def _merge_page_content(self, pages: list[PageContent]) -> list[dict]:
        """
        Merge content that spans multiple pages.

        Returns list of sections with merged content.
        """
        sections = []
        current_section = None

        for page in pages:
            content = page.content

            # Check if this continues previous section
            if page.continues_from_previous and current_section:
                current_section["content"] += " " + content
                current_section["pages"].append(page.page_number)

                # Update section info if new sections found
                current_section["section_numbers"].extend(page.section_numbers)
                current_section["section_titles"].extend(page.section_titles)
            else:
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    "content": content,
                    "pages": [page.page_number],
                    "section_numbers": list(page.section_numbers),
                    "section_titles": list(page.section_titles),
                    "has_table": page.has_table,
                    "has_list": page.has_list,
                    "internal_references": list(page.internal_references),
                    "external_references": list(page.external_references),
                }

            # Check if section continues to next page
            if not page.continues_to_next and current_section:
                sections.append(current_section)
                current_section = None

        # Don't forget last section
        if current_section:
            sections.append(current_section)

        return sections

    def _chunk_section(
        self,
        section: dict,
        source_document: str,
        context: DocumentContext,
    ) -> list[RAGChunk]:
        """
        Split a section into appropriate chunks.
        """
        content = section["content"]
        chunks = []

        # Determine chunk type
        chunk_type = ChunkType.SECTION
        if section.get("has_table"):
            chunk_type = ChunkType.TABLE
        elif section.get("has_list"):
            chunk_type = ChunkType.LIST

        # Check if content is short enough for single chunk
        if len(content) <= self.config.max_chunk_size:
            chunk = self._create_chunk(
                content=content,
                chunk_type=chunk_type,
                source_document=source_document,
                pages=section["pages"],
                section_numbers=section["section_numbers"],
                section_titles=section["section_titles"],
                context=context,
            )
            chunks.append(chunk)
        else:
            # Split into smaller chunks
            sub_chunks = self._split_content(content)
            for i, sub_content in enumerate(sub_chunks):
                chunk = self._create_chunk(
                    content=sub_content,
                    chunk_type=chunk_type,
                    source_document=source_document,
                    pages=section["pages"],
                    section_numbers=section["section_numbers"],
                    section_titles=section["section_titles"],
                    context=context,
                    chunk_index=i,
                )
                chunks.append(chunk)

        return chunks

    def _split_content(self, content: str) -> list[str]:
        """
        Split content into chunks of appropriate size.

        Tries to split at natural boundaries:
        1. Paragraph markers (1), (2), etc.
        2. Sentence boundaries
        3. Word boundaries (last resort)
        """
        target_size = self.config.target_chunk_size
        max_size = self.config.max_chunk_size
        overlap = self.config.chunk_overlap

        chunks = []

        # Try to split by paragraph markers first
        paragraphs = re.split(r'(?=\(\d+\)\s)', content)

        current_chunk = ""
        for para in paragraphs:
            if not para.strip():
                continue

            # If adding this paragraph would exceed max, save current and start new
            if len(current_chunk) + len(para) > max_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from end of previous chunk
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:] + para
                else:
                    current_chunk = para
            else:
                current_chunk += para

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If still too large, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_size:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > max_size and current:
                        final_chunks.append(current.strip())
                        current = sent
                    else:
                        current += " " + sent if current else sent
                if current.strip():
                    final_chunks.append(current.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks if final_chunks else [content]

    def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        source_document: str,
        pages: list[int],
        section_numbers: list[str],
        section_titles: list[str],
        context: DocumentContext,
        chunk_index: Optional[int] = None,
    ) -> RAGChunk:
        """Create a single RAG chunk."""
        # Generate unique ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        section_part = section_numbers[0] if section_numbers else "general"
        index_part = f"-{chunk_index}" if chunk_index is not None else ""
        chunk_id = f"{source_document[:20]}-{section_part}{index_part}-{content_hash}"

        # Extract references from content
        references = self.reference_pattern.findall(content)

        # Determine chapter from context
        chapter = self._determine_chapter(section_numbers, context)

        # Extract keywords (simple approach - can be enhanced)
        keywords = self._extract_keywords(content, context)

        # Extract paragraph number if present
        paragraph = None
        para_match = self.paragraph_pattern.search(content[:50])
        if para_match:
            paragraph = f"({para_match.group(1)})"

        return RAGChunk(
            id=chunk_id,
            text=content,
            chunk_type=chunk_type,
            source_document=source_document,
            source_pages=pages,
            section_number=section_numbers[0] if section_numbers else None,
            section_title=section_titles[0] if section_titles else None,
            chapter=chapter,
            paragraph=paragraph,
            topics=context.main_topics[:3] if context.main_topics else [],
            keywords=keywords,
            related_sections=list(set(references)),
        )

    def _create_metadata_chunk(
        self,
        context: DocumentContext,
        source_document: str,
    ) -> RAGChunk:
        """Create a metadata chunk for document-level information."""
        meta_text = f"""Dokument: {context.title}
Institution: {context.institution}
Typ: {context.document_type.value}
"""
        if context.degree_program:
            meta_text += f"Studiengang: {context.degree_program}\n"
        if context.version_date:
            meta_text += f"Version: {context.version_date}\n"
        if context.chapters:
            meta_text += f"Gliederung: {', '.join(context.chapters)}\n"
        if context.abbreviations:
            abbrevs = [f"{k} = {v}" for k, v in context.abbreviations.items()]
            meta_text += f"Abkürzungen: {'; '.join(abbrevs)}\n"

        return RAGChunk(
            id=f"{source_document[:20]}-metadata",
            text=meta_text,
            chunk_type=ChunkType.METADATA,
            source_document=source_document,
            source_pages=[1],
            topics=["Dokumentinformation", "Metadaten"],
            keywords=["Prüfungsordnung", context.institution] + context.key_terms[:5],
        )

    def _determine_chapter(
        self,
        section_numbers: list[str],
        context: DocumentContext,
    ) -> Optional[str]:
        """Determine which chapter a section belongs to."""
        if not section_numbers or not context.chapters:
            return None

        # Simple heuristic based on section number
        # This could be enhanced with the actual chapter mapping
        try:
            section_num = int(re.search(r'\d+', section_numbers[0]).group())
            if section_num <= 3:
                return context.chapters[0] if context.chapters else None
            elif section_num <= 17:
                return context.chapters[1] if len(context.chapters) > 1 else None
            elif section_num <= 38:
                return context.chapters[2] if len(context.chapters) > 2 else None
            else:
                return context.chapters[3] if len(context.chapters) > 3 else None
        except (AttributeError, ValueError):
            return None

    def _extract_keywords(self, content: str, context: DocumentContext) -> list[str]:
        """Extract keywords from content."""
        keywords = []

        # Check for known key terms
        content_lower = content.lower()
        for term in context.key_terms:
            if term.lower() in content_lower:
                keywords.append(term)

        # Check for common academic terms
        common_terms = [
            "Leistungspunkte", "ECTS", "Modul", "Prüfung", "Klausur",
            "Regelstudienzeit", "Semester", "Vorlesung", "Übung",
            "Bachelorarbeit", "Masterarbeit", "Note", "Bewertung",
        ]
        for term in common_terms:
            if term.lower() in content_lower and term not in keywords:
                keywords.append(term)

        return keywords[:10]  # Limit to 10 keywords

    def _link_related_chunks(self, chunks: list[RAGChunk]) -> None:
        """Link chunks that reference each other."""
        # Build section number index
        section_index = {}
        for chunk in chunks:
            if chunk.section_number:
                section_index[chunk.section_number] = chunk.id

        # Link references
        for chunk in chunks:
            for ref in chunk.related_sections:
                # Clean reference to just section number
                match = re.search(r'§\s*(\d+)', ref)
                if match:
                    section_num = f"§{match.group(1)}"
                    if section_num in section_index:
                        if section_index[section_num] not in chunk.related_sections:
                            chunk.related_sections.append(section_index[section_num])

    def export_jsonl(self, chunks: list[RAGChunk], output_path: str | Path) -> None:
        """Export chunks to JSONL format."""
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk.to_jsonl_entry() + '\n')

    def export_json(self, chunks: list[RAGChunk], output_path: str | Path) -> None:
        """Export chunks to JSON format."""
        output_path = Path(output_path)
        data = {
            "total_chunks": len(chunks),
            "chunks": [chunk.to_dict() for chunk in chunks],
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_stats(self, chunks: list[RAGChunk]) -> ChunkingStats:
        """Get statistics about the generated chunks."""
        if not chunks:
            return ChunkingStats(
                total_chunks=0,
                avg_chunk_length=0,
                min_chunk_length=0,
                max_chunk_length=0,
                chunks_by_type={},
            )

        lengths = [len(chunk.text) for chunk in chunks]
        type_counts = {}
        for chunk in chunks:
            type_name = chunk.chunk_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return ChunkingStats(
            total_chunks=len(chunks),
            avg_chunk_length=sum(lengths) / len(lengths),
            min_chunk_length=min(lengths),
            max_chunk_length=max(lengths),
            chunks_by_type=type_counts,
        )
