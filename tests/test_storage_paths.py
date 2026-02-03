from pathlib import Path

from chunking.models import ChunkingConfig, ChunkingResult
from chunking.storage import ChunkingStorage
from pdf_extractor.models import DocumentContext, DocumentType, ExtractionResult
from pdf_extractor.storage import ExtractionStorage


def test_extraction_storage_paths(tmp_path: Path) -> None:
    storage = ExtractionStorage(str(tmp_path))
    result = ExtractionResult(
        source_file="pdfs/example.pdf",
        context=DocumentContext(
            document_type=DocumentType.OTHER,
            title="Example",
            institution="Test",
            total_pages=1,
        ),
        pages=[],
    )
    paths = storage.save(result)
    assert paths.document_id == "example"
    assert paths.extraction_file.exists()


def test_chunking_storage_paths(tmp_path: Path) -> None:
    storage = ChunkingStorage(str(tmp_path))
    result = ChunkingResult(
        source_file="pdfs/example.pdf",
        document_id="example",
        config=ChunkingConfig(),
        chunks=[],
    )
    paths = storage.save(result)
    assert paths.document_id == "example"
    assert paths.chunk_file.exists()
