from .config import ChunkingServiceConfig
from .chunker import DocumentChunker
from .models import ChunkingResult
from .storage import ChunkingStorage


class ChunkingService:
    def __init__(self, config: ChunkingServiceConfig | None = None):
        self.config = config or ChunkingServiceConfig()
        self.chunker = DocumentChunker(self.config.chunking)
        self.storage = ChunkingStorage(self.config.data_dir)

    def chunk_from_file(self, extraction_path: str) -> ChunkingResult:
        return self.chunker.chunk_from_file(extraction_path)

    def chunk_and_save(self, extraction_path: str) -> tuple[ChunkingResult, str]:
        result = self.chunk_from_file(extraction_path)
        paths = self.storage.save(result)
        return result, str(paths.chunk_file)
