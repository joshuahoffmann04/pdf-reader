from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import ChunkingResult


@dataclass
class ChunkingPaths:
    document_id: str
    chunk_dir: Path
    chunk_file: Path


class ChunkingStorage:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def build_paths(self, document_id: str) -> ChunkingPaths:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        chunk_dir = self.data_dir / document_id / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / f"{document_id}_{timestamp}.json"
        return ChunkingPaths(
            document_id=document_id,
            chunk_dir=chunk_dir,
            chunk_file=chunk_file,
        )

    def save(self, result: ChunkingResult) -> ChunkingPaths:
        paths = self.build_paths(result.document_id)
        result.save(str(paths.chunk_file))
        return paths
