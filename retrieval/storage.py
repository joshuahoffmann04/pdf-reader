import json
from pathlib import Path
from typing import Any

from .models import ChunkInput, DocumentSummary


class ChunkStore:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.data_dir / "chunks.jsonl"

    def save_chunks(self, document_id: str, chunks: list[ChunkInput]) -> None:
        with self.chunks_file.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                payload = {
                    "document_id": document_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def load_chunks(self) -> list[dict[str, Any]]:
        if not self.chunks_file.exists():
            return []
        chunks = []
        with self.chunks_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        return chunks

    def list_documents(self) -> list[DocumentSummary]:
        counts: dict[str, int] = {}
        for chunk in self.load_chunks():
            doc_id = chunk["document_id"]
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return [
            DocumentSummary(document_id=doc_id, chunk_count=count)
            for doc_id, count in sorted(counts.items())
        ]
