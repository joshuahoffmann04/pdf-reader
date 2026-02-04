import json
from typing import Any, Optional

import chromadb

from .embedder import OllamaEmbedder

from .models import RetrievalHit


class VectorIndex:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedder: Optional[OllamaEmbedder] = None,
        chroma_client: Optional[chromadb.ClientAPI] = None,
    ):
        self.embedder = embedder or OllamaEmbedder()
        self._client = chroma_client or chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [self._flatten_metadata(chunk) for chunk in chunks]
        embeddings = self.embedder.embed_batch(documents)
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]:
        query_embedding = self.embedder.embed(query)
        params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            params["where"] = filters
        results = self._collection.query(**params)
        hits: list[RetrievalHit] = []
        if not results["ids"] or not results["ids"][0]:
            return hits
        for idx, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx]
            similarity = 1 - distance
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    score=float(similarity),
                    text=results["documents"][0][idx],
                    metadata=results["metadatas"][0][idx],
                )
            )
        return hits

    @staticmethod
    def _flatten_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
        metadata = chunk.get("metadata", {})
        flat: dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                flat[key] = ",".join(str(item) for item in value)
            elif isinstance(value, dict):
                flat[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat[key] = value
        flat["document_id"] = chunk.get("document_id", "")
        return flat


