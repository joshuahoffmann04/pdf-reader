from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    data_dir: str = "data/retrieval"
    collection_name: str = "retrieval_chunks"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    max_context_tokens: int = 1024
    rrf_k: int = 60
