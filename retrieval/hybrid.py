from collections import defaultdict

from .models import RetrievalHit


def rrf_merge(
    bm25_hits: list[RetrievalHit],
    vector_hits: list[RetrievalHit],
    top_k: int,
    rrf_k: int,
) -> list[RetrievalHit]:
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievalHit] = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
        by_id[hit.chunk_id] = hit

    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
        by_id.setdefault(hit.chunk_id, hit)

    merged = [
        RetrievalHit(
            chunk_id=chunk_id,
            score=score,
            text=by_id[chunk_id].text,
            metadata=by_id[chunk_id].metadata,
        )
        for chunk_id, score in scores.items()
    ]
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged[:top_k]
