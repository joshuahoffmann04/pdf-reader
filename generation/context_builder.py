from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunking.token_counter import count_tokens


@dataclass
class ContextBuildResult:
    context_text: str
    selected_chunks: list[dict[str, Any]]
    available_tokens: int
    used_tokens: int


def _parse_page_numbers(metadata: dict[str, Any]) -> list[int]:
    raw = metadata.get("page_numbers")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(p) for p in raw if isinstance(p, int) or str(p).isdigit()]
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        nums = []
        for p in parts:
            if p.isdigit():
                nums.append(int(p))
        return nums
    return []


def _chunk_block(idx: int, hit: dict[str, Any]) -> str:
    chunk_id = hit.get("chunk_id", "")
    metadata = hit.get("metadata", {}) or {}
    pages = _parse_page_numbers(metadata)
    page_text = ", ".join(str(p) for p in pages) if pages else "unknown"
    header = f"[Chunk {idx}] id={chunk_id} pages={page_text}\n"
    text = (hit.get("text") or "").strip()
    return header + text


def build_context(
    query: str,
    results: list[dict[str, Any]],
    max_context_tokens: int,
    system_prompt: str,
    user_template: str,
) -> ContextBuildResult:
    system_tokens = count_tokens(system_prompt)
    base_user = user_template.format(query=query, context="")
    base_tokens = count_tokens(base_user)

    available = max_context_tokens - system_tokens - base_tokens
    if available < 0:
        available = 0

    parts: list[str] = []
    selected: list[dict[str, Any]] = []
    used = 0

    for idx, hit in enumerate(results, start=1):
        block = _chunk_block(idx, hit)
        block_tokens = count_tokens(block)
        if block_tokens <= available:
            parts.append(block)
            selected.append(hit)
            available -= block_tokens
            used += block_tokens
            continue

        if not parts:
            # If nothing fits, include a truncated prefix.
            prefix = block[: max(200, int(len(block) * 0.5))]
            parts.append(prefix)
            selected.append(hit)
            used += count_tokens(prefix)
        break

    return ContextBuildResult(
        context_text="\n\n".join(parts).strip(),
        selected_chunks=selected,
        available_tokens=available,
        used_tokens=used,
    )
