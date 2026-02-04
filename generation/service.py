from __future__ import annotations

import json
from typing import Any

from .config import GenerationConfig
from .context_builder import build_context, parse_page_numbers
from .http_client import post_json
from .models import GenerateRequest, GenerateResponse, Citation
from .ollama_client import chat
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, RESPONSE_SCHEMA


class GenerationService:
    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig.from_env()

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        mode = request.mode or self.config.retrieval_mode
        if mode not in {"bm25", "vector", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        max_context_tokens = request.max_context_tokens or self.config.max_context_tokens
        output_tokens = request.output_tokens or self.config.output_tokens

        retrieval_payload = {
            "query": request.query,
            "top_k": self.config.candidate_top_k,
        }
        retrieval_url = f"{self.config.retrieval_base_url.rstrip('/')}/retrieve/{mode}"
        retrieval = post_json(retrieval_url, retrieval_payload, timeout=120)
        results = retrieval.get("results", []) or []

        context = build_context(
            query=request.query,
            results=results,
            max_context_tokens=max_context_tokens,
            system_prompt=SYSTEM_PROMPT,
            user_template=USER_PROMPT_TEMPLATE,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            query=request.query,
            context=context.context_text,
        )

        content = chat(
            base_url=self.config.ollama_base_url,
            model=self.config.ollama_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_schema=RESPONSE_SCHEMA,
            temperature=self.config.temperature,
            output_tokens=output_tokens,
        )

        payload = _safe_parse_json(content)
        citations = _normalize_citations(payload.get("citations", []), context.selected_chunks)

        answer = (payload.get("answer") or "").strip()
        missing_info = (payload.get("missing_info") or "").strip()
        if answer:
            missing_info = ""
        elif not missing_info:
            missing_info = "Information nicht im Dokument enthalten."

        return GenerateResponse(
            answer=answer,
            citations=citations,
            missing_info=missing_info,
            metadata={
                "mode": mode,
                "candidate_top_k": self.config.candidate_top_k,
                "selected_chunks": len(context.selected_chunks),
                "max_context_tokens": max_context_tokens,
                "output_tokens": output_tokens,
            },
        )


def _safe_parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json(text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            return {}


def _extract_json(text: str) -> str:
    if "{" in text:
        start = text.find("{")
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return text


def _normalize_citations(raw: list[dict[str, Any]], selected_hits: list[dict[str, Any]]) -> list[Citation]:
    mapping = {hit.get("chunk_id"): hit for hit in selected_hits}
    citations: list[Citation] = []

    for item in raw:
        chunk_id = item.get("chunk_id")
        if not chunk_id or chunk_id not in mapping:
            continue
        hit = mapping[chunk_id]
        metadata = hit.get("metadata", {}) or {}
        pages = parse_page_numbers(metadata)
        snippet = (item.get("snippet") or "").strip()
        if not snippet:
            snippet = (hit.get("text") or "").strip()[:240]
        citations.append(
            Citation(
                chunk_id=chunk_id,
                page_numbers=pages,
                snippet=snippet,
                score=hit.get("score"),
            )
        )

    if not citations and selected_hits:
        hit = selected_hits[0]
        metadata = hit.get("metadata", {}) or {}
        citations.append(
            Citation(
                chunk_id=hit.get("chunk_id", ""),
                page_numbers=parse_page_numbers(metadata),
                snippet=(hit.get("text") or "").strip()[:240],
                score=hit.get("score"),
            )
        )

    return citations
