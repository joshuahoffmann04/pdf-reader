from __future__ import annotations

from .citations import normalize_and_select_citations
from .config import GenerationConfig
from .context_builder import build_context
from .http_client import post_json
from .models import GenerateRequest, GenerateResponse, Citation
from .postprocess import postprocess_answer
from .json_utils import safe_parse_json
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

        payload = safe_parse_json(content)
        answer = postprocess_answer((payload.get("answer") or "").strip(), request.query)
        missing_info = (payload.get("missing_info") or "").strip()
        citations_raw = payload.get("citations", [])
        citations_norm = normalize_and_select_citations(
            citations_raw,
            context.selected_chunks,
            answer=answer,
            query=request.query,
        )
        citations = [Citation(**c) for c in citations_norm]

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
