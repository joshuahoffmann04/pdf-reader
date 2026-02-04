from dataclasses import dataclass
import os


@dataclass
class GenerationConfig:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"
    retrieval_base_url: str = "http://localhost:8000"
    retrieval_mode: str = "hybrid"
    max_context_tokens: int = 2048
    output_tokens: int = 512
    candidate_top_k: int = 30
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        def _int(name: str, default: int) -> int:
            value = os.environ.get(name)
            return int(value) if value else default

        def _float(name: str, default: float) -> float:
            value = os.environ.get(name)
            return float(value) if value else default

        return cls(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_model=os.environ.get("OLLAMA_MODEL", cls.ollama_model),
            retrieval_base_url=os.environ.get("RETRIEVAL_BASE_URL", cls.retrieval_base_url),
            retrieval_mode=os.environ.get("GENERATION_RETRIEVAL_MODE", cls.retrieval_mode),
            max_context_tokens=_int("GENERATION_MAX_CONTEXT_TOKENS", cls.max_context_tokens),
            output_tokens=_int("GENERATION_OUTPUT_TOKENS", cls.output_tokens),
            candidate_top_k=_int("GENERATION_CANDIDATE_TOP_K", cls.candidate_top_k),
            temperature=_float("GENERATION_TEMPERATURE", cls.temperature),
        )
