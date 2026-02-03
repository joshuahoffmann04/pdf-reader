import logging
from typing import Optional

import ollama

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self._client = ollama.Client(host=base_url)
        self._dimensions: Optional[int] = None

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        try:
            response = self._client.embed(model=self.model, input=text)
            embedding = response["embeddings"][0]
            self._dimensions = len(embedding)
            return embedding
        except ollama.ResponseError as e:
            raise RuntimeError(
                f"Ollama embedding failed for model '{self.model}': {e}"
            ) from e
        except Exception as e:
            if "Connection" in type(e).__name__ or "refused" in str(e).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is Ollama running? Start it with: ollama serve"
                ) from e
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        non_empty: list[tuple[int, str]] = [
            (i, t) for i, t in enumerate(texts) if t and t.strip()
        ]
        if not non_empty:
            return []

        try:
            input_texts = [t for _, t in non_empty]
            response = self._client.embed(model=self.model, input=input_texts)
            embeddings = response["embeddings"]

            if embeddings:
                self._dimensions = len(embeddings[0])

            result: list[list[float]] = [[] for _ in texts]
            for (orig_idx, _), embedding in zip(non_empty, embeddings):
                result[orig_idx] = embedding

            return result
        except ollama.ResponseError as e:
            raise RuntimeError(
                f"Ollama batch embedding failed for model '{self.model}': {e}"
            ) from e
        except Exception as e:
            if "Connection" in type(e).__name__ or "refused" in str(e).lower():
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Is Ollama running? Start it with: ollama serve"
                ) from e
            raise RuntimeError(f"Batch embedding failed: {e}") from e

    def health_check(self) -> dict[str, bool | str]:
        result = {
            "healthy": False,
            "ollama_running": False,
            "model_available": False,
            "model": self.model,
            "error": "",
        }

        try:
            models = self._client.list()
            result["ollama_running"] = True

            model_names = [m.model for m in models.models]
            result["model_available"] = any(
                m.startswith(self.model) for m in model_names
            )

            if not result["model_available"]:
                result["error"] = (
                    f"Model '{self.model}' not found. "
                    f"Available: {model_names}. "
                    f"Pull it with: ollama pull {self.model}"
                )
            else:
                result["healthy"] = True

        except Exception as e:
            result["error"] = f"Cannot connect to Ollama: {e}"

        return result
