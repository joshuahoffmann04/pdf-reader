"""
Ollama Embedder - Local embedding generation via Ollama API

Wraps the Ollama Python client to generate text embeddings locally.
Supports single and batch embedding with health checks.

Design:
- Thin wrapper around ollama.embed() (available since ollama 0.4+)
- Batch embedding for efficient ingestion
- Health check to verify Ollama is running and model is available
- No ChromaDB dependency — pure embedding logic

Usage:
    from vector_store.embedder import OllamaEmbedder

    embedder = OllamaEmbedder(model="nomic-embed-text")
    vector = embedder.embed("Ein Beispieltext")
    vectors = embedder.embed_batch(["Text 1", "Text 2"])
"""

import logging
from typing import Optional

import ollama

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """
    Generates text embeddings using a local Ollama model.

    The embedder connects to a running Ollama instance and uses a specified
    embedding model to convert text into dense vector representations.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize the embedder.

        Args:
            model: Ollama model name for embeddings.
            base_url: Ollama API base URL.
        """
        self.model = model
        self.base_url = base_url
        self._client = ollama.Client(host=base_url)
        self._dimensions: Optional[int] = None

    @property
    def dimensions(self) -> Optional[int]:
        """Return the embedding dimensions (available after first embed call)."""
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If embedding generation fails.
        """
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
        """
        Generate embeddings for multiple texts.

        Uses Ollama's batch embedding API for efficiency.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ConnectionError: If Ollama is not reachable.
            RuntimeError: If embedding generation fails.
        """
        if not texts:
            return []

        # Filter out empty texts but track their positions
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

            # Map embeddings back to original positions
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
        """
        Check if Ollama is running and the embedding model is available.

        Returns:
            Dict with 'healthy' (bool), 'ollama_running' (bool),
            'model_available' (bool), and 'error' (str, if any).
        """
        result = {
            "healthy": False,
            "ollama_running": False,
            "model_available": False,
            "model": self.model,
            "error": "",
        }

        try:
            # Check if Ollama is running
            models = self._client.list()
            result["ollama_running"] = True

            # Check if our model is available
            model_names = [m.model for m in models.models]
            # Match by prefix (e.g., "nomic-embed-text" matches "nomic-embed-text:latest")
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
