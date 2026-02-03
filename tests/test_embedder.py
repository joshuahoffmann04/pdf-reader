"""Tests for vector_store.embedder — OllamaEmbedder."""

import pytest
from unittest.mock import MagicMock, patch

from vector_store.embedder import OllamaEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1] * 768  # 768-dimensional fake embedding


@pytest.fixture
def mock_client():
    """Create a mock Ollama client."""
    with patch("vector_store.embedder.ollama.Client") as MockClient:
        client = MockClient.return_value
        client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        client.list.return_value = MagicMock(
            models=[
                MagicMock(model="nomic-embed-text:latest"),
                MagicMock(model="llama3:latest"),
            ]
        )
        yield client


@pytest.fixture
def embedder(mock_client):
    """Create an embedder with mocked Ollama client."""
    return OllamaEmbedder(model="nomic-embed-text")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbed:
    def test_single_text(self, embedder, mock_client):
        result = embedder.embed("Ein Testtext")
        assert len(result) == 768
        assert result == FAKE_EMBEDDING
        mock_client.embed.assert_called_once()

    def test_sets_dimensions(self, embedder, mock_client):
        assert embedder.dimensions is None
        embedder.embed("Test")
        assert embedder.dimensions == 768

    def test_empty_text_raises(self, embedder):
        with pytest.raises(ValueError, match="empty"):
            embedder.embed("")

    def test_whitespace_only_raises(self, embedder):
        with pytest.raises(ValueError, match="empty"):
            embedder.embed("   ")

    def test_connection_error(self, mock_client):
        mock_client.embed.side_effect = ConnectionError("refused")
        embedder = OllamaEmbedder(model="nomic-embed-text")
        with pytest.raises(ConnectionError, match="Ollama"):
            embedder.embed("Test")


class TestEmbedBatch:
    def test_multiple_texts(self, embedder, mock_client):
        mock_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING, FAKE_EMBEDDING, FAKE_EMBEDDING],
        }
        result = embedder.embed_batch(["Text 1", "Text 2", "Text 3"])
        assert len(result) == 3
        assert all(len(v) == 768 for v in result)

    def test_empty_list(self, embedder):
        result = embedder.embed_batch([])
        assert result == []

    def test_handles_empty_strings_in_batch(self, embedder, mock_client):
        mock_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        result = embedder.embed_batch(["", "Text", ""])
        # Only "Text" should be embedded
        assert len(result) == 3
        assert result[1] == FAKE_EMBEDDING
        assert result[0] == []  # Empty string → empty list
        assert result[2] == []

    def test_sets_dimensions(self, embedder, mock_client):
        mock_client.embed.return_value = {
            "embeddings": [FAKE_EMBEDDING],
        }
        assert embedder.dimensions is None
        embedder.embed_batch(["Test"])
        assert embedder.dimensions == 768


class TestHealthCheck:
    def test_healthy(self, embedder, mock_client):
        result = embedder.health_check()
        assert result["healthy"] is True
        assert result["ollama_running"] is True
        assert result["model_available"] is True

    def test_model_not_found(self, mock_client):
        mock_client.list.return_value = MagicMock(
            models=[MagicMock(model="llama3:latest")]
        )
        embedder = OllamaEmbedder(model="nonexistent-model")
        result = embedder.health_check()
        assert result["healthy"] is False
        assert result["ollama_running"] is True
        assert result["model_available"] is False
        assert "not found" in result["error"]

    def test_ollama_not_running(self, mock_client):
        mock_client.list.side_effect = Exception("Connection refused")
        embedder = OllamaEmbedder(model="nomic-embed-text")
        result = embedder.health_check()
        assert result["healthy"] is False
        assert result["ollama_running"] is False
