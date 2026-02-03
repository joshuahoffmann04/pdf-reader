"""
Token Counter for the Chunking Pipeline

Uses tiktoken with the cl100k_base encoding as a conservative approximation
for Ollama models (LLaMA, Mistral, etc.). cl100k_base tends to produce
slightly higher token counts than SentencePiece-based tokenizers, which
provides a safe margin when chunk size is a hard limit.

Usage:
    from chunking.token_counter import count_tokens, count_tokens_batch

    n = count_tokens("Dies ist ein Beispielsatz.")
    counts = count_tokens_batch(["Satz eins.", "Satz zwei."])
"""

import tiktoken

# Singleton encoder - initialized once, reused across calls.
# cl100k_base is used by GPT-4 / GPT-3.5-turbo and is a reasonable
# approximation for BPE-based tokenizers used in LLaMA/Mistral models.
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Get or initialize the tiktoken encoder (singleton)."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to tokenize.

    Returns:
        Number of tokens.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def count_tokens_batch(texts: list[str]) -> list[int]:
    """
    Count tokens for a list of texts.

    Args:
        texts: List of text strings.

    Returns:
        List of token counts, one per input text.
    """
    encoder = _get_encoder()
    return [len(encoder.encode(t)) if t else 0 for t in texts]
