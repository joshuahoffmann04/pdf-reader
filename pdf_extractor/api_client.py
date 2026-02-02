"""
OpenAI API Client for Vision-based PDF Extraction.

This module provides a robust API client with:
- Retry logic with exponential backoff
- Token usage tracking
- Response validation and parsing
- Error handling with custom exceptions

Designed for GPT-4o Vision API calls with base64-encoded images.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from openai import OpenAI, APIError as OpenAIAPIError, RateLimitError, APIConnectionError as OpenAIConnectionError

from .exceptions import (
    APIError,
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
)
from .pdf_utils import PageImage


logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class APIResponse:
    """
    Parsed response from an API call.

    Attributes:
        content: Parsed JSON content (dict or list)
        raw_content: Raw text content from the API
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        model: Model used for the request
        finish_reason: Why the model stopped (stop, length, etc.)
    """

    content: Union[dict, list]
    raw_content: str
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class TokenUsage:
    """
    Cumulative token usage tracker.

    Tracks total tokens used across multiple API calls.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

    def add(self, response: APIResponse) -> None:
        """Add tokens from a response."""
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.request_count += 1

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens

    def estimate_cost(self, model: str = "gpt-4o") -> float:
        """
        Estimate cost in USD based on token usage.

        Args:
            model: Model name for pricing

        Returns:
            Estimated cost in USD
        """
        pricing = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        }
        prices = pricing.get(model, pricing["gpt-4o"])
        return (self.input_tokens * prices["input"]) + (self.output_tokens * prices["output"])


# =============================================================================
# API CLIENT
# =============================================================================


class VisionAPIClient:
    """
    OpenAI Vision API client optimized for PDF extraction.

    Features:
    - Automatic retry with exponential backoff
    - JSON response parsing with validation
    - Token usage tracking
    - Detailed error handling

    Usage:
        client = VisionAPIClient()

        # Single image request
        response = client.call(
            system_prompt="Analyze this document...",
            user_prompt="Extract the structure...",
            images=[page_image],
        )
        print(response.content)  # Parsed JSON

        # Check usage
        print(f"Total tokens: {client.usage.total_tokens}")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Vision API client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (gpt-4o or gpt-4o-mini)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature (0 = deterministic)
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Token usage tracking
        self.usage = TokenUsage()

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[PageImage],
        max_tokens: Optional[int] = None,
    ) -> APIResponse:
        """
        Make a Vision API call with images.

        Args:
            system_prompt: System message content
            user_prompt: User message content
            images: List of PageImage objects to include

        Returns:
            APIResponse with parsed content

        Raises:
            APIConnectionError: Cannot connect to API
            APIRateLimitError: Rate limit exceeded
            APIResponseError: Invalid or refused response
        """
        # Build messages
        messages = self._build_messages(system_prompt, user_prompt, images)

        # Make request with retry
        response = self._call_with_retry(messages, max_tokens or self.max_tokens)

        # Parse and validate response
        return self._parse_response(response)

    def call_text_only(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> APIResponse:
        """
        Make a text-only API call (no images).

        Useful for follow-up questions or text processing.

        Args:
            system_prompt: System message content
            user_prompt: User message content

        Returns:
            APIResponse with parsed content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self._call_with_retry(messages, max_tokens or self.max_tokens)
        return self._parse_response(response)

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[PageImage],
    ) -> list[dict]:
        """Build the messages array for the API call."""
        # Build content array with text and images
        content = [{"type": "text", "text": user_prompt}]

        # Add images
        for img in images:
            content.append(img.to_api_format())

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _call_with_retry(
        self,
        messages: list[dict],
        max_tokens: int,
    ) -> Any:
        """
        Make API call with retry logic.

        Uses exponential backoff for retries.
        """
        last_error: Optional[Exception] = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"API call attempt {attempt + 1}/{self.max_retries}")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                return response

            except RateLimitError as e:
                logger.warning(f"Rate limit hit, waiting {delay}s...")
                last_error = e
                time.sleep(delay)
                delay *= 2  # Exponential backoff

            except OpenAIConnectionError as e:
                logger.warning(f"Connection error, retrying in {delay}s...")
                last_error = e
                time.sleep(delay)
                delay *= 2

            except OpenAIAPIError as e:
                # Check if it's a retryable error (5xx)
                if hasattr(e, "status_code") and e.status_code >= 500:
                    logger.warning(f"Server error ({e.status_code}), retrying...")
                    last_error = e
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise APIError(str(e), e, getattr(e, "status_code", None))

        # All retries exhausted
        if isinstance(last_error, RateLimitError):
            raise APIRateLimitError(original_error=last_error)
        elif isinstance(last_error, OpenAIConnectionError):
            raise APIConnectionError(original_error=last_error)
        else:
            raise APIError("Max retries exceeded", last_error)

    def _parse_response(self, response: Any) -> APIResponse:
        """
        Parse and validate the API response.

        Extracts JSON from the response content, handling various formats.
        """
        choice = response.choices[0]
        raw_content = choice.message.content or ""
        finish_reason = choice.finish_reason

        # Extract usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Check for content policy/refusal
        if finish_reason == "content_filter":
            raise APIResponseError(
                "Response blocked by content filter",
                raw_content,
            )

        if not raw_content.strip():
            raise APIResponseError("Empty response from API", raw_content)

        # Check for refusal patterns
        refusal_patterns = [
            r"I(?:'m| am) (?:sorry|unable|not able)",
            r"cannot (?:help|assist|provide)",
            r"(?:don't|do not) have (?:access|the ability)",
            r"as an AI",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, raw_content, re.IGNORECASE):
                raise APIResponseError(
                    "Model refused to process request",
                    raw_content,
                )

        # Parse JSON
        parsed = self._extract_json(raw_content)

        api_response = APIResponse(
            content=parsed,
            raw_content=raw_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=response.model,
            finish_reason=finish_reason,
        )

        # Track usage
        self.usage.add(api_response)

        return api_response

    def _extract_json(self, text: str) -> Union[dict, list]:
        """
        Extract JSON from text, handling code blocks and extra text.

        Tries multiple strategies to find valid JSON.
        """
        # Strategy 1: Try parsing the whole text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from code block
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object/array boundaries
        # Find first { or [
        start_obj = text.find("{")
        start_arr = text.find("[")

        if start_obj == -1 and start_arr == -1:
            raise APIResponseError(
                "No JSON found in response",
                text,
            )

        # Determine start position and type
        if start_arr == -1 or (start_obj != -1 and start_obj < start_arr):
            start = start_obj
            end_char = "}"
        else:
            start = start_arr
            end_char = "]"

        # Find matching closing bracket
        depth = 0
        end = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            raise APIResponseError(
                "Malformed JSON in response",
                text,
            )

        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise APIResponseError(
                f"Invalid JSON: {e}",
                json_str,
            )

    def reset_usage(self) -> None:
        """Reset token usage counters."""
        self.usage = TokenUsage()

    def get_usage_summary(self) -> dict:
        """Get a summary of token usage."""
        return {
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "request_count": self.usage.request_count,
            "estimated_cost_usd": round(self.usage.estimate_cost(self.model), 4),
        }
