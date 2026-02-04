from __future__ import annotations

from typing import Any

from .http_client import post_json


def chat(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    temperature: float = 0.2,
    output_tokens: int = 512,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": response_schema,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": output_tokens,
        },
    }
    url = f"{base_url.rstrip('/')}/api/chat"
    response = post_json(url, payload, timeout=120)
    message = response.get("message", {})
    return message.get("content", "")
