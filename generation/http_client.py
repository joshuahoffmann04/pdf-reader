import json
from typing import Any
from urllib import request


def post_json(url: str, payload: dict, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)
