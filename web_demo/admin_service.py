from __future__ import annotations

from web_demo.request_store import get_recent_requests


def load_recent_requests(limit: int = 50) -> list[dict]:
    return get_recent_requests(limit=limit)
