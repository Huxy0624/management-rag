from __future__ import annotations

from db_utils import add_demo_request_log, list_recent_demo_requests


def save_request_log(
    request_id: str,
    question: str,
    success: bool,
    session_id: str | None = None,
    selected_from: str | None = None,
    fallback_triggered: bool = False,
    total_latency_ms: int | None = None,
    error_message: str | None = None,
    client_ip: str | None = None,
    user_mode: str = "user",
) -> str:
    return add_demo_request_log(
        request_id=request_id,
        question=question,
        success=success,
        session_id=session_id,
        selected_from=selected_from,
        fallback_triggered=fallback_triggered,
        total_latency_ms=total_latency_ms,
        error_message=error_message,
        client_ip=client_ip,
        user_mode=user_mode,
    )


def get_recent_requests(limit: int = 50) -> list[dict]:
    return list_recent_demo_requests(limit=limit)
