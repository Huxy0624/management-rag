from __future__ import annotations

from db_utils import add_demo_feedback


def save_feedback(
    request_id: str,
    rating: str,
    comment: str | None = None,
    session_id: str | None = None,
    client_ip: str | None = None,
) -> str:
    return add_demo_feedback(
        request_id=request_id,
        rating=rating,
        comment=comment,
        session_id=session_id,
        client_ip=client_ip,
    )
