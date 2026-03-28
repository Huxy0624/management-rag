from __future__ import annotations

from typing import Any

from runtime.experiment_bridge import get_v1, get_v21


def infer_query_type(query: str) -> str:
    v1 = get_v1()
    normalized = v1.normalize_text(query)
    if any(token in normalized for token in ("为什么", "为何", "怎么会", "为何会")):
        return "why"
    if any(token in normalized for token in ("怎么", "如何", "怎样", "应该怎么", "该怎么")):
        return "how"
    return "what"


def route_query(query: str) -> dict[str, Any]:
    query_type = infer_query_type(query)
    v21 = get_v21()
    return v21.router_decision_v21(query, query_type)
