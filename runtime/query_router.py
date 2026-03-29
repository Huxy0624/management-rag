from __future__ import annotations

from typing import Any

from runtime.experiment_bridge import get_v1, get_v21
from runtime.question_diagnoser import diagnosis_to_query_type


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


def route_query_with_diagnosis(query: str, diagnosis_result: dict[str, Any]) -> dict[str, Any]:
    query_type = diagnosis_to_query_type(diagnosis_result)
    v21 = get_v21()
    routed = v21.router_decision_v21(query, query_type)
    routed["question_diagnosis"] = diagnosis_result
    routed["question_type"] = diagnosis_result.get("question_type")
    routed["primary_intent"] = diagnosis_result.get("primary_intent")
    routed["response_mode"] = diagnosis_result.get("response_mode")
    routed["root_cause_mode"] = diagnosis_result.get("root_cause_mode")
    return routed
