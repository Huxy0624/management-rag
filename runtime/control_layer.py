from __future__ import annotations

from typing import Any

from runtime.experiment_bridge import get_v1


GENERIC_FILLER_PHRASES = (
    "优化机制",
    "加强管理",
    "提升协同",
    "完善流程",
    "建立机制",
)


def short_tokens(text: str) -> list[str]:
    v1 = get_v1()
    cleaned = v1.clean_snippet(text)
    for prefix in ("固定字段：", "固定议程：", "广播节奏：", "固定对象："):
        cleaned = cleaned.replace(prefix, "")
    parts = [item.strip() for item in cleaned.replace("+", "、").replace("，", "、").replace("：", "、").split("、")]
    tokens = [item for item in parts if item and len(item) >= 2]
    return tokens[:4]


def step_signal_count(answer: str, step: dict[str, Any]) -> int:
    checks = 0
    deliverable = str(step.get("deliverable", ""))
    if deliverable and deliverable in answer:
        checks += 1
    if any(token in answer for token in short_tokens(str(step.get("object", "")))):
        checks += 1
    if any(token in answer for token in short_tokens(str(step.get("action", "")))):
        checks += 1
    return checks


def is_mechanism_building_row(row: dict[str, Any]) -> bool:
    router = row.get("router_decision", {})
    planner_output = row.get("planner_output_v21", {})
    return router.get("subtype") == "mechanism_building" or planner_output.get("solution_mode") == "mechanism_building"


def missing_mechanism_names(row: dict[str, Any], answer: str) -> list[str]:
    if not is_mechanism_building_row(row):
        return []
    entities = row.get("planner_output_v21", {}).get("mechanism_entities", [])
    return [str(entity.get("name", "")) for entity in entities if str(entity.get("name", "")) and str(entity.get("name", "")) not in answer]


def action_steps_match_count(row: dict[str, Any], answer: str) -> int | None:
    if str(row.get("query_type", "")) != "how":
        return None
    steps = row.get("planner_output_v21", {}).get("action_steps", [])
    return sum(1 for step in steps if step_signal_count(answer, step) >= 2)


def contains_generic_filler(answer: str) -> bool:
    return any(phrase in answer for phrase in GENERIC_FILLER_PHRASES)


def build_control_checks(row: dict[str, Any], answer: str) -> dict[str, Any]:
    missing_names = missing_mechanism_names(row, answer)
    mechanism_pass = not missing_names
    match_count = action_steps_match_count(row, answer)
    expected_step_count = len(row.get("planner_output_v21", {}).get("action_steps", [])) if str(row.get("query_type", "")) == "how" else None
    structure_pass = match_count == expected_step_count if expected_step_count is not None else True
    return {
        "mechanism_name_check_pass": mechanism_pass,
        "missing_mechanism_names": missing_names,
        "action_steps_match_count": match_count,
        "expected_action_steps_count": expected_step_count,
        "structure_check_pass": structure_pass,
        "contains_generic_filler": contains_generic_filler(answer),
    }
