#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Any

import exp_validate_retrieval_v4 as base


base.DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v4_phase22/tagged_chunks/records.json")
base.DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
base.DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v4_phase22/retrieval_eval_v4/report.json")

_BASE_MATRIX_ADJUSTMENT = base.matrix_adjustment


def parse_query_profile(query_text: str) -> dict[str, Any]:
    normalized = base.normalize_text(query_text)
    profile = {
        "query_type": base.query_type(query_text),
        "preferred_answer_roles": [],
        "avoid_answer_roles": [],
        "preferred_root_issues": [],
        "preferred_intents": [],
        "preferred_target_roles": [],
        "preferred_role_profiles": [],
    }

    if any(token in normalized for token in ("什么是", "定义", "本质")):
        profile["preferred_answer_roles"] = ["definition", "principle", "summary"]
        profile["avoid_answer_roles"] = ["solution", "mechanism", "example"]
    elif any(token in normalized for token in ("为什么", "为何")):
        profile["preferred_answer_roles"] = ["mechanism", "cause", "principle", "comparison", "warning"]
        profile["avoid_answer_roles"] = ["definition", "summary", "solution"]
    elif any(token in normalized for token in ("怎么", "如何")):
        profile["preferred_answer_roles"] = ["solution", "mechanism", "example"]
        profile["avoid_answer_roles"] = ["definition", "summary"]

    if any(token in normalized for token in ("信息不通", "信息失真", "汇报", "传导", "压缩", "扩散")):
        profile["preferred_root_issues"].append("information_distortion")
    if any(token in normalized for token in ("评价失效", "评价不公", "评价标准", "奖惩", "激励")):
        profile["preferred_root_issues"].append("evaluation_failure")

    if any(token in normalized for token in ("跨部门", "协作", "部门墙", "资源协调")):
        profile["preferred_intents"].append("resource_coordination")
    if any(token in normalized for token in ("机制", "制度", "流程", "规则", "长期", "体系", "系统性")):
        profile["preferred_intents"].append("mechanism_design")
    if any(token in normalized for token in ("带人", "培养", "辅导")):
        profile["preferred_intents"].append("coaching")
    if any(token in normalized for token in ("选拔", "候选人", "任命")):
        profile["preferred_intents"].append("selection")
    if any(token in normalized for token in ("接班", "梯队", "后备")):
        profile["preferred_intents"].append("succession")

    if any(token in normalized for token in ("老板", "高管", "CEO")):
        profile["preferred_target_roles"].append("executive")
    if any(token in normalized for token in ("总监", "部门负责人")):
        profile["preferred_target_roles"].append("director")
    if any(token in normalized for token in ("经理", "leader", "项目负责人")):
        profile["preferred_target_roles"].append("manager")
    if any(token in normalized for token in ("员工", "一线")):
        profile["preferred_target_roles"].append("employee")

    if any(token in normalized for token in ("滚刀肉", "出工不出力")):
        profile["preferred_role_profiles"].append("slippery_worker")
    if any(token in normalized for token in ("挑战", "不服")):
        profile["preferred_role_profiles"].append("challenger")

    for key, value in profile.items():
        if isinstance(value, list):
            deduped: list[str] = []
            for item in value:
                if item not in deduped:
                    deduped.append(item)
            profile[key] = deduped
    return profile


def role_anchor_adjustment(record: dict[str, Any], profile: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    tags = dict(record.get("governance_tags_v4", {}))
    answer_role = str(tags.get("answer_role", "none"))
    intent = str(tags.get("intent", "none"))
    query_type = str(profile.get("query_type", "what"))

    delta = 0.0
    reasons: list[str] = []

    query_type_weights = {
        "what": {"definition": 0.55, "principle": 0.4, "summary": 0.15, "mechanism": -0.25, "solution": -0.45},
        "why": {"mechanism": 0.55, "cause": 0.35, "principle": 0.22, "definition": -0.25, "solution": -0.45},
        "how": {"solution": 0.55, "mechanism": 0.28, "example": 0.15, "definition": -0.25, "principle": -0.18},
    }
    if answer_role in query_type_weights.get(query_type, {}):
        contribution = float(query_type_weights[query_type][answer_role])
        delta += contribution
        reasons.append(f"query_type:{query_type}:{answer_role}:{contribution:+.2f}")

    if intent == "mechanism_design":
        if answer_role == "mechanism":
            delta += 0.45
            reasons.append("intent_anchor:mechanism_design->mechanism:+0.45")
        elif answer_role in {"principle", "definition"}:
            delta += 0.12
            reasons.append("intent_anchor:mechanism_design->definition_or_principle:+0.12")
        elif answer_role == "solution":
            delta -= 0.35
            reasons.append("intent_anchor:mechanism_design->solution:-0.35")
    elif intent == "coaching":
        if answer_role == "solution":
            delta += 0.45
            reasons.append("intent_anchor:coaching->solution:+0.45")
        elif answer_role == "example":
            delta += 0.2
            reasons.append("intent_anchor:coaching->example:+0.20")
        elif answer_role in {"definition", "principle"}:
            delta -= 0.2
            reasons.append("intent_anchor:coaching->definition_or_principle:-0.20")

    if query_type == "what" and intent == "mechanism_design" and answer_role == "solution":
        delta -= 0.18
        reasons.append("cross_guard:what+mechanism_design+solution:-0.18")
    if query_type == "why" and intent == "mechanism_design" and answer_role == "solution":
        delta -= 0.12
        reasons.append("cross_guard:why+mechanism_design+solution:-0.12")
    if query_type == "how" and intent == "mechanism_design" and answer_role == "mechanism":
        delta += 0.08
        reasons.append("cross_guard:how+mechanism_design+mechanism:+0.08")

    clamped = max(-0.9, min(0.9, delta))
    return clamped, {
        "field_value": {"answer_role": answer_role, "intent": intent, "query_type": query_type},
        "decision": "soft_alignment" if reasons else "neutral",
        "delta": round(clamped, 4),
        "raw_delta": round(delta, 4),
        "reasons": reasons,
    }


def matrix_adjustment(record: dict[str, Any], profile: dict[str, Any], primary_score: float) -> tuple[float, float, dict[str, Any]]:
    base_raw_delta, _, components = _BASE_MATRIX_ADJUSTMENT(record, profile, primary_score)
    anchor_delta, anchor_info = role_anchor_adjustment(record, profile)
    raw_delta = base_raw_delta + anchor_delta

    lower_bound = -0.08 * primary_score
    upper_bound = 0.12 * primary_score
    clamped = max(lower_bound, min(upper_bound, raw_delta))

    components["role_anchor"] = anchor_info
    components["clamp"] = {
        "raw_delta": round(raw_delta, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "clamped_delta": round(clamped, 4),
    }
    return raw_delta, clamped, components


base.parse_query_profile = parse_query_profile
base.matrix_adjustment = matrix_adjustment


if __name__ == "__main__":
    base.main()
