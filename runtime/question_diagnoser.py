from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime.experiment_bridge import get_v1
from runtime.llm_surface_runtime import load_prompt, request_llm_answer
from runtime.runtime_config import SurfaceRuntimeConfig


DIAGNOSIS_PROMPT_PATH = Path("prompts/diagnosis/question_diagnosis_prompt.txt")


DEFAULT_DIAGNOSIS = {
    "question_type": "principle_explanation",
    "primary_intent": "definition",
    "user_role": "unknown",
    "role_confidence": 0.2,
    "task_nature": "mixed",
    "time_horizon": "current_case",
    "response_mode": "human_guidance_first",
    "needs_clarification": False,
    "clarification_question": "",
    "root_cause_mode": "not_applicable",
    "rewrite_for_retrieval": [],
    "rewrite_for_reasoning": [],
    "planner_focus": [],
}


def _normalized(query: str) -> str:
    v1 = get_v1()
    return v1.normalize_text(query)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _infer_user_role(text: str) -> tuple[str, float]:
    if _contains_any(text, ("ceo", "老板", "总经理", "高管", "总监")):
        return "top_leader", 0.92
    if _contains_any(text, ("经理", "主管", "负责人", "带团队", "管理者")):
        return "manager", 0.84
    if _contains_any(text, ("项目经理", "协调", "协同", "pm", "副手", "接口人", "副班长", "班长副手")):
        return "deputy_or_coordinator", 0.76
    if _contains_any(text, ("前端", "后端", "开发", "工程师", "员工", "个人贡献者")):
        return "individual_contributor", 0.74
    return "unknown", 0.28


def _infer_question_type(text: str) -> str:
    if _contains_any(text, ("谁该负责", "责任", "谁的问题", "谁应该拍板", "谁来拍板", "谁说了算", "谁该拍板")):
        return "responsibility_boundary"
    if _contains_any(text, ("为什么", "为何", "根因", "本质", "怎么判断", "谁导致", "谁导致的")):
        return "root_cause_judgment"
    if _contains_any(text, ("机制", "流程", "制度", "规范", "体系")):
        return "mechanism_design"
    if _contains_any(text, ("汇报", "向上", "给老板", "进度说明")):
        return "upward_reporting"
    if _contains_any(text, ("如果我是", "作为经理", "作为主管", "应该怎么带", "怎么执行", "具体该做什么")):
        return "role_execution"
    if _contains_any(text, ("怎么做", "怎么办", "如何处理", "如何推进", "立刻", "怎么收口", "如何收口", "先把事做成", "怎么处理", "临时推进")):
        return "immediate_handling"
    return "principle_explanation"


def _infer_primary_intent(question_type: str) -> str:
    mapping = {
        "root_cause_judgment": "diagnosis",
        "responsibility_boundary": "diagnosis",
        "immediate_handling": "action",
        "mechanism_design": "mechanism",
        "upward_reporting": "action",
        "role_execution": "action",
        "principle_explanation": "definition",
    }
    return mapping.get(question_type, "definition")


def _infer_task_nature(primary_intent: str, text: str) -> str:
    if primary_intent == "action":
        return "deterministic"
    if primary_intent == "diagnosis":
        return "uncertain"
    if _contains_any(text, ("冲突", "争执", "歧义", "责任")):
        return "mixed"
    return "mixed"


def _infer_time_horizon(text: str, question_type: str) -> str:
    if _contains_any(text, ("长期", "机制", "制度", "体系", "长期治理")) or question_type == "mechanism_design":
        return "long_term_governance"
    if _contains_any(text, ("经常", "总是", "反复", "一再", "老是")):
        return "repeat_issue"
    return "current_case"


def _infer_response_mode(question_type: str, primary_intent: str) -> str:
    if question_type in {"root_cause_judgment", "responsibility_boundary", "principle_explanation"}:
        return "human_guidance_first"
    if question_type in {"mechanism_design", "upward_reporting", "role_execution", "immediate_handling"}:
        return "sop_first"
    if primary_intent == "diagnosis":
        return "human_guidance_first"
    return "mixed"


def _infer_root_cause_mode(text: str, question_type: str) -> str:
    if question_type not in {"root_cause_judgment", "responsibility_boundary"} and not _contains_any(
        text, ("冲突", "误解", "责任", "为什么", "根因")
    ):
        return "not_applicable"
    if _contains_any(text, ("信息不通", "文档", "接口", "歧义", "理解偏差", "信息失真")) and _contains_any(
        text, ("评价", "考核", "责任边界", "谁负责", "甩锅", "奖惩")
    ):
        return "mixed_root_causes"
    if _contains_any(text, ("信息不通", "文档", "接口", "歧义", "理解偏差", "信息失真")):
        return "information_distortion"
    if _contains_any(text, ("评价", "考核", "责任边界", "谁负责", "甩锅", "奖惩")):
        return "evaluation_failure"
    return "mixed_root_causes"


def _needs_clarification(role: str, role_confidence: float, question_type: str) -> tuple[bool, str]:
    if role_confidence >= 0.45:
        return False, ""
    if question_type in {"upward_reporting", "role_execution", "immediate_handling"}:
        return True, "你当前更接近一线执行者、协调者，还是团队负责人？"
    return False, ""


def _extract_scene_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in ("前端", "后端", "接口文档", "歧义", "具体写法", "争执升级", "评审", "反馈权限", "上下游", "交付物"):
        if token in query and token not in terms:
            terms.append(token)
    return terms


def _build_retrieval_rewrites(query: str, question_type: str, root_cause_mode: str) -> list[str]:
    scene_terms = _extract_scene_terms(query)
    scene = " ".join(scene_terms)
    rewrites = [query]
    if question_type in {"root_cause_judgment", "responsibility_boundary"}:
        root_hint = {
            "information_distortion": "信息失真 根因 责任边界",
            "evaluation_failure": "评价失效 责任边界 考核激励",
            "mixed_root_causes": "信息失真 评价失效 主因 次因",
        }.get(root_cause_mode, "信息失真 评价失效")
        rewrites.append(f"{scene} {root_hint} 谁导致 冲突升级".strip())
    elif question_type == "mechanism_design":
        rewrites.append(f"{scene} 机制 流程 制度 规范 体系".strip())
    elif question_type in {"upward_reporting", "role_execution", "immediate_handling"}:
        rewrites.append(f"{scene} 管理动作 做法 输出 责任人".strip())
    else:
        rewrites.append(f"{scene} 本质 原理 管理问题".strip())
    return [item for item in rewrites if item]


def _build_reasoning_rewrites(query: str, question_type: str, root_cause_mode: str) -> list[str]:
    scene_terms = "、".join(_extract_scene_terms(query))
    if question_type == "root_cause_judgment":
        return [
            f"先判断这是信息失真、评价失效还是两者兼有；保留场景细节：{scene_terms or query}",
            "先回答为什么会这样，再区分主因和次因，最后再给建议",
        ]
    if question_type == "responsibility_boundary":
        return [
            f"先区分导火索、冲突升级者、系统责任；保留角色冲突细节：{scene_terms or query}",
            f"优先判断责任边界与根因模式：{root_cause_mode}",
        ]
    if question_type == "mechanism_design":
        return ["先判断现有问题属于哪类管理结构缺陷，再给机制化修复方案"]
    if question_type in {"upward_reporting", "immediate_handling", "role_execution"}:
        return ["保留原场景与角色，给可执行动作，但不要脱离原问题语义"]
    return ["先定义问题，再解释边界和本质"]


def _build_planner_focus(question_type: str, root_cause_mode: str) -> list[str]:
    if question_type in {"root_cause_judgment", "responsibility_boundary"}:
        focus = ["先定性", "再归因", "主因/次因", "系统责任", "最后建议"]
        if root_cause_mode == "information_distortion":
            focus.append("信息失真")
        elif root_cause_mode == "evaluation_failure":
            focus.append("评价失效")
        elif root_cause_mode == "mixed_root_causes":
            focus.append("信息失真+评价失效")
        return focus
    if question_type == "mechanism_design":
        return ["机制实体", "责任人", "节奏", "字段", "长期治理"]
    return ["动作对象", "动作行为", "输出结果"]


def fallback_diagnosis(original_question: str) -> dict[str, Any]:
    text = _normalized(original_question)
    question_type = _infer_question_type(text)
    primary_intent = _infer_primary_intent(question_type)
    user_role, role_confidence = _infer_user_role(text)
    task_nature = _infer_task_nature(primary_intent, text)
    time_horizon = _infer_time_horizon(text, question_type)
    response_mode = _infer_response_mode(question_type, primary_intent)
    root_cause_mode = _infer_root_cause_mode(text, question_type)
    needs_clarification, clarification_question = _needs_clarification(user_role, role_confidence, question_type)
    return {
        "question_type": question_type,
        "primary_intent": primary_intent,
        "user_role": user_role,
        "role_confidence": round(role_confidence, 2),
        "task_nature": task_nature,
        "time_horizon": time_horizon,
        "response_mode": response_mode,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "root_cause_mode": root_cause_mode,
        "rewrite_for_retrieval": _build_retrieval_rewrites(original_question, question_type, root_cause_mode),
        "rewrite_for_reasoning": _build_reasoning_rewrites(original_question, question_type, root_cause_mode),
        "planner_focus": _build_planner_focus(question_type, root_cause_mode),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in diagnosis response.")
    return json.loads(cleaned[start : end + 1])


def llm_diagnosis(original_question: str, config: SurfaceRuntimeConfig) -> dict[str, Any]:
    prompt_text = load_prompt(DIAGNOSIS_PROMPT_PATH)
    user_text = json.dumps({"original_question": original_question}, ensure_ascii=False, indent=2)
    answer, _, _ = request_llm_answer(prompt_text, user_text, config=config, temperature=0.0)
    parsed = _extract_json_object(answer)
    merged = dict(DEFAULT_DIAGNOSIS)
    merged.update(parsed)
    return merged


def diagnosis_to_query_type(diagnosis: dict[str, Any]) -> str:
    question_type = str(diagnosis.get("question_type", "principle_explanation"))
    if question_type in {"root_cause_judgment", "responsibility_boundary"}:
        return "why"
    if question_type in {"immediate_handling", "mechanism_design", "upward_reporting", "role_execution"}:
        return "how"
    return "what"


def diagnose_question(original_question: str, config: SurfaceRuntimeConfig | None = None) -> dict[str, Any]:
    fallback = fallback_diagnosis(original_question)
    if config is None or not config.api_key:
        return fallback
    try:
        llm_result = llm_diagnosis(original_question, config)
    except Exception:
        return fallback

    merged = dict(DEFAULT_DIAGNOSIS)
    merged.update(fallback)
    merged.update(llm_result)
    if not merged.get("rewrite_for_retrieval"):
        merged["rewrite_for_retrieval"] = fallback["rewrite_for_retrieval"]
    if not merged.get("rewrite_for_reasoning"):
        merged["rewrite_for_reasoning"] = fallback["rewrite_for_reasoning"]
    if not merged.get("planner_focus"):
        merged["planner_focus"] = fallback["planner_focus"]
    return merged
