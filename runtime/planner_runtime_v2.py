from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime.llm_surface_runtime import load_prompt, request_llm_answer
from runtime.runtime_config import SurfaceRuntimeConfig


PLANNER_PROMPT_PATH = Path("prompts/planner/planner_prompt_v2.txt")

ROOT_CAUSE_ENUMS = {"information_distortion", "evaluation_failure", "mixed_root_causes", "not_applicable"}
ADVICE_MODES = {"current_case_first", "coaching_first", "mechanism_first", "mixed"}
ROLE_DEFAULT = "unknown"


PRINCIPLE_ROLE_ACTIONS: dict[str, dict[str, str]] = {
    "信息通道建设": {
        "top_leader": "建立跨部门治理节奏和固定升级节点，让关键分歧能按节点升级而不是线下扯皮。",
        "manager": "编制一页纸协同单，明确协同对象、目标、交付物和反馈方式，并发给相关负责人确认。",
        "deputy_or_coordinator": "负责版本同步、问题归并、异常上卷，确保上下游拿到的是同一版口径。",
        "individual_contributor": "反馈问题时写清背景、影响和建议，围绕事实补充信息，不越位拍板。",
        ROLE_DEFAULT: "先把信息背景、影响和待确认点整理清楚，再进入协同动作。",
    },
    "评价体系建设": {
        "top_leader": "明确跨团队争议的裁决机制，避免问题长期停留在平级争执。",
        "manager": "明确建议权、验收权和拍板权，避免执行层用争论替代决策。",
        "deputy_or_coordinator": "整理争议点并推入决策节点，确保分歧进入有结论的裁决流程。",
        "individual_contributor": "区分提建议和做最终评判，先提交证据和影响，不自行定义责任归属。",
        ROLE_DEFAULT: "先分清谁提供信息、谁提出建议、谁做最终裁决。",
    },
    "授权与监督": {
        "top_leader": "授权负责人，只看关键节点和偏差，避免自己下场接管所有细节。",
        "manager": "拆责任、定节奏、看偏差，对结果负责但不替代每个人做判断。",
        "deputy_or_coordinator": "充当精力延伸与问题过滤器，把杂音过滤后再升级给负责人。",
        "individual_contributor": "围绕交付物反馈，不替代管理动作，也不越级定义处置方案。",
        ROLE_DEFAULT: "明确谁拥有决策权，谁负责监督偏差，谁只负责反馈事实。",
    },
}

PRINCIPLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "信息通道建设": ("信息", "对齐", "同步", "文档", "接口", "歧义", "误解", "广播", "沟通"),
    "评价体系建设": ("责任", "考核", "评价", "裁决", "拍板", "验收", "边界", "归属"),
    "授权与监督": ("授权", "监督", "负责人", "升级", "偏差", "节奏", "节点"),
}


DEFAULT_PLANNER_OUTPUT = {
    "answer_goal": "",
    "reasoning_order": [],
    "advice_mode": "mixed",
    "root_cause_primary": "not_applicable",
    "root_cause_secondary": "not_applicable",
    "action_translation": [],
    "answer_outline": [],
    "answer_guardrails": [],
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
        raise ValueError("No JSON object found in planner response.")
    return json.loads(cleaned[start : end + 1])


def _preview_chunk(chunk: dict[str, Any], limit: int = 280) -> dict[str, Any]:
    metadata = dict(chunk.get("metadata", {}))
    document = " ".join(str(chunk.get("document", "")).split())
    return {
        "source": metadata.get("source"),
        "title": metadata.get("title"),
        "chunk_id": metadata.get("chunk_id"),
        "answer_role": metadata.get("answer_role"),
        "intent": metadata.get("intent"),
        "root_issue": metadata.get("root_issue"),
        "preview": document[:limit] + ("..." if len(document) > limit else ""),
    }


def summarize_retrieved_chunks(chunks: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    return [_preview_chunk(chunk) for chunk in chunks[:limit]]


def _infer_answer_goal(diagnosis_result: dict[str, Any]) -> str:
    question_type = str(diagnosis_result.get("question_type", ""))
    if question_type == "root_cause_judgment":
        return "先判断问题主因，再补充次因和管理建议"
    if question_type == "responsibility_boundary":
        return "先划清导火索、升级责任和系统责任边界"
    if question_type == "mechanism_design":
        return "把原则转成机制化治理动作"
    if question_type in {"upward_reporting", "role_execution", "immediate_handling"}:
        return "把抽象原则转成与当前角色匹配的具体动作"
    return "先解释原理，再给与角色匹配的建议"


def _infer_advice_mode(diagnosis_result: dict[str, Any]) -> str:
    time_horizon = str(diagnosis_result.get("time_horizon", "current_case"))
    task_nature = str(diagnosis_result.get("task_nature", "mixed"))
    response_mode = str(diagnosis_result.get("response_mode", "mixed"))
    question_type = str(diagnosis_result.get("question_type", ""))
    if time_horizon == "long_term_governance" or question_type == "mechanism_design":
        return "mechanism_first"
    if response_mode == "human_guidance_first":
        return "coaching_first"
    if task_nature == "deterministic":
        return "current_case_first"
    return "mixed"


def _normalize_root_cause(value: str) -> str:
    normalized = str(value or "not_applicable")
    return normalized if normalized in ROOT_CAUSE_ENUMS else "not_applicable"


def _infer_root_causes(diagnosis_result: dict[str, Any], retrieved_chunks: list[dict[str, Any]]) -> tuple[str, str]:
    diagnosis_root = _normalize_root_cause(str(diagnosis_result.get("root_cause_mode", "not_applicable")))
    text = " ".join(" ".join(str(chunk.get("document", "")).split()) for chunk in retrieved_chunks[:4])
    info_hits = sum(token in text for token in ("信息", "文档", "接口", "歧义", "理解偏差", "同步"))
    eval_hits = sum(token in text for token in ("责任", "裁决", "拍板", "考核", "验收", "边界"))
    if diagnosis_root == "mixed_root_causes":
        if info_hits >= eval_hits:
            return "information_distortion", "evaluation_failure"
        return "evaluation_failure", "information_distortion"
    if diagnosis_root in {"information_distortion", "evaluation_failure"}:
        secondary = "evaluation_failure" if diagnosis_root == "information_distortion" and eval_hits else "not_applicable"
        if diagnosis_root == "evaluation_failure" and info_hits:
            secondary = "information_distortion"
        return diagnosis_root, secondary
    if info_hits and eval_hits:
        if info_hits >= eval_hits:
            return "information_distortion", "evaluation_failure"
        return "evaluation_failure", "information_distortion"
    if info_hits:
        return "information_distortion", "not_applicable"
    if eval_hits:
        return "evaluation_failure", "not_applicable"
    return "not_applicable", "not_applicable"


def _detect_principles(retrieved_chunks: list[dict[str, Any]], diagnosis_result: dict[str, Any]) -> list[str]:
    text = " ".join(" ".join(str(chunk.get("document", "")).split()) for chunk in retrieved_chunks[:4])
    principles: list[str] = []
    for principle, keywords in PRINCIPLE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            principles.append(principle)
    if not principles:
        root_mode = str(diagnosis_result.get("root_cause_mode", "not_applicable"))
        if root_mode == "information_distortion":
            principles.append("信息通道建设")
        elif root_mode == "evaluation_failure":
            principles.append("评价体系建设")
        elif root_mode == "mixed_root_causes":
            principles.extend(["信息通道建设", "评价体系建设"])
    if str(diagnosis_result.get("question_type")) == "mechanism_design" and "授权与监督" not in principles:
        principles.append("授权与监督")
    return principles[:3] or ["信息通道建设"]


def _translated_action(principle: str, user_role: str) -> str:
    role_actions = PRINCIPLE_ROLE_ACTIONS.get(principle, {})
    return role_actions.get(user_role) or role_actions.get(ROLE_DEFAULT, "")


def _build_action_translation(diagnosis_result: dict[str, Any], retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    user_role = str(diagnosis_result.get("user_role", ROLE_DEFAULT))
    translations: list[dict[str, str]] = []
    for principle in _detect_principles(retrieved_chunks, diagnosis_result):
        translations.append(
            {
                "principle": principle,
                "role": user_role,
                "translated_action": _translated_action(principle, user_role),
            }
        )
    return translations


def _build_reasoning_order(
    diagnosis_result: dict[str, Any],
    advice_mode: str,
    root_primary: str,
    root_secondary: str,
) -> list[str]:
    question_type = str(diagnosis_result.get("question_type", ""))
    if question_type in {"root_cause_judgment", "responsibility_boundary"}:
        order = [
            "先定义当前问题到底是在问根因还是责任边界",
            f"先判断主因更偏 {root_primary}",
        ]
        if root_secondary != "not_applicable":
            order.append(f"补充次因更偏 {root_secondary}")
        order.append("区分个体责任、升级责任和系统责任")
        order.append("最后再给与角色匹配的收尾建议")
        return order
    if advice_mode == "mechanism_first":
        return ["先说明当前问题为何会反复发生", "再给机制化治理动作", "最后补当前阶段的落地节奏"]
    if advice_mode == "current_case_first":
        return ["先说当前个案怎么处理", "再说与角色匹配的动作", "最后补如何避免重复发生"]
    if advice_mode == "coaching_first":
        return ["先定性当前问题", "再解释判断依据", "最后给经验型辅导动作"]
    return ["先定性问题", "再给角色匹配动作", "最后补治理建议"]


def _build_answer_outline(
    diagnosis_result: dict[str, Any],
    translations: list[dict[str, str]],
    root_primary: str,
    root_secondary: str,
) -> list[str]:
    question_type = str(diagnosis_result.get("question_type", ""))
    outline: list[str] = []
    if question_type in {"root_cause_judgment", "responsibility_boundary"}:
        outline.append(f"先回答主判断：当前更偏 {root_primary}")
        if root_secondary != "not_applicable":
            outline.append(f"补充次因：同时也有 {root_secondary} 成分")
        outline.append("区分导火索、升级责任和系统责任边界")
    else:
        outline.append("先回答当前最应该处理的目标")
    outline.extend(item["translated_action"] for item in translations[:2] if item.get("translated_action"))
    outline.append("结尾收束为当前该怎么做，以及如何避免再次发生")
    return outline[:6]


def _build_answer_guardrails(diagnosis_result: dict[str, Any], advice_mode: str) -> list[str]:
    guards = [
        "不要一上来输出空泛 SOP。",
        "不要把抽象原则原样复述成口号。",
        "动作必须和用户角色匹配，避免角色越位。",
    ]
    if str(diagnosis_result.get("question_type")) in {"root_cause_judgment", "responsibility_boundary"}:
        guards.extend(
            [
                "必须先定性和归因，再给建议。",
                "不要直接甩锅给单一角色，要区分系统责任。",
            ]
        )
    if advice_mode == "mechanism_first":
        guards.append("优先给机制化治理，不要只给临时救火动作。")
    if advice_mode == "current_case_first":
        guards.append("先给当前个案处理动作，再补长期建议。")
    return guards[:8]


def fallback_planner_v2(
    original_query: str,
    diagnosis_result: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    conversation_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root_primary, root_secondary = _infer_root_causes(diagnosis_result, retrieved_chunks)
    advice_mode = _infer_advice_mode(diagnosis_result)
    translations = _build_action_translation(diagnosis_result, retrieved_chunks)
    return {
        "answer_goal": _infer_answer_goal(diagnosis_result),
        "reasoning_order": _build_reasoning_order(diagnosis_result, advice_mode, root_primary, root_secondary),
        "advice_mode": advice_mode,
        "root_cause_primary": root_primary,
        "root_cause_secondary": root_secondary,
        "action_translation": translations,
        "answer_outline": _build_answer_outline(diagnosis_result, translations, root_primary, root_secondary),
        "answer_guardrails": _build_answer_guardrails(diagnosis_result, advice_mode),
    }


def _sanitize_planner_output(parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_PLANNER_OUTPUT)
    merged.update(fallback)
    merged.update(parsed)
    if merged.get("advice_mode") not in ADVICE_MODES:
        merged["advice_mode"] = fallback["advice_mode"]
    merged["root_cause_primary"] = _normalize_root_cause(str(merged.get("root_cause_primary", fallback["root_cause_primary"])))
    merged["root_cause_secondary"] = _normalize_root_cause(str(merged.get("root_cause_secondary", fallback["root_cause_secondary"])))
    for key in ("reasoning_order", "answer_outline", "answer_guardrails", "action_translation"):
        if not isinstance(merged.get(key), list) or not merged.get(key):
            merged[key] = fallback[key]
    return merged


def llm_planner_v2(
    original_query: str,
    diagnosis_result: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    config: SurfaceRuntimeConfig,
    conversation_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prompt_text = load_prompt(PLANNER_PROMPT_PATH)
    payload = {
        "original_query": original_query,
        "diagnosis_result": diagnosis_result,
        "retrieved_chunks": summarize_retrieved_chunks(retrieved_chunks),
        "conversation_context": conversation_context or [],
    }
    user_text = json.dumps(payload, ensure_ascii=False, indent=2)
    answer, _, _ = request_llm_answer(prompt_text, user_text, config=config, temperature=0.0)
    return _extract_json_object(answer)


def build_planner_v2(
    original_query: str,
    diagnosis_result: dict[str, Any],
    retrieved_chunks: list[dict[str, Any]],
    config: SurfaceRuntimeConfig | None = None,
    conversation_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fallback = fallback_planner_v2(original_query, diagnosis_result, retrieved_chunks, conversation_context)
    if config is None or not config.api_key:
        return fallback
    try:
        parsed = llm_planner_v2(original_query, diagnosis_result, retrieved_chunks, config, conversation_context)
    except Exception:
        return fallback
    return _sanitize_planner_output(parsed, fallback)
