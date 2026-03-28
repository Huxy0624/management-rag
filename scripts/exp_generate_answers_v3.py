#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import exp_generate_answers_v1 as v1
import exp_generate_answers_v21 as v21


DEFAULT_V21_ANSWERS_PATH = Path("data/pipeline_candidates/generation_v21/answers_v21.json")
DEFAULT_OUTPUT_DIR = Path("data/pipeline_candidates/generation_v3")
DEFAULT_LLM_MODEL = "gpt-4.1-mini"

PRIORITY_QUERY_IDS = [
    "cross_dept_how_temporary_push",
    "cross_dept_how_mechanism_fix",
    "rule_of_law_fix",
    "upward_compression",
    "downward_diffusion",
]

GENERIC_FILLER_PHRASES = (
    "优化机制",
    "加强管理",
    "提升协同",
    "做好管理",
    "推动落实",
    "持续优化",
    "完善机制",
    "形成合力",
)

SUSPICIOUS_SPECIFICS = (
    "ERP",
    "OA",
    "OKR",
    "KPI",
    "SOP",
    "RACI",
    "九宫格",
    "日报",
    "周报",
    "月报",
)

ANSWER_PROMPT_V3 = """你是一个管理知识问答助手。

你不会自己思考答案结构，
而是基于“已经规划好的回答骨架（planner_output）”，
将其改写为清晰、自然、专业的表达。

【输入信息】
- query
- query_type
- planner_output（结构化骨架）

【你的任务】
你只做一件事：
把 planner_output 改写成用户可以直接理解的答案。

【严格约束】
1. 不允许改变 planner 的结构。
2. 不允许新增 planner 中不存在的逻辑。
3. 不允许删除 planner 的关键步骤。
4. 优先解释清楚，而不是说得高级。
5. 不允许输出空话（如：优化机制、加强管理、提升协同）。
6. 只输出答案正文，不要输出标题、说明、JSON、项目符号或额外备注。

【分类型生成约束】
what：
- 第一句必须是定义句。
- 必须包含“本质是”。
- 禁止把机制罗列当成定义。

why：
- 必须按“结论 -> 因果链（2-3步） -> 管理含义”输出。
- 必须出现“因为 / 导致 / 最终”这类因果词。

how：
- 必须保持动作导向。
- generic_how / temporary_push：每一步都要保留动作对象、动作行为、输出结果。
- mechanism_building：必须保留机制实体，至少写出机制名称、责任人、执行节奏、关键字段或输出。
"""

REWRITE_PROMPT_V3 = """你是一个管理知识问答助手的最后一轮输出纠偏器。

你不会重新规划答案，
也不会新增 planner_output 之外的逻辑。
你只会在保留原有结构的前提下，对当前答案做最小必要改写。

【输入信息】
- query
- query_type
- planner_output
- current_answer
- detected_issues

【你的任务】
只做受控改写，让答案通过控制层校验。

【总约束】
1. 不允许改变 planner_output 的步骤顺序。
2. 不允许新增 planner_output 之外的新动作、新机制、新制度、新判断。
3. 不允许删除 planner_output 中已有的关键步骤、适用条件或风险提醒。
4. 只输出改写后的答案正文。

【mechanism_building rewrite】
- 必须逐字保留 planner_output.mechanism_entities 中的全部名称。
- 不允许用“规则表”“机制实体”“某种机制”替代原名。

【how structure rewrite】
- 必须明确呈现三个步骤，且只能写成：
  第一步：...
  第二步：...
  第三步：...
- 每一步都必须包含：
  动作对象（谁）
  动作行为（做什么）
  输出结果（产出什么）
- 不允许合并步骤。
- 不允许用“接着/然后”模糊步骤边界来替代“第一步/第二步/第三步”。
- 不允许用解释句替代动作句。
- 允许步骤内部自然表达，但步骤边界必须显式。
"""

ANSWER_EVAL_DIMENSIONS_V3 = [
    {
        "name": "structure_preservation_score",
        "scale": "1-5",
        "question": "LLM 是否保持了 planner 的原始结构，而没有随意改写顺序或层级。",
    },
    {
        "name": "hallucination_score",
        "scale": "1-5",
        "question": "答案是否新增 planner 中不存在的具体信息、工具、机制或判断。",
    },
    {
        "name": "action_fidelity_score",
        "scale": "1-5",
        "question": "how 类答案是否保留动作对象、动作行为、交付物，而没有弱化成空话。",
    },
    {
        "name": "mechanism_fidelity_score",
        "scale": "1-5",
        "question": "mechanism_building 是否保留机制名称、owner、cadence、fields 等关键细节。",
    },
    {
        "name": "readability_score",
        "scale": "1-5",
        "question": "LLM 表达是否比 v2.1 规则生成更自然、更顺畅。",
    },
]


class SurfaceGenerationConfigError(Exception):
    """Raised when the LLM surface generation config is invalid."""


class SurfaceGenerationRuntimeError(Exception):
    """Raised when the online LLM request fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation v3 online surface generation experiment over planner_output_v21.")
    parser.add_argument("--v21-answers-path", type=Path, default=DEFAULT_V21_ANSWERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--priority-only", action="store_true", help="Only run the required first-batch priority queries.")
    parser.add_argument("--fail-on-missing-api-key", action="store_true", help="Exit non-zero if OPENAI_API_KEY is not configured.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def should_include(query_id: str, priority_only: bool) -> bool:
    if not priority_only:
        return True
    return query_id in PRIORITY_QUERY_IDS


@lru_cache(maxsize=1)
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SurfaceGenerationConfigError(
            "OPENAI_API_KEY is not set.\n"
            "Windows PowerShell example:\n"
            '$env:OPENAI_API_KEY = "your-openai-api-key"'
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SurfaceGenerationConfigError(
            "Missing dependency 'openai'.\n"
            "Please run:\n"
            'python -m pip install -r requirements.txt'
        ) from exc

    kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def build_style_instruction(row: dict[str, Any]) -> str:
    query_type = str(row["query_type"])
    planner_output = row["planner_output_v21"]
    if query_type == "what":
        return "写成定义先行、边界补充、例子点到即止的短答案。"
    if query_type == "why":
        return "写成结论先行、因果链清晰、管理含义收束的解释。"
    if planner_output.get("solution_mode") == "mechanism_building":
        return "写成机制化方案，优先把机制实体自然嵌入句子，但不能丢掉 owner、cadence、fields。"
    return "写成可执行动作说明，保留步骤顺序，并把对象、动作、交付物说清楚。"


def build_constraints(row: dict[str, Any]) -> list[str]:
    query_type = str(row["query_type"])
    planner_output = row["planner_output_v21"]
    constraints = [
        "不要引用 raw evidence，不要补充 planner_output 之外的新事实。",
        "不要改变 planner_output 中步骤、因果链或定义边界的顺序。",
        "不要删掉 planner_output 里的关键动作、机制实体、适用条件或风险提醒。",
        "避免空话，避免“优化机制、加强管理、提升协同”一类占位表达。",
    ]
    if query_type == "what":
        constraints.extend(
            [
                "第一句必须是定义句。",
                "必须包含“本质是”。",
            ]
        )
    elif query_type == "why":
        constraints.extend(
            [
                "必须出现“因为”或同类因果词。",
                "必须出现“导致”和“最终”。",
                "结尾保留管理含义。",
            ]
        )
    else:
        constraints.extend(
            [
                "第一句必须动作导向。",
                "每一步都要保留动作对象、动作行为、输出结果。",
            ]
        )
        if planner_output.get("solution_mode") == "mechanism_building":
            constraints.append("必须明确写出机制名称、责任人、执行节奏、关键字段或输出。")
    return constraints


def build_prompt_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": row["query"],
        "query_type": row["query_type"],
        "planner_output": row["planner_output_v21"],
        "style_instruction": build_style_instruction(row),
        "constraints": build_constraints(row),
    }


def request_llm_answer(prompt_text: str, user_text: str, model: str, temperature: float = 0.2) -> tuple[str, dict[str, Any]]:
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": user_text},
            ],
        )
    except Exception as exc:
        raise SurfaceGenerationRuntimeError(f"Online surface generation failed: {exc}") from exc

    answer = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    usage_payload = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return answer, usage_payload


def request_llm_surface_answer(prompt_text: str, payload: dict[str, Any], model: str) -> tuple[str, dict[str, Any]]:
    user_text = "请严格根据以下 JSON 输入生成答案，只输出答案正文，不要解释你的做法。\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return request_llm_answer(prompt_text, user_text, model=model, temperature=0.2)


def split_sentences(text: str) -> list[str]:
    chunks = text.replace("！", "。").replace("？", "。").split("。")
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def clamp_score(value: int) -> int:
    return max(1, min(5, value))


def short_tokens(text: str) -> list[str]:
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


def mechanism_signal_count(answer: str, entity: dict[str, Any]) -> int:
    checks = 0
    if str(entity.get("name", "")) in answer:
        checks += 1
    if any(token in answer for token in short_tokens(str(entity.get("owner", "")))):
        checks += 1
    if any(token in answer for token in short_tokens(str(entity.get("cadence", "")))):
        checks += 1
    if any(token in answer for token in short_tokens(str(entity.get("fields_or_rhythm", "")))):
        checks += 1
    return checks


def contains_generic_filler(answer: str) -> bool:
    return any(phrase in answer for phrase in GENERIC_FILLER_PHRASES)


def is_mechanism_building_row(row: dict[str, Any]) -> bool:
    router = row.get("router_decision", {})
    planner_output = row.get("planner_output_v21", {})
    return router.get("subtype") == "mechanism_building" or planner_output.get("solution_mode") == "mechanism_building"


def missing_mechanism_names(row: dict[str, Any], answer: str) -> list[str]:
    if not is_mechanism_building_row(row):
        return []
    entities = row["planner_output_v21"].get("mechanism_entities", [])
    return [str(entity["name"]) for entity in entities if str(entity["name"]) not in answer]


def action_steps_match_count(row: dict[str, Any], answer: str) -> int | None:
    if str(row["query_type"]) != "how":
        return None
    steps = row["planner_output_v21"].get("action_steps", [])
    return sum(1 for step in steps if step_signal_count(answer, step) >= 2)


def build_control_checks(row: dict[str, Any], answer: str) -> dict[str, Any]:
    missing_names = missing_mechanism_names(row, answer)
    mechanism_pass = not missing_names
    match_count = action_steps_match_count(row, answer)
    expected_step_count = len(row["planner_output_v21"].get("action_steps", [])) if str(row["query_type"]) == "how" else None
    structure_pass = match_count == expected_step_count if expected_step_count is not None else True
    return {
        "mechanism_name_check_pass": mechanism_pass,
        "missing_mechanism_names": missing_names,
        "action_steps_match_count": match_count,
        "expected_action_steps_count": expected_step_count,
        "structure_check_pass": structure_pass,
    }


def build_rewrite_payload(row: dict[str, Any], current_answer: str, control_checks: dict[str, Any]) -> dict[str, Any]:
    detected_issues: list[str] = []
    if not control_checks["mechanism_name_check_pass"]:
        detected_issues.append("缺失的 mechanism entity names: " + "、".join(control_checks["missing_mechanism_names"]))
    if not control_checks["structure_check_pass"]:
        detected_issues.append(
            "how 结构未完整保留 3 步动作，当前只匹配到 "
            f"{control_checks['action_steps_match_count']}/{control_checks['expected_action_steps_count']}"
        )
    return {
        "query": row["query"],
        "query_type": row["query_type"],
        "planner_output": row["planner_output_v21"],
        "current_answer": current_answer,
        "detected_issues": detected_issues,
    }


def request_llm_rewrite(row: dict[str, Any], current_answer: str, control_checks: dict[str, Any], model: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    payload = build_rewrite_payload(row, current_answer, control_checks)
    user_text = "请根据以下 JSON 做最后一轮受控改写，只输出改写后的答案正文。\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    answer, usage = request_llm_answer(REWRITE_PROMPT_V3, user_text, model=model, temperature=0.0)
    return answer, usage, payload


def score_structure_preservation(row: dict[str, Any], answer: str) -> tuple[int, dict[str, Any]]:
    query_type = str(row["query_type"])
    planner_output = row["planner_output_v21"]
    if not answer:
        return 1, {"reason": "empty_answer"}

    if query_type == "what":
        sentences = split_sentences(answer)
        checks = {
            "has_definition_first_sentence": bool(sentences and "本质是" in sentences[0]),
            "keeps_boundary": any(token in answer for token in short_tokens(str(planner_output.get("boundary_explanation", "")))),
        }
        score = 1 + sum(1 for ok in checks.values() if ok) * 2
        return clamp_score(score), checks

    if query_type == "why":
        chain = planner_output.get("causal_chain", [])
        chain_hits = sum(1 for item in chain if any(token in answer for token in short_tokens(str(item))))
        checks = {
            "has_conclusion": any(token in answer for token in short_tokens(str(planner_output.get("core_conclusion", "")))),
            "causal_words": all(token in answer for token in ("因为", "导致", "最终")),
            "chain_hits": chain_hits,
        }
        score = 1 + (1 if checks["has_conclusion"] else 0) + (1 if checks["causal_words"] else 0) + min(2, chain_hits)
        return clamp_score(score), checks

    steps = planner_output.get("action_steps", [])
    matched_steps = sum(1 for step in steps if step_signal_count(answer, step) >= 2)
    mechanism_entities = planner_output.get("mechanism_entities", [])
    mechanism_names_preserved = all(entity["name"] in answer for entity in mechanism_entities) if mechanism_entities else True
    checks = {
        "matched_steps": matched_steps,
        "step_count": len(steps),
        "mechanism_names_preserved": mechanism_names_preserved,
    }
    score = 1 + round(4 * (matched_steps / max(1, len(steps))))
    if mechanism_entities and not mechanism_names_preserved:
        score -= 1
    return clamp_score(score), checks


def score_hallucination(row: dict[str, Any], answer: str) -> tuple[int, dict[str, Any]]:
    planner_text = json.dumps(row["planner_output_v21"], ensure_ascii=False)
    introduced_specifics = [token for token in SUSPICIOUS_SPECIFICS if token in answer and token not in planner_text]
    checks = {
        "introduced_specifics": introduced_specifics,
        "generic_filler_detected": contains_generic_filler(answer),
    }
    score = 5
    if introduced_specifics:
        score -= 2
    if contains_generic_filler(answer):
        score -= 1
    return clamp_score(score), checks


def score_action_fidelity(row: dict[str, Any], answer: str) -> tuple[int | None, dict[str, Any]]:
    if str(row["query_type"]) != "how":
        return None, {"not_applicable": True}
    steps = row["planner_output_v21"].get("action_steps", [])
    matched_steps = sum(1 for step in steps if step_signal_count(answer, step) >= 2)
    checks = {
        "matched_steps": matched_steps,
        "step_count": len(steps),
        "generic_filler_detected": contains_generic_filler(answer),
    }
    score = 1 + round(4 * (matched_steps / max(1, len(steps))))
    if contains_generic_filler(answer):
        score -= 1
    return clamp_score(score), checks


def score_mechanism_fidelity(row: dict[str, Any], answer: str) -> tuple[int | None, dict[str, Any]]:
    entities = row["planner_output_v21"].get("mechanism_entities", [])
    if not entities:
        return None, {"not_applicable": True}
    matched_entities = sum(1 for entity in entities if mechanism_signal_count(answer, entity) >= 3)
    all_names_present = all(entity["name"] in answer for entity in entities)
    checks = {
        "matched_entities": matched_entities,
        "entity_count": len(entities),
        "all_names_present": all_names_present,
    }
    score = 1 + round(4 * (matched_entities / max(1, len(entities))))
    if not all_names_present:
        score -= 1
    return clamp_score(score), checks


def score_readability(v21_answer: str, llm_answer: str) -> tuple[int, dict[str, Any]]:
    v21_len = len(v21_answer)
    llm_len = len(llm_answer)
    connective_count = sum(llm_answer.count(token) for token in ("先", "再", "同时", "最后", "因此", "这样"))
    checks = {
        "v21_length": v21_len,
        "llm_length": llm_len,
        "connective_count": connective_count,
        "same_opening_as_v21": llm_answer.startswith("更可执行的做法是") or llm_answer.startswith("更合适的做法是"),
    }
    score = 3
    if v21_len and 0.7 * v21_len <= llm_len <= 1.15 * v21_len:
        score += 1
    if connective_count >= 2:
        score += 1
    if checks["same_opening_as_v21"]:
        score -= 1
    return clamp_score(score), checks


def evaluate_answer(row: dict[str, Any], llm_answer: str) -> dict[str, Any]:
    structure_score, structure_checks = score_structure_preservation(row, llm_answer)
    hallucination_score, hallucination_checks = score_hallucination(row, llm_answer)
    action_score, action_checks = score_action_fidelity(row, llm_answer)
    mechanism_score, mechanism_checks = score_mechanism_fidelity(row, llm_answer)
    readability_score, readability_checks = score_readability(str(row["final_answer"]), llm_answer)
    return {
        "scores": {
            "structure_preservation_score": structure_score,
            "hallucination_score": hallucination_score,
            "action_fidelity_score": action_score,
            "mechanism_fidelity_score": mechanism_score,
            "readability_score": readability_score,
        },
        "checks": {
            "structure": structure_checks,
            "hallucination": hallucination_checks,
            "action": action_checks,
            "mechanism": mechanism_checks,
            "readability": readability_checks,
        },
        "flags": {
            "structure_broken": structure_score <= 3,
            "hallucination_risk": hallucination_score <= 3,
            "action_weakened": action_score is not None and action_score <= 3,
            "mechanism_detail_lost": mechanism_score is not None and mechanism_score <= 3,
            "generic_filler_detected": contains_generic_filler(llm_answer),
        },
    }


def apply_control_layer(
    row: dict[str, Any],
    llm_answer: str | None,
    llm_status: dict[str, Any],
    model: str,
    api_key_missing: bool,
) -> dict[str, Any]:
    if not llm_answer:
        empty_checks = {
            "mechanism_name_check_pass": None,
            "missing_mechanism_names": [],
            "action_steps_match_count": None,
            "expected_action_steps_count": None,
            "structure_check_pass": None,
            "rewrite_triggered": False,
            "fallback_triggered": False,
        }
        return {
            "initial_control_checks": empty_checks,
            "rewrite_triggered": False,
            "fallback_triggered": False,
            "rewrite_prompt_payload": None,
            "rewrite_status": None,
            "rewritten_answer": None,
            "selected_from": None,
            "final_selected_answer": None,
            "final_control_checks": empty_checks,
        }

    initial_checks = build_control_checks(row, llm_answer)
    needs_rewrite = not (initial_checks["mechanism_name_check_pass"] and initial_checks["structure_check_pass"])
    rewrite_answer: str | None = None
    rewrite_prompt_payload: dict[str, Any] | None = None
    rewrite_status: dict[str, Any] | None = None
    rewrite_checks = initial_checks
    fallback_triggered = False
    selected_from = "llm_v3"
    final_selected_answer = llm_answer

    if needs_rewrite:
        if api_key_missing:
            fallback_triggered = True
            selected_from = "fallback_v21"
            final_selected_answer = row["final_answer"]
            rewrite_checks = build_control_checks(row, final_selected_answer)
            rewrite_status = {
                "status": "skipped_missing_api_key",
                "model": model,
            }
        else:
            rewrite_answer, rewrite_usage, rewrite_prompt_payload = request_llm_rewrite(row, llm_answer, initial_checks, model=model)
            rewrite_checks = build_control_checks(row, rewrite_answer)
            rewrite_status = {
                "status": "success",
                "model": model,
                "usage": rewrite_usage,
            }
            if rewrite_checks["mechanism_name_check_pass"] and rewrite_checks["structure_check_pass"]:
                selected_from = "rewrite_v3"
                final_selected_answer = rewrite_answer
            else:
                fallback_triggered = True
                selected_from = "fallback_v21"
                final_selected_answer = row["final_answer"]
                rewrite_checks = build_control_checks(row, final_selected_answer)

    final_checks = dict(rewrite_checks if needs_rewrite else initial_checks)
    final_checks["rewrite_triggered"] = needs_rewrite
    final_checks["fallback_triggered"] = fallback_triggered
    return {
        "initial_control_checks": {
            **initial_checks,
            "rewrite_triggered": needs_rewrite,
            "fallback_triggered": False,
        },
        "rewrite_triggered": needs_rewrite,
        "fallback_triggered": fallback_triggered,
        "rewrite_prompt_payload": rewrite_prompt_payload,
        "rewrite_status": rewrite_status,
        "rewritten_answer": rewrite_answer,
        "selected_from": selected_from,
        "final_selected_answer": final_selected_answer,
        "final_control_checks": final_checks,
    }


def compare_single_row(
    row: dict[str, Any],
    llm_answer: str | None,
    llm_status: dict[str, Any],
    control_result: dict[str, Any],
    final_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    solution_mode = row["planner_output_v21"].get("solution_mode")
    if not control_result["final_selected_answer"] or not final_evaluation:
        return {
            "query_id": row["query_id"],
            "query": row["query"],
            "query_type": row["query_type"],
            "run_status": llm_status,
            "planner_output_v21": row["planner_output_v21"],
            "v21_final_answer": row["final_answer"],
            "v3_llm_answer": llm_answer,
            "control_layer": control_result,
            "final_selected_answer": control_result["final_selected_answer"],
            "selected_from": control_result["selected_from"],
            "comparison": {
                "structure_preserved": None,
                "actionability_weakened": None,
                "mechanism_entities_lost": None,
                "hallucination_detected": None,
                "readability_improved": None,
            },
        }

    scores = final_evaluation["scores"]
    return {
        "query_id": row["query_id"],
        "query": row["query"],
        "query_type": row["query_type"],
        "solution_mode": solution_mode,
        "run_status": llm_status,
        "planner_output_v21": row["planner_output_v21"],
        "v21_final_answer": row["final_answer"],
        "v3_llm_answer": llm_answer,
        "rewritten_answer": control_result["rewritten_answer"],
        "final_selected_answer": control_result["final_selected_answer"],
        "selected_from": control_result["selected_from"],
        "control_layer": control_result,
        "evaluation": final_evaluation,
        "comparison": {
            "structure_preserved": scores["structure_preservation_score"] >= 4,
            "actionability_weakened": scores["action_fidelity_score"] is not None and scores["action_fidelity_score"] <= 3,
            "mechanism_entities_lost": scores["mechanism_fidelity_score"] is not None and scores["mechanism_fidelity_score"] <= 3,
            "hallucination_detected": scores["hallucination_score"] <= 3,
            "readability_improved": scores["readability_score"] >= 4,
        },
    }


def summarize_results(compare_rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in compare_rows if row["run_status"]["status"] == "success"]
    blocked = [row for row in compare_rows if row["run_status"]["status"] != "success"]
    if not successful:
        return {
            "successful_queries": 0,
            "blocked_queries": len(blocked),
            "structure_preserved_count": 0,
            "actionability_weakened_count": 0,
            "mechanism_entities_lost_count": 0,
            "hallucination_detected_count": 0,
            "readability_improved_count": 0,
        }
    return {
        "successful_queries": len(successful),
        "blocked_queries": len(blocked),
        "structure_preserved_count": sum(1 for row in successful if row["comparison"]["structure_preserved"]),
        "actionability_weakened_count": sum(1 for row in successful if row["comparison"]["actionability_weakened"]),
        "mechanism_entities_lost_count": sum(1 for row in successful if row["comparison"]["mechanism_entities_lost"]),
        "hallucination_detected_count": sum(1 for row in successful if row["comparison"]["hallucination_detected"]),
        "readability_improved_count": sum(1 for row in successful if row["comparison"]["readability_improved"]),
    }


def build_pre_integration_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_queries = len(rows)
    final_rows = [row for row in rows if row.get("final_selected_answer")]
    mechanism_rows = [row for row in final_rows if is_mechanism_building_row(row)]
    how_rows = [row for row in final_rows if str(row.get("query_type")) == "how"]
    mechanism_pass_count = sum(1 for row in mechanism_rows if row["control_layer"]["final_control_checks"]["mechanism_name_check_pass"])
    structure_pass_count = sum(1 for row in how_rows if row["control_layer"]["final_control_checks"]["structure_check_pass"])
    rewrite_count = sum(1 for row in final_rows if row["control_layer"]["rewrite_triggered"])
    fallback_count = sum(1 for row in final_rows if row["control_layer"]["fallback_triggered"])
    llm_selected_count = sum(1 for row in final_rows if row["selected_from"] in {"llm_v3", "rewrite_v3"})
    fallback_selected_count = sum(1 for row in final_rows if row["selected_from"] == "fallback_v21")
    problematic_queries = [
        {
            "query_id": row["query_id"],
            "selected_from": row["selected_from"],
            "missing_mechanism_names": row["control_layer"]["final_control_checks"]["missing_mechanism_names"],
            "action_steps_match_count": row["control_layer"]["final_control_checks"]["action_steps_match_count"],
        }
        for row in final_rows
        if row["control_layer"]["rewrite_triggered"] or row["control_layer"]["fallback_triggered"]
    ]
    fallback_rate = fallback_selected_count / total_queries if total_queries else 0.0
    if fallback_rate < 0.2 and all(row["selected_from"] != "fallback_v21" for row in final_rows):
        recommendation = "可以进入主链路预集成，当前控制层已能锁住 mechanism name 和 3-step how 结构。"
    elif fallback_rate < 0.2:
        recommendation = "可以进入主链路预集成，但建议持续监控 fallback query，并优先观察 mechanism_building 的 rewrite 命中率。"
    else:
        recommendation = "暂不建议直接进入主链路预集成，应先继续补强 rewrite prompt 或选择规则。"
    return {
        "total_queries": total_queries,
        "mechanism_name_check_pass_count": mechanism_pass_count,
        "mechanism_name_check_pass_rate": round(mechanism_pass_count / len(mechanism_rows), 4) if mechanism_rows else None,
        "how_structure_check_pass_count": structure_pass_count,
        "how_structure_check_pass_rate": round(structure_pass_count / len(how_rows), 4) if how_rows else None,
        "rewrite_triggered_count": rewrite_count,
        "fallback_triggered_count": fallback_count,
        "final_llm_selected_count": llm_selected_count,
        "final_fallback_selected_count": fallback_selected_count,
        "problematic_queries": problematic_queries,
        "recommendation": recommendation,
    }


def write_design_doc(output_dir: Path) -> None:
    lines = [
        "# Generation V3 Design",
        "",
        "## Pipeline",
        "",
        "retrieval (v4_phase22) -> rerank -> planner (v2.1) -> LLM surface generation -> answer evaluation",
        "",
        "## Guardrails",
        "",
        "- LLM 输入只包含 `query / query_type / planner_output / style_instruction / constraints`。",
        "- 不传 raw evidence，不让 LLM 临场规划，不让 LLM 重新决定 answer structure。",
        "- 评测重点是 `structure_preservation / hallucination / action_fidelity / mechanism_fidelity / readability`。",
        "- mechanism_building 增加 exact name check，若缺失原名则触发 rewrite，仍失败则 fallback 到 `v21_final_answer`。",
        "- how 增加 3-step completeness check，若 `action_steps_match_count < 3` 则触发 rewrite，仍失败则 fallback。",
    ]
    (output_dir / "generation_v3_design.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    v21_answers = load_json(args.v21_answers_path)
    rows = [
        row for row in v21_answers.get("answers", [])
        if should_include(str(row.get("query_id", "")), args.priority_only)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_design_doc(args.output_dir)
    (args.output_dir / "answer_prompt_v3.txt").write_text(ANSWER_PROMPT_V3, encoding="utf-8")
    (args.output_dir / "answer_rewrite_prompt_v3.txt").write_text(REWRITE_PROMPT_V3, encoding="utf-8")

    api_key_missing = not bool(os.getenv("OPENAI_API_KEY"))
    if api_key_missing and args.fail_on_missing_api_key:
        raise SurfaceGenerationConfigError("OPENAI_API_KEY is required for online generation.")

    generated_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for row in rows:
        prompt_payload = build_prompt_payload(row)
        llm_answer: str | None = None
        llm_evaluation: dict[str, Any] | None = None
        final_evaluation: dict[str, Any] | None = None
        if api_key_missing:
            llm_status = {
                "status": "blocked_missing_api_key",
                "model": args.model,
                "message": "OPENAI_API_KEY not configured; prompt and evaluation framework generated, online call skipped.",
            }
            control_result = apply_control_layer(row, None, llm_status, model=args.model, api_key_missing=api_key_missing)
        else:
            llm_answer, usage = request_llm_surface_answer(ANSWER_PROMPT_V3, prompt_payload, args.model)
            llm_evaluation = evaluate_answer(row, llm_answer)
            llm_status = {
                "status": "success",
                "model": args.model,
                "usage": usage,
            }
            control_result = apply_control_layer(row, llm_answer, llm_status, model=args.model, api_key_missing=api_key_missing)
            final_evaluation = evaluate_answer(row, control_result["final_selected_answer"])

        generated_rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "query_type": row["query_type"],
                "router_decision": row["router_decision"],
                "planner_output_v21": row["planner_output_v21"],
                "v21_final_answer": row["final_answer"],
                "prompt_payload_v3": prompt_payload,
                "llm_status": llm_status,
                "llm_answer": llm_answer,
                "llm_answer_evaluation": llm_evaluation,
                "control_layer": control_result,
                "rewritten_answer": control_result["rewritten_answer"],
                "rewrite_prompt_payload_v3": control_result["rewrite_prompt_payload"],
                "rewrite_status": control_result["rewrite_status"],
                "final_selected_answer": control_result["final_selected_answer"],
                "selected_from": control_result["selected_from"],
                "final_evaluation": final_evaluation,
            }
        )
        compare_rows.append(compare_single_row(row, llm_answer, llm_status, control_result, final_evaluation))
        eval_rows.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "query_type": row["query_type"],
                "selected_from": control_result["selected_from"],
                "final_selected_answer": control_result["final_selected_answer"],
                "score_by_dimension": final_evaluation["scores"] if final_evaluation else None,
                "checks": final_evaluation["checks"] if final_evaluation else None,
                "flags": final_evaluation["flags"] if final_evaluation else None,
                "control_layer": control_result["final_control_checks"],
                "run_status": llm_status,
            }
        )

    compare_summary = summarize_results(compare_rows)
    pre_integration_report = build_pre_integration_report(generated_rows)
    (args.output_dir / "answers_v3.json").write_text(
        json.dumps(
            {
                "v21_answers_path": str(args.v21_answers_path),
                "model": args.model,
                "query_count": len(generated_rows),
                "answers": generated_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "answer_eval_v3.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "dimensions": ANSWER_EVAL_DIMENSIONS_V3,
                "query_count": len(eval_rows),
                "evaluations": eval_rows,
                "summary": compare_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "compare_v21_vs_v3.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "query_count": len(compare_rows),
                "priority_query_ids": PRIORITY_QUERY_IDS,
                "summary": compare_summary,
                "comparisons": compare_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "pre_integration_report.json").write_text(
        json.dumps(pre_integration_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved generation v3 outputs to: {args.output_dir}")
    print(f"Prepared rows: {len(generated_rows)}")
    if api_key_missing:
        print("Online generation skipped because OPENAI_API_KEY is not set.")


if __name__ == "__main__":
    main()
