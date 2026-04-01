from __future__ import annotations

import json
import time
from typing import Any

from runtime.llm_surface_runtime import load_prompt, request_llm_answer
from runtime.output_language import answer_language_suffix
from runtime.runtime_config import SurfaceRuntimeConfig


def build_rewrite_payload(row: dict[str, Any], current_answer: str, control_checks: dict[str, Any]) -> dict[str, Any]:
    detected_issues: list[str] = []
    if not control_checks["mechanism_name_check_pass"]:
        detected_issues.append("缺失的 mechanism entity names: " + "、".join(control_checks["missing_mechanism_names"]))
    if not control_checks["structure_check_pass"]:
        detected_issues.append(
            "how 结构未完整保留 3 步动作，当前只匹配到 "
            f"{control_checks['action_steps_match_count']}/{control_checks['expected_action_steps_count']}"
        )
    if not control_checks.get("diagnosis_mode_check_pass", True):
        detected_issues.append("diagnosis 问题被改写偏了：" + str(control_checks.get("diagnosis_fail_reason", "")))
    return {
        "query": row["query"],
        "query_type": row["query_type"],
        "output_language": row.get("output_language", "zh"),
        "question_diagnosis": row.get("question_diagnosis"),
        "planner_meta": row.get("planner_output_v2"),
        "planner_output": row["planner_output_v21"],
        "current_answer": current_answer,
        "detected_issues": detected_issues,
    }


def rewrite_answer(row: dict[str, Any], current_answer: str, control_checks: dict[str, Any], config: SurfaceRuntimeConfig) -> dict[str, Any]:
    prompt_text = load_prompt(config.rewrite_prompt_path)
    payload = build_rewrite_payload(row, current_answer, control_checks)
    user_text = (
        "请根据以下 JSON 做最后一轮受控改写，只输出改写后的答案正文。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + answer_language_suffix(str(row.get("output_language", "zh")))
    )
    started = time.perf_counter()
    answer, usage, retry_count = request_llm_answer(prompt_text, user_text, config=config, temperature=0.0)
    return {
        "answer": answer,
        "prompt_text": prompt_text + "\n\n" + user_text,
        "prompt_length": len(prompt_text) + len(user_text),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "retry_count": retry_count,
        "prompt_payload": payload,
        "usage": usage,
    }
