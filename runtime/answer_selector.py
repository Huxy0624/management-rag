from __future__ import annotations

from typing import Any

from runtime.control_layer import build_control_checks
from runtime.rewrite_runtime import rewrite_answer
from runtime.runtime_config import GenerationRuntimeConfig


def _build_prompt_trace(llm_result: dict[str, Any] | None, rewrite_result: dict[str, Any] | None) -> dict[str, Any]:
    trace: dict[str, Any] = {}
    if llm_result:
        trace["surface_prompt_payload"] = llm_result.get("prompt_payload")
    if rewrite_result:
        trace["rewrite_prompt_payload"] = rewrite_result.get("prompt_payload")
    return trace


def select_answer(row: dict[str, Any], llm_result: dict[str, Any] | None, config: GenerationRuntimeConfig) -> dict[str, Any]:
    flags = config.feature_flags
    fallback_answer = str(row.get("v21_final_answer", "")).strip()
    llm_answer = str((llm_result or {}).get("answer", "")).strip()

    empty_checks = {
        "mechanism_name_check_pass": True,
        "missing_mechanism_names": [],
        "action_steps_match_count": None,
        "expected_action_steps_count": None,
        "structure_check_pass": True,
        "contains_generic_filler": False,
    }
    result: dict[str, Any] = {
        "llm_answer": llm_answer,
        "rewrite_answer": None,
        "final_selected_answer": "",
        "selected_from": "",
        "initial_control_checks": empty_checks,
        "final_control_checks": empty_checks,
        "rewrite_triggered": False,
        "fallback_triggered": False,
        "prompt_trace": _build_prompt_trace(llm_result, None),
        "latency_ms": int((llm_result or {}).get("latency_ms", 0)),
        "retry_count": int((llm_result or {}).get("retry_count", 0)),
        "timings_ms": {
            "generation": int((llm_result or {}).get("latency_ms", 0)),
            "rewrite": 0,
        },
    }

    if not llm_answer:
        if flags.enable_fallback_v21 and fallback_answer:
            fallback_checks = build_control_checks(row, fallback_answer) if flags.enable_control_checks else empty_checks
            result.update(
                {
                    "final_selected_answer": fallback_answer,
                    "selected_from": "fallback_v21",
                    "final_control_checks": fallback_checks,
                    "fallback_triggered": True,
                }
            )
            return result
        raise RuntimeError("LLM returned an empty answer and fallback_v21 is disabled.")

    if not flags.enable_control_checks:
        result.update(
            {
                "final_selected_answer": llm_answer,
                "selected_from": "llm_v3",
            }
        )
        return result

    initial_checks = build_control_checks(row, llm_answer)
    result["initial_control_checks"] = initial_checks
    result["final_control_checks"] = initial_checks

    if initial_checks["mechanism_name_check_pass"] and initial_checks["structure_check_pass"]:
        result.update(
            {
                "final_selected_answer": llm_answer,
                "selected_from": "llm_v3",
            }
        )
        return result

    rewrite_result: dict[str, Any] | None = None
    rewrite_checks = initial_checks
    if flags.enable_rewrite_v3:
        result["rewrite_triggered"] = True
        try:
            rewrite_result = rewrite_answer(row, llm_answer, initial_checks, config.surface)
        except Exception as exc:
            result["rewrite_error"] = str(exc)
            rewrite_result = None
        if rewrite_result is not None:
            rewrite_answer_text = str(rewrite_result.get("answer", "")).strip()
            result["rewrite_answer"] = rewrite_answer_text
            result["prompt_trace"] = _build_prompt_trace(llm_result, rewrite_result)
            result["latency_ms"] += int(rewrite_result.get("latency_ms", 0))
            result["retry_count"] += int(rewrite_result.get("retry_count", 0))
            result["timings_ms"]["rewrite"] = int(rewrite_result.get("latency_ms", 0))
            rewrite_checks = build_control_checks(row, rewrite_answer_text)
            result["final_control_checks"] = rewrite_checks
            if rewrite_checks["mechanism_name_check_pass"] and rewrite_checks["structure_check_pass"]:
                result.update(
                    {
                        "final_selected_answer": rewrite_answer_text,
                        "selected_from": "rewrite_v3",
                    }
                )
                return result

    if flags.enable_fallback_v21 and fallback_answer:
        fallback_checks = build_control_checks(row, fallback_answer)
        result.update(
            {
                "final_selected_answer": fallback_answer,
                "selected_from": "fallback_v21",
                "final_control_checks": fallback_checks,
                "fallback_triggered": True,
            }
        )
        return result

    if rewrite_result and result["rewrite_answer"]:
        result.update(
            {
                "final_selected_answer": result["rewrite_answer"],
                "selected_from": "rewrite_v3",
                "final_control_checks": rewrite_checks,
            }
        )
        return result

    result.update(
        {
            "final_selected_answer": llm_answer,
            "selected_from": "llm_v3",
        }
    )
    return result
