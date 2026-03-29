from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FAILURE_LOG_PATH = Path("logs/generation_failures.jsonl")
FAILURE_TYPES = {
    "sop_opening_on_root_cause",
    "wrong_question_type",
    "role_misclassification",
    "over_mechanism_on_current_case",
    "vague_answer",
    "other",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_failure_type(failure_type: str | None) -> str:
    candidate = str(failure_type or "other").strip()
    return candidate if candidate in FAILURE_TYPES else "other"


def build_failure_case_record(
    *,
    original_query: str,
    diagnosis_result: dict[str, Any] | None,
    planner_result: dict[str, Any] | None,
    final_answer: str,
    triggered_guardrail: bool,
    failure_type: str,
    notes: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "original_query": original_query,
        "diagnosis_result": diagnosis_result or {},
        "planner_result": planner_result or {},
        "final_answer": final_answer,
        "triggered_guardrail": bool(triggered_guardrail),
        "failure_type": normalize_failure_type(failure_type),
        "optional_notes": notes or "",
    }
    if extra:
        record["extra"] = extra
    return record


def record_failure_case(
    *,
    original_query: str,
    diagnosis_result: dict[str, Any] | None,
    planner_result: dict[str, Any] | None,
    final_answer: str,
    triggered_guardrail: bool,
    failure_type: str,
    notes: str | None = None,
    output_path: str | Path = DEFAULT_FAILURE_LOG_PATH,
    enabled: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    if not enabled:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = build_failure_case_record(
        original_query=original_query,
        diagnosis_result=diagnosis_result,
        planner_result=planner_result,
        final_answer=final_answer,
        triggered_guardrail=triggered_guardrail,
        failure_type=failure_type,
        notes=notes,
        extra=extra,
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def detect_failure_case(
    *,
    original_query: str,
    diagnosis_result: dict[str, Any] | None,
    planner_result: dict[str, Any] | None,
    final_answer: str,
    triggered_guardrail: bool,
) -> tuple[str | None, str]:
    diagnosis = dict(diagnosis_result or {})
    planner = dict(planner_result or {})
    answer = str(final_answer or "").strip()
    prefix = answer[:80]

    if triggered_guardrail:
        if diagnosis.get("question_type") in {"root_cause_judgment", "responsibility_boundary"} and prefix.startswith(
            ("第一步", "第二步", "第三步")
        ):
            return "sop_opening_on_root_cause", "Diagnosis-style question still opened as SOP."
        return "other", "Guardrail or control layer was triggered."

    if diagnosis.get("question_type") == "principle_explanation" and any(
        token in original_query for token in ("谁导致", "谁该负责", "谁该拍板", "怎么处理", "怎么收口")
    ):
        return "wrong_question_type", "Question looks like diagnosis/action but stayed in principle_explanation."

    if diagnosis.get("user_role") == "unknown" and any(
        token in original_query for token in ("副班长", "经理", "一号位", "老板", "负责人")
    ):
        return "role_misclassification", "Role cue exists in the question but user_role stayed unknown."

    if diagnosis.get("time_horizon") == "current_case" and planner.get("advice_mode") == "mechanism_first":
        return "over_mechanism_on_current_case", "Current-case question was over-shifted to mechanism-first planning."

    if len(answer) < 24 or any(phrase in answer for phrase in ("优化机制", "加强管理", "提升协同", "完善流程")):
        return "vague_answer", "Answer is too short or contains generic filler."

    return None, ""
