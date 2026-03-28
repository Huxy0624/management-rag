from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from runtime.runtime_config import SurfaceRuntimeConfig


class SurfaceGenerationConfigError(Exception):
    """Raised when the LLM surface generation config is invalid."""


class SurfaceGenerationRuntimeError(Exception):
    """Raised when the online surface generation request fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_text: str = "",
        prompt_length: int = 0,
        retry_count: int = 0,
        latency_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_text = prompt_text
        self.prompt_length = prompt_length
        self.retry_count = retry_count
        self.latency_ms = latency_ms


def load_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise SurfaceGenerationConfigError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_style_instruction(query_type: str, planner_output: dict[str, Any]) -> str:
    if query_type == "what":
        return "写成定义先行、边界补充、例子点到即止的短答案。"
    if query_type == "why":
        return "写成结论先行、因果链清晰、管理含义收束的解释。"
    if planner_output.get("solution_mode") == "mechanism_building":
        return "写成机制化方案，优先把机制实体自然嵌入句子，但不能丢掉 owner、cadence、fields。"
    return "写成可执行动作说明，保留步骤顺序，并把对象、动作、交付物说清楚。"


def build_constraints(query_type: str, planner_output: dict[str, Any]) -> list[str]:
    constraints = [
        "不要引用 raw evidence，不要补充 planner_output 之外的新事实。",
        "不要改变 planner_output 中步骤、因果链或定义边界的顺序。",
        "不要删掉 planner_output 里的关键动作、机制实体、适用条件或风险提醒。",
        "避免空话，避免“优化机制、加强管理、提升协同”一类占位表达。",
    ]
    if query_type == "what":
        constraints.extend(["第一句必须是定义句。", "必须包含“本质是”。"])
    elif query_type == "why":
        constraints.extend(["必须出现“因为”或同类因果词。", "必须出现“导致”和“最终”。", "结尾保留管理含义。"])
    else:
        constraints.extend(["第一句必须动作导向。", "每一步都要保留动作对象、动作行为、输出结果。"])
        if planner_output.get("solution_mode") == "mechanism_building":
            constraints.append("必须明确写出机制名称、责任人、执行节奏、关键字段或输出。")
    return constraints


def build_surface_payload(query: str, query_type: str, planner_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": query,
        "query_type": query_type,
        "planner_output": planner_output,
        "style_instruction": build_style_instruction(query_type, planner_output),
        "constraints": build_constraints(query_type, planner_output),
    }


def get_openai_client(config: SurfaceRuntimeConfig):
    api_key = config.api_key
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

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": config.request_timeout_seconds}
    base_url = config.base_url
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def request_llm_answer(prompt_text: str, user_text: str, config: SurfaceRuntimeConfig, temperature: float) -> tuple[str, dict[str, Any], int]:
    client = get_openai_client(config)
    request_started = time.perf_counter()
    last_error: Exception | None = None
    max_attempts = 1 + max(0, config.max_retries)
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=config.model_name,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": user_text},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            return (
                answer,
                {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                },
                attempt - 1,
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(config.retry_delay_seconds)
                continue

    prompt_length = len(prompt_text) + len(user_text)
    raise SurfaceGenerationRuntimeError(
        f"Surface generation request failed: {last_error}",
        prompt_text=prompt_text + "\n\n" + user_text,
        prompt_length=prompt_length,
        retry_count=max(0, max_attempts - 1),
        latency_ms=int((time.perf_counter() - request_started) * 1000),
    )


def generate_surface_answer(query: str, query_type: str, planner_output: dict[str, Any], config: SurfaceRuntimeConfig) -> dict[str, Any]:
    prompt_text = load_prompt(config.prompt_path)
    payload = build_surface_payload(query, query_type, planner_output)
    user_text = "请严格根据以下 JSON 输入生成答案，只输出答案正文，不要解释你的做法。\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    started = time.perf_counter()
    answer, usage, retry_count = request_llm_answer(prompt_text, user_text, config=config, temperature=0.2)
    return {
        "answer": answer,
        "prompt_text": prompt_text + "\n\n" + user_text,
        "prompt_length": len(prompt_text) + len(user_text),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "retry_count": retry_count,
        "prompt_payload": payload,
        "usage": usage,
    }
