from __future__ import annotations

import json
import os
import time
from typing import Any
from uuid import uuid4

from chat import add_generation_log, add_message, empty_generation_trace, ensure_session_id, get_next_turn_index

NO_OPENAI_KEY_ANSWER = (
    "当前演示需要配置有效的 OPENAI_API_KEY 后才能生成回答。"
    "请在部署环境中设置该变量后重试。"
)


def no_openai_key_response(
    *,
    question: str,
    session_id: str | None,
    debug: bool,
) -> dict[str, object]:
    """Stable JSON payload when the API key is missing (no external LLM calls)."""
    request_started = time.perf_counter()
    request_id = str(uuid4())
    session_id_resolved = ensure_session_id(session_id, question)
    turn_index = get_next_turn_index(session_id_resolved)
    user_message_id = add_message(
        session_id=session_id_resolved,
        role="user",
        content=question,
        turn_index=turn_index,
    )
    assistant_message_id = add_message(
        session_id=session_id_resolved,
        role="assistant",
        content=NO_OPENAI_KEY_ANSWER,
        turn_index=turn_index + 1,
    )
    total_ms = int((time.perf_counter() - request_started) * 1000)
    trace = empty_generation_trace()
    trace["final_output"] = {"final_answer": NO_OPENAI_KEY_ANSWER}
    trace["retrieval"] = {"query": question, "top_chunks": []}
    trace["config"] = {"openai_api_key": "missing"}

    debug_info: dict[str, object] = {
        "request_id": request_id,
        "session_id": session_id_resolved,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "selected_from": "no_openai_key",
        "generation_chain_v2_enabled": False,
        "fallback_triggered": False,
        "timings_ms": {
            "retrieval": 0,
            "planner": 0,
            "generation": 0,
            "rewrite": 0,
            "total": total_ms,
        },
    }
    if debug:
        debug_info["reason"] = "OPENAI_API_KEY is not set"

    return {
        "request_id": request_id,
        "session_id": session_id_resolved,
        "question": question,
        "answer": NO_OPENAI_KEY_ANSWER,
        "sources": [],
        "source_text": "",
        "retrieved_chunks": [],
        "retrieval_query": None,
        "retrieval_count": 0,
        "retrieval_latency_ms": 0,
        "retrieval_backend": None,
        "debug_info": debug_info,
        "selected_from": "no_openai_key",
        "fallback_triggered": False,
        "needs_clarification": False,
        "clarification_question": None,
        "generation_trace": trace,
    }


def _client() -> Any:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run OpenAI completion fallback.")
    base = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base or None)


def openai_completion_fallback_response(
    *,
    question: str,
    session_id: str | None,
    prior_error: str | None,
    debug: bool,
) -> dict[str, object]:
    """Single-turn answer via chat.completions when the full RAG/v3 pipeline fails."""
    request_started = time.perf_counter()
    request_id = str(uuid4())
    session_id_resolved = ensure_session_id(session_id, question)
    turn_index = get_next_turn_index(session_id_resolved)
    user_message_id = add_message(
        session_id=session_id_resolved,
        role="user",
        content=question,
        turn_index=turn_index,
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
    client = _client()
    system = (
        "你是一名管理实践与组织协作方向的助手。"
        "请用清晰、可执行的中文回答用户问题。"
        "若缺少关键信息，可先简要说明假设再给出建议。"
    )
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_tokens=900,
            timeout=timeout,
        )
    except Exception as exc:
        add_generation_log(
            session_id=session_id_resolved,
            message_id=user_message_id,
            provider="openai",
            model_name=model,
            prompt_text=json.dumps({"system": system, "user": question}, ensure_ascii=False),
            prompt_length=len(question),
            answer=None,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            retry_count=0,
            success=False,
            error_message=str(exc),
            metadata_json={
                "runtime_path": "lite_openai_completion_fallback",
                "request_id": request_id,
                "prior_error": prior_error,
            },
        )
        raise

    answer = (response.choices[0].message.content or "").strip() or "（模型未返回文本）"
    gen_ms = int((time.perf_counter() - t0) * 1000)
    add_generation_log(
        session_id=session_id_resolved,
        message_id=user_message_id,
        provider="openai",
        model_name=model,
        prompt_text=json.dumps({"system": system, "user": question}, ensure_ascii=False),
        prompt_length=len(question),
        answer=answer,
        latency_ms=gen_ms,
        retry_count=0,
        success=True,
        error_message=None,
        metadata_json={
            "runtime_path": "lite_openai_completion_fallback",
            "request_id": request_id,
            "prior_error": prior_error,
        },
    )
    assistant_message_id = add_message(
        session_id=session_id_resolved,
        role="assistant",
        content=answer,
        turn_index=turn_index + 1,
    )
    total_ms = int((time.perf_counter() - request_started) * 1000)
    trace = empty_generation_trace()
    trace["final_output"] = {"final_answer": answer}
    trace["retrieval"] = {"query": question, "top_chunks": []}
    trace["lite_fallback"] = {
        "enabled": True,
        "path": "openai_chat_completion",
        "model": model,
        "prior_pipeline_error": prior_error,
    }

    debug_info: dict[str, object] = {
        "request_id": request_id,
        "session_id": session_id_resolved,
        "user_message_id": user_message_id,
        "assistant_message_id": assistant_message_id,
        "selected_from": "lite_openai_completion_fallback",
        "generation_chain_v2_enabled": False,
        "fallback_triggered": True,
        "timings_ms": {
            "retrieval": 0,
            "planner": 0,
            "generation": gen_ms,
            "rewrite": 0,
            "total": total_ms,
        },
    }
    if debug and prior_error:
        debug_info["prior_pipeline_error"] = prior_error

    return {
        "request_id": request_id,
        "session_id": session_id_resolved,
        "question": question,
        "answer": answer,
        "sources": [],
        "source_text": "",
        "retrieved_chunks": [],
        "retrieval_query": question,
        "retrieval_count": 0,
        "retrieval_latency_ms": 0,
        "retrieval_backend": None,
        "debug_info": debug_info,
        "selected_from": "lite_openai_completion_fallback",
        "fallback_triggered": True,
        "needs_clarification": False,
        "clarification_question": None,
        "generation_trace": trace,
    }
