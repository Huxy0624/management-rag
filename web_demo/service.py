from __future__ import annotations

import argparse
from functools import lru_cache
from uuid import uuid4

from chat import (
    CONTEXT_CHAR_LIMIT,
    DEFAULT_COLLECTION,
    DEFAULT_DB_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_OLLAMA_URL,
    OLLAMA_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_READ_TIMEOUT_SECONDS,
    answer_single_turn_payload,
    load_collection_from_args,
)
from embedding_provider import DEFAULT_EMBEDDING_PROVIDER
from runtime.runtime_config import runtime_config_from_args
from web_demo.request_store import save_request_log


def build_demo_args() -> argparse.Namespace:
    args = argparse.Namespace(
        question=None,
        session_id=None,
        no_rag=False,
        db_dir=DEFAULT_DB_DIR,
        collection=DEFAULT_COLLECTION,
        embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
        embedding_model=None,
        top_k=DEFAULT_TOP_K if DEFAULT_TOP_K >= 5 else 5,
        llm_model=DEFAULT_LLM_MODEL,
        ollama_url=DEFAULT_OLLAMA_URL,
        ollama_read_timeout=OLLAMA_READ_TIMEOUT_SECONDS,
        ollama_connect_timeout=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        ollama_max_retries=OLLAMA_MAX_RETRIES,
        context_char_limit=CONTEXT_CHAR_LIMIT,
        debug=False,
        runtime_profile=None,
        enable_llm_surface_generation_v3=None,
        enable_control_checks=None,
        enable_rewrite_v3=None,
        enable_fallback_v21=None,
        debug_return_intermediate=None,
        surface_model=None,
        surface_prompt_path="prompts/generation_v3/answer_prompt_v3.txt",
        rewrite_prompt_path="prompts/generation_v3/answer_rewrite_prompt_v3.txt",
        surface_timeout_seconds=None,
        surface_max_retries=None,
        surface_retry_delay_seconds=2,
        surface_base_url=None,
        openai_api_key=None,
    )
    args.runtime_config = runtime_config_from_args(args)
    return args


@lru_cache(maxsize=1)
def get_base_args() -> argparse.Namespace:
    return build_demo_args()


@lru_cache(maxsize=1)
def get_collection():
    return load_collection_from_args(get_base_args())


def clone_args_for_request(session_id: str | None, debug: bool) -> argparse.Namespace:
    base = get_base_args()
    cloned = argparse.Namespace(**vars(base))
    cloned.session_id = session_id
    cloned.debug = debug
    cloned.runtime_config = runtime_config_from_args(cloned)
    return cloned


def ask_question(
    question: str,
    session_id: str | None = None,
    debug: bool = True,
    client_ip: str | None = None,
    user_mode: str = "user",
) -> dict:
    args = clone_args_for_request(session_id=session_id, debug=debug)
    collection = get_collection()
    try:
        result = answer_single_turn_payload(collection=collection, question=question, args=args)
    except Exception as exc:
        save_request_log(
            request_id=f"failed-{uuid4()}",
            question=question,
            success=False,
            session_id=session_id,
            error_message=str(exc),
            client_ip=client_ip,
            user_mode=user_mode,
        )
        raise

    debug_info = dict(result.get("debug_info", {}))
    total_latency_ms = int(dict(debug_info.get("timings_ms", {})).get("total", 0))
    save_request_log(
        request_id=str(result["request_id"]),
        question=question,
        success=True,
        session_id=str(result["session_id"]),
        selected_from=result.get("selected_from"),
        fallback_triggered=bool(result.get("fallback_triggered", False)),
        total_latency_ms=total_latency_ms,
        client_ip=client_ip,
        user_mode=user_mode,
    )
    return result
