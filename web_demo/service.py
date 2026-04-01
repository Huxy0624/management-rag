from __future__ import annotations

import argparse
import logging
import os
from typing import Any
from functools import lru_cache
from uuid import uuid4

from chat import (
    CONTEXT_CHAR_LIMIT,
    DEFAULT_COLLECTION,
    DEFAULT_DB_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    OLLAMA_CONNECT_TIMEOUT_SECONDS,
    OLLAMA_MAX_RETRIES,
    OLLAMA_READ_TIMEOUT_SECONDS,
    answer_single_turn_payload,
    load_collection_from_args,
)
from embedding_provider import DEFAULT_EMBEDDING_PROVIDER
from runtime.runtime_config import runtime_config_from_args
from web_demo.lite_fallback import no_openai_key_response
from web_demo.request_store import save_request_log

log = logging.getLogger(__name__)


def _has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def build_demo_args() -> argparse.Namespace:
    has_key = _has_openai_key()
    no_rag = not has_key

    args = argparse.Namespace(
        question=None,
        session_id=None,
        no_rag=no_rag,
        db_dir=DEFAULT_DB_DIR,
        collection=DEFAULT_COLLECTION,
        embedding_provider=DEFAULT_EMBEDDING_PROVIDER,
        embedding_model=None,
        top_k=5,
        llm_model=DEFAULT_LLM_MODEL,
        ollama_url=DEFAULT_OLLAMA_URL,
        ollama_read_timeout=OLLAMA_READ_TIMEOUT_SECONDS,
        ollama_connect_timeout=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        ollama_max_retries=OLLAMA_MAX_RETRIES,
        context_char_limit=CONTEXT_CHAR_LIMIT,
        debug=False,
        runtime_profile="minimal_rag",
        enable_generation_chain_v2=False,
        enable_llm_surface_generation_v3=True,
        enable_control_checks=None,
        enable_rewrite_v3=None,
        enable_fallback_v21=None,
        enable_failure_case_logger=None,
        failure_case_log_path=None,
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
def get_chroma_collection_state() -> tuple[Any | None, str | None]:
    """Cached (collection, reason) from load_collection_from_args; reason is always set when collection is None."""
    base = get_base_args()
    return load_collection_from_args(base)


def get_collection():
    """Return collection or None. Load reason is discarded; use get_chroma_collection_state() for diagnostics."""
    col, _reason = get_chroma_collection_state()
    return col


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
    if not _has_openai_key():
        result = no_openai_key_response(question=question, session_id=session_id, debug=debug)
        debug_info = dict(result.get("debug_info", {}))
        total_latency_ms = int(dict(debug_info.get("timings_ms", {})).get("total", 0))
        save_request_log(
            request_id=str(result["request_id"]),
            question=question,
            success=True,
            session_id=str(result["session_id"]),
            selected_from=result.get("selected_from"),
            fallback_triggered=False,
            total_latency_ms=total_latency_ms,
            client_ip=client_ip,
            user_mode=user_mode,
        )
        return result

    args = clone_args_for_request(session_id=session_id, debug=debug)
    collection, chroma_reason = get_chroma_collection_state()

    result = answer_single_turn_payload(
        collection=collection,
        question=question,
        args=args,
        chroma_load_reason=chroma_reason,
    )

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
