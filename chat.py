#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import chromadb
import requests

from db_utils import (
    DEFAULT_DB_PATH as DEFAULT_SESSION_DB_PATH,
    add_generation_log,
    add_message,
    add_retrieval_log,
    create_session,
    get_connection,
)
from embedding_provider import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_MODEL,
    EmbeddingConfigError,
    EmbeddingRuntimeError,
    embed_texts,
)
from rerank import RERANK_CONFIG, build_candidates, rerank_candidates
from runtime.answer_selector import select_answer
from runtime.llm_surface_runtime import (
    SurfaceGenerationConfigError,
    SurfaceGenerationRuntimeError,
    generate_surface_answer,
)
from runtime.planner_runtime import build_planner_context
from runtime.runtime_config import runtime_config_from_args


DEFAULT_DB_DIR = Path("db/chroma")
DEFAULT_COLLECTION = "management_rag"
DEFAULT_LLM_MODEL = "qwen2.5:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_TOP_K = 2
OLLAMA_CONNECT_TIMEOUT_SECONDS = 10
OLLAMA_READ_TIMEOUT_SECONDS = 120
OLLAMA_MAX_RETRIES = 2
OLLAMA_RETRY_DELAY_SECONDS = 4
CONTEXT_CHAR_LIMIT = 2500


class LLMConfigError(Exception):
    """Raised when the LLM client configuration is invalid."""


class LLMRuntimeError(Exception):
    """Raised when the LLM request fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_text: str = "",
        prompt_length: int = 0,
        context_length: int = 0,
        retry_count: int = 0,
        latency_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.prompt_text = prompt_text
        self.prompt_length = prompt_length
        self.context_length = context_length
        self.retry_count = retry_count
        self.latency_ms = latency_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG chat over the local Chroma database.")
    parser.add_argument("--question", help="Ask a single question without interactive mode.")
    parser.add_argument("--session-id", help="Reuse an existing session ID. If omitted, a new session is created.")
    parser.add_argument("--no-rag", action="store_true", help="Skip retrieval and send only the raw question to Ollama.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Persistent Chroma database directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        choices=["local", "openai"],
        help="Embedding provider. Default is local.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Embedding model name. Local default: {DEFAULT_LOCAL_MODEL}",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to use for answering.")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="LLM model name.")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama generate API endpoint.",
    )
    parser.add_argument(
        "--ollama-read-timeout",
        type=int,
        default=OLLAMA_READ_TIMEOUT_SECONDS,
        help="Ollama read timeout in seconds.",
    )
    parser.add_argument(
        "--ollama-connect-timeout",
        type=int,
        default=OLLAMA_CONNECT_TIMEOUT_SECONDS,
        help="Ollama connect timeout in seconds.",
    )
    parser.add_argument(
        "--ollama-max-retries",
        type=int,
        default=OLLAMA_MAX_RETRIES,
        help="How many times to try the Ollama request before failing.",
    )
    parser.add_argument(
        "--context-char-limit",
        type=int,
        default=CONTEXT_CHAR_LIMIT,
        help="Maximum number of context characters to include in the prompt.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print Ollama request diagnostics.",
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["local_dev", "staging", "production"],
        help="Runtime profile for v3 defaults. Falls back to GENERATION_RUNTIME_PROFILE.",
    )
    parser.add_argument(
        "--enable-llm-surface-generation-v3",
        dest="enable_llm_surface_generation_v3",
        action="store_true",
        help="Enable planner_v21 -> llm_v3 -> control/rewrite/fallback runtime.",
    )
    parser.add_argument(
        "--disable-llm-surface-generation-v3",
        dest="enable_llm_surface_generation_v3",
        action="store_false",
        help="Disable planner-based v3 runtime and keep the legacy Ollama path.",
    )
    parser.set_defaults(enable_llm_surface_generation_v3=None)
    parser.add_argument(
        "--enable-control-checks",
        dest="enable_control_checks",
        action="store_true",
        help="Enable mechanism-name and how-structure control checks.",
    )
    parser.add_argument(
        "--disable-control-checks",
        dest="enable_control_checks",
        action="store_false",
        help="Disable post-generation control checks.",
    )
    parser.set_defaults(enable_control_checks=None)
    parser.add_argument(
        "--enable-rewrite-v3",
        dest="enable_rewrite_v3",
        action="store_true",
        help="Enable rewrite_v3 when control checks fail.",
    )
    parser.add_argument(
        "--disable-rewrite-v3",
        dest="enable_rewrite_v3",
        action="store_false",
        help="Disable rewrite_v3.",
    )
    parser.set_defaults(enable_rewrite_v3=None)
    parser.add_argument(
        "--enable-fallback-v21",
        dest="enable_fallback_v21",
        action="store_true",
        help="Enable fallback_v21 as the final fuse.",
    )
    parser.add_argument(
        "--disable-fallback-v21",
        dest="enable_fallback_v21",
        action="store_false",
        help="Disable fallback_v21.",
    )
    parser.set_defaults(enable_fallback_v21=None)
    parser.add_argument(
        "--debug-return-intermediate",
        dest="debug_return_intermediate",
        action="store_true",
        help="Return planner/router/selection debug fields in CLI debug mode.",
    )
    parser.add_argument(
        "--no-debug-return-intermediate",
        dest="debug_return_intermediate",
        action="store_false",
        help="Disable intermediate debug field output even if the profile enables it.",
    )
    parser.set_defaults(debug_return_intermediate=None)
    parser.add_argument(
        "--surface-model",
        default=None,
        help="Surface generation model for the v3 runtime. Falls back to OPENAI_MODEL.",
    )
    parser.add_argument(
        "--surface-prompt-path",
        type=Path,
        default=Path("prompts/generation_v3/answer_prompt_v3.txt"),
        help="Prompt path for the v3 surface generation runtime.",
    )
    parser.add_argument(
        "--rewrite-prompt-path",
        type=Path,
        default=Path("prompts/generation_v3/answer_rewrite_prompt_v3.txt"),
        help="Prompt path for the v3 rewrite runtime.",
    )
    parser.add_argument(
        "--surface-timeout-seconds",
        type=int,
        default=None,
        help="Timeout in seconds for surface and rewrite LLM requests. Falls back to OPENAI_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--surface-max-retries",
        type=int,
        default=None,
        help="Retry count for surface and rewrite LLM requests. Falls back to OPENAI_MAX_RETRIES.",
    )
    parser.add_argument(
        "--surface-retry-delay-seconds",
        type=int,
        default=2,
        help="Delay in seconds between surface/rewrite retries.",
    )
    parser.add_argument(
        "--surface-base-url",
        help="Optional OpenAI-compatible base URL for the v3 runtime.",
    )
    parser.add_argument(
        "--openai-api-key",
        help="Optional OpenAI API key override. Falls back to OPENAI_API_KEY.",
    )
    return parser.parse_args()


def preview_text(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def retrieve_context(
    collection: chromadb.Collection,
    question: str,
    embedding_provider: str,
    embedding_model: str | None,
    top_k: int,
) -> list[dict[str, object]]:
    query_embedding = embed_texts(
        texts=[question],
        provider=embedding_provider,
        model_name=embedding_model,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(int(RERANK_CONFIG["recall_k"]), top_k),
        include=["documents", "metadatas", "distances"],
    )

    candidates = build_candidates(results)
    rerank_config = dict(RERANK_CONFIG)
    rerank_config["final_top_k"] = top_k
    return rerank_candidates(question, candidates, rerank_config)


def build_context_with_limit(
    chunks: list[dict[str, object]],
    char_limit: int,
) -> tuple[str, int, int, list[dict[str, object]]]:
    parts: list[str] = []
    used_chunks: list[dict[str, object]] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = dict(chunk["metadata"])
        chunk_text = "\n".join(
            [
                f"[Chunk {index}]",
                f"source: {metadata.get('source')}",
                f"chunk_id: {metadata.get('chunk_id')}",
                f"title: {metadata.get('title')}",
                "content:",
                str(chunk["document"]),
            ]
        )
        candidate_text = "\n\n".join(parts + [chunk_text]) if parts else chunk_text
        if len(candidate_text) <= char_limit:
            parts.append(chunk_text)
            used_chunks.append(chunk)
            continue

        remaining = char_limit - (len("\n\n".join(parts)) + (2 if parts else 0))
        if remaining > 0:
            parts.append(chunk_text[:remaining].rstrip())
            used_chunks.append(chunk)
        break

    final_context = "\n\n".join(parts)
    original_context = "\n\n".join(
        [
            "\n".join(
                [
                    f"[Chunk {index}]",
                    f"source: {dict(chunk['metadata']).get('source')}",
                    f"chunk_id: {dict(chunk['metadata']).get('chunk_id')}",
                    f"title: {dict(chunk['metadata']).get('title')}",
                    "content:",
                    str(chunk["document"]),
                ]
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
    )
    return final_context, len(original_context), len(final_context), used_chunks


def format_sources(chunks: list[dict[str, object]]) -> str:
    seen: set[tuple[str, object]] = set()
    lines: list[str] = []
    for chunk in chunks:
        metadata = dict(chunk["metadata"])
        source = str(metadata.get("source"))
        chunk_id = metadata.get("chunk_id")
        key = (source, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {source} + {chunk_id}")
    return "\n".join(lines)


def build_retrieved_items_json(chunks: list[dict[str, object]]) -> str:
    items: list[dict[str, object]] = []
    for chunk in chunks:
        metadata = dict(chunk["metadata"])
        breakdown = dict(chunk.get("rerank_breakdown", {}))
        items.append(
            {
                "source": metadata.get("source"),
                "chunk_id": metadata.get("chunk_id"),
                "vector_score": breakdown.get("vector_score"),
                "rerank_score": chunk.get("rerank_score"),
            }
        )
    return json.dumps(items, ensure_ascii=False)


def build_source_items(chunks: list[dict[str, object]], limit: int = 5) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for chunk in chunks[:limit]:
        metadata = dict(chunk["metadata"])
        items.append(
            {
                "source": str(metadata.get("source", "")),
                "chunk_id": metadata.get("chunk_id"),
                "title": str(metadata.get("title", "")),
                "rerank_score": round(float(chunk.get("rerank_score", 0.0)), 4),
                "vector_score": round(float(dict(chunk.get("rerank_breakdown", {})).get("vector_score", 0.0)), 4),
                "preview": preview_text(str(chunk.get("document", "")), limit=260),
            }
        )
    return items


def session_exists(session_id: str, db_path: Path = DEFAULT_SESSION_DB_PATH) -> bool:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
    return row is not None


def get_next_turn_index(session_id: str, db_path: Path = DEFAULT_SESSION_DB_PATH) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) AS max_turn_index FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(row["max_turn_index"]) + 1


def ensure_session_id(
    session_id: str | None,
    question: str,
    db_path: Path = DEFAULT_SESSION_DB_PATH,
) -> str:
    if session_id:
        if not session_exists(session_id, db_path=db_path):
            raise ValueError(f"Session not found: {session_id}")
        return session_id

    title = question.strip()[:50] or "New Session"
    return create_session(title=title, db_path=db_path)


def load_collection_from_args(args: argparse.Namespace) -> chromadb.Collection | None:
    if args.no_rag:
        return None
    if not args.db_dir.exists():
        raise FileNotFoundError(f"Chroma database directory not found: {args.db_dir}")
    chroma_client = chromadb.PersistentClient(path=str(args.db_dir))
    return chroma_client.get_collection(name=args.collection)


def _runtime_debug_payload(planner_context: dict[str, object], selection_result: dict[str, object]) -> dict[str, object]:
    final_checks = dict(selection_result.get("final_control_checks", {}))
    return {
        "query_type": planner_context.get("query_type"),
        "router_decision": planner_context.get("router_decision"),
        "selected_evidence": planner_context.get("selected_evidence"),
        "planner_output_v21": planner_context.get("planner_output_v21"),
        "action_translator_output": planner_context.get("action_translator_output"),
        "mechanism_mapper_output": planner_context.get("mechanism_mapper_output"),
        "llm_answer": selection_result.get("llm_answer"),
        "final_selected_answer": selection_result.get("final_selected_answer"),
        "selected_from": selection_result.get("selected_from"),
        "mechanism_name_check_pass": final_checks.get("mechanism_name_check_pass"),
        "action_steps_match_count": final_checks.get("action_steps_match_count"),
        "rewrite_triggered": selection_result.get("rewrite_triggered", False),
        "fallback_triggered": selection_result.get("fallback_triggered", False),
    }


def _runtime_log_metadata(
    planner_context: dict[str, object],
    selection_result: dict[str, object],
    runtime_profile: str,
    error_message: str | None = None,
) -> dict[str, object]:
    final_checks = dict(selection_result.get("final_control_checks", {}))
    initial_checks = dict(selection_result.get("initial_control_checks", {}))
    metadata: dict[str, object] = {
        "runtime_path": "planner_v21_llm_v3",
        "runtime_profile": runtime_profile,
        "query_type": planner_context.get("query_type"),
        "selected_from": selection_result.get("selected_from"),
        "rewrite_triggered": selection_result.get("rewrite_triggered", False),
        "fallback_triggered": selection_result.get("fallback_triggered", False),
        "mechanism_name_check_pass": final_checks.get("mechanism_name_check_pass"),
        "action_steps_match_count": final_checks.get("action_steps_match_count"),
        "structure_check_pass": final_checks.get("structure_check_pass"),
        "initial_control_checks": initial_checks,
        "final_control_checks": final_checks,
        "control_failure_reasons": {
            "missing_mechanism_names": final_checks.get("missing_mechanism_names", []),
            "structure_failed": not bool(final_checks.get("structure_check_pass", True)),
        },
    }
    if error_message:
        metadata["runtime_error"] = error_message
    return metadata


def _as_llm_runtime_error(exc: Exception, *, context_length: int = 0) -> LLMRuntimeError:
    prompt_text = str(getattr(exc, "prompt_text", ""))
    prompt_length = int(getattr(exc, "prompt_length", len(prompt_text) if prompt_text else 0))
    retry_count = int(getattr(exc, "retry_count", 0))
    latency_ms = getattr(exc, "latency_ms", None)
    return LLMRuntimeError(
        str(exc),
        prompt_text=prompt_text,
        prompt_length=prompt_length,
        context_length=context_length,
        retry_count=retry_count,
        latency_ms=latency_ms if isinstance(latency_ms, int) else None,
    )


def generate_answer_v3_runtime(
    question: str,
    chunks: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    if args.no_rag:
        raise ValueError("The v3 runtime requires retrieval results and does not support --no-rag.")

    runtime_config = args.runtime_config
    total_started = time.perf_counter()
    planner_started = time.perf_counter()
    planner_context = build_planner_context(question, chunks, runtime_config.planner)
    planner_latency_ms = int((time.perf_counter() - planner_started) * 1000)
    context_length = sum(len(str(chunk["document"])) for chunk in chunks)

    try:
        llm_result = generate_surface_answer(
            query=question,
            query_type=str(planner_context["query_type"]),
            planner_output=dict(planner_context["planner_output_v21"]),
            config=runtime_config.surface,
        )
        selection_result = select_answer(planner_context, llm_result, runtime_config)
    except (SurfaceGenerationConfigError, SurfaceGenerationRuntimeError) as exc:
        if runtime_config.feature_flags.enable_fallback_v21:
            fallback_answer = str(planner_context["v21_final_answer"])
            selection_result = {
                "llm_answer": "",
                "rewrite_answer": None,
                "final_selected_answer": fallback_answer,
                "selected_from": "fallback_v21",
                "initial_control_checks": {},
                "final_control_checks": {},
                "rewrite_triggered": False,
                "fallback_triggered": True,
                "prompt_trace": {},
                "latency_ms": getattr(exc, "latency_ms", None) or 0,
                "retry_count": getattr(exc, "retry_count", None) or 0,
                "runtime_error": str(exc),
            }
            debug_payload = _runtime_debug_payload(planner_context, selection_result)
            return {
                "answer": fallback_answer,
                "prompt_text": prompt_text if (prompt_text := getattr(exc, "prompt_text", "")) else json.dumps({"prompt_trace": {}}, ensure_ascii=False),
                "prompt_length": int(getattr(exc, "prompt_length", 0) or 0),
                "context_length": context_length,
                "retry_count": int(getattr(exc, "retry_count", 0) or 0),
                "latency_ms": int(getattr(exc, "latency_ms", 0) or 0),
                "provider": "openai",
                "model_name": runtime_config.surface.model_name,
                "debug_payload": debug_payload,
                "planner_context": planner_context,
                "timings_ms": {
                    "planner": planner_latency_ms,
                    "generation": int(getattr(exc, "latency_ms", 0) or 0),
                    "rewrite": 0,
                    "total": int((time.perf_counter() - total_started) * 1000),
                },
                "log_metadata": _runtime_log_metadata(
                    planner_context,
                    selection_result,
                    runtime_config.metadata.profile_name,
                    error_message=str(exc),
                ),
                "used_chunks": chunks,
            }
        raise _as_llm_runtime_error(exc, context_length=context_length)

    prompt_trace = {
        "surface_prompt_payload": selection_result.get("prompt_trace", {}).get("surface_prompt_payload"),
        "rewrite_prompt_payload": selection_result.get("prompt_trace", {}).get("rewrite_prompt_payload"),
    }
    debug_payload = _runtime_debug_payload(planner_context, selection_result)
    return {
        "answer": str(selection_result["final_selected_answer"]),
        "prompt_text": json.dumps(prompt_trace, ensure_ascii=False, indent=2),
        "prompt_length": len(json.dumps(prompt_trace, ensure_ascii=False)),
        "context_length": context_length,
        "retry_count": int(selection_result.get("retry_count", 0)),
        "latency_ms": int(selection_result.get("latency_ms", 0)),
        "provider": "openai",
        "model_name": runtime_config.surface.model_name,
        "debug_payload": debug_payload,
        "planner_context": planner_context,
        "timings_ms": {
            "planner": planner_latency_ms,
            "generation": int(selection_result.get("timings_ms", {}).get("generation", 0)),
            "rewrite": int(selection_result.get("timings_ms", {}).get("rewrite", 0)),
            "total": int((time.perf_counter() - total_started) * 1000),
        },
        "log_metadata": _runtime_log_metadata(
            planner_context,
            selection_result,
            runtime_config.metadata.profile_name,
        ),
        "used_chunks": chunks,
    }


def answer_single_turn_payload(
    collection: chromadb.Collection | None,
    question: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    runtime_config = args.runtime_config
    request_started = time.perf_counter()
    request_id = str(uuid4())
    session_id = ensure_session_id(args.session_id, question)
    turn_index = get_next_turn_index(session_id)
    user_message_id = add_message(
        session_id=session_id,
        role="user",
        content=question,
        turn_index=turn_index,
    )

    retrieval_started = time.perf_counter()
    chunks: list[dict[str, object]] = []
    if not args.no_rag:
        if collection is None:
            raise ValueError("Collection is not available while RAG mode is enabled.")
        chunks = retrieve_context(
            collection=collection,
            question=question,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
        )
    retrieval_latency_ms = int((time.perf_counter() - retrieval_started) * 1000)

    if not args.no_rag:
        context_length = sum(len(str(chunk["document"])) for chunk in chunks)
        add_retrieval_log(
            session_id=session_id,
            message_id=user_message_id,
            raw_query=question,
            retrieval_query=question,
            top_k=args.top_k,
            retrieved_items_json=build_retrieved_items_json(chunks),
            context_length=context_length,
            rerank_applied=True,
        )

    try:
        if runtime_config.feature_flags.enable_llm_surface_generation_v3:
            generation_result = generate_answer_v3_runtime(question=question, chunks=chunks, args=args)
        else:
            generation_result = generate_answer(
                question=question,
                chunks=chunks,
                llm_model=args.llm_model,
                ollama_url=args.ollama_url,
                ollama_connect_timeout=args.ollama_connect_timeout,
                ollama_read_timeout=args.ollama_read_timeout,
                ollama_max_retries=args.ollama_max_retries,
                no_rag=args.no_rag,
                context_char_limit=args.context_char_limit,
                debug=args.debug,
            )
    except LLMRuntimeError as exc:
        add_generation_log(
            session_id=session_id,
            message_id=user_message_id,
            provider="openai" if runtime_config.feature_flags.enable_llm_surface_generation_v3 else "ollama",
            model_name=runtime_config.surface.model_name
            if runtime_config.feature_flags.enable_llm_surface_generation_v3
            else args.llm_model,
            prompt_text=exc.prompt_text,
            prompt_length=exc.prompt_length,
            answer=None,
            latency_ms=exc.latency_ms,
            retry_count=exc.retry_count,
            success=False,
            error_message=str(exc),
            metadata_json={
                "runtime_path": "planner_v21_llm_v3"
                if runtime_config.feature_flags.enable_llm_surface_generation_v3
                else "legacy_ollama",
                "runtime_profile": runtime_config.metadata.profile_name,
                "request_id": request_id,
            },
        )
        raise

    answer = str(generation_result["answer"])
    add_generation_log(
        session_id=session_id,
        message_id=user_message_id,
        provider=str(generation_result.get("provider", "ollama")),
        model_name=str(generation_result.get("model_name", args.llm_model)),
        prompt_text=str(generation_result["prompt_text"]),
        prompt_length=int(generation_result["prompt_length"]),
        answer=answer,
        latency_ms=int(generation_result["latency_ms"]),
        retry_count=int(generation_result["retry_count"]),
        success=True,
        error_message=None,
        metadata_json={
            **dict(generation_result.get("log_metadata", {})),
            "request_id": request_id,
        },
    )

    assistant_message_id = add_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        turn_index=turn_index + 1,
    )

    runtime_timings = dict(generation_result.get("timings_ms", {}))
    planner_latency_ms = int(runtime_timings.get("planner", 0))
    generation_latency_ms = int(runtime_timings.get("generation", int(generation_result.get("latency_ms", 0))))
    rewrite_latency_ms = int(runtime_timings.get("rewrite", 0))
    total_latency_ms = int(runtime_timings.get("total", 0)) or int((time.perf_counter() - request_started) * 1000)

    debug_info: dict[str, object] = dict(generation_result.get("debug_payload", {}))
    debug_info.update(
        {
            "request_id": request_id,
            "session_id": session_id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "query": question,
            "retrieval_query": question,
            "timings_ms": {
                "retrieval": retrieval_latency_ms,
                "planner": planner_latency_ms,
                "generation": generation_latency_ms,
                "rewrite": rewrite_latency_ms,
                "total": total_latency_ms,
            },
        }
    )

    return {
        "request_id": request_id,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "sources": build_source_items(chunks, limit=args.top_k),
        "source_text": format_sources(chunks),
        "debug_info": debug_info,
        "selected_from": debug_info.get("selected_from"),
        "fallback_triggered": bool(debug_info.get("fallback_triggered", False)),
    }


def generate_answer(
    question: str,
    chunks: list[dict[str, object]],
    llm_model: str,
    ollama_url: str,
    ollama_connect_timeout: int,
    ollama_read_timeout: int,
    ollama_max_retries: int,
    no_rag: bool,
    context_char_limit: int,
    debug: bool,
) -> dict[str, object]:
    request_started = time.perf_counter()
    if no_rag:
        prompt = question
        original_context_length = 0
        final_context_length = 0
        used_chunks: list[dict[str, object]] = []
    else:
        context_text, original_context_length, final_context_length, used_chunks = build_context_with_limit(
            chunks=chunks,
            char_limit=context_char_limit,
        )
        source_text = format_sources(used_chunks)

        system_prompt = (
            "你是一个RAG问答助手。"
            "你必须优先依据提供的检索片段回答，不要脱离上下文编造事实。"
            "回答必须严格分成三部分：\n"
            "第一部分：直接回答问题\n"
            "第二部分：补充解释\n"
            "第三部分：来源\n"
            "来源格式必须是：\n"
            "来源：\n"
            "- 文件名 + chunk_id"
        )
        user_prompt = (
            f"问题：\n{question}\n\n"
            f"可用检索片段：\n{context_text}\n\n"
            f"请基于以上片段回答。可引用的来源如下：\n{source_text}"
        )
        prompt = f"{system_prompt}\n\n{user_prompt}"

    if debug:
        print(f"[debug] no_rag={no_rag}")
        print(f"[debug] model={llm_model}")
        print(f"[debug] retrieved_chunks={len(chunks)}")
        print(f"[debug] original_context_length={original_context_length}")
        print(f"[debug] final_context_length={final_context_length}")
        print(f"[debug] used_chunks_in_prompt={len(used_chunks)}")
        for chunk in used_chunks:
            metadata = dict(chunk["metadata"])
            print(f"[debug] chunk={metadata.get('source')} + {metadata.get('chunk_id')}")
        print(f"[debug] prompt_length={len(prompt)}")
        print("[debug] prompt_tail_preview:")
        print(prompt[-500:])
        print()

    payload = {
        "model": llm_model,
        "prompt": prompt,
        "stream": False,
    }

    max_attempts = max(1, ollama_max_retries)
    response_data: dict[str, object] | None = None

    for attempt in range(1, max_attempts + 1):
        if debug:
            print(
                f"[debug] Ollama request attempt {attempt}/{max_attempts} | "
                f"url={ollama_url} | timeout=({ollama_connect_timeout}, {ollama_read_timeout})"
            )

        try:
            response = requests.post(
                ollama_url,
                json=payload,
                timeout=(ollama_connect_timeout, ollama_read_timeout),
            )
        except requests.exceptions.Timeout:
            if attempt < max_attempts:
                if debug:
                    print(
                        f"[debug] Ollama request timed out on attempt {attempt}. "
                        f"Retrying in {OLLAMA_RETRY_DELAY_SECONDS}s..."
                    )
                time.sleep(OLLAMA_RETRY_DELAY_SECONDS)
                continue
            raise LLMRuntimeError(
                "Ollama request timed out.\n"
                "The model may still be loading. Please retry in a moment.",
                prompt_text=prompt,
                prompt_length=len(prompt),
                context_length=final_context_length,
                retry_count=attempt - 1,
                latency_ms=int((time.perf_counter() - request_started) * 1000),
            )
        except requests.exceptions.ConnectionError:
            raise LLMRuntimeError(
                "Cannot connect to Ollama.\n"
                "Please make sure Ollama is running and the endpoint is reachable:\n"
                f"{ollama_url}",
                prompt_text=prompt,
                prompt_length=len(prompt),
                context_length=final_context_length,
                retry_count=attempt - 1,
                latency_ms=int((time.perf_counter() - request_started) * 1000),
            )
        except requests.RequestException as exc:
            raise LLMRuntimeError(
                "Ollama request failed.\n"
                "Please check your model name, Ollama service status, or endpoint.\n"
                f"Original error: {exc}",
                prompt_text=prompt,
                prompt_length=len(prompt),
                context_length=final_context_length,
                retry_count=attempt - 1,
                latency_ms=int((time.perf_counter() - request_started) * 1000),
            )

        if debug:
            print(f"[debug] Ollama HTTP status={response.status_code}")

        if response.status_code != 200:
            raise LLMRuntimeError(
                "Ollama request failed.\n"
                f"HTTP status: {response.status_code}\n"
                f"Response: {response.text}",
                prompt_text=prompt,
                prompt_length=len(prompt),
                context_length=final_context_length,
                retry_count=attempt - 1,
                latency_ms=int((time.perf_counter() - request_started) * 1000),
            )

        try:
            response_data = response.json()
        except ValueError:
            raise LLMRuntimeError(
                "Ollama returned a non-JSON response.\n"
                f"Raw response: {response.text}",
                prompt_text=prompt,
                prompt_length=len(prompt),
                context_length=final_context_length,
                retry_count=attempt - 1,
                latency_ms=int((time.perf_counter() - request_started) * 1000),
            )
        break

    if response_data is None:
        raise LLMRuntimeError(
            "Ollama request failed without a response.",
            prompt_text=prompt,
            prompt_length=len(prompt),
            context_length=final_context_length,
            retry_count=max_attempts - 1,
            latency_ms=int((time.perf_counter() - request_started) * 1000),
        )

    content = response_data.get("response")
    if not content:
        raise LLMRuntimeError(
            "Ollama returned an empty response.",
            prompt_text=prompt,
            prompt_length=len(prompt),
            context_length=final_context_length,
            retry_count=max_attempts - 1,
            latency_ms=int((time.perf_counter() - request_started) * 1000),
        )
    return {
        "answer": content.strip(),
        "prompt_text": prompt,
        "prompt_length": len(prompt),
        "context_length": final_context_length,
        "retry_count": attempt - 1,
        "latency_ms": int((time.perf_counter() - request_started) * 1000),
    }


def run_single_turn(
    collection: chromadb.Collection | None,
    question: str,
    args: argparse.Namespace,
) -> str:
    runtime_config = args.runtime_config
    payload = answer_single_turn_payload(collection=collection, question=question, args=args)
    session_id = str(payload["session_id"])
    answer = str(payload["answer"])
    sources = str(payload["source_text"])
    print(f"Ask: {question}")
    print("Answer:")
    print(answer)
    if args.debug:
        print(f"[debug] top_k={args.top_k}")
        print(f"[debug] retrieved_chunk_count={len(payload['sources'])}")
        for item in payload["sources"]:
            print(f"[debug] retrieved={item['source']} + {item['chunk_id']}")
        print()
    if args.debug and runtime_config.feature_flags.debug_return_intermediate:
        print("[debug] generation_runtime:")
        print(json.dumps(payload["debug_info"], ensure_ascii=False, indent=2))
    print("Sources:")
    print(sources)
    return session_id


def run() -> None:
    args = parse_args()
    args.runtime_config = runtime_config_from_args(args)

    if not DEFAULT_SESSION_DB_PATH.exists():
        raise FileNotFoundError(
            f"Session database not found: {DEFAULT_SESSION_DB_PATH}. "
            "Please run `python init_db.py` first."
        )

    collection: chromadb.Collection | None = None
    if not args.no_rag:
        collection = load_collection_from_args(args)

    if args.debug:
        print(f"[debug] runtime_profile={args.runtime_config.metadata.profile_name}")
        print(f"[debug] env_overrides={json.dumps(args.runtime_config.metadata.env_overrides, ensure_ascii=False)}")
        print(f"[debug] cli_overrides={json.dumps(args.runtime_config.metadata.cli_overrides, ensure_ascii=False)}")

    if args.question:
        args.session_id = run_single_turn(collection, args.question, args)
        return

    print("Interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        question = input("Ask: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not question:
            continue

        args.session_id = run_single_turn(collection, question, args)
        print()


def main() -> None:
    try:
        run()
    except (
        EmbeddingConfigError,
        EmbeddingRuntimeError,
        LLMConfigError,
        LLMRuntimeError,
        SurfaceGenerationConfigError,
        SurfaceGenerationRuntimeError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
