from __future__ import annotations

from pathlib import Path
import logging
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web_demo.admin_service import load_recent_requests
from web_demo.config import get_demo_config
from web_demo.feedback_store import save_feedback
from embedding_provider import EmbeddingRuntimeError

from chat import RetrievalInfrastructureError, chroma_installation_status, retrieve_chunks_resilient
from runtime.kb_source_policy import filter_chunks_for_user_facing
from web_demo.schemas import (
    AdminRequestsResponse,
    AskRequest,
    AskResponse,
    ConfigResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RetrievalDiagResponse,
)
from web_demo.service import ask_question, get_base_args, get_chroma_collection_state


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = BASE_DIR / "index.html"
ADMIN_FILE = BASE_DIR / "admin.html"
RATE_LIMIT_HISTORY: dict[str, deque[float]] = defaultdict(deque)
log = logging.getLogger(__name__)

app = FastAPI(title="TraceAnswer Demo", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/admin", include_in_schema=False)
def admin_page(token: str | None = None) -> FileResponse:
    if not _admin_authorized(token):
        raise HTTPException(status_code=403, detail="Admin access denied.")
    return FileResponse(ADMIN_FILE)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _debug_authorized(debug_requested: bool, debug_token: str | None) -> bool:
    config = get_demo_config()
    if not debug_requested:
        return False
    if not config.debug_token:
        return False
    return bool(debug_token) and debug_token == config.debug_token


def _admin_authorized(admin_token: str | None) -> bool:
    config = get_demo_config()
    if not config.admin_token:
        return False
    return bool(admin_token) and admin_token == config.admin_token


def _enforce_rate_limit(request: Request) -> None:
    config = get_demo_config()
    ip = _client_ip(request)
    now = time.time()
    history = RATE_LIMIT_HISTORY[ip]
    while history and now - history[0] > config.rate_limit_window_seconds:
        history.popleft()
    if len(history) >= config.rate_limit_max_requests:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment and try again.")
    history.append(now)


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    args = get_base_args()
    return HealthResponse(
        status="ok",
        runtime_profile=args.runtime_config.metadata.profile_name,
        v3_enabled=args.runtime_config.feature_flags.enable_llm_surface_generation_v3,
        model_name=args.runtime_config.surface.model_name,
    )


@app.get("/api/config", response_model=ConfigResponse)
def config(request: Request, debug: int = 0, token: str | None = None) -> ConfigResponse:
    demo_config = get_demo_config()
    debug_enabled = _debug_authorized(bool(debug), token)
    return ConfigResponse(
        title=demo_config.title,
        tagline=demo_config.tagline,
        description=demo_config.description,
        examples=demo_config.examples,
        debug_mode_enabled=debug_enabled,
        show_debug_toggle=demo_config.show_debug_toggle,
        input_placeholder=demo_config.input_placeholder,
        answer_title=demo_config.answer_title,
        feedback_prompt=demo_config.feedback_prompt,
        debug_title=demo_config.debug_title,
        debug_description=demo_config.debug_description,
        admin_enabled=bool(demo_config.admin_token),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    _enforce_rate_limit(request)
    debug_enabled = _debug_authorized(payload.debug, payload.debug_token)
    user_mode = "debug" if debug_enabled else "user"
    try:
        result = ask_question(
            question=payload.question.strip(),
            session_id=payload.session_id,
            debug=debug_enabled,
            client_ip=_client_ip(request),
            user_mode=user_mode,
        )
    except RetrievalInfrastructureError as exc:
        log.error("ask_retrieval_infrastructure: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingRuntimeError as exc:
        log.error("ask_embedding_retrieval_failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "向量检索失败：Chroma 索引与当前查询的 embedding 不一致，请用 OpenAI embedding 重建 db/chroma 或对齐配置。"
                f" 详情: {exc}"
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not debug_enabled:
        result["debug_info"] = {
            "request_id": result["request_id"],
            "session_id": result["session_id"],
            "selected_from": result.get("selected_from"),
            "generation_chain_v2_enabled": dict(result.get("debug_info", {})).get("generation_chain_v2_enabled"),
            "minimal_rag_mode": dict(result.get("debug_info", {})).get("minimal_rag_mode"),
            "fallback_triggered": result.get("fallback_triggered", False),
            "timings_ms": dict(result.get("debug_info", {}).get("timings_ms", {})),
            "retrieval_query": result.get("retrieval_query"),
            "retrieval_count": result.get("retrieval_count"),
            "retrieval_latency_ms": result.get("retrieval_latency_ms"),
            "retrieval_backend": result.get("retrieval_backend"),
            "retrieval_status": result.get("retrieval_status"),
            "retrieval_reason": result.get("retrieval_reason"),
            "output_language": result.get("output_language"),
        }
    return AskResponse(**result)


@app.get("/api/diag/chroma")
def diag_chroma() -> dict[str, object]:
    """Report whether db/chroma can be opened and which collections exist (no embedding call)."""
    args = get_base_args()
    return chroma_installation_status(args)


@app.get("/api/diag/retrieval", response_model=RetrievalDiagResponse)
def diag_retrieval(q: str, request: Request) -> RetrievalDiagResponse:
    """Run retrieval only (strict Chroma when collection is open; empty if unavailable)."""
    _enforce_rate_limit(request)
    args = get_base_args()
    if args.no_rag:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is required for retrieval.")
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter q is required.")
    collection, chroma_unavailable_reason = get_chroma_collection_state()
    try:
        chunks, latency_ms, backend = retrieve_chunks_resilient(collection, query, args)
        chunks = filter_chunks_for_user_facing(chunks)
    except RetrievalInfrastructureError as exc:
        log.error("diag_retrieval_infrastructure: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingRuntimeError as exc:
        log.error("diag_retrieval_embedding_failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "向量检索失败：索引与当前查询的 embedding 方案或维度不一致，需用与线上一致的 OpenAI embedding 重建 Chroma。"
                f" 详情: {exc}"
            ),
        ) from exc
    preview = [
        {
            "source": str(dict(c.get("metadata") or {}).get("source", "")),
            "chunk_id": dict(c.get("metadata") or {}).get("chunk_id"),
            "preview": (str(c.get("document", ""))[:400]),
            "rerank_score": c.get("rerank_score"),
        }
        for c in chunks[:10]
    ]
    return RetrievalDiagResponse(
        ok=collection is not None,
        retrieval_query=query,
        chunk_count=len(chunks),
        latency_ms=latency_ms,
        backend=backend,
        chunks=preview,
        chroma_unavailable_reason=chroma_unavailable_reason if collection is None else None,
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    feedback_id = save_feedback(
        request_id=payload.request_id,
        session_id=payload.session_id,
        rating=payload.rating,
        comment=payload.comment,
        client_ip=_client_ip(request),
    )
    return FeedbackResponse(ok=True, feedback_id=feedback_id)


@app.get("/api/admin/requests", response_model=AdminRequestsResponse)
def admin_requests(token: str | None = None, limit: int = 50) -> AdminRequestsResponse:
    if not _admin_authorized(token):
        raise HTTPException(status_code=403, detail="Admin access denied.")
    return AdminRequestsResponse(items=load_recent_requests(limit=min(max(limit, 1), 100)))
