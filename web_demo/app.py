from __future__ import annotations

from pathlib import Path
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web_demo.admin_service import load_recent_requests
from web_demo.config import get_demo_config
from web_demo.feedback_store import save_feedback
from web_demo.schemas import (
    AdminRequestsResponse,
    AskRequest,
    AskResponse,
    ConfigResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from web_demo.service import ask_question, get_base_args


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = BASE_DIR / "index.html"
ADMIN_FILE = BASE_DIR / "admin.html"
RATE_LIMIT_HISTORY: dict[str, deque[float]] = defaultdict(deque)


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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not debug_enabled:
        result["debug_info"] = {
            "request_id": result["request_id"],
            "session_id": result["session_id"],
            "selected_from": result.get("selected_from"),
            "fallback_triggered": result.get("fallback_triggered", False),
            "timings_ms": dict(result.get("debug_info", {}).get("timings_ms", {})),
        }
    return AskResponse(**result)


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
