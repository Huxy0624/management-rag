from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str | None = None
    debug: bool = True
    debug_token: str | None = None


class AskResponse(BaseModel):
    request_id: str
    session_id: str
    question: str
    answer: str
    sources: list[dict[str, Any]]
    debug_info: dict[str, Any]
    generation_trace: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_query: str | None = None
    retrieval_count: int = 0
    retrieval_latency_ms: int = 0
    retrieval_backend: str | None = None
    retrieval_status: str | None = None
    retrieval_reason: str | None = None
    output_language: str = "zh"
    selected_from: str | None = None
    fallback_triggered: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None


class RetrievalDiagResponse(BaseModel):
    ok: bool
    retrieval_query: str
    chunk_count: int
    latency_ms: int
    backend: str
    chunks: list[dict[str, Any]]
    chroma_unavailable_reason: str | None = None


class HealthResponse(BaseModel):
    status: str
    runtime_profile: str
    v3_enabled: bool
    model_name: str


class ConfigResponse(BaseModel):
    title: str
    tagline: str
    description: str
    examples: list[str]
    debug_mode_enabled: bool
    show_debug_toggle: bool
    input_placeholder: str
    answer_title: str
    feedback_prompt: str
    debug_title: str
    debug_description: str
    admin_enabled: bool


class FeedbackRequest(BaseModel):
    request_id: str
    session_id: str | None = None
    rating: str = Field(pattern="^(up|down)$")
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    ok: bool
    feedback_id: str


class AdminRequestsResponse(BaseModel):
    items: list[dict[str, Any]]
