from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Never surface these as "real KB" in user-facing RAG (generation, sources, trace).
FORBIDDEN_KB_SOURCES: frozenset[str] = frozenset({"kb_fallback"})

# Strict allowlist: if set to a non-empty frozenset, only chunks whose
# metadata["source"] (stripped) is in this set are kept.
# If None, allowlist is disabled: any non-empty source passes except FORBIDDEN_KB_SOURCES.
# Populate with your real corpus source strings (must match Chroma metadata from build_chroma).
REAL_KB_SOURCE_ALLOWLIST: frozenset[str] | None = None


def filter_chunks_for_user_facing(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Drop chunks that must not inform answers or UI (e.g. kb_fallback).
    When REAL_KB_SOURCE_ALLOWLIST is a non-empty frozenset, enforce strict membership.
    """
    before = len(chunks)
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        source = str(metadata.get("source") or "").strip()
        if not source:
            continue
        if source in FORBIDDEN_KB_SOURCES:
            continue
        if REAL_KB_SOURCE_ALLOWLIST is not None and len(REAL_KB_SOURCE_ALLOWLIST) > 0:
            if source not in REAL_KB_SOURCE_ALLOWLIST:
                continue
        out.append(chunk)
    dropped = before - len(out)
    if dropped:
        logger.info(
            "kb_source_filter dropped=%s kept=%s strict_allowlist=%s",
            dropped,
            len(out),
            bool(REAL_KB_SOURCE_ALLOWLIST),
        )
    return out
