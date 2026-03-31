from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from copy import deepcopy

from rerank import RERANK_CONFIG, rerank_candidates

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD_KB_PATH = Path("data/kb_chunks.jsonl")


def kb_jsonl_ready(path: Path | None = None) -> bool:
    """True if the keyword KB file exists and is non-empty (deployment smoke check)."""
    p = path or DEFAULT_KEYWORD_KB_PATH
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def _load_kb_documents(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.info("keyword_kb_missing path=%s", path)
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def keyword_retrieve_fallback(
    query: str,
    top_k: int,
    *,
    kb_path: Path | None = None,
) -> list[dict[str, object]]:
    """
    Lightweight keyword + rerank retrieval when Chroma is unavailable.
    Reads JSONL: each line {"document": "...", "metadata": {"source", "chunk_id", "title", ...}}.
    """
    path = kb_path or DEFAULT_KEYWORD_KB_PATH
    rows = _load_kb_documents(path)
    if not rows:
        return []

    candidates: list[dict[str, object]] = []
    for row in rows:
        doc = str(row.get("document") or row.get("text") or "")
        meta = dict(row.get("metadata") or {})
        if not doc.strip():
            continue
        candidates.append(
            {
                "document": doc,
                "metadata": meta,
                "distance": 0.5,
            }
        )
    if not candidates:
        return []

    cfg = deepcopy(RERANK_CONFIG)
    cfg["final_top_k"] = max(1, int(top_k))
    cfg["recall_k"] = max(len(candidates), int(cfg.get("recall_k", 15)))
    return rerank_candidates(query, candidates, cfg)
