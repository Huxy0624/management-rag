#!/usr/bin/env python
"""
Rebuild db/chroma with OpenAI embeddings only (no torch / sentence-transformers).

Matches web_demo defaults: data/raw -> db/chroma, collection management_rag,
same embedding stack as embedding_provider.DEFAULT_OPENAI_MODEL when model unset.

Prerequisite: OPENAI_API_KEY set in the environment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_chroma import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_DB_DIR,
    DEFAULT_INPUT_DIR,
    build_records,
    embed_and_store,
    validate_environment,
)
from embedding_provider import DEFAULT_OPENAI_MODEL  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot production Chroma rebuild using OpenAI embeddings (Render / demo compatible).",
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / DEFAULT_INPUT_DIR, help="Markdown/txt corpus root.")
    parser.add_argument("--db-dir", type=Path, default=ROOT / DEFAULT_DB_DIR, help="Chroma persistent directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name (must match chat.DEFAULT_COLLECTION).")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"OpenAI embedding model (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Upsert into existing collection instead of deleting it first.",
    )
    args = parser.parse_args()

    validate_environment(args.input_dir, args.chunk_size, args.chunk_overlap, "openai")
    records = build_records(args.input_dir, args.chunk_size, args.chunk_overlap)
    if not records:
        print(f"ERROR: No .md/.txt chunks under {args.input_dir}", file=sys.stderr)
        return 2

    unique_sources = sorted({record.metadata["source"] for record in records})
    print(f"Embedding {len(records)} chunks from {len(unique_sources)} files with OpenAI...")
    embed_and_store(
        records=records,
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedding_provider="openai",
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        reset=not args.no_reset,
    )
    print(f"OK: {args.db_dir} collection={args.collection} provider=openai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
