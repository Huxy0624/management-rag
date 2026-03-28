#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import chromadb

from embedding_provider import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_MODEL,
    EmbeddingConfigError,
    EmbeddingRuntimeError,
    embed_texts,
)
from rerank import (
    DEFAULT_FINAL_TOP_K,
    RERANK_CONFIG,
    build_candidates,
    rerank_candidates,
)


DEFAULT_DB_DIR = Path("db/chroma")
DEFAULT_COLLECTION = "management_rag"


def preview_text(text: str, limit: int = 300) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def print_results(query: str, reranked_results: list[dict[str, object]]) -> None:
    if not reranked_results:
        print("No results found.")
        return

    print(f"Query: {query}")
    print(f"Top {len(reranked_results)} reranked results:\n")

    for index, item in enumerate(reranked_results, start=1):
        metadata = dict(item["metadata"])
        breakdown = dict(item["rerank_breakdown"])
        print(f"[{index}] rerank_score: {float(item['rerank_score']):.6f}")
        print(f"    source: {metadata.get('source')}")
        print(f"    chunk_id: {metadata.get('chunk_id')}")
        print(f"    vector_score: {float(breakdown['vector_score']):.6f}")
        print(f"    keyword_score: {float(breakdown['keyword_score']):.6f}")
        print(f"    definition_bonus: {float(breakdown['weighted_definition']):.6f}")
        print(f"    outline_penalty: {float(breakdown['weighted_outline_penalty']):.6f}")
        print(
            "    score_formula: "
            f"{float(breakdown['vector_score']):.6f} * {RERANK_CONFIG['vector_weight']} "
            f"+ {float(breakdown['keyword_score']):.6f} * {RERANK_CONFIG['keyword_weight']} "
            f"+ {float(breakdown['definition_score']):.6f} * {RERANK_CONFIG['definition_weight']} "
            f"+ {float(breakdown['exact_query_bonus']):.6f} * {RERANK_CONFIG['exact_query_weight']} "
            f"- {float(breakdown['outline_penalty']):.6f} * {RERANK_CONFIG['outline_penalty_weight']}"
        )
        print(f"    text: {preview_text(str(item['document']))}")
        print()


def run_query(
    collection: chromadb.Collection,
    query: str,
    embedding_provider: str,
    embedding_model: str | None,
    top_k: int,
) -> None:
    query_embedding = embed_texts(
        texts=[query],
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

    print("Rerank config:")
    for key, value in rerank_config.items():
        print(f"  {key}: {value}")
    print()

    print("Candidate score breakdown:")
    for candidate in candidates:
        reranked_candidate = rerank_candidates(query, [candidate], rerank_config)[0]
        breakdown = dict(reranked_candidate["rerank_breakdown"])
        metadata = dict(reranked_candidate["metadata"])
        print(f"source: {metadata.get('source')}")
        print(f"chunk_id: {metadata.get('chunk_id')}")
        print(f"vector_score: {float(breakdown['vector_score']):.6f}")
        print(f"keyword_score: {float(breakdown['keyword_score']):.6f}")
        print(f"definition_bonus: {float(breakdown['weighted_definition']):.6f}")
        print(f"outline_penalty: {float(breakdown['weighted_outline_penalty']):.6f}")
        print(f"final_score: {float(breakdown['final_score']):.6f}")
        print()

    reranked_results = rerank_candidates(query, candidates, rerank_config)
    print_results(query, reranked_results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the local Chroma vector database.")
    parser.add_argument("--query", help="The natural language question to search for.")
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
    parser.add_argument("--top-k", type=int, default=DEFAULT_FINAL_TOP_K, help="How many chunks to keep after rerank.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()

    if not args.db_dir.exists():
        raise FileNotFoundError(f"Chroma database directory not found: {args.db_dir}")

    chroma_client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = chroma_client.get_collection(name=args.collection)

    if args.query:
        run_query(
            collection=collection,
            query=args.query,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
        )
        return

    print("Interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not query:
            continue

        run_query(
            collection=collection,
            query=query,
            embedding_provider=args.embedding_provider,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
        )


def main() -> None:
    try:
        run()
    except (EmbeddingConfigError, EmbeddingRuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
