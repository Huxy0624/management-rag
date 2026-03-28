#!/usr/bin/env python
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


DEFAULT_RECALL_K = 15
DEFAULT_FINAL_TOP_K = 5

RERANK_CONFIG: dict[str, Any] = {
    "recall_k": DEFAULT_RECALL_K,
    "final_top_k": DEFAULT_FINAL_TOP_K,
    "vector_weight": 1.0,
    "keyword_weight": 0.5,
    "definition_weight": 0.4,
    "outline_penalty_weight": 0.3,
    "exact_query_weight": 0.2,
    "max_keyword_hits": 5,
    "keyword_hit_unit": 0.2,
    "max_definition_hits": 2,
    "definition_patterns": [
        "本质是",
        "核心是",
        "定义是",
        "可以定义为",
    ],
    "outline_source_terms": [
        "大纲",
        "小结",
        "总结",
        "回顾",
    ],
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def extract_keywords(query: str) -> list[str]:
    normalized_query = normalize_text(query)
    normalized_query = re.sub(r"^(什么是|什么叫|什么叫做|请问|如何|怎么|怎样)", "", normalized_query)
    raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", normalized_query)

    keywords: set[str] = set()
    for token in raw_tokens:
        if len(token) >= 2:
            keywords.add(token)

        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) >= 4:
            for size in (2, 3):
                for index in range(len(token) - size + 1):
                    keywords.add(token[index : index + size])

    return sorted(keywords, key=len, reverse=True)


def build_candidates(results: dict[str, list[list[object]]]) -> list[dict[str, object]]:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    candidates: list[dict[str, object]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        candidates.append(
            {
                "document": document,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )
    return candidates


def compute_rerank_breakdown(
    query: str,
    candidate: dict[str, object],
    config: dict[str, Any],
) -> dict[str, Any]:
    document = str(candidate["document"])
    metadata = dict(candidate["metadata"])
    distance = float(candidate["distance"])
    source = str(metadata.get("source", ""))

    normalized_query = normalize_text(query)
    normalized_document = normalize_text(document)
    normalized_source = normalize_text(source)
    keywords = extract_keywords(query)

    vector_score = 1.0 / (1.0 + max(distance, 0.0))

    keyword_hits = 0
    for keyword in keywords:
        if keyword in normalized_document or keyword in normalized_source:
            keyword_hits += 1

    keyword_score = min(keyword_hits, int(config["max_keyword_hits"])) * float(config["keyword_hit_unit"])

    exact_query_bonus = 1.0 if normalized_query and normalized_query in normalized_document else 0.0

    definition_hits = sum(1 for term in config["definition_patterns"] if term in document)
    definition_score = min(definition_hits, int(config["max_definition_hits"]))

    outline_penalty = 1.0 if any(term in source for term in config["outline_source_terms"]) else 0.0

    weighted_vector = vector_score * float(config["vector_weight"])
    weighted_keyword = keyword_score * float(config["keyword_weight"])
    weighted_definition = definition_score * float(config["definition_weight"])
    weighted_exact_query = exact_query_bonus * float(config["exact_query_weight"])
    weighted_outline_penalty = outline_penalty * float(config["outline_penalty_weight"])

    final_score = (
        weighted_vector
        + weighted_keyword
        + weighted_definition
        + weighted_exact_query
        - weighted_outline_penalty
    )

    return {
        "source": source,
        "chunk_id": metadata.get("chunk_id"),
        "distance": distance,
        "vector_score": vector_score,
        "keyword_score": keyword_score,
        "definition_score": definition_score,
        "outline_penalty": outline_penalty,
        "exact_query_bonus": exact_query_bonus,
        "weighted_vector": weighted_vector,
        "weighted_keyword": weighted_keyword,
        "weighted_definition": weighted_definition,
        "weighted_exact_query": weighted_exact_query,
        "weighted_outline_penalty": weighted_outline_penalty,
        "final_score": final_score,
        "keyword_hits": keyword_hits,
        "definition_hits": definition_hits,
        "keywords": keywords,
    }


def rerank_candidates(
    query: str,
    candidates: list[dict[str, object]],
    config: dict[str, Any] | None = None,
) -> list[dict[str, object]]:
    rerank_config = deepcopy(config or RERANK_CONFIG)
    reranked: list[dict[str, object]] = []

    for candidate in candidates:
        breakdown = compute_rerank_breakdown(query, candidate, rerank_config)
        enriched_candidate = dict(candidate)
        enriched_candidate["rerank_breakdown"] = breakdown
        enriched_candidate["rerank_score"] = breakdown["final_score"]
        reranked.append(enriched_candidate)

    reranked.sort(
        key=lambda item: (
            float(item["rerank_score"]),
            -float(item["distance"]),
        ),
        reverse=True,
    )

    return reranked[: int(rerank_config["final_top_k"])]
