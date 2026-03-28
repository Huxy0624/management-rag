#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


DEFAULT_INPUT_PATH_V1_1 = Path("data/experiments/tagging_v1_1/sample_tagged_chunks.json")
DEFAULT_INPUT_PATH_V1 = Path("data/experiments/tagging_v1/sample_tagged_chunks.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/retrieval_validation_v1/report.json")
DEFAULT_TOP_K = 3

METHODS = (
    "chunk_only",
    "chunk_plus_insights",
    "chunk_insights_tags",
)

TOPIC_TAG_VOCAB = [
    "管理认知",
    "团队治理",
    "执行管理",
    "组织设计",
    "人效管理",
    "绩效管理",
    "文化建设",
    "机制建设",
]

SCENARIO_TAG_VOCAB = [
    "跨部门协作",
    "项目复盘",
    "管理学习",
    "向上汇报",
    "管理选拔",
    "工作负担",
    "角色错位",
    "组织扩张",
    "绩效评价",
    "战略传导",
    "任务分配",
]

ROLE_VOCAB = [
    "高管",
    "总监",
    "经理",
    "一线员工",
    "创业者",
    "项目负责人",
]

CONTENT_TYPE_QUERY_HINTS = {
    "why": {"pitfall": 0.35, "principle": 0.2, "criteria": 0.1},
    "how": {"method": 0.35, "solution": 0.35, "criteria": 0.1},
    "what": {"principle": 0.3, "criteria": 0.2},
}

EVAL_QUERIES = [
    {
        "id": "cross_dept",
        "query": "为什么跨部门协作经常低效？",
        "must_have_groups": [
            ["部门墙", "跨部门", "扯皮"],
            ["管理能力", "无效工作", "管理问题"],
        ],
        "helpful_terms": ["BUG", "事故", "项目失败"],
        "intent": "why",
    },
    {
        "id": "management_goal",
        "query": "管理的核心目标是什么？",
        "must_have_groups": [
            ["人效", "有效工作"],
            ["成本控制", "管理的意义", "有限成本下提升产出"],
        ],
        "helpful_terms": ["无效工作", "价值"],
        "intent": "what",
    },
    {
        "id": "scale_distortion",
        "query": "为什么团队扩大后更容易信息失真？",
        "must_have_groups": [
            ["组织扩张", "规模扩大", "人数", "沟通复杂度"],
            ["信息失真", "信息过载", "噪声"],
        ],
        "helpful_terms": ["沟通路径", "非线性增长"],
        "intent": "why",
    },
    {
        "id": "management_learning",
        "query": "管理学习初期应该怎么学？",
        "must_have_groups": [
            ["管理学习", "系统性学习", "系统框架"],
            ["高管视角", "高管管理知识"],
        ],
        "helpful_terms": ["管理体系", "晋升经理"],
        "intent": "how",
    },
    {
        "id": "review_value",
        "query": "复盘为什么重要？",
        "must_have_groups": [
            ["复盘"],
            ["评价标准", "信息失真", "评价失效", "机制"],
        ],
        "helpful_terms": ["问题打开", "暴露出来", "共同确定评价标准"],
        "intent": "why",
    },
    {
        "id": "effective_work",
        "query": "什么是有效工作？",
        "must_have_groups": [
            ["有效工作"],
            ["人效", "价值", "评价"],
        ],
        "helpful_terms": ["不确定性", "任务"],
        "intent": "what",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval validation for chunk / insight / tag schemes.")
    parser.add_argument("--input-path", type=Path, help="Optional tagged chunk input path.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Validation report output path.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K results to evaluate for each method.")
    parser.add_argument("--query-id", type=str, help="Only run one built-in query by id.")
    return parser.parse_args()


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    if DEFAULT_INPUT_PATH_V1_1.exists():
        return DEFAULT_INPUT_PATH_V1_1
    if DEFAULT_INPUT_PATH_V1.exists():
        return DEFAULT_INPUT_PATH_V1
    raise FileNotFoundError(
        f"Neither {DEFAULT_INPUT_PATH_V1_1} nor {DEFAULT_INPUT_PATH_V1} exists."
    )


def normalize_text(text: str) -> str:
    normalized = text.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def chinese_ngrams(text: str, n: int) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text)
    joined = "".join(chars)
    if len(joined) < n:
        return set()
    return {joined[index : index + n] for index in range(len(joined) - n + 1)}


def build_phrase_lexicon(records: list[dict[str, object]]) -> set[str]:
    lexicon = set(TOPIC_TAG_VOCAB + SCENARIO_TAG_VOCAB + ROLE_VOCAB)
    for query in EVAL_QUERIES:
        lexicon.add(query["query"])
        for group in query["must_have_groups"]:
            lexicon.update(group)
        lexicon.update(query["helpful_terms"])
    for record in records:
        tags = dict(record.get("tags", {}))
        concepts = dict(tags.get("concepts", {}))
        lexicon.update(str(item) for item in tags.get("topic_tags", []))
        lexicon.update(str(item) for item in tags.get("scenario_tags", []))
        lexicon.update(str(item) for item in tags.get("applicable_roles", []))
        lexicon.update(str(item) for item in concepts.get("explicit", []))
        lexicon.update(str(item) for item in concepts.get("inferred", []))
        lexicon.update(str(item) for item in concepts.get("original_terms", []))
    return {item for item in lexicon if len(item) >= 2}


def extract_tokens(text: str, phrase_lexicon: set[str]) -> set[str]:
    normalized = normalize_text(text)
    tokens = {phrase for phrase in phrase_lexicon if phrase in normalized}
    tokens.update(chinese_ngrams(normalized, 2))
    tokens.update(chinese_ngrams(normalized, 3))
    return tokens


def build_idf(records: list[dict[str, object]], phrase_lexicon: set[str]) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    for record in records:
        joined = " ".join(
            [
                str(record.get("chunk_text", "")),
                " ".join(str(item.get("insight_text", "")) for item in record.get("insights", []) if isinstance(item, dict)),
                flatten_tags(dict(record.get("tags", {}))),
            ]
        )
        for token in extract_tokens(joined, phrase_lexicon):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    total_docs = max(1, len(records))
    return {
        token: math.log((total_docs + 1) / (freq + 1)) + 1.0
        for token, freq in doc_freq.items()
    }


def flatten_tags(tags: dict[str, object]) -> str:
    concepts = dict(tags.get("concepts", {}))
    relations = [
        f"{item.get('type', '')} {item.get('from', '')} {item.get('to', '')}"
        for item in tags.get("relations", [])
        if isinstance(item, dict)
    ]
    return " ".join(
        [
            " ".join(str(item) for item in tags.get("topic_tags", [])),
            str(tags.get("content_type", "")),
            " ".join(str(item) for item in concepts.get("explicit", [])),
            " ".join(str(item) for item in concepts.get("inferred", [])),
            " ".join(str(item) for item in concepts.get("original_terms", [])),
            " ".join(str(item) for item in tags.get("scenario_tags", [])),
            " ".join(str(item) for item in tags.get("applicable_roles", [])),
            " ".join(relations),
        ]
    )


def field_weights(method: str) -> dict[str, float]:
    if method == "chunk_only":
        return {"chunk_text": 1.0}
    if method == "chunk_plus_insights":
        return {"chunk_text": 1.0, "insight_text": 1.4}
    return {
        "chunk_text": 1.0,
        "insight_text": 1.4,
        "topic_tags": 1.2,
        "scenario_tags": 1.6,
        "explicit_concepts": 1.9,
        "inferred_concepts": 1.0,
        "original_terms": 1.1,
        "applicable_roles": 0.8,
    }


def query_intent(query: str) -> str:
    if any(marker in query for marker in ("为什么", "为何")):
        return "why"
    if any(marker in query for marker in ("怎么", "如何")):
        return "how"
    return "what"


def score_record(
    record: dict[str, object],
    query: dict[str, object],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    method: str,
) -> tuple[float, dict[str, float]]:
    query_tokens = extract_tokens(str(query["query"]), phrase_lexicon)
    tags = dict(record.get("tags", {}))
    concepts = dict(tags.get("concepts", {}))
    insight_text = " ".join(
        str(item.get("insight_text", ""))
        for item in record.get("insights", [])
        if isinstance(item, dict)
    )
    fields = {
        "chunk_text": str(record.get("chunk_text", "")),
        "insight_text": insight_text,
        "topic_tags": " ".join(str(item) for item in tags.get("topic_tags", [])),
        "scenario_tags": " ".join(str(item) for item in tags.get("scenario_tags", [])),
        "explicit_concepts": " ".join(str(item) for item in concepts.get("explicit", [])),
        "inferred_concepts": " ".join(str(item) for item in concepts.get("inferred", [])),
        "original_terms": " ".join(str(item) for item in concepts.get("original_terms", [])),
        "applicable_roles": " ".join(str(item) for item in tags.get("applicable_roles", [])),
    }

    components: dict[str, float] = {}
    total_score = 0.0
    for field_name, weight in field_weights(method).items():
        tokens = extract_tokens(fields[field_name], phrase_lexicon)
        field_score = sum(idf.get(token, 1.0) for token in query_tokens if token in tokens)
        if field_score > 0:
            weighted = field_score * weight
            components[field_name] = round(weighted, 4)
            total_score += weighted

    if method == "chunk_insights_tags":
        intent = str(query.get("intent") or query_intent(str(query["query"])))
        hint_bonus = CONTENT_TYPE_QUERY_HINTS.get(intent, {})
        content_type = str(tags.get("content_type", ""))
        total_score += hint_bonus.get(content_type, 0.0)
        if hint_bonus.get(content_type, 0.0):
            components["content_type_bonus"] = round(hint_bonus[content_type], 4)

    return total_score, components


def judge_relevance(record: dict[str, object], query: dict[str, object]) -> int:
    text = normalize_text(
        f"{record.get('chunk_text', '')} "
        + " ".join(
            str(item.get("insight_text", ""))
            for item in record.get("insights", [])
            if isinstance(item, dict)
        )
    )

    score = 0
    for group in query["must_have_groups"]:
        if any(term in text for term in group):
            score += 2
    for term in query["helpful_terms"]:
        if term in text:
            score += 1
    return score


def dcg(relevances: list[int]) -> float:
    total = 0.0
    for index, rel in enumerate(relevances, start=1):
        total += (2**rel - 1) / math.log2(index + 1)
    return total


def evaluate_method(
    records: list[dict[str, object]],
    query: dict[str, object],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    method: str,
    top_k: int,
) -> dict[str, object]:
    scored_rows: list[dict[str, object]] = []
    for record in records:
        score, components = score_record(record, query, idf, phrase_lexicon, method)
        relevance = judge_relevance(record, query)
        scored_rows.append(
            {
                "chunk_id": str(record.get("chunk_id", "")),
                "title": str(record.get("title", "")),
                "score": round(score, 4),
                "relevance": relevance,
                "components": components,
            }
        )

    ranked = sorted(scored_rows, key=lambda item: (-item["score"], -item["relevance"], item["chunk_id"]))
    top_rows = ranked[:top_k]
    relevances = [int(item["relevance"]) for item in top_rows]
    ideal = sorted((int(item["relevance"]) for item in ranked), reverse=True)[:top_k]
    ndcg = 0.0
    if ideal and any(ideal):
        ndcg = dcg(relevances) / dcg(ideal)

    return {
        "top_results": top_rows,
        "hit_at_k": any(rel >= 3 for rel in relevances),
        "mean_relevance_at_k": round(sum(relevances) / max(len(relevances), 1), 4),
        "ndcg_at_k": round(ndcg, 4),
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    records = json.loads(input_path.read_text(encoding="utf-8"))
    phrase_lexicon = build_phrase_lexicon(records)
    idf = build_idf(records, phrase_lexicon)

    queries = EVAL_QUERIES
    if args.query_id:
        queries = [query for query in EVAL_QUERIES if query["id"] == args.query_id]
        if not queries:
            raise ValueError(f"Unknown query id: {args.query_id}")

    report_queries: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, float]] = {
        method: {"hit_rate": 0.0, "mean_relevance_at_k": 0.0, "mean_ndcg_at_k": 0.0}
        for method in METHODS
    }

    for query in queries:
        method_results = {
            method: evaluate_method(
                records=records,
                query=query,
                idf=idf,
                phrase_lexicon=phrase_lexicon,
                method=method,
                top_k=args.top_k,
            )
            for method in METHODS
        }
        report_queries.append(
            {
                "query_id": query["id"],
                "query": query["query"],
                "intent": query["intent"],
                "methods": method_results,
            }
        )
        for method, result in method_results.items():
            aggregate[method]["hit_rate"] += 1.0 if result["hit_at_k"] else 0.0
            aggregate[method]["mean_relevance_at_k"] += float(result["mean_relevance_at_k"])
            aggregate[method]["mean_ndcg_at_k"] += float(result["ndcg_at_k"])

    total_queries = max(1, len(report_queries))
    for method in METHODS:
        aggregate[method]["hit_rate"] = round(aggregate[method]["hit_rate"] / total_queries, 4)
        aggregate[method]["mean_relevance_at_k"] = round(
            aggregate[method]["mean_relevance_at_k"] / total_queries,
            4,
        )
        aggregate[method]["mean_ndcg_at_k"] = round(
            aggregate[method]["mean_ndcg_at_k"] / total_queries,
            4,
        )

    report = {
        "input_path": str(input_path),
        "top_k": args.top_k,
        "methods": list(METHODS),
        "queries": report_queries,
        "aggregate": aggregate,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file: {input_path}")
    print(f"Queries evaluated: {len(report_queries)}")
    print(f"Saved report to: {args.output_path}")
    print("Aggregate summary:")
    for method in METHODS:
        summary = aggregate[method]
        print(
            f"- {method}: hit_rate={summary['hit_rate']:.4f}, "
            f"mean_relevance@{args.top_k}={summary['mean_relevance_at_k']:.4f}, "
            f"mean_ndcg@{args.top_k}={summary['mean_ndcg_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()
