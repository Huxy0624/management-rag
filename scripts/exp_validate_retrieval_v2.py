#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH_PIPELINE = Path("data/pipeline_candidates/v1/tagged_chunks/records.json")
DEFAULT_INPUT_PATH_V1_1 = Path("data/experiments/tagging_v1_1/sample_tagged_chunks.json")
DEFAULT_INPUT_PATH_V1 = Path("data/experiments/tagging_v1/sample_tagged_chunks.json")
DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v1/retrieval_eval_v2/query_set.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v1/retrieval_eval_v2/report.json")
DEFAULT_MARKDOWN_OUTPUT_PATH = Path("data/pipeline_candidates/v1/retrieval_eval_v2/report.md")
DEFAULT_TOP_K = 5

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline retrieval validation on full pipeline candidate data.")
    parser.add_argument("--input-path", type=Path, help="Tagged chunk input path.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH, help="Query set JSON path.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="JSON report output path.")
    parser.add_argument(
        "--markdown-output-path",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT_PATH,
        help="Markdown report output path.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K results per method.")
    parser.add_argument("--query-id", type=str, help="Only run one query by id.")
    return parser.parse_args()


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    for candidate in (DEFAULT_INPUT_PATH_PIPELINE, DEFAULT_INPUT_PATH_V1_1, DEFAULT_INPUT_PATH_V1):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No tagged chunk input file found.")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build_phrase_lexicon(records: list[dict[str, object]], queries: list[dict[str, object]]) -> set[str]:
    lexicon = set(TOPIC_TAG_VOCAB + SCENARIO_TAG_VOCAB + ROLE_VOCAB)
    for query in queries:
        lexicon.add(str(query["query"]))
        lexicon.update(str(term) for group in query.get("must_have_groups", []) for term in group)
        lexicon.update(str(term) for term in query.get("helpful_terms", []))
    for record in records:
        tags = dict(record.get("tags", {}))
        concepts = dict(tags.get("concepts", {}))
        lexicon.update(str(item) for item in tags.get("topic_tags", []))
        lexicon.update(str(item) for item in tags.get("scenario_tags", []))
        lexicon.update(str(item) for item in tags.get("applicable_roles", []))
        lexicon.update(str(item) for item in concepts.get("explicit", []))
        lexicon.update(str(item) for item in concepts.get("inferred", []))
        lexicon.update(str(item) for item in concepts.get("original_terms", []))
        for insight in record.get("insights", []):
            if isinstance(insight, dict):
                lexicon.add(str(insight.get("insight_text", "")))
    return {item for item in lexicon if len(item) >= 2}


def extract_tokens(text: str, phrase_lexicon: set[str]) -> set[str]:
    normalized = normalize_text(text)
    tokens = {phrase for phrase in phrase_lexicon if phrase and phrase in normalized}
    tokens.update(chinese_ngrams(normalized, 2))
    tokens.update(chinese_ngrams(normalized, 3))
    return tokens


def build_idf(records: list[dict[str, object]], phrase_lexicon: set[str]) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    for record in records:
        joined = " ".join(
            [
                str(record.get("title", "")),
                str(record.get("chunk_text", "")),
                " ".join(
                    str(item.get("insight_text", ""))
                    for item in record.get("insights", [])
                    if isinstance(item, dict)
                ),
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


def field_weights(method: str) -> dict[str, float]:
    if method == "chunk_only":
        return {"title": 0.5, "chunk_text": 1.0}
    if method == "chunk_plus_insights":
        return {"title": 0.5, "chunk_text": 1.0, "insight_text": 1.45}
    return {
        "title": 0.5,
        "chunk_text": 1.0,
        "insight_text": 1.45,
        "topic_tags": 1.15,
        "scenario_tags": 1.55,
        "explicit_concepts": 1.9,
        "inferred_concepts": 0.95,
        "original_terms": 1.05,
        "applicable_roles": 0.65,
    }


def query_intent(query: str) -> str:
    if any(marker in query for marker in ("为什么", "为何")):
        return "why"
    if any(marker in query for marker in ("怎么", "如何")):
        return "how"
    return "what"


def shared_tokens(text: str, query_tokens: set[str], phrase_lexicon: set[str]) -> list[str]:
    text_tokens = extract_tokens(text, phrase_lexicon)
    matched = [token for token in query_tokens if token in text_tokens]
    return sorted(set(matched), key=lambda item: (-len(item), item))


def score_overlap(text: str, query_tokens: set[str], idf: dict[str, float], phrase_lexicon: set[str]) -> tuple[float, list[str]]:
    matched = shared_tokens(text, query_tokens, phrase_lexicon)
    score = sum(idf.get(token, 1.0) for token in matched)
    return score, matched


def score_record(
    record: dict[str, object],
    query: dict[str, object],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    method: str,
) -> tuple[float, dict[str, dict[str, object]]]:
    query_tokens = extract_tokens(str(query["query"]), phrase_lexicon)
    tags = dict(record.get("tags", {}))
    concepts = dict(tags.get("concepts", {}))
    insight_text = " ".join(
        str(item.get("insight_text", ""))
        for item in record.get("insights", [])
        if isinstance(item, dict)
    )
    fields = {
        "title": str(record.get("title", "")),
        "chunk_text": str(record.get("chunk_text", "")),
        "insight_text": insight_text,
        "topic_tags": " ".join(str(item) for item in tags.get("topic_tags", [])),
        "scenario_tags": " ".join(str(item) for item in tags.get("scenario_tags", [])),
        "explicit_concepts": " ".join(str(item) for item in concepts.get("explicit", [])),
        "inferred_concepts": " ".join(str(item) for item in concepts.get("inferred", [])),
        "original_terms": " ".join(str(item) for item in concepts.get("original_terms", [])),
        "applicable_roles": " ".join(str(item) for item in tags.get("applicable_roles", [])),
    }

    breakdown: dict[str, dict[str, object]] = {}
    total_score = 0.0
    for field_name, weight in field_weights(method).items():
        raw_score, matched_tokens = score_overlap(fields[field_name], query_tokens, idf, phrase_lexicon)
        if raw_score <= 0:
            continue
        weighted_score = raw_score * weight
        breakdown[field_name] = {
            "raw_score": round(raw_score, 4),
            "weight": weight,
            "weighted_score": round(weighted_score, 4),
            "matched_tokens": matched_tokens[:8],
        }
        total_score += weighted_score

    if method == "chunk_insights_tags":
        intent = str(query.get("intent") or query_intent(str(query["query"])))
        bonus = CONTENT_TYPE_QUERY_HINTS.get(intent, {}).get(str(tags.get("content_type", "")), 0.0)
        if bonus:
            breakdown["content_type_bonus"] = {
                "raw_score": round(bonus, 4),
                "weight": 1.0,
                "weighted_score": round(bonus, 4),
                "matched_tokens": [str(tags.get("content_type", ""))],
            }
            total_score += bonus

    return total_score, breakdown


def clip_text(text: str, max_chars: int = 210) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def matched_insights(
    record: dict[str, object],
    query_tokens: set[str],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    limit: int = 3,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for insight in record.get("insights", []):
        if not isinstance(insight, dict):
            continue
        text = str(insight.get("insight_text", ""))
        raw_score, matched = score_overlap(text, query_tokens, idf, phrase_lexicon)
        if raw_score <= 0:
            continue
        rows.append(
            {
                "insight_text": text,
                "matched_tokens": matched[:6],
                "score": round(raw_score, 4),
            }
        )
    rows.sort(key=lambda item: (-float(item["score"]), str(item["insight_text"])))
    return rows[:limit]


def matched_tags(record: dict[str, object], query_tokens: set[str], phrase_lexicon: set[str], method: str) -> dict[str, object]:
    if method != "chunk_insights_tags":
        return {}

    tags = dict(record.get("tags", {}))
    concepts = dict(tags.get("concepts", {}))
    matched: dict[str, object] = {}
    for field_name, values in (
        ("topic_tags", tags.get("topic_tags", [])),
        ("scenario_tags", tags.get("scenario_tags", [])),
        ("explicit_concepts", concepts.get("explicit", [])),
        ("inferred_concepts", concepts.get("inferred", [])),
        ("original_terms", concepts.get("original_terms", [])),
        ("applicable_roles", tags.get("applicable_roles", [])),
    ):
        field_matches = [str(value) for value in values if shared_tokens(str(value), query_tokens, phrase_lexicon)]
        if field_matches:
            matched[field_name] = field_matches

    content_type = str(tags.get("content_type", ""))
    if content_type and CONTENT_TYPE_QUERY_HINTS.get(query_intent("".join(query_tokens)), {}).get(content_type, 0.0):
        matched["content_type"] = content_type
    return matched


def evidence_text_for_relevance(record: dict[str, object]) -> str:
    title = str(record.get("title", ""))
    chunk_text = str(record.get("chunk_text", ""))
    insights_text = " ".join(
        str(item.get("insight_text", ""))
        for item in record.get("insights", [])
        if isinstance(item, dict)
    )
    return normalize_text(f"{title} {chunk_text} {insights_text}")


def judge_relevance(record: dict[str, object], query: dict[str, object]) -> int:
    text = evidence_text_for_relevance(record)
    score = 0
    for group in query.get("must_have_groups", []):
        if any(term in text for term in group):
            score += 2
    for term in query.get("helpful_terms", []):
        if term in text:
            score += 1
    return score


def dcg(relevances: list[int]) -> float:
    total = 0.0
    for index, rel in enumerate(relevances, start=1):
        total += (2**rel - 1) / math.log2(index + 1)
    return total


def build_result_row(
    record: dict[str, object],
    score: float,
    relevance: int,
    breakdown: dict[str, dict[str, object]],
    query_tokens: set[str],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    method: str,
) -> dict[str, object]:
    return {
        "chunk_id": str(record.get("chunk_id", "")),
        "source_file": str(record.get("source_file", "")),
        "title": str(record.get("title", "")),
        "chunk_text_preview": clip_text(str(record.get("chunk_text", "")), max_chars=210),
        "matched_insights": matched_insights(record, query_tokens, idf, phrase_lexicon),
        "matched_tags": matched_tags(record, query_tokens, phrase_lexicon, method),
        "score": round(score, 4),
        "relevance": relevance,
        "score_breakdown": breakdown,
    }


def evaluate_method(
    records: list[dict[str, object]],
    query: dict[str, object],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    method: str,
    top_k: int,
) -> dict[str, object]:
    query_tokens = extract_tokens(str(query["query"]), phrase_lexicon)
    scored_rows: list[dict[str, object]] = []
    for record in records:
        score, breakdown = score_record(record, query, idf, phrase_lexicon, method)
        relevance = judge_relevance(record, query)
        scored_rows.append(
            {
                "chunk_id": str(record.get("chunk_id", "")),
                "score": score,
                "relevance": relevance,
                "result": build_result_row(
                    record=record,
                    score=score,
                    relevance=relevance,
                    breakdown=breakdown,
                    query_tokens=query_tokens,
                    idf=idf,
                    phrase_lexicon=phrase_lexicon,
                    method=method,
                ),
            }
        )

    ranked = sorted(scored_rows, key=lambda item: (-float(item["score"]), -int(item["relevance"]), str(item["chunk_id"])))
    top_rows = ranked[:top_k]
    relevances = [int(item["relevance"]) for item in top_rows]
    ideal = sorted((int(item["relevance"]) for item in ranked), reverse=True)[:top_k]
    ndcg = 0.0
    if ideal and any(ideal):
        ndcg = dcg(relevances) / dcg(ideal)

    return {
        "top_results": [item["result"] for item in top_rows],
        "hit_at_k": any(rel >= 3 for rel in relevances),
        "top1_relevance": int(relevances[0]) if relevances else 0,
        "mean_relevance_at_k": round(sum(relevances) / max(len(relevances), 1), 4),
        "ndcg_at_k": round(ndcg, 4),
    }


def aggregate_metrics(report_queries: list[dict[str, object]], top_k: int) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {
        method: {
            "hit_rate": 0.0,
            "top1_hit_rate": 0.0,
            "mean_top1_relevance": 0.0,
            "mean_relevance_at_k": 0.0,
            "mean_ndcg_at_k": 0.0,
        }
        for method in METHODS
    }
    total_queries = max(1, len(report_queries))
    for report_query in report_queries:
        for method, result in report_query["methods"].items():
            aggregate[method]["hit_rate"] += 1.0 if result["hit_at_k"] else 0.0
            aggregate[method]["top1_hit_rate"] += 1.0 if int(result["top1_relevance"]) >= 3 else 0.0
            aggregate[method]["mean_top1_relevance"] += float(result["top1_relevance"])
            aggregate[method]["mean_relevance_at_k"] += float(result["mean_relevance_at_k"])
            aggregate[method]["mean_ndcg_at_k"] += float(result["ndcg_at_k"])

    for method in METHODS:
        for key in aggregate[method]:
            aggregate[method][key] = round(aggregate[method][key] / total_queries, 4)
    return aggregate


def aggregate_by_category(report_queries: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in report_queries:
        grouped[str(item["category"])].append(item)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for category, rows in grouped.items():
        results[category] = aggregate_metrics(rows, top_k=len(rows))
    return results


def collect_field_contributions(report_queries: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    per_method_fields: dict[str, Counter[str]] = {method: Counter() for method in METHODS}
    for item in report_queries:
        for method in METHODS:
            for result in item["methods"][method]["top_results"]:
                for field_name, payload in result["score_breakdown"].items():
                    per_method_fields[method][field_name] += float(payload.get("weighted_score", 0.0))

    output: dict[str, list[dict[str, object]]] = {}
    for method, counter in per_method_fields.items():
        output[method] = [
            {"field": field, "weighted_score_sum": round(score, 4)}
            for field, score in counter.most_common()
        ]
    return output


def collect_tag_counts(records: list[dict[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        tags = dict(record.get("tags", {}))
        concepts = dict(tags.get("concepts", {}))
        counter.update(str(item) for item in tags.get("topic_tags", []))
        counter.update(str(item) for item in tags.get("scenario_tags", []))
        counter.update(str(item) for item in concepts.get("explicit", []))
    return counter


def collect_matched_tag_counts(report_queries: list[dict[str, object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in report_queries:
        for result in item["methods"]["chunk_insights_tags"]["top_results"]:
            matched = dict(result.get("matched_tags", {}))
            for value in matched.values():
                if isinstance(value, list):
                    counter.update(str(v) for v in value)
                elif isinstance(value, str):
                    counter.update([value])
    return counter


def build_observations(
    records: list[dict[str, object]],
    report_queries: list[dict[str, object]],
    category_summary: dict[str, dict[str, dict[str, float]]],
    field_contributions: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    insights_help = []
    tags_help = []
    weak_queries = []
    for item in report_queries:
        q = str(item["query"])
        chunk_only = item["methods"]["chunk_only"]
        plus_insights = item["methods"]["chunk_plus_insights"]
        plus_tags = item["methods"]["chunk_insights_tags"]
        if (
            float(plus_insights["mean_relevance_at_k"]) - float(chunk_only["mean_relevance_at_k"]) >= 0.4
            or int(plus_insights["top1_relevance"]) > int(chunk_only["top1_relevance"])
        ):
            insights_help.append(q)
        if (
            float(plus_tags["mean_relevance_at_k"]) - float(plus_insights["mean_relevance_at_k"]) >= 0.4
            or int(plus_tags["top1_relevance"]) > int(plus_insights["top1_relevance"])
        ):
            tags_help.append(q)
        if max(
            float(chunk_only["mean_relevance_at_k"]),
            float(plus_insights["mean_relevance_at_k"]),
            float(plus_tags["mean_relevance_at_k"]),
        ) < 2.0:
            weak_queries.append(q)

    category_leaders = {}
    for category, methods in category_summary.items():
        winner = max(methods.items(), key=lambda item: (item[1]["mean_ndcg_at_k"], item[1]["mean_relevance_at_k"]))
        category_leaders[category] = {"best_method": winner[0], "metrics": winner[1]}

    corpus_tag_counts = collect_tag_counts(records)
    matched_tag_counts = collect_matched_tag_counts(report_queries)
    noisy_candidates = []
    for label, corpus_count in corpus_tag_counts.items():
        if corpus_count < 20:
            continue
        matched_count = matched_tag_counts.get(label, 0)
        utility_ratio = matched_count / corpus_count
        noisy_candidates.append(
            {
                "label": label,
                "corpus_count": corpus_count,
                "matched_count": matched_count,
                "utility_ratio": round(utility_ratio, 4),
            }
        )
    noisy_candidates.sort(key=lambda item: (item["utility_ratio"], -item["corpus_count"], item["label"]))

    return {
        "queries_where_insights_help": insights_help[:10],
        "queries_where_tags_help": tags_help[:10],
        "queries_still_weak": weak_queries[:10],
        "category_leaders": category_leaders,
        "tag_signal_fields": field_contributions.get("chunk_insights_tags", []),
        "potential_noise_tags": noisy_candidates[:8],
    }


def markdown_for_result(result: dict[str, object], rank: int) -> list[str]:
    lines = [
        f"{rank}. `{result['chunk_id']}` | score={result['score']} | relevance={result['relevance']} | `{result['source_file']}`",
        f"   Preview: {result['chunk_text_preview']}",
    ]
    matched_insights_rows = list(result.get("matched_insights", []))
    if matched_insights_rows:
        insight_summary = " ; ".join(str(item["insight_text"]) for item in matched_insights_rows[:2])
        lines.append(f"   Matched insights: {insight_summary}")
    matched_tags_rows = dict(result.get("matched_tags", {}))
    if matched_tags_rows:
        tag_parts = [f"{key}={value}" for key, value in matched_tags_rows.items()]
        lines.append(f"   Matched tags: {'; '.join(tag_parts)}")
    breakdown_rows = dict(result.get("score_breakdown", {}))
    if breakdown_rows:
        breakdown_text = "; ".join(
            f"{field}={payload['weighted_score']}" for field, payload in breakdown_rows.items()
        )
        lines.append(f"   Score breakdown: {breakdown_text}")
    return lines


def write_markdown_report(report: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Retrieval Evaluation V2",
        "",
        f"- Input path: `{report['input_path']}`",
        f"- Query set path: `{report['query_set_path']}`",
        f"- Total records: `{report['total_records']}`",
        f"- Unique sources: `{report['unique_sources']}`",
        f"- Top K: `{report['top_k']}`",
        "",
        "## Aggregate",
        "",
    ]
    for method, metrics in report["aggregate"].items():
        lines.append(
            f"- `{method}`: hit_rate={metrics['hit_rate']}, top1_hit_rate={metrics['top1_hit_rate']}, "
            f"mean_relevance@{report['top_k']}={metrics['mean_relevance_at_k']}, "
            f"mean_ndcg@{report['top_k']}={metrics['mean_ndcg_at_k']}"
        )
    lines.extend(["", "## Key Observations", ""])
    observations = dict(report.get("observations", {}))
    for key in (
        "queries_where_insights_help",
        "queries_where_tags_help",
        "queries_still_weak",
    ):
        values = observations.get(key, [])
        lines.append(f"- `{key}`: {values}")

    lines.append("")
    for query in report["queries"]:
        lines.extend(
            [
                f"## {query['query']}",
                "",
                f"- Query id: `{query['query_id']}`",
                f"- Category: `{query['category']}`",
                "",
            ]
        )
        for method in METHODS:
            result = query["methods"][method]
            lines.extend(
                [
                    f"### {method} top{report['top_k']}",
                    "",
                    f"- hit_at_k: `{result['hit_at_k']}`",
                    f"- top1_relevance: `{result['top1_relevance']}`",
                    f"- mean_relevance@{report['top_k']}: `{result['mean_relevance_at_k']}`",
                    f"- ndcg@{report['top_k']}: `{result['ndcg_at_k']}`",
                    "",
                ]
            )
            for index, row in enumerate(result["top_results"], start=1):
                lines.extend(markdown_for_result(row, index))
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    input_path = resolve_input_path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not args.query_set_path.exists():
        raise FileNotFoundError(f"Query set file not found: {args.query_set_path}")

    records = load_json(input_path)
    queries = load_json(args.query_set_path)
    if args.query_id:
        queries = [query for query in queries if str(query.get("id", "")) == args.query_id]
        if not queries:
            raise ValueError(f"Unknown query id: {args.query_id}")

    phrase_lexicon = build_phrase_lexicon(records, queries)
    idf = build_idf(records, phrase_lexicon)

    report_queries: list[dict[str, object]] = []
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
                "query_id": str(query["id"]),
                "query": str(query["query"]),
                "category": str(query.get("category", query.get("intent", query_intent(str(query["query"]))))),
                "intent": str(query.get("intent", query_intent(str(query["query"])))),
                "methods": method_results,
            }
        )

    aggregate = aggregate_metrics(report_queries, args.top_k)
    category_summary = aggregate_by_category(report_queries)
    field_contributions = collect_field_contributions(report_queries)
    observations = build_observations(records, report_queries, category_summary, field_contributions)

    report = {
        "input_path": str(input_path),
        "query_set_path": str(args.query_set_path),
        "top_k": args.top_k,
        "methods": list(METHODS),
        "total_records": len(records),
        "unique_sources": len({str(item.get('source_file', '')) for item in records}),
        "queries": report_queries,
        "aggregate": aggregate,
        "category_summary": category_summary,
        "field_contributions": field_contributions,
        "observations": observations,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(report, args.markdown_output_path)

    print(f"Input file: {input_path}")
    print(f"Query set: {args.query_set_path}")
    print(f"Total records: {len(records)}")
    print(f"Unique sources: {len({str(item.get('source_file', '')) for item in records})}")
    print(f"Queries evaluated: {len(report_queries)}")
    print(f"Saved JSON report to: {args.output_path}")
    print(f"Saved Markdown report to: {args.markdown_output_path}")
    print("Aggregate summary:")
    for method in METHODS:
        summary = aggregate[method]
        print(
            f"- {method}: hit_rate={summary['hit_rate']:.4f}, "
            f"top1_hit_rate={summary['top1_hit_rate']:.4f}, "
            f"mean_relevance@{args.top_k}={summary['mean_relevance_at_k']:.4f}, "
            f"mean_ndcg@{args.top_k}={summary['mean_ndcg_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()
