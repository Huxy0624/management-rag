#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v4/tagged_chunks/records.json")
DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/report.json")
DEFAULT_BASELINE_V3_REPORT = Path("data/pipeline_candidates/v2/retrieval_eval_v3/report.json")
DEFAULT_BASELINE_V2_TAGS = Path("data/pipeline_candidates/v2/tagged_chunks/records.json")
DEFAULT_TOP_K = 5
DEFAULT_TOP_M = 15

METHODS = ("primary_only", "primary_plus_matrix")

MATRIX_FIELD_WEIGHTS = {
    "answer_role": 2.7,
    "root_issue": 1.55,
    "intent": 1.45,
    "target_role": 0.45,
    "role_profile": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 offline retrieval validation for governance matrix v4.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input v4 tagged chunks path.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH, help="Phase2 query set JSON path.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output report JSON path.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Final top-k for evaluation.")
    parser.add_argument("--top-m", type=int, default=DEFAULT_TOP_M, help="Primary recall candidate pool size.")
    parser.add_argument("--query-id", type=str, help="Only run one query by id.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    normalized = text.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def pretty_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chinese_ngrams(text: str, n: int) -> set[str]:
    chars = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text)
    joined = "".join(chars)
    if len(joined) < n:
        return set()
    return {joined[index : index + n] for index in range(len(joined) - n + 1)}


def primary_document_text(record: dict[str, Any]) -> str:
    return " ".join(
        [
            str(record.get("title", "")),
            str(record.get("chunk_text", "")),
            " ".join(
                str(item.get("insight_text", ""))
                for item in record.get("insights", [])
                if isinstance(item, dict)
            ),
        ]
    )


def build_phrase_lexicon(records: list[dict[str, Any]], queries: list[dict[str, Any]]) -> set[str]:
    lexicon: set[str] = set()
    for query in queries:
        lexicon.add(str(query.get("query", "")))
        lexicon.update(str(term) for group in query.get("must_have_groups", []) for term in group)
        lexicon.update(str(term) for term in query.get("helpful_terms", []))
    for record in records:
        lexicon.add(str(record.get("title", "")))
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


def build_idf(records: list[dict[str, Any]], phrase_lexicon: set[str]) -> dict[str, float]:
    doc_freq: dict[str, int] = {}
    for record in records:
        for token in extract_tokens(primary_document_text(record), phrase_lexicon):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total_docs = max(1, len(records))
    return {
        token: math.log((total_docs + 1) / (freq + 1)) + 1.0
        for token, freq in doc_freq.items()
    }


def score_overlap(text: str, query_tokens: set[str], idf: dict[str, float], phrase_lexicon: set[str]) -> tuple[float, list[str]]:
    text_tokens = extract_tokens(text, phrase_lexicon)
    matched = sorted({token for token in query_tokens if token in text_tokens}, key=lambda item: (-len(item), item))
    score = sum(idf.get(token, 1.0) for token in matched)
    return score, matched


def clip_text(text: str, max_chars: int = 210) -> str:
    compact = pretty_text(text)
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars].rstrip()}..."


def judge_relevance(record: dict[str, Any], query: dict[str, Any]) -> int:
    text = normalize_text(primary_document_text(record))
    score = 0
    for group in query.get("must_have_groups", []):
        if any(term in text for term in group):
            score += 2
    for term in query.get("helpful_terms", []):
        if term in text:
            score += 1
    return score


def query_type(query_text: str) -> str:
    if any(marker in query_text for marker in ("为什么", "为何")):
        return "why"
    if any(marker in query_text for marker in ("怎么", "如何")):
        return "how"
    return "what"


def parse_query_profile(query_text: str) -> dict[str, Any]:
    normalized = normalize_text(query_text)
    profile = {
        "query_type": query_type(query_text),
        "preferred_answer_roles": [],
        "avoid_answer_roles": [],
        "preferred_root_issues": [],
        "preferred_intents": [],
        "preferred_target_roles": [],
        "preferred_role_profiles": [],
    }

    if any(token in normalized for token in ("什么是", "定义", "本质")):
        profile["preferred_answer_roles"] = ["definition", "principle", "summary"]
        profile["avoid_answer_roles"] = ["mechanism", "example"]
    elif any(token in normalized for token in ("为什么", "为何")):
        profile["preferred_answer_roles"] = ["mechanism", "cause", "comparison", "warning"]
        profile["avoid_answer_roles"] = ["definition", "summary", "solution"]
    elif any(token in normalized for token in ("怎么", "如何")):
        profile["preferred_answer_roles"] = ["solution", "mechanism", "example"]
        profile["avoid_answer_roles"] = ["definition", "summary"]

    if any(token in normalized for token in ("信息不通", "信息失真", "汇报", "传导", "压缩", "扩散")):
        profile["preferred_root_issues"].append("information_distortion")
    if any(token in normalized for token in ("评价失效", "评价不公", "评价标准", "奖惩", "激励")):
        profile["preferred_root_issues"].append("evaluation_failure")

    if any(token in normalized for token in ("跨部门", "协作", "部门墙", "资源协调")):
        profile["preferred_intents"].append("resource_coordination")
    if any(token in normalized for token in ("机制", "制度", "流程", "规则", "长期")):
        profile["preferred_intents"].append("mechanism_design")
    if any(token in normalized for token in ("带人", "培养", "辅导")):
        profile["preferred_intents"].append("coaching")
    if any(token in normalized for token in ("选拔", "候选人", "任命")):
        profile["preferred_intents"].append("selection")
    if any(token in normalized for token in ("接班", "梯队", "后备")):
        profile["preferred_intents"].append("succession")

    if any(token in normalized for token in ("老板", "高管", "CEO")):
        profile["preferred_target_roles"].append("executive")
    if any(token in normalized for token in ("总监", "部门负责人")):
        profile["preferred_target_roles"].append("director")
    if any(token in normalized for token in ("经理", "leader", "项目负责人")):
        profile["preferred_target_roles"].append("manager")
    if any(token in normalized for token in ("员工", "一线")):
        profile["preferred_target_roles"].append("employee")

    if any(token in normalized for token in ("滚刀肉", "出工不出力")):
        profile["preferred_role_profiles"].append("slippery_worker")
    if any(token in normalized for token in ("挑战", "不服")):
        profile["preferred_role_profiles"].append("challenger")

    for key in profile:
        if isinstance(profile[key], list):
            deduped: list[str] = []
            for item in profile[key]:
                if item not in deduped:
                    deduped.append(item)
            profile[key] = deduped
    return profile


def primary_score_record(
    record: dict[str, Any],
    query_tokens: set[str],
    idf: dict[str, float],
    phrase_lexicon: set[str],
) -> tuple[float, dict[str, dict[str, Any]]]:
    fields = {
        "title": (str(record.get("title", "")), 0.35),
        "chunk_text": (str(record.get("chunk_text", "")), 1.0),
        "insight_text": (
            " ".join(
                str(item.get("insight_text", ""))
                for item in record.get("insights", [])
                if isinstance(item, dict)
            ),
            1.35,
        ),
    }
    total_score = 0.0
    breakdown: dict[str, dict[str, Any]] = {}
    for field_name, (text, weight) in fields.items():
        raw_score, matched_tokens = score_overlap(text, query_tokens, idf, phrase_lexicon)
        if raw_score <= 0:
            continue
        weighted_score = raw_score * weight
        total_score += weighted_score
        breakdown[field_name] = {
            "raw_score": round(raw_score, 4),
            "weight": weight,
            "weighted_score": round(weighted_score, 4),
            "matched_tokens": matched_tokens[:8],
        }
    return total_score, breakdown


def score_matrix_field(value: str, preferred: list[str], avoid: list[str], weight: float) -> tuple[float, dict[str, Any]]:
    if not preferred and not avoid:
        return 0.0, {"field_value": value, "decision": "neutral", "delta": 0.0}
    if value in preferred:
        return weight, {"field_value": value, "decision": "preferred_match", "delta": round(weight, 4)}
    if value in avoid and value != "none":
        delta = -weight * 0.8
        return delta, {"field_value": value, "decision": "avoid_match", "delta": round(delta, 4)}
    if value == "none":
        delta = -weight * 0.2
        return delta, {"field_value": value, "decision": "missing", "delta": round(delta, 4)}
    if preferred:
        delta = -weight * 0.52
        return delta, {"field_value": value, "decision": "mismatch", "delta": round(delta, 4)}
    return 0.0, {"field_value": value, "decision": "neutral", "delta": 0.0}


def score_matrix_list(values: list[str], preferred: list[str], weight: float) -> tuple[float, dict[str, Any]]:
    if not preferred:
        return 0.0, {"field_value": values, "decision": "neutral", "delta": 0.0}
    if any(value in preferred for value in values):
        return weight, {"field_value": values, "decision": "preferred_match", "delta": round(weight, 4)}
    if not values:
        delta = -weight * 0.15
        return delta, {"field_value": values, "decision": "missing", "delta": round(delta, 4)}
    delta = -weight * 0.25
    return delta, {"field_value": values, "decision": "mismatch", "delta": round(delta, 4)}


def matrix_adjustment(record: dict[str, Any], profile: dict[str, Any], primary_score: float) -> tuple[float, float, dict[str, Any]]:
    tags = dict(record.get("governance_tags_v4", {}))
    components: dict[str, Any] = {}
    raw_delta = 0.0

    answer_delta, answer_info = score_matrix_field(
        str(tags.get("answer_role", "none")),
        list(profile.get("preferred_answer_roles", [])),
        list(profile.get("avoid_answer_roles", [])),
        MATRIX_FIELD_WEIGHTS["answer_role"],
    )
    raw_delta += answer_delta
    components["answer_role"] = answer_info

    root_delta, root_info = score_matrix_field(
        str(tags.get("root_issue", "none")),
        list(profile.get("preferred_root_issues", [])),
        [],
        MATRIX_FIELD_WEIGHTS["root_issue"],
    )
    raw_delta += root_delta
    components["root_issue"] = root_info

    intent_delta, intent_info = score_matrix_field(
        str(tags.get("intent", "none")),
        list(profile.get("preferred_intents", [])),
        [],
        MATRIX_FIELD_WEIGHTS["intent"],
    )
    raw_delta += intent_delta
    components["intent"] = intent_info

    target_delta, target_info = score_matrix_list(
        list(tags.get("target_role", [])),
        list(profile.get("preferred_target_roles", [])),
        MATRIX_FIELD_WEIGHTS["target_role"],
    )
    raw_delta += target_delta
    components["target_role"] = target_info

    profile_delta, profile_info = score_matrix_list(
        list(tags.get("role_profile", [])),
        list(profile.get("preferred_role_profiles", [])),
        MATRIX_FIELD_WEIGHTS["role_profile"],
    )
    raw_delta += profile_delta
    components["role_profile"] = profile_info

    lower_bound = -0.08 * primary_score
    upper_bound = 0.12 * primary_score
    clamped = max(lower_bound, min(upper_bound, raw_delta))
    components["clamp"] = {
        "raw_delta": round(raw_delta, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "clamped_delta": round(clamped, 4),
    }
    return raw_delta, clamped, components


def matched_insights(
    record: dict[str, Any],
    query_tokens: set[str],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for insight in record.get("insights", []):
        if not isinstance(insight, dict):
            continue
        text = str(insight.get("insight_text", ""))
        raw_score, matched = score_overlap(text, query_tokens, idf, phrase_lexicon)
        if raw_score <= 0:
            continue
        rows.append({"insight_text": text, "score": round(raw_score, 4), "matched_tokens": matched[:6]})
    rows.sort(key=lambda item: (-float(item["score"]), str(item["insight_text"])))
    return rows[:limit]


def build_result_row(
    record: dict[str, Any],
    relevance: int,
    primary_score: float,
    final_score: float,
    primary_breakdown: dict[str, Any],
    matrix_breakdown: dict[str, Any],
    query_tokens: set[str],
    idf: dict[str, float],
    phrase_lexicon: set[str],
) -> dict[str, Any]:
    return {
        "chunk_id": str(record.get("chunk_id", "")),
        "source_file": str(record.get("source_file", "")),
        "title": str(record.get("title", "")),
        "chunk_text_preview": clip_text(str(record.get("chunk_text", "")), max_chars=210),
        "matched_insights": matched_insights(record, query_tokens, idf, phrase_lexicon),
        "governance_tags_v4": record.get("governance_tags_v4", {}),
        "primary_score": round(primary_score, 4),
        "final_score": round(final_score, 4),
        "relevance": relevance,
        "primary_score_breakdown": primary_breakdown,
        "matrix_score_breakdown": matrix_breakdown,
    }


def dcg(relevances: list[int]) -> float:
    total = 0.0
    for index, rel in enumerate(relevances, start=1):
        total += (2**rel - 1) / math.log2(index + 1)
    return total


def summarize_ranking(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    top_rows = rows[:top_k]
    relevances = [int(item["relevance"]) for item in top_rows]
    ideal = sorted((int(item["relevance"]) for item in rows), reverse=True)[:top_k]
    ndcg = 0.0
    if ideal and any(ideal):
        ndcg = dcg(relevances) / dcg(ideal)
    return {
        "top_results": [item["result"] for item in top_rows],
        "hit_at_k": any(rel >= 3 for rel in relevances),
        "top1_hit": bool(relevances and relevances[0] >= 3),
        "top1_relevance": int(relevances[0]) if relevances else 0,
        "mean_relevance_at_k": round(sum(relevances) / max(len(relevances), 1), 4),
        "ndcg_at_k": round(ndcg, 4),
    }


def evaluate_query(
    records: list[dict[str, Any]],
    query: dict[str, Any],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    top_k: int,
    top_m: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_text = str(query.get("query", ""))
    query_tokens = extract_tokens(query_text, phrase_lexicon)
    profile = parse_query_profile(query_text)

    primary_ranked: list[dict[str, Any]] = []
    for record in records:
        primary_score, primary_breakdown = primary_score_record(record, query_tokens, idf, phrase_lexicon)
        relevance = judge_relevance(record, query)
        result = build_result_row(record, relevance, primary_score, primary_score, primary_breakdown, {}, query_tokens, idf, phrase_lexicon)
        primary_ranked.append(
            {
                "chunk_id": str(record.get("chunk_id", "")),
                "primary_score": primary_score,
                "final_score": primary_score,
                "relevance": relevance,
                "record": record,
                "result": result,
                "primary_breakdown": primary_breakdown,
                "matrix_breakdown": {},
                "matrix_delta_clamped": 0.0,
            }
        )

    primary_ranked.sort(key=lambda item: (-float(item["primary_score"]), -int(item["relevance"]), str(item["chunk_id"])))
    primary_summary = summarize_ranking(primary_ranked, top_k)

    rerank_pool = primary_ranked[:top_m]
    reranked_rows: list[dict[str, Any]] = []
    for row in rerank_pool:
        raw_delta, clamped_delta, matrix_breakdown = matrix_adjustment(row["record"], profile, row["primary_score"])
        final_score = row["primary_score"] + clamped_delta
        result = build_result_row(row["record"], row["relevance"], row["primary_score"], final_score, row["primary_breakdown"], matrix_breakdown, query_tokens, idf, phrase_lexicon)
        reranked_rows.append(
            {
                "chunk_id": row["chunk_id"],
                "primary_score": row["primary_score"],
                "final_score": final_score,
                "relevance": row["relevance"],
                "record": row["record"],
                "result": result,
                "primary_breakdown": row["primary_breakdown"],
                "matrix_breakdown": matrix_breakdown,
                "matrix_delta_raw": raw_delta,
                "matrix_delta_clamped": clamped_delta,
            }
        )

    reranked_rows.sort(key=lambda item: (-float(item["final_score"]), -int(item["relevance"]), str(item["chunk_id"])))
    matrix_summary = summarize_ranking(reranked_rows, top_k)

    return {
        "primary_only": primary_summary,
        "primary_plus_matrix": matrix_summary,
    }, primary_ranked, reranked_rows, profile


def aggregate_metrics(query_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate = {
        method: {"hit_rate": 0.0, "top1_hit_rate": 0.0, "mean_relevance_at_5": 0.0, "mean_ndcg_at_5": 0.0}
        for method in METHODS
    }
    total = max(1, len(query_rows))
    for item in query_rows:
        for method in METHODS:
            result = item["methods"][method]
            aggregate[method]["hit_rate"] += 1.0 if result["hit_at_k"] else 0.0
            aggregate[method]["top1_hit_rate"] += 1.0 if result["top1_hit"] else 0.0
            aggregate[method]["mean_relevance_at_5"] += float(result["mean_relevance_at_k"])
            aggregate[method]["mean_ndcg_at_5"] += float(result["ndcg_at_k"])
    for method in METHODS:
        for key in aggregate[method]:
            aggregate[method][key] = round(aggregate[method][key] / total, 4)
    return aggregate


def breakdown_by_query_type(query_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[str(row["category"])].append(row)
    return {category: aggregate_metrics(rows) for category, rows in grouped.items()}


def scene_keywords(query_text: str) -> list[str]:
    mapping = ["跨部门", "协作", "评价失效", "评价", "汇报", "战略", "机制", "人治", "法治", "资源"]
    normalized = normalize_text(query_text)
    return [keyword for keyword in mapping if keyword in normalized]


def role_mismatch_cases(query_rows: list[dict[str, Any]], raw_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in query_rows:
        preferred = row["query_profile"].get("preferred_answer_roles", [])
        if not preferred:
            continue
        primary_top = raw_rows[row["query_id"]]["primary_ranked"][0]
        primary_role = primary_top["record"].get("governance_tags_v4", {}).get("answer_role", "none")
        if primary_role in preferred or primary_top["relevance"] >= 3:
            continue
        cases.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "preferred_answer_roles": preferred,
                "primary_chunk_id": primary_top["chunk_id"],
                "primary_answer_role": primary_role,
                "primary_relevance": primary_top["relevance"],
            }
        )
    return cases


def same_topic_wrong_answer_cases(query_rows: list[dict[str, Any]], raw_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in query_rows:
        primary_top = raw_rows[row["query_id"]]["primary_ranked"][0]
        matrix_top = raw_rows[row["query_id"]]["reranked"][0]
        query_scene = scene_keywords(row["query"])
        primary_text = normalize_text(primary_document_text(primary_top["record"]))
        if not query_scene or not any(token in primary_text for token in query_scene):
            continue
        primary_role = primary_top["record"].get("governance_tags_v4", {}).get("answer_role", "none")
        preferred_roles = row["query_profile"].get("preferred_answer_roles", [])
        if primary_role in preferred_roles:
            continue
        cases.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "scene_tokens": query_scene,
                "primary_chunk_id": primary_top["chunk_id"],
                "primary_answer_role": primary_role,
                "primary_relevance": primary_top["relevance"],
                "matrix_chunk_id": matrix_top["chunk_id"],
                "matrix_answer_role": matrix_top["record"].get("governance_tags_v4", {}).get("answer_role", "none"),
                "matrix_relevance": matrix_top["relevance"],
            }
        )
    return cases


def strong_primary_but_penalized_cases(raw_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for query_id, payload in raw_rows.items():
        for row in payload["reranked"]:
            if row["primary_score"] < 18:
                continue
            if row["matrix_delta_clamped"] >= -0.25:
                continue
            cases.append(
                {
                    "query_id": query_id,
                    "query": payload["query"],
                    "chunk_id": row["chunk_id"],
                    "primary_score": round(row["primary_score"], 4),
                    "matrix_delta": round(row["matrix_delta_clamped"], 4),
                    "final_score": round(row["final_score"], 4),
                    "relevance": row["relevance"],
                    "governance_tags_v4": row["record"].get("governance_tags_v4", {}),
                    "matrix_score_breakdown": row["matrix_breakdown"],
                }
            )
    cases.sort(key=lambda item: (item["matrix_delta"], -item["primary_score"]))
    return cases[:12]


def queries_where_matrix_helps(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    helped = []
    for row in query_rows:
        primary = row["methods"]["primary_only"]
        matrix = row["methods"]["primary_plus_matrix"]
        if float(matrix["mean_relevance_at_k"]) > float(primary["mean_relevance_at_k"]) or float(matrix["ndcg_at_k"]) > float(primary["ndcg_at_k"]):
            helped.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "primary_ndcg_at_5": primary["ndcg_at_k"],
                    "matrix_ndcg_at_5": matrix["ndcg_at_k"],
                }
            )
    return helped


def queries_where_matrix_hurts(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hurt = []
    for row in query_rows:
        primary = row["methods"]["primary_only"]
        matrix = row["methods"]["primary_plus_matrix"]
        if float(matrix["mean_relevance_at_k"]) + 0.2 < float(primary["mean_relevance_at_k"]) or float(matrix["ndcg_at_k"]) + 0.05 < float(primary["ndcg_at_k"]):
            hurt.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "primary_ndcg_at_5": primary["ndcg_at_k"],
                    "matrix_ndcg_at_5": matrix["ndcg_at_k"],
                }
            )
    return hurt


def field_contributions(query_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positive = Counter()
    negative = Counter()
    for row in query_rows:
        for result in row["methods"]["primary_plus_matrix"]["top_results"]:
            matrix = dict(result.get("matrix_score_breakdown", {}))
            for field in ("answer_role", "root_issue", "intent", "target_role", "role_profile"):
                delta = float(dict(matrix.get(field, {})).get("delta", 0.0))
                if delta > 0:
                    positive[field] += delta
                elif delta < 0:
                    negative[field] += abs(delta)
    return {
        "positive": [{"field": field, "score_sum": round(score, 4)} for field, score in positive.most_common()],
        "negative": [{"field": field, "score_sum": round(score, 4)} for field, score in negative.most_common()],
    }


def tag_stats(records: list[dict[str, Any]], baseline_v2_records: list[dict[str, Any]] | None) -> dict[str, Any]:
    current_root = Counter(str(record.get("governance_tags_v4", {}).get("root_issue", "none")) for record in records)
    current_answer = Counter(str(record.get("governance_tags_v4", {}).get("answer_role", "none")) for record in records)
    current_intent = Counter(str(record.get("governance_tags_v4", {}).get("intent", "none")) for record in records)
    total = max(1, len(records))
    stats = {
        "root_issue_counts": dict(current_root),
        "answer_role_counts": dict(current_answer),
        "intent_counts": dict(current_intent),
        "mixed_ratio_v4": round(current_root.get("mixed", 0) / total, 4),
    }
    if baseline_v2_records is not None:
        baseline_root = Counter(str(record.get("governance_tags_v2", {}).get("root_issue", "none")) for record in baseline_v2_records)
        baseline_total = max(1, len(baseline_v2_records))
        stats["mixed_ratio_v3"] = round(baseline_root.get("mixed", 0) / baseline_total, 4)
    return stats


def maybe_load_v3_baseline() -> dict[str, Any] | None:
    if not DEFAULT_BASELINE_V3_REPORT.exists():
        return None
    baseline = load_json(DEFAULT_BASELINE_V3_REPORT)
    return {
        "aggregate": baseline.get("aggregate", {}),
        "breakdown_by_query_type": baseline.get("breakdown_by_query_type", {}),
    }


def maybe_load_v2_tags() -> list[dict[str, Any]] | None:
    if not DEFAULT_BASELINE_V2_TAGS.exists():
        return None
    return load_json(DEFAULT_BASELINE_V2_TAGS)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.top_m <= 0:
        raise ValueError("--top-k and --top-m must be greater than 0")
    if args.top_m < args.top_k:
        raise ValueError("--top-m must be >= --top-k")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    if not args.query_set_path.exists():
        raise FileNotFoundError(f"Query set file not found: {args.query_set_path}")

    records = load_json(args.input_path)
    queries = load_json(args.query_set_path)
    if args.query_id:
        queries = [query for query in queries if str(query.get("id", "")) == args.query_id]
        if not queries:
            raise ValueError(f"Unknown query id: {args.query_id}")

    phrase_lexicon = build_phrase_lexicon(records, queries)
    idf = build_idf(records, phrase_lexicon)

    report_queries: list[dict[str, Any]] = []
    raw_rows: dict[str, dict[str, Any]] = {}
    for query in queries:
        methods, primary_ranked, reranked, profile = evaluate_query(records, query, idf, phrase_lexicon, args.top_k, args.top_m)
        query_id = str(query.get("id", ""))
        report_queries.append(
            {
                "query_id": query_id,
                "query": str(query.get("query", "")),
                "category": str(query.get("category", query_type(str(query.get("query", ""))))),
                "query_profile": profile,
                "methods": methods,
            }
        )
        raw_rows[query_id] = {"query": str(query.get("query", "")), "primary_ranked": primary_ranked, "reranked": reranked}

    v3_baseline = maybe_load_v3_baseline()
    v2_tags = maybe_load_v2_tags()
    aggregate = aggregate_metrics(report_queries)
    report = {
        "input_path": str(args.input_path),
        "query_set_path": str(args.query_set_path),
        "top_k": args.top_k,
        "top_m": args.top_m,
        "total_records": len(records),
        "unique_sources": len({str(item.get('source_file', '')) for item in records}),
        "methods": list(METHODS),
        "aggregate": aggregate,
        "breakdown_by_query_type": breakdown_by_query_type(report_queries),
        "field_contributions": field_contributions(report_queries),
        "queries_where_matrix_helps": queries_where_matrix_helps(report_queries),
        "queries_where_matrix_hurts": queries_where_matrix_hurts(report_queries),
        "role_mismatch_cases": role_mismatch_cases(report_queries, raw_rows),
        "same_topic_wrong_answer_cases": same_topic_wrong_answer_cases(report_queries, raw_rows),
        "strong_primary_but_penalized_cases": strong_primary_but_penalized_cases(raw_rows),
        "tag_stats": tag_stats(records, v2_tags),
        "baseline_v3": v3_baseline,
        "queries": report_queries,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input file: {args.input_path}")
    print(f"Query set: {args.query_set_path}")
    print(f"Total records: {len(records)}")
    print(f"Unique sources: {len({str(item.get('source_file', '')) for item in records})}")
    print(f"Queries evaluated: {len(report_queries)}")
    print(f"Saved v4 report to: {args.output_path}")
    print("Aggregate summary:")
    for method in METHODS:
        summary = aggregate[method]
        print(
            f"- {method}: hit_rate={summary['hit_rate']:.4f}, "
            f"top1_hit_rate={summary['top1_hit_rate']:.4f}, "
            f"mean_relevance@{args.top_k}={summary['mean_relevance_at_5']:.4f}, "
            f"mean_ndcg@{args.top_k}={summary['mean_ndcg_at_5']:.4f}"
        )
    print(f"mixed_ratio_v4={report['tag_stats']['mixed_ratio_v4']}")
    if "mixed_ratio_v3" in report["tag_stats"]:
        print(f"mixed_ratio_v3={report['tag_stats']['mixed_ratio_v3']}")


if __name__ == "__main__":
    main()
