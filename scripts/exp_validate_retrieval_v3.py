#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v2/tagged_chunks/records.json")
DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v1/retrieval_eval_v2/query_set.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v2/retrieval_eval_v3/report.json")
DEFAULT_BASELINE_V2_REPORT = Path("data/pipeline_candidates/v1/retrieval_eval_v2/report.json")
DEFAULT_TOP_K = 5
DEFAULT_TOP_M = 15

METHODS = ("primary_only", "primary_plus_matrix")

ANSWER_ROLE_WEIGHTS = {
    "definition": 2.4,
    "cause": 2.4,
    "mechanism": 2.2,
    "solution": 2.2,
    "principle": 1.6,
    "example": 1.4,
    "warning": 1.8,
    "comparison": 1.8,
    "summary": 1.2,
    "none": 0.0,
}

MATRIX_FIELD_WEIGHTS = {
    "answer_role": 2.4,
    "root_issue": 1.4,
    "intent": 1.4,
    "gov_mode": 0.9,
    "solution_type": 0.9,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage offline retrieval validation for governance matrix MVP.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input v2 tagged chunks path.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH, help="Query set JSON path.")
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
        "preferred_root_issues": [],
        "preferred_gov_modes": [],
        "preferred_intents": [],
        "preferred_solution_types": [],
        "avoid_answer_roles": [],
    }

    if any(token in normalized for token in ("为什么", "为何")):
        profile["preferred_answer_roles"] = ["cause", "mechanism", "warning", "comparison"]
        profile["avoid_answer_roles"] = ["definition", "summary"]
    elif any(token in normalized for token in ("怎么", "如何")):
        profile["preferred_answer_roles"] = ["solution", "mechanism", "example"]
        profile["avoid_answer_roles"] = ["definition"]
    elif any(token in normalized for token in ("什么是", "本质", "定义")):
        profile["preferred_answer_roles"] = ["definition", "principle", "summary"]
        profile["avoid_answer_roles"] = ["example"]

    if any(token in normalized for token in ("长期", "机制", "体系", "系统性", "制度", "规则", "SOP")):
        profile["preferred_gov_modes"].append("rule_of_law")
        profile["preferred_solution_types"].append("mechanism_building")
    if any(token in normalized for token in ("救火", "短期", "临时", "先推进", "先解决", "先顶住")):
        profile["preferred_gov_modes"].extend(["rule_of_man", "hybrid"])
        profile["preferred_solution_types"].extend(["authority_borrowing", "risk_escalation"])

    if any(token in normalized for token in ("跨部门", "协作", "部门墙")):
        profile["preferred_intents"].append("cross_function_alignment")
        profile["preferred_root_issues"].append("information_distortion")
    if any(token in normalized for token in ("资源协调", "资源", "依赖", "协调")):
        profile["preferred_intents"].append("resource_coordination")
    if any(token in normalized for token in ("绩效设计", "OKR", "考核设计", "绩效指标")):
        profile["preferred_intents"].append("performance_design")
    if any(token in normalized for token in ("绩效补救", "绩效失效", "评价失效", "复盘", "纠偏")):
        profile["preferred_intents"].append("performance_repair")
        profile["preferred_root_issues"].append("evaluation_failure")
    if any(token in normalized for token in ("梯队", "接班", "储备干部", "后备")):
        profile["preferred_intents"].append("succession")
    if any(token in normalized for token in ("带人", "培养", "辅导", "成长")):
        profile["preferred_intents"].append("coaching")
    if any(token in normalized for token in ("机制", "制度", "SOP", "规则", "流程")):
        profile["preferred_intents"].append("mechanism_design")
        profile["preferred_gov_modes"].append("rule_of_law")
    if any(token in normalized for token in ("信息失真", "沟通", "汇报", "传导")):
        profile["preferred_root_issues"].append("information_distortion")
    if any(token in normalized for token in ("评价失效", "评价", "奖惩", "价值")):
        profile["preferred_root_issues"].append("evaluation_failure")

    for key in (
        "preferred_answer_roles",
        "preferred_root_issues",
        "preferred_gov_modes",
        "preferred_intents",
        "preferred_solution_types",
        "avoid_answer_roles",
    ):
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
        delta = -weight * 0.25
        return delta, {"field_value": value, "decision": "missing", "delta": round(delta, 4)}
    if preferred:
        delta = -weight * 0.55
        return delta, {"field_value": value, "decision": "mismatch", "delta": round(delta, 4)}
    return 0.0, {"field_value": value, "decision": "neutral", "delta": 0.0}


def matrix_adjustment(record: dict[str, Any], profile: dict[str, Any], primary_score: float) -> tuple[float, float, dict[str, Any]]:
    tags = dict(record.get("governance_tags_v2", {}))
    components: dict[str, Any] = {}
    raw_delta = 0.0

    answer_delta, answer_info = score_matrix_field(
        value=str(tags.get("answer_role", "none")),
        preferred=list(profile.get("preferred_answer_roles", [])),
        avoid=list(profile.get("avoid_answer_roles", [])),
        weight=MATRIX_FIELD_WEIGHTS["answer_role"],
    )
    raw_delta += answer_delta
    components["answer_role"] = answer_info

    root_delta, root_info = score_matrix_field(
        value=str(tags.get("root_issue", "none")),
        preferred=list(profile.get("preferred_root_issues", [])),
        avoid=[],
        weight=MATRIX_FIELD_WEIGHTS["root_issue"],
    )
    raw_delta += root_delta
    components["root_issue"] = root_info

    intent_delta, intent_info = score_matrix_field(
        value=str(tags.get("intent", "none")),
        preferred=list(profile.get("preferred_intents", [])),
        avoid=[],
        weight=MATRIX_FIELD_WEIGHTS["intent"],
    )
    raw_delta += intent_delta
    components["intent"] = intent_info

    mode_delta, mode_info = score_matrix_field(
        value=str(tags.get("gov_mode", "none")),
        preferred=list(profile.get("preferred_gov_modes", [])),
        avoid=[],
        weight=MATRIX_FIELD_WEIGHTS["gov_mode"],
    )
    raw_delta += mode_delta
    components["gov_mode"] = mode_info

    solution_delta, solution_info = score_matrix_field(
        value=str(tags.get("solution_type", "none")),
        preferred=list(profile.get("preferred_solution_types", [])),
        avoid=[],
        weight=MATRIX_FIELD_WEIGHTS["solution_type"],
    )
    raw_delta += solution_delta
    components["solution_type"] = solution_info

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
        rows.append(
            {
                "insight_text": text,
                "score": round(raw_score, 4),
                "matched_tokens": matched[:6],
            }
        )
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
        "governance_tags_v2": record.get("governance_tags_v2", {}),
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
        result = build_result_row(
            record=record,
            relevance=relevance,
            primary_score=primary_score,
            final_score=primary_score,
            primary_breakdown=primary_breakdown,
            matrix_breakdown={},
            query_tokens=query_tokens,
            idf=idf,
            phrase_lexicon=phrase_lexicon,
        )
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
                "matrix_delta_raw": 0.0,
                "matrix_delta_clamped": 0.0,
            }
        )

    primary_ranked.sort(key=lambda item: (-float(item["primary_score"]), -int(item["relevance"]), str(item["chunk_id"])))
    primary_summary = summarize_ranking(primary_ranked, top_k=top_k)

    rerank_pool = primary_ranked[:top_m]
    reranked_rows: list[dict[str, Any]] = []
    for row in rerank_pool:
        raw_delta, clamped_delta, matrix_breakdown = matrix_adjustment(row["record"], profile, row["primary_score"])
        final_score = row["primary_score"] + clamped_delta
        result = build_result_row(
            record=row["record"],
            relevance=row["relevance"],
            primary_score=row["primary_score"],
            final_score=final_score,
            primary_breakdown=row["primary_breakdown"],
            matrix_breakdown=matrix_breakdown,
            query_tokens=query_tokens,
            idf=idf,
            phrase_lexicon=phrase_lexicon,
        )
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
    matrix_summary = summarize_ranking(reranked_rows, top_k=top_k)

    methods = {
        "primary_only": primary_summary,
        "primary_plus_matrix": matrix_summary,
    }
    return methods, primary_ranked, reranked_rows, profile


def aggregate_metrics(query_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregate: dict[str, dict[str, float]] = {
        method: {
            "hit_rate": 0.0,
            "top1_hit_rate": 0.0,
            "mean_relevance_at_5": 0.0,
            "mean_ndcg_at_5": 0.0,
        }
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


def field_contributions(query_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positive = Counter()
    negative = Counter()
    for row in query_rows:
        for result in row["methods"]["primary_plus_matrix"]["top_results"]:
            matrix = dict(result.get("matrix_score_breakdown", {}))
            for field in ("answer_role", "root_issue", "intent", "gov_mode", "solution_type"):
                payload = dict(matrix.get(field, {}))
                delta = float(payload.get("delta", 0.0))
                if delta > 0:
                    positive[field] += delta
                elif delta < 0:
                    negative[field] += abs(delta)
    return {
        "positive": [{"field": field, "score_sum": round(score, 4)} for field, score in positive.most_common()],
        "negative": [{"field": field, "score_sum": round(score, 4)} for field, score in negative.most_common()],
    }


def detect_queries_where_matrix_helps(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    helped = []
    for row in query_rows:
        primary = row["methods"]["primary_only"]
        matrix = row["methods"]["primary_plus_matrix"]
        if (
            int(matrix["top1_relevance"]) > int(primary["top1_relevance"])
            or float(matrix["mean_relevance_at_k"]) > float(primary["mean_relevance_at_k"])
        ):
            helped.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "primary_top1_relevance": primary["top1_relevance"],
                    "matrix_top1_relevance": matrix["top1_relevance"],
                    "primary_mean_relevance_at_5": primary["mean_relevance_at_k"],
                    "matrix_mean_relevance_at_5": matrix["mean_relevance_at_k"],
                }
            )
    return helped


def detect_queries_where_matrix_hurts(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hurt = []
    for row in query_rows:
        primary = row["methods"]["primary_only"]
        matrix = row["methods"]["primary_plus_matrix"]
        if (
            int(matrix["top1_relevance"]) < int(primary["top1_relevance"])
            or float(matrix["mean_relevance_at_k"]) + 0.2 < float(primary["mean_relevance_at_k"])
        ):
            hurt.append(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "primary_top1_relevance": primary["top1_relevance"],
                    "matrix_top1_relevance": matrix["top1_relevance"],
                    "primary_mean_relevance_at_5": primary["mean_relevance_at_k"],
                    "matrix_mean_relevance_at_5": matrix["mean_relevance_at_k"],
                }
            )
    return hurt


def detect_same_scene_misleading_cases(query_rows: list[dict[str, Any]], raw_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for row in query_rows:
        primary = row["methods"]["primary_only"]
        matrix = row["methods"]["primary_plus_matrix"]
        if int(primary["top1_relevance"]) >= 3 or int(matrix["top1_relevance"]) <= int(primary["top1_relevance"]):
            continue
        primary_top = raw_rows[row["query_id"]]["primary_ranked"][0]
        matrix_top = raw_rows[row["query_id"]]["reranked"][0]
        primary_tags = dict(primary_top["record"].get("legacy_tags", {}))
        matrix_tags = dict(matrix_top["record"].get("governance_tags_v2", {}))
        cases.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "misleading_primary_chunk_id": primary_top["chunk_id"],
                "misleading_primary_title": str(primary_top["record"].get("title", "")),
                "misleading_primary_relevance": primary_top["relevance"],
                "misleading_primary_legacy_tags": {
                    "topic_tags": primary_tags.get("topic_tags", []),
                    "scenario_tags": primary_tags.get("scenario_tags", []),
                },
                "matrix_promoted_chunk_id": matrix_top["chunk_id"],
                "matrix_promoted_relevance": matrix_top["relevance"],
                "matrix_promoted_governance_tags": matrix_tags,
            }
        )
    return cases


def detect_high_primary_but_penalized_cases(raw_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for query_id, payload in raw_rows.items():
        for row in payload["reranked"]:
            if row["primary_score"] < 18:
                continue
            if row["matrix_delta_clamped"] >= -0.3:
                continue
            cases.append(
                {
                    "query_id": query_id,
                    "query": payload["query"],
                    "chunk_id": row["chunk_id"],
                    "title": str(row["record"].get("title", "")),
                    "primary_score": round(row["primary_score"], 4),
                    "matrix_delta": round(row["matrix_delta_clamped"], 4),
                    "final_score": round(row["final_score"], 4),
                    "relevance": row["relevance"],
                    "governance_tags_v2": row["record"].get("governance_tags_v2", {}),
                    "matrix_score_breakdown": row["matrix_breakdown"],
                }
            )
    cases.sort(key=lambda item: (item["matrix_delta"], -item["primary_score"]))
    return cases[:12]


def maybe_load_v2_baseline() -> dict[str, Any] | None:
    if not DEFAULT_BASELINE_V2_REPORT.exists():
        return None
    baseline = load_json(DEFAULT_BASELINE_V2_REPORT)
    return dict(baseline.get("aggregate", {}))


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
        methods, primary_ranked, reranked_rows, profile = evaluate_query(
            records=records,
            query=query,
            idf=idf,
            phrase_lexicon=phrase_lexicon,
            top_k=args.top_k,
            top_m=args.top_m,
        )
        query_id = str(query.get("id", ""))
        report_queries.append(
            {
                "query_id": query_id,
                "query": str(query.get("query", "")),
                "category": str(query.get("category", query_type(str(query.get("query", ""))))),
                "intent": str(query.get("intent", query_type(str(query.get("query", ""))))),
                "query_profile": profile,
                "methods": methods,
            }
        )
        raw_rows[query_id] = {
            "query": str(query.get("query", "")),
            "primary_ranked": primary_ranked,
            "reranked": reranked_rows,
        }

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
        "queries_where_matrix_helps": detect_queries_where_matrix_helps(report_queries),
        "queries_where_matrix_hurts": detect_queries_where_matrix_hurts(report_queries),
        "same_scene_misleading_cases": detect_same_scene_misleading_cases(report_queries, raw_rows),
        "high_primary_but_penalized_cases": detect_high_primary_but_penalized_cases(raw_rows),
        "baseline_v2_aggregate": maybe_load_v2_baseline(),
        "queries": report_queries,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input file: {args.input_path}")
    print(f"Query set: {args.query_set_path}")
    print(f"Total records: {len(records)}")
    print(f"Unique sources: {len({str(item.get('source_file', '')) for item in records})}")
    print(f"Queries evaluated: {len(report_queries)}")
    print(f"Saved v3 report to: {args.output_path}")
    print("Aggregate summary:")
    for method in METHODS:
        summary = aggregate[method]
        print(
            f"- {method}: hit_rate={summary['hit_rate']:.4f}, "
            f"top1_hit_rate={summary['top1_hit_rate']:.4f}, "
            f"mean_relevance@{args.top_k}={summary['mean_relevance_at_5']:.4f}, "
            f"mean_ndcg@{args.top_k}={summary['mean_ndcg_at_5']:.4f}"
        )


if __name__ == "__main__":
    main()
