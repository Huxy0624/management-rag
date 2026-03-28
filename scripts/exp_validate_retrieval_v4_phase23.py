#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Any

import exp_validate_retrieval_v4_phase22 as phase22


base = phase22.base

base.DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v4_phase22/tagged_chunks/records.json")
base.DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
base.DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v4_phase23/report.json")

STRONG_PRIMARY_CRITERIA = {
    "min_primary_score": 44.0,
    "max_primary_rank": 2,
    "min_gap_to_next": 4.0,
    "or_min_score_ratio_to_next": 1.08,
}

META_LIKE_KEYWORDS = (
    "管理本质",
    "核心",
    "根源",
    "下限",
    "上限",
    "信息失真",
    "评价失效",
    "人治",
    "法治",
    "平衡",
    "治理框架",
    "治理演进",
    "公司阶段",
    "从经理到总监",
    "角色跃迁",
    "管理视角",
    "系统性思考",
    "结构影响行为",
    "机制和架构",
)

PENALTY_DECAY_POLICY = {
    "strong_primary_global_decay": 0.34,
    "meta_high_primary_decay": 0.28,
    "meta_mid_primary_decay": 0.44,
    "answer_role_tier_decay": {
        "light": 0.22,
        "medium": 0.4,
        "heavy": 0.82,
    },
    "intent_tier_decay": {
        "light": 0.2,
        "medium": 0.45,
        "heavy": 0.82,
    },
    "role_anchor_tier_decay": {
        "light": 0.25,
        "medium": 0.5,
        "heavy": 0.88,
    },
    "meta_root_issue_mixed_decay": 0.5,
    "meta_root_issue_none_decay": 0.72,
}

MISMATCH_TIER_POLICY = {
    "light": [
        "mechanism vs principle",
        "principle vs mechanism",
        "definition vs principle",
        "principle vs definition",
        "meta-like definition vs mechanism",
    ],
    "medium": [
        "definition vs mechanism",
        "mechanism vs solution",
        "how vs principle",
        "what vs mechanism on non-meta chunk",
    ],
    "heavy": [
        "how query vs definition/summary",
        "why query vs pure solution/example",
        "what query vs pure action solution",
    ],
}


def parse_query_profile(query_text: str) -> dict[str, Any]:
    return phase22.parse_query_profile(query_text)


def derive_meta_like(record: dict[str, Any]) -> dict[str, Any]:
    text = base.normalize_text(
        " ".join(
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
    )
    hits = [keyword for keyword in META_LIKE_KEYWORDS if keyword in text]
    return {"meta_like": bool(hits), "meta_hits": hits[:8]}


def answer_role_mismatch_tier(query_type: str, answer_role: str, preferred_roles: list[str], meta_like: bool) -> str:
    if answer_role in preferred_roles:
        return "match"
    if query_type == "how":
        if answer_role in {"definition", "summary"}:
            return "heavy"
        if answer_role in {"principle", "warning"}:
            return "medium"
    if query_type == "why":
        if answer_role in {"solution", "example", "summary"}:
            return "heavy"
        if answer_role == "definition":
            return "light" if meta_like else "medium"
    if query_type == "what":
        if answer_role == "solution":
            return "heavy"
        if answer_role == "mechanism":
            return "light" if meta_like else "medium"
        if answer_role == "example":
            return "medium"
    semantic_cluster = {"definition", "principle", "mechanism", "cause"}
    if answer_role in semantic_cluster and any(role in semantic_cluster for role in preferred_roles):
        return "light"
    return "medium"


def intent_mismatch_tier(preferred_intents: list[str], intent: str, meta_like: bool) -> str:
    if not preferred_intents or intent in preferred_intents:
        return "match"
    if intent == "none":
        return "light"
    if meta_like and (intent == "mechanism_design" or "mechanism_design" in preferred_intents):
        return "light"
    if {intent, *preferred_intents} & {"mechanism_design", "resource_coordination"}:
        return "medium"
    return "heavy"


def strong_primary_context(rows: list[dict[str, Any]], row_index: int) -> dict[str, Any]:
    row = rows[row_index]
    primary_score = float(row.get("primary_score", 0.0))
    next_score = float(rows[row_index + 1]["primary_score"]) if row_index + 1 < len(rows) else None
    ratio = (primary_score / next_score) if next_score not in (None, 0.0) else None
    gap_to_next = primary_score - next_score if next_score is not None else primary_score
    is_strong = bool(
        primary_score >= STRONG_PRIMARY_CRITERIA["min_primary_score"]
        and row_index + 1 <= STRONG_PRIMARY_CRITERIA["max_primary_rank"]
        and (
            gap_to_next >= STRONG_PRIMARY_CRITERIA["min_gap_to_next"]
            or (ratio is not None and ratio >= STRONG_PRIMARY_CRITERIA["or_min_score_ratio_to_next"])
        )
    )
    return {
        "primary_rank": row_index + 1,
        "primary_score": round(primary_score, 4),
        "gap_to_next": round(gap_to_next, 4) if next_score is not None else None,
        "score_ratio_to_next": round(ratio, 4) if ratio is not None else None,
        "is_strong_primary": is_strong,
    }


def penalty_protection_scale(ranking_context: dict[str, Any], meta_like: bool) -> tuple[float, str]:
    primary_rank = int(ranking_context.get("primary_rank", 999))
    primary_score = float(ranking_context.get("primary_score", 0.0))
    if bool(ranking_context.get("is_strong_primary", False)):
        return PENALTY_DECAY_POLICY["strong_primary_global_decay"], "strong_primary"
    if meta_like and primary_score >= 55:
        return PENALTY_DECAY_POLICY["meta_high_primary_decay"], "meta_high_primary"
    if meta_like and primary_score >= 44 and primary_rank <= 5:
        return PENALTY_DECAY_POLICY["meta_mid_primary_decay"], "meta_mid_primary"
    return 1.0, "none"


def scaled_negative(delta: float, scale: float) -> float:
    if delta >= 0:
        return delta
    return round(delta * scale, 4)


def matrix_adjustment(
    record: dict[str, Any],
    profile: dict[str, Any],
    primary_score: float,
    ranking_context: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    _, _, components = phase22.matrix_adjustment(record, profile, primary_score)
    components = dict(components)
    tags = dict(record.get("governance_tags_v4", {}))
    meta_info = derive_meta_like(record)
    meta_like = bool(meta_info["meta_like"])
    query_type = str(profile.get("query_type", "what"))
    answer_role = str(tags.get("answer_role", "none"))
    intent = str(tags.get("intent", "none"))
    preferred_roles = list(profile.get("preferred_answer_roles", []))
    preferred_intents = list(profile.get("preferred_intents", []))
    preferred_root_issues = list(profile.get("preferred_root_issues", []))
    is_strong_primary = bool(ranking_context.get("is_strong_primary", False))

    answer_role_tier = answer_role_mismatch_tier(query_type, answer_role, preferred_roles, meta_like)
    intent_tier = intent_mismatch_tier(preferred_intents, intent, meta_like)
    global_negative_scale, protection_level = penalty_protection_scale(ranking_context, meta_like)

    if "answer_role" in components and isinstance(components["answer_role"], dict):
        delta = float(components["answer_role"].get("delta", 0.0))
        if delta < 0:
            scale = PENALTY_DECAY_POLICY["answer_role_tier_decay"].get(answer_role_tier, 1.0)
            components["answer_role"]["delta_before_phase23"] = round(delta, 4)
            components["answer_role"]["mismatch_tier"] = answer_role_tier
            components["answer_role"]["delta"] = scaled_negative(delta, scale * global_negative_scale)

    if "intent" in components and isinstance(components["intent"], dict):
        delta = float(components["intent"].get("delta", 0.0))
        if delta < 0:
            scale = PENALTY_DECAY_POLICY["intent_tier_decay"].get(intent_tier, 1.0)
            components["intent"]["delta_before_phase23"] = round(delta, 4)
            components["intent"]["mismatch_tier"] = intent_tier
            components["intent"]["delta"] = scaled_negative(delta, scale * global_negative_scale)

    if "root_issue" in components and isinstance(components["root_issue"], dict):
        delta = float(components["root_issue"].get("delta", 0.0))
        root_issue = str(tags.get("root_issue", "none"))
        if delta < 0 and preferred_root_issues:
            scale = 1.0
            if root_issue == "mixed":
                scale = PENALTY_DECAY_POLICY["meta_root_issue_mixed_decay"] if meta_like else 0.68
            elif root_issue == "none" and meta_like:
                scale = PENALTY_DECAY_POLICY["meta_root_issue_none_decay"]
            if protection_level != "none":
                scale *= global_negative_scale
            components["root_issue"]["delta_before_phase23"] = round(delta, 4)
            components["root_issue"]["delta"] = scaled_negative(delta, scale)

    if "role_anchor" in components and isinstance(components["role_anchor"], dict):
        delta = float(components["role_anchor"].get("delta", 0.0))
        if delta < 0:
            scale = PENALTY_DECAY_POLICY["role_anchor_tier_decay"].get(answer_role_tier, 1.0)
            components["role_anchor"]["delta_before_phase23"] = round(delta, 4)
            components["role_anchor"]["mismatch_tier"] = answer_role_tier
            components["role_anchor"]["delta"] = scaled_negative(delta, scale * global_negative_scale)

    raw_delta = 0.0
    for field, payload in components.items():
        if field == "clamp" or not isinstance(payload, dict):
            continue
        raw_delta += float(payload.get("delta", 0.0))

    lower_bound = -0.08 * primary_score
    upper_bound = 0.12 * primary_score
    clamped = max(lower_bound, min(upper_bound, raw_delta))
    components["phase23"] = {
        "meta_like": meta_like,
        "meta_hits": meta_info["meta_hits"],
        "is_strong_primary": is_strong_primary,
        "strong_primary_context": ranking_context,
        "penalty_protection_level": protection_level,
        "global_negative_scale": round(global_negative_scale, 4),
        "answer_role_mismatch_tier": answer_role_tier,
        "intent_mismatch_tier": intent_tier,
    }
    components["clamp"] = {
        "raw_delta": round(raw_delta, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "clamped_delta": round(clamped, 4),
    }
    return raw_delta, clamped, components


def evaluate_query(
    records: list[dict[str, Any]],
    query: dict[str, Any],
    idf: dict[str, float],
    phrase_lexicon: set[str],
    top_k: int,
    top_m: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_text = str(query.get("query", ""))
    query_tokens = base.extract_tokens(query_text, phrase_lexicon)
    profile = parse_query_profile(query_text)

    primary_ranked: list[dict[str, Any]] = []
    for record in records:
        primary_score, primary_breakdown = base.primary_score_record(record, query_tokens, idf, phrase_lexicon)
        relevance = base.judge_relevance(record, query)
        result = base.build_result_row(record, relevance, primary_score, primary_score, primary_breakdown, {}, query_tokens, idf, phrase_lexicon)
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
    primary_summary = base.summarize_ranking(primary_ranked, top_k)

    rerank_pool = primary_ranked[:top_m]
    reranked_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rerank_pool):
        ranking_context = strong_primary_context(rerank_pool, index)
        raw_delta, clamped_delta, matrix_breakdown = matrix_adjustment(row["record"], profile, row["primary_score"], ranking_context)
        final_score = row["primary_score"] + clamped_delta
        result = base.build_result_row(row["record"], row["relevance"], row["primary_score"], final_score, row["primary_breakdown"], matrix_breakdown, query_tokens, idf, phrase_lexicon)
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
    matrix_summary = base.summarize_ranking(reranked_rows, top_k)

    return {
        "primary_only": primary_summary,
        "primary_plus_matrix": matrix_summary,
    }, primary_ranked, reranked_rows, profile


base.parse_query_profile = parse_query_profile
base.evaluate_query = evaluate_query


if __name__ == "__main__":
    base.main()
