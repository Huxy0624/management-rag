#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import exp_validate_retrieval_v4_phase22 as phase22


DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v4_phase22/tagged_chunks/records.json")
DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
DEFAULT_SOURCE_REPORT_PATH = Path("data/pipeline_candidates/v4_phase22/retrieval_eval_v4/report.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v4_phase23/error_audit_strong_primary.json")
DEFAULT_TOP_K = 5
DEFAULT_TOP_M = 15

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit strong-primary-but-penalized cases before phase23 local rerank patch.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input phase22 tagged records.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH, help="Query set path.")
    parser.add_argument("--source-report-path", type=Path, default=DEFAULT_SOURCE_REPORT_PATH, help="Phase22 report path for reference.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output audit JSON path.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k for evaluation.")
    parser.add_argument("--top-m", type=int, default=DEFAULT_TOP_M, help="Top-m rerank pool size.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def derive_meta_like(record: dict[str, Any]) -> dict[str, Any]:
    text = phase22.base.normalize_text(
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
    return {"meta_like": bool(hits), "meta_hits": hits}


def mismatch_tier(query_type: str, answer_role: str, preferred_roles: list[str], meta_like: bool) -> str:
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
    if {answer_role, *preferred_roles} & {"definition", "principle", "mechanism", "cause"}:
        return "light"
    return "medium"


def is_light_misalignment(query_type: str, answer_role: str, preferred_roles: list[str], meta_like: bool) -> bool:
    return mismatch_tier(query_type, answer_role, preferred_roles, meta_like) == "light"


def strong_primary_context(rows: list[dict[str, Any]], row_index: int) -> dict[str, Any]:
    row = rows[row_index]
    primary_score = float(row.get("primary_score", 0.0))
    next_score = float(rows[row_index + 1]["primary_score"]) if row_index + 1 < len(rows) else None
    top_score = float(rows[0]["primary_score"]) if rows else primary_score
    gap_to_next = primary_score - next_score if next_score is not None else primary_score
    top_gap = top_score - primary_score
    is_strong = bool(
        primary_score >= 44
        and row_index <= 1
        and (gap_to_next >= 4.0 or (next_score is not None and primary_score >= next_score * 1.08))
    )
    return {
        "primary_rank": row_index + 1,
        "primary_score": round(primary_score, 4),
        "gap_to_next": round(gap_to_next, 4) if next_score is not None else None,
        "gap_from_top": round(top_gap, 4),
        "is_strong_primary": is_strong,
        "criteria": {
            "min_primary_score": 44.0,
            "max_primary_rank": 2,
            "min_gap_to_next": 4.0,
            "or_min_score_ratio_to_next": 1.08,
        },
    }


def negative_components(matrix_breakdown: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, payload in matrix_breakdown.items():
        if not isinstance(payload, dict) or field == "clamp":
            continue
        delta = float(payload.get("delta", 0.0))
        if delta >= 0:
            continue
        entry = {
            "field": field,
            "delta": round(delta, 4),
            "decision": str(payload.get("decision", "")),
        }
        if field == "role_anchor":
            entry["reasons"] = [str(item) for item in payload.get("reasons", []) if item]
        else:
            entry["field_value"] = payload.get("field_value")
        rows.append(entry)
    rows.sort(key=lambda item: item["delta"])
    return rows


def why_penalized(case: dict[str, Any], meta_like_info: dict[str, Any], preferred_roles: list[str], query_type: str) -> str:
    answer_role = str(case.get("governance_tags_v4", {}).get("answer_role", "none"))
    tier = mismatch_tier(query_type, answer_role, preferred_roles, bool(meta_like_info["meta_like"]))
    parts = [f"answer_role={answer_role} vs query_type={query_type} is treated as {tier} mismatch"]
    negatives = negative_components(dict(case.get("matrix_score_breakdown", {})))
    if negatives:
        parts.append(f"largest penalty from {negatives[0]['field']}{negatives[0]['delta']:+.2f}")
    if meta_like_info["meta_like"]:
        parts.append("chunk is meta-like but current rerank still applies ordinary mismatch penalties")
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    records = load_json(args.input_path)
    queries = load_json(args.query_set_path)
    phrase_lexicon = phase22.base.build_phrase_lexicon(records, queries)
    idf = phase22.base.build_idf(records, phrase_lexicon)

    query_rows: list[dict[str, Any]] = []
    raw_rows: dict[str, dict[str, Any]] = {}
    for query in queries:
        methods, primary_ranked, reranked, profile = phase22.base.evaluate_query(records, query, idf, phrase_lexicon, args.top_k, args.top_m)
        query_id = str(query.get("id", ""))
        query_rows.append(
            {
                "query_id": query_id,
                "query": str(query.get("query", "")),
                "query_profile": profile,
                "methods": methods,
            }
        )
        raw_rows[query_id] = {
            "query": str(query.get("query", "")),
            "primary_ranked": primary_ranked,
            "reranked": reranked,
        }

    query_map = {row["query_id"]: row for row in query_rows}
    audit_cases: list[dict[str, Any]] = []
    for case in phase22.base.strong_primary_but_penalized_cases(raw_rows):
        query_id = str(case.get("query_id", ""))
        query_entry = query_map[query_id]
        primary_rows = raw_rows[query_id]["primary_ranked"]
        reranked_rows = raw_rows[query_id]["reranked"]
        chunk_id = str(case.get("chunk_id", ""))
        primary_index = next(index for index, row in enumerate(primary_rows) if str(row.get("chunk_id", "")) == chunk_id)
        final_index = next(index for index, row in enumerate(reranked_rows) if str(row.get("chunk_id", "")) == chunk_id)
        primary_row = primary_rows[primary_index]
        final_row = reranked_rows[final_index]
        record = dict(primary_row.get("record", {}))
        meta_like_info = derive_meta_like(record)
        query_type = str(query_entry["query_profile"].get("query_type", "what"))
        preferred_roles = list(query_entry["query_profile"].get("preferred_answer_roles", []))
        context = strong_primary_context(primary_rows, primary_index)
        audit_cases.append(
            {
                "query_id": query_id,
                "query": case.get("query"),
                "query_profile": query_entry["query_profile"],
                "chunk_id": chunk_id,
                "title": str(record.get("title", "")),
                "primary_rank": primary_index + 1,
                "final_rank": final_index + 1,
                "primary_score": round(float(primary_row.get("primary_score", 0.0)), 4),
                "final_score": round(float(final_row.get("final_score", 0.0)), 4),
                "score_drop": round(float(primary_row.get("primary_score", 0.0)) - float(final_row.get("final_score", 0.0)), 4),
                "strong_primary_context": context,
                "current_labels": {
                    "answer_role": str(case.get("governance_tags_v4", {}).get("answer_role", "none")),
                    "intent": str(case.get("governance_tags_v4", {}).get("intent", "none")),
                    "root_issue": str(case.get("governance_tags_v4", {}).get("root_issue", "none")),
                },
                "penalty_sources": negative_components(dict(case.get("matrix_score_breakdown", {}))),
                "meta_like": meta_like_info,
                "mismatch_tier": mismatch_tier(
                    query_type,
                    str(case.get("governance_tags_v4", {}).get("answer_role", "none")),
                    preferred_roles,
                    bool(meta_like_info["meta_like"]),
                ),
                "light_misalignment_but_heavily_penalized": (
                    is_light_misalignment(
                        query_type,
                        str(case.get("governance_tags_v4", {}).get("answer_role", "none")),
                        preferred_roles,
                        bool(meta_like_info["meta_like"]),
                    )
                    and float(case.get("matrix_delta", 0.0)) <= -2.5
                ),
                "why_penalized": why_penalized(case, meta_like_info, preferred_roles, query_type),
                "chunk_text_preview": phase22.base.clip_text(str(record.get("chunk_text", "")), max_chars=260),
            }
        )

    summary = {
        "source_report": str(args.source_report_path),
        "total_cases": len(audit_cases),
        "meta_like_cases": sum(1 for case in audit_cases if case["meta_like"]["meta_like"]),
        "light_misalignment_but_heavily_penalized_cases": sum(
            1 for case in audit_cases if case["light_misalignment_but_heavily_penalized"]
        ),
        "strong_primary_by_heuristic": sum(1 for case in audit_cases if case["strong_primary_context"]["is_strong_primary"]),
    }

    output = {"summary": summary, "audit_cases": audit_cases}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved strong-primary audit to: {args.output_path}")
    print(f"Total cases: {len(audit_cases)}")


if __name__ == "__main__":
    main()
