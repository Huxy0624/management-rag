#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/report.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/error_audit_phase22.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v4 retrieval errors before local answer_role patching.")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH, help="Input v4 retrieval report.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output audit JSON path.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_tags(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("governance_tags_v4", {}))


def matrix_rule_traces(row: dict[str, Any]) -> list[str]:
    traces: list[str] = []
    matrix = dict(row.get("matrix_score_breakdown", {}))
    for field, payload in matrix.items():
        if not isinstance(payload, dict):
            continue
        if field == "clamp":
            continue
        if field == "role_anchor":
            reasons = [str(item) for item in payload.get("reasons", []) if item]
            if reasons:
                traces.extend(reasons)
            elif float(payload.get("delta", 0.0)) != 0:
                traces.append(f"role_anchor:{payload.get('decision')}:{float(payload.get('delta', 0.0)):+.2f}")
            continue
        delta = float(payload.get("delta", 0.0))
        if delta < 0:
            traces.append(f"{field}:{payload.get('decision')}:{delta:+.2f}")
    return traces


def summarize_labels(row: dict[str, Any]) -> dict[str, Any]:
    tags = extract_tags(row)
    return {
        "answer_role": tags.get("answer_role"),
        "intent": tags.get("intent"),
        "root_issue": tags.get("root_issue"),
        "target_role": tags.get("target_role", []),
        "role_profile": tags.get("role_profile", []),
    }


def summarize_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summarized: list[dict[str, Any]] = []
    for row in rows[:5]:
        final_score = float(row.get("final_score", row.get("primary_score", 0.0)) or 0.0)
        primary_score = float(row.get("primary_score", 0.0) or 0.0)
        summarized.append(
            {
                "chunk_id": str(row.get("chunk_id", "")),
                "title": str(row.get("title", "")),
                "primary_score": row.get("primary_score"),
                "final_score": row.get("final_score"),
                "matrix_delta": round(final_score - primary_score, 4),
                "relevance": row.get("relevance"),
                "current_labels": summarize_labels(row),
                "chunk_text_preview": str(row.get("chunk_text_preview", "")),
                "matrix_score_breakdown": row.get("matrix_score_breakdown", {}),
                "negative_rule_traces": matrix_rule_traces(row),
            }
        )
    return summarized


def best_relevance(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return max(int(row.get("relevance", 0) or 0) for row in rows[:5])


def first_row(query_entry: dict[str, Any], method: str) -> dict[str, Any]:
    rows = list(query_entry.get("methods", {}).get(method, {}).get("top_results", []))
    return dict(rows[0]) if rows else {}


def diagnose_case(query_entry: dict[str, Any], case_kind: str) -> dict[str, Any]:
    profile = dict(query_entry.get("query_profile", {}))
    primary_rows = list(query_entry.get("methods", {}).get("primary_only", {}).get("top_results", []))
    matrix_rows = list(query_entry.get("methods", {}).get("primary_plus_matrix", {}).get("top_results", []))
    primary_top = first_row(query_entry, "primary_only")
    matrix_top = first_row(query_entry, "primary_plus_matrix")
    primary_tags = extract_tags(primary_top)
    matrix_tags = extract_tags(matrix_top)
    preferred_roles = list(profile.get("preferred_answer_roles", []))
    preferred_intents = list(profile.get("preferred_intents", []))
    primary_relevance = int(primary_top.get("relevance", 0) or 0)
    matrix_relevance = int(matrix_top.get("relevance", 0) or 0)
    best_primary_relevance = best_relevance(primary_rows)
    best_matrix_relevance = best_relevance(matrix_rows)

    issue_fields: list[str] = []
    trigger_rules: list[str] = []
    recommended_action = "no_patch"
    likely_fault_source = "unknown"

    if preferred_roles and str(primary_tags.get("answer_role", "none")) not in preferred_roles:
        issue_fields.append("answer_role")
    if preferred_intents:
        current_intent = str(primary_tags.get("intent", "none"))
        if current_intent not in preferred_intents:
            issue_fields.append("intent")

    if case_kind == "strong_primary_but_penalized_cases":
        likely_fault_source = "rerank_rule"
        recommended_action = "tune_soft_rerank_or_local_relabel"
        trigger_rules.extend(matrix_rule_traces(matrix_top))
    elif primary_relevance < 3 and best_primary_relevance <= 2:
        likely_fault_source = "query_parser_or_primary_recall"
        recommended_action = "leave_for_mainline_recall_fix"
        trigger_rules.append("low_relevance_top1")
    elif "answer_role" in issue_fields and primary_relevance >= 3:
        likely_fault_source = "chunk_answer_role_rule"
        recommended_action = "local_answer_role_patch"
        trigger_rules.append("answer_role_boundary")
    elif "answer_role" in issue_fields and best_primary_relevance > primary_relevance:
        likely_fault_source = "query_parser_or_primary_recall"
        recommended_action = "keep_for_mainline_recall_fix"
        trigger_rules.append("higher_relevance_exists_but_not_top1")
    elif "answer_role" in issue_fields and "intent" not in issue_fields:
        likely_fault_source = "chunk_answer_role_rule"
        recommended_action = "local_answer_role_patch"
        trigger_rules.append("answer_role_boundary")
    elif "intent" in issue_fields and "answer_role" not in issue_fields:
        likely_fault_source = "chunk_intent_rule"
        recommended_action = "light_intent_check_only"
        trigger_rules.append("intent_mapping")
    elif issue_fields:
        likely_fault_source = "chunk_boundary_rule"
        recommended_action = "local_patch_if_text_supports"
        trigger_rules.extend(issue_fields)
    elif best_primary_relevance == 0 and best_matrix_relevance == 0:
        likely_fault_source = "query_parser_or_primary_recall"
        recommended_action = "leave_for_mainline_recall_fix"
        trigger_rules.append("no_relevant_top5")
    elif preferred_roles and preferred_intents:
        likely_fault_source = "query_parser_or_primary_recall"
        recommended_action = "leave_for_mainline_recall_fix"
        trigger_rules.append("query_profile_soft_constraint")

    return {
        "issue_fields": issue_fields,
        "likely_fault_source": likely_fault_source,
        "recommended_action": recommended_action,
        "trigger_rules": trigger_rules,
        "preferred_answer_roles": preferred_roles,
        "preferred_intents": preferred_intents,
        "primary_top_answer_role": primary_tags.get("answer_role"),
        "primary_top_intent": primary_tags.get("intent"),
        "matrix_top_answer_role": matrix_tags.get("answer_role"),
        "matrix_top_intent": matrix_tags.get("intent"),
        "primary_top_relevance": primary_relevance,
        "matrix_top_relevance": matrix_relevance,
        "best_primary_top5_relevance": best_primary_relevance,
        "best_matrix_top5_relevance": best_matrix_relevance,
        "primary_negative_rule_traces": matrix_rule_traces(primary_top),
        "matrix_negative_rule_traces": matrix_rule_traces(matrix_top),
    }


def build_case_entry(query_entry: dict[str, Any], case_kind: str) -> dict[str, Any]:
    diagnosis = diagnose_case(query_entry, case_kind)
    primary_rows = list(query_entry.get("methods", {}).get("primary_only", {}).get("top_results", []))
    matrix_rows = list(query_entry.get("methods", {}).get("primary_plus_matrix", {}).get("top_results", []))
    target_chunk_ids = sorted(
        {
            str(row.get("chunk_id", ""))
            for row in primary_rows[:5] + matrix_rows[:5]
            if str(row.get("chunk_id", ""))
        }
    )
    return {
        "case_type": case_kind,
        "query_id": str(query_entry.get("query_id", "")),
        "query": str(query_entry.get("query", "")),
        "query_profile": query_entry.get("query_profile", {}),
        "diagnosis": diagnosis,
        "primary_top5": summarize_result_rows(primary_rows),
        "matrix_top5": summarize_result_rows(matrix_rows),
        "target_chunk_ids": target_chunk_ids,
    }


def main() -> None:
    args = parse_args()
    if not args.report_path.exists():
        raise FileNotFoundError(f"Report file not found: {args.report_path}")

    report = load_json(args.report_path)
    queries = {str(item.get("query_id", "")): item for item in report.get("queries", [])}

    audit_cases: list[dict[str, Any]] = []
    for case_kind in (
        "role_mismatch_cases",
        "same_topic_wrong_answer_cases",
        "strong_primary_but_penalized_cases",
    ):
        for raw_case in report.get(case_kind, []):
            query_id = str(raw_case.get("query_id", ""))
            query_entry = queries.get(query_id)
            if not query_entry:
                continue
            audit_cases.append(build_case_entry(query_entry, case_kind))

    unique_target_chunk_ids = sorted(
        {
            chunk_id
            for case in audit_cases
            for chunk_id in case.get("target_chunk_ids", [])
            if chunk_id
        }
    )

    summary = {
        "total_audit_cases": len(audit_cases),
        "role_mismatch_cases": len(report.get("role_mismatch_cases", [])),
        "same_topic_wrong_answer_cases": len(report.get("same_topic_wrong_answer_cases", [])),
        "strong_primary_but_penalized_cases": len(report.get("strong_primary_but_penalized_cases", [])),
        "unique_target_chunk_count": len(unique_target_chunk_ids),
        "unique_target_chunk_ids": unique_target_chunk_ids,
    }

    output = {
        "source_report": str(args.report_path),
        "summary": summary,
        "audit_cases": audit_cases,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Source report: {args.report_path}")
    print(f"Audit cases: {len(audit_cases)}")
    print(f"Unique target chunks: {len(unique_target_chunk_ids)}")
    print(f"Saved audit to: {args.output_path}")


if __name__ == "__main__":
    main()
