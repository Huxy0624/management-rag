from __future__ import annotations

from typing import Any

from runtime.experiment_bridge import get_v1, get_v2, get_v21
from runtime.query_router import infer_query_type, route_query
from runtime.runtime_config import PlannerRuntimeConfig


DEFINITION_PATTERNS = ("本质是", "核心是", "定义是", "可以定义为", "就是")
MECHANISM_PATTERNS = ("机制", "规则", "流程", "结构", "治理", "复盘", "周会", "考核", "边界表", "复盘单")
SOLUTION_PATTERNS = ("编制", "指派", "对齐", "广播", "拆解", "追踪", "记录", "升级", "复盘", "设定", "同步", "明确")


def _preview_text(text: str, limit: int = 400) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _guess_answer_role(document: str, query_type: str) -> str:
    if any(token in document for token in DEFINITION_PATTERNS):
        return "definition" if query_type == "what" else "principle"
    if any(token in document for token in MECHANISM_PATTERNS):
        return "mechanism"
    if any(token in document for token in SOLUTION_PATTERNS):
        return "solution"
    if query_type == "why":
        return "mechanism"
    if query_type == "how":
        return "solution"
    return "principle"


def _row_from_chunk(chunk: dict[str, Any], final_rank: int, query_type: str) -> dict[str, Any]:
    v1 = get_v1()
    metadata = dict(chunk.get("metadata", {}))
    breakdown = dict(chunk.get("rerank_breakdown", {}))
    document = str(chunk.get("document", ""))
    return {
        "chunk_id": str(metadata.get("chunk_id", final_rank - 1)),
        "title": str(metadata.get("title", metadata.get("source", ""))),
        "primary_rank": None,
        "final_rank": final_rank,
        "primary_score": round(float(breakdown.get("vector_score", 0.0)), 4),
        "final_score": round(float(chunk.get("rerank_score", 0.0)), 4),
        "relevance": 0,
        "answer_role": str(metadata.get("answer_role", _guess_answer_role(document, query_type))),
        "intent": str(metadata.get("intent", "none")),
        "root_issue": str(metadata.get("root_issue", "none")),
        "snippets": [v1.clean_snippet(item) for item in v1.split_sentences(document)[:3] if v1.clean_snippet(item)],
        "chunk_text_preview": _preview_text(document),
    }


def _normalize_runtime_rows(chunks: list[dict[str, Any]], query_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        row = _row_from_chunk(chunk, index, query_type)
        if row["answer_role"] in {"definition", "principle", "mechanism", "solution", "example", "summary"}:
            rows.append(row)
        else:
            row["answer_role"] = _guess_answer_role(str(chunk.get("document", "")), query_type)
            rows.append(row)
    return rows


def build_selected_evidence(question: str, chunks: list[dict[str, Any]], query_type: str, config: PlannerRuntimeConfig) -> dict[str, Any]:
    v1 = get_v1()
    rows = _normalize_runtime_rows(chunks, query_type)
    policy = v1.QUERY_TYPE_POLICY[query_type]
    main: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []

    for row in rows:
        if row["answer_role"] in policy["main_roles"] and len(main) < config.main_evidence_count:
            main.append(row)
            continue
        if row["answer_role"] in policy["support_roles"] and len(support) < config.support_evidence_count:
            support.append(row)

    if not main and rows:
        main.append(rows[0])

    if len(support) < config.support_evidence_count:
        used_chunk_ids = {str(item["chunk_id"]) for item in main + support}
        for row in rows:
            if str(row["chunk_id"]) in used_chunk_ids:
                continue
            support.append(row)
            used_chunk_ids.add(str(row["chunk_id"]))
            if len(support) >= config.support_evidence_count:
                break

    return {
        "query_type": query_type,
        "policy": policy,
        "main_evidence": main,
        "support_evidence": support,
        "role_conflict_notes": [],
    }


def build_planner_context(question: str, chunks: list[dict[str, Any]], config: PlannerRuntimeConfig) -> dict[str, Any]:
    query_type = infer_query_type(question)
    router = route_query(question)
    selected = build_selected_evidence(question, chunks, query_type, config)
    v2 = get_v2()
    v21 = get_v21()

    if query_type == "how":
        planner_output, action_output, mechanism_output = v21.plan_how_v21(question, router, selected)
    elif query_type == "what":
        planner_output = v2.plan_what(question, router, selected)
        action_output = {"translator_name": "not_applicable", "action_steps": []}
        mechanism_output = {"mapper_name": "not_applicable", "mechanism_entities": []}
    else:
        planner_output = v2.plan_why(question, router, selected)
        action_output = {"translator_name": "not_applicable", "action_steps": []}
        mechanism_output = {"mapper_name": "not_applicable", "mechanism_entities": []}

    fallback_answer = v21.surface_generate_v21(question, query_type, router, planner_output)
    return {
        "query": question,
        "query_type": query_type,
        "router_decision": router,
        "selected_evidence": selected,
        "planner_output_v21": planner_output,
        "action_translator_output": action_output,
        "mechanism_mapper_output": mechanism_output,
        "v21_final_answer": fallback_answer,
    }
