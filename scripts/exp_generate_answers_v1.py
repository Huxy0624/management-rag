#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
DEFAULT_RETRIEVAL_REPORT_PATH = Path("data/pipeline_candidates/v4_phase22/retrieval_eval_v4/report.json")
DEFAULT_OUTPUT_DIR = Path("data/pipeline_candidates/generation_v1")

QUERY_TYPE_POLICY = {
    "what": {
        "main_roles": ["definition", "principle"],
        "support_roles": ["mechanism", "summary"],
        "structure": "先给定义或本质，再补边界/解释，必要时补一个很短的例子。",
    },
    "why": {
        "main_roles": ["mechanism", "principle"],
        "support_roles": ["definition", "warning", "summary"],
        "structure": "先给核心结论，再给2到3步因果链，最后补一句管理含义或风险提醒。",
    },
    "how": {
        "main_roles": ["solution", "mechanism"],
        "support_roles": ["principle", "example"],
        "structure": "先给可执行做法，再说明适用条件，最后补一句常见误区或风险。",
    },
}

ANSWER_PROMPT_V1 = """你是一个管理知识问答助手。你会收到：
- 用户问题
- query_type（what / why / how）
- 主证据（main evidence）
- 辅证据（support evidence）
- 证据冲突提示（如果有）

你的任务：
1. 必须先正面回答问题，第一句就回应用户真正关心的点。
2. 必须按 query_type 对应结构作答：
   - what：先定义/本质，再补边界或解释，必要时补一个很短的例子。
   - why：先给核心结论，再给2到3步因果链，最后补一句管理含义或风险。
   - how：先给可执行做法，再说明适用条件，最后补一句常见误区或风险。
3. 优先使用主证据；如果主证据与辅证据有轻微冲突，优先采用 primary_score 更高、answer_role 更匹配的证据。
4. 不要只复述检索原文，要做整合表达。
5. 不要脱离证据自由发挥，不要编造未被证据支持的机制、步骤或结论。
6. 默认输出 3 段以内的紧凑中文，不需要机械罗列太多小点。
"""

ANSWER_EVAL_TEMPLATE = {
    "dimensions": [
        {
            "name": "question_alignment",
            "scale": "1-5",
            "question": "是否先正面回答了用户真正的问题，而不是绕到邻近主题。",
            "anchors": {"1": "明显答偏", "3": "部分回答到点", "5": "首句即正面回应且整体不跑偏"},
        },
        {
            "name": "answer_role_alignment",
            "scale": "1-5",
            "question": "回答结构是否符合 query type。",
            "anchors": {"1": "结构明显错位", "3": "大体符合但起手不稳", "5": "what先定义/why先解释原因/how先给做法"},
        },
        {
            "name": "evidence_usage",
            "scale": "1-5",
            "question": "是否优先使用主证据，没有被辅证据带偏。",
            "anchors": {"1": "明显错用证据", "3": "主辅证据混用但可接受", "5": "主证据使用准确且整合自然"},
        },
        {
            "name": "clarity",
            "scale": "1-5",
            "question": "回答是否清楚、紧凑、不绕。",
            "anchors": {"1": "混乱或冗长", "3": "能读懂但略绕", "5": "清楚、紧凑、层次自然"},
        },
        {
            "name": "overgeneralization",
            "scale": "1-5",
            "question": "是否脱离证据乱发挥。",
            "anchors": {"1": "大量超出证据", "3": "有少量泛化", "5": "基本贴证据，不乱扩展"},
        },
    ],
    "per_answer_fields": ["query_id", "query", "score_by_dimension", "overall_comment", "needs_policy_v2"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline generation prototype over phase22 retrieval report.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH, help="Query set JSON path.")
    parser.add_argument("--retrieval-report-path", type=Path, default=DEFAULT_RETRIEVAL_REPORT_PATH, help="Retrieval report JSON path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, help="Optional number of queries to generate.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_snippet(text: str) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"^[0-9]+[\.．\)]\s*", "", cleaned)
    cleaned = re.sub(r"^[一二三四五六七八九十]+[、\.．]\s*", "", cleaned)
    cleaned = cleaned.strip(" 。；：:")
    return cleaned


def query_type_from_row(query_row: dict[str, Any]) -> str:
    return str(query_row.get("category", query_row.get("query_profile", {}).get("query_type", "what")))


def subject_from_query(query: str, query_type: str) -> str:
    subject = query.strip().rstrip("？?。")
    if query_type == "what":
        subject = re.sub(r"^(什么是|什么叫|何谓)", "", subject)
    elif query_type == "why":
        subject = subject.replace("为什么", "", 1).replace("为何", "", 1)
    elif query_type == "how":
        subject = subject.replace("怎么", "", 1).replace("如何", "", 1)
    return subject.strip(" ，,：:")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", normalize_text(text))
    return [part.strip(" 。") for part in parts if part.strip(" 。")]


def evidence_snippets(result: dict[str, Any], limit: int = 3) -> list[str]:
    snippets: list[str] = []
    for insight in result.get("matched_insights", []):
        snippet = clean_snippet(str(insight.get("insight_text", "")))
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            return snippets
    for sentence in split_sentences(str(result.get("chunk_text_preview", ""))):
        cleaned = clean_snippet(sentence)
        if cleaned and cleaned not in snippets:
            snippets.append(cleaned)
        if len(snippets) >= limit:
            break
    return snippets


def title_match_bonus(query: str, title: str) -> int:
    normalized_query = normalize_text(query)
    normalized_title = normalize_text(title)
    score = 0
    subject = subject_from_query(query, "what")
    if subject and subject in normalized_title:
        score += 2
    for keyword in ("跨部门", "协作", "评价失效", "信息", "汇报", "战略", "机制", "人治", "法治"):
        if keyword in normalized_query and keyword in normalized_title:
            score += 1
    return score


def evidence_priority(query: str, query_type: str, result: dict[str, Any]) -> tuple[int, int, float, float, int]:
    policy = QUERY_TYPE_POLICY[query_type]
    answer_role = str(result.get("governance_tags_v4", {}).get("answer_role", "none"))
    if answer_role in policy["main_roles"]:
        tier = 0
    elif answer_role in policy["support_roles"]:
        tier = 1
    else:
        tier = 2
    return (
        tier,
        -title_match_bonus(query, str(result.get("title", ""))),
        -float(result.get("final_score", 0.0)),
        -float(result.get("primary_score", 0.0)),
        -int(result.get("relevance", 0)),
    )


def select_evidence(query_row: dict[str, Any], max_main: int = 2, max_support: int = 2) -> dict[str, Any]:
    query_type = query_type_from_row(query_row)
    top_results = list(query_row.get("methods", {}).get("primary_plus_matrix", {}).get("top_results", []))
    ranked = sorted(top_results, key=lambda result: evidence_priority(str(query_row.get("query", "")), query_type, result))
    policy = QUERY_TYPE_POLICY[query_type]
    main: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    used_chunk_ids: set[str] = set()

    for result in ranked:
        chunk_id = str(result.get("chunk_id", ""))
        if not chunk_id or chunk_id in used_chunk_ids:
            continue
        answer_role = str(result.get("governance_tags_v4", {}).get("answer_role", "none"))
        payload = {
            "chunk_id": chunk_id,
            "title": str(result.get("title", "")),
            "primary_rank": next(
                (
                    index + 1
                    for index, row in enumerate(query_row.get("methods", {}).get("primary_only", {}).get("top_results", []))
                    if str(row.get("chunk_id", "")) == chunk_id
                ),
                None,
            ),
            "final_rank": next(
                (
                    index + 1
                    for index, row in enumerate(top_results)
                    if str(row.get("chunk_id", "")) == chunk_id
                ),
                None,
            ),
            "primary_score": round(float(result.get("primary_score", 0.0)), 4),
            "final_score": round(float(result.get("final_score", 0.0)), 4),
            "relevance": int(result.get("relevance", 0)),
            "answer_role": answer_role,
            "intent": str(result.get("governance_tags_v4", {}).get("intent", "none")),
            "root_issue": str(result.get("governance_tags_v4", {}).get("root_issue", "none")),
            "snippets": evidence_snippets(result),
            "chunk_text_preview": str(result.get("chunk_text_preview", "")),
        }
        if answer_role in policy["main_roles"] and len(main) < max_main:
            main.append(payload)
            used_chunk_ids.add(chunk_id)
            continue
        if answer_role in policy["support_roles"] and len(support) < max_support:
            support.append(payload)
            used_chunk_ids.add(chunk_id)

    if not main and support:
        promoted = support.pop(0)
        main.append(promoted)
        used_chunk_ids.add(promoted["chunk_id"])
        role_conflict = ["no ideal main evidence role found; promoted best support evidence to main"]
    else:
        role_conflict = []

    if len(support) < max_support:
        for result in ranked:
            chunk_id = str(result.get("chunk_id", ""))
            if chunk_id in used_chunk_ids:
                continue
            payload = {
                "chunk_id": chunk_id,
                "title": str(result.get("title", "")),
                "primary_rank": next(
                    (
                        index + 1
                        for index, row in enumerate(query_row.get("methods", {}).get("primary_only", {}).get("top_results", []))
                        if str(row.get("chunk_id", "")) == chunk_id
                    ),
                    None,
                ),
                "final_rank": next(
                    (
                        index + 1
                        for index, row in enumerate(top_results)
                        if str(row.get("chunk_id", "")) == chunk_id
                    ),
                    None,
                ),
                "primary_score": round(float(result.get("primary_score", 0.0)), 4),
                "final_score": round(float(result.get("final_score", 0.0)), 4),
                "relevance": int(result.get("relevance", 0)),
                "answer_role": str(result.get("governance_tags_v4", {}).get("answer_role", "none")),
                "intent": str(result.get("governance_tags_v4", {}).get("intent", "none")),
                "root_issue": str(result.get("governance_tags_v4", {}).get("root_issue", "none")),
                "snippets": evidence_snippets(result),
                "chunk_text_preview": str(result.get("chunk_text_preview", "")),
            }
            support.append(payload)
            used_chunk_ids.add(chunk_id)
            if len(support) >= max_support:
                break

    main_roles = {item["answer_role"] for item in main}
    support_roles = {item["answer_role"] for item in support}
    if any(role in {"solution", "mechanism"} for role in support_roles) and query_type == "what":
        role_conflict.append("support evidence contains mechanism/solution for a what query")
    if "solution" in main_roles and query_type == "why":
        role_conflict.append("main evidence includes solution-like role for a why query")
    if "definition" in main_roles and query_type == "how":
        role_conflict.append("main evidence includes definition-like role for a how query")

    return {
        "query_type": query_type,
        "policy": policy,
        "main_evidence": main,
        "support_evidence": support,
        "role_conflict_notes": role_conflict,
    }


def render_prompt(query: str, query_type: str, selected: dict[str, Any]) -> str:
    def render_bucket(title: str, rows: list[dict[str, Any]]) -> str:
        rendered: list[str] = [title]
        for index, row in enumerate(rows, start=1):
            rendered.append(
                "\n".join(
                    [
                        f"{index}. chunk_id={row['chunk_id']}",
                        f"   title={row['title']}",
                        f"   primary_rank={row['primary_rank']} final_rank={row['final_rank']}",
                        f"   primary_score={row['primary_score']} final_score={row['final_score']}",
                        f"   answer_role={row['answer_role']} intent={row['intent']} root_issue={row['root_issue']}",
                        f"   snippets={json.dumps(row['snippets'], ensure_ascii=False)}",
                    ]
                )
            )
        return "\n".join(rendered)

    return "\n\n".join(
        [
            ANSWER_PROMPT_V1.strip(),
            f"query_type: {query_type}",
            f"question: {query}",
            f"policy_note: {selected['policy']['structure']}",
            f"role_conflict_notes: {json.dumps(selected['role_conflict_notes'], ensure_ascii=False)}",
            render_bucket("main_evidence:", selected["main_evidence"]),
            render_bucket("support_evidence:", selected["support_evidence"]),
        ]
    )


def first_snippet(rows: list[dict[str, Any]], fallback: str = "") -> str:
    for row in rows:
        for snippet in row.get("snippets", []):
            if snippet:
                return snippet
    return fallback


def next_distinct_snippets(rows: list[dict[str, Any]], used: set[str], limit: int) -> list[str]:
    snippets: list[str] = []
    for row in rows:
        for snippet in row.get("snippets", []):
            cleaned = snippet.strip(" 。")
            if not cleaned or cleaned in used:
                continue
            used.add(cleaned)
            snippets.append(cleaned)
            if len(snippets) >= limit:
                return snippets
    return snippets


def generate_answer(query: str, query_type: str, selected: dict[str, Any]) -> str:
    subject = subject_from_query(query, query_type)
    main = selected["main_evidence"]
    support = selected["support_evidence"]
    main_roles = {item["answer_role"] for item in main}
    used: set[str] = set()
    lead = first_snippet(main, first_snippet(support, subject))
    if lead:
        used.add(clean_snippet(lead))
    followups = next_distinct_snippets(main + support, used, limit=3)

    if query_type == "what":
        if main_roles & {"definition", "principle"}:
            first = f"{subject}，本质上是{clean_snippet(lead)}。"
        else:
            first = f"从当前证据看，{subject}更接近这样一类问题：{clean_snippet(lead)}。"
        second = ""
        if followups:
            second = f"再往下看，关键边界在于{clean_snippet(followups[0])}。"
        third = ""
        if len(followups) > 1:
            third = f"放到管理场景里，常见表现会落到{clean_snippet(followups[1])}。"
        return " ".join(part for part in (first, second, third) if part)

    if query_type == "why":
        chain = [clean_snippet(lead)] + [clean_snippet(item) for item in followups[:2]]
        first = f"核心原因是，{chain[0]}。"
        second_parts = []
        if len(chain) > 1:
            second_parts.append(f"第一步是{chain[1]}")
        if len(chain) > 2:
            second_parts.append(f"第二步是{chain[2]}")
        second = "；".join(second_parts) + "。" if second_parts else ""
        third = "这意味着如果只盯表面冲突而不回到机制或评价结构，问题会反复出现。"
        return " ".join(part for part in (first, second, third) if part)

    steps = [clean_snippet(lead)] + [clean_snippet(item) for item in followups[:2]]
    first = f"更可执行的做法是，先围绕“{steps[0]}”推进。"
    second_parts = []
    if len(steps) > 1:
        second_parts.append(f"再把“{steps[1]}”补上")
    if len(steps) > 2:
        second_parts.append(f"同时注意“{steps[2]}”")
    second = "；".join(second_parts) + "。" if second_parts else ""
    third = "这类做法更适合先稳住局面再推动机制修正，误区是还没形成约束条件就先讲大而空的定义。"
    return " ".join(part for part in (first, second, third) if part)


def summarize_batch(rows: list[dict[str, Any]]) -> str:
    by_type: dict[str, list[dict[str, Any]]] = {"what": [], "why": [], "how": []}
    for row in rows:
        by_type[row["query_type"]].append(row)
    lines = [
        "# Generation V1 Summary",
        "",
        f"- total_queries: {len(rows)}",
        f"- baseline_retrieval_report: `{DEFAULT_RETRIEVAL_REPORT_PATH.as_posix()}`",
        "- strongest_query_type: `how`，因为主证据里的 `solution/mechanism` 更容易转成可执行回答。",
        "- easiest_to_drift: `what`，因为检索结果里经常混入 `mechanism/solution`，若不强制定义起手很容易答偏。",
        "- main_generation_risk: `why` 在 evidence 带有 `solution` 或 meta-like 概括时，容易被解释链和做法混写。",
        "- recommendation: 需要进入 `answer policy v2`，优先补充 evidence conflict handling 和更细的 sentence planner。",
        "",
        "## By Query Type",
    ]
    for query_type in ("what", "why", "how"):
        lines.append(f"- `{query_type}` queries: {len(by_type[query_type])}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    queries = load_json(args.query_set_path)
    report = load_json(args.retrieval_report_path)
    query_rows = list(report.get("queries", []))
    if args.limit is not None:
        query_rows = query_rows[: args.limit]
    query_map = {str(row.get("query_id", "")): row for row in query_rows}

    generated_rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query.get("id", ""))
        query_row = query_map.get(query_id)
        if not query_row:
            continue
        query_type = query_type_from_row(query_row)
        selected = select_evidence(query_row)
        prompt = render_prompt(str(query.get("query", "")), query_type, selected)
        answer = generate_answer(str(query.get("query", "")), query_type, selected)
        generated_rows.append(
            {
                "query_id": query_id,
                "query": str(query.get("query", "")),
                "query_type": query_type,
                "answer_policy": QUERY_TYPE_POLICY[query_type],
                "selected_evidence": {
                    "main_evidence": selected["main_evidence"],
                    "support_evidence": selected["support_evidence"],
                    "role_conflict_notes": selected["role_conflict_notes"],
                },
                "answer_prompt_v1": prompt,
                "final_answer": answer,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "answer_policy_v1.md").write_text(
        "\n".join(
            [
                "# Answer Policy V1",
                "",
                "- `what`: 先给定义/本质，再补边界或解释，必要时补一个很短的例子。",
                "- `why`: 先给核心结论，再给 2-3 步因果链，最后补一句管理含义或风险提醒。",
                "- `how`: 先给可执行做法，再说明适用条件，最后补一句常见误区或风险。",
                "",
                "## Evidence Assembly",
                "",
                "- `what` 主证据优先 `definition/principle`，辅证据优先 `mechanism/summary`。",
                "- `why` 主证据优先 `mechanism/principle`，辅证据优先 `definition/warning/summary`。",
                "- `how` 主证据优先 `solution/mechanism`，辅证据优先 `principle/example`。",
                "- 主辅证据分开组装，避免 5 条同质内容直接平铺。",
                "- 若 answer_role 明显冲突，则在 prompt 中显式记录冲突，提示优先主证据。",
            ]
        ),
        encoding="utf-8",
    )
    (args.output_dir / "answer_prompt_v1.txt").write_text(ANSWER_PROMPT_V1, encoding="utf-8")
    (args.output_dir / "answer_eval_template.json").write_text(
        json.dumps(ANSWER_EVAL_TEMPLATE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "answers_v1.json").write_text(
        json.dumps(
            {
                "retrieval_report_path": str(args.retrieval_report_path),
                "query_count": len(generated_rows),
                "answers": generated_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "generation_summary.md").write_text(summarize_batch(generated_rows), encoding="utf-8")
    print(f"Saved generation prototype outputs to: {args.output_dir}")
    print(f"Generated answers: {len(generated_rows)}")


if __name__ == "__main__":
    main()
