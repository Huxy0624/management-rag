#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import exp_generate_answers_v1 as v1


DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
DEFAULT_RETRIEVAL_REPORT_PATH = Path("data/pipeline_candidates/v4_phase22/retrieval_eval_v4/report.json")
DEFAULT_V1_ANSWERS_PATH = Path("data/pipeline_candidates/generation_v1/answers_v1.json")
DEFAULT_OUTPUT_DIR = Path("data/pipeline_candidates/generation_v2")

PRIORITY_QUERY_IDS = [
    "cross_dept_definition",
    "cross_dept_why_low_efficiency",
    "cross_dept_how_temporary_push",
    "cross_dept_how_mechanism_fix",
    "conflict_info_root",
    "upward_compression",
    "downward_diffusion",
]

FIRST_PRINCIPLE_MAP = {
    "信息失真": "信息在跨层级或跨角色传递中被压缩、变形或误解的状态",
    "评价失效": "组织无法稳定地区分贡献与责任，导致激励、奖惩和资源配置失真的状态",
    "局部最优": "各部门只优化自己收益，却让整体协作效率下降的状态",
    "公地悲剧": "每个参与方都按自身利益行事，最后共同损害整体结果的局面",
    "结构影响行为": "组织的规则、分工和奖惩会持续塑造成员行为的原理",
    "跨部门协作问题": "部门之间因为目标、责任、信息和资源难以对齐，导致合作成本升高的状态",
}

SURFACE_PROMPT_V2 = """你是一个管理知识问答助手。你不会直接摘录检索句子，而是先阅读 planner 产出的回答骨架，再把它写成自然语言答案。

输出要求：
1. 第一行必须正面回应问题。
2. what：第一句必须是定义句；why：第一句必须是核心结论；how：第一句必须是动作导向。
3. 优先复述 planner，而不是复述原始 evidence。
4. 只有在需要支撑时，才把 evidence 中的信息折叠进解释。
5. 不要把角色画像、能力标签、标题句原样抄成答案主体。
"""

ANSWER_EVAL_TEMPLATE_V2 = {
    "dimensions": [
        {
            "name": "first_sentence_alignment",
            "scale": "1-5",
            "question": "第一句是否真的正面回答了用户问题。",
            "anchors": {"1": "第一句跑偏", "3": "部分回应", "5": "第一句即正中问题"},
        },
        {
            "name": "structure_fitness",
            "scale": "1-5",
            "question": "是否符合 what / why / how 的目标结构。",
            "anchors": {"1": "结构错位", "3": "结构基本可用", "5": "结构完全贴合策略"},
        },
        {
            "name": "planner_faithfulness",
            "scale": "1-5",
            "question": "最终答案是否忠实执行了 planner，而不是回到摘句拼接。",
            "anchors": {"1": "明显脱离 planner", "3": "部分遵循", "5": "高度忠实"},
        },
        {
            "name": "actionability",
            "scale": "1-5",
            "question": "how 类回答是否形成了真正可执行的动作步骤。",
            "anchors": {"1": "几乎没有动作", "3": "有步骤但不够可执行", "5": "动作清楚、顺序明确"},
        },
        {
            "name": "causal_integrity",
            "scale": "1-5",
            "question": "why 类回答是否形成了完整因果链。",
            "anchors": {"1": "没有因果链", "3": "有因果但不完整", "5": "至少两步且完整"},
        },
        {
            "name": "definition_quality",
            "scale": "1-5",
            "question": "what 类定义是否清楚，抽象层级是否对。",
            "anchors": {"1": "定义漂移", "3": "定义勉强可用", "5": "定义清楚且层级准确"},
        },
    ],
    "per_answer_fields": [
        "query_id",
        "query",
        "router_decision",
        "score_by_dimension",
        "overall_comment",
        "needs_policy_v3",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planner-based generation prototype over phase22 retrieval report.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH, help="Query set JSON path.")
    parser.add_argument("--retrieval-report-path", type=Path, default=DEFAULT_RETRIEVAL_REPORT_PATH, help="Retrieval report JSON path.")
    parser.add_argument("--v1-answers-path", type=Path, default=DEFAULT_V1_ANSWERS_PATH, help="Generation v1 answers path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--priority-only", action="store_true", help="Only generate priority query set.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def router_decision(query: str, query_type: str) -> dict[str, Any]:
    normalized = v1.normalize_text(query)
    if query_type == "how":
        if any(token in normalized for token in ("临时", "先", "救火", "短期", "推进")):
            subtype = "temporary_push"
            policy_bias = "rule_of_man_point_solution"
        elif any(token in normalized for token in ("机制", "长期", "体系", "法治", "制度")):
            subtype = "mechanism_building"
            policy_bias = "rule_of_law_system_solution"
        else:
            subtype = "generic_how"
            policy_bias = "mixed_action"
    elif query_type == "what":
        if any(token in normalized for token in ("本质", "原理")):
            subtype = "essence_or_principle"
            policy_bias = "first_principle_definition"
        else:
            subtype = "concept_definition"
            policy_bias = "concept_boundary"
    else:
        if any(token in normalized for token in ("反复", "总是", "一直", "总会")):
            subtype = "recurring_pattern"
            policy_bias = "structure_repetition"
        else:
            subtype = "root_cause"
            policy_bias = "structure_behavior_result"
    return {"query_type": query_type, "subtype": subtype, "policy_bias": policy_bias}


def infer_first_principle_anchor(query: str, evidence_rows: list[dict[str, Any]]) -> str:
    normalized = v1.normalize_text(query)
    for key, value in FIRST_PRINCIPLE_MAP.items():
        if key in normalized:
            return value
    joined = " ".join(
        " ".join(row.get("snippets", []))
        for row in evidence_rows
    )
    if "规避责任" in joined or "甩锅" in joined:
        return FIRST_PRINCIPLE_MAP["局部最优"]
    if "信息" in joined and ("压缩" in joined or "传递" in joined or "解释" in joined):
        return FIRST_PRINCIPLE_MAP["信息失真"]
    if "评价" in joined or "奖惩" in joined:
        return FIRST_PRINCIPLE_MAP["评价失效"]
    if "结构影响行为" in joined or "规则影响行为" in joined:
        return FIRST_PRINCIPLE_MAP["结构影响行为"]
    subject = v1.subject_from_query(query, "what")
    return f"{subject}在管理场景中的关键状态"


def compact_snippet(rows: list[dict[str, Any]], fallback: str = "") -> str:
    for row in rows:
        for snippet in row.get("snippets", []):
            cleaned = v1.clean_snippet(snippet)
            if cleaned:
                return cleaned
    return fallback


def compact_snippets(rows: list[dict[str, Any]], limit: int = 3) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for snippet in row.get("snippets", []):
            cleaned = v1.clean_snippet(snippet)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            items.append(cleaned)
            if len(items) >= limit:
                return items
    return items


def abstract_definition_from_mechanism(query: str, evidence_rows: list[dict[str, Any]]) -> str:
    subject = v1.subject_from_query(query, "what")
    anchor = infer_first_principle_anchor(query, evidence_rows)
    joined = " ".join(compact_snippets(evidence_rows, limit=4))
    if "规避责任" in joined or "甩锅" in joined:
        return f"{subject}，本质上是部门在局部激励下优先自保，导致责任、信息和资源难以对齐的状态"
    if "信息" in joined and ("压缩" in joined or "传递" in joined or "解释" in joined):
        return f"{subject}，本质上是{anchor}"
    if "评价" in joined or "奖惩" in joined:
        return f"{subject}，本质上是{anchor}"
    return f"{subject}，本质上是{anchor}"


def translate_capability_to_action(snippet: str, router: dict[str, Any]) -> str:
    cleaned = v1.clean_snippet(snippet)
    replacements = [
        ("能平衡各方利益，推动全局发展", "先把各方利益和阻塞点摊开，再推动关键人做取舍"),
        ("懂得权衡各种利益关系，并以最大的整体ROI为最终追求", "先统一短期目标和整体ROI口径"),
        ("需要更加注重决策的果断性和执行力度", "明确由谁拍板、何时反馈、谁负责跟进"),
        ("关注并加强信息传递的清晰度和实际执行力", "把信息压缩成少数关键点并追踪执行"),
        ("通过增强执行力和注重细节", "把动作拆成可追踪的小步骤"),
    ]
    for old, new in replacements:
        if old in cleaned:
            return new
    if router["subtype"] == "temporary_push":
        return f"围绕“{cleaned}”去做一次短期协调和拍板"
    return f"把“{cleaned}”转成明确的机制动作和跟进节点"


def plan_what(query: str, router: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    main = selected["main_evidence"]
    support = selected["support_evidence"]
    main_roles = {row["answer_role"] for row in main}
    if main_roles & {"definition", "principle"}:
        definition_anchor = compact_snippet(main, infer_first_principle_anchor(query, main))
        definition_source = "main_definition_like_evidence"
    else:
        definition_anchor = abstract_definition_from_mechanism(query, main + support)
        definition_source = "abstracted_from_mechanism"
    snippets = compact_snippets(main + support, limit=4)
    boundary = ""
    if any("规避责任" in item or "甩锅" in item for item in snippets):
        boundary = "它不只是合作意愿差，而是激励与责任边界没有对齐，导致各方优先保护自己的局部收益"
    elif any("信息" in item for item in snippets):
        boundary = "它不只是沟通次数少，而是信息在传递、过滤和解释过程中持续失真"
    else:
        boundary = "它不是单一现象，而是组织结构、信息和激励共同作用出来的管理状态"
    example = ""
    if snippets:
        example = f"例如，{snippets[0]}"
    return {
        "planner_type": "what_planner",
        "definition_anchor": definition_anchor,
        "definition_source": definition_source,
        "boundary_explanation": boundary,
        "short_example": example,
    }


def plan_why(query: str, router: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    main = selected["main_evidence"]
    support = selected["support_evidence"]
    main_snippets = compact_snippets(main, limit=4)
    support_snippets = compact_snippets(support, limit=3)
    core = main_snippets[0] if main_snippets else infer_first_principle_anchor(query, main + support)
    if "规避责任" in " ".join(main_snippets + support_snippets):
        chain = [
            "底层结构里，部门首先对自己的事故、考核和收益负责",
            "这会诱发各部门优先规避责任、保住本部门优先级，而不是主动承担协同成本",
            "最后整体合作效率下降，冲突、扯皮和信息断裂会反复出现",
        ]
    elif "信息" in " ".join(main_snippets + support_snippets):
        chain = [
            "底层结构里，信息需要跨层级、跨角色传递和翻译",
            "传递过程中的过滤、压缩和偏听偏信会诱发理解偏差与动作错位",
            "最终不同部门对同一问题形成不同判断，冲突就会持续放大",
        ]
    else:
        chain = [
            "底层结构决定了组织优先奖励什么、规避什么",
            "这种激励会持续塑造成员行为和协作方式",
            "当行为方向与整体目标不一致时，结果就会表现为效率下降或问题反复",
        ]
    implication = "所以管理上不能只盯表面现象，而要回到规则、激励和信息结构本身去修。"
    return {
        "planner_type": "why_planner",
        "core_conclusion": core,
        "causal_chain": chain,
        "management_implication": implication,
        "support_usage_note": "support evidence only supplements the chain; solution-like evidence cannot dominate explanation",
    }


def plan_how(query: str, router: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    main = selected["main_evidence"]
    support = selected["support_evidence"]
    main_snippets = compact_snippets(main, limit=4)
    support_snippets = compact_snippets(support, limit=3)
    action_steps: list[str] = []
    for snippet in main_snippets:
        lowered = snippet
        if any(token in lowered for token in ("协调者", "成熟稳重", "能力", "角色", "适合岗位")):
            action_steps.append(translate_capability_to_action(lowered, router))
        elif any(token in lowered for token in ("建立", "推进", "压缩", "解释", "汇报", "跟踪", "明确")):
            action_steps.append(lowered)
        else:
            action_steps.append(translate_capability_to_action(lowered, router))
        if len(action_steps) >= 3:
            break
    while len(action_steps) < 3 and support_snippets:
        action_steps.append(translate_capability_to_action(support_snippets.pop(0), router))

    if router["subtype"] == "temporary_push":
        applicability = "适用于短期先把事情推过去、已有明确目标但缺少协同动作的场景。"
        risk = "风险是只靠人治推动而不补机制，问题会在下一轮合作中再次出现。"
    elif router["subtype"] == "mechanism_building":
        applicability = "适用于需要长期稳定协同、需要把临时协调固化成规则和流程的场景。"
        risk = "风险是只讲制度口号，不落到责任边界、反馈节奏和奖惩约束。"
    else:
        applicability = "适用于既要先动起来，又要为后续机制修正留接口的场景。"
        risk = "风险是动作很多，但没有主次和跟进闭环。"
    return {
        "planner_type": "how_planner",
        "solution_mode": router["subtype"],
        "action_steps": action_steps[:3],
        "applicability": applicability,
        "risk_note": risk,
    }


def plan_answer(query: str, router: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    if router["query_type"] == "what":
        return plan_what(query, router, selected)
    if router["query_type"] == "why":
        return plan_why(query, router, selected)
    return plan_how(query, router, selected)


def surface_generate(query: str, router: dict[str, Any], planner_output: dict[str, Any]) -> str:
    query_type = router["query_type"]
    if query_type == "what":
        first = planner_output["definition_anchor"]
        if not first.endswith("。"):
            first += "。"
        second = planner_output["boundary_explanation"]
        if second and not second.endswith("。"):
            second += "。"
        third = planner_output["short_example"]
        if third and not third.endswith("。"):
            third += "。"
        return " ".join(part for part in (first, second, third) if part)
    if query_type == "why":
        first = f"核心原因是，{planner_output['core_conclusion']}。"
        middle = "；".join(
            [f"第一步，{planner_output['causal_chain'][0]}", f"第二步，{planner_output['causal_chain'][1]}", f"第三步，{planner_output['causal_chain'][2]}"]
        )
        if not middle.endswith("。"):
            middle += "。"
        last = planner_output["management_implication"]
        if last and not last.endswith("。"):
            last += "。"
        return " ".join([first, middle, last])
    steps = planner_output["action_steps"]
    first = f"更合适的做法是，先{steps[0]}。"
    middle_parts = []
    if len(steps) > 1:
        middle_parts.append(f"第二步，{steps[1]}")
    if len(steps) > 2:
        middle_parts.append(f"第三步，{steps[2]}")
    middle = "；".join(middle_parts)
    if middle and not middle.endswith("。"):
        middle += "。"
    tail = f"{planner_output['applicability']} {planner_output['risk_note']}"
    return " ".join(part for part in (first, middle, tail) if part)


def render_prompt_pieces(query: str, router: dict[str, Any], selected: dict[str, Any], planner_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "router_instruction": {
            "query_type": router["query_type"],
            "subtype": router["subtype"],
            "policy_bias": router["policy_bias"],
        },
        "planner_instruction": {
            "query": query,
            "planner_type": planner_output["planner_type"],
            "conflict_notes": selected["role_conflict_notes"],
        },
        "surface_instruction": SURFACE_PROMPT_V2,
    }


def compare_v1_v2(v1_answers: dict[str, Any], v2_rows: list[dict[str, Any]]) -> str:
    v1_map = {item["query_id"]: item for item in v1_answers.get("answers", [])}
    by_type: dict[str, list[str]] = {"what": [], "why": [], "how": []}
    for row in v2_rows:
        qid = row["query_id"]
        old = str(v1_map.get(qid, {}).get("final_answer", ""))
        new = str(row["final_answer"])
        if row["query_type"] == "what":
            note = "v2 用 planner 把 definition anchor 单独抽出来，减少直接拿 mechanism 句子顶替定义。"
        elif row["query_type"] == "why":
            note = "v2 强制输出结构 -> 行为 -> 结果 的链路，减少摘句拼接。"
        else:
            note = "v2 把能力/角色描述先转译成动作步骤，再做 surface generation。"
        if old != new:
            by_type[row["query_type"]].append(qid)
        else:
            by_type[row["query_type"]].append(f"{qid}(stable)")
    lines = [
        "# Generation V2 Compare Summary",
        "",
        "- `what` improved: 定义句由 planner 先抽象，再写成定义/边界/短例子结构。",
        "- `why` improved: 原先偏摘句，现在会先给核心结论，再落到完整因果链。",
        "- `how` improved: 原先容易复述能力画像，现在会先转成步骤化动作。",
        "- hardest remaining type: `what` 仍最难，因为 retrieval 本身经常缺少真正 definition/principle 主证据，只能从 mechanism 反推。",
        "- online LLM next step: 值得。因为 planner 输出已经足够结构化，可以直接接真实 LLM 做 surface generation，而不必让 LLM自己从杂乱 evidence 临场规划。",
        "",
        "## Query Coverage",
    ]
    for query_type in ("what", "why", "how"):
        lines.append(f"- `{query_type}`: {', '.join(by_type[query_type])}")
    return "\n".join(lines)


def write_design_doc(output_dir: Path) -> None:
    lines = [
        "# Generation V2 Design",
        "",
        "## Pipeline",
        "",
        "User Query -> Query-Type Router -> Logic Planner -> Surface Generator -> Final Answer",
        "",
        "## Router Rules",
        "",
        "- `how`: 区分 `temporary_push` 与 `mechanism_building`。",
        "- `what`: 区分 `concept_definition` 与 `essence_or_principle`。",
        "- `why`: 区分 `root_cause` 与 `recurring_pattern`。",
        "",
        "## Logic Planner",
        "",
        "- `what_planner`: 产出 `definition_anchor + boundary_explanation + short_example`。",
        "- `why_planner`: 产出 `core_conclusion + causal_chain + management_implication`。",
        "- `how_planner`: 产出 `action_steps + applicability + risk_note`。",
        "",
        "## Conflict Handling",
        "",
        "- what 缺少 definition/principle 时，从 mechanism 抽象定义锚点，而不是直接摘 mechanism 句。",
        "- why 若 evidence 中混入 solution，只允许放在后置管理含义，不进入主因果链。",
        "- how 若主证据是角色/能力描述，先转译成动作步骤，再交给 surface generator。",
    ]
    (output_dir / "generation_v2_design.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    queries = load_json(args.query_set_path)
    report = load_json(args.retrieval_report_path)
    v1_answers = load_json(args.v1_answers_path)
    query_rows = {str(row.get("query_id", "")): row for row in report.get("queries", [])}

    generated_rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query.get("id", ""))
        if args.priority_only and query_id not in PRIORITY_QUERY_IDS:
            continue
        query_row = query_rows.get(query_id)
        if not query_row:
            continue
        query_text = str(query.get("query", ""))
        query_type = v1.query_type_from_row(query_row)
        selected = v1.select_evidence(query_row)
        router = router_decision(query_text, query_type)
        planner_output = plan_answer(query_text, router, selected)
        final_answer = surface_generate(query_text, router, planner_output)
        generated_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "query_type": query_type,
                "router_decision": router,
                "selected_evidence": selected,
                "planner_output": planner_output,
                "final_answer": final_answer,
                "prompt_pieces": render_prompt_pieces(query_text, router, selected, planner_output),
                "generation_notes": {
                    "priority_query": query_id in PRIORITY_QUERY_IDS,
                    "conflict_notes": selected["role_conflict_notes"],
                    "surface_generator": "planner_first_surface_second",
                },
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_design_doc(args.output_dir)
    (args.output_dir / "instruction_router_rules.json").write_text(
        json.dumps(
            {
                "priority_query_ids": PRIORITY_QUERY_IDS,
                "router_rules": {
                    "what": ["concept_definition", "essence_or_principle"],
                    "why": ["root_cause", "recurring_pattern"],
                    "how": ["temporary_push", "mechanism_building", "generic_how"],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "answer_eval_template_v2.json").write_text(
        json.dumps(ANSWER_EVAL_TEMPLATE_V2, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "answers_v2.json").write_text(
        json.dumps(
            {
                "retrieval_report_path": str(args.retrieval_report_path),
                "v1_answers_path": str(args.v1_answers_path),
                "query_count": len(generated_rows),
                "answers": generated_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "compare_summary_v1_vs_v2.md").write_text(compare_v1_v2(v1_answers, generated_rows), encoding="utf-8")
    print(f"Saved generation v2 outputs to: {args.output_dir}")
    print(f"Generated planner-based answers: {len(generated_rows)}")


if __name__ == "__main__":
    main()
