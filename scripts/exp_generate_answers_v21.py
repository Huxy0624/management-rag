#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import exp_generate_answers_v1 as v1
import exp_generate_answers_v2 as v2


DEFAULT_QUERY_SET_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/query_set_phase2.json")
DEFAULT_RETRIEVAL_REPORT_PATH = Path("data/pipeline_candidates/v4_phase22/retrieval_eval_v4/report.json")
DEFAULT_V2_ANSWERS_PATH = Path("data/pipeline_candidates/generation_v2/answers_v2.json")
DEFAULT_OUTPUT_DIR = Path("data/pipeline_candidates/generation_v21")

HOW_PRIORITY_QUERY_IDS = [
    "cross_dept_how_temporary_push",
    "cross_dept_how_mechanism_fix",
    "upward_compression",
    "downward_diffusion",
    "rule_of_man_fix",
    "rule_of_law_fix",
]

FORBIDDEN_ACTION_PHRASES = (
    "具备",
    "能够",
    "注重",
    "意识到",
    "感受到",
    "平衡利益",
    "推动发展",
    "转成明确机制动作",
    "做一次协调和拍板",
    "围绕“",
)

ACTION_VERBS = ("编制", "指派", "对齐", "广播", "拆解", "追踪", "记录", "升级", "复盘", "设定", "考核", "归档", "同步")

MECHANISM_TEMPLATE_LIBRARY = {
    "cross_dept": [
        {
            "name": "跨部门项目周会",
            "type": "information",
            "fields_or_rhythm": "固定字段：事项、主责部门、协作部门、阻塞项、截止时间、升级条件",
            "owner": "项目负责人",
            "cadence": "每周一次",
        },
        {
            "name": "责任边界表",
            "type": "evaluation",
            "fields_or_rhythm": "固定字段：事项、输入、输出、主责人、配合人、超时责任",
            "owner": "部门负责人",
            "cadence": "项目启动时建立，变更时更新",
        },
        {
            "name": "跨部门复盘单",
            "type": "correction",
            "fields_or_rhythm": "固定字段：事件、根因、责任边界、纠偏动作、复查日期",
            "owner": "项目负责人+上级管理者",
            "cadence": "里程碑后或事故后触发",
        },
    ],
    "info": [
        {
            "name": "信息-行为映射表",
            "type": "information",
            "fields_or_rhythm": "固定字段：信息类型、生成动作、接收角色、更新时间、异常升级人",
            "owner": "部门接口人",
            "cadence": "每周更新",
        },
        {
            "name": "向上汇报模板",
            "type": "information",
            "fields_or_rhythm": "固定字段：目标、当前进度、偏差、风险、需要拍板事项",
            "owner": "汇报责任人",
            "cadence": "日报或关键节点更新",
        },
        {
            "name": "反馈广播机制",
            "type": "correction",
            "fields_or_rhythm": "广播节奏：事项同步、偏差反馈、决策回传、责任确认",
            "owner": "项目负责人",
            "cadence": "当日闭环",
        },
    ],
    "strategy": [
        {
            "name": "战略拆解表",
            "type": "information",
            "fields_or_rhythm": "固定字段：战略目标、团队目标、岗位动作、里程碑、检查点",
            "owner": "部门负责人",
            "cadence": "月度更新",
        },
        {
            "name": "战略向下解释周会",
            "type": "information",
            "fields_or_rhythm": "固定议程：目标解释、任务拆解、疑问收集、偏差澄清",
            "owner": "直接主管",
            "cadence": "每周一次",
        },
        {
            "name": "理解偏差复盘单",
            "type": "correction",
            "fields_or_rhythm": "固定字段：误解点、触发原因、纠偏动作、责任人、复查时间",
            "owner": "主管+项目负责人",
            "cadence": "发现偏差后48小时内",
        },
    ],
    "evaluation": [
        {
            "name": "评价标准表",
            "type": "evaluation",
            "fields_or_rhythm": "固定字段：岗位、目标、行为标准、结果指标、扣分项",
            "owner": "部门负责人+HR",
            "cadence": "季度校准",
        },
        {
            "name": "交叉考核规则",
            "type": "evaluation",
            "fields_or_rhythm": "固定对象：主责部门、协作部门、评分项、申诉入口、复核人",
            "owner": "上级管理者",
            "cadence": "月度执行",
        },
        {
            "name": "问题复盘机制",
            "type": "correction",
            "fields_or_rhythm": "固定字段：问题、根因、责任边界、纠偏动作、下次检查时间",
            "owner": "业务负责人",
            "cadence": "每次问题关闭后触发",
        },
    ],
    "generic_governance": [
        {
            "name": "问题分级表",
            "type": "information",
            "fields_or_rhythm": "固定字段：问题级别、触发条件、责任人、升级路径、处理时限",
            "owner": "部门负责人",
            "cadence": "问题出现时即时更新",
        },
        {
            "name": "周节奏治理会",
            "type": "information",
            "fields_or_rhythm": "固定议程：本周问题、纠偏动作、阻塞项、下周检查点",
            "owner": "直属上级",
            "cadence": "每周一次",
        },
        {
            "name": "复盘与考核联动表",
            "type": "correction",
            "fields_or_rhythm": "固定字段：问题、根因、纠偏动作、责任人、考核项",
            "owner": "直属上级+HR",
            "cadence": "每月复核",
        },
    ],
}

ANSWER_EVAL_TEMPLATE_V21 = {
    "dimensions": [
        {
            "name": "first_sentence_alignment",
            "scale": "1-5",
            "question": "第一句是否正面回应问题。",
        },
        {
            "name": "structure_fitness",
            "scale": "1-5",
            "question": "是否符合目标结构，尤其是 how 是否先给动作。",
        },
        {
            "name": "planner_faithfulness",
            "scale": "1-5",
            "question": "最终答案是否忠实执行 planner，而不是退回摘句。",
        },
        {
            "name": "actionability_score",
            "scale": "1-5",
            "question": "action_steps 是否包含明确对象、动作、交付物，并避免空话。",
        },
        {
            "name": "mechanism_specificity_score",
            "scale": "1-5",
            "question": "mechanism_building 是否落到真实机制实体，包含责任人、节奏、字段或考核对象。",
        },
        {
            "name": "definition_quality",
            "scale": "1-5",
            "question": "what 类定义是否清楚且层级正确。",
        },
        {
            "name": "causal_integrity",
            "scale": "1-5",
            "question": "why 类因果链是否完整。",
        },
    ],
    "per_answer_fields": [
        "query_id",
        "query",
        "router_decision",
        "score_by_dimension",
        "planner_comment",
        "needs_online_llm",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation v2.1 action translator and mechanism mapper prototype.")
    parser.add_argument("--query-set-path", type=Path, default=DEFAULT_QUERY_SET_PATH)
    parser.add_argument("--retrieval-report-path", type=Path, default=DEFAULT_RETRIEVAL_REPORT_PATH)
    parser.add_argument("--v2-answers-path", type=Path, default=DEFAULT_V2_ANSWERS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--priority-only", action="store_true", help="Only emit the key how queries plus regression coverage.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def router_decision_v21(query: str, query_type: str) -> dict[str, Any]:
    base = v2.router_decision(query, query_type)
    normalized = v1.normalize_text(query)
    if query_type != "how":
        base["human_vs_system"] = "not_applicable"
        return base
    if any(token in normalized for token in ("临时", "先", "救火", "推进", "老板问进度", "向上汇报", "向下解释")):
        base["human_vs_system"] = "human_intervention"
        base["execution_frame"] = "temporary_push"
    elif any(token in normalized for token in ("机制", "长期", "体系", "法治", "根治")):
        base["human_vs_system"] = "system_governance"
        base["execution_frame"] = "mechanism_building"
    else:
        base["human_vs_system"] = "mixed"
        base["execution_frame"] = "generic_how"
    return base


def compact_snippets(rows: list[dict[str, Any]], limit: int = 6) -> list[str]:
    return v2.compact_snippets(rows, limit=limit)


def contains_forbidden_phrase(text: str) -> bool:
    return any(token in text for token in FORBIDDEN_ACTION_PHRASES)


def ensure_dynamic_action(text: str) -> str:
    cleaned = v1.clean_snippet(text)
    if contains_forbidden_phrase(cleaned):
        return cleaned
    if any(cleaned.startswith(verb) for verb in ACTION_VERBS):
        return cleaned
    return f"拆解并{cleaned}"


def build_step(step: int, goal: str, action: str, deliverable: str, obj: str) -> dict[str, Any]:
    return {
        "step": step,
        "goal": goal,
        "object": obj,
        "action": action,
        "deliverable": deliverable,
    }


def action_translator(query: str, router: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    normalized = v1.normalize_text(query)
    main = selected["main_evidence"]
    snippets = compact_snippets(main + selected["support_evidence"], limit=8)

    if "跨部门合作" in normalized and router["execution_frame"] == "temporary_push":
        steps = [
            build_step(
                1,
                "先把协同目标对齐到一张纸上",
                "对齐依赖部门负责人和内部负责人，编制一页协同单，写清目标、阻塞项、截止时间",
                "跨部门协同单",
                "依赖部门负责人、内部负责人",
            ),
            build_step(
                2,
                "把责任和拍板链路固定下来",
                "指派单一责任人，同步拍板人和升级条件，并在群内广播完成时点",
                "责任人表+升级规则",
                "责任人、拍板人、相关群组",
            ),
            build_step(
                3,
                "用短周期追踪推进结果",
                "追踪关键阻塞项，当日记录偏差并升级未闭环事项",
                "当日进度更新",
                "项目负责人、阻塞事项",
            ),
        ]
    elif "跨部门协作" in normalized and router["execution_frame"] == "mechanism_building":
        steps = [
            build_step(
                1,
                "先把协同责任从口头要求改成规则",
                "设定主责部门、协作部门、输入输出和超时责任，归档到责任边界表",
                "责任边界表",
                "主责部门、协作部门",
            ),
            build_step(
                2,
                "让跨部门信息按固定节奏流动",
                "编制跨部门周会模板，按周同步事项、阻塞项、截止时间和升级条件",
                "跨部门周会纪要",
                "项目负责人、各部门接口人",
            ),
            build_step(
                3,
                "把偏差纠偏变成固定闭环",
                "记录协同事故并复盘根因，对重复问题升级到上级管理者处理",
                "复盘单+升级记录",
                "项目负责人、上级管理者",
            ),
        ]
    elif "向上汇报" in normalized or "压缩信息" in normalized:
        steps = [
            build_step(
                1,
                "先把汇报内容压成决策视角",
                "拆解当前进度，只保留目标、偏差、风险和需要拍板事项",
                "一页汇报卡",
                "老板/高管",
            ),
            build_step(
                2,
                "把细节折成少数关键数字",
                "同步进度数字、风险等级和下个时间点，不展开过程噪音",
                "进度摘要",
                "汇报责任人",
            ),
            build_step(
                3,
                "让汇报形成后续动作",
                "记录老板拍板项并广播到执行责任人，追踪下次更新时间",
                "拍板事项清单",
                "执行责任人、老板",
            ),
        ]
    elif "向下解释" in normalized or "战略传到一线" in normalized:
        steps = [
            build_step(
                1,
                "先把战略翻译成团队任务",
                "拆解战略目标，编制团队目标表，写清岗位动作和里程碑",
                "团队目标拆解表",
                "主管、一线团队",
            ),
            build_step(
                2,
                "把抽象目标讲成具体动作",
                "同步每个岗位本周要做的动作、判断标准和交付时间",
                "岗位动作清单",
                "一线员工、直接主管",
            ),
            build_step(
                3,
                "追踪理解偏差并纠偏",
                "记录一线疑问，汇总误解点，并在下一次周会统一广播纠偏",
                "Q&A记录+纠偏说明",
                "主管、团队成员",
            ),
        ]
    elif "人治" in normalized:
        steps = [
            build_step(
                1,
                "先让问题挂到具体责任人身上",
                "指派临时责任人，写清当天必须完成的动作和反馈时间",
                "临时责任单",
                "临时责任人",
            ),
            build_step(
                2,
                "补上拍板与升级链路",
                "同步上级管理者作为拍板人，明确什么情况要直接升级",
                "拍板链路说明",
                "上级管理者、执行责任人",
            ),
            build_step(
                3,
                "用短周期反馈把事压过去",
                "当天同步结果、记录卡点、继续升级未完成事项",
                "当日闭环记录",
                "执行责任人、相关方",
            ),
        ]
    elif "机制" in normalized or "长期" in normalized:
        steps = [
            build_step(
                1,
                "先定义规则和责任边界",
                "设定标准、责任人和超时后果，归档到规则表",
                "规则表",
                "部门负责人、协作对象",
            ),
            build_step(
                2,
                "再建立固定节奏的同步机制",
                "编制周会或月会模板，按节奏同步问题、风险和决策项",
                "节奏化会议纪要",
                "机制 owner、相关部门",
            ),
            build_step(
                3,
                "最后用复盘和考核把机制锁住",
                "记录重复问题、复盘根因，并把执行结果纳入考核",
                "复盘记录+考核项",
                "机制 owner、上级管理者",
            ),
        ]
    else:
        steps = [
            build_step(
                1,
                "先把问题拆成动作",
                "拆解目标、责任人和时点，编制执行清单",
                "执行清单",
                "直接责任人",
            ),
            build_step(
                2,
                "再按节奏推进",
                "同步进度、记录偏差、追踪未完成事项",
                "进度记录",
                "相关方",
            ),
            build_step(
                3,
                "最后形成闭环",
                "复盘结果并归档下次复用的做法",
                "复盘记录",
                "负责人、团队",
            ),
        ]

    sanitized_steps: list[dict[str, Any]] = []
    for item in steps:
        action = str(item["action"])
        if contains_forbidden_phrase(action):
            action = action.replace("围绕“", "编制").replace("”去做一次短期协调和拍板", "执行单")
        item["action"] = action
        sanitized_steps.append(item)

    return {
        "translator_name": "action_translator_v21",
        "action_steps": sanitized_steps,
        "actionability_checks": {
            "has_object": all(bool(step["object"]) for step in sanitized_steps),
            "has_deliverable": all(bool(step["deliverable"]) for step in sanitized_steps),
            "avoids_forbidden_phrases": all(not contains_forbidden_phrase(step["action"]) for step in sanitized_steps),
        },
    }


def choose_mechanism_templates(query: str, selected: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = v1.normalize_text(query)
    joined = " ".join(compact_snippets(selected["main_evidence"] + selected["support_evidence"], limit=10))
    if "跨部门" in normalized:
        return MECHANISM_TEMPLATE_LIBRARY["cross_dept"]
    if "汇报" in normalized or "信息" in normalized or "压缩" in normalized:
        return MECHANISM_TEMPLATE_LIBRARY["info"]
    if "战略" in normalized or "一线" in normalized:
        return MECHANISM_TEMPLATE_LIBRARY["strategy"]
    if "评价" in normalized or "奖惩" in joined or "复盘" in normalized:
        return MECHANISM_TEMPLATE_LIBRARY["evaluation"]
    if any(token in normalized for token in ("机制", "长期", "体系", "法治", "规则")):
        return MECHANISM_TEMPLATE_LIBRARY["generic_governance"]
    return []


def mechanism_template_mapper(query: str, router: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    if router["execution_frame"] != "mechanism_building":
        return {"mapper_name": "mechanism_template_mapper_v21", "mechanism_entities": []}
    entities = choose_mechanism_templates(query, selected)
    return {
        "mapper_name": "mechanism_template_mapper_v21",
        "mechanism_entities": entities,
        "specificity_checks": {
            "has_entity": bool(entities),
            "has_owner": all(bool(entity["owner"]) for entity in entities),
            "has_cadence": all(bool(entity["cadence"]) for entity in entities),
            "has_fields_or_rhythm": all(bool(entity["fields_or_rhythm"]) for entity in entities),
        },
    }


def plan_how_v21(query: str, router: dict[str, Any], selected: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    action_output = action_translator(query, router, selected)
    mechanism_output = mechanism_template_mapper(query, router, selected)
    applicability = ""
    risk_note = ""
    if router["execution_frame"] == "temporary_push":
        applicability = "适用于事情已经卡住、需要先拿结果再补机制的场景。"
        risk_note = "常见误区是只催人不留记录，导致下一次还会回到同样的协同卡点。"
    elif router["execution_frame"] == "mechanism_building":
        applicability = "适用于要把跨人、跨部门、跨周期的问题固化进规则和节奏的场景。"
        risk_note = "常见误区是只说建立机制，却没有 owner、节奏、字段或考核对象。"
    else:
        applicability = "适用于需要先落动作，再逐步沉淀流程的通用 how 场景。"
        risk_note = "常见误区是步骤很多，但没有责任人和交付物。"

    planner_output = {
        "planner_type": "how_planner_v21",
        "human_vs_system": router["human_vs_system"],
        "solution_mode": router["execution_frame"],
        "action_steps": action_output["action_steps"],
        "mechanism_entities": mechanism_output["mechanism_entities"],
        "applicability": applicability,
        "risk_note": risk_note,
    }
    return planner_output, action_output, mechanism_output


def surface_generate_v21(query: str, query_type: str, router: dict[str, Any], planner_output: dict[str, Any]) -> str:
    if query_type != "how":
        return v2.surface_generate(query, router, planner_output)
    steps = planner_output["action_steps"]
    first = f"更可执行的做法是，第一步先{steps[0]['action']}，产出《{steps[0]['deliverable']}》。"
    second_parts: list[str] = []
    if len(steps) > 1:
        second_parts.append(f"第二步对{steps[1]['object']}{steps[1]['action']}，形成《{steps[1]['deliverable']}》")
    if len(steps) > 2:
        second_parts.append(f"第三步继续对{steps[2]['object']}{steps[2]['action']}，留下《{steps[2]['deliverable']}》")
    second = "；".join(second_parts) + "。" if second_parts else ""
    mechanism_line = ""
    if planner_output["mechanism_entities"]:
        names = "、".join(entity["name"] for entity in planner_output["mechanism_entities"][:3])
        mechanism_line = f"如果要长期固化，就把动作沉淀为{names}，并明确责任人和执行节奏。"
    return " ".join(part for part in (first, second, planner_output["applicability"], mechanism_line, planner_output["risk_note"]) if part)


def diff_summary(v2_row: dict[str, Any] | None, v21_row: dict[str, Any]) -> str:
    if not v2_row:
        return "new_output_only"
    if v21_row["query_type"] != "how":
        return "non_how_regression_check_only"
    old_steps = v2_row.get("planner_output", {}).get("action_steps", [])
    new_steps = v21_row.get("planner_output_v21", {}).get("action_steps", [])
    old_mech = v2_row.get("planner_output", {}).get("mechanism_entities", [])
    new_mech = v21_row.get("planner_output_v21", {}).get("mechanism_entities", [])
    return (
        f"action_steps: {len(old_steps)} -> {len(new_steps)}; "
        f"mechanism_entities: {len(old_mech)} -> {len(new_mech)}; "
        f"router: {v2_row.get('router_decision', {}).get('subtype')} -> {v21_row.get('router_decision', {}).get('execution_frame')}"
    )


def compare_v2_v21(v2_answers: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    v2_map = {item["query_id"]: item for item in v2_answers.get("answers", [])}
    lines = [
        "# Generation V2.1 Compare Summary",
        "",
        "- `what`: 只做回归验证，planner 与 v2 基本保持一致。",
        "- `why`: 只做回归验证，继续沿用 v2 的结构 -> 行为 -> 结果链路。",
        "- `how`: v2.1 重点把 how 从句子级 action 改成 WBS 风格的 `goal/action/deliverable`，并把 `mechanism_building` 映射为真实机制实体。",
        "- hardest remaining type: `how` 里的 generic_how 仍依赖 retrieval 证据质量，如果 evidence 太抽象，动作步骤仍会偏模板化。",
        "- online LLM next step: 值得，但前提是复用 v2.1 的 planner_output，不要让在线 LLM直接从 raw evidence 生成。",
        "",
        "## Priority How Queries",
    ]
    for query_id in HOW_PRIORITY_QUERY_IDS:
        row = next((item for item in rows if item["query_id"] == query_id), None)
        if not row:
            continue
        lines.append(f"- `{query_id}`: {diff_summary(v2_map.get(query_id), row)}")
    return "\n".join(lines)


def write_design_doc(output_dir: Path) -> None:
    lines = [
        "# Generation V2.1 Design",
        "",
        "## Focus",
        "",
        "- Action Translator: 把原则句、角色画像句、能力句翻译成 `goal/action/deliverable`。",
        "- Mechanism Template Mapper: 把 mechanism_building 映射成真实机制实体。",
        "- Human-vs-System Router: 区分 temporary_push 与 mechanism_building。",
        "",
        "## Router",
        "",
        "- `temporary_push / firefighting` -> 偏人治动作、责任人指定、快速反馈。",
        "- `mechanism_building / long_term_fix` -> 偏规则、流程、节奏、考核约束。",
        "",
        "## Planner Output",
        "",
        "- `action_steps[]` 每步都包含 `goal/object/action/deliverable`。",
        "- `mechanism_entities[]` 必须包含 `name/type/fields_or_rhythm/owner/cadence`。",
        "",
        "## Guardrails",
        "",
        "- 禁止把 `具备/能够/注重/围绕某句话推进/转成明确机制动作` 这类空泛表达当 action 主体。",
        "- 若 mechanism_building 说不出 owner/cadence/fields，则不允许泛泛使用“机制”一词。",
    ]
    (output_dir / "generation_v21_design.md").write_text("\n".join(lines), encoding="utf-8")


def should_include(query_id: str, query_type: str, priority_only: bool) -> bool:
    if not priority_only:
        return True
    if query_type != "how":
        return query_id in {"cross_dept_definition", "cross_dept_why_low_efficiency"}
    return query_id in HOW_PRIORITY_QUERY_IDS


def main() -> None:
    args = parse_args()
    queries = load_json(args.query_set_path)
    report = load_json(args.retrieval_report_path)
    v2_answers = load_json(args.v2_answers_path)
    v2_map = {item["query_id"]: item for item in v2_answers.get("answers", [])}
    query_rows = {str(row.get("query_id", "")): row for row in report.get("queries", [])}

    generated_rows: list[dict[str, Any]] = []
    for query in queries:
        query_id = str(query.get("id", ""))
        query_row = query_rows.get(query_id)
        if not query_row:
            continue
        query_text = str(query.get("query", ""))
        query_type = v1.query_type_from_row(query_row)
        if not should_include(query_id, query_type, args.priority_only):
            continue
        selected = v1.select_evidence(query_row)
        router = router_decision_v21(query_text, query_type)
        if query_type == "how":
            planner_output, action_output, mechanism_output = plan_how_v21(query_text, router, selected)
        elif query_type == "what":
            planner_output = v2.plan_what(query_text, router, selected)
            action_output = {"translator_name": "action_translator_v21", "action_steps": []}
            mechanism_output = {"mapper_name": "mechanism_template_mapper_v21", "mechanism_entities": []}
        else:
            planner_output = v2.plan_why(query_text, router, selected)
            action_output = {"translator_name": "action_translator_v21", "action_steps": []}
            mechanism_output = {"mapper_name": "mechanism_template_mapper_v21", "mechanism_entities": []}
        final_answer = surface_generate_v21(query_text, query_type, router, planner_output)
        generated_rows.append(
            {
                "query_id": query_id,
                "query": query_text,
                "query_type": query_type,
                "router_decision": router,
                "selected_evidence": selected,
                "planner_output_v21": planner_output,
                "action_translator_output": action_output,
                "mechanism_mapper_output": mechanism_output,
                "final_answer": final_answer,
                "v2_vs_v21_diff": diff_summary(v2_map.get(query_id), {
                    "query_id": query_id,
                    "query_type": query_type,
                    "router_decision": router,
                    "planner_output_v21": planner_output,
                }),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_design_doc(args.output_dir)
    (args.output_dir / "answer_eval_template_v21.json").write_text(
        json.dumps(ANSWER_EVAL_TEMPLATE_V21, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "answers_v21.json").write_text(
        json.dumps(
            {
                "retrieval_report_path": str(args.retrieval_report_path),
                "v2_answers_path": str(args.v2_answers_path),
                "query_count": len(generated_rows),
                "answers": generated_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "compare_summary_v2_vs_v21.md").write_text(compare_v2_v21(v2_answers, generated_rows), encoding="utf-8")
    print(f"Saved generation v2.1 outputs to: {args.output_dir}")
    print(f"Generated answers: {len(generated_rows)}")


if __name__ == "__main__":
    main()
