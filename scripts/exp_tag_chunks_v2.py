#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v1/tagged_chunks/records.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v2/tagged_chunks/records.json")

ANSWER_ROLE_VOCAB = (
    "definition",
    "cause",
    "mechanism",
    "solution",
    "principle",
    "example",
    "warning",
    "comparison",
    "summary",
    "none",
)

ROOT_ISSUE_VOCAB = (
    "information_distortion",
    "evaluation_failure",
    "mixed",
    "none",
)

GOV_MODE_VOCAB = (
    "rule_of_man",
    "rule_of_law",
    "hybrid",
    "none",
)

INTENT_VOCAB = (
    "selection",
    "succession",
    "resource_coordination",
    "coaching",
    "mechanism_design",
    "performance_design",
    "performance_repair",
    "cross_function_alignment",
    "none",
)

SOLUTION_TYPE_VOCAB = (
    "energy_extension",
    "priority_alignment",
    "risk_escalation",
    "mechanism_building",
    "authority_borrowing",
    "responsibility_binding",
    "none",
)

LLM_TAGGING_PROMPT = """你是一个治理语义标注器。你的任务不是总结主题，而是判断这段内容在治理逻辑上回答了什么问题。

必须遵守：
1. 只能从给定词表中选择，不允许生成新词。
2. 先判断 answer_role，再判断 root_issue / gov_mode / intent / solution_type。
3. 如果证据不足，使用 none，不要硬猜。
4. 标签优先服务检索匹配，而不是内容概述。
5. 输出必须是合法 JSON，不要输出解释。

固定词表：
- answer_role: definition | cause | mechanism | solution | principle | example | warning | comparison | summary | none
- root_issue: information_distortion | evaluation_failure | mixed | none
- gov_mode: rule_of_man | rule_of_law | hybrid | none
- intent: selection | succession | resource_coordination | coaching | mechanism_design | performance_design | performance_repair | cross_function_alignment | none
- solution_type: energy_extension | priority_alignment | risk_escalation | mechanism_building | authority_borrowing | responsibility_binding | none

优先判断顺序：
1. answer_role
2. root_issue
3. gov_mode
4. intent
5. solution_type

输入：
- title
- chunk_text
- insights
- legacy_tags（仅辅助，不得主导）

输出格式：
{
  "answer_role": "...",
  "root_issue": "...",
  "gov_mode": "...",
  "intent": "...",
  "solution_type": "..."
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag chunk + insight records with governance matrix MVP fields.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input tagged chunk JSON path.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output tagged JSON path.")
    parser.add_argument("--limit", type=int, help="Optional source file limit for debug runs.")
    parser.add_argument("--source-file", type=str, help="Optional source_file substring filter.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = text.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def filter_records(records: list[dict[str, Any]], source_file: str | None, limit: int | None) -> list[dict[str, Any]]:
    filtered = records
    if source_file:
        filtered = [record for record in filtered if source_file in str(record.get("source_file", ""))]
    if limit is None:
        return filtered

    kept_sources: list[str] = []
    for record in filtered:
        current = str(record.get("source_file", ""))
        if current and current not in kept_sources:
            kept_sources.append(current)
        if len(kept_sources) >= limit:
            break
    kept_set = set(kept_sources)
    return [record for record in filtered if str(record.get("source_file", "")) in kept_set]


def joined_text(record: dict[str, Any]) -> tuple[str, str, str]:
    title = normalize_text(str(record.get("title", "")))
    chunk_text = normalize_text(str(record.get("chunk_text", "")))
    insights_text = normalize_text(
        " ".join(
            str(item.get("insight_text", ""))
            for item in record.get("insights", [])
            if isinstance(item, dict)
        )
    )
    return title, chunk_text, insights_text


def legacy_text(record: dict[str, Any]) -> str:
    tags = dict(record.get("tags", {}))
    concepts = dict(tags.get("concepts", {}))
    return normalize_text(
        " ".join(
            [
                " ".join(str(item) for item in tags.get("topic_tags", [])),
                " ".join(str(item) for item in tags.get("scenario_tags", [])),
                " ".join(str(item) for item in concepts.get("explicit", [])),
                " ".join(str(item) for item in concepts.get("inferred", [])),
                " ".join(str(item) for item in concepts.get("original_terms", [])),
                str(tags.get("content_type", "")),
            ]
        )
    )


def keyword_hits(text: str, keywords: tuple[str, ...] | list[str]) -> list[str]:
    normalized = compact_text(text)
    return [keyword for keyword in keywords if keyword and keyword in normalized]


def choose_answer_role(record: dict[str, Any], full_text: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    insight_types = [
        str(item.get("insight_type", ""))
        for item in record.get("insights", [])
        if isinstance(item, dict)
    ]

    if keyword_hits(full_text, ["什么是", "本质是", "是指", "定义", "管理的本质", "意味着"]):
        hits.append("definition_rule")
        return "definition", hits
    if keyword_hits(full_text, ["为什么", "根源", "原因", "之所以", "归因于", "导致", "引发"]):
        if keyword_hits(full_text, ["机制", "长期", "系统", "传递", "失真", "失效", "结构", "层级"]):
            hits.append("mechanism_rule")
            return "mechanism", hits
        hits.append("cause_rule")
        return "cause", hits
    if keyword_hits(full_text, ["应该", "怎么", "如何", "做法", "解法", "路径", "通过", "建立", "优化", "先", "处理"]):
        if keyword_hits(full_text, ["案例", "比如", "举例", "真实案例", "事故"]):
            hits.append("example_rule")
            return "example", hits
        hits.append("solution_rule")
        return "solution", hits
    if keyword_hits(full_text, ["不要", "不能", "风险", "反噬", "误区", "失灵", "陷阱", "代价"]):
        hits.append("warning_rule")
        return "warning", hits
    if keyword_hits(full_text, ["对比", "平衡", "并非对立", "不是...而是", "什么时候", "相反"]):
        hits.append("comparison_rule")
        return "comparison", hits
    if keyword_hits(full_text, ["综上", "总结", "总之", "可以看出"]):
        hits.append("summary_rule")
        return "summary", hits
    if "case_takeaway" in insight_types or keyword_hits(full_text, ["案例", "事故", "扯皮", "举个例子", "真实案例"]):
        hits.append("example_fallback")
        return "example", hits
    if keyword_hits(full_text, ["核心", "原则", "关键", "重点", "目标", "意义"]):
        hits.append("principle_fallback")
        return "principle", hits
    return "none", hits


def choose_root_issue(record: dict[str, Any], full_text: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    info_hits = keyword_hits(full_text, ["信息失真", "信息差", "沟通复杂度", "层级", "战略传导", "向下传递", "噪声"])
    eval_hits = keyword_hits(full_text, ["评价失效", "评价标准", "奖惩", "资源配置", "绩效评价", "评价权", "是否有价值"])
    legacy = compact_text(legacy_text(record))
    if "信息失真" in legacy and "评价失效" in legacy:
        info_hits.append("legacy_information_distortion")
        eval_hits.append("legacy_evaluation_failure")
    elif "信息失真" in legacy:
        info_hits.append("legacy_information_distortion")
    elif "评价失效" in legacy:
        eval_hits.append("legacy_evaluation_failure")

    if info_hits and eval_hits:
        hits.extend(info_hits[:2] + eval_hits[:2])
        return "mixed", hits
    if info_hits:
        hits.extend(info_hits[:3])
        return "information_distortion", hits
    if eval_hits:
        hits.extend(eval_hits[:3])
        return "evaluation_failure", hits
    return "none", hits


def choose_gov_mode(full_text: str) -> tuple[str, list[str]]:
    hits: list[str] = []
    man_hits = keyword_hits(full_text, ["人治", "站台", "拍板", "借力", "投诉", "老板出面", "硬推", "个人魅力", "救火", "先顶住", "临时"])
    law_hits = keyword_hits(full_text, ["法治", "机制", "制度", "流程", "规则", "体系", "复盘", "长期", "标准", "SOP", "结构性"])
    if man_hits and law_hits:
        hits.extend(man_hits[:2] + law_hits[:2])
        return "hybrid", hits
    if law_hits:
        hits.extend(law_hits[:3])
        return "rule_of_law", hits
    if man_hits:
        hits.extend(man_hits[:3])
        return "rule_of_man", hits
    return "none", hits


def choose_intent(full_text: str) -> tuple[str, list[str]]:
    mapping = [
        ("succession", ["接班", "梯队", "储备干部", "后备", "继任"]),
        ("selection", ["选拔", "候选人", "晋升", "提拔", "任命"]),
        ("cross_function_alignment", ["跨部门", "部门墙", "协作", "对齐", "扯皮"]),
        ("resource_coordination", ["资源协调", "资源", "协调", "排期", "依赖方", "控制权"]),
        ("coaching", ["培养", "辅导", "带人", "成长", "教练", "反馈"]),
        ("mechanism_design", ["机制", "制度", "规则", "SOP", "流程", "体系建设"]),
        ("performance_design", ["绩效设计", "绩效指标", "OKR", "考核设计", "评价标准"]),
        ("performance_repair", ["评价失效", "绩效失效", "复盘", "纠偏", "修复", "补救"]),
    ]
    for label, keywords in mapping:
        hits = keyword_hits(full_text, keywords)
        if hits:
            return label, hits[:3]
    return "none", []


def choose_solution_type(answer_role: str, gov_mode: str, intent: str, full_text: str) -> tuple[str, list[str]]:
    if answer_role not in {"solution", "mechanism", "warning", "example"}:
        return "none", []

    mapping = [
        ("mechanism_building", ["机制", "制度", "规则", "流程", "复盘", "体系", "标准"]),
        ("authority_borrowing", ["领导站台", "老板出面", "借力", "借势", "抄送CEO", "拉着老板"]),
        ("risk_escalation", ["升级", "上升", "暴露风险", "提前汇报", "风险上报", "预警"]),
        ("responsibility_binding", ["责任", "负责", "绑定", "问责", "拍板权", "说了算"]),
        ("priority_alignment", ["优先级", "目标对齐", "聚焦", "最重要", "减少无效工作"]),
        ("energy_extension", ["加班", "冲刺", "扛", "顶住", "扩大战线", "多投入"]),
    ]
    for label, keywords in mapping:
        hits = keyword_hits(full_text, keywords)
        if hits:
            return label, hits[:3]

    if gov_mode == "rule_of_law" and intent in {"mechanism_design", "performance_repair"}:
        return "mechanism_building", ["derived_mechanism_building"]
    if gov_mode in {"rule_of_man", "hybrid"} and intent in {"cross_function_alignment", "resource_coordination"}:
        return "authority_borrowing", ["derived_authority_borrowing"]
    return "none", []


def sanitize_vocab(value: str, vocab: tuple[str, ...]) -> str:
    return value if value in vocab else "none"


def build_prompt(record: dict[str, Any]) -> str:
    title, chunk_text, insights_text = joined_text(record)
    return (
        f"{LLM_TAGGING_PROMPT}\n\n"
        f"title:\n{title}\n\n"
        f"chunk_text:\n{chunk_text}\n\n"
        f"insights:\n{insights_text}\n\n"
        f"legacy_tags:\n{legacy_text(record)}\n"
    )


def extract_with_llm(record: dict[str, Any]) -> dict[str, str]:
    raise NotImplementedError("LLM extraction is intentionally not enabled in the MVP experiment.")


def extract_with_rules(record: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    title, chunk_text, insights_text = joined_text(record)
    full_text = compact_text(" ".join([title, insights_text, chunk_text, legacy_text(record)]))

    answer_role, answer_hits = choose_answer_role(record, full_text)
    root_issue, root_hits = choose_root_issue(record, full_text)
    gov_mode, gov_hits = choose_gov_mode(full_text)
    intent, intent_hits = choose_intent(full_text)
    solution_type, solution_hits = choose_solution_type(answer_role, gov_mode, intent, full_text)

    tags = {
        "answer_role": sanitize_vocab(answer_role, ANSWER_ROLE_VOCAB),
        "root_issue": sanitize_vocab(root_issue, ROOT_ISSUE_VOCAB),
        "gov_mode": sanitize_vocab(gov_mode, GOV_MODE_VOCAB),
        "intent": sanitize_vocab(intent, INTENT_VOCAB),
        "solution_type": sanitize_vocab(solution_type, SOLUTION_TYPE_VOCAB),
    }
    debug = {
        "answer_role_hits": answer_hits,
        "root_issue_hits": root_hits,
        "gov_mode_hits": gov_hits,
        "intent_hits": intent_hits,
        "solution_type_hits": solution_hits,
    }
    return tags, debug


def compute_manual_review(tags: dict[str, str], debug: dict[str, list[str]]) -> bool:
    none_count = sum(1 for value in tags.values() if value == "none")
    if none_count >= 4:
        return True
    if tags["solution_type"] != "none" and tags["answer_role"] not in {"solution", "mechanism", "warning", "example"}:
        return True
    if tags["root_issue"] == "none" and tags["intent"] in {"performance_repair", "cross_function_alignment", "resource_coordination"}:
        return True
    if tags["gov_mode"] == "none" and tags["solution_type"] != "none":
        return True
    if not any(debug.values()):
        return True
    return False


def transform_record(record: dict[str, Any]) -> dict[str, Any]:
    governance_tags_v2, tagging_debug = extract_with_rules(record)
    return {
        "source_file": str(record.get("source_file", "")),
        "title": str(record.get("title", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "chunk_order": int(record.get("chunk_order", 0)),
        "chunk_text": str(record.get("chunk_text", "")),
        "insights": record.get("insights", []),
        "legacy_tags": record.get("tags", {}),
        "governance_tags_v2": governance_tags_v2,
        "tagging_debug": tagging_debug,
        "needs_manual_review": compute_manual_review(governance_tags_v2, tagging_debug),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    records = json.loads(args.input_path.read_text(encoding="utf-8"))
    filtered = filter_records(records, args.source_file, args.limit)
    transformed = [transform_record(record) for record in filtered]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input file: {args.input_path}")
    print(f"Total input records: {len(records)}")
    print(f"Selected records: {len(filtered)}")
    print(f"Unique sources: {len({item['source_file'] for item in transformed})}")
    print(f"Needs manual review: {sum(1 for item in transformed if item['needs_manual_review'])}")
    print(f"Saved v2 governance tags to: {args.output_path}")


if __name__ == "__main__":
    main()
