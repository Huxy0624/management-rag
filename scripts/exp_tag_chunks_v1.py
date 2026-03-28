#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_PATH_V2 = Path("data/experiments/insights_v2/sample_insights.json")
DEFAULT_INPUT_PATH_V1 = Path("data/experiments/insights_v1/sample_insights.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/tagging_v1/sample_tagged_chunks.json")
DEFAULT_LIMIT = 5

TOPIC_TAG_VOCAB = [
    "管理认知",
    "人效管理",
    "组织设计",
    "跨部门协作",
    "向上管理",
    "执行管理",
    "绩效管理",
    "团队治理",
    "管理选拔",
    "组织扩张",
    "文化建设",
    "机制建设",
]

SCENARIO_TAG_VOCAB = [
    "任务分配",
    "跨部门协作",
    "向上汇报",
    "工作负担",
    "角色错位",
    "项目复盘",
    "组织扩张",
    "绩效评价",
    "管理选拔",
    "管理学习",
    "战略传导",
]

APPLICABLE_ROLE_VOCAB = [
    "高管",
    "总监",
    "经理",
    "一线员工",
    "创业者",
    "项目负责人",
]

CONTENT_TYPE_VOCAB = [
    "principle",
    "method",
    "case",
    "pitfall",
    "criteria",
    "solution",
]

STANDARD_CONCEPTS = [
    "管理能力",
    "管理体系",
    "管理本质",
    "人效",
    "有效工作",
    "无效工作",
    "信息失真",
    "评价失效",
    "不确定性",
    "资源配置",
    "成本控制",
    "组织扩张",
    "沟通复杂度",
    "部门墙",
    "管理选拔",
    "高管视角",
    "管理学习",
    "文化建设",
    "机制建设",
    "人治",
    "法治",
    "复盘",
]

TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "管理认知": ("管理本质", "管理认知", "管理体系", "管理能力", "高管视角", "动态过程", "计划", "组织", "领导", "控制"),
    "人效管理": ("人效", "有效工作", "无效工作", "成本控制", "资源浪费"),
    "组织设计": ("组织设计", "组织扩张", "组织架构", "资源配置", "沟通复杂度"),
    "跨部门协作": ("部门墙", "跨部门", "协作", "扯皮"),
    "向上管理": ("向上汇报", "向上沟通", "进度与风险", "汇报"),
    "执行管理": ("执行", "任务执行", "战略传导", "指令"),
    "绩效管理": ("绩效", "评价", "人效", "奖惩", "价值"),
    "团队治理": ("团队", "团队治理", "管理能力", "无效工作"),
    "管理选拔": ("管理职位", "候选人", "晋升", "管理选拔"),
    "组织扩张": ("规模", "人数", "组织扩张", "沟通复杂度"),
    "文化建设": ("文化建设", "价值观", "企业文化", "奉献"),
    "机制建设": ("机制建设", "法治", "复盘", "流程", "机制"),
}

SCENARIO_RULES: dict[str, tuple[str, ...]] = {
    "任务分配": ("任务", "指令", "任务执行", "工作分配"),
    "跨部门协作": ("部门墙", "跨部门", "协作", "BUG", "事故"),
    "向上汇报": ("汇报", "向上沟通", "进度与风险", "回执"),
    "工作负担": ("无效工作", "脏活", "累活", "工作负担", "资源浪费"),
    "角色错位": ("高管视角", "角色", "晋升", "管理岗", "管理能力"),
    "项目复盘": ("复盘", "项目失败", "事故", "复盘"),
    "组织扩张": ("规模", "人数", "沟通复杂度", "组织扩张"),
    "绩效评价": ("评价", "绩效", "价值", "奖惩", "人效"),
    "管理选拔": ("提拔", "候选人", "管理职位", "晋升"),
    "管理学习": ("管理学习", "高管视角", "系统框架", "管理体系", "管理本质", "定义"),
    "战略传导": ("战略", "战略传导", "信息失真", "向下沟通"),
}

ROLE_RULES: dict[str, tuple[str, ...]] = {
    "高管": ("CEO", "高管", "VP", "战略", "高管视角"),
    "总监": ("总监", "中层", "总监管理"),
    "经理": ("经理", "一线经理", "管理岗", "Leader"),
    "一线员工": ("员工", "一线", "基层"),
    "创业者": ("初创", "创业", "公司初期"),
    "项目负责人": ("项目负责人", "项目", "PMO"),
}

CONCEPT_RULES: dict[str, tuple[str, ...]] = {
    "管理能力": ("管理能力",),
    "管理体系": ("管理体系",),
    "管理本质": ("管理本质", "管理的本质"),
    "人效": ("人效",),
    "有效工作": ("有效工作",),
    "无效工作": ("无效工作", "资源浪费"),
    "信息失真": ("信息失真",),
    "评价失效": ("评价失效",),
    "不确定性": ("不确定性",),
    "资源配置": ("资源配置", "资源", "资源分配"),
    "成本控制": ("控制成本", "成本控制", "成本"),
    "组织扩张": ("组织扩张", "公司规模", "人数增加", "规模扩大"),
    "沟通复杂度": ("沟通复杂度", "沟通路径"),
    "部门墙": ("部门墙",),
    "管理选拔": ("管理职位", "候选人", "晋升"),
    "高管视角": ("高管视角", "高管管理知识"),
    "管理学习": ("管理学习", "系统性学习", "系统框架"),
    "文化建设": ("文化建设", "企业文化", "价值观"),
    "机制建设": ("机制建设", "机制", "流程"),
    "人治": ("人治",),
    "法治": ("法治",),
    "复盘": ("复盘",),
}

INFERRED_CONCEPT_RULES = [
    ({"信息失真"}, "战略传导"),
    ({"部门墙"}, "跨部门协作"),
    ({"高管视角"}, "管理认知"),
    ({"无效工作"}, "人效"),
    ({"复盘"}, "机制建设"),
]

CHUNK_SUPPORT_CONCEPTS = {
    "管理能力",
    "人效",
    "有效工作",
    "无效工作",
    "信息失真",
    "评价失效",
    "不确定性",
    "组织扩张",
    "沟通复杂度",
    "部门墙",
    "高管视角",
    "管理学习",
    "文化建设",
    "机制建设",
    "人治",
    "法治",
    "复盘",
    "成本控制",
    "资源配置",
}

ORIGINAL_TERM_PATTERNS = (
    "部门墙",
    "圣旨",
    "画大饼",
    "扯皮",
    "嚎一嗓子",
    "三抓三放",
    "白嫖",
    "磨洋工",
    "见风使舵",
    "九阴真经",
    "九阳神功",
    "易筋经",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated tagging experiments for chunk + insight samples.")
    parser.add_argument("--input-path", type=Path, help="Optional input path. Defaults to insights_v2, then insights_v1.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output tagged JSON path.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Only process the first N chunks after filtering.")
    parser.add_argument("--source-file", type=str, help="Optional source_file substring filter.")
    return parser.parse_args()


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    if DEFAULT_INPUT_PATH_V2.exists():
        return DEFAULT_INPUT_PATH_V2
    if DEFAULT_INPUT_PATH_V1.exists():
        return DEFAULT_INPUT_PATH_V1
    raise FileNotFoundError(
        f"Neither {DEFAULT_INPUT_PATH_V2} nor {DEFAULT_INPUT_PATH_V1} exists."
    )


def normalize_text(text: str) -> str:
    normalized = text.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def text_pool(record: dict[str, object]) -> tuple[str, str, str, str]:
    insights = " ".join(
        str(item.get("insight_text", ""))
        for item in record.get("insights", [])
        if isinstance(item, dict)
    )
    chunk_text = str(record.get("chunk_text", ""))
    title = str(record.get("title", ""))
    source_file = str(record.get("source_file", ""))
    return (
        normalize_text(insights),
        normalize_text(chunk_text),
        normalize_text(title),
        normalize_text(source_file),
    )


def score_vocab_items(
    vocab: list[str],
    rules: dict[str, tuple[str, ...]],
    primary_text: str,
    chunk_text: str,
    title: str,
    source_file: str,
    max_items: int,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in vocab:
        score = 0.0
        for keyword in rules.get(item, ()):
            if keyword in primary_text:
                score += 1.6
            if keyword in chunk_text:
                score += 1.0
            if keyword in title or keyword in source_file:
                score += 0.4
        if score > 0:
            scored.append((score, item))
    scored = [value for value in scored if value[0] >= 1.0]
    scored.sort(key=lambda value: (-value[0], vocab.index(value[1])))
    return [item for _, item in scored[:max_items]]


def choose_content_type(insights: list[dict[str, object]], chunk_text: str) -> str:
    insight_types = [str(item.get("insight_type", "")) for item in insights]
    insight_text = " ".join(str(item.get("insight_text", "")) for item in insights)
    search_text = normalize_text(insight_text)

    if any(kind == "case_takeaway" for kind in insight_types) or re.search(r"(案例|BUG|事故|扯皮)", search_text):
        return "case"
    if any(kind in {"warning"} for kind in insight_types) or re.search(r"(风险|失真|失效|浪费|低效|埋下种子)", search_text):
        return "pitfall"
    if any(kind in {"definition", "principle", "judgment"} for kind in insight_types):
        if re.search(r"(评价|标准|是否有价值|判断)", search_text):
            return "criteria"
        return "principle"
    if re.search(r"(需要|应该|通过|机制|解法|路径|建立|重塑|优化)", search_text):
        if re.search(r"(解决|降低|缓解|优化|建设)", search_text):
            return "solution"
        return "method"
    return "principle"


def extract_concepts(
    insights_text: str,
    chunk_text: str,
) -> dict[str, list[str]]:
    explicit: list[str] = []
    inferred: list[str] = []
    original_terms: list[str] = []

    for concept, patterns in CONCEPT_RULES.items():
        if any(pattern in insights_text for pattern in patterns):
            explicit.append(concept)
        elif concept in CHUNK_SUPPORT_CONCEPTS and any(pattern in chunk_text for pattern in patterns):
            explicit.append(concept)

    for pattern in ORIGINAL_TERM_PATTERNS:
        if pattern in chunk_text:
            original_terms.append(pattern)

    explicit = explicit[:5]
    original_terms = original_terms[:5]

    for required_concepts, inferred_concept in INFERRED_CONCEPT_RULES:
        if inferred_concept in explicit:
            continue
        if required_concepts.issubset(set(explicit)):
            inferred.append(inferred_concept)
    inferred = inferred[:3]

    return {
        "explicit": explicit,
        "inferred": inferred,
        "original_terms": original_terms,
    }


def derive_relations(concepts: dict[str, list[str]], chunk_text: str, insights_text: str) -> list[dict[str, str]]:
    explicit = set(concepts.get("explicit", []))
    search_text = f"{chunk_text} {insights_text}"
    relations: list[dict[str, str]] = []

    def add(relation_type: str, from_value: str, to_value: str) -> None:
        relation = {
            "type": relation_type,
            "from": from_value,
            "to": to_value,
        }
        if relation not in relations:
            relations.append(relation)

    if {"信息失真", "无效工作"}.issubset(explicit) and re.search(r"(导致|引发|结果)", search_text):
        add("cause", "信息失真", "无效工作")
    if {"组织扩张", "信息失真"}.issubset(explicit) and re.search(r"(人数|规模|沟通复杂度)", search_text):
        add("cause", "组织扩张", "信息失真")
    if {"复盘", "信息失真"}.issubset(explicit):
        add("solution", "复盘", "信息失真")
    if {"复盘", "评价失效"}.issubset(explicit):
        add("solution", "复盘", "评价失效")
    if {"人治", "法治"}.issubset(explicit):
        add("contrast", "人治", "法治")
    if {"高管视角", "管理学习"}.issubset(explicit):
        add("dependency", "高管视角", "管理学习")
    if {"部门墙", "管理能力"}.issubset(explicit):
        add("cause", "部门墙", "管理能力")

    return relations


def tag_record(record: dict[str, object]) -> dict[str, object]:
    insights_text, chunk_text, title, source_file = text_pool(record)
    insights = [item for item in record.get("insights", []) if isinstance(item, dict)]

    concepts = extract_concepts(insights_text=insights_text, chunk_text=chunk_text)
    concept_text = " ".join(concepts["explicit"] + concepts["inferred"] + concepts["original_terms"])

    topic_tags = score_vocab_items(
        vocab=TOPIC_TAG_VOCAB,
        rules=TOPIC_RULES,
        primary_text=f"{insights_text} {concept_text}",
        chunk_text=chunk_text,
        title=title,
        source_file=source_file,
        max_items=3,
    )
    scenario_tags = score_vocab_items(
        vocab=SCENARIO_TAG_VOCAB,
        rules=SCENARIO_RULES,
        primary_text=f"{insights_text} {concept_text}",
        chunk_text=chunk_text,
        title=title,
        source_file=source_file,
        max_items=3,
    )
    applicable_roles = score_vocab_items(
        vocab=APPLICABLE_ROLE_VOCAB,
        rules=ROLE_RULES,
        primary_text=f"{insights_text} {concept_text}",
        chunk_text=chunk_text,
        title=title,
        source_file=source_file,
        max_items=3,
    )

    content_type = choose_content_type(insights=insights, chunk_text=chunk_text)
    relations = derive_relations(concepts=concepts, chunk_text=chunk_text, insights_text=insights_text)

    needs_manual_review = False
    if not topic_tags:
        needs_manual_review = True
    if not concepts["explicit"]:
        needs_manual_review = True
    if not scenario_tags:
        needs_manual_review = True
    if len(concepts["inferred"]) > 2 and not concepts["explicit"]:
        needs_manual_review = True

    return {
        "source_file": str(record.get("source_file", "")),
        "title": str(record.get("title", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "chunk_order": int(record.get("chunk_order", 0)),
        "chunk_text": str(record.get("chunk_text", "")),
        "insights": insights,
        "tags": {
            "topic_tags": topic_tags,
            "content_type": content_type if content_type in CONTENT_TYPE_VOCAB else "principle",
            "concepts": concepts,
            "scenario_tags": scenario_tags,
            "applicable_roles": applicable_roles,
            "relations": relations,
        },
        "needs_manual_review": needs_manual_review,
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0")

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if args.source_file:
        records = [record for record in records if args.source_file in str(record.get("source_file", ""))]
    records = records[: args.limit]

    tagged_records = [tag_record(record) for record in records]

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(tagged_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file: {input_path}")
    print(f"Processed records: {len(tagged_records)}")
    print(f"Saved tagged samples to: {args.output_path}")


if __name__ == "__main__":
    main()
