#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_PATH_V2 = Path("data/experiments/insights_v2/sample_insights.json")
DEFAULT_INPUT_PATH_V1 = Path("data/experiments/insights_v1/sample_insights.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/tagging_v1_1/sample_tagged_chunks.json")
DEFAULT_LIMIT = 5

TOPIC_TAG_VOCAB = [
    "管理认知",
    "团队治理",
    "执行管理",
    "组织设计",
    "人效管理",
    "绩效管理",
    "文化建设",
    "机制建设",
]

SCENARIO_TAG_VOCAB = [
    "跨部门协作",
    "项目复盘",
    "管理学习",
    "向上汇报",
    "管理选拔",
    "工作负担",
    "角色错位",
    "组织扩张",
    "绩效评价",
    "战略传导",
    "任务分配",
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

CONTROLLED_CONCEPTS = [
    "管理能力",
    "管理体系",
    "管理本质",
    "管理学习",
    "高管视角",
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
    "文化建设",
    "机制建设",
    "人治",
    "法治",
    "复盘",
]

TOPIC_RULES: dict[str, tuple[str, ...]] = {
    "管理认知": (
        "管理本质",
        "管理体系",
        "管理能力",
        "高管视角",
        "动态过程",
        "计划",
        "组织",
        "领导",
        "控制",
    ),
    "团队治理": (
        "团队",
        "团队治理",
        "管理能力",
        "人治",
        "法治",
        "文化建设",
        "机制建设",
    ),
    "执行管理": (
        "执行",
        "任务执行",
        "战略传导",
        "指令",
        "战略",
        "传导",
    ),
    "组织设计": (
        "组织设计",
        "组织扩张",
        "组织架构",
        "资源配置",
        "沟通复杂度",
        "规模扩大",
    ),
    "人效管理": (
        "人效",
        "有效工作",
        "无效工作",
        "成本控制",
        "资源浪费",
    ),
    "绩效管理": (
        "绩效",
        "评价",
        "奖惩",
        "价值",
        "评价失效",
    ),
    "文化建设": (
        "文化建设",
        "价值观",
        "企业文化",
        "奉献",
    ),
    "机制建设": (
        "机制建设",
        "法治",
        "复盘",
        "流程",
        "机制",
    ),
}

SCENARIO_RULES: dict[str, tuple[str, ...]] = {
    "跨部门协作": ("部门墙", "跨部门", "协作", "BUG", "事故", "扯皮"),
    "项目复盘": ("复盘", "项目失败", "事故", "复盘"),
    "管理学习": ("管理学习", "高管视角", "系统框架", "管理体系", "管理本质"),
    "向上汇报": ("汇报", "向上沟通", "进度与风险", "回执"),
    "管理选拔": ("管理职位", "候选人", "晋升", "提拔"),
    "工作负担": ("无效工作", "脏活", "累活", "工作负担", "资源浪费"),
    "角色错位": ("高管视角", "角色", "晋升经理", "管理岗"),
    "组织扩张": ("规模", "人数", "沟通复杂度", "组织扩张"),
    "绩效评价": ("评价", "绩效", "价值", "奖惩", "是否有价值"),
    "战略传导": ("战略", "战略传导", "信息失真", "向下沟通"),
    "任务分配": ("任务", "指令", "任务执行", "工作分配"),
}

CONCEPT_PATTERNS: dict[str, tuple[str, ...]] = {
    "管理能力": ("管理能力",),
    "管理体系": ("管理体系",),
    "管理本质": ("管理本质", "管理的本质", "管理是涵盖计划、组织、领导与控制的动态过程"),
    "管理学习": ("管理学习", "系统性学习", "系统框架"),
    "高管视角": ("高管视角", "高管管理知识"),
    "人效": ("人效",),
    "有效工作": ("有效工作",),
    "无效工作": ("无效工作", "资源浪费"),
    "信息失真": ("信息失真",),
    "评价失效": ("评价失效",),
    "不确定性": ("不确定性",),
    "资源配置": ("资源配置", "资源", "资源分配", "优化配置"),
    "成本控制": ("控制成本", "成本控制",),
    "组织扩张": ("组织扩张", "公司规模", "规模扩大", "人数增加"),
    "沟通复杂度": ("沟通复杂度", "沟通路径"),
    "部门墙": ("部门墙",),
    "管理选拔": ("管理职位", "候选人", "晋升",),
    "文化建设": ("文化建设", "企业文化", "价值观"),
    "机制建设": ("机制建设", "机制", "流程"),
    "人治": ("人治",),
    "法治": ("法治",),
    "复盘": ("复盘",),
}

INSIGHT_FIRST_CONCEPTS = {
    "管理能力",
    "管理体系",
    "管理本质",
    "管理学习",
    "高管视角",
    "人效",
    "有效工作",
    "无效工作",
    "信息失真",
    "评价失效",
    "不确定性",
    "沟通复杂度",
    "部门墙",
    "管理选拔",
    "文化建设",
    "机制建设",
    "人治",
    "法治",
    "复盘",
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
    parser = argparse.ArgumentParser(description="Run isolated tagging v1.1 experiments for chunk + insight samples.")
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
    insights_text = " ".join(
        str(item.get("insight_text", ""))
        for item in record.get("insights", [])
        if isinstance(item, dict)
    )
    chunk_text = str(record.get("chunk_text", ""))
    title = str(record.get("title", ""))
    source_file = str(record.get("source_file", ""))
    return (
        normalize_text(insights_text),
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
    min_score: float = 1.2,
) -> list[str]:
    scored: list[tuple[float, str]] = []
    for item in vocab:
        score = 0.0
        for keyword in rules.get(item, ()):
            if keyword in primary_text:
                score += 1.8
            if keyword in chunk_text:
                score += 0.8
            if keyword in title or keyword in source_file:
                score += 0.3
        if score >= min_score:
            scored.append((score, item))
    scored.sort(key=lambda value: (-value[0], vocab.index(value[1])))
    return [item for _, item in scored[:max_items]]


def choose_content_type(insights: list[dict[str, object]], chunk_text: str) -> str:
    insight_types = [str(item.get("insight_type", "")) for item in insights]
    insights_text = normalize_text(" ".join(str(item.get("insight_text", "")) for item in insights))
    definition_like = sum(1 for kind in insight_types if kind in {"definition", "principle", "judgment"})
    method_like = sum(1 for kind in insight_types if kind == "method")

    if any(kind == "case_takeaway" for kind in insight_types) or re.search(r"(案例|BUG|事故|扯皮)", insights_text):
        return "case"
    if re.search(r"(评价|标准|判断|是否有价值|如何评价)", insights_text):
        return "criteria"
    if definition_like >= method_like and definition_like > 0:
        if any(kind == "warning" for kind in insight_types) or re.search(r"(风险|失真|失效|浪费|低效|埋下种子)", insights_text):
            return "pitfall"
        return "principle"
    if re.search(r"(需要|应该|通过|建立|重塑|优化|学习)", insights_text):
        if re.search(r"(解决|降低|缓解|优化|建设)", insights_text):
            return "solution"
        return "method"
    if any(kind == "warning" for kind in insight_types) or re.search(r"(风险|失真|失效|浪费|低效|埋下种子)", insights_text):
        return "pitfall"
    return "principle"


def extract_concepts(insights_text: str, chunk_text: str) -> dict[str, list[str]]:
    explicit: list[str] = []
    inferred: list[str] = []
    original_terms: list[str] = []

    for concept, patterns in CONCEPT_PATTERNS.items():
        if concept in INSIGHT_FIRST_CONCEPTS and any(pattern in insights_text for pattern in patterns):
            explicit.append(concept)

    for concept, patterns in CONCEPT_PATTERNS.items():
        if concept in explicit:
            continue
        if any(pattern in chunk_text for pattern in patterns):
            if concept in {"资源配置", "成本控制", "组织扩张"} and not any(pattern in insights_text for pattern in patterns):
                continue
            explicit.append(concept)

    for pattern in ORIGINAL_TERM_PATTERNS:
        if pattern in chunk_text:
            original_terms.append(pattern)

    if "跨部门失败常被归因于管理能力不足" in insights_text and "部门墙" not in explicit:
        explicit.append("部门墙")
    if "项目失败通常暴露管理问题" in insights_text and "管理能力" not in explicit:
        explicit.append("管理能力")
    if "糟糕管理会为失败埋下种子" in insights_text and "管理能力" not in explicit:
        explicit.append("管理能力")
    if "好的管理未必直接导致成功" in insights_text and "管理能力" not in explicit:
        explicit.append("管理能力")
    if "管理的核心目标是提升人效" in insights_text and "人效" not in explicit:
        explicit.append("人效")
    if "管理的意义是以更高效率控制成本" in insights_text and "成本控制" not in explicit:
        explicit.append("成本控制")

    explicit = explicit[:5]
    original_terms = original_terms[:5]

    if "部门墙" in explicit:
        inferred.append("跨部门协作")
    if "高管视角" in explicit:
        inferred.append("管理认知")
    if "复盘" in explicit:
        inferred.append("机制建设")
    if "无效工作" in explicit and "人效" not in explicit:
        inferred.append("人效")

    inferred = [item for item in inferred if item not in explicit][:3]
    return {
        "explicit": explicit,
        "inferred": inferred,
        "original_terms": original_terms,
    }


def choose_roles(
    topic_tags: list[str],
    scenario_tags: list[str],
    content_type: str,
    concepts: dict[str, list[str]],
    insights_text: str,
    chunk_text: str,
) -> list[str]:
    scores = {role: 0.0 for role in APPLICABLE_ROLE_VOCAB}
    explicit = set(concepts.get("explicit", []))
    inferred = set(concepts.get("inferred", []))
    search_text = f"{insights_text} {chunk_text}"

    if "高管视角" in explicit or "组织设计" in topic_tags or "组织扩张" in scenario_tags:
        scores["高管"] += 2.0
        scores["创业者"] += 1.4
    if "团队治理" in topic_tags or "跨部门协作" in scenario_tags:
        scores["经理"] += 1.8
        scores["项目负责人"] += 1.8
        scores["总监"] += 1.0
    if "管理学习" in scenario_tags:
        scores["经理"] += 1.6
        scores["一线员工"] += 1.4
        scores["总监"] += 0.8
    if "向上汇报" in scenario_tags:
        scores["经理"] += 1.4
        scores["总监"] += 1.0
    if "绩效管理" in topic_tags or "绩效评价" in scenario_tags:
        scores["经理"] += 1.2
        scores["总监"] += 1.2
    if "管理选拔" in scenario_tags:
        scores["总监"] += 1.0
        scores["高管"] += 1.0
    if "人效管理" in topic_tags:
        scores["经理"] += 1.0
        scores["总监"] += 1.0

    if content_type == "method":
        scores["经理"] += 0.8
        scores["项目负责人"] += 0.8
    if content_type == "solution":
        scores["总监"] += 0.8
        scores["高管"] += 0.8
    if content_type == "pitfall":
        scores["经理"] += 0.6
        scores["项目负责人"] += 0.6

    if "晋升经理前应先建立高管视角" in insights_text:
        scores["一线员工"] += 2.0
        scores["经理"] += 1.4
    if "项目失败通常暴露管理问题" in insights_text:
        scores["项目负责人"] += 1.4
        scores["经理"] += 1.0
    if re.search(r"(CEO|高管|VP)", search_text):
        scores["高管"] += 0.8
    if re.search(r"(经理|Leader)", search_text):
        scores["经理"] += 0.5
    if re.search(r"(总监|中层)", search_text):
        scores["总监"] += 0.5

    if "管理认知" in topic_tags and not scenario_tags:
        scores["经理"] += 0.6
        scores["总监"] += 0.6
    if "管理认知" in topic_tags and "管理学习" not in scenario_tags:
        scores["一线员工"] -= 0.6

    ordered = sorted(scores.items(), key=lambda item: (-item[1], APPLICABLE_ROLE_VOCAB.index(item[0])))
    selected = [role for role, score in ordered if score >= 1.5][:2]
    return selected


def derive_relations(concepts: dict[str, list[str]], insights_text: str) -> list[dict[str, str]]:
    explicit = set(concepts.get("explicit", []))
    relations: list[dict[str, str]] = []

    def add(relation_type: str, from_value: str, to_value: str) -> None:
        relation = {
            "type": relation_type,
            "from": from_value,
            "to": to_value,
        }
        if relation not in relations:
            relations.append(relation)

    if {"组织扩张", "沟通复杂度"}.issubset(explicit) and "团队扩大将显著增加沟通复杂度" in insights_text:
        add("cause", "组织扩张", "沟通复杂度")
    if {"信息失真", "无效工作"}.issubset(explicit) and re.search(r"(信息失真.+无效工作|无效工作.+信息失真)", insights_text):
        add("cause", "信息失真", "无效工作")
    if {"复盘", "机制建设"}.issubset(explicit) and re.search(r"(复盘.+机制|机制.+复盘)", insights_text):
        add("solution", "复盘", "机制建设")
    if {"人治", "法治"}.issubset(explicit) and re.search(r"(人治.+法治|法治.+人治)", insights_text):
        add("contrast", "人治", "法治")
    if {"高管视角", "管理学习"}.issubset(explicit) and "晋升经理前应先建立高管视角" in insights_text:
        add("dependency", "高管视角", "管理学习")

    return relations


def tag_record(record: dict[str, object]) -> dict[str, object]:
    insights_text, chunk_text, title, source_file = text_pool(record)
    insights = [item for item in record.get("insights", []) if isinstance(item, dict)]

    concepts = extract_concepts(insights_text=insights_text, chunk_text=chunk_text)
    concept_text = " ".join(concepts["explicit"] + concepts["inferred"])

    topic_tags = score_vocab_items(
        vocab=TOPIC_TAG_VOCAB,
        rules=TOPIC_RULES,
        primary_text=f"{insights_text} {concept_text}",
        chunk_text=chunk_text,
        title=title,
        source_file=source_file,
        max_items=3,
        min_score=1.6,
    )
    scenario_tags = score_vocab_items(
        vocab=SCENARIO_TAG_VOCAB,
        rules=SCENARIO_RULES,
        primary_text=f"{insights_text} {concept_text}",
        chunk_text=chunk_text,
        title=title,
        source_file=source_file,
        max_items=3,
        min_score=1.4,
    )
    content_type = choose_content_type(insights=insights, chunk_text=chunk_text)
    applicable_roles = choose_roles(
        topic_tags=topic_tags,
        scenario_tags=scenario_tags,
        content_type=content_type,
        concepts=concepts,
        insights_text=insights_text,
        chunk_text=chunk_text,
    )
    relations = derive_relations(concepts=concepts, insights_text=insights_text)

    needs_manual_review = False
    if not topic_tags:
        needs_manual_review = True
    if not concepts["explicit"]:
        needs_manual_review = True
    if content_type == "principle" and not insights:
        needs_manual_review = True
    if len(relations) > 1:
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
